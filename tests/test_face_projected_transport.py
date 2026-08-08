import math

import numpy as np

from pf_sintering.bc_ops import face_gradient_x, face_gradient_y, flux_divergence
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import (
    dissipation_density_face_projected,
    exact_dissipation_face_projected,
    face_projected_tangentiality,
    fdot_chain_face_projected,
    m_s_ref,
    surface_divergence_update,
    surface_flux,
    surface_flux_face_projected,
    surface_mobility_tensor,
    variational_surface_diffusion_step,
)

BC_X, BC_Y = "reflecting", "periodic"


def _circle_params(dx=2e-9, W=20e-9, Nx=150, Ny=150):
    p = build_params(ModelConfig(
        preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W, use_aniso_surface=False, surface_mobility_scale=0.3,
    ))
    return p


def _circle_state(p, R=100e-9, n_modes=3, mu_amplitude=0.3):
    dx = p.dx
    x = (np.arange(1, p.Nx + 1)) * dx
    y = (np.arange(1, p.Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    cx, cy = x.mean(), y.mean()
    r = np.hypot(X - cx, Y - cy)
    theta = np.arctan2(Y - cy, X - cx)
    W = p.interface_width
    f = 0.5 * (1 - np.tanh((r - R) / W))
    mu = mu_amplitude * np.sin(n_modes * theta)
    return f, mu


def test_face_gradient_matches_grad_bc_tangential_component_on_planar_field():
    # For a field with NO y-variation, the y-derivative at any x-face must
    # be exactly zero (both the direct one-sided x-derivative and the
    # averaged-tangential y-derivative reduce to trivial cases we can
    # check by hand).
    dx = 2e-9
    Nx = Ny = 40
    x = (np.arange(1, Nx + 1)) * dx
    X = np.tile(x, (Ny, 1))
    a = X ** 2  # varies only in x
    gx_face, gy_face = face_gradient_x(a, dx, bc_x=BC_X, bc_y=BC_Y)
    assert np.allclose(gy_face, 0.0, atol=1e-20)
    # normal derivative should match the analytic d(x^2)/dx = 2x at the face midpoint
    x_face = x + 0.5 * dx
    expected = 2 * x_face[:-1]
    assert np.allclose(gx_face[:, :-1], np.tile(expected, (Ny, 1)), rtol=1e-6)


def test_face_projected_P_is_symmetric_and_psd():
    p = _circle_params()
    f, mu = _circle_state(p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    for face in (fp["xface"], fp["yface"]):
        Mxx, Mxy, Myy = face["Mxx"], face["Mxy"], face["Myy"]
        # symmetry is by construction (Mxy used for both off-diagonal terms);
        # PSD: trace >= 0 and determinant >= 0 (2x2 PSD test)
        assert np.all(Mxx >= -1e-30)
        assert np.all(Myy >= -1e-30)
        det = Mxx * Myy - Mxy * Mxy
        assert np.all(det >= -1e-30 * np.maximum(Mxx * Myy, 1e-300))


def test_P_face_dot_n_face_is_zero():
    # P . n = n*(1 - |n|^2), exactly zero only for a TRUE unit vector;
    # interface_normal's eps_n regularization makes |n|<1 strictly, with
    # the deviation growing wherever |grad f| is small relative to eps_n
    # (deep in the bulk/vapor, far from any interface). That is expected
    # and harmless -- q(f) (hence M_tensor, hence the actual flux) is ~0
    # in exactly those same regions -- so this check is restricted to
    # where q(f) is non-negligible, the only place P.n~0 needs to hold
    # for the flux itself to be tangential.
    p = _circle_params()
    f, mu = _circle_state(p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    for face in (fp["xface"], fp["yface"]):
        nx, ny = face["nx"], face["ny"]
        q = face["q_face"]
        valid = q > 1e-3 * np.max(q)
        Pxx = 1.0 - nx * nx
        Pxy = -nx * ny
        Pyy = 1.0 - ny * ny
        vx = Pxx * nx + Pxy * ny
        vy = Pxy * nx + Pyy * ny
        assert np.max(np.abs(vx[valid])) < 1e-6
        assert np.max(np.abs(vy[valid])) < 1e-6


def test_J_face_dot_n_face_is_near_roundoff_on_high_curvature_circle():
    # The exact case that showed face_x max ~1.0 under the legacy scheme
    # (Milestone 13C) -- must be roundoff-level under face projection.
    p = _circle_params(dx=2e-9, Nx=200, Ny=200)
    f, mu = _circle_state(p, R=150e-9)
    M_s = m_s_ref(p.M_f, p.interface_width)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    tang = face_projected_tangentiality(fp)
    for ratio_key, mag_key in (("ratio_x", "mag_x"), ("ratio_y", "mag_y")):
        ratio, mag = tang[ratio_key], tang[mag_key]
        valid = np.isfinite(ratio) & (mag > 1e-6 * np.nanmax(mag))
        assert np.max(ratio[valid]) < 1e-4


def test_exact_mass_conservation_face_projected():
    p = _circle_params()
    f, mu = _circle_state(p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, BC_X, BC_Y,
                                             face_flux_mode="face_projected")
    mass0 = float(f.sum())
    mass1 = float(f_new.sum())
    assert abs(mass1 - mass0) / mass0 < 1e-12


def test_planar_interface_zero_flux_face_projected():
    # A flat, uniform-mu planar interface: grad(mu)=0 everywhere, so the
    # face-projected flux must be exactly zero (both components, both
    # face families), and f must not change.
    dx = 2e-9
    Nx, Ny = 60, 20
    p = build_params(ModelConfig(
        preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=20e-9, use_aniso_surface=False, surface_mobility_scale=0.3,
    ))
    x = (np.arange(1, Nx + 1)) * dx
    W = p.interface_width
    f = np.tile(0.5 * (1 + np.tanh((x - x.mean()) / W)), (Ny, 1))
    mu = np.zeros_like(f)
    M_s = m_s_ref(p.M_f, W)
    f_new, diag = surface_divergence_update(f, mu, dx, p.dt, W, M_s, BC_X, BC_Y, face_flux_mode="face_projected")
    assert np.allclose(diag["Jx_face"], 0.0, atol=1e-25)
    assert np.allclose(diag["Jy_face"], 0.0, atol=1e-25)
    assert np.allclose(f_new, f, atol=1e-20)


def test_energy_descent_face_projected():
    p = _circle_params(dx=2e-9, Nx=100, Ny=100)
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
    f = np.clip(0.5 + 0.2 * gaussian_filter(rng.normal(size=(p.Ny, p.Nx)), sigma=4), 0.02, 0.98)
    e1 = e2 = e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    F0 = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    for frac in (1.0, 0.1, 0.01):
        f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt * frac, p.interface_width, M_s, BC_X, BC_Y,
                                                  face_flux_mode="face_projected")
        F1 = exact_free_energy_isotropic(f_new, e1, e2, e3, s, p)
        assert F1 <= F0 + 1e-6 * abs(F0)
        # Milestone 13E: exact_dissipation_face_projected's PER-FACE partial
        # terms are not individually sign-definite (each is only "half" of
        # the full 2x2 quadratic form at that face -- see the module
        # docstring) -- only the GLOBAL sum D_h is guaranteed non-negative
        # (confirmed via direct <mu,L_h mu> probing, not pointwise PSD-ness).
        Dx, Dy = diag["D_density_x"], diag["D_density_y"]
        D_h = (p.dx * p.dx) * (float(np.sum(Dx)) + float(np.sum(Dy)))
        assert D_h >= -1e-30 * max(abs(D_h), 1.0)


def test_legacy_path_unchanged_by_new_mode_default():
    # face_flux_mode defaults to "cell_average_legacy"; calling without
    # the new argument must reproduce the pre-Milestone-13D behavior
    # exactly (same dict keys/values as the explicit legacy call).
    p = _circle_params()
    f, mu = _circle_state(p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    f_new_default, diag_default = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, BC_X, BC_Y)
    f_new_explicit, diag_explicit = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, BC_X, BC_Y,
                                                                face_flux_mode="cell_average_legacy")
    assert np.array_equal(f_new_default, f_new_explicit)
    assert set(diag_default.keys()) == {"Jx", "Jy", "Mxx", "Mxy", "Myy", "div", "D_density"}
    assert np.array_equal(diag_default["Jx"], diag_explicit["Jx"])


def test_fdot_chain_equals_negative_exact_dissipation():
    # Milestone 13E Sections 3-4: Fdot_chain = dx^2*sum(mu*f_dot), computed
    # directly from the chain rule using the ACTUAL implemented flux/
    # divergence, must equal -D_h (exact_dissipation_face_projected's sum)
    # to machine precision -- an algebraic identity from discrete
    # summation by parts, not a dt->0 limit.
    p = _circle_params()
    f, mu = _circle_state(p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    Fdot_chain, _ = fdot_chain_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y, fp=fp)
    Dx, Dy = exact_dissipation_face_projected(fp)
    D_h = (p.dx * p.dx) * (float(np.sum(Dx)) + float(np.sum(Dy)))
    assert math.isclose(Fdot_chain, -D_h, rel_tol=1e-9, abs_tol=1e-30)


def test_quadratic_form_negative_semidefinite_at_frozen_f():
    # Milestone 13E Section 6: <mu, L_h mu> <= 0 for diverse mu (smooth,
    # high-frequency, single-spike) at a fixed f -- the decisive physical
    # test (only the SYMMETRIC part of L_h enters a quadratic form, so
    # this is insensitive to -- and does not require -- exact self-
    # adjointness of L_h itself).
    p = _circle_params(dx=2e-9, Nx=60, Ny=60)
    f, _ = _circle_state(p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    rng = np.random.default_rng(7)
    from scipy.ndimage import gaussian_filter

    def L_h(mu):
        fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
        return -flux_divergence(fp["Jx_face"], fp["Jy_face"], p.dx, bc_x=BC_X, bc_y=BC_Y)

    test_vectors = []
    for _ in range(20):
        test_vectors.append(gaussian_filter(rng.normal(size=(p.Ny, p.Nx)), sigma=rng.uniform(1, 6)))
    test_vectors.append(rng.normal(size=(p.Ny, p.Nx)))  # unsmoothed / high-frequency
    spike = np.zeros((p.Ny, p.Nx))
    spike[p.Ny // 2, p.Nx // 2] = 1.0
    test_vectors.append(spike)  # single-point localized

    for mu in test_vectors:
        q = (p.dx * p.dx) * float(np.sum(mu * L_h(mu)))
        assert q <= 1e-30 * max(abs(q), 1.0)


def test_exact_dissipation_converges_to_unity_ratio_as_dt_to_zero():
    # Milestone 13E Section 8: the CORRECTED D_h converges to a ratio of
    # 1.0 (not the ~0.655 the deprecated dissipation_density_face_projected
    # gave) as dt->0.
    from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
    p = _circle_params(dx=2e-9, Nx=100, Ny=100)
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    f = np.clip(0.5 + 0.2 * gaussian_filter(rng.normal(size=(p.Ny, p.Nx)), sigma=4), 0.02, 0.98)
    e1 = e2 = e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    mu = mu_isotropic(f, e1, e2, e3, s, p)

    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    Dx, Dy = exact_dissipation_face_projected(fp)
    D_h = (p.dx * p.dx) * (float(np.sum(Dx)) + float(np.sum(Dy)))

    F0 = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    ratios = []
    for frac in (1.0, 0.1, 0.01, 0.001):
        dt = p.dt * frac
        f_new, _ = surface_divergence_update(f, mu, p.dx, dt, p.interface_width, M_s, BC_X, BC_Y,
                                              face_flux_mode="face_projected")
        F1 = exact_free_energy_isotropic(f_new, e1, e2, e3, s, p)
        ratios.append((F0 - F1) / dt / D_h)
    assert ratios[-1] > 0.999
    assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)  # monotonically improving toward 1


def test_variational_surface_diffusion_step_defaults_to_face_projected():
    # Milestone 13E Section 12 (promotion gate): the NEW canonical wrapper
    # defaults to face_projected (diag carries Jx_face, not the legacy
    # cell-centered Jx) ...
    p = _circle_params(dx=2e-9, Nx=60, Ny=60)
    f, mu = _circle_state(p)
    M_s = m_s_ref(p.M_f, p.interface_width)

    f_new_default, diag_default = variational_surface_diffusion_step(
        f, mu, p.dx, p.dt, p.interface_width, M_s, BC_X, BC_Y)
    f_new_explicit, diag_explicit = surface_divergence_update(
        f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y,
        face_flux_mode="face_projected")

    assert "Jx_face" in diag_default and "Jx" not in diag_default
    assert np.array_equal(f_new_default, f_new_explicit)
    assert np.array_equal(diag_default["Jx_face"], diag_explicit["Jx_face"])

    # ... while surface_divergence_update's OWN bare default is UNCHANGED
    # (still cell_average_legacy, still cell-centered Jx/Jy) -- flipping
    # that shared default would silently alter ~15 pre-13D/13E milestone
    # scripts/tests that call it without face_flux_mode.
    f_legacy, diag_legacy = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s,
                                                        bc_x=BC_X, bc_y=BC_Y)
    assert "Jx" in diag_legacy and "Jx_face" not in diag_legacy

    # cell_average_legacy stays explicitly reachable through the new wrapper too.
    f_wrap_legacy, diag_wrap_legacy = variational_surface_diffusion_step(
        f, mu, p.dx, p.dt, p.interface_width, M_s, BC_X, BC_Y, face_flux_mode="cell_average_legacy")
    assert np.array_equal(f_wrap_legacy, f_legacy)
    assert "Jx" in diag_wrap_legacy and "Jx_face" not in diag_wrap_legacy
