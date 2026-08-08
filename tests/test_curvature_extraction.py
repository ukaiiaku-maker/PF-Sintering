import math

import numpy as np

from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.curvature_extraction import _walk_branch, branch_mu_J_profile, window_curvature
from pf_sintering.model import ModelConfig, Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_flux, surface_mobility_tensor
from pf_sintering.tj_force import compute_neck_tj_forces


def _flat_grid(dx=1e-9, W=20e-9, Nx=400, Ny=200):
    p = build_params(ModelConfig(
        preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W, use_aniso_surface=False,
    ))
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    return p, x, y


def test_flat_interface_kappa_A_is_zero():
    # A planar interface is the exact 1D equilibrium profile: mu vanishes
    # identically (both bulk and gradient terms cancel), so kappa_A must be
    # exactly zero regardless of window position.
    p, x, y = _flat_grid()
    X, Y = np.meshgrid(x, y)
    xc = x[len(x) // 2]
    f = 0.5 * (1 - np.tanh((X - xc) / p.interface_width))
    e1 = e2 = e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    tj_xy = (xc, y.mean())
    wc = window_curvature(f, e1, e2, e3, s, p, tj_xy, np.array([0.0, 1.0]), 0.0, 3 * p.interface_width, mu_field=mu)
    assert wc.resolved
    assert abs(wc.kappa_A) < 1e-6 / p.interface_width  # zero to numerical roundoff


def test_circle_curvature_methods_A_and_B_match_analytic_kappa():
    # Method A (mu/(1.5*gamma_s), see module docstring for the 1.5 factor's
    # derivation) and Method B (geometric Kasa fit) must both recover the
    # known 1/R curvature of a synthetic circular interface, independently.
    p, x, y = _flat_grid(dx=1e-9, W=20e-9, Nx=400, Ny=200)
    X, Y = np.meshgrid(x, y)
    e1 = e2 = e3 = np.zeros_like(X)
    s = Sink(threshold=math.inf)
    xc = x[len(x) // 2]

    for R in (300e-9, 500e-9, 800e-9):
        cy0 = -R + y.mean()
        r = np.hypot(X - xc, Y - cy0)
        f = 0.5 * (1 - np.tanh((r - R) / p.interface_width))
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        tj_xy = (xc, y.mean())
        wc = window_curvature(f, e1, e2, e3, s, p, tj_xy, np.array([1.0, 0.0]),
                               0.0, 2 * p.interface_width, mu_field=mu)
        assert wc.resolved
        kappa_exact = 1.0 / R
        assert math.isclose(wc.kappa_A, kappa_exact, rel_tol=0.01)
        assert math.isclose(wc.kappa_B, kappa_exact, rel_tol=0.01)


def test_window_curvature_unresolved_when_too_few_contour_points():
    p, x, y = _flat_grid()
    f = np.zeros((len(y), len(x)))  # no interface anywhere
    e1 = e2 = e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    wc = window_curvature(f, e1, e2, e3, s, p, (x.mean(), y.mean()), np.array([0.0, 1.0]),
                           0.0, 3 * p.interface_width)
    assert not wc.resolved


def _sinusoidal_state(dx_nm=5.0):
    p = build_params(ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", dx=dx_nm * 1e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6, seed=42,
        sinusoid_wavelength=480e-9, sinusoid_amplitude=24e-9,
        interface_width_override=20e-9, eta_diffusivity_fixed_physical=True,
        use_aniso_surface=False, surface_mobility_scale=0.3,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def test_walk_branch_diverges_for_distinct_tj_branches():
    # Regression test for a real bug found this milestone: an earlier
    # global-nearest-neighbor version of the branch walk converged onto
    # the IDENTICAL path for v_s1 and v_s2 (two branches under 90 degrees
    # apart at a real TJ project positively onto each other's direction,
    # so a direction-only pre-filter cannot separate them). The
    # local-circle-marching walk must not have this failure.
    p, f, e1, e2, e3 = _sinusoidal_state()
    s = Sink(threshold=math.inf)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    assert rep.top.resolved
    path1, _ = _walk_branch(f, p, rep.top.tj_xy, rep.top.v_s1, 1e-7)
    path2, _ = _walk_branch(f, p, rep.top.tj_xy, rep.top.v_s2, 1e-7)
    assert path1 is not None and path2 is not None
    assert len(path1) >= 3 and len(path2) >= 3
    # paths share only the TJ-adjacent start point; must diverge well before
    # the end of a 100nm walk
    assert np.linalg.norm(path1[-1] - path2[-1]) > 5 * p.dx


def test_branch_mu_J_profile_resolves_on_real_tj():
    p, f, e1, e2, e3 = _sinusoidal_state()
    s = Sink(threshold=math.inf)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    assert rep.top.resolved
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    M_s = m_s_ref(p.M_f, p.interface_width)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, "reflecting", "periodic",
                                             eps_n=1e-6 / p.interface_width)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, "reflecting", "periodic")
    prof = branch_mu_J_profile(f, mu, Jx, Jy, p, rep.top.tj_xy, rep.top.v_s1, 100e-9, n_samples=15)
    assert prof is not None
    assert len(prof["s"]) == 15
    assert all(math.isfinite(v) for v in prof["mu"])
