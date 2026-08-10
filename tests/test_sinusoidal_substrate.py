import math

import numpy as np
from skimage.measure import find_contours

from pf_sintering.gb_obstacle_energy import obstacle_ell, obstacle_profile
from pf_sintering.gb_signed_distance import sinusoid_signed_distance
from pf_sintering.model import ModelConfig, Sink, build_params, compute_stress, initialize_fields
from pf_sintering.tj_force import locate_neck_tjs
from pf_sintering.tj_subgrid import compute_subgrid_contact


def _cfg(**overrides):
    return ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=20e-9,
        **overrides,
    )


def test_domain_height_matches_exactly_one_wavelength():
    p = build_params(_cfg())
    assert math.isclose(p.Ny * p.dx, p.sinusoid_wavelength, rel_tol=1e-12)


def test_existing_geometries_unaffected():
    p_flat = build_params(ModelConfig(preset="dev", geometry="substrate", dx=5e-9, r2=80e-9,
                                       aspect_ratio=2.0, contact_orientation="short_plane",
                                       initial_overlap=20e-9, t_total=1e-6))
    assert p_flat.sinusoid_wavelength == 0.0
    assert p_flat.sinusoid_amplitude == 0.0
    f, e1, e2, e3 = initialize_fields(p_flat)
    assert f.shape == (p_flat.Ny, p_flat.Nx)  # unchanged construction path


def test_fields_are_finite_and_in_unit_range():
    p = build_params(_cfg())
    f, e1, e2, e3 = initialize_fields(p)
    for arr in (f, e1, e2, e3):
        assert np.all(np.isfinite(arr))
        assert arr.min() >= -1e-6
        assert arr.max() <= 1.0 + 1e-6


def test_e1_e2_do_not_exceed_f():
    # Milestone 15B: eta1=f*(1-phi_GB), eta2=f*phi_GB with phi_GB in [0,1],
    # so e1<=f and e2<=f holds by construction, exactly (not just
    # approximately in a transition zone).
    p = build_params(_cfg())
    f, e1, e2, e3 = initialize_fields(p)
    assert np.all(e1 <= f + 1e-12)
    assert np.all(e2 <= f + 1e-12)


def test_f_contour_follows_the_imposed_sinusoid_away_from_the_particle():
    p = build_params(_cfg())
    f, e1, e2, e3 = initialize_fields(p)
    x = (np.arange(1, p.Nx + 1) - p.Nx / 2) * p.dx
    y = (np.arange(1, p.Ny + 1) - p.Ny / 2) * p.dx

    wall_mean = (p.substrate_wall_frac - .5) * p.Nx * p.dx
    # sample the f=0.5 crossing along a few rows far from the particle (large |Y|)
    # where the substrate's own free surface, not the particle, sets f=0.5.
    for row_idx in (5, p.Ny - 6):
        row = f[row_idx, :]
        # locate the crossing column via linear interpolation
        above = np.where(row > 0.5)[0]
        below = np.where(row <= 0.5)[0]
        if len(above) == 0 or len(below) == 0:
            continue
        ci = above.max() if above.max() < below.max() else above.min()
        # crude crossing estimate
        x_cross = x[ci]
        y_val = y[row_idx]
        x_s_expected = wall_mean + p.sinusoid_amplitude * math.cos(2 * math.pi * y_val / p.sinusoid_wavelength)
        assert abs(x_cross - x_s_expected) < 3 * p.interface_width


def test_particle_centered_at_the_crest():
    p = build_params(_cfg())
    f, e1, e2, e3 = initialize_fields(p)
    y = (np.arange(1, p.Ny + 1) - p.Ny / 2) * p.dx
    Y = np.tile(y[:, None], (1, p.Nx))
    total = float(e2.sum())
    y_centroid = float((e2 * Y).sum()) / total
    assert abs(y_centroid) < p.dx * 5  # centered near Y=0, the crest


def test_geometry_resolves_tjs_and_stress():
    p = build_params(_cfg())
    f, e1, e2, e3 = initialize_fields(p)
    s = Sink(threshold=math.inf)
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    assert not stop, reason
    assert math.isfinite(st.sigma)

    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
    assert sub.L_contact_TJ_sub > 0
    assert sub.top.resolved and sub.bottom.resolved


def test_v1_v2_positive_and_reasonable():
    p = build_params(_cfg())
    f, e1, e2, e3 = initialize_fields(p)
    V1 = float(e1.sum()) * p.dx * p.dx
    V2 = float(e2.sum()) * p.dx * p.dx
    assert V1 > 0
    assert V2 > 0
    # V2 should be in the right ballpark for a particle of radius R2 (short_plane aspect 2)
    expected_order = math.pi * p.Rx * p.Ry
    assert 0.3 * expected_order < V2 < 1.2 * expected_order


def test_gb_ownership_split_uses_exact_signed_distance_and_obstacle_profile():
    # Milestone 15B Sections 2-3: eta1=f*(1-phi_GB), eta2=f*phi_GB, with
    # phi_GB=obstacle_profile(d_GB, ell) and d_GB the EXACT nearest-point
    # signed Euclidean distance to the sinusoidal GB curve x=x_s(y) (not
    # the raw horizontal difference X-x_s(Y), which is not the normal
    # distance wherever the curve's slope is nonzero). eta2 additionally
    # caps at the particle's own raw level set e2_raw (min(f*phi_GB,
    # e2_raw)) -- phi_GB alone, unconditionally, assigns spurious grain-2
    # ownership deep in pure-substrate bulk far from the particle, where
    # e2_raw~0 (see model.py's initialize_fields docstring).
    p = build_params(_cfg(gamma_gb_override=1.0))
    f, e1, e2, e3 = initialize_fields(p)
    x = (np.arange(1, p.Nx + 1) - p.Nx / 2) * p.dx
    y = (np.arange(1, p.Ny + 1) - p.Ny / 2) * p.dx
    X, Y = np.meshgrid(x, y)
    W = p.interface_width
    wall_mean = (p.substrate_wall_frac - .5) * p.Nx * p.dx
    x_s = wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)
    x_crest = wall_mean + p.sinusoid_amplitude * math.cos(p.sinusoid_phase)
    cx = x_crest + p.Rx - p.initial_overlap
    rr = np.sqrt(((X - cx) / p.Rx) ** 2 + (Y / p.Ry) ** 2)
    e2_raw = .5 * (1 - np.tanh((rr - 1) * min(p.Rx, p.Ry) / W))
    ell = obstacle_ell(p.k_eta, p.W_cpl_f)
    d_GB = sinusoid_signed_distance(X, Y, p.sinusoid_amplitude, p.sinusoid_wavelength, p.sinusoid_phase, wall_mean)
    phi_GB_expected = obstacle_profile(d_GB, ell)
    e2_expected = np.minimum(f * phi_GB_expected, e2_raw)
    assert np.allclose(e2, e2_expected, atol=1e-12)
    assert np.allclose(e1, f - e2_expected, atol=1e-12)


def test_eta_sum_matches_f_exactly_everywhere():
    # Milestone 15B Section 2 invariant: eta1=f*(1-phi_GB), eta2=f*phi_GB
    # makes sum(eta_i)=f identically, for ANY phi_GB in [0,1] -- not just
    # in the solid core (Milestone 15's construction, e1_raw*t1+e2_raw*t2,
    # only achieved that approximately, with a real deficit of up to ~0.3
    # in the free-surface diffuse tail near the TJ; this construction has
    # no such deficit anywhere).
    p = build_params(_cfg(gamma_gb_override=1.0))
    f, e1, e2, e3 = initialize_fields(p)
    resid = np.abs((e1 + e2 + e3) - np.clip(f, 0.0, 1.0))
    assert resid.max() < 1e-9


def test_eta_nonnegative_everywhere():
    p = build_params(_cfg(gamma_gb_override=1.0))
    f, e1, e2, e3 = initialize_fields(p)
    assert np.all(e1 >= -1e-12)
    assert np.all(e2 >= -1e-12)


def test_tj_position_unchanged_relative_to_tanh_split_reconstruction():
    # Milestone 15B: swapping the ownership-split construction (Milestone
    # 15's e1_raw*t1/e2_raw*t2, tanh-based coordinate -> this milestone's
    # f*(1-phi_GB)/f*phi_GB, exact-signed-distance-based) must not move
    # the located TJs -- f (e1_raw, e2_raw, the union f=max(e1_raw,e2_raw),
    # particle/substrate geometry) is untouched by the split construction,
    # and locate_neck_tjs's neck-column search is f-driven.
    p = build_params(_cfg(gamma_gb_override=1.0))
    f, e1, e2, e3 = initialize_fields(p)
    tj_new = locate_neck_tjs(f, e1, e2, p)

    x = (np.arange(1, p.Nx + 1) - p.Nx / 2) * p.dx
    y = (np.arange(1, p.Ny + 1) - p.Ny / 2) * p.dx
    X, Y = np.meshgrid(x, y)
    W = p.interface_width
    wall_mean = (p.substrate_wall_frac - .5) * p.Nx * p.dx
    x_s = wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)
    e1_raw = .5 * (1 - np.tanh((X - x_s) / W))
    x_crest = wall_mean + p.sinusoid_amplitude * math.cos(p.sinusoid_phase)
    cx = x_crest + p.Rx - p.initial_overlap
    rr = np.sqrt(((X - cx) / p.Rx) ** 2 + (Y / p.Ry) ** 2)
    e2_raw = .5 * (1 - np.tanh((rr - 1) * min(p.Rx, p.Ry) / W))
    f_old = np.maximum(e1_raw, e2_raw)
    t1_old = .5 * (1 - np.tanh((X - x_s) / W))
    t2_old = .5 * (1 + np.tanh((X - x_s) / W))
    tj_old = locate_neck_tjs(f_old, e1_raw * t1_old, e2_raw * t2_old, p)

    assert np.array_equal(f, f_old)
    assert np.allclose(tj_new["tj_top"], tj_old["tj_top"])
    assert np.allclose(tj_new["tj_bottom"], tj_old["tj_bottom"])
    assert tj_new["nc"] == tj_old["nc"]


def test_gb_phi_half_contour_matches_macroscopic_gb_curve():
    # Milestone 15B Section 3: the phi_GB=0.5 contour (where d_GB=0, i.e.
    # exactly on the GB curve x=x_s(y)) must coincide with the SAME
    # macroscopic GB location the old (Milestone 15) tanh-coordinate
    # split used (X=x_s(Y), the substrate's own free-surface line) --
    # confirmed by checking d_GB's own zero-crossing sits on x_s(Y) to
    # near machine precision (the Newton/coarse-search solver's own
    # accuracy), independent of any grid discretization.
    p = build_params(_cfg())
    W = p.interface_width
    y_probe = np.array([-100e-9, -30e-9, 0.0, 40e-9, 120e-9])
    wall_mean = (p.substrate_wall_frac - .5) * p.Nx * p.dx
    x_s_probe = wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * y_probe / p.sinusoid_wavelength + p.sinusoid_phase)
    d0 = sinusoid_signed_distance(x_s_probe, y_probe, p.sinusoid_amplitude, p.sinusoid_wavelength,
                                   p.sinusoid_phase, wall_mean)
    assert np.max(np.abs(d0)) < 1e-9 * W  # a point ON the curve has zero signed distance to it


def test_signed_distance_matches_brute_force_nearest_point():
    # Independent check of gb_signed_distance.sinusoid_signed_distance
    # against brute-force fine sampling of the curve, at this milestone's
    # actual geometry (large amplitude/wavelength ratio -> large slope,
    # where the raw-horizontal-difference approximation would be poor).
    A, lam, phase, x_mean = 100e-9, 320e-9, 0.0, 0.0
    rng = np.random.default_rng(0)
    Xs = rng.uniform(-150e-9, 150e-9, 20)
    Ys = rng.uniform(-150e-9, 150e-9, 20)
    d = sinusoid_signed_distance(Xs, Ys, A, lam, phase, x_mean)

    def brute(x0, y0, n=200000):
        Yp = np.linspace(y0 - 2 * lam, y0 + 2 * lam, n)
        xs = x_mean + A * np.cos(2 * math.pi * Yp / lam + phase)
        d2 = (x0 - xs) ** 2 + (y0 - Yp) ** 2
        i = np.argmin(d2)
        return math.copysign(math.sqrt(d2[i]), x0 - xs[i])

    for i in range(len(Xs)):
        expected = brute(Xs[i], Ys[i])
        assert math.isclose(d[i], expected, abs_tol=1e-9)


def test_custom_wavelength_and_amplitude_respected():
    p = build_params(_cfg(sinusoid_wavelength=400e-9, sinusoid_amplitude=15e-9))
    assert math.isclose(p.sinusoid_wavelength, 400e-9, rel_tol=0.05)  # snapped to Ny*dx
    assert p.sinusoid_amplitude == 15e-9
    f, e1, e2, e3 = initialize_fields(p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
