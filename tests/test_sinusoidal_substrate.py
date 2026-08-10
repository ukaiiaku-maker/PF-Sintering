import math

import numpy as np
from skimage.measure import find_contours

from pf_sintering.gb_obstacle_energy import obstacle_ell, obstacle_profile
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
    # f is np.maximum(e1_raw, e2_raw) of the RAW (pre-ownership-weighting)
    # fields; the returned e1/e2 are weighted by the t1/t2 ownership
    # partition and are strictly <= f, but not necessarily equal to it in
    # the transition zone -- identical, pre-existing behavior to the flat
    # substrate case (same construction pattern), not specific to this
    # geometry.
    p = build_params(_cfg())
    f, e1, e2, e3 = initialize_fields(p)
    assert np.all(e1 <= f + 1e-9)
    assert np.all(e2 <= f + 1e-9)


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


def test_gb_ownership_split_uses_calibrated_obstacle_profile():
    # Milestone 15 Section 2: the t1/t2 ownership split across the GB uses
    # gb_obstacle_energy's compact-support sine (ell=obstacle_ell(p.k_eta,
    # p.W_cpl_f)), not a tanh -- reconstruct t2 directly from the same
    # x_s(Y) coordinate initialize_fields uses and confirm e2/e1_raw
    # (the un-weighted particle level set) matches obstacle_profile exactly.
    p = build_params(_cfg(gamma_gb_override=1.0))
    f, e1, e2, e3 = initialize_fields(p)
    x = (np.arange(1, p.Nx + 1) - p.Nx / 2) * p.dx
    y = (np.arange(1, p.Ny + 1) - p.Ny / 2) * p.dx
    X, Y = np.meshgrid(x, y)
    wall_mean = (p.substrate_wall_frac - .5) * p.Nx * p.dx
    x_s = wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)
    ell = obstacle_ell(p.k_eta, p.W_cpl_f)
    t2_expected = obstacle_profile(X - x_s, ell)
    e1_raw = .5 * (1 - np.tanh((X - x_s) / p.interface_width))
    x_crest = wall_mean + p.sinusoid_amplitude * math.cos(p.sinusoid_phase)
    cx = x_crest + p.Rx - p.initial_overlap
    rr = np.sqrt(((X - cx) / p.Rx) ** 2 + (Y / p.Ry) ** 2)
    e2_raw = .5 * (1 - np.tanh((rr - 1) * min(p.Rx, p.Ry) / p.interface_width))
    assert np.allclose(e2, e2_raw * t2_expected, atol=1e-12)
    assert np.allclose(e1, e1_raw * (1 - t2_expected), atol=1e-12)


def test_eta_sum_matches_f_exactly_in_the_solid_core_near_the_tj():
    # Milestone 15 Section 2 invariant: sum(eta_i)=f holds EXACTLY in the
    # solid core (f>0.99) near the GB/TJ, where the calibrated obstacle
    # split is the only thing determining ownership. It does NOT hold in
    # the free-surface diffuse tail there (where f itself has not
    # saturated and BOTH grains' own raw level sets are still
    # transitioning simultaneously) -- a pre-existing characteristic of
    # this 1D (X-x_s-only) ownership split, present at comparable
    # magnitude with the OLD tanh split too (see
    # test_gb_split_deficit_not_worse_than_tanh_baseline below); fixing
    # that would mean redesigning the split's 2D geometry, out of this
    # milestone's explicit scope ("update ONLY the structural eta
    # initialization").
    p = build_params(_cfg(gamma_gb_override=1.0))
    f, e1, e2, e3 = initialize_fields(p)
    tjs = locate_neck_tjs(f, e1, e2, p)
    assert tjs is not None
    nc = tjs["nc"]
    solid = np.flatnonzero(f[:, nc] > 0.5)
    r_lo, r_hi = solid[0], solid[-1]
    margin = round(40e-9 / p.dx)
    col_margin = round(60e-9 / p.dx)
    rlo, rhi = max(0, r_lo - margin), min(p.Ny, r_hi + margin + 1)
    clo, chi = max(0, nc - col_margin), min(p.Nx, nc + col_margin + 1)
    fb = np.clip(f, 0.0, 1.0)
    core = fb > 0.99
    resid = np.abs((e1 + e2 + e3) - fb)
    sub_resid, sub_core = resid[rlo:rhi, clo:chi], core[rlo:rhi, clo:chi]
    assert sub_core.any()
    assert sub_resid[sub_core].max() < 1e-9


def test_gb_split_deficit_not_worse_than_tanh_baseline():
    # In the free-surface diffuse tail near the TJ (see previous test),
    # sum(eta_i) undershoots f for BOTH the old tanh split and the new
    # calibrated-obstacle split, by comparable magnitude -- a pre-existing
    # limitation of the 1D split, not something this milestone's profile-
    # shape change introduces or meaningfully worsens.
    p = build_params(_cfg(gamma_gb_override=1.0))
    f, e1, e2, e3 = initialize_fields(p)
    tjs = locate_neck_tjs(f, e1, e2, p)
    nc = tjs["nc"]
    solid = np.flatnonzero(f[:, nc] > 0.5)
    r_lo, r_hi = solid[0], solid[-1]
    margin = round(40e-9 / p.dx)
    col_margin = round(60e-9 / p.dx)
    rlo, rhi = max(0, r_lo - margin), min(p.Ny, r_hi + margin + 1)
    clo, chi = max(0, nc - col_margin), min(p.Nx, nc + col_margin + 1)
    resid_new = np.abs((e1 + e2 + e3) - np.clip(f, 0.0, 1.0))[rlo:rhi, clo:chi]

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
    resid_old = np.abs((e1_raw * t1_old + e2_raw * t2_old) - f_old)[rlo:rhi, clo:chi]

    assert resid_new.max() < 2.0 * resid_old.max()


def test_tj_position_unchanged_relative_to_tanh_split_reconstruction():
    # Milestone 15 Section 2: swapping the ownership-split SHAPE (tanh ->
    # calibrated obstacle profile) must not move the located TJs -- f, e1
    # (substrate level set), e2 (particle level set) are untouched by the
    # split, and locate_neck_tjs's neck-column search is f-driven.
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


def test_custom_wavelength_and_amplitude_respected():
    p = build_params(_cfg(sinusoid_wavelength=400e-9, sinusoid_amplitude=15e-9))
    assert math.isclose(p.sinusoid_wavelength, 400e-9, rel_tol=0.05)  # snapped to Ny*dx
    assert p.sinusoid_amplitude == 15e-9
    f, e1, e2, e3 = initialize_fields(p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
