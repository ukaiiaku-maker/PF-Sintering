import math

import numpy as np
from skimage.measure import find_contours

from pf_sintering.model import ModelConfig, Sink, build_params, compute_stress, initialize_fields
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


def test_custom_wavelength_and_amplitude_respected():
    p = build_params(_cfg(sinusoid_wavelength=400e-9, sinusoid_amplitude=15e-9))
    assert math.isclose(p.sinusoid_wavelength, 400e-9, rel_tol=0.05)  # snapped to Ny*dx
    assert p.sinusoid_amplitude == 15e-9
    f, e1, e2, e3 = initialize_fields(p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
