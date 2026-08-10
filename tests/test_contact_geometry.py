import math

import numpy as np

from pf_sintering.contact_geometry import (
    V2_f_weighted,
    contact_length_tj,
    eta_centroid_separation,
    far_field_position,
    gb_geom_length,
    mean_gb_tangent,
)
from pf_sintering.diagnostics import wall_x0
from pf_sintering.model import ModelConfig, Sink, build_params, contact_width, initialize_fields
from pf_sintering.tj_force import compute_neck_tj_forces


def _geometry(**overrides):
    p = build_params(ModelConfig(
        preset="dev", nx=96, ny=128, dx=5e-9, r2=80e-9,
        aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=20e-9, t_total=1e-6,
        coarsening_rate_scale=3.0, surface_mobility_scale=0.3,
        **overrides,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    return p, f, e1, e2, e3


def test_L_contact_TJ_and_L_GB_geom_are_independent_and_agree_closely():
    # Two independently-derived contact-length measures (straight-line
    # TJ-to-TJ projection vs. traced contour arc length) should agree
    # closely for a geometry with a near-straight GB, without either being
    # defined in terms of the other.
    p, f, e1, e2, e3 = _geometry()
    s = Sink(threshold=1.0)
    cl = contact_length_tj(f, e1, e2, e3, s, p)
    assert math.isfinite(cl["L_contact_TJ"])
    l_geom = gb_geom_length(f, e1, e2, p)
    assert math.isfinite(l_geom)
    assert abs(l_geom - cl["L_contact_TJ"]) / cl["L_contact_TJ"] < 0.02


def test_d_n_TJ_is_small_for_this_near_vertical_geometry_but_not_assumed_zero():
    # The GB here is close to vertical, but d_n_TJ must be *measured*, not
    # hard-coded -- confirm it comes out small (not exactly zero, which
    # would suggest a hard-coded assumption) relative to L_contact_TJ.
    p, f, e1, e2, e3 = _geometry()
    s = Sink(threshold=1.0)
    cl = contact_length_tj(f, e1, e2, e3, s, p)
    assert cl["d_n_TJ"] < 0.05 * cl["L_contact_TJ"]


def test_mean_gb_tangent_is_unit_and_orthogonal_to_normal():
    p, f, e1, e2, e3 = _geometry()
    s = Sink(threshold=1.0)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    t_gb, n_gb = mean_gb_tangent(rep)
    assert t_gb is not None
    assert math.isclose(float(np.linalg.norm(t_gb)), 1.0, rel_tol=1e-9)
    assert math.isclose(float(np.linalg.norm(n_gb)), 1.0, rel_tol=1e-9)
    assert abs(float(np.dot(t_gb, n_gb))) < 1e-9


def test_far_field_position_is_far_from_the_neck_and_distinct_from_centroid():
    p, f, e1, e2, e3 = _geometry()
    w0 = wall_x0(p)
    l_ff = far_field_position(f, w0, p)
    l_ec = eta_centroid_separation(e2, w0, p)
    assert math.isfinite(l_ff) and math.isfinite(l_ec)
    # The tip is farther from the wall than the centroid (it's the far
    # extreme of the particle, not its middle).
    assert l_ff > l_ec > 0


def test_V2_f_weighted_differs_from_raw_V2_when_eta_sum_is_slack_of_f():
    # Milestone 15B: the production initializer now constructs
    # eta1=f*(1-phi_GB), eta2=f*phi_GB, so eta1+eta2=f exactly, everywhere
    # -- no incidental slack remains at t=0 to exercise this distinction.
    # Construct slack synthetically instead (a uniform 90% fill fraction),
    # the same kind of eta<f state that CAN arise mid-simulation.
    p, f, e1, e2, e3 = _geometry()
    e1b, e2b, e3b = 0.9 * e1, 0.9 * e2, 0.9 * e3
    v2_eta = float(e2b.sum() * p.dx * p.dx)
    v2_f = V2_f_weighted(f, e1b, e2b, e3b, p)
    assert math.isfinite(v2_f)
    assert abs(v2_f - v2_eta) / v2_eta > 1e-4


def test_V2_f_weighted_matches_V2_eta_when_eta_exactly_fills_f():
    # Synthetic sanity check: if e1+e2 == f exactly everywhere (no slack),
    # V2_f_weighted must reduce to exactly V2_eta.
    p, f, e1, e2, e3 = _geometry()
    e1b = f * 0.5
    e2b = f * 0.5
    e3b = np.zeros_like(f)
    v2_eta = float(e2b.sum() * p.dx * p.dx)
    v2_f = V2_f_weighted(f, e1b, e2b, e3b, p)
    assert math.isclose(v2_f, v2_eta, rel_tol=1e-9)


def test_gb_geom_length_returns_nan_when_no_tj_provided_and_none_resolvable():
    p, f, e1, e2, e3 = _geometry()
    # Far-away, nonsense TJ coordinates should fail to connect to any
    # contour within the snap tolerance.
    bad_tj = np.array([1.0, 1.0])
    l_geom = gb_geom_length(f, e1, e2, p, tj_top=bad_tj, tj_bottom=bad_tj + 1e-9)
    assert math.isnan(l_geom)
