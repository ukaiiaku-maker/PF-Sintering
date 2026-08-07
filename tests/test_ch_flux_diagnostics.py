import math

import numpy as np

from pf_sintering.ch_flux_diagnostics import evolve_f_diagnostic, surface_flux_near_tj
from pf_sintering.model import ModelConfig, Sink, build_params, evolve_f, initialize_fields
from pf_sintering.tj_force import locate_neck_tjs


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


def test_diagnostic_kernel_matches_production_exactly():
    p, f, e1, e2, e3 = _geometry()
    s = Sink(threshold=1.0)
    f1 = evolve_f(f.copy(), e1, e2, e3, s, Sink(), p)
    f2, diag = evolve_f_diagnostic(f.copy(), e1, e2, e3, s, p)
    assert np.array_equal(f1, f2)
    assert np.allclose(f2, f + diag.df)
    assert np.all(np.isfinite(diag.mu))
    assert np.all(diag.M >= 0)


def test_surface_flux_near_tj_returns_finite_components():
    p, f, e1, e2, e3 = _geometry()
    s = Sink(threshold=1.0)
    _, diag = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    tjs = locate_neck_tjs(f, e1, e2, p)
    assert tjs is not None
    res = surface_flux_near_tj(f, diag.Jx, diag.Jy, tjs["tj_top"], p)
    assert res is not None
    assert math.isfinite(res["J_tangent"])
    assert math.isfinite(res["J_normal"])
    assert res["n_points"] >= 3


def test_surface_flux_tangent_points_away_from_tj():
    # Sign convention: tangent must be oriented away from the TJ (into the
    # bulk of that branch), matching tj_force._branch_dir's convention, so
    # J_tangent's sign is physically meaningful (positive = flux leaving the
    # neck) rather than an arbitrary SVD sign.
    p, f, e1, e2, e3 = _geometry()
    s = Sink(threshold=1.0)
    _, diag = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    tjs = locate_neck_tjs(f, e1, e2, p)
    assert tjs is not None
    for name in ("tj_top", "tj_bottom"):
        res = surface_flux_near_tj(f, diag.Jx, diag.Jy, tjs[name], p)
        assert res is not None
        tangent = np.array(res["tangent"])
        # Re-derive the band centroid the same way surface_flux_near_tj does
        # and confirm the tangent has a non-negative component away from TJ.
        from skimage.measure import find_contours
        pts = []
        for rc in find_contours(f, 0.5):
            pts.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
        P = np.vstack(pts)
        d = np.hypot(P[:, 0] - tjs[name][0], P[:, 1] - tjs[name][1])
        band = P[(d >= 1.0 * p.interface_width) & (d <= 3.0 * p.interface_width)]
        away = band.mean(0) - np.asarray(tjs[name])
        assert np.dot(tangent, away) >= -1e-12
