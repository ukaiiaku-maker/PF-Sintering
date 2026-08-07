import numpy as np

from pf_sintering.model import ModelConfig, build_params, initialize_fields, ostwald_substrate
from pf_sintering.ostwald_diagnostics import (
    centroid,
    near_neck_fractions,
    ostwald_addition_only,
    ostwald_removal_only,
    ostwald_substrate_diagnostic,
)
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
    f1, e11, e21, e31 = ostwald_substrate(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
    f2, e12, e22, e32, diag = ostwald_substrate_diagnostic(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
    assert np.array_equal(f1, f2)
    assert np.array_equal(e11, e12)
    assert np.array_equal(e21, e22)
    assert np.array_equal(e31, e32)
    assert diag.actual_removed > 0
    assert np.isclose(diag.actual_removed, diag.actual_added, rtol=1e-9)


def test_removal_only_matches_production_e2_but_not_e1():
    p, f, e1, e2, e3 = _geometry()
    f0, e1_prod, e2_prod, e3_prod = ostwald_substrate(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
    f_r, e1_r, e2_r, e3_r, reservoir = ostwald_removal_only(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
    # e2 side matches production exactly (same removal field).
    assert np.array_equal(e2_r, e2_prod)
    # e1 (destination) is untouched in the removal-only diagnostic.
    assert np.array_equal(e1_r, e1)
    assert reservoir > 0
    assert np.isclose(float(f.sum() - f_r.sum()), reservoir, rtol=1e-9)


def test_addition_only_matches_production_e1_but_not_e2():
    p, f, e1, e2, e3 = _geometry()
    f0, e1_prod, e2_prod, e3_prod = ostwald_substrate(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
    f_a, e1_a, e2_a, e3_a, artificial = ostwald_addition_only(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
    assert np.array_equal(e1_a, e1_prod)
    assert np.array_equal(e2_a, e2)  # untouched: no removal in this diagnostic
    assert artificial > 0
    # Deliberately non-conservative: total f mass increases by the artificial amount.
    assert np.isclose(float(f_a.sum() - f.sum()), artificial, rtol=1e-9)


def test_near_neck_fractions_sum_reasonably_and_are_bounded():
    p, f, e1, e2, e3 = _geometry()
    _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    tjs = locate_neck_tjs(f, e1, e2, p)
    assert tjs is not None
    fr = near_neck_fractions(diag.remove, tjs["tj_top"], tjs["tj_bottom"], p)
    for k, v in fr.items():
        assert 0.0 <= v <= 1.0 + 1e-9, (k, v)
    # Wider bands must contain at least as much mass as narrower ones.
    assert fr["frac_within_1W"] <= fr["frac_within_2W"] <= fr["frac_within_3W"] <= fr["frac_within_5W"]


def test_centroid_is_finite_and_within_domain():
    p, f, e1, e2, e3 = _geometry()
    _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    cx, cy = centroid(diag.remove, p)
    assert 0 < cx < p.Nx * p.dx
    assert 0 < cy < p.Ny * p.dx
