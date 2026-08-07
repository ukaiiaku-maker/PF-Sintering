import numpy as np

from pf_sintering.model import (
    ModelConfig,
    build_params,
    initialize_fields,
    ostwald_substrate,
    overlap_col,
)


def test_eta_mobility_scale_is_independent_and_multiplicative():
    p1 = build_params(ModelConfig(preset="dev", nx=64, ny=64, t_total=1e-6))
    p_half = build_params(ModelConfig(preset="dev", nx=64, ny=64, t_total=1e-6, eta_mobility_scale=0.5))
    p_zero = build_params(ModelConfig(preset="dev", nx=64, ny=64, t_total=1e-6, eta_mobility_scale=0.0))
    assert np.isclose(p_half.M_eta, 0.5 * p1.M_eta)
    assert p_zero.M_eta == 0.0
    # M_f (free-surface mobility) must be unaffected by the GB/eta control.
    assert p_half.M_f == p1.M_f == p_zero.M_f


def test_reservoir_neck_unprotected_increases_near_neck_removal_fraction():
    def near_neck_removed_fraction(unprotected):
        p = build_params(ModelConfig(
            preset="dev", nx=96, ny=128, dx=5e-9, r2=80e-9, t_total=1e-6,
            reservoir_neck_unprotected=unprotected,
        ))
        f, e1, e2, e3 = initialize_fields(p)
        cr = p.Ny // 2
        _, col = overlap_col(e1[cr] * e2[cr])
        nc = int(round(col)) - 1
        _, e1b, e2b, _ = ostwald_substrate(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
        removed = e2 - e2b
        win = max(3, round(3 * p.interface_width / p.dx))
        near = slice(max(0, nc - win), nc + win + 1)
        total = float(removed.sum())
        assert total > 0
        return float(removed[:, near].sum()) / total

    protected_frac = near_neck_removed_fraction(False)
    unprotected_frac = near_neck_removed_fraction(True)
    assert unprotected_frac > protected_frac


def test_reservoir_default_preserves_baseline_localization():
    p_default = build_params(ModelConfig(preset="dev", nx=64, ny=64, t_total=1e-6))
    p_explicit_off = build_params(ModelConfig(preset="dev", nx=64, ny=64, t_total=1e-6, reservoir_neck_unprotected=False))
    assert p_default.reservoir_neck_unprotected is False
    f, e1, e2, e3 = initialize_fields(p_default)
    out_default = ostwald_substrate(f.copy(), e1.copy(), e2.copy(), e3.copy(), p_default)
    out_explicit = ostwald_substrate(f.copy(), e1.copy(), e2.copy(), e3.copy(), p_explicit_off)
    for a, b in zip(out_default, out_explicit):
        assert np.allclose(a, b)
