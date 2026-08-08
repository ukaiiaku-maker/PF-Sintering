import math

import numpy as np

from pf_sintering.ch_crossover_diagnostics import (
    g_ch_probe,
    neck_ch_mass_balance,
    neck_region_mask,
    trace_branch_profile,
)
from pf_sintering.ch_flux_diagnostics import evolve_f_diagnostic
from pf_sintering.model import ModelConfig, Sink, build_params, initialize_fields
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact


def _state():
    p = build_params(ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=20e-9,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    return p, f, e1, e2, e3


def test_g_ch_probe_resolves_and_is_dt_stable():
    p, f, e1, e2, e3 = _state()
    s = Sink(threshold=math.inf)
    probe = g_ch_probe(f, e1, e2, e3, s, p)
    assert probe.resolved
    assert math.isfinite(probe.L_contact_before)
    vals = [probe.g_ch[frac] for frac in (1.0, 0.5, 0.25)]
    assert all(math.isfinite(v) for v in vals)
    # not a timestep artifact: same sign at every fraction, values close
    signs = {v > 0 for v in vals}
    assert len(signs) == 1
    spread = (max(vals) - min(vals)) / abs(vals[0])
    assert spread < 0.01


def test_g_ch_probe_does_not_mutate_inputs():
    p, f, e1, e2, e3 = _state()
    s = Sink(threshold=math.inf)
    f_orig, e1_orig = f.copy(), e1.copy()
    g_ch_probe(f, e1, e2, e3, s, p)
    assert np.array_equal(f, f_orig)
    assert np.array_equal(e1, e1_orig)


def test_neck_region_mask_is_localized_and_nonempty():
    p, f, e1, e2, e3 = _state()
    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
    mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
    assert 0 < mask.sum() < mask.size  # localized, not the whole domain


def test_neck_ch_mass_balance_matches_direct_sum():
    p, f, e1, e2, e3 = _state()
    s = Sink(threshold=math.inf)
    sub = compute_subgrid_contact(f, e1, e2, p)
    mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
    f_after, _ = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    mb = neck_ch_mass_balance(f, f_after, mask, p)
    expected = float(((f_after - f) * mask).sum()) * p.dx * p.dx
    assert math.isclose(mb, expected, rel_tol=1e-12)


def test_neck_mass_balance_sign_matches_g_ch_sign_at_start():
    # At t=0, g_CH > 0 (contact widening) -- the neck region should be
    # gaining, not losing, mass over the same CH-only step.
    p, f, e1, e2, e3 = _state()
    s = Sink(threshold=math.inf)
    probe = g_ch_probe(f, e1, e2, e3, s, p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
    f_after, _ = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    mb = neck_ch_mass_balance(f, f_after, mask, p)
    assert probe.g_ch[1.0] > 0
    assert mb > 0


def test_trace_branch_profile_resolves_with_sensible_arclength():
    p, f, e1, e2, e3 = _state()
    s = Sink(threshold=math.inf)
    f_after, diag = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    assert rep.top.resolved
    prof = trace_branch_profile(f, diag.mu, diag.Jx, diag.Jy, p, rep.top.tj_xy, rep.top.v_s1,
                                 max_arclength=150e-9, n_samples=20)
    assert prof is not None
    s_arr = np.array(prof["s"])
    assert s_arr[0] == 0.0
    assert s_arr[-1] <= 150e-9 + 1e-15
    assert np.all(np.diff(s_arr) >= 0)  # monotonically increasing arclength
    for key in ("mu", "J_tangent", "J_normal", "kappa", "theta", "dJ_tangent_ds"):
        assert len(prof[key]) == len(prof["s"])
        assert all(math.isfinite(v) for v in prof[key])


def test_trace_branch_profile_returns_none_far_from_any_contour():
    p, f, e1, e2, e3 = _state()
    s = Sink(threshold=math.inf)
    f_after, diag = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    far_xy = (1e-6, 1e-6)  # well outside the domain
    prof = trace_branch_profile(f, diag.mu, diag.Jx, diag.Jy, p, far_xy, (1.0, 0.0),
                                 max_arclength=50e-9, n_samples=10)
    assert prof is None
