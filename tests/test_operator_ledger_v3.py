import math

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.operator_ledger import OPERATORS, run_ledger_trajectory_v3, summarize_ledger_v3


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


def test_v3_ledger_never_activates_sink_and_holds_strain_zero():
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory_v3(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=30)
    assert len(steps) >= 2
    for ledger in steps:
        for stage in ledger.stage_samples.values():
            assert stage["sink_active"] == 0
            assert stage["strain"] == 0.0


def test_v3_ledger_closure_is_exact():
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory_v3(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=30)
    summ = summarize_ledger_v3(steps)
    for k, err in summ["closure_error"].items():
        if err is None:
            continue
        assert err == 0.0, f"{k} closure error {err} is not exactly zero"


def test_subgrid_contact_resolves_and_varies_smoothly_step_to_step():
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory_v3(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=15)
    values = []
    for ledger in steps:
        v = ledger.stage_samples["start"]["L_contact_TJ_sub"]
        assert math.isfinite(v), "sub-grid contact length should resolve throughout this short trajectory"
        values.append(v)
    # No single-step change should be anywhere near a full grid cell (the
    # failure mode the legacy locator has); this is the core Milestone-6C
    # continuity requirement.
    for i in range(len(values) - 1):
        assert abs(values[i + 1] - values[i]) < 0.5 * p.dx


def test_raw_CH_dominates_subgrid_contact_change_in_this_regime():
    # Empirically established (see MILESTONE_6C report): in the primary
    # case, raw CH's effect on L_contact_TJ_sub is consistently positive and
    # roughly two orders of magnitude larger than any other single operator.
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory_v3(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=10)
    summ = summarize_ledger_v3(steps)
    ch = summ["totals"]["CH"]["L_contact_TJ_sub"]
    assert ch > 0
    for op in OPERATORS:
        if op == "CH":
            continue
        other = abs(summ["totals"][op]["L_contact_TJ_sub"])
        assert other < 0.1 * ch
