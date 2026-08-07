import math

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.operator_ledger import OPERATORS, run_ledger_trajectory, summarize_ledger


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


def test_ledger_never_activates_sink_and_holds_strain_zero():
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=500)
    assert len(steps) >= 2
    for ledger in steps:
        for stage in ledger.stage_samples.values():
            assert stage["sink_active"] == 0
            assert stage["strain"] == 0.0


def test_raw_CH_step_cannot_change_x_neck_or_A_GB_by_construction():
    # x_neck/A_GB depend only on e1,e2 (contact_width/contact_area), which
    # the raw CH step (evolve_f) never touches -- this must hold exactly,
    # not approximately.
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=500)
    for ledger in steps:
        d = ledger.operator_deltas.get("CH")
        if d is None:
            continue
        assert d["x_neck_m"] == 0.0
        assert d["A_GB_m2"] == 0.0


def test_V2_change_is_entirely_attributed_to_ostwald():
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=500)
    summ = summarize_ledger(steps)
    total_v2 = summ["total_observed"]["V2"]
    assert total_v2 < 0  # coarsening: V2 decreases
    for op in OPERATORS:
        if op == "Ostwald":
            continue
        assert abs(summ["totals"][op]["V2"]) < 1e-6 * abs(total_v2)


def test_ledger_closure_is_exact():
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    steps = run_ledger_trajectory(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=500)
    summ = summarize_ledger(steps)
    for k, err in summ["closure_error"].items():
        if err is None:
            continue
        assert err == 0.0, f"{k} closure error {err} is not exactly zero"
