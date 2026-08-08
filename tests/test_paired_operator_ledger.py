import math

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.operator_ledger import OPERATORS
from pf_sintering.paired_operator_ledger import (
    LEDGER_PAIRED_KEYS,
    run_paired_operator_ledger_trajectory,
    summarize_paired_ledger,
)


def _geometry(**overrides):
    p = build_params(ModelConfig(
        preset="dev", nx=96, ny=64, dx=5e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        coarsening_rate_scale=3.0, surface_mobility_scale=0.3, eta_mobility_scale=1.0,
        **overrides,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    return p, f, e1, e2, e3


def _c0_params():
    return build_params(ModelConfig(
        preset="dev", nx=96, ny=64, dx=5e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        coarsening_rate_scale=0.0, surface_mobility_scale=0.3, eta_mobility_scale=1.0,
    ))


def test_paired_ledger_closure_is_exact():
    p1, f0, e1_0, e2_0, e3_0 = _geometry()
    p0 = _c0_params()
    steps, l0, l1 = run_paired_operator_ledger_trajectory(p0, p1, f0, e1_0, e2_0, e3_0,
                                                           target_dv2_frac=5e-5, max_steps=40)
    summ = summarize_paired_ledger(steps)
    for k, err in summ["closure_error"].items():
        if err is None:
            continue
        assert err == 0.0, f"{k} closure error {err} is not exactly zero"


def test_paired_ledger_total_matches_differential_coarsening_delta():
    # The paired ledger's total_observed L_contact_TJ_sub change must equal
    # last-step delta_L_coarsening minus first-step delta_L_coarsening
    # (== last, since both start identical) from the plain (non-ledger)
    # paired-trajectory machinery -- same physics, two different
    # decompositions of the same quantity.
    from pf_sintering.differential_coarsening import paired_delta_series, run_paired_trajectory

    p1, f0, e1_0, e2_0, e3_0 = _geometry()
    p0 = _c0_params()
    steps, l0, l1 = run_paired_operator_ledger_trajectory(p0, p1, f0, e1_0, e2_0, e3_0,
                                                           target_dv2_frac=5e-5, max_steps=40)
    summ = summarize_paired_ledger(steps)
    n = summ["n_steps"]

    rows_c0, rows_c1, reason = run_paired_trajectory(p0, p1, f0, e1_0, e2_0, e3_0,
                                                       target_dv2_frac=5e-5, max_steps=40)
    deltas = paired_delta_series(rows_c0, rows_c1)
    assert math.isclose(summ["total_observed"]["L_contact_TJ_sub"], deltas[n]["L_contact_TJ_sub"], abs_tol=1e-20)


def test_paired_ledger_operator_totals_cover_all_five_operators():
    p1, f0, e1_0, e2_0, e3_0 = _geometry()
    p0 = _c0_params()
    steps, l0, l1 = run_paired_operator_ledger_trajectory(p0, p1, f0, e1_0, e2_0, e3_0,
                                                           target_dv2_frac=5e-5, max_steps=20)
    summ = summarize_paired_ledger(steps)
    assert set(summ["totals"].keys()) == set(OPERATORS)
    for op in OPERATORS:
        assert "L_contact_TJ_sub" in summ["totals"][op]


def test_ostwald_fn_override_changes_c1_trajectory():
    from pf_sintering.differential_coarsening import run_single_trajectory
    from pf_sintering.ostwald_diagnostics import ostwald_removal_only

    p1, f0, e1_0, e2_0, e3_0 = _geometry()
    rows_normal, stop_n, _ = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, 20)
    rows_removal_only, stop_r, _ = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, 20,
                                                          ostwald_fn=ostwald_removal_only)
    assert not stop_n and not stop_r
    # removal-only doesn't deposit on e1/f, so V2 should differ from the O0 case.
    assert rows_normal[-1]["V2"] != rows_removal_only[-1]["V2"]


def test_ostwald_fn_default_matches_production_step():
    from pf_sintering.differential_coarsening import run_single_trajectory

    p1, f0, e1_0, e2_0, e3_0 = _geometry()
    rows_default, stop_d, _ = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, 20)
    from pf_sintering.model import ostwald_substrate
    rows_explicit, stop_e, _ = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, 20, ostwald_fn=ostwald_substrate)
    assert rows_default[-1]["L_contact_TJ_sub"] == rows_explicit[-1]["L_contact_TJ_sub"]
