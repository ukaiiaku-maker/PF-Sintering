import math

import numpy as np
import pytest

from pf_sintering.model import ModelConfig, build_params, initialize_fields, ostwald_substrate
from pf_sintering.differential_coarsening import (
    paired_delta,
    paired_delta_series,
    run_paired_trajectory,
    run_single_trajectory,
)


def _geometry(coarsening_rate_scale, **overrides):
    p = build_params(ModelConfig(
        preset="dev", nx=96, ny=64, dx=5e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        coarsening_rate_scale=coarsening_rate_scale, surface_mobility_scale=0.3,
        eta_mobility_scale=1.0, **overrides,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    return p, f, e1, e2, e3


def test_coarsening_rate_scale_zero_gives_infinite_tau_ripening():
    p, *_ = _geometry(0.0)
    assert p.tau_ripening == math.inf


def test_ostwald_substrate_is_exact_noop_when_coarsening_off():
    p, f, e1, e2, e3 = _geometry(0.0)
    f2, e1b, e2b, e3b = ostwald_substrate(f.copy(), e1.copy(), e2.copy(), e3.copy(), p)
    assert np.array_equal(f2, f)
    assert np.array_equal(e1b, e1)
    assert np.array_equal(e2b, e2)
    assert np.array_equal(e3b, e3)


def test_c0_c1_share_identical_dt():
    p0, *_ = _geometry(0.0)
    p1, *_ = _geometry(3.0)
    assert p0.dt == p1.dt


def test_paired_trajectory_rejects_mismatched_dt():
    p0, f0, e1_0, e2_0, e3_0 = _geometry(0.0)
    p1, *_ = _geometry(3.0, dt_override=p0.dt / 2)
    with pytest.raises(ValueError):
        run_paired_trajectory(p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=5)


def test_c0_conserves_v2_to_near_machine_precision():
    p0, f0, e1_0, e2_0, e3_0 = _geometry(0.0)
    p1, *_ = _geometry(3.0)
    rows_c0, rows_c1, reason = run_paired_trajectory(
        p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=5e-5, max_steps=60,
    )
    assert reason == ""
    v20 = rows_c0[0]["V2"]
    for row in rows_c0:
        assert abs(row["V2"] - v20) / v20 < 1e-10
    # C1 should show a real, non-trivial V2 change over the same steps.
    assert abs(rows_c1[-1]["V2"] - v20) / v20 > 1e-6


def test_paired_delta_series_is_c1_minus_c0_and_starts_at_zero():
    p0, f0, e1_0, e2_0, e3_0 = _geometry(0.0)
    p1, *_ = _geometry(3.0)
    rows_c0, rows_c1, reason = run_paired_trajectory(
        p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=5e-5, max_steps=60,
    )
    assert reason == ""
    deltas = paired_delta_series(rows_c0, rows_c1)
    assert len(deltas) == len(rows_c0) == len(rows_c1)
    # Identical initial state -> zero delta at step 0 for every paired key.
    for k, v in deltas[0].items():
        assert v == 0.0, f"{k}: expected exact zero delta at step 0, got {v}"
    # Direct spot check against paired_delta on the final row.
    d_last = paired_delta(rows_c0[-1], rows_c1[-1])
    assert d_last["L_contact_TJ_sub"] == deltas[-1]["L_contact_TJ_sub"]


def test_run_single_trajectory_matches_paired_c1_branch():
    # run_single_trajectory (used for the rate-competition series) must
    # reproduce exactly the same per-step state as the C1 branch of
    # run_paired_trajectory for the same config and step count -- both call
    # the same _step_once sequence, so this should be bit-for-bit identical.
    p1, f0, e1_0, e2_0, e3_0 = _geometry(3.0)
    p0, *_ = _geometry(0.0)
    rows_c0, rows_c1, reason = run_paired_trajectory(
        p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=5e-5, max_steps=40,
    )
    assert reason == ""
    n = len(rows_c1) - 1
    rows_single, stop, _ = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, n)
    assert not stop
    assert len(rows_single) == len(rows_c1)
    assert rows_single[-1]["V2"] == rows_c1[-1]["V2"]
    assert rows_single[-1]["L_contact_TJ_sub"] == rows_c1[-1]["L_contact_TJ_sub"]
