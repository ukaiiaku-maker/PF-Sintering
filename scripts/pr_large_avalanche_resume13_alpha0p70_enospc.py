#!/usr/bin/env python3
"""Resume avalanche-3 reload after the resume-12 ENOSPC flush.

The exact passive PF state, root hazard and threshold, RNG stream, event
physics, 1.50 eV initial descendant lowering, and h[j+1]=0.70*h[j] decay are
preserved.  This wrapper changes only output/restart bookkeeping.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_large_avalanche_resume11_alpha0p70_enospc as base  # noqa: E402


SOURCE = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_"
    "resume12_macro_alpha0p70_coarse_diagnostics")
OUT = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_"
    "resume13_macro_alpha0p70_enospc")
PREFIX = ROOT / "runs/pr_large_avalanche_resume13_prefix_through_a3_reload.csv"
SEGMENT_LABEL = "avalanche3_alpha0p70_enospc_atomic_resume13"

PASSIVE_ACCEPTED_MACRO_DT_MODEL = 1.0
PASSIVE_INTERNAL_MONITOR_DT_MODEL = 0.025
SPARSE_COPY_DIAGNOSTIC_DT_MODEL = 25.0

SPARSE_PROBE_KEYS = (
    "reaction_probe_available",
    "reaction_probe_q_over_b",
    "reaction_probe_copy_only",
    "reaction_probe_state_mutated",
    "reaction_probe_volume_conserving",
    "reaction_probe_delta_G_J",
    "Fq_reaction_N",
    "dVsource_dq_m2",
    "delta_mu_q_Pa",
    "reaction_probe_event_C4",
    "reaction_probe_event_integrator",
)


def main() -> None:
    base.SOURCE = SOURCE
    base.CHECKPOINT = SOURCE / "checkpoints/avalanche3_waiting_latest.npz"
    base.OUT = OUT
    base.PREFIX = PREFIX
    base.SEGMENT_LABEL = SEGMENT_LABEL
    base.PASSIVE_MACRO_DT_MODEL = PASSIVE_ACCEPTED_MACRO_DT_MODEL
    base.HAZARD_QUADRATURE_DT_MODEL = PASSIVE_INTERNAL_MONITOR_DT_MODEL
    base.INITIAL_DESCENDANT_LOWERING_EV = 1.50
    base.DECAY_ALPHA = 0.70

    original_production_main = base.production.main

    def production_main_after_enospc() -> None:
        with np.load(base.CHECKPOINT, allow_pickle=False) as saved:
            checkpoint_t = float(saved["t_model"])
        rows = base.helpers.typed_rows(
            SOURCE / "avalanche_renewal_history.csv")
        tolerance = 64.0 * math.ulp(max(abs(checkpoint_t), 1.0))
        matching = [row for row in rows if math.isclose(
            float(row["t_model"]), checkpoint_t,
            rel_tol=0.0, abs_tol=tolerance)]
        if len(matching) != 1:
            raise RuntimeError(
                "checkpoint does not map to one sparse-probe source row")
        preserved_probe = {
            key: matching[0][key] for key in SPARSE_PROBE_KEYS
            if key in matching[0]
        }
        if len(preserved_probe) != len(SPARSE_PROBE_KEYS):
            missing = sorted(set(SPARSE_PROBE_KEYS) - set(preserved_probe))
            raise RuntimeError(f"sparse-probe ledger is incomplete: {missing}")

        base.production.CONTINUATION["sparse_thermo_probe_result"] = (
            preserved_probe)
        base.production.CONTINUATION["continuation_reason"] = (
            "ENOSPC during resume-12 scalar-history flush after an exact "
            "atomic avalanche-3 passive checkpoint; storage recovery only")
        base.renewal.SPARSE_THERMO_PROBE_DT_MODEL = (
            SPARSE_COPY_DIAGNOSTIC_DT_MODEL)
        base.renewal.ROOT_ANALYSIS_DT = PASSIVE_ACCEPTED_MACRO_DT_MODEL
        base.renewal.ROOT_MOVIE_DT = PASSIVE_ACCEPTED_MACRO_DT_MODEL
        base.production.EXTRA_RUN_METADATA.update(dict(
            restart_reason=(
                "resume-12 scalar CSV flush failed with errno 28 after exact "
                "atomic checkpoint; output storage recovered only"),
            physical_parameters_changed=False,
            stochastic_parameters_changed=False,
            initial_descendant_lowering_eV=1.50,
            descendant_decay_alpha=0.70,
            intended_descendant_decay_verified="0.70",
            sparse_copy_diagnostic_dt_model=(
                SPARSE_COPY_DIAGNOSTIC_DT_MODEL),
            passive_accepted_macro_dt_model=(
                PASSIVE_ACCEPTED_MACRO_DT_MODEL),
            passive_internal_monitor_dt_model=(
                PASSIVE_INTERNAL_MONITOR_DT_MODEL),
            root_crossing_bisection_preserved=True,
            full_event_copy_diagnostics_preserved=True,
            curvature_monitor_preserved=True,
            output_storage_recovered_only=True,
            diagnostic_cadence_change_only=False))
        original_production_main()

    base.production.main = production_main_after_enospc
    base.main()


if __name__ == "__main__":
    main()
