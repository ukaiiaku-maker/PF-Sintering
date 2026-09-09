#!/usr/bin/env python3
"""Resume avalanche-3 reload after an ENOSPC scalar-history flush.

Only output/restart bookkeeping changes.  The exact passive PF state,
accumulated root hazard, threshold stream, initial 1.50 eV descendant
lowering, and h[j+1]=0.70*h[j] facilitation decay are preserved.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_renewal_five as renewal  # noqa: E402
import pr_full_corrected_production_campaign as campaign  # noqa: E402
import pr_full_corrected_production_slowgb_large_avalanche as production  # noqa: E402
import pr_large_avalanche_current_state_transfer_from_event2_onset as source_driver  # noqa: E402
import pr_large_avalanche_current_state_transfer_production as replacement  # noqa: E402
import pr_large_avalanche_no_space_resume as helpers  # noqa: E402
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25 as parent  # noqa: E402
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_resume2 as recovery  # noqa: E402
from pr_coarsening_driven_fourier_loading import integral  # noqa: E402


SOURCE = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_"
    "resume10_macro_alpha0p70")
CHECKPOINT = SOURCE / "checkpoints/avalanche3_waiting_latest.npz"
PREVIOUS_EVENT_SOURCE = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_"
    "resume9_macro_all_passive_currentstate")
AVALANCHE1_SOURCE = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_resume3")
OUT = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_"
    "resume11_macro_alpha0p70_enospc")
PREFIX = ROOT / "runs/pr_large_avalanche_resume11_prefix_through_a3_reload.csv"
SEGMENT_LABEL = "avalanche3_alpha0p70_enospc_atomic_resume11"

PASSIVE_MAX_DT_MODEL = 8.086604871225672e-05
PASSIVE_MACRO_DT_MODEL = 0.25
HAZARD_QUADRATURE_DT_MODEL = 0.025
CROSSING_TOLERANCE_MODEL = 1.0e-4
INITIAL_DESCENDANT_LOWERING_EV = 1.50
DECAY_ALPHA = 0.70


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    with np.load(CHECKPOINT, allow_pickle=False) as saved:
        state = tuple(np.asarray(saved[key]).copy()
                      for key in ("f", "particle", "substrate"))
        checkpoint = {key: saved[key].item() for key in saved.files
                      if saved[key].shape == ()}

    manifest = json.loads((SOURCE / "launch_manifest.json").read_text())
    seed_record = json.loads((SOURCE / "rng_seed_record.json").read_text())
    checkpoint_t = float(checkpoint["t_model"])
    checkpoint_h = float(checkpoint["H"])
    checkpoint_hstar = float(checkpoint["Hstar"])
    if not 0.0 <= checkpoint_h < checkpoint_hstar:
        raise RuntimeError("avalanche-3 passive checkpoint is not restartable")

    parent_rows = helpers.typed_rows(SOURCE / "avalanche_renewal_history.csv")
    tolerance = 64.0 * math.ulp(max(abs(checkpoint_t), 1.0))
    matches = [row for row in parent_rows if math.isclose(
        float(row["t_model"]), checkpoint_t, rel_tol=0.0,
        abs_tol=tolerance)]
    if len(matches) != 1:
        raise RuntimeError("checkpoint does not map to exactly one scalar row")
    if not math.isclose(float(matches[0]["H"]), checkpoint_h,
                        rel_tol=0.0,
                        abs_tol=8.0 * math.ulp(checkpoint_h)):
        raise RuntimeError("checkpoint hazard differs from scalar history")
    if not math.isclose(float(matches[0]["H_threshold"]), checkpoint_hstar,
                        rel_tol=0.0,
                        abs_tol=8.0 * math.ulp(checkpoint_hstar)):
        raise RuntimeError("checkpoint threshold differs from scalar history")

    prefix_rows = helpers.write_prefix_before_checkpoint(
        SOURCE / "avalanche_renewal_history.csv", PREFIX, checkpoint_t)
    prior_subevents = helpers.typed_rows(
        PREVIOUS_EVENT_SOURCE / "one_b_subevents.csv")
    if len(prior_subevents) != 11:
        raise RuntimeError(
            f"expected 11 completed 1b events, found {len(prior_subevents)}")
    prior_avalanches = (
        helpers.typed_rows(AVALANCHE1_SOURCE / "avalanche_summary.csv")
        + helpers.typed_rows(PREVIOUS_EVENT_SOURCE / "avalanche_summary.csv"))
    if [int(row["avalanche_id"]) for row in prior_avalanches] != [1, 2]:
        raise RuntimeError("expected complete avalanche summaries [1, 2]")

    preserved_descendant_thresholds = [
        float(row["descendant_Hstar"]) for row in prior_subevents]
    preserved_root_thresholds = [
        float(value) for value in seed_record["root_thresholds"]]
    if len(preserved_root_thresholds) != 3:
        raise RuntimeError("root RNG prefix must include avalanche 3")
    if not math.isclose(preserved_root_thresholds[-1], checkpoint_hstar,
                        rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("avalanche-3 root threshold is not preserved")

    _, setup = campaign.padded_case_builder()
    current_particle_volume = integral(state[1], setup)
    vp_cycle0 = current_particle_volume / float(checkpoint["Vp_over_Vp_cycle"])
    previous_contours = list(manifest.get("continuation_contour_sources", []))
    previous_movies = list(manifest.get("continuation_movie_sources", []))
    continuation = dict(
        parent_run=str(SOURCE.resolve()),
        parent_checkpoint=str(CHECKPOINT.resolve()),
        t_model=checkpoint_t,
        root_hazard=checkpoint_h,
        root_threshold=checkpoint_hstar,
        preserved_root_thresholds=preserved_root_thresholds,
        preserved_descendant_thresholds=preserved_descendant_thresholds,
        total_completed=11,
        vp_cycle0_m3=vp_cycle0,
        solid_volume_initial_m3=float(manifest["solid_volume_initial_m3"]),
        completed_avalanches_before=2,
        start_avalanche_id=3,
        root_already_nucleated=False,
        active_avalanche=False,
        active_event=False,
        wait_start_model=float(prior_avalanches[-1]["end_time_model"]),
        prior_subevents=prior_subevents,
        prior_avalanches=prior_avalanches,
        history_sources=[str(PREFIX.resolve())],
        contour_sources=(previous_contours + [str((
            SOURCE / "avalanche_renewal_sparse_contours.npz").resolve())]),
        movie_sources=(previous_movies + [str((
            SOURCE / "avalanche_renewal_movie_geometry.h5").resolve())]),
        movie_source_valid_through_t_model=checkpoint_t,
        repair_partition_on_load=False,
        RNG_state_preserved=True,
        morphology_restarted=False,
        continuation_reason=(
            "ENOSPC during scalar-history flush after an atomic avalanche-3 "
            "passive checkpoint; output storage recovered only"))

    preflight = parent.cheap_transport_preflight()
    recovery.OUT = OUT
    recovery.SEGMENT_LABEL = SEGMENT_LABEL
    recovery._configure_production(
        preflight, seed=int(seed_record["seed"]), continuation=continuation,
        provenance=parent.source_provenance())
    production.OUT = OUT
    production.EVENT_QUALIFICATION = source_driver.QUALIFICATION
    production.quasistatic_event_call = replacement.current_state_event_call
    production.AVALANCHES_REQUESTED = 8
    production.CONTINUATION = continuation

    production.DESCENDANT_DELTA_G0_EV = INITIAL_DESCENDANT_LOWERING_EV
    production.DESCENDANT_FACILITATION_DECAY_ALPHA = DECAY_ALPHA
    production.DESCENDANT_ESCALATION_AFTER_AVALANCHES = None
    production.DESCENDANT_ESCALATION_IF_ALL_S_BELOW = None
    production.DESCENDANT_ESCALATION_NEW_DELTA_G_EV = None

    renewal.PASSIVE_MACROSTEP_ENABLED = True
    renewal.PASSIVE_MAX_NUMERICAL_DT_MODEL = PASSIVE_MAX_DT_MODEL
    renewal.PASSIVE_MIN_NUMERICAL_DT_MODEL = setup["dt"] / 16.0
    renewal.PASSIVE_HAZARD_QUADRATURE_DT_MODEL = HAZARD_QUADRATURE_DT_MODEL
    renewal.PASSIVE_CROSSING_TOLERANCE_MODEL = CROSSING_TOLERANCE_MODEL
    renewal.ROOT_ANALYSIS_DT = PASSIVE_MACRO_DT_MODEL
    renewal.ROOT_MOVIE_DT = PASSIVE_MACRO_DT_MODEL

    production.EXTRA_RUN_METADATA.update(dict(
        campaign="current-state transfer alpha-0.70 ENOSPC continuation",
        campaign_timestamp_label=SEGMENT_LABEL,
        parent_run=str(SOURCE.resolve()),
        parent_checkpoint=str(CHECKPOINT.resolve()),
        restart_kind="exact atomic avalanche-3 passive checkpoint",
        restart_reason="scalar CSV rewrite failed with errno 28",
        checkpoint_t_model=checkpoint_t,
        checkpoint_root_hazard=checkpoint_h,
        checkpoint_root_threshold=checkpoint_hstar,
        checkpoint_H_over_Hstar=checkpoint_h / checkpoint_hstar,
        history_prefix_rows_before_checkpoint=prefix_rows,
        parent_files_modified=False,
        stochastic_seed_preserved=True,
        root_threshold_preserved=True,
        descendant_rng_stream_preserved=True,
        morphology_restarted=False,
        physical_parameters_changed=False,
        numerical_parameters_changed=False,
        output_storage_recovered_only=True,
        initial_descendant_lowering_changed=False,
        initial_descendant_lowering_eV=INITIAL_DESCENDANT_LOWERING_EV,
        descendant_decay_alpha=DECAY_ALPHA,
        target_event_count_forced=False,
        parameter_sweep_performed=False,
        atomic_checkpoint_each_passive_macrostep=True,
        curvature_monitor_preserved=True,
        continuous_scalar_history=True,
        history_checkpoint_row_written_exactly_once=True))
    production.main()


if __name__ == "__main__":
    main()
