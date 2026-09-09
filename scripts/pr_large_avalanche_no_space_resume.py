#!/usr/bin/env python3
"""Resume the accepted large-avalanche trajectory after an ENOSPC stop.

Only restart/output bookkeeping changes.  The PF state, accumulated root
hazard, RNG streams, thresholds, transport coefficients, event propagator,
and avalanche physics are restored exactly from the last atomic checkpoint.
"""
from __future__ import annotations

import csv
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
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25 as parent  # noqa: E402
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_resume2 as recovery  # noqa: E402
from pr_coarsening_driven_fourier_loading import integral  # noqa: E402


SOURCE = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_resume3")
CHECKPOINT = SOURCE / "checkpoints/avalanche2_waiting_latest.npz"
OUT = ROOT / (
    "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1_resume4")
PREFIX = SOURCE / "recovery_history_prefix_before_enospc_checkpoint.csv"
SEGMENT_LABEL = "current_state_transfer_resume4_after_enospc"


def typed_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            row = {}
            for key, value in source.items():
                if value == "":
                    row[key] = math.nan
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def write_prefix_before_checkpoint(source: Path, target: Path,
                                   checkpoint_t: float) -> int:
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    tolerance = 64.0 * math.ulp(max(abs(checkpoint_t), 1.0))
    committed = [row for row in rows
                 if float(row["t_model"]) <= checkpoint_t + tolerance]
    if not committed or not math.isclose(
            float(committed[-1]["t_model"]), checkpoint_t,
            rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError("history does not contain the atomic checkpoint")
    # The continuation writes the checkpoint state as its initial row.  Keep
    # all strictly earlier rows so the combined scalar history contains that
    # state exactly once.
    committed.pop()
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(committed)
    return len(committed)


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
    prior_avalanches = typed_rows(SOURCE / "avalanche_summary.csv")
    if len(prior_avalanches) != 1 or int(
            prior_avalanches[0]["avalanche_id"]) != 1:
        raise RuntimeError("expected exactly one completed prior avalanche")
    prior_subevents = typed_rows(
        source_driver.OUT / "one_b_subevents.csv")
    if len(prior_subevents) != 5:
        raise RuntimeError("expected five committed avalanche-1 events")

    checkpoint_t = float(checkpoint["t_model"])
    checkpoint_h = float(checkpoint["H"])
    checkpoint_hstar = float(checkpoint["Hstar"])
    parent_rows = typed_rows(SOURCE / "avalanche_renewal_history.csv")
    matching = [row for row in parent_rows if math.isclose(
        float(row["t_model"]), checkpoint_t, rel_tol=0.0,
        abs_tol=64.0 * math.ulp(max(abs(checkpoint_t), 1.0)))]
    if len(matching) != 1:
        raise RuntimeError("checkpoint does not map to one scalar row")
    if not math.isclose(float(matching[0]["H"]), checkpoint_h,
                        rel_tol=0.0, abs_tol=8.0*math.ulp(checkpoint_h)):
        raise RuntimeError("checkpoint hazard differs from scalar history")
    if not math.isclose(float(matching[0]["H_threshold"]), checkpoint_hstar,
                        rel_tol=0.0, abs_tol=8.0*math.ulp(checkpoint_hstar)):
        raise RuntimeError("checkpoint threshold differs from scalar history")

    prefix_rows = write_prefix_before_checkpoint(
        SOURCE / "avalanche_renewal_history.csv", PREFIX, checkpoint_t)
    geom, setup = campaign.padded_case_builder()
    current_particle_volume = integral(state[1], setup)
    vp_cycle0 = current_particle_volume / float(
        checkpoint["Vp_over_Vp_cycle"])

    previous_contours = list(manifest.get("continuation_contour_sources", []))
    previous_movies = list(manifest.get("continuation_movie_sources", []))
    continuation = dict(
        parent_run=str(SOURCE.resolve()),
        parent_checkpoint=str(CHECKPOINT.resolve()),
        t_model=checkpoint_t,
        root_hazard=checkpoint_h,
        root_threshold=checkpoint_hstar,
        preserved_root_thresholds=[float(value) for value in
                                   seed_record["root_thresholds"]],
        preserved_descendant_thresholds=[float(value) for value in
            seed_record["preserved_descendant_thresholds"]],
        total_completed=5,
        vp_cycle0_m3=vp_cycle0,
        solid_volume_initial_m3=float(manifest["solid_volume_initial_m3"]),
        completed_avalanches_before=1,
        start_avalanche_id=2,
        root_already_nucleated=False,
        active_avalanche=False,
        active_event=False,
        wait_start_model=float(prior_avalanches[0]["end_time_model"]),
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
        continuation_reason="ENOSPC during post-checkpoint movie append")

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
    production.EXTRA_RUN_METADATA.update(dict(
        campaign="current-state transfer ENOSPC atomic continuation",
        campaign_timestamp_label=SEGMENT_LABEL,
        parent_run=str(SOURCE.resolve()),
        parent_checkpoint=str(CHECKPOINT.resolve()),
        restart_kind="exact atomic avalanche-2 passive checkpoint",
        restart_reason="movie HDF5 append failed with errno 28",
        checkpoint_t_model=checkpoint_t,
        checkpoint_root_hazard=checkpoint_h,
        checkpoint_root_threshold=checkpoint_hstar,
        checkpoint_H_over_Hstar=checkpoint_h/checkpoint_hstar,
        history_prefix_rows_before_checkpoint=prefix_rows,
        parent_movie_valid_through_t_model=checkpoint_t,
        parent_movie_contains_uncommitted_tail=True,
        parent_files_modified=False,
        stochastic_seed_preserved=True,
        root_threshold_preserved=True,
        descendant_rng_stream_preserved=True,
        morphology_restarted=False,
        physical_parameters_changed=False,
        numerical_parameters_changed=False,
        output_storage_recovered_only=True,
        prior_avalanche_records_carried=True,
        continuous_scalar_history=True,
        history_checkpoint_row_written_exactly_once=True))
    production.main()


if __name__ == "__main__":
    main()
