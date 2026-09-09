#!/usr/bin/env python3
"""Resume the accepted stochastic trajectory at avalanche-1 event-2 q=0.

The parent loading history and root/event-1 realization are immutable.  Only
the event propagator is replaced: event 2 begins from its exact saved child-
crossing fields and q is reset to zero transferred material.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_renewal_five as renewal  # noqa: E402
import pr_full_corrected_production_campaign as campaign  # noqa: E402
import pr_full_corrected_production_slowgb_large_avalanche as production  # noqa: E402
import pr_large_avalanche_current_state_transfer_production as replacement  # noqa: E402
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25 as parent  # noqa: E402
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_resume2 as recovery  # noqa: E402
from pf_sintering.rigid_rbm_deposition import _axisym_weighted_sum  # noqa: E402
from pr_full_deterministic_cycle import restart_arrays  # noqa: E402


PARENT0 = ROOT / "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904"
PARENT1 = ROOT / "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_resume1"
ONSET_SEGMENT = ROOT / "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_resume3"
ONSET_STATE = ONSET_SEGMENT / "checkpoints/avalanche1_event2_before_descendant.npz"
CONTROLLER_SOURCE = ROOT / (
    "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_"
    "resume6_autorestart47/checkpoints/current_event_latest.npz")
PRIOR_EVENT = ROOT / (
    "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_"
    "resume1_event1_complete_record.json")
QUALIFICATION = ROOT / "runs/pr_current_state_mass_transfer_qualification_v1/decision.json"
REJECTED_METHOD = PARENT0 / "QUASISTATIC_GEOMETRIC_Q_CONTINUATION_NOT_PRODUCTION_VIABLE.json"
RESTART_INPUT = ROOT / "runs/pr_current_state_transfer_event2_onset_restart_input.npz"
OUT = ROOT / "runs/pr_large_avalanche_current_state_transfer_event2_onset_v1"


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


def source_provenance() -> dict:
    sources = (
        Path(__file__),
        ROOT / "pf_sintering/current_state_mass_transfer.py",
        ROOT / "pf_sintering/production_mass_transfer_event.py",
        ROOT / "scripts/pr_large_avalanche_current_state_transfer_production.py",
        ROOT / "scripts/pr_avalanche_renewal_five.py",
    )
    combined = hashlib.sha256()
    entries = {}
    for path in sources:
        data = path.read_bytes()
        name = str(path.relative_to(ROOT))
        entries[name] = hashlib.sha256(data).hexdigest()
        combined.update(name.encode())
        combined.update(b"\0")
        combined.update(data)
    return dict(
        git_HEAD=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_snapshot_sha256=combined.hexdigest(),
        source_files_sha256=entries)


def exact_rows() -> tuple[dict, dict]:
    rows = typed_rows(ONSET_SEGMENT / "avalanche_renewal_history.csv")
    roots = [row for row in rows
             if row.get("frame_type") == "root_nucleation"
             and int(row.get("avalanche_id", -1)) == 1]
    onsets = [row for row in rows
              if row.get("frame_type") == "nucleated_transport_wait"
              and int(row.get("avalanche_id", -1)) == 1
              and int(row.get("event_number", -1)) == 2
              and math.isclose(float(row.get("q_event_over_b", math.nan)), 0.0,
                               abs_tol=1.0e-15)]
    if len(roots) != 1 or len(onsets) != 1:
        raise RuntimeError(
            f"expected one root and one event-2 onset, got {len(roots)}, "
            f"{len(onsets)}")
    return roots[0], onsets[0]


def build_restart_input(state: tuple[np.ndarray, ...], setup: dict,
                        onset_row: dict, decision: dict) -> None:
    restart = dict(
        base_fields=tuple(field.copy() for field in state),
        union_previous_fields=tuple(field.copy() for field in state),
        cumulative_q_m=0.0,
        cumulative_source_weighted=0.0,
        event_time_model=0.0,
        accepted_steps_total=0,
        mass_initial_weighted=_axisym_weighted_sum(state[0], setup["r_c"]),
        cumulative_grain_flux_volume_m3={1: 0.0, 2: 0.0},
        quasistatic_trial_rejections_total=0,
        restart_fast_precondition_calls_total=0,
        slow_clock_quadrature_refinements_total=0,
        active_minimum_step_over_b=0.0025,
        ordinary_fixed_q_calls_total=0,
        event_integrator_decision=dict(decision),
        event_origin_row=dict(onset_row),
        explicit_max_fourth_order_courant=math.nan,
        event_branch_time_integrator=replacement.EVENT_INTEGRATOR_NAME)
    payload = restart_arrays(state, restart)
    temporary = RESTART_INPUT.with_suffix(RESTART_INPUT.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, RESTART_INPUT)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    gate = json.loads(QUALIFICATION.read_text())
    if (gate.get("status") != "PASS_CURRENT_STATE_TRANSFER_TWO_EVENT_GATE"
            or not gate.get("production_launch_authorized")):
        raise RuntimeError("replacement event qualification is not accepted")
    if not REJECTED_METHOD.exists():
        raise RuntimeError("rejected geometric-q propagator lacks failure record")

    root_row, onset_row = exact_rows()
    with np.load(ONSET_STATE, allow_pickle=False) as saved:
        state = tuple(np.asarray(saved[key]).copy()
                      for key in ("f", "particle", "substrate"))
        onset_time = float(saved["t_model"])
        descendant_hazard = float(saved["descendant_hazard"])
        descendant_threshold = float(saved["descendant_threshold"])
    if not math.isclose(onset_time, float(onset_row["t_model"]),
                        rel_tol=0.0, abs_tol=8.0*math.ulp(onset_time)):
        raise RuntimeError("event-2 onset field time and scalar row differ")
    if not math.isclose(descendant_hazard, descendant_threshold,
                        rel_tol=0.0,
                        abs_tol=8.0*math.ulp(descendant_threshold)):
        raise RuntimeError("event-2 onset is not the committed child crossing")

    with np.load(CONTROLLER_SOURCE, allow_pickle=False) as saved:
        controller_manifest = json.loads(saved["controller_manifest_json"].item())
        root_threshold = float(saved["root_threshold"])
        solid_volume = float(saved["solid_volume_initial_m3"])
    controller_state = controller_manifest["state"]
    for key, expected in (("avalanche_id", 1), ("S_completed", 1)):
        if int(controller_state[key]) != expected:
            raise RuntimeError(f"controller {key} does not identify event 2")
    if not controller_state["window_triggered"]:
        raise RuntimeError("controller has not committed descendant event 2")
    if not math.isclose(float(controller_state["descendant_threshold"]),
                        descendant_threshold, rel_tol=0.0,
                        abs_tol=8.0*math.ulp(descendant_threshold)):
        raise RuntimeError("controller threshold differs from onset checkpoint")

    geom, setup = campaign.padded_case_builder()
    evaluator = production.corrected.evaluator_builder(setup, geom)
    transport = replace(
        production.make_transport(geom),
        D_gb_m2_per_model_time=parent.D_GB)
    current_record = evaluator(*state)
    onset_for_selection = dict(onset_row)
    onset_for_selection["contact_area_m2"] = float(
        current_record["contact_area_m2"])
    # Configure the selector exactly as the production wrapper will before
    # constructing the serialized q=0 event restart record.
    renewal.EVENT_B_PF_MODEL = production.corrected.EVENT_B_PF
    renewal.EVENT_SURFACE_ACTIVE_LENGTH_M = parent.EVENT_SURFACE_ACTIVE_LENGTH_M
    renewal.EVENT_INTEGRATOR_REQUESTED = (
        renewal.DEFAULT_PRODUCTION_EVENT_INTEGRATOR)
    decision = renewal.event_integrator_decision(
        onset_for_selection, setup, transport, event_restart=None)
    if decision["event_integrator_selected"] != "quasistatic":
        raise RuntimeError("slow-GB onset no longer selects fast-manifold event mode")
    build_restart_input(state, setup, onset_row, decision)

    seed_record = json.loads((PARENT1 / "rng_seed_record.json").read_text())
    seed = int(seed_record["seed"])
    if seed != parent.PRESERVED_PRELAUNCH_OS_SEED:
        raise RuntimeError("stochastic seed differs from accepted trajectory")
    preserved_root_thresholds = [float(value)
                                 for value in seed_record["root_thresholds"]]
    prior_event = json.loads(PRIOR_EVENT.read_text())
    prior_event["descendant_Hstar"] = descendant_threshold
    vp_cycle0 = renewal.integral(state[1], setup)

    parent_manifest = json.loads(
        (ONSET_SEGMENT / "launch_manifest.json").read_text())
    contour_sources = list(parent_manifest.get("continuation_contour_sources", []))
    movie_sources = list(parent_manifest.get("continuation_movie_sources", []))
    for collection, path in (
            (contour_sources,
             ONSET_SEGMENT / "avalanche_renewal_sparse_contours.npz"),
            (movie_sources,
             ONSET_SEGMENT / "avalanche_renewal_movie_geometry.h5")):
        if path.exists():
            collection.append(str(path.resolve()))

    continuation = dict(
        parent_run=str(ONSET_SEGMENT.resolve()),
        parent_checkpoint=str(RESTART_INPUT.resolve()),
        t_model=onset_time,
        root_hazard=float(root_row["H"]),
        root_threshold=root_threshold,
        preserved_root_thresholds=preserved_root_thresholds,
        preserved_descendant_thresholds=[descendant_threshold],
        total_completed=1,
        vp_cycle0_m3=vp_cycle0,
        solid_volume_initial_m3=solid_volume,
        completed_avalanches_before=0,
        start_avalanche_id=1,
        root_already_nucleated=True,
        active_avalanche=True,
        active_event=True,
        active_event_number=2,
        active_event_start_model=onset_time,
        active_event_pre_row=onset_row,
        root_row=root_row,
        controller_manifest=controller_manifest,
        source_amplitude=float(controller_state["source_amplitude"]),
        avalanche_start_model=float(root_row["t_model"]),
        wait_start_model=0.0,
        prior_subevents=[prior_event],
        history_sources=[str((ONSET_SEGMENT /
                              "avalanche_renewal_history.csv").resolve())],
        contour_sources=list(dict.fromkeys(contour_sources)),
        movie_sources=list(dict.fromkeys(movie_sources)),
        repair_partition_on_load=False,
        RNG_state_preserved=True,
        morphology_restarted=False,
        event_1_replayed=False,
        event_2_prefix_replayed=False,
        event_2_q_reset_to_transferred_material_zero=True)

    preflight = parent.cheap_transport_preflight()
    recovery.OUT = OUT
    recovery._configure_production(
        preflight, seed=seed, continuation=continuation,
        provenance=parent.source_provenance())
    production.OUT = OUT
    production.EVENT_QUALIFICATION = QUALIFICATION
    production.quasistatic_event_call = replacement.current_state_event_call
    production.AVALANCHES_REQUESTED = 8
    production.CONTINUATION = continuation
    production.EXTRA_RUN_METADATA.update(dict(
        campaign="current-state transfer from exact event-2 onset",
        restart_kind="exact accepted avalanche-1 event-2 q=0 child crossing",
        parent_event2_onset_state=str(ONSET_STATE.resolve()),
        synthetic_restart_input=str(RESTART_INPUT.resolve()),
        event_propagator=replacement.EVENT_INTEGRATOR_NAME,
        q_definition="accumulated transferred material only",
        loading_replayed=False,
        event_1_replayed=False,
        event_2_prefix_replayed=False,
        start_q_over_b=0.0,
        start_t_model=onset_time,
        stochastic_seed_preserved=True,
        descendant_threshold_preserved=True,
        physical_parameters_changed=False,
        current_state_source_provenance=source_provenance()))

    if os.environ.get("PR_VALIDATE_CONTINUATION_ONLY") == "1":
        streams = np.random.SeedSequence(seed).spawn(2)
        root_rng = np.random.default_rng(streams[0])
        descendant_rng = np.random.default_rng(streams[1])
        rebuilt_root = parent.ROOT_THRESHOLD_MULTIPLIER * float(
            root_rng.exponential())
        rebuilt_descendant = float(descendant_rng.exponential())
        if not math.isclose(rebuilt_root, root_threshold, rel_tol=0.0,
                            abs_tol=8.0*math.ulp(root_threshold)):
            raise RuntimeError("root threshold reconstruction failed")
        if not math.isclose(rebuilt_descendant, descendant_threshold,
                            rel_tol=0.0,
                            abs_tol=8.0*math.ulp(descendant_threshold)):
            raise RuntimeError("descendant threshold reconstruction failed")
        print(json.dumps(dict(
            status="PASS_EVENT2_ONSET_CONTINUATION",
            avalanche_id=1, event_number=2, q_over_b=0.0,
            t_model=onset_time, sigma_local_MPa=(
                float(onset_row["sigma_local_Pa"])*1.0e-6),
            sigma_integral_MPa=(
                float(onset_row["sigma_integral_Pa"])*1.0e-6),
            descendant_threshold=descendant_threshold,
            event_integrator_selected=decision["event_integrator_selected"],
            event_propagator=replacement.EVENT_INTEGRATOR_NAME), sort_keys=True))
        return
    production.main()


if __name__ == "__main__":
    main()
