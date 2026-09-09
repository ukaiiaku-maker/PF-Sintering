#!/usr/bin/env python3
"""Resume avalanche 1 after its committed root 1b bookkeeping stop."""
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
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25 as parent_driver  # noqa: E402
from pr_full_deterministic_cycle import load_event_save  # noqa: E402


PARENT0 = ROOT / "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904"
PARENT1 = ROOT / "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_resume1"
EVENT_SAVE = PARENT1 / "checkpoints/avalanche1_event1_q1.00b.npz"
RECOVERY_CHECKPOINT = ROOT / (
    "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_"
    "resume1_event1_complete_recovery_input.npz")
RECOVERED_EVENT_RECORD = ROOT / (
    "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_"
    "resume1_event1_complete_record.json")
OUT = ROOT / "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904_resume2"
SEGMENT_LABEL = "20260904_resume2"


def _typed_row(row: dict[str, str]) -> dict:
    result = {}
    for key, value in row.items():
        if value == "":
            result[key] = math.nan
            continue
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def _history_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return [_typed_row(row) for row in csv.DictReader(handle)]


def _configure_production(preflight: dict, *, seed: int,
                          continuation: dict, provenance: dict) -> None:
    production.OUT = OUT
    production.D_GB_REFERENCE = parent_driver.QUALIFIED_D_GB
    production.D_GB_TARGET = parent_driver.D_GB
    production.REFERENCE_MEDIAN_EVENT_S = (
        parent_driver.QUALIFIED_EVENT_DURATION_S)
    production.TARGET_EVENTS_PER_AVALANCHE = 15
    production.TARGET_AVALANCHE_RELAXATION_S = 0.42
    production.TARGET_MEDIAN_EVENT_S = preflight[
        "predicted_candidate_event_duration_s"]
    production.DESCENDANT_DELTA_G0_EV = (
        parent_driver.DELTA_G_DESC_INITIAL_EV)
    production.ROOT_THRESHOLD_MULTIPLIER = (
        parent_driver.ROOT_THRESHOLD_MULTIPLIER)
    production.DESCENDANT_ESCALATION_AFTER_AVALANCHES = 2
    production.DESCENDANT_ESCALATION_IF_ALL_S_BELOW = 10
    production.DESCENDANT_ESCALATION_NEW_DELTA_G_EV = (
        parent_driver.DELTA_G_DESC_ESCALATED_EV)
    production.SOLID_VOLUME_RELATIVE_TOLERANCE = (
        parent_driver.VOLUME_ERROR_CEILING)
    production.AVALANCHES_REQUESTED = parent_driver.AVALANCHES_REQUESTED
    production.MAX_DESCENDANTS_SAFETY = 25
    production.REUSED_SEED = seed
    production.CONTINUATION = continuation
    production.EXTRA_RUN_METADATA = dict(
        campaign="large-avalanche q-continuation D_GB1.5x threshold1.25",
        campaign_timestamp_label=SEGMENT_LABEL,
        parent_runs=[str(PARENT0.resolve()), str(PARENT1.resolve())],
        parent_checkpoint=str(EVENT_SAVE.resolve()),
        restart_kind="exact committed-root-event continuation",
        restart_reason=(
            "post-event reporting NameError; physical q=b event completed "
            "and is not replayed"),
        source_fixes=[
            ("recover event_integrator_decision from returned event restart "
             "record before ONE_B reporting"),
            ("map a zero trial qdot to infinite reciprocal-rate change so the "
             "existing adaptive rollback and step-halving path executes"),
            ("expose every accepted prescribed-q state through an atomic live "
             "progress record and periodic exact event restart checkpoint"),
            ("reduce only the adaptive continuation minimum increment after a "
             "diagnostic trial proved the unchanged acceptance gates pass at "
             "1.953125e-5 b"),
        ],
        event_minimum_increment_fraction_b=(
            production.EVENT_MINIMUM_INCREMENT_FRACTION_B),
        stochastic_identity_preserved=True,
        morphology_restarted=False,
        root_event_replayed=False,
        root_threshold_redrawn=False,
        descendant_threshold_redrawn=False,
        continuous_scalar_history=True,
        history_boundary_duplicates_removed=True,
        prior_contour_movie_segments_preserved=True,
        predeclared_descendant_escalation_rule_preserved=True,
        production_source_snapshot_sha256=(
            provenance["production_source_snapshot_sha256"]),
        parent_production_source_snapshot_sha256=json.loads(
            (PARENT1 / "launch_manifest.json").read_text())[
                "production_source_snapshot_sha256"],
        current_git_HEAD=provenance["git_HEAD"])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not EVENT_SAVE.exists():
        raise FileNotFoundError(EVENT_SAVE)

    manifest0 = json.loads((PARENT0 / "launch_manifest.json").read_text())
    manifest1 = json.loads((PARENT1 / "launch_manifest.json").read_text())
    seed_record = json.loads((PARENT1 / "rng_seed_record.json").read_text())
    seed = int(seed_record["seed"])
    root_threshold = float(manifest1["first_root_threshold_effective"])
    if seed != parent_driver.PRESERVED_PRELAUNCH_OS_SEED:
        raise RuntimeError("parent stochastic seed changed")

    rows1 = _history_rows(PARENT1 / "avalanche_renewal_history.csv")
    root_rows = [row for row in rows1 if row.get("frame_type") == "root_nucleation"]
    event_end_rows = [
        row for row in rows1
        if row.get("frame_type") == "active_1b_transit"
        and math.isclose(float(row["q_event_over_b"]), 1.0)]
    if len(root_rows) != 1 or len(event_end_rows) != 1:
        raise RuntimeError("cannot uniquely reconstruct the committed root event")
    root_row = root_rows[0]
    post_row = event_end_rows[0]

    event_state, event_restart = load_event_save(EVENT_SAVE)
    event_end_model = (
        float(root_row["t_model"]) + float(event_restart["event_time_model"]))
    if not math.isclose(
            event_end_model, float(post_row["t_model"]),
            rel_tol=0.0, abs_tol=64.0 * math.ulp(event_end_model)):
        raise RuntimeError("event checkpoint time differs from scalar history")
    if not math.isclose(
            float(event_restart["cumulative_q_m"]), renewal.B_EVENT_M,
            rel_tol=0.0, abs_tol=8.0 * math.ulp(renewal.B_EVENT_M)):
        raise RuntimeError("recovery event did not reach exactly q=b")

    geom, setup = campaign.padded_case_builder()
    solid_volume = float(manifest0["solid_volume_initial_m3"])
    post_particle_volume = renewal.integral(event_state[1], setup)
    renewal.save_state(
        RECOVERY_CHECKPOINT, event_state, t_model=event_end_model,
        H=float(root_row["H"]), Hstar=root_threshold,
        total_completed=1, recovered_from=str(EVENT_SAVE.resolve()))

    exported, _root_slice = renewal.authoritative.configure_barrier()
    streams = np.random.SeedSequence(seed).spawn(2)
    root_rng = np.random.default_rng(streams[0])
    reconstructed_root = (
        parent_driver.ROOT_THRESHOLD_MULTIPLIER
        * float(root_rng.exponential()))
    if not math.isclose(
            reconstructed_root, root_threshold, rel_tol=0.0,
            abs_tol=8.0 * math.ulp(root_threshold)):
        raise RuntimeError("root RNG stream does not reproduce")
    descendant_rng = np.random.default_rng(streams[1])
    controller = renewal.AvalancheController(
        barrier=renewal.DescendantBarrier(
            renewal.base.BARRIER, parent_driver.DELTA_G_DESC_INITIAL_EV),
        temperature_K=renewal.TEMPERATURE_K,
        attempt_frequency_per_s=float(exported["constants"]["nu0_sinv"]),
        b_m=renewal.B_EVENT_M,
        correlation_time_s=renewal.TAU_CORR_S,
        rng=descendant_rng,
        facilitation_decay_alpha=parent_driver.FACILITATION_DECAY)
    controller.start(
        avalanche_id=1, root_cycle=1,
        start_time_s=float(root_row["t_s"]))
    controller.begin_transit()
    controller.complete_transit(event_end_model * renewal.SECONDS_PER_MODEL_TIME)
    descendant_threshold = float(controller.state.descendant_threshold)

    decision = dict(event_restart["event_integrator_decision"])
    pre_local = float(root_row["sigma_local_Pa"])
    post_local = float(post_row["sigma_local_Pa"])
    pre_integral = float(root_row["sigma_integral_Pa"])
    post_integral = float(post_row["sigma_integral_Pa"])
    prior_event = dict(
        **decision,
        avalanche_id=1,
        event_number=1,
        event_type="root",
        start_time_model=float(root_row["t_model"]),
        end_time_model=event_end_model,
        start_time_s=float(root_row["t_s"]),
        end_time_s=event_end_model * renewal.SECONDS_PER_MODEL_TIME,
        duration_s=float(event_restart["event_time_model"])
        * renewal.SECONDS_PER_MODEL_TIME,
        sigma_start_local_Pa=pre_local,
        sigma_end_local_Pa=post_local,
        delta_sigma_local_Pa=pre_local-post_local,
        delta_sigma_integral_Pa=pre_integral-post_integral,
        delta_G_J=(float(post_row["G_phasefield_J"])
                   - float(root_row["G_phasefield_J"])),
        delta_z_TJ_m=float(post_row["z_TJ_m"])-float(root_row["z_TJ_m"]),
        delta_r_TJ_m=float(post_row["r_n_m"])-float(root_row["r_n_m"]),
        delta_volume_relative=(
            float(post_row["V_solid_m3"])/float(root_row["V_solid_m3"])-1.0),
        delta_mu_GB_minus_TJ_local_Pa=(
            float(post_row["delta_mu_GB_minus_TJ_local_Pa"])
            - float(root_row["delta_mu_GB_minus_TJ_local_Pa"])),
        mu_GB_minus_mu_TJ_before_Pa=float(
            root_row["delta_mu_GB_minus_TJ_local_Pa"]),
        mu_GB_minus_mu_TJ_after_Pa=float(
            post_row["delta_mu_GB_minus_TJ_local_Pa"]),
        G_PF_native_before_J=float(root_row["G_phasefield_J"]),
        G_PF_native_after_J=float(post_row["G_phasefield_J"]),
        G_gamma_before_J=9.297575734246754e-13,
        G_gamma_after_J=9.298963691113445e-13,
        r_neck_after_m=float(post_row["r_n_m"]),
        A1_after_m=float(post_row["A1_cos_m"]),
        descendant_Gstar_eV=controller.barrier_eV(post_local),
        h=controller.state.source_amplitude,
        facilitation_eV=(controller.state.source_amplitude
                         * controller.barrier.delta_G_step_eV),
        descendant_rate_per_s=controller.rate(
            post_local, float(post_row["r_n_m"])),
        descendant_H=controller.state.descendant_hazard,
        descendant_Hstar=descendant_threshold,
        remaining_corr_window_ms=renewal.TAU_CORR_S*1.0e3,
        descendants_committed_during_transit=0,
        pending_children_after=0,
        C4=math.nan,
        event_branch_time_integrator=(
            "direct adaptive prescribed-q fast-manifold continuation"),
        event_packets_total=int(event_restart["accepted_steps_total"]),
        event_max_increment_fraction_b=production.EVENT_MAX_INCREMENT_FRACTION_B,
        D_GB_m2_per_model_time=parent_driver.D_GB,
        G_PF_equilibrated_copy_before_J=9.310853572468677e-13,
        G_PF_equilibrated_copy_after_J=9.31213734433472e-13,
        delta_G_PF_equilibrated_copy_J=1.2837718660428083e-16,
        G_gamma_geometry_before_J=9.297575734246754e-13,
        G_gamma_geometry_after_J=9.298963691113445e-13,
        delta_G_gamma_geometry_J=1.3879568666907012e-16,
        copy_pre_ownership_iterations=400,
        copy_post_ownership_iterations=400,
        copy_pre_volume_relative_error=-7.771561172376096e-16,
        copy_post_volume_relative_error=-1.6653345369377348e-15,
        copy_equilibration_inputs_unchanged=1,
        copy_equilibration_advanced_model_time=0,
        recovery_source=str(EVENT_SAVE.resolve()),
        recovery_reason="post-event reporting NameError")
    RECOVERED_EVENT_RECORD.write_text(
        json.dumps(prior_event, indent=2) + "\n")

    history_sources = [
        str((PARENT0 / "avalanche_renewal_history.csv").resolve()),
        str((PARENT1 / "avalanche_renewal_history.csv").resolve()),
    ]
    contour_sources = [
        str((PARENT0 / "avalanche_renewal_sparse_contours.npz").resolve()),
        str((PARENT1 / "avalanche_renewal_sparse_contours.npz").resolve()),
    ]
    movie_sources = [
        str((PARENT0 / "avalanche_renewal_movie_geometry.h5").resolve()),
        str((PARENT1 / "avalanche_renewal_movie_geometry.h5").resolve()),
    ]
    continuation = dict(
        parent_run=str(PARENT1.resolve()),
        parent_checkpoint=str(RECOVERY_CHECKPOINT.resolve()),
        t_model=event_end_model,
        root_hazard=float(root_row["H"]),
        root_threshold=root_threshold,
        preserved_root_thresholds=[root_threshold],
        preserved_descendant_thresholds=[descendant_threshold],
        total_completed=1,
        vp_cycle0_m3=post_particle_volume,
        solid_volume_initial_m3=solid_volume,
        completed_avalanches_before=0,
        start_avalanche_id=1,
        root_already_nucleated=True,
        active_avalanche=True,
        root_row=root_row,
        controller_manifest=controller.manifest(),
        source_amplitude=1.0,
        avalanche_start_model=float(root_row["t_model"]),
        wait_start_model=0.0,
        prior_subevents=[prior_event],
        history_sources=history_sources,
        contour_sources=contour_sources,
        movie_sources=movie_sources,
        repair_partition_on_load=False,
        RNG_state_preserved=True,
        morphology_restarted=False,
        event_already_committed=True,
        stop_cause="post-event reporting NameError")

    preflight = parent_driver.cheap_transport_preflight()
    provenance = parent_driver.source_provenance()
    parent_hashes = manifest0["production_source_files_sha256"]
    current_hashes = provenance["production_source_files_sha256"]
    changed_sources = sorted(
        key for key in parent_hashes
        if current_hashes.get(key) != parent_hashes[key])
    if changed_sources != [
            "pf_sintering/production_q_event.py",
            "scripts/pr_avalanche_renewal_five.py",
            "scripts/pr_full_corrected_production_slowgb_large_avalanche.py"]:
        raise RuntimeError(
            "unexpected production source changes before recovery launch: "
            f"{changed_sources}")
    _configure_production(
        preflight, seed=seed, continuation=continuation,
        provenance=provenance)
    production.main()


if __name__ == "__main__":
    main()
