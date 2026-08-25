#!/usr/bin/env python3
"""Exact event-5 continuation with production-consistent volume referencing."""
from __future__ import annotations

import csv
from dataclasses import fields
import json
import math
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
from numba import set_num_threads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pr_authoritative_stochastic_ten_event as authoritative  # noqa: E402
import pr_coarsening_stochastic_two_event as base  # noqa: E402
import pr_avalanche_renewal_five as renewal  # noqa: E402
from pf_sintering.pr_avalanche import (  # noqa: E402
    AvalancheController, AvalancheState, DescendantBarrier,
)
from pf_sintering.pr_movie_geometry import (  # noqa: E402
    FRAME_TYPES, MovieGeometryArchive,
)
from pr_coarsening_driven_fourier_loading import integral  # noqa: E402
from pr_experimental_long_sinkoff import build_case  # noqa: E402
from pr_full_deterministic_cycle import make_transport  # noqa: E402
from pr_tj_node_coupling_gate import make_evaluator  # noqa: E402


SOURCE = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_five_deltaG0p300_taucorr9ms")
OUT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_three_deltaG0p300_taucorr9ms_vpcycle_reset")
SEED = 198902775941295178708233632700209944609
TARGET_AVALANCHES = 3
RESUME_EVENT = 5


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        state = tuple(np.asarray(data[name]).copy()
                      for name in ("f", "particle", "substrate"))
        metadata = {
            name: data[name].item() if data[name].shape == () else data[name].copy()
            for name in data.files if name not in ("f", "particle", "substrate")
        }
    return state, metadata


def restore_controller(manifest: dict, descendant_rng, barrier, exported):
    recorded = np.asarray(manifest["state"]["thresholds_drawn"], dtype=float)
    replayed = np.asarray(
        descendant_rng.exponential(size=recorded.size), dtype=float)
    if not np.array_equal(recorded, replayed):
        raise RuntimeError("descendant RNG replay does not match saved thresholds")
    controller = AvalancheController(
        barrier=barrier, temperature_K=renewal.TEMPERATURE_K,
        attempt_frequency_per_s=float(exported["constants"]["nu0_sinv"]),
        b_m=renewal.B_EVENT_M, correlation_time_s=renewal.TAU_CORR_S,
        rng=descendant_rng)
    allowed = {item.name for item in fields(AvalancheState)}
    state_values = {key: value for key, value in manifest["state"].items()
                    if key in allowed}
    controller.state = AvalancheState(**state_values)
    controller.crossings = list(manifest["crossings"])
    return controller


def truncate_and_correct_movie(path: Path, cutoff_model: float) -> int:
    """Keep through event 5 and re-reference each completed-event interval."""
    with h5py.File(path, "r+") as archive:
        times = np.asarray(archive["time/t_model"], dtype=float)
        keep = int(np.searchsorted(times, cutoff_model + 1e-12, side="right"))
        if keep <= 0 or abs(times[keep - 1] - cutoff_model) > 1e-9:
            raise RuntimeError("event-5 time is absent from source movie")
        absolute_ratio = np.asarray(
            archive["state/Vp_over_Vp_cycle"][:keep], dtype=float)
        frame_type = np.asarray(archive["state/frame_type"][:keep], dtype=int)
        reference = 1.0
        corrected = np.empty_like(absolute_ratio)
        for index, value in enumerate(absolute_ratio):
            if frame_type[index] == FRAME_TYPES["child_completion"]:
                reference = value
                corrected[index] = 1.0
            else:
                corrected[index] = value / reference
        for group_name in ("geometry", "time", "state"):
            for dataset in archive[group_name].values():
                dataset.resize((keep,) + dataset.shape[1:])
        archive["state/Vp_over_Vp_cycle"][:] = corrected
        metadata = json.loads(archive.attrs["metadata_json"])
        metadata.update(
            continuation="exact event-5 q=b checkpoint",
            volume_reference=(
                "reset to particle volume at every completed one-b event"),
            source_archive=str(SOURCE))
        archive.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        archive.flush()
    return keep


def load_scalar_history(out: Path, cutoff_model: float) -> renewal.ScalarOutput:
    scalar = renewal.ScalarOutput(out)
    source_rows = read_rows(out / "avalanche_renewal_history.csv")
    retained = [row for row in source_rows
                if float(row["t_model"]) <= cutoff_model + 1e-12]
    reference = 1.0
    for row in retained:
        absolute = float(row["Vp_over_Vp_cycle"])
        if (float(row.get("q_event_over_b", 0.0)) >= 1.0 - 1e-12
                and int(float(row.get("event_number", 0))) > 0):
            reference = absolute
            row["Vp_over_Vp_cycle"] = 1.0
        else:
            row["Vp_over_Vp_cycle"] = absolute / reference
    scalar.rows = retained
    with np.load(out / "avalanche_renewal_sparse_contours.npz") as contours:
        for sample_id in range(len(retained)):
            scalar.contours.append(dict(
                sample_id=sample_id,
                z_negative=np.asarray(
                    contours[f"sample_{sample_id:05d}_z_negative"]).copy(),
                r_negative=np.asarray(
                    contours[f"sample_{sample_id:05d}_r_negative"]).copy(),
                z_positive=np.asarray(
                    contours[f"sample_{sample_id:05d}_z_positive"]).copy(),
                r_positive=np.asarray(
                    contours[f"sample_{sample_id:05d}_r_positive"]).copy()))
    scalar.flush()
    return scalar


def prepare_output(cutoff_model: float) -> tuple[renewal.ScalarOutput, int]:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    shutil.copytree(SOURCE, OUT)
    for name in (
            "avalanche_renewal_result.json", "movie_geometry_validation.json",
            "avalanche_renewal_five_aligned.png",
            "avalanche_renewal_five_aligned.pdf",
            "avalanche_renewal_morphology.gif"):
        path = OUT / name
        if path.exists():
            path.rename(OUT / f"pre_bookkeeping_{name}")
    keep = truncate_and_correct_movie(
        OUT / "avalanche_renewal_movie_geometry.h5", cutoff_model)
    scalar = load_scalar_history(OUT, cutoff_model)
    trace = [row for row in read_rows(OUT / "descendant_hazard_trace.csv")
             if float(row["t_s"]) <= (
                 cutoff_model * renewal.SECONDS_PER_MODEL_TIME + 1e-12)]
    renewal.write_csv(OUT / "descendant_hazard_trace.csv", trace)
    provenance = dict(
        classification="exact checkpoint continuation",
        source_directory=str(SOURCE), source_seed=SEED,
        source_checkpoint=(
            "checkpoints/avalanche1_event5_complete.npz"),
        resume_event=RESUME_EVENT, retained_movie_frames=keep,
        physics_change="none",
        bookkeeping_change=(
            "Vp_cycle0 reset to current particle volume at every completed 1b event"),
        validity_cutoff=renewal.V_CUTOFF,
        target_complete_avalanches=TARGET_AVALANCHES)
    (OUT / "continuation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    return scalar, keep


def reset_volume_reference(
        state, *, t_model, avalanche_id, total_completed, root_row,
        root_threshold, setup, geom, evaluator):
    vp_cycle0 = integral(state[1], setup)
    row, branches = renewal.measure(
        state, t_model=t_model, cycle=avalanche_id, sink=1, q=1.0,
        qcum=total_completed, hazard=root_row["H"],
        threshold=root_threshold, setup=setup, geom=geom,
        evaluator=evaluator, vp_cycle0=vp_cycle0)
    if abs(row["Vp_over_Vp_cycle"] - 1.0) > 5e-12:
        raise RuntimeError("post-event volume-reference reset failed")
    return vp_cycle0, row, branches


def main() -> None:
    set_num_threads(8)
    assert renewal.PRODUCTION_C4 == 0.05
    assert renewal.DELTA_G_STEP_EV == 0.300000
    assert renewal.TAU_CORR_S == 9.0e-3
    exported, root_slice = authoritative.configure_barrier()
    assert base.BARRIER.G0_eV == root_slice["G0_eV"]
    renewal.OUT = OUT
    base.OUT = OUT
    base.MIN_SOLVABLE_VOLUME_RATIO = -math.inf

    checkpoint = SOURCE / "checkpoints/avalanche1_event5_complete.npz"
    state, checkpoint_meta = load_checkpoint(checkpoint)
    t_model = float(checkpoint_meta["t_model"])
    total_completed = int(checkpoint_meta["total_completed"])
    if total_completed != RESUME_EVENT:
        raise RuntimeError("resume checkpoint is not event 5 complete")
    saved_controller = json.loads(
        str(checkpoint_meta["controller_manifest_json"]))

    streams = np.random.SeedSequence(SEED).spawn(2)
    root_rng = np.random.default_rng(streams[0])
    descendant_rng = np.random.default_rng(streams[1])
    root_threshold = float(root_rng.exponential())
    seed_record = json.loads((SOURCE / "rng_seed_record.json").read_text())
    if root_threshold != float(seed_record["root_thresholds"][0]):
        raise RuntimeError("root RNG replay does not match saved threshold")
    barrier = DescendantBarrier(base.BARRIER, renewal.DELTA_G_STEP_EV)
    controller = restore_controller(
        saved_controller, descendant_rng, barrier, exported)
    if (not controller.state.avalanche_active
            or controller.state.S_completed != RESUME_EVENT):
        raise RuntimeError("controller checkpoint is not active after event 5")

    scalar, retained_frames = prepare_output(t_model)
    geom, setup = build_case()
    evaluator = make_evaluator(setup, geom)
    transport = make_transport(geom)
    if transport.b_m != renewal.B_EVENT_M:
        raise RuntimeError("event displacement changed")
    if state[0].shape != geom["f"].shape:
        raise RuntimeError("checkpoint grid does not match qualified geometry")

    subevents = read_rows(OUT / "one_b_subevents.csv")
    descendant_trace = read_rows(OUT / "descendant_hazard_trace.csv")
    avalanches: list[dict] = []
    outcome = "RUNNING"
    blocker = None
    started = time.monotonic()

    with h5py.File(OUT / "avalanche_renewal_movie_geometry.h5", "r") as source:
        root_indices = np.flatnonzero(
            np.asarray(source["state/frame_type"]) == FRAME_TYPES["root_nucleation"])
        if root_indices.size != 1:
            raise RuntimeError("source archive must contain exactly one root frame")
        index = int(root_indices[0])
        root_row = dict(
            H=root_threshold, sigma_local_Pa=float(
                source["state/sigma_local_Pa"][index]),
            sigma_integral_Pa=float(source["state/sigma_integral_Pa"][index]),
            r_n_m=float(source["state/r_neck_m"][index]))
        avalanche_start_model = float(source["time/t_model"][index])
    root_local = root_row["sigma_local_Pa"]
    root_integral = root_row["sigma_integral_Pa"]
    wait_start_model = 0.0
    vp_cycle0, current_row, current_branches = reset_volume_reference(
        state, t_model=t_model, avalanche_id=1,
        total_completed=total_completed, root_row=root_row,
        root_threshold=root_threshold, setup=setup, geom=geom,
        evaluator=evaluator)
    event_number = RESUME_EVENT
    needs_window = True

    movie_path = OUT / "avalanche_renewal_movie_geometry.h5"
    with MovieGeometryArchive(
            movie_path, nbranch=renewal.N_BRANCH,
            seconds_per_model_time=renewal.SECONDS_PER_MODEL_TIME,
            mode="a") as movie:
        try:
            for avalanche_id in range(1, TARGET_AVALANCHES + 1):
                if avalanche_id > 1:
                    root_threshold = float(root_rng.exponential())
                    seed_record["root_thresholds"].append(root_threshold)
                    (OUT / "rng_seed_record.json").write_text(
                        json.dumps(seed_record, indent=2) + "\n")
                    wait_start_model = t_model
                    status, state, t_model, root_row, current_branches, vp_cycle0, reasons = (
                        renewal.wait_for_root_or_censor(
                            state, cycle=avalanche_id, t_model=t_model,
                            threshold=root_threshold,
                            total_completed=total_completed, setup=setup,
                            geom=geom, evaluator=evaluator, scalar=scalar,
                            movie=movie, first_reload=True))
                    if status == "censored":
                        blocker = dict(
                            reason="geometry_validity_boundary_during_root_reload",
                            avalanche_id=avalanche_id, reasons=reasons,
                            Vp_over_Vp_cycle=root_row["Vp_over_Vp_cycle"],
                            r_n_m=root_row["r_n_m"])
                        outcome = "STOPPED_AT_EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                        break
                    controller.start(
                        avalanche_id=avalanche_id, root_cycle=avalanche_id,
                        start_time_s=t_model * renewal.SECONDS_PER_MODEL_TIME)
                    avalanche_start_model = t_model
                    root_local = root_row["sigma_local_Pa"]
                    root_integral = root_row["sigma_integral_Pa"]
                    current_row = root_row
                    event_number = 0
                    needs_window = False

                avalanche_blocker = None
                while controller.state.avalanche_active:
                    if needs_window:
                        (window_status, state, t_model, current_row,
                         current_branches, reasons) = (
                            renewal.wait_for_descendant_or_extinction(
                                state, avalanche_id=avalanche_id,
                                event_number=event_number, t_model=t_model,
                                total_completed=total_completed,
                                root_hazard=root_row["H"],
                                root_threshold=root_threshold,
                                vp_cycle0=vp_cycle0, setup=setup, geom=geom,
                                evaluator=evaluator, scalar=scalar, movie=movie,
                                controller=controller,
                                descendant_trace=descendant_trace))
                        if window_status == "invalid":
                            avalanche_blocker = dict(
                                reason="geometry_validity_boundary_during_avalanche",
                                avalanche_id=avalanche_id,
                                event_number=event_number, reasons=reasons,
                                Vp_over_Vp_cycle=current_row["Vp_over_Vp_cycle"],
                                r_n_m=current_row["r_n_m"])
                            break
                        if window_status == "extinct":
                            avalanches.append(dict(
                                avalanche_id=avalanche_id,
                                root_threshold=root_threshold,
                                start_time_model=avalanche_start_model,
                                end_time_model=t_model,
                                start_time_s=avalanche_start_model
                                * renewal.SECONDS_PER_MODEL_TIME,
                                end_time_s=t_model * renewal.SECONDS_PER_MODEL_TIME,
                                duration_s=(t_model - avalanche_start_model)
                                * renewal.SECONDS_PER_MODEL_TIME,
                                wait_time_model=avalanche_start_model
                                - wait_start_model,
                                wait_time_s=(avalanche_start_model - wait_start_model)
                                * renewal.SECONDS_PER_MODEL_TIME,
                                S=controller.state.S_completed,
                                Q_avalanche_nm=controller.state.S_completed
                                * renewal.B_EVENT_M * 1e9,
                                sigma_root_local_Pa=root_local,
                                sigma_final_local_Pa=current_row["sigma_local_Pa"],
                                delta_sigma_local_Pa=root_local
                                - current_row["sigma_local_Pa"],
                                delta_sigma_integral_Pa=root_integral
                                - current_row["sigma_integral_Pa"],
                                r_n_root_m=root_row["r_n_m"],
                                r_n_final_m=current_row["r_n_m"],
                                descendant_thresholds_drawn=len(
                                    controller.state.thresholds_drawn)))
                            renewal.write_csv(
                                OUT / "avalanche_summary.csv", avalanches)
                            renewal.save_state(
                                OUT / "checkpoints"
                                / f"avalanche{avalanche_id}_extinct.npz",
                                state, t_model=t_model,
                                total_completed=total_completed,
                                controller_manifest_json=json.dumps(
                                    controller.manifest()))
                            print("AVALANCHE COMPLETE", json.dumps(avalanches[-1]),
                                  flush=True)
                            break

                    event_number += 1
                    controller.begin_transit()
                    event_start_model = t_model
                    pre_local = current_row["sigma_local_Pa"]
                    pre_integral = current_row["sigma_integral_Pa"]
                    try:
                        (state, t_model, restart, current_row, current_branches,
                         samples) = renewal.run_one_b_event(
                            state, avalanche_id=avalanche_id,
                            event_number=event_number,
                            total_completed=total_completed, t_model=t_model,
                            current_row=current_row,
                            root_threshold=root_threshold,
                            vp_cycle0=vp_cycle0, setup=setup, geom=geom,
                            evaluator=evaluator, transport=transport,
                            scalar=scalar, movie=movie, controller=controller)
                    except Exception as error:
                        avalanche_blocker = dict(
                            reason="qualified_1b_event_failed",
                            avalanche_id=avalanche_id,
                            event_number=event_number,
                            message=f"{type(error).__name__}: {error}")
                        break
                    trace = controller.integrate_transit_samples(samples)
                    descendant_trace.extend(renewal.controller_trace_rows(
                        trace, avalanche_id=avalanche_id,
                        event_number=event_number, phase="active_1b_frozen"))
                    controller.complete_transit(
                        t_model * renewal.SECONDS_PER_MODEL_TIME)
                    total_completed += 1
                    vp_cycle0, current_row, current_branches = reset_volume_reference(
                        state, t_model=t_model, avalanche_id=avalanche_id,
                        total_completed=total_completed, root_row=root_row,
                        root_threshold=root_threshold, setup=setup, geom=geom,
                        evaluator=evaluator)
                    subevents.append(dict(
                        avalanche_id=avalanche_id,
                        event_number=event_number,
                        event_type="root" if event_number == 1 else "descendant",
                        start_time_model=event_start_model,
                        end_time_model=t_model,
                        start_time_s=event_start_model
                        * renewal.SECONDS_PER_MODEL_TIME,
                        end_time_s=t_model * renewal.SECONDS_PER_MODEL_TIME,
                        duration_s=(t_model - event_start_model)
                        * renewal.SECONDS_PER_MODEL_TIME,
                        sigma_start_local_Pa=pre_local,
                        sigma_end_local_Pa=current_row["sigma_local_Pa"],
                        delta_sigma_local_Pa=pre_local
                        - current_row["sigma_local_Pa"],
                        delta_sigma_integral_Pa=pre_integral
                        - current_row["sigma_integral_Pa"],
                        descendants_committed_during_transit=0,
                        pending_children_after=controller.state.pending_children,
                        C4=restart["explicit_max_fourth_order_courant"]))
                    renewal.write_csv(OUT / "one_b_subevents.csv", subevents)
                    renewal.write_csv(
                        OUT / "descendant_hazard_trace.csv", descendant_trace)
                    renewal.append_movie(
                        movie, current_branches, current_row,
                        frame_type="child_completion",
                        avalanche_id=avalanche_id,
                        event_number=event_number, avalanche_active=1,
                        sink_state=1, q_event_over_b=1.0,
                        Q_avalanche_over_b=controller.state.S_completed,
                        Q_cumulative_over_b=total_completed,
                        S_completed=controller.state.S_completed,
                        controller=controller, flush=True)
                    renewal.save_state(
                        OUT / "checkpoints"
                        / f"avalanche{avalanche_id}_event{event_number}_complete.npz",
                        state, t_model=t_model,
                        total_completed=total_completed,
                        Vp_cycle0=vp_cycle0,
                        controller_manifest_json=json.dumps(
                            controller.manifest()))
                    renewal.save_controller_checkpoint(
                        controller, avalanche_id=avalanche_id,
                        event_number=event_number)
                    invalid = renewal.geometry_invalid(current_row)
                    if invalid:
                        avalanche_blocker = dict(
                            reason="geometry_validity_boundary_during_avalanche",
                            avalanche_id=avalanche_id,
                            event_number=event_number, reasons=invalid,
                            Vp_over_Vp_cycle=current_row["Vp_over_Vp_cycle"],
                            r_n_m=current_row["r_n_m"])
                        break
                    needs_window = True

                if avalanche_blocker is not None:
                    blocker = avalanche_blocker
                    outcome = (
                        "STOPPED_AT_EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                        if blocker["reason"].startswith("geometry_validity")
                        else "STOPPED_ON_QUALIFIED_EVENT_BLOCKER")
                    break
                if len(avalanches) < avalanche_id:
                    blocker = dict(
                        reason="avalanche_did_not_reach_extinction",
                        avalanche_id=avalanche_id)
                    outcome = "STOPPED_ON_AVALANCHE_CONTROLLER_BLOCKER"
                    break
            else:
                outcome = "THREE_COMPLETE_AVALANCHES"
        except Exception as error:
            blocker = dict(
                reason="unexpected_exception",
                message=f"{type(error).__name__}: {error}")
            outcome = "STOPPED_ON_UNEXPECTED_BLOCKER"
        finally:
            scalar.flush()
            movie.flush()

    renewal.final_figure(movie_path, avalanches)
    movie_result = renewal.morphology_movie(movie_path)
    result = dict(
        outcome=outcome, avalanches_completed=len(avalanches),
        avalanches_requested=TARGET_AVALANCHES,
        total_one_b_events=total_completed, blocker=blocker,
        seed=SEED, root_thresholds=seed_record["root_thresholds"],
        descendant_delta_G_step_eV=renewal.DELTA_G_STEP_EV,
        tau_corr_s=renewal.TAU_CORR_S,
        C4_received_and_asserted=renewal.PRODUCTION_C4,
        volume_reference_reset_after_each_one_b=True,
        resumed_from_event=RESUME_EVENT,
        retained_source_movie_frames=retained_frames,
        movie=movie_result, wall_seconds=time.monotonic() - started,
        avalanches=avalanches)
    (OUT / "avalanche_renewal_result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
