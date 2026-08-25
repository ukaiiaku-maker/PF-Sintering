#!/usr/bin/env python3
"""Matched never-on and always-on controls for the avalanche renewal run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from numba import set_num_threads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pr_avalanche_renewal_five as renewal  # noqa: E402
from pf_sintering.axisym_numba_kernel import NumbaScratch  # noqa: E402
from pf_sintering.pr_movie_geometry import MovieGeometryArchive  # noqa: E402
from pr_coarsening_driven_fourier_loading import integral  # noqa: E402
from pr_experimental_long_sinkoff import build_case  # noqa: E402
from pr_full_deterministic_cycle import event_call, make_transport  # noqa: E402
from pr_tj_node_coupling_gate import make_evaluator  # noqa: E402


MAIN_DEFAULT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_five_deltaG0p300_taucorr9ms")
CONTROL_ROOT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_limit_controls_deltaG0p300")
MOVIE_DT_MODEL = 0.025
PINCH_R_M_FACTOR = 1.0


def read_main(main_dir: Path) -> dict:
    result = json.loads((main_dir / "avalanche_renewal_result.json").read_text())
    validation = json.loads((main_dir / "movie_geometry_validation.json").read_text())
    if validation["status"] != "PASS":
        raise RuntimeError("main compact geometry archive is not qualified")
    return dict(
        result=result,
        t_end_model=float(validation["time_model_range"][1]),
        t_end_s=float(validation["time_s_range"][1]),
        Q_main_over_b=float(result["total_one_b_events"]),
        main_dir=str(main_dir))


def common_setup():
    geom, setup = build_case()
    evaluator = make_evaluator(setup, geom)
    state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    return geom, setup, evaluator, state


def measure(state, *, t_model, sink, q, qcum, setup, geom, evaluator, vp0):
    return renewal.measure(
        state, t_model=t_model, cycle=0, sink=sink, q=q, qcum=qcum,
        hazard=0.0, threshold=1.0, setup=setup, geom=geom,
        evaluator=evaluator, vp_cycle0=vp0)


def movie_add(movie, branches, row, *, frame_type, sink, event_number,
              q_event, q_cumulative, flush=False):
    renewal.append_movie(
        movie, branches, row, frame_type=frame_type,
        avalanche_id=0, event_number=event_number,
        avalanche_active=0, sink_state=sink,
        q_event_over_b=q_event, Q_avalanche_over_b=0.0,
        Q_cumulative_over_b=q_cumulative, S_completed=0,
        pending_children=0, flush=flush)


def scalar_row(row: dict, *, mode: str, event_number: int,
               q_event: float, q_cumulative: float) -> dict:
    return dict(
        row, mode=mode, event_number=event_number,
        q_event_over_b=q_event, Q_cumulative_over_b=q_cumulative,
        avalanche_active=0, pending_children=0,
        t_s=float(row["t_model"] * renewal.SECONDS_PER_MODEL_TIME))


def base_manifest(mode: str, main: dict, setup, geom) -> dict:
    return dict(
        mode=mode,
        matched_main_directory=main["main_dir"],
        matched_main_t_end_model=main["t_end_model"],
        matched_main_t_end_s=main["t_end_s"],
        matched_main_Q_over_b=main["Q_main_over_b"],
        implementation_commit=os.popen("git rev-parse HEAD").read().strip(),
        seconds_per_model_time=renewal.SECONDS_PER_MODEL_TIME,
        C4=renewal.PRODUCTION_C4,
        b_event_m=renewal.B_EVENT_M,
        V_cutoff=renewal.V_CUTOFF,
        r_neck_cutoff_m=renewal.RN_CUTOFF_M,
        tau_coarsen_over_tau_surface=20.0,
        grid_shape=list(geom["f"].shape), dr_m=setup["dr"], dz_m=setup["dz"],
        W_m=setup["W"], R_cyl_m=geom["R_cyl"], lambda_m=geom["lam"],
        movie_dt_model=MOVIE_DT_MODEL, N_branch=renewal.N_BRANCH)


def run_never_on(main: dict) -> None:
    out = CONTROL_ROOT / "never_on_PR"
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "checkpoints").mkdir()
    geom, setup, evaluator, state = common_setup()
    vp0 = integral(state[1], setup)
    manifest = base_manifest("never_on_PR", main, setup, geom)
    manifest.update(
        sink_always_off=True, q_identically_zero=True,
        root_hazard_disabled=True, descendant_hazard_disabled=True,
        event_operator_disabled=True,
        pinch_resolution_r_m=PINCH_R_M_FACTOR * setup["dr"])
    (out / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    renewal.save_state(out / "checkpoints/initial_state.npz", state, t_model=0.0)

    rows = []
    t_model = 0.0
    outcome = "RUNNING"
    blocker = None
    started = time.monotonic()
    movie_path = out / "control_movie_geometry.h5"
    scratches = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
    which = 0
    movie_steps = max(1, int(round(MOVIE_DT_MODEL / setup["dt"])))
    with MovieGeometryArchive(
            movie_path, nbranch=renewal.N_BRANCH,
            seconds_per_model_time=renewal.SECONDS_PER_MODEL_TIME,
            metadata=manifest) as movie:
        row, branches = measure(
            state, t_model=0.0, sink=0, q=0.0, qcum=0.0,
            setup=setup, geom=geom, evaluator=evaluator, vp0=vp0)
        rows.append(scalar_row(
            row, mode="never_on_PR", event_number=0,
            q_event=0.0, q_cumulative=0.0))
        movie_add(movie, branches, row, frame_type="loading", sink=0,
                  event_number=0, q_event=0.0, q_cumulative=0.0, flush=True)
        try:
            while t_model < main["t_end_model"] - 1e-14:
                remaining = main["t_end_model"] - t_model
                steps_left = int(math.floor(remaining / setup["dt"] + 1e-12))
                if steps_left <= 0:
                    t_model = main["t_end_model"]
                    row["t_model"] = t_model
                    rows.append(scalar_row(
                        row, mode="never_on_PR", event_number=0,
                        q_event=0.0, q_cumulative=0.0))
                    movie_add(
                        movie, branches, row, frame_type="loading", sink=0,
                        event_number=0, q_event=0.0, q_cumulative=0.0,
                        flush=True)
                    break
                step_count = min(movie_steps, steps_left)
                state, which = renewal.advance_pf_steps(
                    state, step_count, setup=setup,
                    scratches=scratches, which=which)
                t_model += step_count * setup["dt"]
                row, branches = measure(
                    state, t_model=t_model, sink=0, q=0.0, qcum=0.0,
                    setup=setup, geom=geom, evaluator=evaluator, vp0=vp0)
                rows.append(scalar_row(
                    row, mode="never_on_PR", event_number=0,
                    q_event=0.0, q_cumulative=0.0))
                movie_add(
                    movie, branches, row, frame_type="loading", sink=0,
                    event_number=0, q_event=0.0, q_cumulative=0.0,
                    flush=len(rows) % 10 == 0)
                if row["r_n_m"] <= manifest["pinch_resolution_r_m"]:
                    outcome = "PR_PINCH_OFF_ENDPOINT"
                    break
                if len(rows) % 10 == 0:
                    print(
                        f"NEVER_ON t={t_model:.3f} "
                        f"sigL={row['sigma_local_Pa']/1e6:.3f} "
                        f"rn={row['r_n_m']*1e9:.2f}", flush=True)
            if outcome == "RUNNING":
                outcome = "MATCHED_MAIN_PHYSICAL_HORIZON"
        except Exception as error:
            outcome = "STOPPED_ON_PR_TOPOLOGY_OR_NUMERICAL_ENDPOINT"
            blocker = f"{type(error).__name__}: {error}"
        movie.flush()
    renewal.write_csv(out / "control_history.csv", rows)
    renewal.save_state(
        out / "checkpoints/final_state.npz", state, t_model=t_model,
        Q_over_b=0.0, outcome=outcome)
    result = dict(
        outcome=outcome, blocker=blocker, t_end_model=t_model,
        t_end_s=t_model * renewal.SECONDS_PER_MODEL_TIME,
        Q_over_b=0.0, final_sigma_local_Pa=row["sigma_local_Pa"],
        final_sigma_integral_Pa=row["sigma_integral_Pa"],
        final_r_n_m=row["r_n_m"], wall_seconds=time.monotonic() - started)
    (out / "control_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main_event_wall_estimate(main_dir: Path) -> dict:
    rows = list(csv.DictReader((main_dir / "one_b_subevents.csv").open()))
    durations = [float(row["duration_s"]) for row in rows]
    result = json.loads((main_dir / "avalanche_renewal_result.json").read_text())
    mean_event_physical_s = float(np.mean(durations))
    mean_event_wall_s = float(result["wall_seconds"]) / max(len(rows), 1)
    return dict(
        measured_event_count=len(rows),
        mean_event_physical_s=mean_event_physical_s,
        mean_event_wall_s=mean_event_wall_s)


def run_always_on(main: dict) -> None:
    out = CONTROL_ROOT / "always_on_diffusion"
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "checkpoints").mkdir()
    geom, setup, evaluator, state = common_setup()
    transport = make_transport(geom)
    assert transport.b_m == renewal.B_EVENT_M
    vp0 = integral(state[1], setup)
    estimate = main_event_wall_estimate(Path(main["main_dir"]))
    projected_events = main["t_end_s"] / estimate["mean_event_physical_s"]
    manifest = base_manifest("always_on_diffusion", main, setup, geom)
    manifest.update(
        sink_continuously_available=True, zero_nucleation_wait=True,
        root_hazard_disabled=True, descendant_hazard_disabled=True,
        one_b_segmentation_bookkeeping_only=True,
        cost_estimate=dict(
            **estimate, projected_events_to_main_horizon=projected_events,
            projected_wall_s_to_main_horizon=(
                projected_events * estimate["mean_event_wall_s"])),
        runtime_safeguard=(
            "stop at completed 1b boundary after exceeding main Q/b"))
    (out / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    renewal.save_state(out / "checkpoints/initial_state.npz", state, t_model=0.0)

    rows = []
    t_model = 0.0
    q_cumulative = 0.0
    event_number = 0
    outcome = "RUNNING"
    blocker = None
    started = time.monotonic()
    movie_path = out / "control_movie_geometry.h5"
    with MovieGeometryArchive(
            movie_path, nbranch=renewal.N_BRANCH,
            seconds_per_model_time=renewal.SECONDS_PER_MODEL_TIME,
            metadata=manifest) as movie:
        row, branches = measure(
            state, t_model=0.0, sink=1, q=0.0, qcum=0.0,
            setup=setup, geom=geom, evaluator=evaluator, vp0=vp0)
        rows.append(scalar_row(
            row, mode="always_on_diffusion", event_number=0,
            q_event=0.0, q_cumulative=0.0))
        movie_add(movie, branches, row, frame_type="active_1b_transit",
                  sink=1, event_number=0, q_event=0.0,
                  q_cumulative=0.0, flush=True)
        wait_scratches = [
            NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
        wait_which = 0
        wait_steps = max(1, int(round(MOVIE_DT_MODEL / setup["dt"])))
        try:
            while True:
                event_number += 1
                callback_elapsed = 0.0
                next_movie_q = renewal.EVENT_MOVIE_DQ
                restart = None
                event_start_model = t_model

                def accepted(packet, accepted_state):
                    nonlocal callback_elapsed, next_movie_q
                    callback_elapsed += float(packet["transport_dt_model"])
                    q = float(packet["q_end_over_b"])
                    if q + 1e-12 < next_movie_q or q >= 1.0 - 1e-12:
                        return
                    frame_row, frame_branches = measure(
                        accepted_state,
                        t_model=event_start_model + callback_elapsed,
                        sink=1, q=q, qcum=q_cumulative + q,
                        setup=setup, geom=geom, evaluator=evaluator, vp0=vp0)
                    movie_add(
                        movie, frame_branches, frame_row,
                        frame_type="active_1b_transit", sink=1,
                        event_number=event_number, q_event=q,
                        q_cumulative=q_cumulative + q)
                    while next_movie_q <= q + 1e-12:
                        next_movie_q += renewal.EVENT_MOVIE_DQ

                first_result = None
                while first_result is None:
                    event_start_model = t_model
                    result = event_call(
                        state, setup, geom, evaluator, transport,
                        renewal.EVENT_CHECKPOINTS[0],
                        event_restart=None, state_callback=accepted,
                        explicit_max_fourth_order_courant=renewal.PRODUCTION_C4)
                    if result[3]:
                        first_result = result
                        break
                    reason = result[4].get("stop_reason")
                    if reason != "nonpositive instantaneous GB-to-TJ-node affinity":
                        raise RuntimeError(
                            f"always-on event {event_number} could not start: "
                            f"{reason}")
                    state, wait_which = renewal.advance_pf_steps(
                        state, wait_steps, setup=setup,
                        scratches=wait_scratches, which=wait_which)
                    t_model += wait_steps * setup["dt"]
                    row, branches = measure(
                        state, t_model=t_model, sink=1, q=0.0,
                        qcum=q_cumulative, setup=setup, geom=geom,
                        evaluator=evaluator, vp0=vp0)
                    rows.append(scalar_row(
                        row, mode="always_on_diffusion",
                        event_number=event_number, q_event=0.0,
                        q_cumulative=q_cumulative))
                    movie_add(
                        movie, branches, row, frame_type="loading", sink=1,
                        event_number=event_number, q_event=0.0,
                        q_cumulative=q_cumulative, flush=len(rows) % 10 == 0)
                    invalid = renewal.geometry_invalid(row)
                    print(
                        f"ALWAYS_ON AVAILABLE_WAIT e={event_number} "
                        f"t={t_model:.3f} sigL={row['sigma_local_Pa']/1e6:.3f}",
                        flush=True)
                    if invalid:
                        outcome = "EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                        blocker = invalid
                        break
                    if t_model >= main["t_end_model"]:
                        outcome = "MATCHED_MAIN_PHYSICAL_HORIZON"
                        break
                if first_result is None:
                    break

                for target_index, target in enumerate(renewal.EVENT_CHECKPOINTS):
                    if target_index == 0:
                        result = first_result
                    else:
                        result = event_call(
                            state, setup, geom, evaluator, transport, target,
                            event_restart=restart, state_callback=accepted,
                            explicit_max_fourth_order_courant=(
                                renewal.PRODUCTION_C4))
                    actual_c4 = result[4].get(
                        "explicit_max_fourth_order_courant")
                    if not result[3]:
                        raise RuntimeError(
                            f"always-on event {event_number} failed at "
                            f"q/b={result[4].get('event_progress_over_b')}: "
                            f"{result[4].get('stop_reason')}")
                    if actual_c4 != renewal.PRODUCTION_C4:
                        raise RuntimeError(
                            "event integrator C4 mismatch: "
                            f"received={actual_c4!r}, "
                            f"required={renewal.PRODUCTION_C4!r}")
                    state = tuple(field.copy() for field in result[:3])
                    restart = result[4]["event_restart"]
                    restart["explicit_max_fourth_order_courant"] = actual_c4
                    row, branches = measure(
                        state,
                        t_model=event_start_model + restart["event_time_model"],
                        sink=1, q=target, qcum=q_cumulative + target,
                        setup=setup, geom=geom, evaluator=evaluator, vp0=vp0)
                    rows.append(scalar_row(
                        row, mode="always_on_diffusion",
                        event_number=event_number, q_event=target,
                        q_cumulative=q_cumulative + target))
                    movie_add(
                        movie, branches, row,
                        frame_type="active_1b_transit", sink=1,
                        event_number=event_number, q_event=target,
                        q_cumulative=q_cumulative + target,
                        flush=target == 1.0)
                    print(
                        f"ALWAYS_ON e={event_number} q={target:.2f} "
                        f"sigL={row['sigma_local_Pa']/1e6:.3f}", flush=True)
                t_model = event_start_model + restart["event_time_model"]
                q_cumulative += 1.0
                renewal.save_state(
                    out / "checkpoints"
                    / f"event{event_number}_complete.npz",
                    state, t_model=t_model, Q_over_b=q_cumulative,
                    C4=actual_c4)
                invalid = renewal.geometry_invalid(row)
                if invalid:
                    outcome = "EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                    blocker = invalid
                    break
                if t_model >= main["t_end_model"]:
                    outcome = "MATCHED_MAIN_PHYSICAL_HORIZON"
                    break
                if q_cumulative > main["Q_main_over_b"]:
                    outcome = "MAIN_CUMULATIVE_DISPLACEMENT_EXCEEDED"
                    break
        except Exception as error:
            outcome = "STOPPED_ON_EVENT_OR_NUMERICAL_ENDPOINT"
            blocker = f"{type(error).__name__}: {error}"
        movie.flush()
    renewal.write_csv(out / "control_history.csv", rows)
    result = dict(
        outcome=outcome, blocker=blocker, t_end_model=t_model,
        t_end_s=t_model * renewal.SECONDS_PER_MODEL_TIME,
        Q_over_b=q_cumulative, events_completed=int(q_cumulative),
        final_sigma_local_Pa=row["sigma_local_Pa"],
        final_sigma_integral_Pa=row["sigma_integral_Pa"],
        final_r_n_m=row["r_n_m"], cost_estimate=manifest["cost_estimate"],
        wall_seconds=time.monotonic() - started)
    (out / "control_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("never_on", "always_on"), required=True)
    parser.add_argument("--main-dir", type=Path, default=MAIN_DEFAULT)
    args = parser.parse_args()
    set_num_threads(8)
    main_result = read_main(args.main_dir)
    if args.mode == "never_on":
        run_never_on(main_result)
    else:
        run_always_on(main_result)


if __name__ == "__main__":
    main()
