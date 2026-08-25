#!/usr/bin/env python3
"""One continuous five-avalanche PR renewal trajectory.

The qualified production loading and one-b event mechanics are unchanged.  A
D2 descendant controller is active only between a root crossing and avalanche
extinction.  High-cadence geometry is a diagnostics-only state callback.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads

os.environ["PR_RENEWAL_CASE"] = "fourier"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pr_authoritative_stochastic_ten_event as authoritative  # noqa: E402
import pr_coarsening_stochastic_two_event as base  # noqa: E402
from pf_sintering.axisym_numba_kernel import (  # noqa: E402
    NumbaScratch, axisym_gb_face_projected_step_fast,
)
from pf_sintering.pr_avalanche import AvalancheController, DescendantBarrier  # noqa: E402
from pf_sintering.pr_movie_geometry import FRAME_TYPES, MovieGeometryArchive  # noqa: E402
from pr_coarsening_driven_fourier_loading import (  # noqa: E402
    integral, reservoir_remove_particle,
)
from pr_experimental_long_sinkoff import build_case  # noqa: E402
from pr_full_deterministic_cycle import (  # noqa: E402
    event_call, make_transport, restart_arrays,
)
from pr_tj_node_coupling_gate import make_evaluator  # noqa: E402


OUT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_five_deltaG0p300_taucorr9ms")
TEMPERATURE_K = 1830.15
SECONDS_PER_MODEL_TIME = 0.01557994316955921
CLOCK_SCALE = 15579943169.55921
B_EVENT_M = 0.25e-9
PRODUCTION_C4 = 0.05
DELTA_G_STEP_EV = 0.300000
TAU_CORR_S = 9.0e-3
V_CUTOFF = 0.88
RN_CUTOFF_M = 35e-9
ROOT_ANALYSIS_DT = 0.25
ROOT_MOVIE_DT = 0.025
COMPETING_CROSSING_DT = 0.005
FACILITATED_MOVIE_DT = 0.025
EVENT_MOVIE_DQ = 0.02
N_BRANCH = 256
AVALANCHES_REQUESTED = 5
EVENT_CHECKPOINTS = (0.10, 0.25, 0.50, 0.75, 1.00)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_state(path: Path, state, **metadata) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, f=state[0], particle=state[1], substrate=state[2],
            **{key: np.asarray(value) for key, value in metadata.items()})
    os.replace(temporary, path)


class ScalarOutput:
    """Normal-cadence scalar/raw-contour output; independent of movie HDF5."""

    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self.contours: list[dict] = []

    def add(self, row: dict, branches: dict, **extra) -> dict:
        merged = dict(row, **extra, sample_id=len(self.rows))
        merged["t_s"] = merged["t_model"] * SECONDS_PER_MODEL_TIME
        self.rows.append(merged)
        self.contours.append(dict(
            sample_id=merged["sample_id"],
            z_negative=np.asarray(branches["negative"]["z_m"]),
            r_negative=np.asarray(branches["negative"]["r_m"]),
            z_positive=np.asarray(branches["positive"]["z_m"]),
            r_positive=np.asarray(branches["positive"]["r_m"])))
        return merged

    def flush(self) -> None:
        write_csv(self.out / "avalanche_renewal_history.csv", self.rows)
        arrays = {}
        for item in self.contours:
            sample_id = item["sample_id"]
            for key, value in item.items():
                if key != "sample_id":
                    arrays[f"sample_{sample_id:05d}_{key}"] = value
        temporary = self.out / "avalanche_renewal_sparse_contours.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, self.out / "avalanche_renewal_sparse_contours.npz")


def geometry_invalid(row: dict) -> list[str]:
    reasons = []
    if row["Vp_over_Vp_cycle"] < V_CUTOFF:
        reasons.append("Vp_over_Vp_cycle_below_0.88")
    if row["r_n_m"] < RN_CUTOFF_M:
        reasons.append("r_n_below_35_nm")
    return reasons


def movie_payload(row: dict, *, avalanche_id: int, event_number: int,
                  avalanche_active: int, sink_state: int,
                  q_event_over_b: float, Q_avalanche_over_b: float,
                  Q_cumulative_over_b: float, S_completed: int,
                  pending_children: int, controller=None) -> dict:
    state = None if controller is None else controller.state
    return dict(
        t_model=float(row["t_model"]),
        t_s=float(row["t_model"] * SECONDS_PER_MODEL_TIME),
        cycle=int(row["cycle"]), avalanche_id=int(avalanche_id),
        event_number=int(event_number), sink_state=int(sink_state),
        avalanche_active=int(avalanche_active),
        q_event_over_b=float(q_event_over_b),
        Q_avalanche_over_b=float(Q_avalanche_over_b),
        Q_cumulative_over_b=float(Q_cumulative_over_b),
        sigma_local_Pa=float(row["sigma_local_Pa"]),
        sigma_integral_Pa=float(row["sigma_integral_Pa"]),
        r_neck_m=float(row["r_n_m"]),
        Vp_over_Vp_cycle=float(row["Vp_over_Vp_cycle"]),
        z_TJ_m=float(row["z_TJ_m"]), r_TJ_m=float(row["r_n_m"]),
        z_GB_m=float(row["z_TJ_m"]), r_GB_m=float(row["r_n_m"]),
        root_hazard=float(row.get("H", np.nan)),
        root_threshold=float(row.get("H_threshold", np.nan)),
        descendant_hazard=(float(state.descendant_hazard)
                           if state is not None else np.nan),
        descendant_threshold=(float(state.descendant_threshold)
                              if state is not None else np.nan),
        S_completed=int(S_completed),
        pending_children=(int(state.pending_children)
                          if state is not None else int(pending_children)))


def append_movie(movie: MovieGeometryArchive, branches: dict, row: dict,
                 *, frame_type: str, avalanche_id: int, event_number: int,
                 avalanche_active: int, sink_state: int,
                 q_event_over_b: float, Q_avalanche_over_b: float,
                 Q_cumulative_over_b: float, S_completed: int,
                 pending_children: int = 0, controller=None,
                 flush: bool = False) -> int:
    payload = movie_payload(
        row, avalanche_id=avalanche_id, event_number=event_number,
        avalanche_active=avalanche_active, sink_state=sink_state,
        q_event_over_b=q_event_over_b,
        Q_avalanche_over_b=Q_avalanche_over_b,
        Q_cumulative_over_b=Q_cumulative_over_b,
        S_completed=S_completed, pending_children=pending_children,
        controller=controller)
    return movie.append(branches, payload, frame_type=frame_type, flush=flush)


def output_extra(*, avalanche_id: int, event_number: int,
                 avalanche_active: int, q_event: float,
                 Q_avalanche: float, Q_cumulative: float,
                 S_completed: int, controller=None, frame_type: str) -> dict:
    state = None if controller is None else controller.state
    return dict(
        avalanche_id=avalanche_id, event_number=event_number,
        avalanche_active=avalanche_active, q_event_over_b=q_event,
        Q_avalanche_over_b=Q_avalanche,
        Q_cumulative_over_b=Q_cumulative, S_completed=S_completed,
        pending_children=(state.pending_children if state is not None else 0),
        descendant_hazard=(state.descendant_hazard if state is not None else np.nan),
        descendant_threshold=(state.descendant_threshold if state is not None else np.nan),
        frame_type=frame_type)


def measure(state, *, t_model: float, cycle: int, sink: int, q: float,
            qcum: float, hazard: float, threshold: float, setup, geom,
            evaluator, vp_cycle0: float):
    return base.measure(
        state, t_model=t_model, cycle=cycle, sink=sink, q=q, qcum=qcum,
        hazard=hazard, threshold=threshold, clock_scale=CLOCK_SCALE,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp_cycle0)


def advance_pf_steps(state, nstep: int, *, setup, scratches, which: int):
    for _ in range(nstep):
        destination = 0 if which != 0 else 1
        state = axisym_gb_face_projected_step_fast(
            *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
            setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
            setup["M_eta"], setup["W"], scratches[destination])
        which = destination
        f, particle, substrate, _ = reservoir_remove_particle(state, setup)
        state = f, particle, substrate
    return state, which


def append_wait_buffer(movie, buffer, *, avalanche_id, total_completed,
                       first_reload, root_nucleation=False, flush=True):
    for index, (row, branches) in enumerate(buffer):
        if root_nucleation and index == len(buffer) - 1:
            frame_type = "root_nucleation"
        elif first_reload and index == 0:
            frame_type = "post_avalanche_reload"
        else:
            frame_type = "loading"
        append_movie(
            movie, branches, row, frame_type=frame_type,
            avalanche_id=avalanche_id if root_nucleation else 0,
            event_number=0, avalanche_active=0, sink_state=0,
            q_event_over_b=0.0, Q_avalanche_over_b=0.0,
            Q_cumulative_over_b=total_completed, S_completed=0,
            flush=flush and index == len(buffer) - 1)


def wait_for_root_or_censor(
        state, *, cycle: int, t_model: float, threshold: float,
        total_completed: int, setup, geom, evaluator, scalar, movie,
        first_reload: bool):
    vp_cycle0 = integral(state[1], setup)
    initial, _ = measure(
        state, t_model=t_model, cycle=cycle, sink=0, q=0.0,
        qcum=total_completed, hazard=0.0, threshold=threshold,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp_cycle0)
    hazard = 0.0
    previous_gamma = initial["Gamma_per_model_time"]
    current = initial
    scratches = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
    which = 0
    analysis_steps = int(round(ROOT_ANALYSIS_DT / setup["dt"]))
    movie_steps = max(1, int(round(ROOT_MOVIE_DT / setup["dt"])))
    elapsed = 0.0
    while True:
        state_before = tuple(field.copy() for field in state)
        elapsed_before = elapsed
        hazard_before = hazard
        previous_before = previous_gamma
        current_before = current
        block_buffer = []
        remaining = analysis_steps
        while remaining:
            step_count = min(movie_steps, remaining)
            state, which = advance_pf_steps(
                state, step_count, setup=setup, scratches=scratches, which=which)
            elapsed += step_count * setup["dt"]
            movie_row, movie_branches = measure(
                state, t_model=t_model + elapsed, cycle=cycle, sink=0, q=0.0,
                qcum=total_completed, hazard=hazard, threshold=threshold,
                setup=setup, geom=geom, evaluator=evaluator,
                vp_cycle0=vp_cycle0)
            block_buffer.append((movie_row, movie_branches))
            remaining -= step_count
        trial, branches = block_buffer[-1]
        block_dt = elapsed - elapsed_before
        hazard += 0.5 * (previous_gamma + trial["Gamma_per_model_time"]) * block_dt
        trial["H"] = hazard
        trial["H_over_threshold"] = hazard / threshold
        block_buffer[-1][0]["H"] = hazard
        block_buffer[-1][0]["H_over_threshold"] = hazard / threshold
        invalid = geometry_invalid(trial)
        crossing = hazard >= threshold

        refine = False
        if invalid and crossing and hazard_before < threshold:
            alpha_h = ((threshold - hazard_before)
                       / max(hazard - hazard_before, 1e-300))
            alpha_v = ((current_before["Vp_over_Vp_cycle"] - V_CUTOFF)
                       / max(current_before["Vp_over_Vp_cycle"]
                             - trial["Vp_over_Vp_cycle"], 1e-300))
            alpha_r = ((current_before["r_n_m"] - RN_CUTOFF_M)
                       / max(current_before["r_n_m"] - trial["r_n_m"], 1e-300))
            refine = alpha_h < min(alpha_v, alpha_r)

        if refine:
            state = state_before
            elapsed = elapsed_before
            hazard = hazard_before
            previous_gamma = previous_before
            scratches = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
            which = 0
            ref_steps = max(1, int(round(COMPETING_CROSSING_DT / setup["dt"])))
            movie_stride = max(1, int(round(ROOT_MOVIE_DT / COMPETING_CROSSING_DT)))
            ref_index = 0
            ref_buffer = []
            while True:
                state, which = advance_pf_steps(
                    state, ref_steps, setup=setup, scratches=scratches, which=which)
                ref_dt = ref_steps * setup["dt"]
                elapsed += ref_dt
                refined, branches = measure(
                    state, t_model=t_model + elapsed, cycle=cycle, sink=0,
                    q=0.0, qcum=total_completed, hazard=hazard,
                    threshold=threshold, setup=setup, geom=geom,
                    evaluator=evaluator, vp_cycle0=vp_cycle0)
                hazard += 0.5 * (
                    previous_gamma + refined["Gamma_per_model_time"]) * ref_dt
                refined["H"] = hazard
                refined["H_over_threshold"] = hazard / threshold
                ref_index += 1
                if ref_index % movie_stride == 0:
                    ref_buffer.append((dict(refined), branches))
                invalid = geometry_invalid(refined)
                crossing = hazard >= threshold
                if crossing or invalid:
                    if not ref_buffer or ref_buffer[-1][0]["t_model"] < refined["t_model"]:
                        ref_buffer.append((dict(refined), branches))
                    append_wait_buffer(
                        movie, ref_buffer, avalanche_id=cycle,
                        total_completed=total_completed, first_reload=first_reload,
                        root_nucleation=crossing and not invalid)
                    scalar.add(
                        refined, branches, **output_extra(
                            avalanche_id=cycle if crossing and not invalid else 0,
                            event_number=0, avalanche_active=0, q_event=0.0,
                            Q_avalanche=0.0, Q_cumulative=total_completed,
                            S_completed=0,
                            frame_type=("root_nucleation"
                                        if crossing and not invalid else "loading")))
                    scalar.flush()
                    if crossing and not invalid:
                        save_state(
                            OUT / "checkpoints" / f"avalanche{cycle}_before_root.npz",
                            state, t_model=t_model + elapsed, H=hazard,
                            Hstar=threshold, Vp_over_Vp_cycle=refined["Vp_over_Vp_cycle"],
                            r_n_m=refined["r_n_m"])
                        return "nucleated", state, t_model + elapsed, refined, branches, vp_cycle0, None
                    save_state(
                        OUT / "checkpoints" / f"avalanche{cycle}_validity_censored.npz",
                        state, t_model=t_model + elapsed, H=hazard,
                        Hstar=threshold, Vp_over_Vp_cycle=refined["Vp_over_Vp_cycle"],
                        r_n_m=refined["r_n_m"])
                    return "censored", state, t_model + elapsed, refined, branches, vp_cycle0, invalid
                previous_gamma = refined["Gamma_per_model_time"]

        append_wait_buffer(
            movie, block_buffer, avalanche_id=cycle,
            total_completed=total_completed, first_reload=first_reload,
            root_nucleation=crossing and not invalid)
        first_reload = False
        scalar.add(
            trial, branches, **output_extra(
                avalanche_id=cycle if crossing and not invalid else 0,
                event_number=0, avalanche_active=0, q_event=0.0,
                Q_avalanche=0.0, Q_cumulative=total_completed,
                S_completed=0,
                frame_type=("root_nucleation"
                            if crossing and not invalid else "loading")))
        scalar.flush()
        save_state(
            OUT / "checkpoints" / f"avalanche{cycle}_waiting_latest.npz",
            state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
            Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"], r_n_m=trial["r_n_m"])
        print(
            f"RENEWAL WAIT a{cycle} t={elapsed:.3f} "
            f"V={trial['Vp_over_Vp_cycle']:.5f} rn={trial['r_n_m']*1e9:.2f} "
            f"sigL={trial['sigma_local_Pa']/1e6:.3f} H/H*={hazard/threshold:.4f}",
            flush=True)
        if invalid:
            save_state(
                OUT / "checkpoints" / f"avalanche{cycle}_validity_censored.npz",
                state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
                Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"], r_n_m=trial["r_n_m"])
            return "censored", state, t_model + elapsed, trial, branches, vp_cycle0, invalid
        if crossing:
            save_state(
                OUT / "checkpoints" / f"avalanche{cycle}_before_root.npz",
                state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
                Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"], r_n_m=trial["r_n_m"])
            return "nucleated", state, t_model + elapsed, trial, branches, vp_cycle0, None
        previous_gamma = trial["Gamma_per_model_time"]
        current = trial


def wait_for_descendant_or_extinction(
        state, *, avalanche_id: int, event_number: int, t_model: float,
        total_completed: int, root_hazard: float, root_threshold: float,
        vp_cycle0: float, setup, geom, evaluator, scalar, movie, controller,
        descendant_trace: list[dict]):
    """Evolve sink-OFF PF physics during one facilitated-source window."""
    if not math.isfinite(controller.state.descendant_threshold):
        raise RuntimeError("descendant window was not reset after the event")
    if controller.state.window_triggered:
        raise RuntimeError("cannot wait on an already-triggered source")

    current, branches = measure(
        state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
        qcum=total_completed, hazard=root_hazard, threshold=root_threshold,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp_cycle0)
    rate_previous = controller.rate(
        current["sigma_local_Pa"], current["r_n_m"])
    scratches = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
    which = 0
    movie_steps = max(1, int(round(FACILITATED_MOVIE_DT / setup["dt"])))
    sample_index = 0

    while True:
        time_s = t_model * SECONDS_PER_MODEL_TIME
        remaining_s = controller.state.window_deadline_s - time_s
        tolerance_s = 64.0 * math.ulp(max(abs(time_s), 1.0))
        if remaining_s <= tolerance_s:
            controller.expire_window(controller.state.window_deadline_s)
            t_model = controller.state.window_deadline_s / SECONDS_PER_MODEL_TIME
            current, branches = measure(
                state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
                qcum=total_completed, hazard=root_hazard,
                threshold=root_threshold, setup=setup, geom=geom,
                evaluator=evaluator, vp_cycle0=vp_cycle0)
            scalar.add(
                current, branches, **output_extra(
                    avalanche_id=avalanche_id, event_number=event_number,
                    avalanche_active=0, q_event=0.0,
                    Q_avalanche=controller.state.S_completed,
                    Q_cumulative=total_completed,
                    S_completed=controller.state.S_completed,
                    controller=controller, frame_type="avalanche_extinction"))
            append_movie(
                movie, branches, current, frame_type="avalanche_extinction",
                avalanche_id=avalanche_id, event_number=event_number,
                avalanche_active=0, sink_state=0, q_event_over_b=0.0,
                Q_avalanche_over_b=controller.state.S_completed,
                Q_cumulative_over_b=total_completed,
                S_completed=controller.state.S_completed,
                controller=controller, flush=True)
            scalar.flush()
            return "extinct", state, t_model, current, branches, None

        remaining_model = remaining_s / SECONDS_PER_MODEL_TIME
        remaining_steps = int(math.floor(
            remaining_model / setup["dt"] + 1.0e-12))
        if remaining_steps <= 0:
            result = controller.accumulate_window_segment(
                rate_start_per_s=rate_previous,
                rate_end_per_s=rate_previous,
                start_time_s=time_s,
                end_time_s=controller.state.window_deadline_s)
            if result["crossed"]:
                t_model = (float(result["crossing_time_s"])
                           / SECONDS_PER_MODEL_TIME)
                current["t_model"] = t_model
                frame_type = "correlation_window"
                scalar.add(
                    current, branches, **output_extra(
                        avalanche_id=avalanche_id,
                        event_number=event_number,
                        avalanche_active=1, q_event=0.0,
                        Q_avalanche=controller.state.S_completed,
                        Q_cumulative=total_completed,
                        S_completed=controller.state.S_completed,
                        controller=controller, frame_type=frame_type))
                append_movie(
                    movie, branches, current, frame_type=frame_type,
                    avalanche_id=avalanche_id, event_number=event_number,
                    avalanche_active=1, sink_state=0, q_event_over_b=0.0,
                    Q_avalanche_over_b=controller.state.S_completed,
                    Q_cumulative_over_b=total_completed,
                    S_completed=controller.state.S_completed,
                    controller=controller, flush=True)
                scalar.flush()
                return "continued", state, t_model, current, branches, None
            t_model = controller.state.window_deadline_s / SECONDS_PER_MODEL_TIME
            continue

        step_count = min(movie_steps, remaining_steps)
        segment_start_s = time_s
        state, which = advance_pf_steps(
            state, step_count, setup=setup, scratches=scratches, which=which)
        t_model += step_count * setup["dt"]
        trial, branches = measure(
            state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
            qcum=total_completed, hazard=root_hazard,
            threshold=root_threshold, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp_cycle0)
        rate_trial = controller.rate(
            trial["sigma_local_Pa"], trial["r_n_m"])
        result = controller.accumulate_window_segment(
            rate_start_per_s=rate_previous, rate_end_per_s=rate_trial,
            start_time_s=segment_start_s,
            end_time_s=t_model * SECONDS_PER_MODEL_TIME)
        crossed = bool(result["crossed"])
        trace = controller._trace_row(
            descendant_sample(trial), rate_trial, int(crossed))
        descendant_trace.append(dict(
            trace, avalanche_id=avalanche_id, event_number=event_number,
            phase="facilitated_sink_off"))
        write_csv(OUT / "descendant_hazard_trace.csv", descendant_trace)
        invalid = geometry_invalid(trial)
        frame_type = "correlation_window"
        append_movie(
            movie, branches, trial, frame_type=frame_type,
            avalanche_id=avalanche_id, event_number=event_number,
            avalanche_active=1, sink_state=0, q_event_over_b=0.0,
            Q_avalanche_over_b=controller.state.S_completed,
            Q_cumulative_over_b=total_completed,
            S_completed=controller.state.S_completed,
            controller=controller,
            flush=crossed or bool(invalid) or sample_index % 10 == 0)
        sample_index += 1
        if crossed or invalid or sample_index % 10 == 0:
            scalar.add(
                trial, branches, **output_extra(
                    avalanche_id=avalanche_id,
                    event_number=event_number,
                    avalanche_active=1, q_event=0.0,
                    Q_avalanche=controller.state.S_completed,
                    Q_cumulative=total_completed,
                    S_completed=controller.state.S_completed,
                    controller=controller, frame_type=frame_type))
            scalar.flush()
        save_state(
            OUT / "checkpoints"
            / f"avalanche{avalanche_id}_facilitated_wait_latest.npz",
            state, t_model=t_model,
            descendant_hazard=controller.state.descendant_hazard,
            descendant_threshold=controller.state.descendant_threshold,
            window_deadline_s=controller.state.window_deadline_s,
            Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"],
            r_n_m=trial["r_n_m"])
        if invalid:
            return "invalid", state, t_model, trial, branches, invalid
        if crossed:
            save_state(
                OUT / "checkpoints"
                / f"avalanche{avalanche_id}_event{event_number + 1}_before_descendant.npz",
                state, t_model=t_model,
                descendant_hazard=controller.state.descendant_hazard,
                descendant_threshold=controller.state.descendant_threshold)
            return "continued", state, t_model, trial, branches, None
        rate_previous = rate_trial


def descendant_sample(row: dict) -> dict:
    return dict(
        t_s=float(row["t_model"] * SECONDS_PER_MODEL_TIME),
        sigma_local_Pa=float(row["sigma_local_Pa"]),
        sigma_integral_Pa=float(row["sigma_integral_Pa"]),
        r_TJ_m=float(row["r_n_m"]))


def run_one_b_event(
        state, *, avalanche_id: int, event_number: int, total_completed: int,
        t_model: float, current_row: dict, root_threshold: float,
        vp_cycle0: float, setup, geom, evaluator, transport, scalar, movie,
        controller):
    start_model = t_model
    restart = None
    callback_elapsed = 0.0
    next_movie_q = EVENT_MOVIE_DQ
    checkpoint_samples = [descendant_sample(current_row)]
    final_branches = None

    def accepted(packet, accepted_state):
        nonlocal callback_elapsed, next_movie_q
        callback_elapsed += float(packet["transport_dt_model"])
        q = float(packet["q_end_over_b"])
        if q + 1e-12 < next_movie_q or q >= 1.0 - 1e-12:
            return
        row, branches = measure(
            accepted_state, t_model=start_model + callback_elapsed,
            cycle=avalanche_id, sink=1, q=q,
            qcum=total_completed + q, hazard=current_row["H"],
            threshold=root_threshold, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp_cycle0)
        append_movie(
            movie, branches, row, frame_type="active_1b_transit",
            avalanche_id=avalanche_id, event_number=event_number,
            avalanche_active=1, sink_state=1, q_event_over_b=q,
            Q_avalanche_over_b=controller.state.S_completed + q,
            Q_cumulative_over_b=total_completed + q,
            S_completed=controller.state.S_completed, controller=controller)
        while next_movie_q <= q + 1e-12:
            next_movie_q += EVENT_MOVIE_DQ

    for target in EVENT_CHECKPOINTS:
        result = event_call(
            state, setup, geom, evaluator, transport, target,
            event_restart=restart, state_callback=accepted,
            explicit_max_fourth_order_courant=PRODUCTION_C4)
        actual_c4 = result[4].get("explicit_max_fourth_order_courant")
        assert actual_c4 == PRODUCTION_C4
        if not result[3]:
            save_state(
                OUT / "checkpoints"
                / f"avalanche{avalanche_id}_event{event_number}_failed.npz",
                result[:3], q_over_b=result[4]["event_progress_over_b"])
            raise RuntimeError(
                f"avalanche {avalanche_id} event {event_number} failed at "
                f"q/b={result[4]['event_progress_over_b']}: "
                f"{result[4].get('stop_reason')}")
        state = tuple(field.copy() for field in result[:3])
        restart = result[4]["event_restart"]
        restart["explicit_max_fourth_order_courant"] = actual_c4
        row, final_branches = measure(
            state, t_model=start_model + restart["event_time_model"],
            cycle=avalanche_id, sink=1, q=target,
            qcum=total_completed + target, hazard=current_row["H"],
            threshold=root_threshold, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp_cycle0)
        scalar.add(
            row, final_branches, **output_extra(
                avalanche_id=avalanche_id, event_number=event_number,
                avalanche_active=1, q_event=target,
                Q_avalanche=controller.state.S_completed + target,
                Q_cumulative=total_completed + target,
                S_completed=controller.state.S_completed,
                controller=controller, frame_type="active_1b_transit"))
        checkpoint_samples.append(descendant_sample(row))
        path = (OUT / "checkpoints"
                / f"avalanche{avalanche_id}_event{event_number}_q{target:.2f}b.npz")
        with path.open("wb") as handle:
            np.savez_compressed(handle, **restart_arrays(state, restart))
        scalar.flush()
        print(
            f"RENEWAL EVENT a{avalanche_id} e{event_number} q={target:.2f} "
            f"sigL={row['sigma_local_Pa']/1e6:.3f}", flush=True)
    return (state, start_model + restart["event_time_model"], restart,
            row, final_branches, checkpoint_samples)


def controller_trace_rows(trace: list[dict], *, avalanche_id: int,
                          event_number: int,
                          phase: str = "active_1b_frozen") -> list[dict]:
    return [dict(item, avalanche_id=avalanche_id,
                 event_number=event_number, phase=phase)
            for item in trace]


def save_controller_checkpoint(controller, *, avalanche_id: int,
                               event_number: int) -> None:
    path = OUT / "controller_latest.json"
    path.write_text(json.dumps(dict(
        avalanche_id=avalanche_id, event_number=event_number,
        manifest=controller.manifest()), indent=2) + "\n")


def final_figure(movie_path: Path, avalanches: list[dict]) -> None:
    with h5py.File(movie_path, "r") as file:
        t_ms = np.asarray(file["time/t_s"]) * 1e3
        local = np.asarray(file["state/sigma_local_Pa"]) / 1e6
        integral_stress = np.asarray(file["state/sigma_integral_Pa"]) / 1e6
        cumulative = np.asarray(file["state/Q_cumulative_over_b"])
        sink = np.asarray(file["state/sink_state"])
        active = np.asarray(file["state/avalanche_active"])
        neck = np.asarray(file["state/r_neck_m"]) * 1e9
    fig, axes = plt.subplots(
        6, 1, figsize=(12, 14), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.7, 1.1, 1.0, .65, .65, 1.0]})
    axes[0].plot(t_ms, local, color="#174a73", label="local")
    axes[0].plot(t_ms, integral_stress, color="#9a6b28", linestyle="--",
                 label="integral")
    axes[0].set_ylabel("Stress (MPa)")
    axes[0].legend(frameon=False)
    axes[1].plot(t_ms, cumulative, color="#2f6f58")
    axes[1].set_ylabel(r"$Q_{cumulative}/b$")
    axes[2].set_ylabel("Avalanche size S")
    for avalanche in avalanches:
        start = avalanche["start_time_s"] * 1e3
        end = avalanche["end_time_s"] * 1e3
        midpoint = 0.5 * (start + end)
        size = avalanche["S"]
        axes[2].scatter([midpoint], [size], color="#67507e", zorder=3)
        axes[2].text(midpoint, size, f"  S={size}", va="center", fontsize=9)
        for ax in axes:
            ax.axvspan(start, end, color="#c65f37", alpha=.13, linewidth=0)
    axes[3].step(t_ms, sink, where="post", color="#30343b")
    axes[3].set_ylabel("Sink")
    axes[3].set_yticks([0, 1], ["OFF", "ON"])
    axes[4].step(t_ms, active, where="post", color="#8b4f43")
    axes[4].set_ylabel("Avalanche")
    axes[4].set_yticks([0, 1], ["OFF", "ON"])
    axes[5].plot(t_ms, neck, color="#4d6d72")
    axes[5].set_ylabel("Neck radius (nm)")
    axes[5].set_xlabel("Physical time (ms)")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Continuous stochastic PR avalanche renewal")
    fig.savefig(OUT / "avalanche_renewal_five_aligned.png", dpi=220)
    fig.savefig(OUT / "avalanche_renewal_five_aligned.pdf")
    plt.close(fig)


def morphology_movie(movie_path: Path) -> dict:
    with h5py.File(movie_path, "r") as file:
        frame_types = np.asarray(file["state/frame_type"])
        nframe = len(frame_types)
        active_codes = {
            FRAME_TYPES["root_nucleation"], FRAME_TYPES["active_1b_transit"],
            FRAME_TYPES["correlation_window"], FRAME_TYPES["avalanche_extinction"],
            FRAME_TYPES["child_completion"]}
        active_indices = np.asarray(
            [i for i, code in enumerate(frame_types) if code in active_codes],
            dtype=int)
        loading_indices = np.asarray(
            [i for i, code in enumerate(frame_types) if code not in active_codes],
            dtype=int)
        keep_active_stride = max(1, math.ceil(len(active_indices) / 320))
        keep_loading_stride = max(1, math.ceil(len(loading_indices) / 100))
        indices = sorted(set(
            active_indices[::keep_active_stride].tolist()
            + loading_indices[::keep_loading_stride].tolist()
            + [0, nframe - 1]))
        all_z = np.concatenate([
            np.asarray(file["geometry/z_negative_m"]),
            np.asarray(file["geometry/z_positive_m"])]) * 1e9
        all_r = np.concatenate([
            np.asarray(file["geometry/r_negative_m"]),
            np.asarray(file["geometry/r_positive_m"])]) * 1e9
        xlim = (float(np.min(all_z) - 5), float(np.max(all_z) + 5))
        rmax = float(np.max(all_r) + 5)
        target = OUT / "avalanche_renewal_morphology.gif"
        with imageio.get_writer(target, mode="I", duration=0.08, loop=0) as writer:
            for index in indices:
                fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=90)
                for side, color in (("negative", "#b45309"),
                                    ("positive", "#1d4ed8")):
                    z = np.asarray(file[f"geometry/z_{side}_m"][index]) * 1e9
                    r = np.asarray(file[f"geometry/r_{side}_m"][index]) * 1e9
                    ax.plot(z, r, color=color, linewidth=1.8)
                    ax.plot(z, -r, color=color, linewidth=1.8)
                ztj = float(file["state/z_TJ_m"][index]) * 1e9
                rtj = float(file["state/r_TJ_m"][index]) * 1e9
                ax.scatter([ztj, ztj], [rtj, -rtj], c="crimson", s=18)
                t_ms = float(file["time/t_s"][index]) * 1e3
                aid = int(file["state/avalanche_id"][index])
                event = int(file["state/event_number"][index])
                sigma = float(file["state/sigma_local_Pa"][index]) / 1e6
                sink = int(file["state/sink_state"][index])
                ax.set_xlim(*xlim)
                ax.set_ylim(-rmax, rmax)
                ax.set_aspect("equal")
                ax.set_title(
                    f"t={t_ms:.3f} ms   avalanche={aid} event={event}   "
                    f"sigma={sigma:.1f} MPa   sink={'ON' if sink else 'OFF'}")
                ax.set_xlabel("z (nm)")
                ax.set_ylabel("r (nm)")
                ax.grid(alpha=.15)
                fig.canvas.draw()
                image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(image)
                plt.close(fig)
    return dict(source_frames=nframe, rendered_frames=len(indices), path=str(target))


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    set_num_threads(8)
    assert PRODUCTION_C4 == 0.05
    assert DELTA_G_STEP_EV == 0.300000
    assert TAU_CORR_S == 9.0e-3
    exported, root_slice = authoritative.configure_barrier()
    assert base.BARRIER.G0_eV == root_slice["G0_eV"]
    base.OUT = OUT
    base.MIN_SOLVABLE_VOLUME_RATIO = -math.inf

    seed = int.from_bytes(os.urandom(16), "big")
    seed_record = dict(
        seed=seed, source="128 bits from os.urandom",
        recorded_unix_time=time.time(), thresholds_generated=False,
        root_thresholds=[], stream_model="SeedSequence spawned root/descendant streams")
    (OUT / "rng_seed_record.json").write_text(json.dumps(seed_record, indent=2) + "\n")
    streams = np.random.SeedSequence(seed).spawn(2)
    root_rng = np.random.default_rng(streams[0])
    descendant_rng = np.random.default_rng(streams[1])
    root_threshold = float(root_rng.exponential())
    seed_record["thresholds_generated"] = True
    seed_record["root_thresholds"].append(root_threshold)
    (OUT / "rng_seed_record.json").write_text(json.dumps(seed_record, indent=2) + "\n")

    geom, setup = build_case()
    evaluator = make_evaluator(setup, geom)
    transport = make_transport(geom)
    assert transport.b_m == B_EVENT_M
    state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    root_params = base.BARRIER
    controller = AvalancheController(
        barrier=DescendantBarrier(root_params, DELTA_G_STEP_EV),
        temperature_K=TEMPERATURE_K,
        attempt_frequency_per_s=float(exported["constants"]["nu0_sinv"]),
        b_m=B_EVENT_M,
        correlation_time_s=TAU_CORR_S,
        rng=descendant_rng)
    manifest = dict(
        accepted_architecture="D2 serialized finite-1b avalanche renewal",
        production_parent_commit="d2aafbf7df99728acb67af941f2c078126455e1f",
        implementation_commit=os.popen("git rev-parse HEAD").read().strip(),
        temperature_K=TEMPERATURE_K,
        descendant_delta_G_step_eV=DELTA_G_STEP_EV,
        descendant_barrier_definition=(
            "max(G_floor, G_root_star(sigma_local)-delta_G_step)"),
        tau_corr_s=controller.correlation_time_s,
        root_barrier_export=str(authoritative.EXPORT),
        root_barrier_export_sha256=hashlib.sha256(
            authoritative.EXPORT.read_bytes()).hexdigest(),
        root_barrier_slice=root_slice, seconds_per_model_time=SECONDS_PER_MODEL_TIME,
        clock_scale=CLOCK_SCALE, b_event_m=B_EVENT_M, C4=PRODUCTION_C4,
        V_cutoff=V_CUTOFF, r_neck_cutoff_m=RN_CUTOFF_M,
        root_analysis_dt_model=ROOT_ANALYSIS_DT,
        loading_movie_dt_model=ROOT_MOVIE_DT,
        event_movie_dq_over_b=EVENT_MOVIE_DQ, N_branch=N_BRANCH,
        facilitated_movie_dt_model=FACILITATED_MOVIE_DT,
        descendant_clock_frozen_during_active_1b=True,
        descendant_pending_queue_enabled=False,
        correlation_window_evolves_sink_off_PF=True,
        no_descendant_memory_between_avalanches=True,
        no_elastic_facilitation=True, no_parameter_retuning=True,
        grid_shape=list(state[0].shape), dr_m=setup["dr"], dz_m=setup["dz"],
        W_m=setup["W"], R_cyl_m=geom["R_cyl"], lambda_m=geom["lam"])
    (OUT / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    scalar = ScalarOutput(OUT)
    movie_path = OUT / "avalanche_renewal_movie_geometry.h5"
    t_model = 0.0
    total_completed = 0
    avalanches = []
    subevents = []
    descendant_trace = []
    censor = None
    blocker = None
    outcome = "RUNNING"
    started = time.monotonic()

    with MovieGeometryArchive(
            movie_path, nbranch=N_BRANCH,
            seconds_per_model_time=SECONDS_PER_MODEL_TIME,
            metadata=manifest) as movie:
        vp0 = integral(state[1], setup)
        initial, branches = measure(
            state, t_model=0.0, cycle=1, sink=0, q=0.0, qcum=0.0,
            hazard=0.0, threshold=root_threshold, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp0)
        scalar.add(
            initial, branches, **output_extra(
                avalanche_id=0, event_number=0, avalanche_active=0,
                q_event=0.0, Q_avalanche=0.0, Q_cumulative=0.0,
                S_completed=0, frame_type="loading"))
        append_movie(
            movie, branches, initial, frame_type="loading", avalanche_id=0,
            event_number=0, avalanche_active=0, sink_state=0,
            q_event_over_b=0.0, Q_avalanche_over_b=0.0,
            Q_cumulative_over_b=0.0, S_completed=0, flush=True)
        scalar.flush()
        save_state(OUT / "checkpoints" / "initial_state.npz", state, t_model=0.0)

        try:
            for avalanche_id in range(1, AVALANCHES_REQUESTED + 1):
                wait_start_model = t_model
                status, state, t_model, root_row, branches, vp_cycle0, reasons = (
                    wait_for_root_or_censor(
                        state, cycle=avalanche_id, t_model=t_model,
                        threshold=root_threshold, total_completed=total_completed,
                        setup=setup, geom=geom, evaluator=evaluator,
                        scalar=scalar, movie=movie,
                        first_reload=avalanche_id > 1))
                if status == "censored":
                    censor = dict(
                        avalanche_id=avalanche_id, reasons=reasons,
                        t_model=t_model, t_s=t_model * SECONDS_PER_MODEL_TIME,
                        wait_time_model=t_model - wait_start_model,
                        H=root_row["H"], Hstar=root_threshold,
                        H_over_Hstar=root_row["H_over_threshold"],
                        sigma_local_Pa=root_row["sigma_local_Pa"],
                        Vp_over_Vp_cycle=root_row["Vp_over_Vp_cycle"],
                        r_n_m=root_row["r_n_m"])
                    outcome = "STOPPED_AT_EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                    break

                controller.start(
                    avalanche_id=avalanche_id, root_cycle=avalanche_id,
                    start_time_s=t_model * SECONDS_PER_MODEL_TIME)
                avalanche_start_model = t_model
                root_local = root_row["sigma_local_Pa"]
                root_integral = root_row["sigma_integral_Pa"]
                event_number = 0
                current_row = root_row
                current_branches = branches
                avalanche_blocker = None

                while controller.state.avalanche_active:
                    event_number += 1
                    controller.begin_transit()
                    event_start_model = t_model
                    pre_local = current_row["sigma_local_Pa"]
                    pre_integral = current_row["sigma_integral_Pa"]
                    try:
                        (state, t_model, restart, current_row, current_branches,
                         samples) = run_one_b_event(
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
                    descendant_trace.extend(controller_trace_rows(
                        trace, avalanche_id=avalanche_id,
                        event_number=event_number, phase="active_1b_frozen"))
                    controller.complete_transit(t_model * SECONDS_PER_MODEL_TIME)
                    total_completed += 1
                    completion_local = current_row["sigma_local_Pa"]
                    completion_integral = current_row["sigma_integral_Pa"]
                    subevents.append(dict(
                        avalanche_id=avalanche_id,
                        event_number=event_number,
                        event_type="root" if event_number == 1 else "descendant",
                        start_time_model=event_start_model, end_time_model=t_model,
                        start_time_s=event_start_model * SECONDS_PER_MODEL_TIME,
                        end_time_s=t_model * SECONDS_PER_MODEL_TIME,
                        duration_s=(t_model - event_start_model)
                        * SECONDS_PER_MODEL_TIME,
                        sigma_start_local_Pa=pre_local,
                        sigma_end_local_Pa=completion_local,
                        delta_sigma_local_Pa=pre_local - completion_local,
                        delta_sigma_integral_Pa=pre_integral - completion_integral,
                        descendants_committed_during_transit=0,
                        pending_children_after=controller.state.pending_children,
                        C4=restart["explicit_max_fourth_order_courant"]))
                    write_csv(OUT / "one_b_subevents.csv", subevents)
                    write_csv(OUT / "descendant_hazard_trace.csv", descendant_trace)
                    append_movie(
                        movie, current_branches, current_row,
                        frame_type="child_completion",
                        avalanche_id=avalanche_id,
                        event_number=event_number, avalanche_active=1,
                        sink_state=1, q_event_over_b=1.0,
                        Q_avalanche_over_b=controller.state.S_completed,
                        Q_cumulative_over_b=total_completed,
                        S_completed=controller.state.S_completed,
                        controller=controller, flush=True)
                    save_state(
                        OUT / "checkpoints"
                        / f"avalanche{avalanche_id}_event{event_number}_complete.npz",
                        state, t_model=t_model,
                        total_completed=total_completed,
                        controller_manifest_json=json.dumps(controller.manifest()))
                    save_controller_checkpoint(
                        controller, avalanche_id=avalanche_id,
                        event_number=event_number)
                    invalid = geometry_invalid(current_row)
                    if invalid:
                        avalanche_blocker = dict(
                            reason="geometry_validity_boundary_during_avalanche",
                            avalanche_id=avalanche_id,
                            event_number=event_number, reasons=invalid,
                            Vp_over_Vp_cycle=current_row["Vp_over_Vp_cycle"],
                            r_n_m=current_row["r_n_m"])
                        break
                    (window_status, state, t_model, current_row,
                     current_branches, window_reasons) = (
                        wait_for_descendant_or_extinction(
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
                            event_number=event_number,
                            reasons=window_reasons,
                            Vp_over_Vp_cycle=current_row["Vp_over_Vp_cycle"],
                            r_n_m=current_row["r_n_m"])
                        break
                    continued = window_status == "continued"
                    if not continued:
                        avalanches.append(dict(
                            avalanche_id=avalanche_id,
                            root_threshold=root_threshold,
                            start_time_model=avalanche_start_model,
                            end_time_model=t_model,
                            start_time_s=avalanche_start_model
                            * SECONDS_PER_MODEL_TIME,
                            end_time_s=t_model * SECONDS_PER_MODEL_TIME,
                            duration_s=(t_model - avalanche_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            wait_time_model=avalanche_start_model
                            - wait_start_model,
                            wait_time_s=(avalanche_start_model - wait_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            S=controller.state.S_completed,
                            Q_avalanche_nm=controller.state.S_completed
                            * B_EVENT_M * 1e9,
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
                        write_csv(OUT / "avalanche_summary.csv", avalanches)
                        save_state(
                            OUT / "checkpoints"
                            / f"avalanche{avalanche_id}_extinct.npz",
                            state, t_model=t_model,
                            total_completed=total_completed,
                            controller_manifest_json=json.dumps(
                                controller.manifest()))
                        print("AVALANCHE COMPLETE", json.dumps(avalanches[-1]),
                              flush=True)
                        break

                if avalanche_blocker is not None:
                    blocker = avalanche_blocker
                    outcome = ("STOPPED_AT_EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                               if blocker["reason"].startswith("geometry_validity")
                               else "STOPPED_ON_QUALIFIED_EVENT_BLOCKER")
                    break
                if len(avalanches) < avalanche_id:
                    blocker = dict(
                        reason="avalanche_did_not_reach_extinction",
                        avalanche_id=avalanche_id)
                    outcome = "STOPPED_ON_AVALANCHE_CONTROLLER_BLOCKER"
                    break
                if avalanche_id < AVALANCHES_REQUESTED:
                    root_threshold = float(root_rng.exponential())
                    seed_record["root_thresholds"].append(root_threshold)
                    (OUT / "rng_seed_record.json").write_text(
                        json.dumps(seed_record, indent=2) + "\n")
            else:
                outcome = "FIVE_COMPLETE_AVALANCHES"
        except Exception as error:
            blocker = dict(reason="unexpected_exception",
                           message=f"{type(error).__name__}: {error}")
            outcome = "STOPPED_ON_UNEXPECTED_BLOCKER"
        finally:
            scalar.flush()
            movie.flush()

    final_figure(movie_path, avalanches)
    movie_result = morphology_movie(movie_path)
    result = dict(
        outcome=outcome, avalanches_completed=len(avalanches),
        avalanches_requested=AVALANCHES_REQUESTED,
        total_one_b_events=total_completed, censor=censor, blocker=blocker,
        seed=seed, root_thresholds=seed_record["root_thresholds"],
        descendant_delta_G_step_eV=DELTA_G_STEP_EV,
        tau_corr_s=controller.correlation_time_s,
        C4_received_and_asserted=PRODUCTION_C4,
        movie=movie_result, wall_seconds=time.monotonic() - started,
        avalanches=avalanches)
    (OUT / "avalanche_renewal_result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
