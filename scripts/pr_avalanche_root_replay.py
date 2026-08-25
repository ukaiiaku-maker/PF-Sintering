#!/usr/bin/env python3
"""Replay frozen production root states through the D2 avalanche controller.

The qualified root barrier and one-b PF event are imported unchanged.  This
driver adds only the external facilitated descendant clock and never repeats a
sink-OFF stochastic wait.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["PR_RENEWAL_CASE"] = "fourier"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pf_sintering.pr_avalanche import (  # noqa: E402
    AvalancheController,
    DescendantBarrier,
    descendant_rate_per_s,
)


PRODUCTION = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "production_ensemble_10x5_courant_verified")
DEFAULT_OUT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_v1_root_replay")
SECONDS_PER_MODEL_TIME = 0.01557994316955921
TEMPERATURE_K = 1830.15
B_EVENT_M = 0.25e-9
PRODUCTION_C4 = 0.05
PRODUCTION_COMMIT = "d2aafbf7df99728acb67af941f2c078126455e1f"
BARRIER_EXPORT = Path(
    "/Volumes/Data/Data/INRL_lambert_onsager/Forward_Model/data/zro2/"
    "bicrystal_creep_barrier_export.json")
MAX_TRANSITS_DEFAULT = 50
TARGET_REDUCED_MEDIAN_S = 5.0
REDUCED_CALIBRATION_REPLICATES_PER_ROOT = 256
REDUCED_CALIBRATION_CAP = 100
REDUCED_CALIBRATION_SEED = 20260824


def load_pf_runtime() -> None:
    """Load the expensive PF/Numba call graph only for actual replays."""
    global authoritative, base, integral, build_case, make_transport
    global make_evaluator, set_num_threads
    from numba import set_num_threads
    import pr_authoritative_stochastic_ten_event as authoritative
    import pr_coarsening_stochastic_two_event as base
    from pr_coarsening_driven_fourier_loading import integral
    from pr_experimental_long_sinkoff import build_case
    from pr_full_deterministic_cycle import make_transport
    from pr_tj_node_coupling_gate import make_evaluator


def numeric_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_authoritative_barrier():
    exported = json.loads(BARRIER_EXPORT.read_text())
    item = next(
        value for value in exported["barrier_model"]["temperature_slices"]
        if math.isclose(value["T_K"], TEMPERATURE_K, abs_tol=1e-10))
    from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
    root = CompleteExpFloorParams(
        G0_eV=item["G0_eV"], G_floor_eV=item["Gfloor_eV"],
        a=item["a"], sigma_hat_pa=item["sigmahat_Pa"], n=item["n"])
    return exported, item, root


def frozen_event_duration_median_s() -> float:
    values = [row["t_event_s"] for row in numeric_csv(
        PRODUCTION / "ensemble_events.csv")]
    if len(values) != 20 or not all(value > 0.0 for value in values):
        raise RuntimeError("the verified 20-event lifetime archive is incomplete")
    return float(np.median(values))


def discover_roots() -> list[dict]:
    events = numeric_csv(PRODUCTION / "ensemble_events.csv")
    roots = []
    for event in events:
        realization = int(event["realization"])
        cycle = int(event["event_number"])
        checkpoint = (PRODUCTION / f"realization_{realization:02d}" /
                      "checkpoints" / f"cycle{cycle}_before_nucleation.npz")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        result = json.loads((PRODUCTION / f"realization_{realization:02d}" /
                             "result.json").read_text())
        roots.append(dict(
            case_id=f"R{realization:02d}_E{cycle:02d}",
            realization=realization, root_cycle=cycle,
            root_sigma_saved_Pa=event["sigma_nuc_Pa"],
            root_integral_saved_Pa=event["sigma_nuc_integral_Pa"],
            checkpoint=str(checkpoint), production_seed=int(result["seed"])))
    roots.sort(key=lambda row: row["root_sigma_saved_Pa"])
    if len(roots) != 20:
        raise RuntimeError(f"expected 20 saved roots, found {len(roots)}")
    return roots


def archived_event_profiles() -> list[dict]:
    profiles = []
    for realization in range(1, 11):
        path = PRODUCTION / f"realization_{realization:02d}" / "history.csv"
        rows = numeric_csv(path)
        for cycle in sorted({int(row["cycle"]) for row in rows}):
            active = [row for row in rows
                      if int(row["cycle"]) == cycle
                      and int(row["sink_state"]) == 1]
            if not active:
                continue
            before = max(
                (row for row in rows if int(row["cycle"]) == cycle
                 and int(row["sink_state"]) == 0
                 and row["t_model"] <= active[0]["t_model"]),
                key=lambda row: row["t_model"])
            profiles.append(dict(
                realization=realization, root_cycle=cycle,
                samples=[before] + sorted(
                    active, key=lambda row: row["t_model"])))
    if len(profiles) != 20:
        raise RuntimeError("verified event profiles are incomplete")
    return profiles


def expected_offspring_for_profile(
        profile: list[dict], barrier: DescendantBarrier, nu0: float,
        tau_corr_s: float) -> dict:
    def rate(row):
        return descendant_rate_per_s(
            row["sigma_local_Pa"], row["r_n_m"], barrier=barrier,
            temperature_K=TEMPERATURE_K, attempt_frequency_per_s=nu0,
            b_m=B_EVENT_M)
    rates = [rate(row) for row in profile]
    transit_hazard = 0.0
    for left, right, gl, gr in zip(profile, profile[1:], rates, rates[1:]):
        dt_s = (right["t_model"] - left["t_model"]) * SECONDS_PER_MODEL_TIME
        transit_hazard += 0.5 * (gl + gr) * dt_s
    correlation_hazard = rates[-1] * tau_corr_s
    # Every transit admits Poisson descendants.  The correlation window is
    # entered only if transit hazard committed none, and can rescue at most
    # the first child before the next transit begins.
    expected = (
        transit_hazard
        + math.exp(-transit_hazard) * (1.0 - math.exp(-correlation_hazard)))
    return dict(
        transit_hazard=transit_hazard,
        correlation_hazard=correlation_hazard,
        expected_offspring=expected)


def reduced_avalanche_size(
        transit_hazard: float, correlation_hazard: float, *, rng,
        cap: int = REDUCED_CALIBRATION_CAP) -> int:
    """Queue replay used only to choose the single descendant barrier scale.

    It repeats one archived finite-transit hazard profile, applies the same
    exponential residual logic as the full controller, and right-censors at a
    declared diagnostic cap.  Common random numbers make the scalar bisection
    deterministic and independent of the later production-root descendant
    seeds.
    """
    threshold = max(float(rng.exponential()), np.finfo(float).tiny)
    hazard = 0.0
    pending = 0
    completed = 0
    while completed < cap:
        remaining = float(transit_hazard)
        while hazard + remaining >= threshold:
            remaining -= threshold - hazard
            hazard = 0.0
            pending += 1
            threshold = max(float(rng.exponential()), np.finfo(float).tiny)
        hazard += remaining
        completed += 1
        if pending:
            pending -= 1
            continue
        if hazard + correlation_hazard >= threshold:
            # The next transit starts at the first correlation-window crossing;
            # unused window time is not consumed.
            hazard = 0.0
            threshold = max(float(rng.exponential()), np.finfo(float).tiny)
            continue
        break
    return completed


def barrier_preflight(out: Path) -> dict:
    exported, item, root = load_authoritative_barrier()
    nu0 = float(exported["constants"]["nu0_sinv"])
    tau_corr = frozen_event_duration_median_s()
    profiles = archived_event_profiles()

    calibration_seeds = np.random.default_rng(
        REDUCED_CALIBRATION_SEED).integers(
            0, np.iinfo(np.uint64).max,
            size=(len(profiles), REDUCED_CALIBRATION_REPLICATES_PER_ROOT),
            dtype=np.uint64)

    def evaluate(g0_step):
        law = DescendantBarrier(root=root, G0_step_eV=g0_step)
        rows = [expected_offspring_for_profile(p["samples"], law, nu0, tau_corr)
                for p in profiles]
        sizes = []
        for index, row in enumerate(rows):
            for seed in calibration_seeds[index]:
                sizes.append(reduced_avalanche_size(
                    row["transit_hazard"], row["correlation_hazard"],
                    rng=np.random.default_rng(int(seed))))
        return float(np.median(sizes)), np.asarray(sizes), rows

    lo, hi = 0.25, root.G0_eV
    # Expected offspring decreases monotonically with G0_step.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value, _, _ = evaluate(mid)
        if value > TARGET_REDUCED_MEDIAN_S:
            lo = mid
        else:
            hi = mid
    selected = 0.5 * (lo + hi)
    reduced_median, reduced_sizes, profile_rows = evaluate(selected)
    rows = []
    roots = discover_roots()
    root_by_key = {
        (root_record["realization"], root_record["root_cycle"]): root_record
        for root_record in roots}
    for profile, values in zip(profiles, profile_rows):
        root_record = root_by_key[
            (profile["realization"], profile["root_cycle"])]
        rows.append(dict(**root_record, **values))
    rows.sort(key=lambda row: row["root_sigma_saved_Pa"])
    write_csv(out / "barrier_preflight_profiles.csv", rows)
    audit = dict(
        production_commit=PRODUCTION_COMMIT,
        root_barrier_slice=item,
        root_barrier_export=str(BARRIER_EXPORT),
        root_barrier_export_sha256=hashlib.sha256(
            BARRIER_EXPORT.read_bytes()).hexdigest(),
        descendant_model="D2 absolute step envelope; no elastic facilitation",
        G0_step_eV=selected,
        G0_effective_eV=min(root.G0_eV, selected),
        scalar_calibration_target="median reduced avalanche size",
        target_reduced_median_S=TARGET_REDUCED_MEDIAN_S,
        achieved_reduced_median_S=reduced_median,
        reduced_mean_S=float(np.mean(reduced_sizes)),
        reduced_S95=float(np.quantile(reduced_sizes, .95)),
        reduced_right_censored_fraction=float(np.mean(
            reduced_sizes >= REDUCED_CALIBRATION_CAP)),
        reduced_calibration_replicates_per_root=(
            REDUCED_CALIBRATION_REPLICATES_PER_ROOT),
        reduced_calibration_cap=REDUCED_CALIBRATION_CAP,
        reduced_calibration_seed=REDUCED_CALIBRATION_SEED,
        reduced_calibration_seed_independent_of_root_replay_seeds=True,
        tau_corr_s=tau_corr,
        tau_corr_rule="median physical duration of 20 verified production 1b events",
        event_profiles=20,
        stress_range_MPa=[
            min(item["root_sigma_saved_Pa"] for item in roots) / 1e6,
            max(item["root_sigma_saved_Pa"] for item in roots) / 1e6],
        physical_seconds_per_model_time=SECONDS_PER_MODEL_TIME,
        b_event_m=B_EVENT_M,
        N_sites="2*pi*r_TJ/b_event")
    (out / "barrier_preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


class ReplayOutput:
    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self.contours: list[dict] = []
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    def add(self, state, row, branches):
        value = dict(row)
        value["sample_id"] = len(self.rows)
        self.rows.append(value)
        self.contours.append(dict(
            sample_id=value["sample_id"],
            z_negative=np.asarray(branches["negative"]["z_m"]),
            r_negative=np.asarray(branches["negative"]["r_m"]),
            z_positive=np.asarray(branches["positive"]["z_m"]),
            r_positive=np.asarray(branches["positive"]["r_m"])))

    def flush(self):
        write_csv(self.out / "pf_event_history.csv", self.rows)
        arrays = {}
        for row in self.contours:
            sid = row["sample_id"]
            for key in ("z_negative", "r_negative", "z_positive", "r_positive"):
                arrays[f"sample_{sid:04d}_{key}"] = row[key]
        if arrays:
            np.savez_compressed(self.out / "pf_event_contours.npz", **arrays)


def derived_descendant_seed(production_seed: int, root_cycle: int) -> int:
    payload = f"sintering-avalanche-v1|{production_seed}|{root_cycle}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def trace_sample(row: dict, *, t_s: float, q_event: float,
                 q_avalanche: float, sink: int, event_number: int) -> dict:
    return dict(
        t_model=float(row["t_model"]), t_s=float(t_s),
        sigma_local_Pa=float(row["sigma_local_Pa"]),
        sigma_integral_Pa=float(row["sigma_integral_Pa"]),
        r_TJ_m=float(row["r_n_m"]),
        N_sites=2.0 * math.pi * float(row["r_n_m"]) / B_EVENT_M,
        q_event_over_b=float(q_event),
        q_avalanche_over_b=float(q_avalanche),
        sink_state=int(sink), event_number=int(event_number))


def run_root(root_record: dict, settings: dict) -> dict:
    load_pf_runtime()
    out = Path(settings["out"]) / root_record["case_id"]
    out.mkdir(parents=True, exist_ok=True)
    set_num_threads(2)
    authoritative.configure_barrier()
    base.OUT = out
    base.MIN_SOLVABLE_VOLUME_RATIO = -math.inf
    assert PRODUCTION_C4 == 0.05

    exported, item, root_params = load_authoritative_barrier()
    tau_corr_s = float(settings["tau_corr_s"])
    g0_step = float(settings["G0_step_eV"])
    seed = derived_descendant_seed(
        root_record["production_seed"], root_record["root_cycle"])
    rng = np.random.default_rng(seed)
    controller = AvalancheController(
        barrier=DescendantBarrier(root_params, g0_step),
        temperature_K=TEMPERATURE_K,
        attempt_frequency_per_s=float(exported["constants"]["nu0_sinv"]),
        b_m=B_EVENT_M, correlation_time_s=tau_corr_s, rng=rng)

    geom, setup = build_case()
    evaluator = make_evaluator(setup, geom)
    transport = make_transport(geom)
    assert transport.b_m == B_EVENT_M
    checkpoint = np.load(root_record["checkpoint"])
    state = tuple(np.asarray(checkpoint[key]).copy()
                  for key in ("f", "particle", "substrate"))
    t_model = float(checkpoint["t_model"])
    root_H = float(checkpoint["H"])
    root_Hstar = float(checkpoint["Hstar"])
    checkpoint.close()
    vp0 = integral(state[1], setup)
    root_row, root_branches = base.measure(
        state, t_model=t_model, cycle=root_record["root_cycle"], sink=0,
        q=0.0, qcum=0.0, hazard=root_H, threshold=root_Hstar,
        clock_scale=float(settings["clock_scale"]), setup=setup, geom=geom,
        evaluator=evaluator, vp_cycle0=vp0)
    stress_difference = abs(
        root_row["sigma_local_Pa"] - root_record["root_sigma_saved_Pa"])
    if stress_difference > 1e-5 * max(root_record["root_sigma_saved_Pa"], 1.0):
        raise RuntimeError(
            f"reconstructed root stress mismatch: {stress_difference} Pa")

    output = ReplayOutput(out)
    controller.start(
        avalanche_id=1, root_cycle=root_record["root_cycle"],
        start_time_s=t_model * SECONDS_PER_MODEL_TIME)
    events = []
    trace = []
    blocker = None
    termination = None
    started = time.monotonic()
    current_row = root_row
    current_branches = root_branches

    while controller.state.avalanche_active:
        if controller.state.S_completed >= int(settings["max_transits"]):
            termination = "max_transits_right_censored"
            break
        if controller.state.S_completed > 0:
            if not controller.launch_pending_child():
                raise RuntimeError("child launch requested with an empty queue")
        event_number = controller.state.S_completed + 1
        controller.begin_transit()
        start_model = t_model
        start_s = start_model * SECONDS_PER_MODEL_TIME
        start_sample = trace_sample(
            current_row, t_s=start_s, q_event=0.0,
            q_avalanche=float(controller.state.S_completed), sink=1,
            event_number=event_number)
        row_start = len(output.rows)
        pre_local = current_row["sigma_local_Pa"]
        pre_integral = current_row["sigma_integral_Pa"]
        nucleation = dict(current_row)
        nucleation["H"] = controller.state.descendant_hazard
        try:
            state, t_model, restart = base.run_event(
                state, cycle=event_number,
                completed=controller.state.S_completed,
                t_model=t_model, nucleation=nucleation,
                threshold=controller.state.descendant_threshold,
                clock_scale=float(settings["clock_scale"]),
                setup=setup, geom=geom, evaluator=evaluator,
                transport=transport, output=output,
                explicit_max_fourth_order_courant=PRODUCTION_C4)
        except Exception as exc:
            blocker = dict(
                reason="qualified_1b_event_failed", event_number=event_number,
                message=repr(exc), S_completed=controller.state.S_completed)
            termination = "event_operator_blocker"
            break
        actual_c4 = restart.get("explicit_max_fourth_order_courant")
        assert actual_c4 == PRODUCTION_C4
        event_rows = output.rows[row_start:]
        if [round(row["q_over_b"], 10) for row in event_rows] != [
                0.1, 0.25, 0.5, 0.75, 1.0]:
            raise RuntimeError("qualified event checkpoint sequence changed")
        samples = [start_sample]
        completed_before = controller.state.S_completed
        for row in event_rows:
            samples.append(trace_sample(
                row, t_s=row["t_model"] * SECONDS_PER_MODEL_TIME,
                q_event=row["q_over_b"],
                q_avalanche=completed_before + row["q_over_b"],
                sink=1, event_number=event_number))
        event_trace = controller.integrate_transit_samples(samples)
        trace.extend(event_trace)
        controller.complete_transit(t_model * SECONDS_PER_MODEL_TIME)
        current_row = event_rows[-1]
        final_local = current_row["sigma_local_Pa"]
        final_integral = current_row["sigma_integral_Pa"]
        events.append(dict(
            case_id=root_record["case_id"], event_number=event_number,
            event_type="root" if event_number == 1 else "descendant",
            start_time_s=start_s,
            end_time_s=t_model * SECONDS_PER_MODEL_TIME,
            duration_s=(t_model - start_model) * SECONDS_PER_MODEL_TIME,
            sigma_start_Pa=pre_local, sigma_end_Pa=final_local,
            delta_sigma_local_Pa=pre_local-final_local,
            delta_sigma_integral_Pa=pre_integral-final_integral,
            descendants_committed_during_transit=sum(
                item["crossings_in_segment"] for item in event_trace),
            pending_children_after=controller.state.pending_children,
            C4=actual_c4))
        write_csv(out / "subevents.csv", events)

        if controller.state.pending_children > 0:
            continue

        corr_start_s = t_model * SECONDS_PER_MODEL_TIME
        corr = controller.correlation_window(
            sigma_local_Pa=current_row["sigma_local_Pa"],
            r_TJ_m=current_row["r_n_m"], start_time_s=corr_start_s)
        t_model += float(corr["elapsed_s"]) / SECONDS_PER_MODEL_TIME
        corr_row, current_branches = base.measure(
            state, t_model=t_model, cycle=root_record["root_cycle"], sink=0,
            q=0.0, qcum=controller.state.S_completed,
            hazard=controller.state.descendant_hazard,
            threshold=controller.state.descendant_threshold,
            clock_scale=float(settings["clock_scale"]), setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=integral(state[1], setup))
        current_row = corr_row
        rate = controller.rate(corr_row["sigma_local_Pa"], corr_row["r_n_m"])
        corr_sample = trace_sample(
            corr_row, t_s=t_model * SECONDS_PER_MODEL_TIME, q_event=0.0,
            q_avalanche=controller.state.S_completed, sink=0,
            event_number=event_number)
        corr_trace = controller._trace_row(corr_sample, rate, int(corr["continued"]))
        corr_trace["correlation_window"] = 1
        trace.append(corr_trace)
        if not corr["continued"]:
            termination = "correlation_window_extinction"
            break

    output.flush()
    write_csv(out / "avalanche_time_trace.csv", trace)
    write_csv(out / "descendant_crossings.csv", controller.crossings)
    result = dict(
        **root_record,
        descendant_seed=seed,
        G0_step_eV=g0_step,
        G0_effective_eV=controller.barrier.G0_effective_eV,
        tau_corr_s=tau_corr_s,
        S=controller.state.S_completed,
        Q_avalanche_over_b=controller.state.q_avalanche_over_b,
        Q_avalanche_nm=controller.state.q_avalanche_over_b * B_EVENT_M * 1e9,
        duration_s=(t_model * SECONDS_PER_MODEL_TIME
                    - controller.state.avalanche_start_time_s),
        root_stress_Pa=root_row["sigma_local_Pa"],
        final_stress_Pa=current_row["sigma_local_Pa"],
        total_local_stress_drop_Pa=(root_row["sigma_local_Pa"]
                                    - current_row["sigma_local_Pa"]),
        final_integral_stress_Pa=current_row["sigma_integral_Pa"],
        termination=termination,
        right_censored=(termination == "max_transits_right_censored"),
        blocker=blocker,
        thresholds_drawn=controller.state.thresholds_drawn,
        crossings=len(controller.crossings),
        C4_received_and_asserted=PRODUCTION_C4,
        wall_seconds=time.monotonic()-started,
        subevents=events,
        controller_manifest=controller.manifest())
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def representative_figure(result: dict, out: Path) -> None:
    rows = numeric_csv(out / result["case_id"] / "avalanche_time_trace.csv")
    t0 = rows[0]["t_s"]
    t_us = np.asarray([(row["t_s"] - t0) * 1e6 for row in rows])
    fig, axes = plt.subplots(
        6, 1, figsize=(8.0, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.35, .72, .72, 1.2, 1.1]},
        constrained_layout=True)
    axes[0].plot(t_us, [row["sigma_local_Pa"]/1e6 for row in rows],
                 color="#174a73", label="local")
    axes[0].plot(t_us, [row["sigma_integral_Pa"]/1e6 for row in rows],
                 color="#9a6b28", linestyle="--", label="integral")
    axes[0].set_ylabel("Stress (MPa)"); axes[0].legend(frameon=False)
    axes[1].step(t_us, [row["q_avalanche_over_b"] for row in rows],
                 where="post", color="#2f6f58", label=r"$Q/b$")
    axes[1].plot(t_us, [row["q_event_over_b"] for row in rows],
                 color="#67507e", marker=".", label=r"$q_j/b$")
    axes[1].set_ylabel("Progress"); axes[1].legend(frameon=False, ncol=2)
    axes[2].step(t_us, [row["sink_state"] for row in rows], where="post",
                 color="#30343b"); axes[2].set_ylabel("Sink"); axes[2].set_yticks([0,1],["OFF","ON"])
    axes[3].step(t_us, [row["avalanche_active"] for row in rows], where="post",
                 color="#8b4f43"); axes[3].set_ylabel("Avalanche"); axes[3].set_yticks([0,1],["OFF","ON"])
    axes[4].plot(t_us, [row["H_over_threshold"] for row in rows],
                 color="#4d6d72", marker=".")
    axes[4].axhline(1.0, color="#555555", linestyle=":", linewidth=1)
    axes[4].set_ylabel(r"$H/H^*$")
    axes[5].plot(t_us, [row["G_desc_star_eV"] for row in rows],
                 color="#7a4d27")
    axes[5].set_ylabel(r"$G^*_{desc}$ (eV)"); axes[5].set_xlabel("Physical time from root nucleation (µs)")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(
        f"{result['case_id']} D2 avalanche: S={result['S']}, "
        f"G0_step={result['G0_step_eV']:.4f} eV")
    fig.savefig(out / "representative_avalanche.png", dpi=240)
    fig.savefig(out / "representative_avalanche.pdf")
    plt.close(fig)


def aggregate(results: list[dict], out: Path, settings: dict) -> dict:
    write_csv(out / "avalanche_results.csv", [
        {key: value for key, value in row.items()
         if key not in ("subevents", "controller_manifest", "thresholds_drawn")}
        for row in results])
    complete = [row for row in results if not row["right_censored"] and not row["blocker"]]
    sizes = np.asarray([row["S"] for row in complete], dtype=float)
    observed_sizes = np.asarray([row["S"] for row in results], dtype=float)
    observations = []
    for row in results:
        if row["blocker"] is not None:
            status = "solver_blocker"
        elif row["right_censored"]:
            status = "right_censored"
        else:
            status = "exact_extinction"
        observations.append(dict(
            case_id=row["case_id"], S_observed=int(row["S"]), status=status,
            is_lower_bound=bool(row["right_censored"]),
            blocker=row["blocker"]))
    write_csv(out / "avalanche_size_observations.csv", observations)
    counts = []
    if len(sizes):
        for size in sorted(set(int(value) for value in sizes)):
            counts.append(dict(S=size, count=int(np.sum(sizes == size)),
                               probability_exact_complete_only=float(np.mean(sizes == size)),
                               denominator_complete_only=len(sizes)))
    write_csv(out / "avalanche_size_distribution.csv", counts)

    # Discrete Kaplan-Meier extinction curve.  A size-cap termination is a
    # right-censored lower bound, not a finite avalanche at the cap.
    km_rows = []
    survival = 1.0
    km_median = math.nan
    for size in sorted(set(int(row["S"]) for row in results)):
        at_risk = sum(int(row["S"]) >= size for row in results)
        extinctions = sum(
            int(row["S"]) == size and not row["right_censored"]
            and row["blocker"] is None for row in results)
        censored = sum(
            int(row["S"]) == size and row["right_censored"]
            for row in results)
        survival_before = survival
        if at_risk:
            survival *= 1.0 - extinctions / at_risk
        extinction_probability = survival_before - survival
        km_rows.append(dict(
            S=size, at_risk=at_risk, extinctions=extinctions,
            right_censored=censored, survival_before=survival_before,
            extinction_probability=extinction_probability,
            survival_after=survival))
        if math.isnan(km_median) and survival <= 0.5 + 1e-15:
            km_median = float(size)
    write_csv(out / "avalanche_size_kaplan_meier.csv", km_rows)

    summary = dict(
        settings=settings,
        N_roots=len(results), N_complete=len(complete),
        N_right_censored=sum(row["right_censored"] for row in results),
        N_blockers=sum(row["blocker"] is not None for row in results),
        size_statistic_scope="exact self-terminated avalanches only",
        mean_S=float(np.mean(sizes)) if len(sizes) else math.nan,
        median_S=float(np.median(sizes)) if len(sizes) else math.nan,
        S95=float(np.quantile(sizes, .95)) if len(sizes) else math.nan,
        mean_S_complete=float(np.mean(sizes)) if len(sizes) else math.nan,
        median_S_complete=float(np.median(sizes)) if len(sizes) else math.nan,
        S95_complete=float(np.quantile(sizes, .95)) if len(sizes) else math.nan,
        observed_lower_bound_mean_S=(float(np.mean(observed_sizes))
                                     if len(observed_sizes) else math.nan),
        observed_lower_bound_median_S=(float(np.median(observed_sizes))
                                       if len(observed_sizes) else math.nan),
        observed_lower_bound_S95=(float(np.quantile(observed_sizes, .95))
                                  if len(observed_sizes) else math.nan),
        kaplan_meier_median_S=km_median,
        kaplan_meier_survival_at_largest_S=(survival if km_rows else math.nan),
        results=results)
    (out / "avalanche_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if complete:
        representative = min(complete, key=lambda row: (
            abs(row["S"] - summary["median_S"]),
            abs(row["root_stress_Pa"] - 80e6)))
        representative_figure(representative, out)
        summary["representative_case"] = representative["case_id"]
        (out / "avalanche_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    censored_rows = [row for row in results if row["right_censored"]]
    if len(sizes):
        axes[0,0].hist(sizes, bins=np.arange(sizes.min()-.5, sizes.max()+1.5), color="#174a73")
        axes[0,0].set(xlabel="Avalanche size S", ylabel="Count")
        axes[0,1].scatter([row["root_stress_Pa"]/1e6 for row in complete], sizes, color="#2f6f58")
        axes[0,1].set(xlabel="Root stress (MPa)", ylabel="S")
        axes[1,0].scatter(sizes, [row["total_local_stress_drop_Pa"]/1e6 for row in complete], color="#67507e")
        axes[1,0].set(xlabel="S", ylabel="Local stress drop (MPa)")
        axes[1,1].scatter(sizes, [row["duration_s"]*1e6 for row in complete], color="#9a6b28")
        axes[1,1].set(xlabel="S", ylabel="Avalanche duration (µs)")
    if censored_rows:
        censored_sizes = np.asarray([row["S"] for row in censored_rows])
        axes[0,0].scatter(censored_sizes, np.zeros_like(censored_sizes),
                          marker="^", facecolors="none", edgecolors="#a33d34",
                          label="right-censored lower bound")
        axes[0,0].legend(fontsize=8)
        axes[0,1].scatter(
            [row["root_stress_Pa"]/1e6 for row in censored_rows], censored_sizes,
            marker="^", facecolors="none", edgecolors="#a33d34")
        axes[1,0].scatter(
            censored_sizes,
            [row["total_local_stress_drop_Pa"]/1e6 for row in censored_rows],
            marker="^", facecolors="none", edgecolors="#a33d34")
        axes[1,1].scatter(
            censored_sizes, [row["duration_s"]*1e6 for row in censored_rows],
            marker="^", facecolors="none", edgecolors="#a33d34")
    for ax in axes.ravel(): ax.grid(alpha=.2)
    fig.savefig(out / "avalanche_calibration_summary.png", dpi=220)
    fig.savefig(out / "avalanche_calibration_summary.pdf")
    plt.close(fig)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--g0-step-eV", type=float)
    parser.add_argument("--preflight-source", type=Path,
                        help="reuse a completed scalar-calibration JSON without recalibrating")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="rebuild statistics/figures from existing result.json files")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--root-limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-transits", type=int, default=MAX_TRANSITS_DEFAULT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        settings = json.loads((args.out / "replay_manifest.json").read_text())
        result_paths = sorted(args.out.glob("R*/result.json"))
        if not result_paths:
            raise RuntimeError("no root replay result.json files found")
        results = [json.loads(path.read_text()) for path in result_paths]
        results.sort(key=lambda row: row["root_sigma_saved_Pa"])
        previous_summary_path = args.out / "avalanche_summary.json"
        previous = (json.loads(previous_summary_path.read_text())
                    if previous_summary_path.exists() else {})
        summary = aggregate(results, args.out, settings)
        if "campaign_wall_seconds" in previous:
            summary["campaign_wall_seconds"] = previous["campaign_wall_seconds"]
            previous_summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: summary.get(key) for key in (
            "N_roots", "N_complete", "N_right_censored", "N_blockers",
            "mean_S_complete", "median_S_complete", "S95_complete",
            "observed_lower_bound_median_S", "kaplan_meier_median_S")}, indent=2))
        return
    if args.preflight_source is not None:
        preflight = json.loads(args.preflight_source.read_text())
        (args.out / "barrier_preflight.json").write_text(
            json.dumps(preflight, indent=2) + "\n")
    else:
        preflight = barrier_preflight(args.out)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    g0_step = (preflight["G0_step_eV"] if args.g0_step_eV is None
               else float(args.g0_step_eV))
    roots = discover_roots()
    if args.case_id:
        requested = set(args.case_id)
        roots = [root for root in roots if root["case_id"] in requested]
        missing = requested - {root["case_id"] for root in roots}
        if missing:
            raise ValueError(f"unknown root case(s): {sorted(missing)}")
    roots = roots[:args.root_limit]
    settings = dict(
        production_commit=PRODUCTION_COMMIT,
        production_root=str(PRODUCTION),
        out=str(args.out), G0_step_eV=g0_step,
        tau_corr_s=preflight["tau_corr_s"],
        clock_scale=15579943169.55921,
        seconds_per_model_time=SECONDS_PER_MODEL_TIME,
        C4=PRODUCTION_C4, b_event_m=B_EVENT_M,
        max_transits=args.max_transits,
        roots=[root["case_id"] for root in roots],
        descendant_seed_rule="SHA256(avalanche-v1|production_seed|root_cycle)",
        no_elastic_facilitation=True,
        root_event_solver_unchanged=True)
    (args.out / "replay_manifest.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.monotonic()
    if args.workers == 1:
        results = [run_root(root, settings) for root in roots]
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers) as pool:
            results = list(pool.map(
                run_root, roots, [settings] * len(roots)))
    results.sort(key=lambda row: row["root_sigma_saved_Pa"])
    summary = aggregate(results, args.out, settings)
    summary["campaign_wall_seconds"] = time.monotonic() - started
    (args.out / "avalanche_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "N_roots", "N_complete", "N_right_censored", "N_blockers",
        "mean_S", "median_S", "S95", "campaign_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
