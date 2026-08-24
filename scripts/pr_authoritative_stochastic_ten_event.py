"""First unbiased ten-event stochastic Fourier production trajectory."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads

import pr_coarsening_stochastic_two_event as base
from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
from pr_coarsening_driven_fourier_loading import integral
from pr_experimental_long_sinkoff import build_case
from pr_full_deterministic_cycle import make_transport
from pr_tj_node_coupling_gate import make_evaluator


ROOT = Path(__file__).resolve().parents[1]
PARENT = (ROOT / "runs/pr_current_head_regression/coarsening_driven_fourier")
OUT = PARENT / "authoritative_stochastic_ten_event"
EXPORT = Path(
    "/Volumes/Data/Data/INRL_lambert_onsager/Forward_Model/data/zro2/"
    "bicrystal_creep_barrier_export.json")
TEMPERATURE_K = 1830.15
TARGET_CLOCK_TIME = 1.5
EVENTS_REQUESTED = 10


def numeric_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        result = []
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            result.append(row)
        return result


def configure_barrier() -> tuple[dict, dict]:
    exported = json.loads(EXPORT.read_text())
    slices = exported["barrier_model"]["temperature_slices"]
    matches = [item for item in slices
               if math.isclose(item["T_K"], TEMPERATURE_K, abs_tol=1e-10)]
    if len(matches) != 1:
        raise RuntimeError("exact 1830.15 K barrier slice is unavailable")
    item = matches[0]
    base.OUT = OUT
    base.TEMPERATURE_K = TEMPERATURE_K
    base.MIN_SOLVABLE_VOLUME_RATIO = -math.inf
    base.BARRIER = CompleteExpFloorParams(
        G0_eV=item["G0_eV"], G_floor_eV=item["Gfloor_eV"],
        a=item["a"], sigma_hat_pa=item["sigmahat_Pa"], n=item["n"])
    return exported, item


def deterministic_replay(exported: dict, item: dict) -> tuple[float, dict]:
    rows = numeric_rows(PARENT / "coarsening_history.csv")
    nu0 = exported["constants"]["nu0_sinv"]
    replay = []
    for row in rows:
        barrier, total_unit_attempt = base.rate(
            row["sigma_local_Pa"], row["r_n_m"], 1.0)
        sites = 2.0 * math.pi * row["r_n_m"] / base.B_M
        per_site_s = nu0 * math.exp(
            -barrier * base.EV / (1.380649e-23 * TEMPERATURE_K))
        replay.append(dict(
            t_model=row["t_model"], sigma_local_Pa=row["sigma_local_Pa"],
            G_star_eV=barrier, Gamma_site_per_s=per_site_s,
            N_sites=sites, Gamma_total_per_s=sites * per_site_s,
            unit_attempt_total_per_model_time=total_unit_attempt))

    target_indices = [i for i, row in enumerate(replay)
                      if row["t_model"] <= TARGET_CLOCK_TIME + 1e-12]
    hazard_physical_per_model_second = 0.0
    for i in target_indices[1:]:
        left, right = replay[i - 1], replay[i]
        dt = right["t_model"] - left["t_model"]
        hazard_physical_per_model_second += 0.5 * dt * (
            left["Gamma_total_per_s"] + right["Gamma_total_per_s"])
    seconds_per_model_time = 1.0 / hazard_physical_per_model_second
    clock_scale = nu0 * seconds_per_model_time

    with (OUT / "exact_slice_deterministic_replay.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay[0]))
        writer.writeheader(); writer.writerows(replay)
    t = np.asarray([row["t_model"] for row in replay])
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True,
                             constrained_layout=True)
    for ax, key, label, log in zip(
            axes,
            ("G_star_eV", "Gamma_site_per_s", "N_sites", "Gamma_total_per_s"),
            ("G* (eV)", "Gamma_site (s^-1)", "N_sites", "Gamma_total (s^-1)"),
            (False, True, False, True)):
        values = [row[key] for row in replay]
        (ax.semilogy if log else ax.plot)(t, values)
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[-1].set_xlabel("model time")
    fig.suptitle("Exact exported EXP-floor slice, T=1830.15 K")
    fig.savefig(OUT / "exact_slice_deterministic_replay.png", dpi=180)
    plt.close(fig)
    first, last = replay[0], replay[-1]
    audit = dict(
        status="PASS", temperature_K=TEMPERATURE_K, exact_slice=item,
        interpolation_used=False, extrapolation_used=False,
        nu0_per_s=nu0, b_site_count_m=base.B_M,
        clock_mapping=dict(
            rule="expected cumulative hazard equals 1 at deterministic t=1.5",
            target_time_model=TARGET_CLOCK_TIME,
            physical_hazard_integral_per_model_second=hazard_physical_per_model_second,
            seconds_per_model_time=seconds_per_model_time,
            fixed_clock_scale_per_model_time=clock_scale),
        endpoint_ratios=dict(
            stress=last["sigma_local_Pa"] / first["sigma_local_Pa"],
            G_star=last["G_star_eV"] / first["G_star_eV"],
            Gamma_site=last["Gamma_site_per_s"] / first["Gamma_site_per_s"],
            N_sites=last["N_sites"] / first["N_sites"],
            Gamma_total=last["Gamma_total_per_s"] / first["Gamma_total_per_s"]),
        guardrails=dict(
            barrier_decreases=last["G_star_eV"] < first["G_star_eV"],
            per_site_rate_increases=last["Gamma_site_per_s"] > first["Gamma_site_per_s"],
            total_rate_increases=last["Gamma_total_per_s"] > first["Gamma_total_per_s"]))
    if not all(audit["guardrails"].values()):
        raise RuntimeError("exact-slice deterministic rate-response gate failed")
    (OUT / "exact_slice_preflight.json").write_text(json.dumps(audit, indent=2)+"\n")
    return clock_scale, audit


def save_event_table(events: list[dict]) -> None:
    if not events:
        return
    with (OUT / "event_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]))
        writer.writeheader(); writer.writerows(events)


def event_products(events: list[dict]) -> None:
    if not events:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    axes[0].hist([x["t_wait_model"] for x in events], bins="auto")
    axes[0].set_xlabel("waiting time (model)")
    axes[1].hist([x["sigma_nuc_Pa"] / 1e6 for x in events], bins="auto")
    axes[1].set_xlabel("nucleation stress (MPa)")
    axes[2].bar([x["cycle"] for x in events],
                [x["delta_sigma_event_Pa"] / 1e6 for x in events])
    axes[2].set_xlabel("cycle"); axes[2].set_ylabel("event stress drop (MPa)")
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(OUT / "event_distributions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    exported, barrier_slice = configure_barrier()
    clock_scale, preflight = deterministic_replay(exported, barrier_slice)

    # The seed is committed to disk before an RNG exists or a threshold is drawn.
    seed = int.from_bytes(os.urandom(16), "big")
    seed_record = dict(seed=seed, source="128 bits from os.urandom",
                       recorded_unix_time=time.time(), thresholds_generated=False)
    (OUT / "rng_seed_record.json").write_text(json.dumps(seed_record, indent=2)+"\n")
    rng = np.random.default_rng(seed)
    thresholds = rng.exponential(size=EVENTS_REQUESTED)
    seed_record["thresholds_generated"] = True
    seed_record["thresholds"] = thresholds.tolist()
    (OUT / "rng_seed_record.json").write_text(json.dumps(seed_record, indent=2)+"\n")

    launch = dict(
        status="LAUNCHED", events_requested=EVENTS_REQUESTED,
        seed=seed, thresholds=thresholds.tolist(),
        temperature_K=TEMPERATURE_K, fixed_clock_scale=clock_scale,
        preflight=preflight, no_retuning=True)
    (OUT / "launch_record.json").write_text(json.dumps(launch, indent=2)+"\n")
    print("AUTHORITATIVE LAUNCH", json.dumps(launch, indent=2), flush=True)

    geom, setup = build_case(); evaluator = make_evaluator(setup, geom)
    transport = make_transport(geom)
    state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    output = base.Output(geom); events = []; t_model = 0.0
    started = time.monotonic(); outcome = "RUNNING"; blocker = None
    try:
        for index, threshold in enumerate(thresholds):
            cycle = index + 1; wait_start = t_model
            state, t_model, nucleation, _ = base.wait_cycle(
                state, cycle=cycle, completed=index, t_model=t_model,
                threshold=float(threshold), clock_scale=clock_scale,
                setup=setup, geom=geom, evaluator=evaluator, output=output)
            before_sigma = nucleation["sigma_local_Pa"]
            state, t_model, restart = base.run_event(
                state, cycle=cycle, completed=index, t_model=t_model,
                nucleation=nucleation, threshold=float(threshold),
                clock_scale=clock_scale, setup=setup, geom=geom,
                evaluator=evaluator, transport=transport, output=output)
            final_row = output.rows[-1]
            events.append(dict(
                cycle=cycle, sigma_nuc_Pa=before_sigma,
                delta_sigma_event_Pa=before_sigma-final_row["sigma_local_Pa"],
                sigma_post_event_Pa=final_row["sigma_local_Pa"],
                t_wait_model=nucleation["t_model"]-wait_start,
                Vp_over_Vp_cycle=nucleation["Vp_over_Vp_cycle"],
                r_n_m=nucleation["r_n_m"], N_sites=nucleation["N_sites"],
                G_star_eV=nucleation["G_star_eV"], H_star=float(threshold),
                accepted_steps=restart["accepted_steps_total"],
                event_time_model=restart["event_time_model"]))
            save_event_table(events)
            print("EVENT COMPLETE", json.dumps(events[-1]), flush=True)
        outcome = "TEN_STOCHASTIC_ONE_B_EVENTS_COMPLETE"
    except Exception as exc:
        outcome = "STOPPED_ON_GENUINE_BLOCKER"
        blocker = f"{type(exc).__name__}: {exc}"
        print("PRODUCTION BLOCKER", blocker, flush=True)
    finally:
        if output.rows:
            base.final_products(output)
        save_event_table(events); event_products(events)
        result = dict(outcome=outcome, blocker=blocker,
                      events_completed=len(events), events_requested=EVENTS_REQUESTED,
                      seed=seed, thresholds=thresholds.tolist(),
                      temperature_K=TEMPERATURE_K,
                      fixed_clock_scale=clock_scale,
                      wall_seconds=time.monotonic()-started, events=events)
        (OUT / "production_result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
