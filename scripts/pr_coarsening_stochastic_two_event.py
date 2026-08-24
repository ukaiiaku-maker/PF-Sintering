"""Two fixed-seed stochastic one-b events on the qualified coarsening path."""
from __future__ import annotations

import csv
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
from numba import set_num_threads
from PIL import Image

os.environ["PR_RENEWAL_CASE"] = "fourier"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))

from pf_sintering.axisym_numba_kernel import NumbaScratch, axisym_gb_face_projected_step_fast  # noqa: E402
from pf_sintering.exp_barrier_nucleation import (  # noqa: E402
    CompleteExpFloorParams, EV, delta_G_complete_exp_floor_eV,
    gamma_complete_exp_floor_per_model_time, tj_site_count,
)
from pf_sintering.experimental_pr_metrology import measure_experimental_pr_state  # noqa: E402
from pr_coarsening_driven_fourier_loading import (  # noqa: E402
    OUT as PARENT, integral, observe, reservoir_remove_particle,
)
from pr_event_metrology_resolution_audit import B_M  # noqa: E402
from pr_experimental_long_sinkoff import build_case  # noqa: E402
from pr_full_deterministic_cycle import event_call, make_transport, restart_arrays  # noqa: E402
from pr_tj_node_coupling_gate import TEMPERATURE_K, make_evaluator  # noqa: E402

OUT = PARENT/"stochastic_two_event"
SEED = 42
SAMPLE_DT = 0.25
MIN_SOLVABLE_VOLUME_RATIO = 0.90
BARRIER = CompleteExpFloorParams(
    G0_eV=1.20, G_floor_eV=0.20, a=1.0,
    sigma_hat_pa=200e6, n=6.0)


def read_rows(path):
    with path.open(newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
        return rows


def rate(sigma, radius, clock_scale):
    barrier = delta_G_complete_exp_floor_eV(sigma, BARRIER)
    gamma = gamma_complete_exp_floor_per_model_time(
        sigma, radius, barrier=BARRIER,
        clock_scale_per_model_time=clock_scale, b_m=B_M,
        temperature_K=TEMPERATURE_K)
    return barrier, gamma


def calibrate(threshold):
    rows = read_rows(PARENT/"coarsening_history.csv")
    eligible = [i for i, row in enumerate(rows)
                if 0.90 <= row["Vp_over_Vp0"] <= 0.95
                and 69.5e6 <= row["sigma_local_Pa"] <= 80e6]
    if not eligible:
        raise RuntimeError("deterministic history has no target window")
    target = eligible[0]
    base = np.asarray([rate(row["sigma_local_Pa"], row["r_n_m"], 1.0)[1]
                       for row in rows])
    hazard = np.zeros(len(rows))
    for i in range(1, len(rows)):
        hazard[i] = hazard[i-1]+0.5*(base[i-1]+base[i])*(
            rows[i]["t_model"]-rows[i-1]["t_model"])
    scale = threshold/hazard[target]
    return scale, dict(
        target_time_model=rows[target]["t_model"],
        target_Vp_over_Vp0=rows[target]["Vp_over_Vp0"],
        target_sigma_local_Pa=rows[target]["sigma_local_Pa"],
        target_r_n_m=rows[target]["r_n_m"],
        unscaled_cumulative_hazard=hazard[target],
        clock_scale_per_model_time=scale,
        initial_gamma_per_model_time=base[0]*scale,
        target_gamma_per_model_time=base[target]*scale)


def save_fields(path, state, **scalars):
    np.savez_compressed(path, f=state[0], particle=state[1],
                        substrate=state[2],
                        **{k: np.asarray(v) for k, v in scalars.items()})


class Output:
    def __init__(self, geom):
        self.rows = []; self.contours = []; self.frames = []; self.geom = geom
        (OUT/"frames").mkdir(parents=True, exist_ok=True)
        (OUT/"checkpoints").mkdir(parents=True, exist_ok=True)

    def add(self, state, row, branches):
        row = dict(row); row["sample_id"] = len(self.rows)
        self.rows.append(row)
        self.contours.append(dict(
            sample_id=row["sample_id"],
            z_negative=np.asarray(branches["negative"]["z_m"]),
            r_negative=np.asarray(branches["negative"]["r_m"]),
            z_positive=np.asarray(branches["positive"]["z_m"]),
            r_positive=np.asarray(branches["positive"]["r_m"])))
        frame = OUT/"frames"/f"frame_{row['sample_id']:04d}.png"
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=130)
        for side, color in (("negative", "#b45309"), ("positive", "#1d4ed8")):
            z = np.asarray(branches[side]["z_m"])*1e9
            r = np.asarray(branches[side]["r_m"])*1e9
            ax.plot(z, r, color=color); ax.plot(z, -r, color=color)
        ax.scatter([row["z_TJ_m"]*1e9]*2,
                   [row["r_n_m"]*1e9, -row["r_n_m"]*1e9],
                   color="crimson", s=24)
        ax.set_xlim(self.geom["z_min_m"]*1e9-5,
                    self.geom["z_max_m"]*1e9+5)
        radial = self.geom["r_c"][-1]*1e9
        ax.set_ylim(-radial, radial); ax.set_aspect("equal")
        ax.set_title(
            f"t={row['t_model']:.3f}  cycle={row['cycle']}  "
            f"qcum/b={row['q_cumulative_over_b']:.2f}  "
            f"sigmaL={row['sigma_local_Pa']/1e6:.1f} MPa  "
            f"sink={'ON' if row['sink_state'] else 'OFF'}")
        ax.set_xlabel("z (nm)"); ax.set_ylabel("r (nm)"); ax.grid(alpha=.15)
        fig.savefig(frame, bbox_inches="tight"); plt.close(fig)
        self.frames.append(frame)

    def flush(self):
        if self.rows:
            with (OUT/"stochastic_history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader(); writer.writerows(self.rows)
        arrays = {}
        for item in self.contours:
            sid = item["sample_id"]
            for key in ("z_negative", "r_negative", "z_positive", "r_positive"):
                arrays[f"sample_{sid:04d}_{key}"] = item[key]
        np.savez_compressed(OUT/"stochastic_contours.npz", **arrays)


def measure(state, *, t_model, cycle, sink, q, qcum, hazard, threshold,
            clock_scale, setup, geom, evaluator, vp_cycle0):
    base = observe(state, 0.0, 0, setup, geom, evaluator, vp_cycle0, 0.0)
    measured, branches, _ = measure_experimental_pr_state(state, setup, evaluator)
    barrier, gamma = rate(base["sigma_local_Pa"], base["r_n_m"], clock_scale)
    row = dict(
        t_model=float(t_model), cycle=int(cycle), sink_state=int(sink),
        q_over_b=float(q), q_cumulative_over_b=float(qcum),
        Vp_over_Vp_cycle=float(base["Vp_over_Vp0"]),
        r_n_m=float(base["r_n_m"]), z_TJ_m=float(measured["z_TJ_m"]),
        sigma_local_Pa=float(base["sigma_local_Pa"]),
        sigma_integral_Pa=float(base["sigma_integral_Pa"]),
        G_star_eV=float(barrier), Gamma_per_model_time=float(0.0 if sink else gamma),
        H=float(hazard), H_threshold=float(threshold),
        H_over_threshold=float(hazard/threshold),
        N_sites=float(tj_site_count(base["r_n_m"], B_M)),
        Eq4_lambda_over_lambda_c=float(base["Eq4_lambda_over_lambda_c"]),
        lambda_min_H=float(base["lambda_min_H"]))
    return row, branches


def wait_cycle(state, *, cycle, completed, t_model, threshold, clock_scale,
               setup, geom, evaluator, output):
    vp0 = integral(state[1], setup); hazard = 0.0; removed = 0.0
    row, branches = measure(
        state, t_model=t_model, cycle=cycle, sink=0, q=0, qcum=completed,
        hazard=hazard, threshold=threshold, clock_scale=clock_scale,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp0)
    output.add(state, row, branches); previous_gamma = row["Gamma_per_model_time"]
    scratches = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
    which = 0; sample_steps = int(round(SAMPLE_DT/setup["dt"])); elapsed = 0.0
    while True:
        for _ in range(sample_steps):
            out = 0 if which != 0 else 1
            state = axisym_gb_face_projected_step_fast(
                *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
                setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
                setup["M_eta"], setup["W"], scratches[out])
            which = out
            f, p, s, loss = reservoir_remove_particle(state, setup)
            state = f, p, s; removed += loss
        elapsed += sample_steps*setup["dt"]
        trial, branches = measure(
            state, t_model=t_model+elapsed, cycle=cycle, sink=0, q=0,
            qcum=completed, hazard=hazard, threshold=threshold,
            clock_scale=clock_scale, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp0)
        hazard += 0.5*(previous_gamma+trial["Gamma_per_model_time"])*SAMPLE_DT
        trial["H"] = hazard; trial["H_over_threshold"] = hazard/threshold
        output.add(state, trial, branches); output.flush()
        save_fields(OUT/"checkpoints"/f"cycle{cycle}_waiting_latest.npz",
                    state, t_model=t_model+elapsed, hazard=hazard,
                    threshold=threshold, Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"])
        print("STOCHASTIC WAIT", cycle, f"t={elapsed:.3f}",
              f"V={trial['Vp_over_Vp_cycle']:.5f}",
              f"sigL={trial['sigma_local_Pa']/1e6:.3f}",
              f"H/H*={hazard/threshold:.5f}", flush=True)
        previous_gamma = trial["Gamma_per_model_time"]
        if hazard >= threshold:
            if trial["Vp_over_Vp_cycle"] < MIN_SOLVABLE_VOLUME_RATIO:
                raise RuntimeError("clock too slow: threshold crossed outside event window")
            save_fields(OUT/"checkpoints"/f"cycle{cycle}_before_nucleation.npz",
                        state, t_model=t_model+elapsed, hazard=hazard,
                        threshold=threshold)
            return state, t_model+elapsed, trial, branches
        if trial["Vp_over_Vp_cycle"] < MIN_SOLVABLE_VOLUME_RATIO:
            save_fields(OUT/"checkpoints"/f"cycle{cycle}_clock_too_slow.npz",
                        state, t_model=t_model+elapsed, hazard=hazard,
                        threshold=threshold)
            raise RuntimeError("clock too slow before nucleation")


def run_event(state, *, cycle, completed, t_model, nucleation, threshold,
              clock_scale, setup, geom, evaluator, transport, output,
              explicit_max_fourth_order_courant=None):
    vp0 = integral(state[1], setup); restart = None
    for target in (0.10, 0.25, 0.50, 0.75, 1.00):
        event_kwargs = dict(event_restart=restart)
        if explicit_max_fourth_order_courant is not None:
            event_kwargs["explicit_max_fourth_order_courant"] = (
                explicit_max_fourth_order_courant)
        result = event_call(state, setup, geom, evaluator, transport, target,
                            **event_kwargs)
        actual_courant = result[4].get(
            "explicit_max_fourth_order_courant")
        if explicit_max_fourth_order_courant is not None:
            assert actual_courant == explicit_max_fourth_order_courant, (
                "event integrator received C4="
                f"{actual_courant!r}, expected "
                f"{explicit_max_fourth_order_courant!r}")
        if not result[3]:
            save_fields(OUT/"checkpoints"/f"cycle{cycle}_event_failed.npz",
                        result[:3], q_over_b=result[4]["event_progress_over_b"])
            raise RuntimeError(f"cycle {cycle} event failed: {result[4].get('stop_reason')}")
        state = tuple(value.copy() for value in result[:3]); restart = result[4]["event_restart"]
        restart["explicit_max_fourth_order_courant"] = actual_courant
        row, branches = measure(
            state, t_model=t_model+restart["event_time_model"], cycle=cycle,
            sink=1, q=target, qcum=completed+target,
            hazard=nucleation["H"], threshold=threshold,
            clock_scale=clock_scale, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp0)
        output.add(state, row, branches); output.flush()
        path = OUT/"checkpoints"/f"cycle{cycle}_event_q{target:.2f}b.npz"
        with path.open("wb") as handle:
            np.savez_compressed(handle, **restart_arrays(state, restart))
        print("STOCHASTIC EVENT", cycle, target,
              f"sigL={row['sigma_local_Pa']/1e6:.3f}", flush=True)
    return state, t_model+restart["event_time_model"], restart


def final_products(output):
    output.flush(); rows = output.rows; t = np.asarray([r["t_model"] for r in rows])
    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True,
                             constrained_layout=True)
    axes[0].plot(t, [r["sigma_local_Pa"]/1e6 for r in rows], label="local")
    axes[0].plot(t, [r["sigma_integral_Pa"]/1e6 for r in rows], label="integral")
    axes[1].plot(t, [r["r_n_m"]*1e9 for r in rows]); axes[1].set_ylabel("r_n nm")
    axes[2].plot(t, [r["Vp_over_Vp_cycle"] for r in rows]); axes[2].set_ylabel("Vp/Vcycle")
    axes[3].plot(t, [r["H_over_threshold"] for r in rows]); axes[3].set_ylabel("H/H*")
    axes[4].plot(t, [r["q_cumulative_over_b"] for r in rows]); axes[4].set_ylabel("qcum/b")
    axes[5].step(t, [r["sink_state"] for r in rows], where="post"); axes[5].set_ylabel("sink")
    axes[0].set_ylabel("stress MPa"); axes[0].legend(); axes[5].set_xlabel("model time")
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(OUT/"two_event_stochastic_history.png", dpi=170); plt.close(fig)
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE)
              for path in output.frames]
    images[0].save(OUT/"two_event_morphology.gif", save_all=True,
                   append_images=images[1:], duration=350, loop=0)
    thumbs = [Image.open(path).convert("RGB") for path in output.frames]
    choose = np.linspace(0, len(thumbs)-1, min(20, len(thumbs))).astype(int)
    w, h = thumbs[0].size; sheet = Image.new("RGB", (4*w, math.ceil(len(choose)/4)*h), "white")
    for k, index in enumerate(choose): sheet.paste(thumbs[index], ((k%4)*w, (k//4)*h))
    sheet.save(OUT/"two_event_contact_sheet.jpg", quality=90)


def main():
    set_num_threads(8); OUT.mkdir(parents=True, exist_ok=True)
    geom, setup = build_case(); evaluator = make_evaluator(setup, geom)
    transport = make_transport(geom); state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    rng = np.random.default_rng(SEED); thresholds = rng.exponential(size=2)
    clock_scale, calibration = calibrate(float(thresholds[0]))
    forecast = dict(seed=SEED, thresholds=thresholds.tolist(),
                    fixed_clock_scale=clock_scale, calibration=calibration,
                    barrier=dict(G0_eV=1.2, G_floor_eV=.2, a=1., sigma_hat_Pa=200e6, n=6.),
                    same_multiplier_all_cycles=True, no_parameter_matrix=True)
    (OUT/"launch_forecast.json").write_text(json.dumps(forecast, indent=2)+"\n")
    print("LAUNCH FORECAST", json.dumps(forecast, indent=2), flush=True)
    output = Output(geom); t_model = 0.0; events = []; started = time.monotonic()
    for index in range(2):
        cycle = index+1
        state, t_model, nucleation, branches = wait_cycle(
            state, cycle=cycle, completed=index, t_model=t_model,
            threshold=float(thresholds[index]), clock_scale=clock_scale,
            setup=setup, geom=geom, evaluator=evaluator, output=output)
        state, t_model, restart = run_event(
            state, cycle=cycle, completed=index, t_model=t_model,
            nucleation=nucleation, threshold=float(thresholds[index]),
            clock_scale=clock_scale, setup=setup, geom=geom,
            evaluator=evaluator, transport=transport, output=output)
        events.append(dict(cycle=cycle, nucleation=nucleation,
                           accepted_steps=restart["accepted_steps_total"],
                           event_time_model=restart["event_time_model"]))
    final_products(output)
    result = dict(outcome="TWO_STOCHASTIC_ONE_B_EVENTS_COMPLETE",
                  events=events, clock_scale=clock_scale, seed=SEED,
                  thresholds=thresholds.tolist(), wall_seconds=time.monotonic()-started)
    (OUT/"stochastic_result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
