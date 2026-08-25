#!/usr/bin/env python3
"""Create the final finite-barrier presentation from the saved trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pf_sintering.pr_movie_geometry import FRAME_TYPES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_three_deltaG0p300_taucorr9ms_vpcycle_reset")
B_NM = 0.25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run
    result = json.loads((run / "avalanche_renewal_result.json").read_text())
    with h5py.File(run / "avalanche_renewal_movie_geometry.h5", "r") as file:
        t_ms = np.asarray(file["time/t_s"], dtype=float) * 1e3
        local = np.asarray(file["state/sigma_local_Pa"], dtype=float) / 1e6
        integral = np.asarray(
            file["state/sigma_integral_Pa"], dtype=float) / 1e6
        cumulative = np.asarray(
            file["state/Q_cumulative_over_b"], dtype=float)
        sink = np.asarray(file["state/sink_state"], dtype=int)
        active = np.asarray(file["state/avalanche_active"], dtype=int)
        avalanche_id = np.asarray(file["state/avalanche_id"], dtype=int)
        event_number = np.asarray(file["state/event_number"], dtype=int)
        completed = np.asarray(file["state/S_completed"], dtype=int)
        q_event = np.asarray(file["state/q_event_over_b"], dtype=float)
        neck_nm = np.asarray(file["state/r_neck_m"], dtype=float) * 1e9
        frame_type = np.asarray(file["state/frame_type"], dtype=int)

    segments = []
    complete_by_id = {
        int(item["avalanche_id"]): item for item in result["avalanches"]}
    for aid in sorted(set(avalanche_id[avalanche_id > 0])):
        indices = np.flatnonzero(avalanche_id == aid)
        roots = indices[frame_type[indices] == FRAME_TYPES["root_nucleation"]]
        start = int(roots[0]) if roots.size else int(indices[0])
        extinctions = indices[
            frame_type[indices] == FRAME_TYPES["avalanche_extinction"]]
        is_complete = extinctions.size > 0
        end = int(extinctions[-1]) if is_complete else int(indices[-1])
        interval = np.arange(start, end + 1)
        S = int(np.max(completed[interval]))
        partial_q = float(q_event[end]) if not is_complete else 0.0
        root_reload_ms = (
            float(t_ms[start]) if aid == 1
            else float(t_ms[start] - segments[-1]["end_ms"]))
        segment = dict(
            avalanche_id=int(aid), complete=bool(is_complete),
            start_index=start, end_index=end,
            start_ms=float(t_ms[start]), end_ms=float(t_ms[end]),
            duration_ms=float(t_ms[end] - t_ms[start]),
            sigma_root_MPa=float(local[start]),
            sigma_end_MPa=float(local[end]),
            delta_sigma_root_to_end_MPa=float(local[start] - local[end]),
            maximum_resolved_relaxation_from_root_MPa=float(
                local[start] - np.min(local[interval])),
            minimum_stress_MPa=float(np.min(local[interval])),
            S_completed=S, partial_event_q_over_b=partial_q,
            displacement_completed_nm=float(S * B_NM),
            root_reload_ms=root_reload_ms,
            max_event_number=int(np.max(event_number[interval])))
        if aid in complete_by_id:
            segment["saved_summary"] = complete_by_id[aid]
        segments.append(segment)

    colors = ("#d97706", "#7c3aed", "#059669", "#dc2626")
    fig, axes = plt.subplots(
        6, 1, figsize=(13.2, 14.4), sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.8, 1.05, 1.0, .65, .65, 1.0]})
    for segment, color in zip(segments, colors):
        label = f"Avalanche {segment['avalanche_id']}"
        if not segment["complete"]:
            label += " (blocked, partial)"
        for ax in axes:
            ax.axvspan(
                segment["start_ms"], segment["end_ms"], color=color,
                alpha=.10, linewidth=0, label=label if ax is axes[0] else None)

    axes[0].plot(t_ms, local, color="#174a73", linewidth=1.8, label="local")
    axes[0].plot(
        t_ms, integral, color="#9a6b28", linewidth=1.5,
        linestyle="--", label="integral")
    axes[0].set_ylabel("Stress (MPa)")
    axes[0].legend(frameon=False, ncol=2, loc="best")

    axes[1].plot(t_ms, cumulative, color="#2f6f58", linewidth=1.8)
    axes[1].set_ylabel(r"Cumulative $Q/b$")

    axes[2].set_ylabel("Avalanche size")
    for segment, color in zip(segments, colors):
        midpoint = .5 * (segment["start_ms"] + segment["end_ms"])
        size = segment["S_completed"]
        axes[2].vlines(midpoint, 0, size, color=color, linewidth=2)
        axes[2].scatter([midpoint], [size], color=color, s=34, zorder=3)
        suffix = "" if segment["complete"] else (
            f" + q={segment['partial_event_q_over_b']:.2f} partial")
        axes[2].annotate(
            f"A{segment['avalanche_id']}: S={size}{suffix}",
            (midpoint, size), xytext=(5, 3), textcoords="offset points",
            fontsize=9, color=color)
    axes[2].set_ylim(bottom=0)

    axes[3].step(t_ms, sink, where="post", color="#30343b")
    axes[3].set_ylabel("Sink")
    axes[3].set_yticks([0, 1], ["OFF", "ON"])

    axes[4].step(t_ms, active, where="post", color="#8b4f43")
    axes[4].set_ylabel("Source")
    axes[4].set_yticks([0, 1], ["root", "facilitated"])

    axes[5].plot(t_ms, neck_nm, color="#4d6d72", linewidth=1.8)
    axes[5].axhline(35.0, color="#991b1b", linestyle=":", linewidth=1.2,
                    label="35 nm validity safeguard")
    axes[5].set_ylabel("Neck radius (nm)")
    axes[5].set_xlabel("Physical time from original Fourier state (ms)")
    axes[5].legend(frameon=False, loc="best")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(
        "Fixed-decrement stochastic PR avalanche renewal\n"
        r"$\Delta G_{step}=0.300000$ eV, $\tau_{corr}=9$ ms; "
        f"outcome: {result['outcome']}")
    png = run / "finite_barrier_avalanche_final.png"
    pdf = run / "finite_barrier_avalanche_final.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    presentation = dict(
        outcome=result["outcome"], blocker=result["blocker"],
        seed=result["seed"], delta_G_step_eV=result[
            "descendant_delta_G_step_eV"], tau_corr_s=result["tau_corr_s"],
        C4=result["C4_received_and_asserted"],
        total_complete_one_b_events=result["total_one_b_events"],
        complete_avalanches=result["avalanches_completed"],
        requested_avalanches=result["avalanches_requested"],
        t_end_ms=float(t_ms[-1]), Q_end_over_b=float(cumulative[-1]),
        r_neck_end_nm=float(neck_nm[-1]),
        sigma_local_end_MPa=float(local[-1]),
        sigma_integral_end_MPa=float(integral[-1]),
        avalanches=segments,
        figure_png=str(png), figure_pdf=str(pdf),
        morphology_gif=result["movie"]["path"])
    (run / "finite_barrier_avalanche_final.json").write_text(
        json.dumps(presentation, indent=2) + "\n")
    print(json.dumps(presentation, indent=2))


if __name__ == "__main__":
    main()
