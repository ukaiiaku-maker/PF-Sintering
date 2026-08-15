"""Milestone 16M Section 23 (+ user addendum) plots for the independent
Poisson multi-sink qualification run (runs/m16m_multisink_qualification/).

Produces:
  A. sigma(t) with birth/completion event markers
  B. N_active(t)
  C. cumulative event count (born, completed) vs time
  D. cumulative_RBM/b vs time
  E. sigma vs cumulative_RBM/b
  F. X_neck vs cumulative_RBM/b
  G. strain (cumulative_RBM/(2*a_contact)) vs time  [user addendum]
  H. Lambda_birth vs sigma
  I. B=Lambda*tau_event vs sigma
  J. event waiting-time distribution (birth-to-birth intervals)
  K. event lifetime distribution (birth-to-completion durations)
  L. combined sigma(t) / strain(t) / N_active(t) summary panel [user addendum]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16m_multisink_qualification")
OUT_DIR = os.path.join(RUN_DIR, "plots")


def load():
    hist = pd.read_csv(os.path.join(RUN_DIR, "history.csv"))
    events = []
    events_path = os.path.join(RUN_DIR, "events.jsonl")
    if os.path.exists(events_path):
        with open(events_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return hist, events


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hist, events = load()
    births = [e for e in events if e["kind"] == "birth"]
    completes = [e for e in events if e["kind"] == "complete"]

    b = 2.5e-10

    # A: sigma(t) with event markers
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist["time"], hist["sigma_Hussein_MPa"], lw=1.0, color="tab:blue", label="sigma_Hussein (operative)")
    for e in births:
        ax.axvline(e["t"], color="tab:green", alpha=0.3, lw=0.8)
    for e in completes:
        ax.axvline(e["t"], color="tab:red", alpha=0.3, lw=0.8)
    if births:
        ax.scatter([e["t"] for e in births], [e["sigma_MPa"] for e in births], marker="^", color="tab:green",
                   s=40, zorder=5, label="birth")
    if completes:
        ax.scatter([e["t"] for e in completes], [e["sigma_MPa"] for e in completes], marker="v", color="tab:red",
                   s=40, zorder=5, label="completion")
    ax.set_xlabel("time (model units)")
    ax.set_ylabel("sigma_Hussein (MPa)")
    ax.set_title("M16M: sintering stress vs time, with Poisson birth/completion events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "A_sigma_vs_time.png"), dpi=150)
    plt.close(fig)

    # B: N_active(t)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(hist["time"], hist["N_active"], where="post", color="tab:purple")
    ax.set_xlabel("time (model units)")
    ax.set_ylabel("N_active")
    ax.set_title("M16M: number of simultaneously-active sink events vs time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "B_N_active_vs_time.png"), dpi=150)
    plt.close(fig)

    # C: cumulative event counts
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["time"], hist["N_born"], label="N_born (cumulative)", color="tab:green")
    ax.plot(hist["time"], hist["N_completed"], label="N_completed (cumulative)", color="tab:red")
    ax.set_xlabel("time (model units)")
    ax.set_ylabel("cumulative count")
    ax.legend()
    ax.set_title("M16M: cumulative birth/completion counts vs time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "C_cumulative_events_vs_time.png"), dpi=150)
    plt.close(fig)

    # D: cumulative RBM/b vs time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["time"], hist["cumulative_RBM_over_b"], color="tab:orange")
    ax.set_xlabel("time (model units)")
    ax.set_ylabel("cumulative RBM / b")
    ax.set_title("M16M: cumulative RBM displacement (Burgers-vector units) vs time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "D_cumulative_RBM_vs_time.png"), dpi=150)
    plt.close(fig)

    # E: sigma vs cumulative RBM/b
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(hist["cumulative_RBM_over_b"], hist["sigma_Hussein_MPa"], c=hist["time"], cmap="viridis", s=8)
    plt.colorbar(sc, ax=ax, label="time")
    ax.set_xlabel("cumulative RBM / b")
    ax.set_ylabel("sigma_Hussein (MPa)")
    ax.set_title("M16M: sigma vs cumulative RBM/b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E_sigma_vs_cumulative_RBM.png"), dpi=150)
    plt.close(fig)

    # F: X_neck vs cumulative RBM/b
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["cumulative_RBM_over_b"], hist["X_neck_nm"], color="tab:brown", lw=1.0, marker=".", ms=2)
    ax.set_xlabel("cumulative RBM / b")
    ax.set_ylabel("X_neck (nm)")
    ax.set_title("M16M: neck width vs cumulative RBM/b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "F_Xneck_vs_cumulative_RBM.png"), dpi=150)
    plt.close(fig)

    # G: strain vs time (user addendum)
    strain = hist["cumulative_RBM_nm"] / (2.0 * hist["a_contact_nm"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["time"], strain, color="tab:cyan")
    ax.set_xlabel("time (model units)")
    ax.set_ylabel("RBM strain = cumulative_RBM / (2*a_contact)")
    ax.set_title("M16M: densification strain vs time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "G_strain_vs_time.png"), dpi=150)
    plt.close(fig)

    # H: Lambda_birth vs sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(hist["sigma_Hussein_MPa"].values)
    ax.plot(hist["sigma_Hussein_MPa"].values[order], hist["Lambda_birth_per_s"].values[order], ".", ms=3,
            color="tab:green")
    ax.set_xlabel("sigma_Hussein (MPa)")
    ax.set_ylabel("Lambda_birth (events/s)")
    ax.set_yscale("log")
    ax.set_title("M16M: calibrated birth intensity vs instantaneous stress")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "H_Lambda_vs_sigma.png"), dpi=150)
    plt.close(fig)

    # I: B = Lambda*tau_event vs sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["sigma_Hussein_MPa"].values[order], hist["B_overlap"].values[order], ".", ms=3, color="tab:red")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="B=1 (appreciable overlap)")
    ax.set_xlabel("sigma_Hussein (MPa)")
    ax.set_ylabel("B = Lambda * tau_event")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("M16M: overlap number vs instantaneous stress")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "I_B_vs_sigma.png"), dpi=150)
    plt.close(fig)

    # J: waiting-time distribution
    birth_times = sorted(e["t"] for e in births)
    waits = np.diff(birth_times) if len(birth_times) > 1 else np.array([])
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(waits) > 0:
        ax.hist(waits, bins=min(20, max(3, len(waits))), color="tab:green", alpha=0.8)
    ax.set_xlabel("inter-birth waiting time (model time units)")
    ax.set_ylabel("count")
    ax.set_title(f"M16M: event waiting-time distribution (n={len(waits)} intervals)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "J_waiting_time_distribution.png"), dpi=150)
    plt.close(fig)

    # K: event lifetime distribution
    lifetimes = [e["t"] - next(b_["t"] for b_ in births if b_["event_id"] == e["event_id"])
                 for e in completes if any(b_["event_id"] == e["event_id"] for b_ in births)]
    fig, ax = plt.subplots(figsize=(7, 5))
    if lifetimes:
        ax.hist(lifetimes, bins=min(20, max(3, len(lifetimes))), color="tab:red", alpha=0.8)
    ax.set_xlabel("event lifetime (birth to completion, model time units)")
    ax.set_ylabel("count")
    ax.set_title(f"M16M: event lifetime distribution (n={len(lifetimes)} completed)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "K_lifetime_distribution.png"), dpi=150)
    plt.close(fig)

    # L: combined summary panel (user addendum: sigma/strain/N_active vs time together)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(hist["time"], hist["sigma_Hussein_MPa"], color="tab:blue")
    for e in births:
        axes[0].axvline(e["t"], color="tab:green", alpha=0.25, lw=0.8)
    for e in completes:
        axes[0].axvline(e["t"], color="tab:red", alpha=0.25, lw=0.8)
    axes[0].set_ylabel("sigma_Hussein (MPa)")
    axes[0].set_title("M16M summary: sintering stress, densification strain, active-sink count vs time")

    axes[1].plot(hist["time"], strain, color="tab:cyan")
    axes[1].set_ylabel("RBM strain")

    axes[2].step(hist["time"], hist["N_active"], where="post", color="tab:purple")
    axes[2].set_ylabel("N_active")
    axes[2].set_xlabel("time (model units)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "L_summary_sigma_strain_Nactive_vs_time.png"), dpi=150)
    plt.close(fig)

    print(f"DONE -- wrote plots to {OUT_DIR} (n_history_rows={len(hist)}, n_births={len(births)}, "
          f"n_completes={len(completes)})")


if __name__ == "__main__":
    main()
