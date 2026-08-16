"""M16O Section 19: pathway figure set (C, D, E, F, G -- A/B require
morphology PNGs / analytic-map metadata not central to the core finding
and are described in the report text instead, per the time budget)."""
from __future__ import annotations

import csv
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from m16e_exact_hussein_two_mode import psi_to_gamma_gb  # noqa: E402
from pf_sintering.m16o_loading_criterion import loading_terms_dt  # noqa: E402

GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(160.0, GAMMA_S)
C_GB = math.sqrt(1.0 - (GAMMA_GB / (2.0 * GAMMA_S)) ** 2)

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16o_topology_screen")
FLAT_CSV = os.path.join(os.path.dirname(__file__), "..", "runs", "m16n_sinkoff_screen", "ratio0.185_W6dx0.85.csv")
OUT_DIR = os.path.join(RUN_DIR, "figures")


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["time"]) for r in rows])
    r = np.array([float(r["r_neck_nm"]) for r in rows]) * 1e-9
    X = np.array([float(r["X_neck_nm"]) for r in rows]) * 1e-9
    sigma = np.array([float(r["sigma_MPa"]) for r in rows])
    return t, r, X, sigma


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = {"flat (M16N)": FLAT_CSV}
    for f in sorted(glob.glob(os.path.join(RUN_DIR, "chi*_ratio0.185_W6dx0.85.csv"))):
        label = os.path.basename(f).split("_ratio")[0]
        cases[label] = f

    data = {label: load(path) for label, path in cases.items() if os.path.exists(path)}

    # Figure C: sigma(t)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (t, r, X, sigma) in data.items():
        ax.plot(t, sigma, marker=".", ms=3, label=label)
    ax.set_xlabel("time (model units)")
    ax.set_ylabel("sigma_Hussein (MPa)")
    ax.set_title("M16O Figure C: sigma(t) across the chi topology family (ratio=0.185)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "C_sigma_vs_time.png"), dpi=150)
    plt.close(fig)

    # Figure D: r(t) and X(t)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for label, (t, r, X, sigma) in data.items():
        axes[0].plot(t, r * 1e9, marker=".", ms=3, label=label)
        axes[1].plot(t, X * 1e9, marker=".", ms=3, label=label)
    axes[0].set_ylabel("r_neck (nm)")
    axes[1].set_ylabel("X_neck (nm)")
    axes[1].set_xlabel("time (model units)")
    axes[0].set_title("M16O Figure D: r_neck(t) and X_neck(t)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "D_r_and_X_vs_time.png"), dpi=150)
    plt.close(fig)

    # Figure E: L_r, L_X, L_total decomposition (one panel per case)
    fig, axes = plt.subplots(len(data), 1, figsize=(8, 3 * len(data)), sharex=False)
    if len(data) == 1:
        axes = [axes]
    for ax, (label, (t, r, X, sigma)) in zip(axes, data.items()):
        terms = loading_terms_dt(t, r, X, GAMMA_S, C_GB)
        ax.plot(terms["t_mid"], terms["L_r"], label="L_r (curvature)", color="tab:blue")
        ax.plot(terms["t_mid"], terms["L_X"], label="L_X (contact-width)", color="tab:orange")
        ax.plot(terms["t_mid"], terms["L_total"], label="L_total", color="black", lw=2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(label)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time (model units)")
    fig.suptitle("M16O Figure E: sigma-derivative decomposition (L_r, L_X, L_total)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "E_decomposition.png"), dpi=150)
    plt.close(fig)

    # Figure F: sigma vs r_neck (geometric coordinate)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (t, r, X, sigma) in data.items():
        ax.plot(r * 1e9, sigma, marker=".", ms=3, label=label)
    ax.set_xlabel("r_neck (nm)")
    ax.set_ylabel("sigma_Hussein (MPa)")
    ax.set_title("M16O Figure F: sigma vs r_neck")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "F_sigma_vs_rneck.png"), dpi=150)
    plt.close(fig)

    # Figure G: morphology sequence proxy -- r_neck/X_neck trajectory in
    # (r,X) space for the loading case, annotated with sigma
    if "chi1.5" in data:
        t, r, X, sigma = data["chi1.5"]
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(r * 1e9, X * 1e9, c=sigma, cmap="viridis", s=25)
        ax.plot(r * 1e9, X * 1e9, color="gray", lw=0.5, zorder=0)
        plt.colorbar(sc, ax=ax, label="sigma_Hussein (MPa)")
        ax.set_xlabel("r_neck (nm)")
        ax.set_ylabel("X_neck (nm)")
        ax.set_title("M16O Figure G: chi=1.5 loading trajectory in (r_neck, X_neck) space")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "G_loading_trajectory_rX.png"), dpi=150)
        plt.close(fig)

    print(f"DONE -- wrote figures to {OUT_DIR} for cases: {list(data.keys())}")


if __name__ == "__main__":
    main()
