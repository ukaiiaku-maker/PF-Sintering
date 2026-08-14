"""Milestone 16J Section 33: required search figures A-I.

Builds all nine figures from whatever Stage-0/Stage-1 data is currently
on disk. Figures A-C use the Stage-0 static-screen matrix (candidates.csv,
15 candidates across chi and X0/(2Rp)). Figures D-G use the Stage-1
short-PF-evolution time series for whichever candidates have completed
(or partially completed) runs. Figure H reports the observed/predicted
time-to-50-MPa directly from the Stage-1 trajectories. Figure I is a
morphology montage from the Stage-0 initial-condition snapshots.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.plotting_style import apply_style, style_axes  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16j_geometry_search")
STAGE1_ROOT = os.path.join(ROOT, "stage1")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def stage1_history(cdir_name):
    path = os.path.join(STAGE1_ROOT, cdir_name, "history.csv")
    return load_csv(path)


def main():
    apply_style()
    stage0 = load_csv(os.path.join(ROOT, "stage0_candidates.csv"))

    # --- Figures A, B, C: static-screen matrix vs R_s/R_p ---
    ratios = sorted({r["X0_over_Rp"] for r in stage0 if isinstance(r.get("X0_over_Rp"), float)})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ratios)))

    for fig_name, field, ylabel, title in [
        ("figureA_sigma_vs_chi", "sigma_hussein_MPa", "sigma_H (MPa, t=0 construction)",
         "Figure A: initial sigma vs R_s/R_p"),
        ("figureB_rneck_over_Rp_vs_chi", "r_neck_over_Rp", "r_neck/R_p", "Figure B: r_neck/R_p vs R_s/R_p"),
        ("figureC_Xneck_over_Rp_vs_chi", "X_neck_over_Rp", "X_neck/R_p", "Figure C: X_neck/R_p vs R_s/R_p"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for ratio, color in zip(ratios, colors):
            rows = sorted((r for r in stage0 if r.get("X0_over_Rp") == ratio and r.get("qc_pass")),
                          key=lambda r: r["R_s_over_R_p"])
            chi_vals = [r["R_s_over_R_p"] for r in rows]
            y_vals = [r[field] for r in rows]
            chi_plot = [1e4 if not np.isfinite(c) else c for c in chi_vals]  # place flat (inf) at a large finite x
            ax.plot(chi_plot, y_vals, "o-", color=color, label=f"X0/Rp={ratio:.3f}")
        ax.set_xscale("log")
        style_axes(ax, "R_s/R_p  (rightmost point = exact flat)", ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{fig_name}.png"), dpi=140)
        plt.close(fig)

    # --- Figures D, E, F, G: Stage-1 time series for leading geometries ---
    leading = [
        ("Rp1000nm_flat_X0over2Rp0.100_t8_ext_archive", "flat, X0/(2Rp)=0.10 (W=10nm, coarse)", "#B4530A"),
        ("Rp1000nm_flat_X0over2Rp0.100_refined_W6dx1", "flat, X0/(2Rp)=0.10 (W=6nm, refined)", "#D62728"),
        ("Rp1000nm_flat_X0over2Rp0.050", "flat, X0/(2Rp)=0.05", "#8A2BE2"),
        ("Rp1000nm_chi1_X0over2Rp0.050", "chi=1, X0/(2Rp)=0.05", "#2E8B57"),
        ("Rp1000nm_chi2_X0over2Rp0.050", "chi=2, X0/(2Rp)=0.05", "#4C4C9D"),
        ("Rp1000nm_chi5_X0over2Rp0.050", "chi=5, X0/(2Rp)=0.05", "#B22222"),
        ("Rp1000nm_chi10_X0over2Rp0.050", "chi=10, X0/(2Rp)=0.05", "0.4"),
    ]
    data = {}
    for cdir, label, color in leading:
        rows = stage1_history(cdir)
        if rows:
            data[cdir] = (rows, label, color)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cdir, (rows, label, color) in data.items():
        t = [r["time"] for r in rows]
        sig = [r["sigma_MPa"] for r in rows]
        ax.plot(t, sig, color=color, linewidth=1.3, label=label)
    ax.axhline(50.0, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    style_axes(ax, "time", "sigma_s (MPa)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure D: sigma_s(t), Stage-1 leading geometries (dashed: 50 MPa target)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figureD_sigma_vs_time.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cdir, (rows, label, color) in data.items():
        t = [r["time"] for r in rows]
        rn = [r["r_neck_nm"] for r in rows]
        ax.plot(t, rn, color=color, linewidth=1.3, label=label)
    style_axes(ax, "time", "r_neck (nm)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure E: r_neck(t), Stage-1 leading geometries", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figureE_rneck_vs_time.png"), dpi=140)
    plt.close(fig)

    fig_f_source = "Rp1000nm_flat_X0over2Rp0.100_refined_W6dx1" if "Rp1000nm_flat_X0over2Rp0.100_refined_W6dx1" in data \
        else "Rp1000nm_flat_X0over2Rp0.100_t8_ext_archive"
    if fig_f_source in data:
        rows, label, _ = data[fig_f_source]
        t = [r["time"] for r in rows]
        sig_c = [r["sigma_curvature_MPa"] for r in rows]
        sig_w = [r["sigma_width_MPa"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(t, sig_c, color="#B4530A", label="sigma_curvature = gamma_s/r_neck")
        ax.plot(t, sig_w, color="#2E8B57", label="sigma_width = -gamma_s*C_GB/X_neck")
        style_axes(ax, "time", "MPa")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Figure F: stress decomposition, {label}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "figureF_stress_decomposition.png"), dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cdir, (rows, label, color) in data.items():
        t = [r["time"] for r in rows]
        q = [r["Q"] for r in rows]
        ax.plot(t, q, color=color, linewidth=1.3, label=label)
    ax.axhline(1.0, color="0.3", linewidth=0.7, linestyle="--")
    style_axes(ax, "time", "Q = X_neck/(C_GB*r_neck)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure G: Q(t) (dashed: Q=1, sigma_s=0 boundary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figureG_Q_vs_time.png"), dpi=140)
    plt.close(fig)

    # --- Figure H: observed/predicted time-to-50-MPa ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels_h, times_h, reached = [], [], []
    for cdir, (rows, label, color) in data.items():
        t_cross = None
        for r in rows:
            if r["sigma_MPa"] >= 50.0:
                t_cross = r["time"]
                break
        labels_h.append(label)
        times_h.append(t_cross if t_cross is not None else np.nan)
        reached.append(t_cross is not None)
    y_pos = np.arange(len(labels_h))
    bar_colors = ["#B4530A" if r else "0.75" for r in reached]
    bar_vals = [t if np.isfinite(t) else max([tt for tt in times_h if np.isfinite(tt)] + [1.0]) * 1.15
                for t in times_h]
    ax.barh(y_pos, bar_vals, color=bar_colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels_h, fontsize=8)
    for i, (t, r) in enumerate(zip(times_h, reached)):
        txt = f"t={t:.2f}" if r else "not reached in tested window"
        ax.text(bar_vals[i] * 1.01, i, txt, va="center", fontsize=7.5)
    ax.set_xlabel("time to first reach sigma_s=50 MPa")
    ax.set_title("Figure H: time-to-50-MPa by geometry", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figureH_time_to_50MPa.png"), dpi=140)
    plt.close(fig)

    # --- Figure I: morphology montage (Stage-0 initial conditions) ---
    morph_files = sorted(glob.glob(os.path.join(ROOT, "stage0_morphology", "*.png")))
    if morph_files:
        n = len(morph_files)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
        axs = np.atleast_2d(axs)
        for i, path in enumerate(morph_files):
            r, c = divmod(i, ncols)
            img = plt.imread(path)
            axs[r, c].imshow(img)
            axs[r, c].axis("off")
            axs[r, c].set_title(os.path.basename(path).replace(".png", ""), fontsize=6)
        for i in range(n, nrows * ncols):
            r, c = divmod(i, ncols)
            axs[r, c].axis("off")
        fig.suptitle("Figure I: Stage-0 morphology montage (initial conditions)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "figureI_morphology_montage.png"), dpi=110)
        plt.close(fig)

    print(f"wrote figures to {FIG_DIR}")
    print("Figure H crossing times:", dict(zip(labels_h, times_h)))


if __name__ == "__main__":
    main()
