"""M16I barrier-ladder correction: standalone figure generation for the
50 MPa-target FINITE run (runs/m16i_three_barrier_highstress/finite).

This regime lives in its own campaign root (see the milestone report,
Section 9, for why) and was never run through the standard
final_regime_figures()/postprocess() pipeline, which assumes all three
regimes share one campaign root. This script produces the same kind of
per-regime evolution plots directly from history.jsonl: sigma_s(t),
sintering_strain(t), X_neck(t), and r_neck(t) (the last of which isn't
part of the standard final_regime_figures() set at all). Read-only:
only reads history.jsonl/events.jsonl, writes only new figure PNGs.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.plotting_style import apply_style, mark_events, style_axes  # noqa: E402

RUN_DIR = "runs/m16i_three_barrier_highstress/finite"
COLOR = "#8A2BE2"  # matches the highstress line color used in barrier_ladder_comparison.png


def load_history(run_dir):
    path = os.path.join(run_dir, "history.jsonl")
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_events(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l)["t"] for l in fh if l.strip()]


def main():
    apply_style()
    rows = load_history(RUN_DIR)
    events = load_events(RUN_DIR)
    t = [r["time"] for r in rows]
    fig_dir = os.path.join(RUN_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    specs = [
        ("sigma_s_t", "sigma_s_1p5W_MPa", "sigma_s (MPa)",
         "Hussein Eq. 1b stress, sigma_target_hazard = 50 MPa (never reached)"),
        ("sintering_strain_t", "sintering_strain", "sintering strain",
         "sintering strain (no RBM events occurred -- strain is pure PF coarsening)"),
        ("X_neck_t", "X_neck_nm", "X_neck (nm)", "neck width X_neck = 2*a_contact"),
        ("r_neck_t", "r_neck_1p5W_nm", "r_neck (nm)", "neck (fillet) radius of curvature, 1.5W window"),
    ]
    for name, field, ylabel, title in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        y = [r.get(field) for r in rows]
        ax.plot(t, y, color=COLOR, linewidth=1.5)
        if events:
            mark_events(ax, events)
        style_axes(ax, "time", ylabel)
        ax.set_title(title, fontsize=10, color="0.3")
        fig.tight_layout()
        out_path = os.path.join(fig_dir, f"highstress_{name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}  (t=[{t[0]:.2f},{t[-1]:.2f}], "
              f"{field}: {y[0]:.5g} -> {y[-1]:.5g})")

    # Combined 2x2 panel for a single-glance overview.
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    panel_map = [(0, 0, "sigma_s_1p5W_MPa", "sigma_s (MPa)"),
                 (0, 1, "sintering_strain", "sintering strain"),
                 (1, 0, "X_neck_nm", "X_neck (nm)"),
                 (1, 1, "r_neck_1p5W_nm", "r_neck (nm)")]
    for ri, ci, field, ylabel in panel_map:
        ax = axs[ri, ci]
        ax.plot(t, [r.get(field) for r in rows], color=COLOR, linewidth=1.4)
        style_axes(ax, "time" if ri == 1 else None, ylabel)
    fig.suptitle(f"High-stress FINITE (sigma_target=50 MPa, V0=1000*Omega, A0=3.339 eV) -- "
                 f"n_events={len(events)} over t=[{t[0]:.1f},{t[-1]:.1f}]", fontsize=11, color="0.25")
    fig.tight_layout()
    combined_path = os.path.join(fig_dir, "highstress_overview_2x2.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()
