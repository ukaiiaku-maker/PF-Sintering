"""M16I Barrier-Ladder Correction, Section: final comparison figure.

Overlays baseline-FINITE (sigma_target=2.5MPa, runs/m16i_three_barrier_prod),
high-stress-FINITE (sigma_target=50MPa, runs/m16i_three_barrier_highstress),
and OFF (runs/m16i_three_barrier_prod) on the same axes. The two FINITE cases
live in separate campaign roots because launching the high-stress case into
the prod root's `finite/` subdirectory collided with (and resumed) the
already-completed baseline checkpoint -- see the milestone report for detail.
This script is read-only: it loads history.jsonl from each regime directory
directly and never touches a checkpoint or sink state.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.plotting_style import apply_style, mark_events, style_axes  # noqa: E402

PROD_ROOT = "runs/m16i_three_barrier_prod"
HIGHSTRESS_ROOT = "runs/m16i_three_barrier_highstress"

CASES = [
    ("baseline_finite", os.path.join(PROD_ROOT, "finite"), "#B4530A", "FINITE, sigma_target=2.5 MPa (baseline)"),
    ("highstress_finite", os.path.join(HIGHSTRESS_ROOT, "finite"), "#8A2BE2", "FINITE, sigma_target=50 MPa (V0=1000*Omega)"),
    ("off", os.path.join(PROD_ROOT, "off"), "#4C4C9D", "OFF (infinite barrier)"),
]


def load_history(run_dir):
    path = os.path.join(run_dir, "history.jsonl")
    if not os.path.exists(path):
        return []
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
    data = {name: load_history(run_dir) for name, run_dir, _, _ in CASES}
    baseline_events = load_events(os.path.join(PROD_ROOT, "finite"))

    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    specs = [("sigma_s_1p5W_MPa", "sigma_s (MPa)"),
             ("r_neck_1p5W_nm", "r_neck (nm)"),
             ("X_neck_nm", "X_neck (nm)")]
    for ax, (field, ylabel) in zip(axs, specs):
        for name, run_dir, color, label in CASES:
            rows = data[name]
            if not rows:
                continue
            t = [r.get("time") for r in rows]
            y = [r.get(field) for r in rows]
            ax.plot(t, y, color=color, linewidth=1.3, label=label)
        mark_events(ax, baseline_events)
        style_axes(ax, None, ylabel)

    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    axs[-1].set_xlabel("time")
    fig.suptitle(
        "Barrier-ladder comparison: baseline FINITE (2.5 MPa target, 4 events)\n"
        "vs. high-stress FINITE (50 MPa target, 0 events) vs. OFF (0 events) -- "
        "dashed lines mark baseline event times",
        fontsize=10.5, color="0.25")
    fig.tight_layout()

    out_dir = os.path.join(HIGHSTRESS_ROOT, "finite", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "barrier_ladder_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # quantitative summary printed to stdout for the report
    for name, run_dir, _, label in CASES:
        rows = data[name]
        if not rows:
            continue
        last = rows[-1]
        n_events = len(load_events(run_dir))
        print(f"{label:45s} final t={last.get('time'):.2f}  "
              f"sigma_s={last.get('sigma_s_1p5W_MPa'):.4f} MPa  "
              f"X_neck={last.get('X_neck_nm'):.3f} nm  "
              f"r_neck={last.get('r_neck_1p5W_nm'):.3f} nm  "
              f"n_events={n_events}")


if __name__ == "__main__":
    main()
