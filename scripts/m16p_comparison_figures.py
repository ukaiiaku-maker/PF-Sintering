"""M16P Section 25: comparison figures across the 2x2 (geometry x barrier)
matrix, plus Section 24's unified comparison table."""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUN_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16p_barrier_run")
OUT_DIR = os.path.join(RUN_ROOT, "comparison_figures")

RUNS = {
    "A: loading+finite": "chi1.5_ratio0.185_finite",
    "B: loading+zero": "chi1.5_ratio0.185_zero",
    "C: relaxing+finite": "chi0.5_ratio0.185_finite",
    "D: relaxing+zero": "chi0.5_ratio0.185_zero",
}


def load(label):
    run_dir = os.path.join(RUN_ROOT, RUNS[label])
    hist_path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(hist_path):
        return None
    with open(hist_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return dict(
        t=np.array([float(r["time"]) for r in rows]),
        sigma=np.array([float(r["sigma_MPa"]) for r in rows]),
        r_neck=np.array([float(r.get("r_neck_nm", "nan") or "nan") for r in rows]) if "r_neck_nm" in rows[0] else None,
        X_neck=np.array([float(r["X_neck_nm"]) for r in rows]),
        n_active=np.array([int(r["N_active"]) for r in rows]),
        n_born=np.array([int(r["N_born"]) for r in rows]),
        n_completed=np.array([int(r["N_completed"]) for r in rows]),
        cum_sink_over_b=np.array([float(r["cumulative_delta_sink_over_b"]) for r in rows]),
        strain=np.array([float(r["cumulative_delta_sink_over_b"]) for r in rows]) * 2.5e-10 /
               np.maximum(np.array([float(r["a_contact_nm"]) for r in rows]) * 1e-9 * 2, 1e-30),
        mass_drift=np.array([float(r["mass_drift"]) for r in rows]),
    )


def load_meta(label):
    path = os.path.join(RUN_ROOT, RUNS[label], "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {label: load(label) for label in RUNS}
    available = {k: v for k, v in data.items() if v is not None}
    print(f"Available runs: {list(available.keys())}")
    if not available:
        print("No runs available yet -- nothing to plot.")
        return

    colors = {"A: loading+finite": "tab:red", "B: loading+zero": "tab:orange",
              "C: relaxing+finite": "tab:blue", "D: relaxing+zero": "tab:cyan"}

    # A: sigma(t)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, d in available.items():
        ax.plot(d["t"], d["sigma"], label=label, color=colors.get(label))
    ax.set_xlabel("time"); ax.set_ylabel("sigma (MPa)"); ax.legend(); ax.set_title("M16P Figure A: sigma(t), all runs")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "A_sigma_vs_time.png"), dpi=150); plt.close(fig)

    # B: r_neck(t) (only available for runs generated after the r_neck_nm
    # logging fix -- Run A/B predate it and will be skipped here)
    fig, ax = plt.subplots(figsize=(9, 5))
    any_r = False
    for label, d in available.items():
        if d["r_neck"] is not None and np.any(np.isfinite(d["r_neck"])):
            ax.plot(d["t"], d["r_neck"], label=label, color=colors.get(label))
            any_r = True
    ax.set_xlabel("time"); ax.set_ylabel("r_neck (nm)")
    if any_r:
        ax.legend()
    ax.set_title("M16P Figure B: r_neck(t)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "B_rneck_vs_time.png"), dpi=150); plt.close(fig)

    # C: X_neck(t)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, d in available.items():
        ax.plot(d["t"], d["X_neck"], label=label, color=colors.get(label))
    ax.set_xlabel("time"); ax.set_ylabel("X_neck (nm)"); ax.legend(); ax.set_title("M16P Figure C: X_neck(t)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "C_Xneck_vs_time.png"), dpi=150); plt.close(fig)

    # D: cumulative delta_sink/b vs time
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, d in available.items():
        ax.plot(d["t"], d["cum_sink_over_b"], label=label, color=colors.get(label))
    ax.set_xlabel("time"); ax.set_ylabel("cumulative delta_sink / b"); ax.legend()
    ax.set_title("M16P Figure D: cumulative delta_sink/b vs time")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "D_cumsink_vs_time.png"), dpi=150); plt.close(fig)

    # F: N_active(t) for finite-barrier cases
    fig, ax = plt.subplots(figsize=(9, 4))
    for label in ["A: loading+finite", "C: relaxing+finite"]:
        if label in available:
            ax.step(available[label]["t"], available[label]["n_active"], where="post", label=label,
                    color=colors.get(label))
    ax.set_xlabel("time"); ax.set_ylabel("N_active"); ax.legend()
    ax.set_title("M16P Figure F: N_active(t), finite-barrier cases")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "F_Nactive_vs_time.png"), dpi=150); plt.close(fig)

    # G: sigma vs cumulative delta_sink/b
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, d in available.items():
        ax.plot(d["cum_sink_over_b"], d["sigma"], label=label, color=colors.get(label))
    ax.set_xlabel("cumulative delta_sink / b"); ax.set_ylabel("sigma (MPa)"); ax.legend()
    ax.set_title("M16P Figure G: sigma vs cumulative delta_sink/b")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "G_sigma_vs_cumsink.png"), dpi=150); plt.close(fig)

    # H: Run A with birth/completion markers
    if "A: loading+finite" in available:
        d = available["A: loading+finite"]
        events_path = os.path.join(RUN_ROOT, RUNS["A: loading+finite"], "events.jsonl")
        births, completes = [], []
        if os.path.exists(events_path):
            with open(events_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        ev = json.loads(line)
                        (births if ev["kind"] == "birth" else completes).append(ev)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(d["t"], d["sigma"], color="tab:red", lw=1.0)
        if births:
            ax.scatter([e["t"] for e in births], [e["sigma_MPa"] for e in births], marker="^", color="green", s=30, zorder=5, label="birth")
        if completes:
            ax.scatter([e["t"] for e in completes], [e["sigma_MPa"] for e in completes], marker="v", color="black", s=30, zorder=5, label="completion")
        ax.set_xlabel("time"); ax.set_ylabel("sigma (MPa)"); ax.legend()
        ax.set_title("M16P Figure H: Run A (loading+finite) with birth/completion markers")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "H_RunA_events.png"), dpi=150); plt.close(fig)

    # Unified comparison table (Section 24)
    table_rows = []
    for label in RUNS:
        d = data.get(label)
        meta = load_meta(label)
        if d is None:
            table_rows.append(dict(run=label, status="not yet run"))
            continue
        table_rows.append(dict(
            run=label, sigma_initial=d["sigma"][0], sigma_max=float(np.max(d["sigma"])),
            sigma_min=float(np.min(d["sigma"])), delta_sigma_net=d["sigma"][-1] - d["sigma"][0],
            N_born=int(d["n_born"][-1]), N_completed=int(d["n_completed"][-1]),
            max_N_active=int(np.max(d["n_active"])), cum_delta_sink_over_b=float(d["cum_sink_over_b"][-1]),
            X_neck_initial=d["X_neck"][0], X_neck_final=d["X_neck"][-1],
            mass_drift_final=float(d["mass_drift"][-1]),
            wall_time_s=meta.get("wall_time_s") if meta else None,
            stop_reason=meta.get("stop_reason") if meta else None,
        ))
    table_path = os.path.join(OUT_DIR, "unified_comparison_table.csv")
    fieldnames = sorted({k for r in table_rows for k in r.keys()})
    with open(table_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in table_rows:
            w.writerow(r)
    print(f"DONE -- wrote figures and {table_path} to {OUT_DIR}")


if __name__ == "__main__":
    main()
