#!/usr/bin/env python3
"""Post-process the matched avalanche, always-on, and never-on trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MAIN_DEFAULT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_five_deltaG0p300_taucorr9ms")
CONTROL_DEFAULT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_limit_controls_deltaG0p300")


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as archive:
        data = {
            "t_ms": np.asarray(archive["time/t_s"], dtype=float) * 1e3,
            "stress_MPa": np.asarray(
                archive["state/sigma_local_Pa"], dtype=float) / 1e6,
            "Q_over_b": np.asarray(
                archive["state/Q_cumulative_over_b"], dtype=float),
            "r_neck_nm": np.asarray(
                archive["state/r_neck_m"], dtype=float) * 1e9,
            "avalanche_active": np.asarray(
                archive["state/avalanche_active"], dtype=int),
        }
    lengths = {values.size for values in data.values()}
    if lengths == {0} or len(lengths) != 1:
        raise RuntimeError(f"inconsistent or empty archive: {path}")
    for name, values in data.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"nonfinite {name} in {path}")
    if np.any(np.diff(data["t_ms"]) <= 0):
        raise RuntimeError(f"non-increasing time in {path}")
    return data


def active_intervals(t: np.ndarray, active: np.ndarray) -> list[tuple[float, float]]:
    mask = active.astype(bool)
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(float(t[start]), float(t[stop])) for start, stop in zip(starts, stops)]


def endpoint_label(ax, x: float, y: float, label: str, color: str) -> None:
    ax.scatter([x], [y], s=22, color=color, zorder=5)
    ax.annotate(
        label, (x, y), xytext=(5, 0), textcoords="offset points",
        va="center", fontsize=8, color=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, default=MAIN_DEFAULT)
    parser.add_argument("--control-root", type=Path, default=CONTROL_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.control_root
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "Avalanche renewal": (
            args.main_dir / "avalanche_renewal_movie_geometry.h5"),
        "Always-on sink": (
            args.control_root / "always_on_diffusion/control_movie_geometry.h5"),
        "Never-on sink": (
            args.control_root / "never_on_PR/control_movie_geometry.h5"),
    }
    series = {name: load_archive(path) for name, path in paths.items()}
    colors = {
        "Avalanche renewal": "#174a73",
        "Always-on sink": "#b45309",
        "Never-on sink": "#2f6f58",
    }
    fields = (
        ("stress_MPa", "Local stress (MPa)"),
        ("Q_over_b", r"Cumulative displacement $Q/b$"),
        ("r_neck_nm", "Neck radius (nm)"),
    )
    fig, axes = plt.subplots(
        3, 1, figsize=(12.2, 9.2), sharex=True, constrained_layout=True)
    main_series = series["Avalanche renewal"]
    intervals = active_intervals(
        main_series["t_ms"], main_series["avalanche_active"])
    for ax, (field, ylabel) in zip(axes, fields):
        for start, stop in intervals:
            ax.axvspan(start, stop, color="#c65f37", alpha=.12, linewidth=0)
        for name, values in series.items():
            ax.plot(
                values["t_ms"], values[field], color=colors[name],
                linewidth=2.0 if name == "Avalanche renewal" else 1.7,
                label=name)
            endpoint_label(
                ax, float(values["t_ms"][-1]), float(values[field][-1]),
                f"{values['t_ms'][-1]:.3g} ms", colors[name])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, ncol=3, loc="best")
    axes[-1].set_xlabel("Physical time from common initial geometry (ms)")
    axes[-1].set_xlim(left=0.0)
    fig.suptitle(
        "Stochastic avalanche renewal and matched sink-limit controls\n"
        "shading: main-trajectory facilitated/avalanche-active interval")
    png = output / "avalanche_limit_controls_comparison.png"
    pdf = output / "avalanche_limit_controls_comparison.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "source_archives": {name: str(path) for name, path in paths.items()},
        "main_avalanche_active_intervals_ms": intervals,
        "trajectories": {
            name: {
                "frames": int(values["t_ms"].size),
                "t_end_ms": float(values["t_ms"][-1]),
                "Q_end_over_b": float(values["Q_over_b"][-1]),
                "sigma_local_end_MPa": float(values["stress_MPa"][-1]),
                "r_neck_end_nm": float(values["r_neck_nm"][-1]),
            }
            for name, values in series.items()
        },
        "outputs": {"png": str(png), "pdf": str(pdf)},
    }
    (output / "avalanche_limit_controls_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
