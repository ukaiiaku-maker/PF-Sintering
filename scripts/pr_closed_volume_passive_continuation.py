#!/usr/bin/env python3
"""Restart the qualified closed-volume passive PR trajectory at t=34.

No source, sink, event, hazard, or avalanche operation is present in this
driver.  The evolution call graph is exactly the one used by
``pr_closed_volume_no_event_t34.py``; this file only adds compact geometric
metrology and the declared t=150/t=300 continuation decision.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from pf_sintering.axisym import axisym_volume  # noqa: E402
from pf_sintering.axisym_numba_kernel import (  # noqa: E402
    NumbaScratch, axisym_gb_face_projected_step_fast,
)
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.experimental_pr_metrology import (  # noqa: E402
    measure_experimental_pr_state,
)
from pr_closed_volume_no_event_t34 import (  # noqa: E402
    build_case, contour_volume, make_tracker_evaluator, outer_radius,
)


SOURCE = ROOT / "runs/pr_closed_volume_no_event_t34_eps0p25_12W"
OUT = Path(os.environ.get(
    "PR_CLOSED_CONTINUATION_OUT",
    str(ROOT / "runs/pr_closed_volume_passive_continuation_eps0p25_12W")))
SOURCE_CHECKPOINT = SOURCE / "checkpoint_latest.npz"
SOURCE_HISTORY = SOURCE / "closed_volume_history.csv"
SOURCE_FIELDS = SOURCE / "closed_volume_fields.h5"

SECONDS_PER_MODEL_TIME = 0.01557994316955921
INITIAL_TARGET_MODEL = 150.0
EXTENDED_TARGET_MODEL = 300.0
ANALYSIS_DT_MODEL = 0.25
CHECKPOINT_DT_MODEL = 0.50
INTERNAL_AUDIT_STRIDE = 256
VOLUME_TOLERANCE = 1.0e-8
PHASE_CLOSURE_TOLERANCE = 1.0e-12
BOUNDARY_STOP_W = 6.0
RESOLVED_RELATIVE_AMPLITUDE_CHANGE = 5.0e-3
RESOLVED_STRESS_CHANGE_PA = 1.0e5
TREND_START_MODEL = 75.0
SNAPSHOT_TIMES = (0.0, 34.0, 75.0, 150.0, 300.0)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class CompactContourArchive:
    def __init__(self, path: Path, nz: int, *, resume: bool):
        self.handle = h5py.File(path, "a" if resume else "w")
        if not resume:
            self.handle.attrs.update(
                source_checkpoint=str(SOURCE_CHECKPOINT),
                system_mass_boundary="closed",
                external_reservoir_mass_exchange="disabled",
                analysis_dt_model=ANALYSIS_DT_MODEL,
                seconds_per_model_time=SECONDS_PER_MODEL_TIME,
                contour_level=0.5,
                profile_branch="fixed original Fourier interval crest-to-crest")
            self.handle.create_dataset(
                "radius_m", shape=(0, nz), maxshape=(None, nz),
                chunks=(1, nz), dtype="f4", compression="lzf")
            for name in ("t_model", "t_s"):
                self.handle.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype="f8")

    def append(self, radius: np.ndarray, t_model: float) -> None:
        n = len(self.handle["t_model"])
        ds = self.handle["radius_m"]
        ds.resize(n + 1, axis=0)
        ds[n] = np.asarray(radius, dtype=np.float32)
        for name, value in (
                ("t_model", t_model),
                ("t_s", t_model * SECONDS_PER_MODEL_TIME)):
            ds = self.handle[name]
            ds.resize(n + 1, axis=0)
            ds[n] = value
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def profile_metrics(radius: np.ndarray, setup: dict, geom: dict) -> dict:
    """Robust metrics on the fixed original Fourier crest-to-crest branch."""
    z = np.asarray(setup["z"], dtype=float)
    z0 = -float(geom["z_raw_min_m"])
    z1 = z0 + float(geom["lam"])
    use = np.isfinite(radius) & (z >= z0) & (z <= z1)
    zz = z[use]
    rr = np.asarray(radius, dtype=float)[use]
    if len(rr) < 21:
        raise RuntimeError("fixed Fourier branch has too few contour points")

    # A ~2W window removes cell-to-cell contour interpolation noise while
    # remaining much shorter than lambda.  Raw radii remain archived.
    window = max(7, int(round(2.0 * setup["W"] / setup["dz"])))
    if window % 2 == 0:
        window += 1
    window = min(window, len(rr) - (1 - len(rr) % 2))
    smooth = savgol_filter(rr, window_length=window, polyorder=3,
                           mode="interp")
    rbar = float(np.trapezoid(smooth, zz) / (zz[-1] - zz[0]))

    phase = 2.0 * math.pi * (zz - z0) / float(geom["lam"])
    design = np.column_stack([
        np.ones_like(phase), np.cos(phase), np.sin(phase)])
    coefficient, *_ = np.linalg.lstsq(design, smooth, rcond=None)
    a_cos = float(coefficient[1])
    a_sin = float(coefficient[2])

    center = float(geom["z1"])
    neck_use = np.abs(zz - center) <= 0.25 * float(geom["lam"])
    left_lobe = zz <= center - 0.15 * float(geom["lam"])
    right_lobe = zz >= center + 0.15 * float(geom["lam"])
    r_neck = float(np.min(smooth[neck_use]))
    r_lobe_left = float(np.max(smooth[left_lobe]))
    r_lobe_right = float(np.max(smooth[right_lobe]))
    r_bulge = 0.5 * (r_lobe_left + r_lobe_right)
    r_min = float(np.min(smooth))
    r_max = float(np.max(smooth))
    a_pr = 0.5 * (r_max - r_min)
    return dict(
        r_min_m=r_min, r_max_m=r_max, A_PR_m=a_pr,
        r_bar_branch_m=rbar, A_PR_over_r_bar=a_pr / rbar,
        A1_cos_m=a_cos, A1_sin_m=a_sin,
        A1_magnitude_m=float(math.hypot(a_cos, a_sin)),
        r_neck_smooth_m=r_neck, r_lobe_left_m=r_lobe_left,
        r_lobe_right_m=r_lobe_right, r_bulge_m=r_bulge,
        neck_to_bulge_ratio=r_neck / r_bulge,
        profile_smoothing_window_points=window,
        profile_smoothing_window_m=window * float(setup["dz"]),
        profile_z_start_m=float(zz[0]), profile_z_end_m=float(zz[-1]))


def late_fit(rows: list[dict], final_t: float) -> dict:
    start = max(TREND_START_MODEL,
                final_t - (150.0 if final_t > 150.0 else 75.0))
    selected = [row for row in rows if start <= row["t_model"] <= final_t]
    if len(selected) < 5:
        raise RuntimeError("insufficient late-time samples for trend fit")
    t = np.asarray([row["t_model"] for row in selected], dtype=float)

    def slope(name: str) -> float:
        return float(np.polyfit(t, [row[name] for row in selected], 1)[0])

    amplitude = np.asarray(
        [max(row["A1_magnitude_m"], 1e-300) for row in selected])
    omega = float(np.polyfit(t, np.log(amplitude), 1)[0])
    first, last = selected[0], selected[-1]
    return dict(
        fit_start_model=float(t[0]), fit_end_model=float(t[-1]),
        fit_samples=len(selected), omega_PR_per_model_time=omega,
        omega_PR_per_s=omega / SECONDS_PER_MODEL_TIME,
        A1_relative_change=(
            last["A1_magnitude_m"] / first["A1_magnitude_m"] - 1.0),
        A_PR_relative_change=(last["A_PR_m"] / first["A_PR_m"] - 1.0),
        sigma_integral_change_Pa=(
            last["sigma_integral_Pa"] - first["sigma_integral_Pa"]),
        sigma_local_change_Pa=(
            last["sigma_local_Pa"] - first["sigma_local_Pa"]),
        d_sigma_integral_Pa_per_model_time=slope("sigma_integral_Pa"),
        d_sigma_integral_Pa_per_s=(
            slope("sigma_integral_Pa") / SECONDS_PER_MODEL_TIME),
        d_sigma_local_Pa_per_model_time=slope("sigma_local_Pa"),
        d_sigma_local_Pa_per_s=(
            slope("sigma_local_Pa") / SECONDS_PER_MODEL_TIME),
        d_A_PR_m_per_model_time=slope("A_PR_m"))


def classify(rows: list[dict], final_t: float) -> tuple[str, dict]:
    fit = late_fit(rows, final_t)
    amplitude_change = fit["A1_relative_change"]
    integral_change = fit["sigma_integral_change_Pa"]
    local_change = fit["sigma_local_change_Pa"]
    amplitude_grows = (
        amplitude_change >= RESOLVED_RELATIVE_AMPLITUDE_CHANGE
        and fit["omega_PR_per_model_time"] > 0.0)
    amplitude_decays = (
        amplitude_change <= -RESOLVED_RELATIVE_AMPLITUDE_CHANGE
        and fit["omega_PR_per_model_time"] < 0.0)
    integral_grows = integral_change >= RESOLVED_STRESS_CHANGE_PA
    local_grows = local_change >= RESOLVED_STRESS_CHANGE_PA
    integral_declines = integral_change <= -RESOLVED_STRESS_CHANGE_PA

    if amplitude_grows and integral_grows and local_grows:
        decision = "CASE_A_PR_AMPLITUDE_AND_STRESS_GROW"
    elif amplitude_grows and integral_declines:
        decision = "CASE_B_PR_AMPLITUDE_GROWS_STRESS_DOES_NOT"
    elif amplitude_decays:
        decision = "CASE_C_PR_AMPLITUDE_DECAYS"
    elif final_t < EXTENDED_TARGET_MODEL:
        decision = "CONTINUE_SMOOTH_UNRESOLVED_TO_T300"
    else:
        decision = "CASE_D_UNRESOLVED_OR_ESSENTIALLY_UNCHANGED_AT_T300"
    fit.update(
        classification=decision, amplitude_grows=amplitude_grows,
        amplitude_decays=amplitude_decays,
        integral_stress_grows=integral_grows,
        local_stress_grows=local_grows,
        integral_stress_declines=integral_declines)
    return decision, fit


def initialize_source_history(setup: dict, geom: dict):
    with SOURCE_HISTORY.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    rows = []
    with h5py.File(SOURCE_FIELDS, "r") as source:
        radii = np.asarray(source["radius_m"], dtype=float)
        times = np.asarray(source["t_model"], dtype=float)
    if len(source_rows) != len(times):
        raise RuntimeError("source scalar and contour archives are misaligned")
    for raw, t_model, radius in zip(source_rows, times, radii):
        row = dict(
            phase="source_closed_t0_to_t34", t_model=float(t_model),
            t_s=float(t_model * SECONDS_PER_MODEL_TIME),
            step=float(raw["step"]), Vf_m3=float(raw["Vf_m3"]),
            V1_m3=float(raw["V1_m3"]), V2_m3=float(raw["V2_m3"]),
            partition_volume_error_m3=float(raw["partition_volume_error_m3"]),
            closure_max=float(raw["closure_max"]),
            Vcontour_m3=float(raw["Vcontour_m3"]),
            Vf_relative=float(raw["Vf_relative"]),
            Vcontour_relative=float(raw["Vcontour_relative"]),
            radial_margin_m=float(raw["radial_margin_m"]),
            left_z_margin_m=float(raw["left_z_margin_m"]),
            right_z_margin_m=float(raw["right_z_margin_m"]),
            boundary_f_max=float(raw["boundary_f_max"]),
            z_TJ_m=float(raw["z_TJ_m"]),
            r_TJ_m=float(raw["r_TJ_m"]), tj_candidate_count=1,
            tj_jump_m=0.0,
            sigma_local_Pa=float(raw["sigma_local_Pa"]),
            sigma_integral_Pa=float(raw["sigma_integral_Pa"]),
            **profile_metrics(radius, setup, geom))
        rows.append(row)
    return rows, radii


def save_snapshot(path: Path, state, *, step: int, t_model: float,
                  V0_m3: float) -> None:
    atomic_npz(
        path, f=state[0], e1=state[1], e2=state[2], step=step,
        t_model=t_model, V0_m3=V0_m3,
        system_mass_boundary=np.asarray("closed"),
        external_reservoir_mass_exchange=np.asarray("disabled"))


def diagnostic_figure(rows: list[dict], setup: dict) -> None:
    t = np.asarray([row["t_model"] for row in rows])
    archive_path = OUT / "passive_contours.h5"
    with h5py.File(archive_path, "r") as archive:
        archive_t = np.asarray(archive["t_model"])
        radii = archive["radius_m"]
        wanted = [value for value in SNAPSHOT_TIMES if value <= t[-1] + 1e-9]
        snapshot_indices = [int(np.argmin(np.abs(archive_t - value)))
                            for value in wanted]

        fig = plt.figure(figsize=(15, 16), constrained_layout=True)
        grid = fig.add_gridspec(6, max(len(wanted), 1), height_ratios=[1.2, 1, 1, 1, 1, 1])
        initial_radius = np.asarray(radii[0], dtype=float) * 1e9
        z_nm = np.asarray(setup["z"]) * 1e9
        for column, (target, index) in enumerate(zip(wanted, snapshot_indices)):
            ax = fig.add_subplot(grid[0, column])
            radius = np.asarray(radii[index], dtype=float) * 1e9
            ax.plot(z_nm, initial_radius, "k--", lw=.7, alpha=.35)
            ax.plot(z_nm, radius, color="tab:blue", lw=1.1)
            ax.fill_between(z_nm, 0.0, radius, color="tab:blue", alpha=.12)
            row = rows[index]
            ax.plot(row["z_TJ_m"] * 1e9, row["r_TJ_m"] * 1e9,
                    "ro", ms=3)
            ax.set_xlim(0.0, setup["z"][-1] * 1e9)
            ax.set_ylim(0.0, setup["r_f"][-1] * 1e9)
            ax.set_aspect("equal")
            ax.set_title(f"t={archive_t[index]:.0f}")
            ax.set_xlabel("z (nm)")
            if column == 0:
                ax.set_ylabel("r (nm)")

    axes = [fig.add_subplot(grid[row_index, :]) for row_index in range(1, 6)]
    axes[0].plot(t, np.asarray([row["A1_cos_m"] for row in rows]) * 1e9,
                 label=r"$A_1$ cosine")
    axes[0].plot(t, np.asarray([row["A_PR_over_r_bar"] for row in rows]),
                 label=r"$A_{PR}/\bar r$")
    axes[0].set_ylabel("mode amplitude")
    axes[0].legend(ncol=2)

    axes[1].plot(t, np.asarray([row["r_neck_smooth_m"] for row in rows]) * 1e9,
                 label="smooth neck")
    axes[1].plot(t, np.asarray([row["r_bulge_m"] for row in rows]) * 1e9,
                 label="bulge")
    axes[1].plot(t, np.asarray([row["r_TJ_m"] for row in rows]) * 1e9,
                 label="field TJ")
    axes[1].set_ylabel("radius (nm)")
    axes[1].legend(ncol=3)

    local = np.asarray([row["sigma_local_Pa"] for row in rows]) / 1e6
    integral = np.asarray([row["sigma_integral_Pa"] for row in rows]) / 1e6
    window = min(101, len(t) - (1 - len(t) % 2))
    if window >= 7:
        axes[2].plot(t, local, color="tab:blue", alpha=.22, lw=.7)
        axes[2].plot(t, integral, color="tab:orange", alpha=.22, lw=.7)
        axes[2].plot(t, savgol_filter(local, window, 3), color="tab:blue",
                     label="local 3W (trend only)")
        axes[2].plot(t, savgol_filter(integral, window, 3), color="tab:orange",
                     label="integral (trend only)")
    else:
        axes[2].plot(t, local, label="local 3W")
        axes[2].plot(t, integral, label="integral")
    axes[2].set_ylabel("stress (MPa)")
    axes[2].legend(ncol=2)

    axes[3].plot(t, [row["Vf_relative"] for row in rows], label="field volume")
    axes[3].plot(t, [row["Vcontour_relative"] for row in rows],
                 label="contour volume")
    axes[3].axhline(0.0, color="k", lw=.7)
    axes[3].set_ylabel("relative volume error")
    axes[3].legend(ncol=2)

    axes[4].plot(t, np.asarray([row["z_TJ_m"] for row in rows]) * 1e9,
                 label="TJ z")
    axes[4].plot(t, np.asarray([row["r_TJ_m"] for row in rows]) * 1e9,
                 label="TJ r")
    axes[4].set_ylabel("TJ coordinate (nm)")
    axes[4].set_xlabel("model time")
    axes[4].legend(ncol=2)
    secondary = axes[0].secondary_xaxis(
        "top", functions=(lambda value: value * SECONDS_PER_MODEL_TIME,
                          lambda value: value / SECONDS_PER_MODEL_TIME))
    secondary.set_xlabel("mapped physical time (s)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "closed_passive_PR_diagnostic.png", dpi=180)
    fig.savefig(OUT / "closed_passive_PR_diagnostic.pdf")
    plt.close(fig)


def main() -> None:
    set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    geom, setup = build_case()
    manifest = dict(
        purpose="closed-volume passive PR direction and timescale",
        source_checkpoint=str(SOURCE_CHECKPOINT), source_t_model=34.0,
        restart_not_reconstructed=True,
        system_mass_boundary="closed",
        external_reservoir_mass_exchange="disabled",
        total_solid_volume_conserved=True,
        explicit_source_sink=False, nucleation_enabled=False,
        event_enabled=False, hazard_enabled=False, avalanche_enabled=False,
        evolution="same conservative no-flux PF and ownership operator as t0-to-t34",
        bounded_conservative_projection=True,
        epsilon1=0.25, epsilon2=0.0,
        lambda_m=float(geom["lam"]),
        lambda_over_R_cyl=float(geom["lambda_over_R_cyl"]),
        grid_shape=list(geom["f"].shape), dr_m=setup["dr"], dz_m=setup["dz"],
        W_m=setup["W"], tail_margin_W=12.0, radial_margin_W=12.0,
        initial_target_model=INITIAL_TARGET_MODEL,
        extended_target_model=EXTENDED_TARGET_MODEL,
        volume_tolerance=VOLUME_TOLERANCE,
        phase_closure_tolerance=PHASE_CLOSURE_TOLERANCE,
        boundary_stop_clearance_W=BOUNDARY_STOP_W,
        local_stress_window_W=3.0,
        analysis_dt_model=ANALYSIS_DT_MODEL,
        checkpoint_dt_model=CHECKPOINT_DT_MODEL,
        seconds_per_model_time=SECONDS_PER_MODEL_TIME)
    atomic_json(OUT / "launch_manifest.json", manifest)

    checkpoint = OUT / "checkpoint_latest.npz"
    history_path = OUT / "passive_continuation_history.csv"
    contour_path = OUT / "passive_contours.h5"
    if checkpoint.exists() and history_path.exists() and contour_path.exists():
        with np.load(checkpoint) as saved:
            state = tuple(np.asarray(saved[key]).copy()
                          for key in ("f", "e1", "e2"))
            step = int(saved["step"])
            V0 = float(saved["V0_m3"])
        with history_path.open(newline="") as stream:
            rows = [{key: (value if key == "phase" else float(value))
                     for key, value in row.items()}
                    for row in csv.DictReader(stream)]
        archive = CompactContourArchive(
            contour_path, state[0].shape[0], resume=True)
    else:
        with np.load(SOURCE_CHECKPOINT) as saved:
            state = tuple(np.asarray(saved[key]).copy()
                          for key in ("f", "e1", "e2"))
            step = int(saved["step"])
            V0 = float(saved["V0_m3"])
        if not math.isclose(step * setup["dt"], 34.0, abs_tol=1e-12):
            raise RuntimeError("source checkpoint is not the exact t=34 state")
        rows, source_radii = initialize_source_history(setup, geom)
        archive = CompactContourArchive(
            contour_path, state[0].shape[0], resume=False)
        for row, radius in zip(rows, source_radii):
            archive.append(radius, row["t_model"])
        write_csv(history_path, rows)
        with h5py.File(SOURCE_FIELDS, "r") as source:
            state0 = tuple(np.asarray(source[key][0], dtype=float)
                           for key in ("f", "e1", "e2"))
        save_snapshot(
            OUT / "snapshots/snapshot_t000.npz", state0,
            step=0, t_model=0.0, V0_m3=V0)
        save_snapshot(
            OUT / "snapshots/snapshot_t034.npz", state,
            step=step, t_model=34.0, V0_m3=V0)

    previous = (rows[-1]["z_TJ_m"], rows[-1]["r_TJ_m"])
    evaluator = make_tracker_evaluator(setup, geom, previous)
    projector = ConservativeBoundedPhaseProjector()
    scratches = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
    current = step % 2
    analysis_steps = max(1, int(round(ANALYSIS_DT_MODEL / setup["dt"])))
    checkpoint_steps = max(1, int(round(CHECKPOINT_DT_MODEL / setup["dt"])))
    target_model = (EXTENDED_TARGET_MODEL
                    if step * setup["dt"] >= INITIAL_TARGET_MODEL
                    and not (OUT / "decision_at_t150.json").exists()
                    else INITIAL_TARGET_MODEL)
    if (OUT / "decision_at_t150.json").exists():
        decision150 = json.loads((OUT / "decision_at_t150.json").read_text())
        if decision150["classification"] == "CONTINUE_SMOOTH_UNRESOLVED_TO_T300":
            target_model = EXTENDED_TARGET_MODEL
    started = time.monotonic()

    def fail(reason: str, row: dict | None = None) -> None:
        save_snapshot(
            OUT / "solver_failure_checkpoint.npz", state, step=step,
            t_model=step * setup["dt"], V0_m3=V0)
        atomic_json(OUT / "run_state.json", dict(
            status="STOPPED_ON_GENUINE_NUMERICAL_FAILURE", reason=reason,
            latest=row, wall_seconds=time.monotonic() - started))
        raise RuntimeError(reason)

    def observe() -> dict:
        f, e1, e2 = state
        try:
            scalar, _, node = measure_experimental_pr_state(
                state, setup, evaluator)
        except Exception as error:
            fail(f"loss of continuous field TJ: {type(error).__name__}: {error}")
        tracking = node.get("tj_tracking", {})
        if int(tracking.get("candidate_count", 0)) != 1:
            fail("field TJ is no longer unique")
        radius = outer_radius(f, setup["r_c"])
        finite = np.flatnonzero(np.isfinite(radius))
        if len(finite) < 2:
            fail("exterior f=0.5 contour is not recoverable")
        Vf = axisym_volume(f, setup["r_c"], setup["dr"], setup["dz"])
        V1 = axisym_volume(e1, setup["r_c"], setup["dr"], setup["dz"])
        V2 = axisym_volume(e2, setup["r_c"], setup["dr"], setup["dz"])
        closure = float(np.max(np.abs(e1 + e2 - f)))
        margins = dict(
            radial_margin_m=float(setup["r_f"][-1] - np.nanmax(radius)),
            left_z_margin_m=float(
                setup["z"][finite[0]] - 0.5 * setup["dz"]),
            right_z_margin_m=float(
                setup["z"][-1] + 0.5 * setup["dz"]
                - setup["z"][finite[-1]]))
        row = dict(
            phase="continued_closed_passive",
            t_model=step * setup["dt"],
            t_s=step * setup["dt"] * SECONDS_PER_MODEL_TIME,
            step=float(step), Vf_m3=Vf, V1_m3=V1, V2_m3=V2,
            partition_volume_error_m3=V1 + V2 - Vf,
            closure_max=closure, Vcontour_m3=contour_volume(setup["z"], radius),
            Vf_relative=Vf / V0 - 1.0,
            Vcontour_relative=(
                contour_volume(setup["z"], radius)
                / rows[0]["Vcontour_m3"] - 1.0),
            **margins,
            boundary_f_max=float(max(
                np.max(f[0]), np.max(f[-1]), np.max(f[:, -1]))),
            z_TJ_m=float(scalar["z_TJ_m"]),
            r_TJ_m=float(scalar["r_TJ_m"]),
            tj_candidate_count=int(tracking["candidate_count"]),
            tj_jump_m=float(tracking["jump_m"]),
            sigma_local_Pa=float(scalar["sigma_local_Pa"]),
            sigma_integral_Pa=float(scalar["sigma_integral_Pa"]),
            **profile_metrics(radius, setup, geom))
        if abs(row["Vf_relative"]) >= VOLUME_TOLERANCE:
            fail("closed total-solid-volume invariant failed", row)
        if closure >= PHASE_CLOSURE_TOLERANCE:
            fail("phase closure invariant failed", row)
        if min(margins.values()) <= BOUNDARY_STOP_W * setup["W"]:
            fail("exterior interface approached padded-box boundary", row)
        rows.append(row)
        write_csv(history_path, rows)
        archive.append(radius, row["t_model"])
        atomic_json(OUT / "run_state.json", dict(
            status="RUNNING", target_model=target_model, latest=row,
            wall_seconds=time.monotonic() - started,
            projector=projector.manifest()))
        print(
            "PASSIVE",
            f"t={row['t_model']:.2f}",
            f"A1={row['A1_cos_m'] * 1e9:+.5f}nm",
            f"APR/rbar={row['A_PR_over_r_bar']:.7f}",
            f"neck/bulge={row['neck_to_bulge_ratio']:.7f}",
            f"sigmaI={row['sigma_integral_Pa'] / 1e6:.5f}MPa",
            f"sigmaL={row['sigma_local_Pa'] / 1e6:.5f}MPa",
            f"dV/V={row['Vf_relative']:+.2e}", flush=True)
        return row

    outcome = "RUNNING"
    decision = None
    fit = None
    try:
        while True:
            final_steps = int(round(target_model / setup["dt"]))
            while step < final_steps:
                destination = 0 if current != 0 else 1
                state = axisym_gb_face_projected_step_fast(
                    *state, setup["p"], setup["Wc"], setup["dr"],
                    setup["dz"], setup["r_c"], setup["r_f"], setup["dt"],
                    setup["M_s"], setup["M_eta"], setup["W"],
                    scratches[destination])
                current = destination
                step += 1
                state, _ = projector(state, setup)
                if step % INTERNAL_AUDIT_STRIDE == 0:
                    volume = axisym_volume(
                        state[0], setup["r_c"], setup["dr"], setup["dz"])
                    if abs(volume / V0 - 1.0) >= VOLUME_TOLERANCE:
                        fail("internal closed total-solid-volume invariant failed")
                    closure = float(np.max(np.abs(state[1] + state[2] - state[0])))
                    if closure >= PHASE_CLOSURE_TOLERANCE:
                        fail("internal phase closure invariant failed")
                if step % analysis_steps == 0 or step == final_steps:
                    row = observe()
                    for target in SNAPSHOT_TIMES:
                        path = OUT / "snapshots" / f"snapshot_t{int(target):03d}.npz"
                        if (not path.exists()
                                and row["t_model"] >= target - 0.5 * setup["dt"]):
                            save_snapshot(
                                path, state, step=step,
                                t_model=row["t_model"], V0_m3=V0)
                if step % checkpoint_steps == 0 or step == final_steps:
                    save_snapshot(
                        checkpoint, state, step=step,
                        t_model=step * setup["dt"], V0_m3=V0)

            decision, fit = classify(rows, target_model)
            atomic_json(
                OUT / f"decision_at_t{int(target_model)}.json", fit)
            if (target_model == INITIAL_TARGET_MODEL
                    and decision == "CONTINUE_SMOOTH_UNRESOLVED_TO_T300"):
                target_model = EXTENDED_TARGET_MODEL
                print("AUTO-CONTINUE t=150 decision unresolved; extending to t=300",
                      json.dumps(fit), flush=True)
                continue
            outcome = decision
            break
    finally:
        archive.close()

    diagnostic_figure(rows, setup)
    result = dict(
        outcome=outcome, final_t_model=rows[-1]["t_model"],
        final_t_s=rows[-1]["t_s"], late_time_fit=fit,
        maximum_abs_field_volume_relative_drift=max(
            abs(row["Vf_relative"]) for row in rows),
        maximum_phase_closure=max(row["closure_max"] for row in rows),
        minimum_boundary_margin_m=min(
            min(row["radial_margin_m"], row["left_z_margin_m"],
                row["right_z_margin_m"]) for row in rows),
        initial_A1_m=rows[0]["A1_cos_m"], final_A1_m=rows[-1]["A1_cos_m"],
        initial_A_PR_over_r_bar=rows[0]["A_PR_over_r_bar"],
        final_A_PR_over_r_bar=rows[-1]["A_PR_over_r_bar"],
        initial_sigma_local_Pa=rows[0]["sigma_local_Pa"],
        final_sigma_local_Pa=rows[-1]["sigma_local_Pa"],
        initial_sigma_integral_Pa=rows[0]["sigma_integral_Pa"],
        final_sigma_integral_Pa=rows[-1]["sigma_integral_Pa"],
        wall_seconds=time.monotonic() - started)
    atomic_json(OUT / "passive_continuation_result.json", result)
    atomic_json(OUT / "run_state.json", dict(status="COMPLETE", result=result))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
