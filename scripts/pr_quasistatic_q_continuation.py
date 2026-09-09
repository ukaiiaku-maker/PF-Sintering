#!/usr/bin/env python3
"""Deterministic fast-surface/slow-GB continuation from the qualified root.

No stochastic clock is evaluated.  Small increments of the already-qualified
one-b geometric operator advance the disconnection coordinate.  After every
increment, the corrected closed-volume PF operator settles the fast local
surface/TJ response while the slow P--R amplitude is retained and monitored.
The requested slow GB clock is then integrated as ``dt=dq/qdot`` strictly in
post-processing.
"""
from __future__ import annotations

import csv
from dataclasses import replace
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


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_renewal_five as renewal  # noqa: E402
import pr_corrected_normalized_ownership_production as corrected  # noqa: E402
import pr_full_corrected_production_campaign as campaign  # noqa: E402
from pf_sintering.axisym import axisym_volume  # noqa: E402
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.corrected_axisym_dynamics import (  # noqa: E402
    CorrectedNumbaScratch,
    axisym_corrected_normalized_ownership_step_fast,
)
from pf_sintering.corrected_pr_thermodynamics import (  # noqa: E402
    corrected_total_energy,
)
from pf_sintering.experimental_pr_metrology import (  # noqa: E402
    measure_experimental_pr_state,
)
from pf_sintering.quasistatic_event_continuation import (  # noqa: E402
    default_q_schedule,
    fast_manifold_converged,
    fast_manifold_increment,
    integrate_slow_clock,
)
from pr_full_deterministic_cycle import event_call, make_transport  # noqa: E402


SOURCE = ROOT / (
    "runs/pr_full_corrected_production_campaign_8avalanche_v3/"
    "checkpoints/avalanche1_before_root.npz")
OUT = ROOT / "runs/pr_quasistatic_q_continuation_v1"
STATUS = ROOT / (
    "runs/pr_full_corrected_production_campaign_8avalanche_v7_"
    "recover_v6_latest/NEXT_RUN_PARAMETER_STATUS.json")
SECONDS_PER_MODEL_TIME = renewal.SECONDS_PER_MODEL_TIME
FAST_BLOCK_STEPS = 10
FAST_MIN_BLOCKS = 3
FAST_MAX_BLOCKS = 60
FAST_CONSECUTIVE_BLOCKS = 3
FAST_TOLERANCES = {
    "affinity_MPa": 0.03,
    "sigma_local_MPa": 0.005,
    "sigma_integral_MPa": 0.005,
    "A1_nm": 0.001,
    "neck_nm": 0.001,
    "energy_relative": 1.5e-8,
}
VOLUME_TOLERANCE = 1.0e-10
PHASE_CLOSURE_TOLERANCE = 1.0e-12


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_state(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        state = tuple(np.asarray(saved[key]).copy()
                      for key in ("f", "particle", "substrate"))
        t_model = float(saved["t_model"])
    return state, t_model


def scalar_state(state, q_over_b, setup, geom, evaluator, volume0):
    scalar, _, _ = measure_experimental_pr_state(state, setup, evaluator)
    scalar.update(corrected.geometry_metrics(state, setup, geom))
    scalar.update(corrected.transport_diagnostic(
        state, setup, geom, evaluator))
    volume = axisym_volume(state[0], setup["r_c"], setup["dr"], setup["dz"])
    closure = float(np.max(np.abs(state[1] + state[2] - state[0])))
    scalar.update(
        q_over_b=float(q_over_b),
        G_phasefield_J=float(corrected_total_energy(state, setup)),
        total_volume_m3=float(volume),
        total_volume_relative_error=float(volume / volume0 - 1.0),
        phase_closure_max=closure,
        sigma_local_MPa=float(scalar["sigma_local_Pa"]) * 1.0e-6,
        sigma_integral_MPa=float(scalar["sigma_integral_Pa"]) * 1.0e-6,
        transport_affinity_MPa=(
            float(scalar["transport_affinity_Pa"]) * 1.0e-6),
        r_neck_nm=float(scalar["r_neck_smooth_m"]) * 1.0e9,
        r_TJ_nm=float(scalar["r_TJ_m"]) * 1.0e9,
        A1_nm=float(scalar["A1_magnitude_m"]) * 1.0e9)
    return scalar


def assert_invariants(row):
    if abs(row["total_volume_relative_error"]) > VOLUME_TOLERANCE:
        raise RuntimeError(
            "closed-volume invariant failed: "
            f"{row['total_volume_relative_error']:+.3e}")
    if row["phase_closure_max"] > PHASE_CLOSURE_TOLERANCE:
        raise RuntimeError(
            f"phase closure failed: {row['phase_closure_max']:.3e}")
    if int(row["tj_candidate_count"]) != 1:
        raise RuntimeError("unique continuous field TJ was lost")


def relax_fast_manifold(state, q_over_b, setup, geom, volume0):
    """Settle fast local modes without demanding that slow P--R growth vanish."""
    projector = ConservativeBoundedPhaseProjector()
    scratches = [CorrectedNumbaScratch(*state[0].shape) for _ in range(2)]
    which = 0
    evaluator = corrected.evaluator_builder(setup, geom)
    history = [scalar_state(
        state, q_over_b, setup, geom, evaluator, volume0)]
    assert_invariants(history[-1])
    consecutive = 0
    for block in range(1, FAST_MAX_BLOCKS + 1):
        for _ in range(FAST_BLOCK_STEPS):
            destination = 1 - which
            state = axisym_corrected_normalized_ownership_step_fast(
                *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
                setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
                setup["M_eta"], setup["W"], scratches[destination])
            which = destination
            state, _ = projector(state, setup)
            state = tuple(np.asarray(field).copy() for field in state)
        row = scalar_state(state, q_over_b, setup, geom, evaluator, volume0)
        assert_invariants(row)
        increment = fast_manifold_increment(history[-1], row)
        row.update({f"fast_increment_{key}": value
                    for key, value in increment.items()})
        row["fast_relax_blocks"] = block
        row["fast_relax_model_time"] = (
            block * FAST_BLOCK_STEPS * setup["dt"])
        history.append(row)
        if block >= FAST_MIN_BLOCKS and fast_manifold_converged(
                increment, FAST_TOLERANCES):
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= FAST_CONSECUTIVE_BLOCKS:
            row["fast_manifold_converged"] = 1
            return state, row, history, projector.manifest()
    history[-1]["fast_manifold_converged"] = 0
    return state, history[-1], history, projector.manifest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key, value in row.items()
                     if np.isscalar(value) and not isinstance(value, str)})
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields}
                         for row in rows)
    os.replace(temporary, path)


def render(rows):
    q = np.asarray([row["q_over_b"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(q, [r["transport_affinity_MPa"] for r in rows], "o-")
    axes[0, 0].axhline(0.0, color="k", ls="--", lw=1)
    axes[0, 0].set_ylabel("GB-to-TJ affinity (MPa)")
    axes[0, 1].plot(q, [r["sigma_local_MPa"] for r in rows], "o-", label="local 3W")
    axes[0, 1].plot(q, [r["sigma_integral_MPa"] for r in rows], "s-", label="integral")
    axes[0, 1].set_ylabel("sintering stress (MPa)")
    axes[0, 1].legend()
    axes[1, 0].plot(q, [r["r_neck_nm"] for r in rows], "o-", label="neck")
    axes[1, 0].plot(q, [r["r_TJ_nm"] for r in rows], "s-", label="TJ")
    axes[1, 0].set_ylabel("radius (nm)")
    axes[1, 0].legend()
    finite = [math.isfinite(float(r.get("slow_time_s", math.inf))) for r in rows]
    if any(finite):
        axes[1, 1].plot(q[finite], [rows[i]["slow_time_s"]
                                    for i in np.flatnonzero(finite)], "o-")
    axes[1, 1].set_ylabel("integrated slow time (s)")
    for ax in axes.flat:
        ax.set_xlabel("q/b")
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "quasistatic_q_continuation.png", dpi=200)
    fig.savefig(OUT / "quasistatic_q_continuation.pdf")
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    (OUT / "states").mkdir(parents=True)
    set_num_threads(min(8, os.cpu_count() or 1))
    geom, setup = campaign.padded_case_builder()
    state, source_t_model = load_state(SOURCE)
    if state[0].shape != geom["f"].shape:
        raise RuntimeError("source checkpoint and padded case grids differ")
    volume0 = axisym_volume(
        state[0], setup["r_c"], setup["dr"], setup["dz"])
    status = json.loads(STATUS.read_text())
    D_gb_slow = float(status["requested_changes"]["D_GB_m2_per_model_time"])
    path_transport = make_transport(geom)
    clock_transport = replace(
        path_transport, D_gb_m2_per_model_time=D_gb_slow)
    schedule = default_q_schedule()
    rows = []
    event_restart = None
    q_current = 0.0
    started = time.monotonic()
    manifest = dict(
        purpose="deterministic fast-surface/slow-GB q continuation",
        source_checkpoint=str(SOURCE.resolve()),
        stochastic_hazards_enabled=False,
        event_barriers_evaluated=False,
        production_state_modified=False,
        q_schedule=schedule.tolist(),
        q_path_transport_D_GB_m2_per_model_time=(
            path_transport.D_gb_m2_per_model_time),
        q_path_transport_role=(
            "numerical construction only; physical time discarded"),
        slow_clock_D_GB_m2_per_model_time=D_gb_slow,
        slow_clock_definition="dt=dq/[M_GB(q)*affinity(q)]",
        fast_relaxation_operator=(
            "corrected normalized-ownership closed-volume PF"),
        fast_relaxation_does_not_advance_physical_time=True,
        slow_PR_coordinate_not_minimized_away=True,
        fast_block_steps=FAST_BLOCK_STEPS,
        fast_max_blocks=FAST_MAX_BLOCKS,
        fast_tolerances=FAST_TOLERANCES)
    atomic_json(OUT / "manifest.json", manifest)

    state, row, fast_history, projector = relax_fast_manifold(
        state, 0.0, setup, geom, volume0)
    if not row.get("fast_manifold_converged", 0):
        atomic_json(OUT / "result.json", dict(
            status="BLOCKED_AT_Q0_FAST_MANIFOLD",
            final=row, wall_time_s=time.monotonic()-started))
        raise RuntimeError("q=0 fast local surface/TJ manifold did not settle")
    rows.append(row)
    atomic_npz(OUT / "states/q_0p000000.npz", f=state[0],
               particle=state[1], substrate=state[2], q_over_b=0.0)
    print("QCONT q/b=0.000000 "
          f"affinity={row['transport_affinity_MPa']:.6f}MPa "
          f"sigL={row['sigma_local_MPa']:.6f}MPa "
          f"fast_t={row['fast_relax_model_time']:.6e}", flush=True)

    stopped = None
    for target in schedule[1:]:
        event_evaluator = corrected.evaluator_builder(setup, geom)
        result = event_call(
            state, setup, geom, event_evaluator, path_transport, float(target),
            event_restart=event_restart,
            max_increment_fraction_b=0.0025,
            explicit_max_fourth_order_courant=renewal.PRODUCTION_C4,
            surface_flux_mobility_m6_per_J_model_time=(
                corrected.EVENT_SURFACE_MOBILITY),
            explicit_stability_B_m4_per_model_time=corrected.EVENT_B_PF)
        state = tuple(np.asarray(field).copy() for field in result[:3])
        event_restart = result[4]["event_restart"]
        q_actual = float(result[4]["event_progress_over_b"])
        if not result[3]:
            evaluator = corrected.evaluator_builder(setup, geom)
            row = scalar_state(
                state, q_actual, setup, geom, evaluator, volume0)
            row.update(fast_manifold_converged=0,
                       continuation_stop_reason=result[4].get("stop_reason"))
            if q_actual > rows[-1]["q_over_b"] + 1.0e-14:
                rows.append(row)
            stopped = dict(
                reason=result[4].get("stop_reason"), q_over_b=q_actual,
                target_q_over_b=float(target))
            print("QCONT PHYSICAL STOP " + json.dumps(stopped), flush=True)
            break
        q_current = float(target)
        state, row, fast_history, projector = relax_fast_manifold(
            state, q_current, setup, geom, volume0)
        rows.append(row)
        state_name = (f"q_{q_current:.6f}".replace(".", "p") + ".npz")
        atomic_npz(
            OUT / "states" / state_name,
            f=state[0], particle=state[1], substrate=state[2],
            q_over_b=q_current,
            event_time_model=float(result[4]["event_time_model"]))
        write_csv(OUT / "continuation_history.partial.csv", rows)
        print(
            f"QCONT q/b={q_current:.6f} "
            f"affinity={row['transport_affinity_MPa']:.6f}MPa "
            f"sigL={row['sigma_local_MPa']:.6f}MPa "
            f"sigI={row['sigma_integral_MPa']:.6f}MPa "
            f"neck={row['r_neck_nm']:.6f}nm "
            f"fast_t={row['fast_relax_model_time']:.6e} "
            f"converged={int(row.get('fast_manifold_converged', 0))}",
            flush=True)
        if not row.get("fast_manifold_converged", 0):
            stopped = dict(
                reason="fast_local_surface_TJ_manifold_did_not_settle",
                q_over_b=q_current)
            break
        if row["transport_affinity_Pa"] <= 0.0:
            stopped = dict(
                reason="nonpositive_relaxed_transport_affinity",
                q_over_b=q_current)
            break

    q = np.asarray([row["q_over_b"] for row in rows])
    affinity = np.asarray([row["transport_affinity_Pa"] for row in rows])
    if len(rows) > 1:
        slow_model, intervals = integrate_slow_clock(q, affinity, clock_transport)
    else:
        slow_model = intervals = np.zeros(1)
    for index, row in enumerate(rows):
        row["slow_interval_model_time"] = float(intervals[index])
        row["slow_time_model"] = float(slow_model[index])
        row["slow_interval_s"] = float(intervals[index] * SECONDS_PER_MODEL_TIME)
        row["slow_time_s"] = float(slow_model[index] * SECONDS_PER_MODEL_TIME)
    write_csv(OUT / "continuation_history.csv", rows)
    render(rows)
    completed = bool(rows[-1]["q_over_b"] >= 1.0 - 1e-12 and stopped is None)
    result = dict(
        status=("AFFINITY_POSITIVE_THROUGH_ONE_B" if completed else
                "CONTINUATION_STOPPED_BEFORE_ONE_B"),
        completed_one_b=completed,
        final_q_over_b=float(rows[-1]["q_over_b"]),
        initial_affinity_MPa=float(rows[0]["transport_affinity_MPa"]),
        final_affinity_MPa=float(rows[-1]["transport_affinity_MPa"]),
        integrated_slow_time_s=float(rows[-1]["slow_time_s"]),
        stop=stopped,
        states_saved=len(rows),
        wall_time_s=time.monotonic()-started,
        production_launched=False)
    atomic_json(OUT / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
