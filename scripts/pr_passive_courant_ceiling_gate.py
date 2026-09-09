#!/usr/bin/env python3
"""One-time passive explicit-Courant qualification for closed PR loading."""
from __future__ import annotations

import csv
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
from pf_sintering.fourier_max_pr_geometry import (  # noqa: E402
    build_fourier_max_pr_geometry,
)
from pf_sintering.model_time_transport import (  # noqa: E402
    mullins_pf_coefficient_m4_per_model_time,
)
from pr_closed_volume_no_event_t34 import (  # noqa: E402
    build_case as build_eps025_case, make_tracker_evaluator, outer_radius,
)
from pr_closed_volume_passive_continuation import profile_metrics  # noqa: E402


OUT = ROOT / "runs/pr_passive_courant_ceiling_eps0p40_closed_qualified"
EPSILON1 = 0.40
R0_OVER_RCYL = 0.958442910521405
COURANT_CASES = (0.05, 0.10, 0.20)
INTERVAL_MODEL = 1.0
GEOMETRY_RELATIVE_TOLERANCE = 1.0e-3
STRESS_ABSOLUTE_TOLERANCE_PA = 0.05e6
FIELD_RELATIVE_L2_TOLERANCE = 1.0e-3
VOLUME_TOLERANCE = 1.0e-8


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def build_eps040_case():
    reference, _ = build_eps025_case()
    geom = build_fourier_max_pr_geometry(
        R_cyl=100e-9, W=10e-9, spacing=1.25e-9, eps1=EPSILON1,
        r0_over_rcyl=R0_OVER_RCYL, tail_margin_W=12.0,
        radial_margin_W=12.0, close_left_substrate=True)
    _, setup = build_eps025_case()
    setup = {
        **setup, "dr": geom["dr"], "dz": geom["dz"],
        "r_c": geom["r_c"], "r_f": geom["r_f"], "z": geom["z"],
        "lam": geom["lam"]}
    relative = geom["diffuse_volume_m3"] / reference["diffuse_volume_m3"] - 1.0
    if abs(relative) >= 1.0e-10:
        raise RuntimeError(f"epsilon=0.40 initial volume mismatch {relative:+.9e}")
    return geom, setup, reference["diffuse_volume_m3"], relative


def final_metrics(state, setup, geom):
    evaluator = make_tracker_evaluator(setup, geom)
    scalar, _, node = measure_experimental_pr_state(state, setup, evaluator)
    profile = profile_metrics(outer_radius(state[0], setup["r_c"]), setup, geom)
    return dict(
        A1_m=profile["A1_cos_m"],
        A1_magnitude_m=profile["A1_magnitude_m"],
        A_PR_over_r_bar=profile["A_PR_over_r_bar"],
        neck_to_bulge_ratio=profile["neck_to_bulge_ratio"],
        sigma_integral_Pa=float(scalar["sigma_integral_Pa"]),
        sigma_local_Pa=float(scalar["sigma_local_Pa"]),
        r_TJ_m=float(scalar["r_TJ_m"]),
        tj_candidate_count=int(node["tj_tracking"]["candidate_count"]))


def run_case(initial, setup, geom, courant_max):
    state = tuple(np.asarray(field).copy() for field in initial)
    projector = ConservativeBoundedPhaseProjector()
    scratch = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
    current = None
    B = mullins_pf_coefficient_m4_per_model_time(
        setup["M_s"], setup["gamma_s"])
    dx = min(setup["dr"], setup["dz"])
    dt_ceiling = 0.9 * courant_max * dx ** 4 / B
    elapsed = 0.0
    steps = 0
    V0 = axisym_volume(state[0], setup["r_c"], setup["dr"], setup["dz"])
    started = time.monotonic()
    while elapsed < INTERVAL_MODEL - 1e-15:
        dt = min(dt_ceiling, INTERVAL_MODEL - elapsed)
        destination = 0 if current != 0 else 1
        state = axisym_gb_face_projected_step_fast(
            *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
            setup["r_c"], setup["r_f"], dt, setup["M_s"], setup["M_eta"],
            setup["W"], scratch[destination])
        current = destination
        state, _ = projector(state, setup)
        elapsed += dt
        steps += 1
    volume = axisym_volume(state[0], setup["r_c"], setup["dr"], setup["dz"])
    closure = float(np.max(np.abs(state[1] + state[2] - state[0])))
    result = dict(
        C4_max=courant_max, dt_ceiling_model=dt_ceiling,
        explicit_steps=steps, elapsed_model=elapsed,
        wall_seconds=time.monotonic() - started,
        Vf_relative=volume / V0 - 1.0, closure_max=closure,
        projector_active_calls=projector.active_calls,
        **final_metrics(state, setup, geom))
    if abs(result["Vf_relative"]) >= VOLUME_TOLERANCE:
        raise RuntimeError("Courant trial violated total volume")
    if closure >= 1.0e-12:
        raise RuntimeError("Courant trial violated phase closure")
    return state, result


def main():
    set_num_threads(8)
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    geom, setup, target_volume, initial_volume_relative = build_eps040_case()
    initial = (geom["f"], geom["e1"], geom["e2"])
    with (OUT / "initial_eps0p40_volume_matched.npz").open("wb") as stream:
        np.savez_compressed(
            stream, f=initial[0], e1=initial[1], e2=initial[2],
            R0_over_Rcyl=R0_OVER_RCYL, epsilon1=EPSILON1,
            target_eps0p25_volume_m3=target_volume,
            initial_volume_relative=initial_volume_relative)
    rows = []
    states = {}
    for courant in COURANT_CASES:
        state, row = run_case(initial, setup, geom, courant)
        states[courant] = tuple(np.asarray(field).copy() for field in state)
        if not rows:
            row.update(
                A1_relative_disagreement=0.0,
                neck_bulge_relative_disagreement=0.0,
                sigma_integral_disagreement_Pa=0.0,
                sigma_local_disagreement_Pa=0.0,
                field_relative_L2=0.0,
                agreement_pass=True)
        else:
            reference = rows[0]
            reference_state = states[COURANT_CASES[0]]
            row.update(
                A1_relative_disagreement=abs(
                    row["A1_m"] / reference["A1_m"] - 1.0),
                neck_bulge_relative_disagreement=abs(
                    row["neck_to_bulge_ratio"]
                    / reference["neck_to_bulge_ratio"] - 1.0),
                sigma_integral_disagreement_Pa=abs(
                    row["sigma_integral_Pa"]
                    - reference["sigma_integral_Pa"]),
                sigma_local_disagreement_Pa=abs(
                    row["sigma_local_Pa"] - reference["sigma_local_Pa"]),
                field_relative_L2=float(
                    np.linalg.norm(state[0] - reference_state[0])
                    / np.linalg.norm(reference_state[0])))
            row["agreement_pass"] = bool(
                row["A1_relative_disagreement"]
                <= GEOMETRY_RELATIVE_TOLERANCE
                and row["neck_bulge_relative_disagreement"]
                <= GEOMETRY_RELATIVE_TOLERANCE
                and row["sigma_integral_disagreement_Pa"]
                <= STRESS_ABSOLUTE_TOLERANCE_PA
                and row["sigma_local_disagreement_Pa"]
                <= STRESS_ABSOLUTE_TOLERANCE_PA
                and row["field_relative_L2"]
                <= FIELD_RELATIVE_L2_TOLERANCE)
        rows.append(row)
        print("COURANT", json.dumps(row), flush=True)
        if not row["agreement_pass"]:
            print(
                f"STOPPING COURANT LADDER after significant disagreement at "
                f"C4,max={courant}", flush=True)
            break
    selected = max(row["C4_max"] for row in rows if row["agreement_pass"])
    write_csv(OUT / "passive_courant_comparison.csv", rows)
    result = dict(
        outcome="PASSIVE_C4_CEILING_FROZEN",
        selected_passive_C4_max=selected, active_event_C4_unchanged=0.05,
        tested_C4_cases=[row["C4_max"] for row in rows],
        stopped_before_next_case=bool(len(rows) < len(COURANT_CASES)),
        interval_model=INTERVAL_MODEL, epsilon1=EPSILON1,
        epsilon2=0.0, R0_over_Rcyl=R0_OVER_RCYL,
        target_eps0p25_volume_m3=target_volume,
        initial_eps0p40_volume_relative_error=initial_volume_relative,
        geometry_relative_tolerance=GEOMETRY_RELATIVE_TOLERANCE,
        stress_absolute_tolerance_Pa=STRESS_ABSOLUTE_TOLERANCE_PA,
        field_relative_L2_tolerance=FIELD_RELATIVE_L2_TOLERANCE,
        rows=rows)
    atomic_json(OUT / "passive_courant_result.json", result)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    c = [row["C4_max"] for row in rows]
    axes[0].plot(c, [row["field_relative_L2"] for row in rows], "o-")
    axes[0].axhline(FIELD_RELATIVE_L2_TOLERANCE, color="k", ls="--")
    axes[0].set_ylabel("field relative L2")
    axes[1].plot(c, [row["A1_relative_disagreement"] for row in rows],
                 "o-", label="A1")
    axes[1].plot(c, [row["neck_bulge_relative_disagreement"] for row in rows],
                 "s-", label="neck/bulge")
    axes[1].axhline(GEOMETRY_RELATIVE_TOLERANCE, color="k", ls="--")
    axes[1].set_ylabel("relative disagreement"); axes[1].legend()
    axes[2].plot(c, [row["sigma_integral_disagreement_Pa"] / 1e6 for row in rows],
                 "o-", label="integral")
    axes[2].plot(c, [row["sigma_local_disagreement_Pa"] / 1e6 for row in rows],
                 "s-", label="local")
    axes[2].axhline(STRESS_ABSOLUTE_TOLERANCE_PA / 1e6, color="k", ls="--")
    axes[2].set_ylabel("stress disagreement (MPa)"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("passive C4,max"); ax.grid(alpha=.2)
    fig.savefig(OUT / "passive_courant_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
