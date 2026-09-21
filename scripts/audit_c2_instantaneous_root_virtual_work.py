#!/usr/bin/env python3
"""Literal signed rigid-body virtual work at the preserved first C2 RIGHT root."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np

n = types.ModuleType("numba")
n.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else lambda f: f)
n.prange = range
n.get_num_threads = lambda: 1
n.set_num_threads = lambda _: None
sys.modules.setdefault("numba", n)
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
sk = types.ModuleType("skimage")
me = types.ModuleType("skimage.measure")
me.find_contours = lambda *a, **k: None
sk.measure = me
sys.modules.setdefault("skimage", sk)
sys.modules.setdefault("skimage.measure", me)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from pf_sintering.constrained_densification_relaxation import multigrain_energy
from pf_sintering.constrained_envelope_force import recover_kkt_multipliers
from pf_sintering.three_particle_diagnostics import radius_profile
from pf_sintering.three_particle_phase_a import FrozenPhysics
from pf_sintering.three_particle_sharp_initial import load_mapped_sharp_state
from pf_sintering.work_conjugate_densification import (
    signed_one_contact_frame_state,
    triple_junction_support,
)
from report_bicrystal_force_mismatch import cc_row, cmc_side_fit, local_side_fit

B = 0.25e-9
SOURCE = ROOT / "runs/three_particle_c2_source/post_cleanup_state.npz"
RUN = ROOT / "runs/three_particle_c2_campaign/stochastic_seed20260915"
SNAPSHOT = RUN / "snapshots/t_0002.772296143_ROOT_CROSSING.npz"
DOC = ROOT / "docs/three_particle/c2_corrected_rigid_frame_audit"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_change(current, previous):
    return abs(current - previous) / max(abs(current), 1e-300)


def main():
    kinematic = json.loads((DOC / "kinematic_summary.json").read_text())
    if not kinematic["pass_all_kinematic_gates"] or kinematic["energy_evaluated"]:
        raise RuntimeError("energy-free kinematic audit has not passed")

    geometry, _, _ = load_mapped_sharp_state(SOURCE)
    with np.load(SNAPSHOT) as data:
        f0 = data["f"].copy()
        ownership0 = data["ownership"].copy()
        gb0 = data["gb"].copy()
        time_s = float(data["time_s"])

    radius = radius_profile(f0, geometry)
    valid = np.isfinite(radius)
    right_neck = float(np.interp(gb0[1], geometry["z"][valid], radius[valid]))
    support = triple_junction_support(
        f0,
        geometry["z"],
        geometry["r_c"],
        z_tj=gb0[1],
        r_tj=right_neck,
        width=geometry["config"].width,
    )

    def state(approach_m):
        return signed_one_contact_frame_state(
            f0,
            ownership0,
            geometry["z"],
            geometry["r_c"],
            dr=geometry["dr"],
            dz=geometry["dz"],
            approach_m=approach_m,
            gb_z_m=gb0[1],
            tj_r_m=right_neck,
            width_m=geometry["config"].width,
            moving_grain=2,
            neighbor_grain=1,
            source_support=support,
        )

    root_energy = multigrain_energy(f0, ownership0, geometry)
    rows = []
    for delta_over_b in (
        0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125,
        0.00015625, 0.000078125, 0.0000390625, 0.00001953125,
        0.000009765625,
    ):
        delta = delta_over_b * B
        plus = state(delta)
        minus = state(-delta)
        plus_energy = multigrain_energy(plus.f, plus.ownership, geometry)
        minus_energy = multigrain_energy(minus.f, minus.ownership, geometry)
        approach_force = -(plus_energy - root_energy) / delta
        separation_force = -(root_energy - minus_energy) / delta
        rows.append(dict(
            delta_u_over_b=delta_over_b,
            delta_u_m=delta,
            G_minus_J=minus_energy,
            G_root_J=root_energy,
            G_plus_J=plus_energy,
            symmetric_quotient_N=-(plus_energy - minus_energy) / (2.0 * delta),
            approach_side_force_N=approach_force,
            separation_side_force_N=separation_force,
            one_sided_jump_N=approach_force - separation_force,
            plus_mass_relative_error=plus.mass_relative_error,
            minus_mass_relative_error=minus.mass_relative_error,
            plus_partition_residual=plus.partition_residual,
            minus_partition_residual=minus.partition_residual,
            plus_f_min=plus.f_bounds[0],
            plus_f_max=plus.f_bounds[1],
            minus_f_min=minus.f_bounds[0],
            minus_f_max=minus.f_bounds[1],
            plus_U_L_m=0.0,
            plus_U_C_m=0.0,
            plus_U_R_m=plus.moving_frame_translation_z_m,
            minus_U_L_m=0.0,
            minus_U_C_m=0.0,
            minus_U_R_m=minus.moving_frame_translation_z_m,
            plus_q_L_m=0.0,
            plus_q_R_m=delta,
            minus_q_L_m=0.0,
            minus_q_R_m=-delta,
            plus_ownership_plane_translation_m=plus.ownership_plane_translation_z_m,
            minus_ownership_plane_translation_m=minus.ownership_plane_translation_z_m,
        ))

    physics = FrozenPhysics()
    local_minus = local_side_fit(radius, geometry["z"], gb0[1], -1, 15e-9)
    local_plus = local_side_fit(radius, geometry["z"], gb0[1], 1, 15e-9)
    a, b = local_minus["slope"], local_plus["slope"]
    psi_deg = math.degrees(math.acos(np.clip(
        -(1.0 + a * b) / math.sqrt((1.0 + a * a) * (1.0 + b * b)),
        -1.0,
        1.0,
    )))
    cmc_minus = cmc_side_fit(
        radius, geometry["z"], gb0[1], -1, geometry["config"].width
    )
    cmc_plus = cmc_side_fit(
        radius, geometry["z"], gb0[1], 1, geometry["config"].width
    )
    cmc_delta = cmc_plus["kappa_total_per_m"] - cmc_minus["kappa_total_per_m"]
    cc_cmc = cc_row("global_CMC", right_neck, psi_deg, cmc_delta, physics.gamma_s)

    kkt = recover_kkt_multipliers(
        f0,
        ownership0,
        geometry,
        active_mask=np.ones_like(f0, bool),
        include_moments=False,
        box_minimum_step=0.01,
    )
    volume_normalization = (
        2.0 * math.pi * geometry["dr"] * geometry["dz"]
        * geometry["config"].outer_radius
    )
    pressures = -np.asarray(kkt["multipliers"]) / volume_normalization
    kkt_delta = (pressures[2] - pressures[1]) / physics.gamma_s
    cc_kkt = cc_row("KKT", right_neck, psi_deg, kkt_delta, physics.gamma_s)

    finest = rows[-1]
    previous = rows[-2]
    approach_converged = relative_change(
        finest["approach_side_force_N"], previous["approach_side_force_N"]
    ) < 0.02
    separation_converged = relative_change(
        finest["separation_side_force_N"], previous["separation_side_force_N"]
    ) < 0.02
    one_sided_close = (
        abs(finest["one_sided_jump_N"])
        / max(
            abs(finest["approach_side_force_N"]),
            abs(finest["separation_side_force_N"]),
            1e-300,
        )
        <= 0.05
    )
    derivative_exists = bool(approach_converged and separation_converged and one_sided_close)
    approach_force = finest["approach_side_force_N"]
    cmc_relative_difference = (
        approach_force - cc_cmc["total_force_N"]
    ) / abs(cc_cmc["total_force_N"])
    kkt_relative_difference = (
        approach_force - cc_kkt["total_force_N"]
    ) / abs(cc_kkt["total_force_N"])
    conjugacy_closed = bool(
        derivative_exists and abs(cmc_relative_difference) <= 0.05
    )

    summary = dict(
        classification=(
            "C2_INSTANTANEOUS_CONJUGACY_CLOSED"
            if conjugacy_closed
            else "C2_INSTANTANEOUS_DERIVATIVE_NOT_DEFINED"
            if not derivative_exists
            else "C2_INSTANTANEOUS_CONJUGACY_NOT_CLOSED"
        ),
        scientific_decision=(
            "EVENT_COORDINATE_REDEFINED_AND_THERMODYNAMICALLY_CLOSED"
            if conjugacy_closed
            else "UNRESOLVED_EVENT_THERMODYNAMICS"
        ),
        finite_event_path_authorized=conjugacy_closed,
        finite_event_path_launched=False,
        root_stationarized_before_test=False,
        coordinate_definition="U_L=U_C=0; U_R=-u; q_L=0; q_R=u",
        ownership_definition="Delta z_GB,L=0; Delta z_GB,R=-u/2",
        virtual_copy_definition="literal signed RIGHT-body translation with fixed LEFT/CENTER frames",
        material_centroids_used_as_constraints=False,
        global_minimization_used=False,
        time_s=time_s,
        instantaneous_derivative_exists=derivative_exists,
        instantaneous_force_N=(
            finest["symmetric_quotient_N"] if derivative_exists else None
        ),
        symmetric_quotient_N=finest["symmetric_quotient_N"],
        approach_side_force_N=approach_force,
        separation_side_force_N=finest["separation_side_force_N"],
        approach_side_step_converged=approach_converged,
        separation_side_step_converged=separation_converged,
        one_sided_derivatives_close=one_sided_close,
        one_sided_jump_N=finest["one_sided_jump_N"],
        contact_cusp_detected=bool(
            approach_converged and separation_converged and not one_sided_close
        ),
        global_CMC_CC_force_N=cc_cmc["total_force_N"],
        approach_to_global_CMC_relative_difference=cmc_relative_difference,
        KKT_CC_force_N=cc_kkt["total_force_N"],
        approach_to_KKT_relative_difference=kkt_relative_difference,
        root_KKT_residual=kkt["projected_KKT_Linf"],
        KKT_pressure_qualified=bool(kkt["projected_KKT_Linf"] <= 1e-7),
        pressure_source="far-field global CMC and full-domain KKT multipliers; no diffuse-TJ curvature",
        right_neck_radius_m=right_neck,
        measured_dihedral_deg=psi_deg,
        CMC_pressure_difference_Pa=physics.gamma_s * cmc_delta,
        KKT_pressure_difference_Pa=pressures[2] - pressures[1],
        source_snapshot_sha256=sha256(SNAPSHOT),
        source_history_sha256=sha256(RUN / "history.json"),
        source_trajectory_sha256=sha256(RUN / "trajectory.npz"),
    )

    with (DOC / "instantaneous_virtual_work.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (DOC / "instantaneous_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
