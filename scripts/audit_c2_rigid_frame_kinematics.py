#!/usr/bin/env python3
"""Energy-free kinematic audit of the C2 RIGHT rigid-frame coordinate."""
from __future__ import annotations

import csv
import json
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
sys.path.insert(0, str(ROOT))

from pf_sintering.three_particle_diagnostics import radius_profile
from pf_sintering.three_particle_sharp_initial import load_mapped_sharp_state
from pf_sintering.work_conjugate_densification import (
    conservative_signed_shift_toward_minus_z,
    signed_one_contact_frame_state,
    triple_junction_support,
)

B = 0.25e-9
SOURCE = ROOT / "runs/three_particle_c2_source/post_cleanup_state.npz"
SNAPSHOT = (
    ROOT
    / "runs/three_particle_c2_campaign/stochastic_seed20260915/snapshots"
    / "t_0002.772296143_ROOT_CROSSING.npz"
)
OUT = ROOT / "runs/c2_corrected_rigid_frame_audit"
DOC = ROOT / "docs/three_particle/c2_corrected_rigid_frame_audit"


def centroid(field, z, r_c):
    weight = np.asarray(field) * np.asarray(r_c)[None, :]
    return float(np.sum(weight * np.asarray(z)[:, None]) / np.sum(weight))


def ownership_plane(ownership, f, z, r_c, left_grain, right_grain, crossing):
    pair = ownership[left_grain] + ownership[right_grain]
    right_fraction = np.divide(
        ownership[right_grain], pair, out=np.zeros_like(f), where=pair > 1e-30
    )
    weight = f * pair * r_c[None, :]
    profile = np.sum(weight * right_fraction, axis=1) / np.maximum(
        np.sum(weight, axis=1), 1e-300
    )
    crossings = np.flatnonzero((profile[:-1] - 0.5) * (profile[1:] - 0.5) <= 0)
    i = crossings[crossing]
    return float(
        z[i]
        + (0.5 - profile[i])
        * (z[i + 1] - z[i])
        / (profile[i + 1] - profile[i])
    )


def write_csv(path, rows, fields=None):
    fields = tuple(rows[0]) if fields is None else fields
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    geometry, _, _ = load_mapped_sharp_state(SOURCE)
    with np.load(SNAPSHOT) as data:
        f0 = data["f"].copy()
        ownership0 = data["ownership"].copy()
        gb0 = data["gb"].copy()

    radius0 = radius_profile(f0, geometry)
    valid0 = np.isfinite(radius0)
    right_neck = float(np.interp(gb0[1], geometry["z"][valid0], radius0[valid0]))
    support = triple_junction_support(
        f0,
        geometry["z"],
        geometry["r_c"],
        z_tj=gb0[1],
        r_tj=right_neck,
        width=geometry["config"].width,
    )

    z = geometry["z"]
    r_c = geometry["r_c"]
    left_marker = np.where(z[:, None] < gb0[0], f0, 0.0)
    center_marker = np.where(
        (z[:, None] >= gb0[0]) & (z[:, None] < gb0[1]), f0, 0.0
    )
    right_marker = np.where(z[:, None] >= gb0[1], f0, 0.0)
    frame_reference = [centroid(marker, z, r_c) for marker in (
        left_marker, center_marker, right_marker
    )]
    material_reference = [centroid(f0 * ownership0[k], z, r_c) for k in range(3)]
    left_plane0 = ownership_plane(ownership0, f0, z, r_c, 0, 1, 0)
    right_plane0 = ownership_plane(ownership0, f0, z, r_c, 1, 2, -1)

    rows = []
    surface_rows = []
    for q_over_b in (0.0, 0.1, 0.5, 1.0):
        u = q_over_b * B
        state = signed_one_contact_frame_state(
            f0,
            ownership0,
            z,
            r_c,
            dr=geometry["dr"],
            dz=geometry["dz"],
            approach_m=u,
            gb_z_m=gb0[1],
            tj_r_m=right_neck,
            width_m=geometry["config"].width,
            moving_grain=2,
            neighbor_grain=1,
            source_support=support,
        )

        frame_markers = (
            left_marker,
            center_marker,
            conservative_signed_shift_toward_minus_z(
                right_marker, u, geometry["dz"]
            ),
        )
        frame_shifts = [
            centroid(marker, z, r_c) - reference
            for marker, reference in zip(frame_markers, frame_reference)
        ]
        material_shifts = [
            centroid(state.f * state.ownership[k], z, r_c) - material_reference[k]
            for k in range(3)
        ]
        measured_left_plane = ownership_plane(
            state.ownership, state.f, z, r_c, 0, 1, 0
        )
        measured_right_plane = ownership_plane(
            state.ownership, state.f, z, r_c, 1, 2, -1
        )
        radius = radius_profile(state.f, geometry)
        valid = np.isfinite(radius)
        prescribed_right_plane = gb0[1] - 0.5 * u

        row = dict(
            q_over_b=q_over_b,
            u_m=u,
            U_L_m=0.0,
            U_C_m=0.0,
            U_R_m=-u,
            q_L_m=0.0,
            q_R_m=u,
            left_frame_tracer_shift_m=frame_shifts[0],
            center_frame_tracer_shift_m=frame_shifts[1],
            right_frame_tracer_shift_m=frame_shifts[2],
            left_frame_tracer_error_m=frame_shifts[0],
            center_frame_tracer_error_m=frame_shifts[1],
            right_frame_tracer_error_m=frame_shifts[2] + u,
            left_GB_plane_m=gb0[0],
            right_GB_plane_m=prescribed_right_plane,
            left_GB_shift_m=0.0,
            right_GB_shift_m=-0.5 * u,
            measured_left_ownership_plane_m=measured_left_plane,
            measured_left_ownership_plane_shift_m=measured_left_plane - left_plane0,
            left_ownership_plane_error_m=measured_left_plane - left_plane0,
            measured_right_ownership_plane_m=measured_right_plane,
            measured_right_ownership_plane_shift_m=measured_right_plane - right_plane0,
            right_ownership_plane_error_m=(measured_right_plane - right_plane0) + 0.5 * u,
            left_material_centroid_shift_m=material_shifts[0],
            center_material_centroid_shift_m=material_shifts[1],
            right_material_centroid_shift_m=material_shifts[2],
            left_neck_radius_m=float(np.interp(gb0[0], z[valid], radius[valid])),
            right_neck_radius_m=float(
                np.interp(prescribed_right_plane, z[valid], radius[valid])
            ),
            mass_relative_error=state.mass_relative_error,
            partition_residual=state.partition_residual,
            f_min=state.f_bounds[0],
            f_max=state.f_bounds[1],
        )
        rows.append(row)
        surface_rows.extend(
            dict(
                q_over_b=q_over_b,
                u_m=u,
                z_m=float(z_value),
                surface_radius_m=float(radius_value),
            )
            for z_value, radius_value in zip(z[valid], radius[valid])
        )
        np.savez_compressed(
            OUT / f"kinematic_q_{q_over_b:.3f}b.npz",
            f=state.f,
            ownership=state.ownership,
            **row,
        )

    passed = all(
        abs(row["U_L_m"]) < 1e-30
        and abs(row["U_C_m"]) < 1e-30
        and row["U_R_m"] == -row["u_m"]
        and row["q_L_m"] == 0.0
        and row["q_R_m"] == row["u_m"]
        and abs(row["left_frame_tracer_error_m"]) < 2e-20
        and abs(row["center_frame_tracer_error_m"]) < 2e-20
        and abs(row["right_frame_tracer_error_m"]) < 2e-20
        and abs(row["left_GB_shift_m"]) < 1e-30
        and abs(row["right_GB_shift_m"] + 0.5 * row["u_m"]) < 1e-30
        and abs(row["left_ownership_plane_error_m"]) < 3e-11
        and abs(row["right_ownership_plane_error_m"]) < 3e-11
        and row["mass_relative_error"] < 3e-13
        for row in rows
    )

    write_csv(DOC / "kinematic_audit.csv", rows)
    write_csv(
        DOC / "rigid_frames.csv",
        rows,
        (
            "q_over_b", "u_m", "U_L_m", "U_C_m", "U_R_m", "q_L_m", "q_R_m",
            "left_frame_tracer_shift_m", "center_frame_tracer_shift_m",
            "right_frame_tracer_shift_m", "left_frame_tracer_error_m",
            "center_frame_tracer_error_m", "right_frame_tracer_error_m",
        ),
    )
    write_csv(
        DOC / "ownership_planes.csv",
        rows,
        (
            "q_over_b", "u_m", "left_GB_plane_m", "right_GB_plane_m",
            "left_GB_shift_m", "right_GB_shift_m",
            "measured_left_ownership_plane_m", "measured_left_ownership_plane_shift_m",
            "left_ownership_plane_error_m", "measured_right_ownership_plane_m",
            "measured_right_ownership_plane_shift_m", "right_ownership_plane_error_m",
        ),
    )
    write_csv(
        DOC / "material_centroids.csv",
        rows,
        (
            "q_over_b", "u_m", "left_material_centroid_shift_m",
            "center_material_centroid_shift_m", "right_material_centroid_shift_m",
        ),
    )
    write_csv(DOC / "free_surface_geometry.csv", surface_rows)

    summary = dict(
        pass_all_kinematic_gates=passed,
        energy_evaluated=False,
        coordinate_definition="U_L=U_C=0; U_R=-u; q_L=0; q_R=u",
        ownership_definition="Delta z_GB,L=0; Delta z_GB,R=-u/2",
        centroids_used_as_constraints=False,
        max_fixed_frame_tracer_error_m=max(
            max(abs(row["left_frame_tracer_error_m"]), abs(row["center_frame_tracer_error_m"]))
            for row in rows
        ),
        max_moving_frame_tracer_error_m=max(
            abs(row["right_frame_tracer_error_m"]) for row in rows
        ),
        max_left_ownership_plane_error_m=max(
            abs(row["left_ownership_plane_error_m"]) for row in rows
        ),
        max_right_ownership_plane_error_m=max(
            abs(row["right_ownership_plane_error_m"]) for row in rows
        ),
        max_mass_relative_error=max(row["mass_relative_error"] for row in rows),
    )
    (DOC / "kinematic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
