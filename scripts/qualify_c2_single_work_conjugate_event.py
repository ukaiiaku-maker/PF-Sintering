#!/usr/bin/env python3
"""Trace one full-domain stationary C2 event from the first genuine root.

The historical stochastic trajectory is read-only.  This script creates a
discarded-copy branch with fixed individual grain volumes, prescribed RIGHT
ownership-frame displacement, and no centroid constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

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
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]

from pf_sintering.constrained_densification_relaxation import (
    multigrain_energy, volume_targets)
from pf_sintering.constrained_newton_krylov import solve_constrained_stationary
from pf_sintering.three_particle_diagnostics import radius_profile
from pf_sintering.three_particle_geometry import topology_status
from pf_sintering.three_particle_sharp_initial import load_mapped_sharp_state
from pf_sintering.work_conjugate_densification import one_contact_state

B = 0.25e-9
SOURCE_RUN = ROOT/"runs/three_particle_c2_campaign/stochastic_seed20260915"
SOURCE_STATE = ROOT/"runs/three_particle_c2_source/post_cleanup_state.npz"
ROOT_SNAPSHOT = SOURCE_RUN/"snapshots/t_0002.772296143_ROOT_CROSSING.npz"
OUT = ROOT/"runs/c2_thermodynamic_single_event_first_right_root"
Q_VALUES = np.round(np.linspace(0.0, 1.0, 21), 10)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def centroid(field, z, r_c):
    weight = np.asarray(field)*np.asarray(r_c)[None, :]
    return float(np.sum(weight*np.asarray(z)[:, None])/np.sum(weight))


def load_problem():
    g, _, source_meta = load_mapped_sharp_state(SOURCE_STATE)
    with np.load(ROOT_SNAPSHOT, allow_pickle=False) as data:
        f0 = data["f"].copy()
        phi0 = data["ownership"].copy()
        gb0 = data["gb"].copy()
        snapshot_meta = dict(
            time_s=float(data["time_s"]), phase=str(data["phase"]),
            event_number=int(data["event_number"]),
            completed_avalanches=int(data["completed_avalanches"]))
    g.update(f=f0, ownership=phi0, gb=gb0)
    radius = radius_profile(f0, g)
    finite = np.isfinite(radius)
    if not np.any(finite):
        raise RuntimeError("the preserved root has no resolved f=0.5 surface")
    right_neck_radius = float(np.interp(gb0[1], g["z"][finite], radius[finite]))
    candidate_one = one_contact_state(
        f0, phi0, g["z"], g["r_c"], dr=g["dr"], dz=g["dz"],
        u_m=B, gb_z_m=gb0[1], tj_r_m=right_neck_radius,
        width_m=g["config"].width, moving_grain=2, neighbor_grain=1)
    phi_one = candidate_one.ownership

    def ownership_at(q):
        # The qualified conservative subcell advection is affine in u over
        # one cell.  This form also supplies exact infinitesimal extensions
        # outside [0,1] for centered endpoint derivatives.
        return phi0+float(q)*(phi_one-phi0)

    return g, f0, phi0, gb0, ownership_at, snapshot_meta, source_meta


def state_path(index):
    return OUT/f"q_{Q_VALUES[index]:.4f}b.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--lbfgs", type=int, default=3000)
    parser.add_argument("--newton", type=int, default=2000)
    args = parser.parse_args()
    index = args.index
    if not 0 <= index < len(Q_VALUES):
        raise SystemExit("index outside stationary branch")
    q = float(Q_VALUES[index])
    OUT.mkdir(parents=True, exist_ok=True)
    g, f0, phi0, gb0, ownership_at, snapshot_meta, source_meta = load_problem()
    target = volume_targets(
        f0, phi0, g["z"], g["r_c"], g["config"].outer_radius)
    ownership = ownership_at(q)
    checkpoint = OUT/f"q_{q:.4f}b_checkpoint.npz"
    if state_path(index).exists():
        print("ALREADY_COMPLETE", state_path(index), flush=True)
        return
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as data:
            initial = data["f"].copy()
        source = str(checkpoint)
    elif index > 0 and state_path(index-1).exists():
        with np.load(state_path(index-1), allow_pickle=False) as data:
            initial = data["f"].copy()
        source = str(state_path(index-1))
    elif index == 0:
        initial = f0.copy()
        source = str(ROOT_SNAPSHOT)
    else:
        raise RuntimeError("previous continuation state is missing")
    mask = np.ones_like(initial, dtype=bool)
    print(json.dumps(dict(index=index, q_over_b=q, initial=source,
        cells=int(initial.size), target=target.tolist())), flush=True)

    def callback(field, row):
        print(json.dumps(row), flush=True)
        if row["stage"] == "newton" or int(row["iteration"]) % 10 == 0:
            np.savez(checkpoint, f=field, q_over_b=q)

    result = solve_constrained_stationary(
        initial, ownership, g, target, active_mask=mask,
        lbfgs_max_iterations=args.lbfgs,
        newton_max_iterations=args.newton, include_moments=False,
        kkt_tolerance=1e-7, callback=callback,
        lbfgs_box_active_step=0.01,
        newton_box_active_step=0.01, gmres_maxiter=8,
        newton_minimum_damping=1e-4)
    energy = multigrain_energy(result.f, ownership, g)
    topology = topology_status(result.f, g)
    np.savez_compressed(
        state_path(index), f=result.f, ownership=ownership, target=target,
        active_mask=mask, q_over_b=q, energy_J=energy,
        converged=result.converged,
        projected_KKT_residual=result.projected_kkt_residual,
        normalized_constraint_residual=result.normalized_constraint_residual)
    metadata = dict(
        index=index, q_over_b=q, converged=result.converged,
        reason=result.reason, projected_KKT_residual=result.projected_kkt_residual,
        normalized_constraint_residual=result.normalized_constraint_residual,
        energy_J=energy, active_cells=result.active_cells,
        gram_condition=result.gram_condition,
        lbfgs_iterations=result.lbfgs_iterations,
        newton_iterations=result.newton_iterations,
        f_min=float(result.f.min()), f_max=float(result.f.max()),
        topology_stop=bool(topology["stop"]),
        topology={key:(value.item() if isinstance(value, np.generic) else value)
                  for key, value in topology.items()},
        initial_source=source, source_snapshot=str(ROOT_SNAPSHOT.relative_to(ROOT)),
        source_snapshot_sha256=sha256(ROOT_SNAPSHOT),
        source_history_sha256=sha256(SOURCE_RUN/"history.json"),
        source_trajectory_sha256=sha256(SOURCE_RUN/"trajectory.npz"),
        snapshot_metadata=snapshot_meta,
        stochastic_source_metadata_sha256=hashlib.sha256(
            json.dumps(source_meta, sort_keys=True, default=str).encode()).hexdigest(),
        individual_grain_volumes_fixed=True, first_moments_fixed=False,
        active_domain="full", clipping_used=False)
    (OUT/f"q_{q:.4f}b.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("RESULT", json.dumps(metadata), flush=True)
    if not result.converged or topology["stop"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
