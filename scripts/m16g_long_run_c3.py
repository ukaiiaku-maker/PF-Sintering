"""Milestone 16G revision: long, checkpointed sink-OFF trajectory on the
C3-smooth particle/asperity geometry, run until a/a0<=0.85 (preferred) or
a/a0<=0.90 (minimum), or until a large step budget/topology failure
forces a stop. Checkpoints (f,e1,e2,step) to .npz periodically so the run
can be resumed across multiple invocations; appends full diagnostic rows
(Section 17 of the handoff: a_contact, X_neck, A_GB, Vp/Vp0, Vs, GB/TJ
position, r_neck (both fit windows), sigma_sintering_paper and its
decomposition, mu_N, mass, F) to an append-only JSONL log.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, mu_N_from_sigma, neck_curvature_windows  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import (  # noqa: E402
    build_particle_asperity_geometry_c3, find_gb_trough, find_stable_dt, grain_volumes,
)

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16g_campaign")
OMEGA = 1.0e-29  # placeholder atomic volume (m^3), diagnostic-only (mu_N is not fed back)


def diagnostics_row(step, t, f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam,
                     Vp0, Vs0, V0, a0, gamma_s, gamma_gb, W):
    R_of_z = measure_R_of_z(f, r_c)
    z_gb, a = find_gb_trough(R_of_z, z, z_gb_prev, lam=lam)
    Vp, Vs = grain_volumes(e1, e2, r_c, dr, dz)
    V = axisym_volume(f, r_c, dr, dz)
    F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)

    X_neck = 2.0 * a if np.isfinite(a) else float("nan")
    row = dict(step=step, t=t, a_nm=a * 1e9 if np.isfinite(a) else float("nan"),
               a_over_a0=a / a0 if np.isfinite(a) else float("nan"),
               X_neck_nm=X_neck * 1e9 if np.isfinite(X_neck) else float("nan"),
               A_GB_nm2=math.pi * (a * 1e9) ** 2 if np.isfinite(a) else float("nan"),
               z_gb_nm=z_gb * 1e9 if np.isfinite(z_gb) else float("nan"),
               Vp_frac=Vp / Vp0, Vs_frac=Vs / Vs0, mass_drift=(V - V0) / V0, F=F)

    if np.isfinite(z_gb) and np.isfinite(a):
        wins = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5, 2.5))
        for w in wins:
            tag = f"w{w['window_W']:g}W"
            row[f"r_neck_{tag}_nm"] = w["r_neck_signed"] * 1e9 if np.isfinite(w["r_neck_signed"]) else float("nan")
            row[f"n_pts_{tag}"] = w["n_points"]
            if np.isfinite(w["r_neck_signed"]) and np.isfinite(X_neck):
                sigma, sc, sg, CGB = hussein_eq1b_sigma(w["r_neck_signed"], X_neck, gamma_s, gamma_gb)
                row[f"sigma_sintering_paper_{tag}"] = sigma
                row[f"sigma_curvature_{tag}"] = sc
                row[f"sigma_contact_GB_{tag}"] = sg
                row[f"mu_N_{tag}"] = mu_N_from_sigma(sigma, OMEGA)
        row["C_GB"] = math.sqrt(1.0 - (gamma_gb / (2.0 * gamma_s)) ** 2)
    return row, z_gb if np.isfinite(z_gb) else z_gb_prev


def run(psi_deg=160.0, R_cyl_nm=100.0, W_nm=10.0, dx_nm=1.25, target_a_over_a0=0.85,
        min_a_over_a0=0.90, max_steps=20_000_000, checkpoint_every_steps=20000,
        diag_every_steps=2000, tag="c3_long", L_frac=3.0, a_cap_frac=4.0, wall_budget_s=None):
    R_cyl = R_cyl_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    gamma_gb = psi_to_gamma_gb(psi_deg)
    gamma_s = 1.0
    p = P(gamma_s=gamma_s, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]

    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    ckpt_path = os.path.join(CAMPAIGN_DIR, f"{tag}_checkpoint.npz")
    log_path = os.path.join(CAMPAIGN_DIR, f"{tag}_log.jsonl")
    meta_path = os.path.join(CAMPAIGN_DIR, f"{tag}_meta.json")

    if os.path.exists(ckpt_path) and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        data = np.load(ckpt_path)
        f, e1, e2 = data["f"], data["e1"], data["e2"]
        step = int(data["step"])
        z = data["z"]
        r_c, r_f = r_centers_faces(meta["Nr"], dr)
        lam, z1, a0, dt = meta["lam"], meta["z1"], meta["a0"], meta["dt"]
        Vp0, Vs0, V0 = meta["Vp0"], meta["Vs0"], meta["V0"]
        z_gb_prev = meta.get("z_gb_prev", z1)
        print(f"resumed from checkpoint at step={step}, t={step*dt:.4f}")
    else:
        geom = build_particle_asperity_geometry_c3(R_cyl, W, dr, dz, L_frac=L_frac, a_cap_frac=a_cap_frac)
        f, e1, e2 = geom["f"], geom["e1"], geom["e2"]
        z, r_c, r_f = geom["z"], geom["r_c"], geom["r_f"]
        z1, lam = geom["z1"], geom["lam"]
        M_s = 1e-33
        M_eta = 1e-33 / (W * (32.0 / 35.0))
        dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100)
        dt *= 0.4
        R_of_z0 = measure_R_of_z(f, r_c)
        a0 = float(np.interp(z1, z, R_of_z0))
        Vp0, Vs0 = grain_volumes(e1, e2, r_c, dr, dz)
        V0 = axisym_volume(f, r_c, dr, dz)
        step = 0
        z_gb_prev = z1
        meta = dict(Nz=geom["Nz"], Nr=geom["Nr"], lam=lam, z1=z1, a0=a0, dt=dt,
                    Vp0=Vp0, Vs0=Vs0, V0=V0, z_gb_prev=z_gb_prev, psi_deg=psi_deg,
                    R_cyl_nm=R_cyl_nm, W_nm=W_nm, dx_nm=dx_nm, L_frac=L_frac, a_cap_frac=a_cap_frac)
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        row0, z_gb_prev = diagnostics_row(0, 0.0, f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev,
                                           lam, Vp0, Vs0, V0, a0, gamma_s, gamma_gb, W)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row0) + "\n")
        print(f"[init] a0={a0*1e9:.4f}nm dt={dt:.4e} target a/a0<={target_a_over_a0}")

    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    t_wall0 = time.time()
    last_ckpt = step
    last_diag = step
    a_over_a0_last = 1.0
    milestones_hit = set()

    while step < max_steps:
        f, e1, e2, diag = axisym_gb_face_projected_step(
            f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W, bc_z="noflux")
        step += 1
        if not np.all(np.isfinite(f)):
            print(f"BLOWUP at step {step} -- stopping for review")
            break

        if step - last_diag >= diag_every_steps:
            last_diag = step
            row, z_gb_prev = diagnostics_row(step, step * dt, f, e1, e2, p, Wc, dr, dz, r_c, r_f,
                                              z, z_gb_prev, lam, Vp0, Vs0, V0, a0, gamma_s, gamma_gb, W)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            a_over_a0_last = row["a_over_a0"]
            print(f"  [{tag}] step={step} t={row['t']:.4f} a/a0={a_over_a0_last:.5f} "
                  f"Vp/Vp0={row['Vp_frac']:.6f} mass_drift={row['mass_drift']:.2e} "
                  f"wall={time.time()-t_wall0:.0f}s")

            for thresh in (0.95, 0.90, 0.85):
                if a_over_a0_last <= thresh and thresh not in milestones_hit:
                    milestones_hit.add(thresh)
                    snap_path = os.path.join(CAMPAIGN_DIR, f"{tag}_a{thresh:.2f}.npz")
                    np.savez_compressed(snap_path, f=f, e1=e1, e2=e2, step=step, z=z)
                    print(f"  *** reached a/a0={thresh}: snapshot saved to {snap_path} ***")

            if a_over_a0_last <= target_a_over_a0:
                print(f"TARGET REACHED: a/a0={a_over_a0_last:.5f} <= {target_a_over_a0}")
                break
            if not np.isfinite(a_over_a0_last):
                print("GB tracking lost (topology change?) -- stopping for review")
                break

        if step - last_ckpt >= checkpoint_every_steps:
            last_ckpt = step
            # np.savez_compressed silently APPENDS ".npz" to any path that
            # doesn't already end with it -- ckpt_path+".tmp" doesn't, so
            # the actual file written was "..._checkpoint.npz.tmp.npz",
            # and os.replace(tmp_path, ckpt_path) below raised
            # FileNotFoundError (discovered when this crashed the first
            # production run at its first checkpoint, step 20000). Fixed
            # by giving the temp file its own valid ".npz" name instead.
            tmp_path = ckpt_path.replace(".npz", "_tmp.npz")
            np.savez_compressed(tmp_path, f=f, e1=e1, e2=e2, step=step, z=z)
            os.replace(tmp_path, ckpt_path)
            meta["z_gb_prev"] = z_gb_prev
            with open(meta_path, "w") as fh:
                json.dump(meta, fh)

        if wall_budget_s is not None and (time.time() - t_wall0) > wall_budget_s:
            print(f"wall budget {wall_budget_s}s exhausted at step {step}, a/a0={a_over_a0_last:.5f} -- checkpointing and stopping")
            break

    # final checkpoint
    np.savez_compressed(ckpt_path, f=f, e1=e1, e2=e2, step=step, z=z)
    meta["z_gb_prev"] = z_gb_prev
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    print(f"stopped at step={step}, t={step*dt:.4f}, a/a0={a_over_a0_last:.5f}, milestones_hit={sorted(milestones_hit)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi", type=float, default=160.0)
    ap.add_argument("--tag", type=str, default="c3_long")
    ap.add_argument("--target-a-over-a0", type=float, default=0.85)
    ap.add_argument("--max-steps", type=int, default=20_000_000)
    ap.add_argument("--wall-budget-s", type=float, default=None)
    ap.add_argument("--L-frac", type=float, default=3.0)
    ap.add_argument("--a-cap-frac", type=float, default=4.0)
    args = ap.parse_args()
    run(psi_deg=args.psi, tag=args.tag, target_a_over_a0=args.target_a_over_a0,
        max_steps=args.max_steps, wall_budget_s=args.wall_budget_s,
        L_frac=args.L_frac, a_cap_frac=args.a_cap_frac)
