"""Milestone 16G Section 10: parent Figure-4 (full periodic, two-GB)
trajectory at the SAME left-hand contact, for direct comparison against
the reduced particle/asperity geometry (scripts/
m16g_pr_derived_particle_asperity.py). Reuses the exact M16E/M16F
two-mode construction (periodic bc_z, unchanged) and tracks the LEFT GB
trough (z1) specifically -- M16F's own run only logged R(0) and
R(lambda), not the trough radius itself.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, build_exact_two_mode, find_stable_dt, grain_volumes, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_gb_trough  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16g_campaign")


def run_case(psi_deg=160.0, R_cyl_nm=100.0, W_nm=10.0, dx_nm=1.25, t_target=30.0, n_sample=40):
    R_cyl = R_cyl_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]

    state = build_exact_two_mode(R_cyl, W, dr, dz)
    f, e1, e2 = state["f"], state["e1"], state["e2"]
    Nz, Nr = state["Nz"], state["Nr"]
    r_c, r_f = r_centers_faces(Nr, dr)
    z = state["z"]
    z1 = state["z1"]
    lam = state["lam"]
    a0 = float(np.interp(z1, z, measure_R_of_z(f, r_c)))

    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    V1_0, V2_0 = grain_volumes(e1, e2, r_c, dr, dz)

    rows = []
    step = 0
    z_gb_prev = z1
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"[psi={psi_deg}] blew up at step {step}")
        R_of_z = measure_R_of_z(f, r_c)
        z_gb, a = find_gb_trough(R_of_z, z, z_gb_prev, lam=lam)
        if np.isfinite(z_gb):
            z_gb_prev = z_gb
        V1, V2 = grain_volumes(e1, e2, r_c, dr, dz)
        V = axisym_volume(f, r_c, dr, dz)
        rows.append(dict(step=step, t=step * dt, V1_frac=V1 / V1_0, V2_frac=V2 / V2_0,
                          mass_drift=(V - V0) / V0, z_gb_nm=z_gb * 1e9, a_nm=a * 1e9, a_over_a0=a / a0))
        print(f"  [PARENT psi={psi_deg:.0f}] t={rows[-1]['t']:.4f} V1/V1_0={rows[-1]['V1_frac']:.6f} "
              f"a={a*1e9:.4f}nm a/a0={rows[-1]['a_over_a0']:.6f} z_gb={z_gb*1e9:.3f}nm "
              f"mass_drift={rows[-1]['mass_drift']:.2e}")

    return dict(psi_deg=psi_deg, a0_nm=a0 * 1e9, Nz=Nz, Nr=Nr, dt=dt, n_steps_total=n_steps_total, rows=rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi", type=float, default=160.0)
    ap.add_argument("--t-target", type=float, default=30.0)
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    out_path = os.path.join(CAMPAIGN_DIR, f"parent_psi{args.psi:g}{args.tag}.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
    else:
        result = run_case(args.psi, t_target=args.t_target, n_sample=args.n_sample)
        with open(out_path, "w") as fh:
            json.dump(result, fh, default=str)
        print(f"saved to {out_path}")
