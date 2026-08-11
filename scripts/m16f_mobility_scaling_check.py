"""Milestone 16F correction, item 10: before concluding anything from the
slow-evolving t=60 runs, verify whether scaling (M_s, M_GB) -> C*(M_s,
M_GB) at fixed ratio M_GB/M_s simply rescales physical time by 1/C (exact
symmetry of the gradient-flow PDE: dR/dt = M*L[R] with L independent of
M implies R_C(t') = R_1(C*t')). If true, running at a larger C and a
proportionally smaller t' produces the identical physical state as the
baseline at t=C*t' -- but for an EXPLICIT time-stepping scheme, the
CFL-stable dt is also expected to scale as 1/C (same symmetry applied to
the stability bound itself), so the number of steps to reach a given
reduced/dimensionless time -- and hence wall-clock cost -- should be
UNCHANGED. This script checks both parts empirically: (a) does the state
collapse under t->C*t, and (b) is there any real wall-clock benefit.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, build_exact_two_mode, find_stable_dt, grain_volumes, psi_to_gamma_gb  # noqa: E402


def run_scaled(psi_deg, C, t_target_baseline, R_cyl_nm=100.0, W_nm=10.0, dx_nm=1.25):
    """Runs the C*M system for t' = t_target_baseline / C, which should
    reproduce the baseline (C=1) state at t=t_target_baseline if the
    time-rescaling symmetry holds exactly in this discretization."""
    R_cyl = R_cyl_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]

    state = build_exact_two_mode(R_cyl, W, dr, dz)
    f, e1, e2 = state["f"], state["e1"], state["e2"]
    r_c, r_f = r_centers_faces(state["Nr"], dr)

    M_s = C * 1e-33
    M_eta = M_s / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100)
    dt *= 0.4
    t_prime_target = t_target_baseline / C
    n_steps = max(1, int(round(t_prime_target / dt)))

    V1_0, V2_0 = grain_volumes(e1, e2, r_c, dr, dz)
    V0 = axisym_volume(f, r_c, dr, dz)

    t0 = time.time()
    for _ in range(n_steps):
        f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
    wall = time.time() - t0

    V1, V2 = grain_volumes(e1, e2, r_c, dr, dz)
    V = axisym_volume(f, r_c, dr, dz)
    R_of_z = measure_R_of_z(f, r_c)
    return dict(C=C, dt=dt, n_steps=n_steps, t_prime_reached=n_steps * dt,
                t_equiv_baseline=C * n_steps * dt, wall_s=wall,
                V1_frac=V1 / V1_0, V2_frac=V2 / V2_0, mass_drift=(V - V0) / V0,
                R_of_0_nm=float(R_of_z[0]) * 1e9, R_of_lam_nm=float(R_of_z[len(R_of_z) // 2]) * 1e9)


if __name__ == "__main__":
    psi = 160.0
    t_target = 0.3
    print(f"psi={psi}, target baseline-equivalent t={t_target}")
    results = []
    for C in (1.0, 10.0, 100.0):
        r = run_scaled(psi, C, t_target)
        results.append(r)
        print(f"  C={C:6.1f}  n_steps={r['n_steps']:7d}  dt={r['dt']:.4e}  wall={r['wall_s']:7.2f}s  "
              f"t'_reached={r['t_prime_reached']:.4e}  t_equiv={r['t_equiv_baseline']:.4f}  "
              f"V1/V1_0={r['V1_frac']:.8f}  R(0)={r['R_of_0_nm']:.5f}nm  R(lam)={r['R_of_lam_nm']:.5f}nm  "
              f"mass_drift={r['mass_drift']:.2e}")

    base = results[0]
    print("\nCollapse check (state vs baseline C=1):")
    for r in results[1:]:
        dV1 = r["V1_frac"] - base["V1_frac"]
        dR0 = r["R_of_0_nm"] - base["R_of_0_nm"]
        print(f"  C={r['C']:6.1f}: dV1_frac={dV1:+.3e}  dR(0)={dR0:+.3e}nm  "
              f"speedup(steps_base/steps_C)={base['n_steps']/r['n_steps']:.3f}x")
