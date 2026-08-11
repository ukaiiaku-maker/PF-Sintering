"""Milestone 16A Section 10 (corrected): the first convergence attempt
(m16a_stage_convergence.py) used the SAME step COUNT as the dx=2.5nm
baseline at finer dx/smaller W -- but finer dx forces a much smaller dt
(4th-order stability), so those runs only reached t~1.8 vs the
baseline's t~97.7, nowhere near the linear regime (confirmed: the
fitted "omega" there was dominated by the initial-transient relaxation,
not genuine PR dynamics). This version sizes the step budget from a
TARGET PHYSICAL TIME instead of a fixed step count, for a single
bounded, cost-controlled confirmatory case: lambda/R0=8 (clearly
unstable at baseline) at dx=1.25nm/W=20nm.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m16a_pr_benchmark import build_rod, find_stable_dt, r_centers_faces, run_case  # noqa: E402


class _P:
    def __init__(self, gamma_s=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W


def n_steps_for_target_t(R0_nm, lam_over_R0, dx_nm, W_nm, t_target, eps0_frac=0.05):
    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    lam = lam_over_R0 * R0
    Nz = max(16, round(lam / dz))
    Nr = max(24, round((R0 + 6 * W) / dr))
    p = _P(W=W)
    f, _ = build_rod(Nz, Nr, dr, dz, R0, W, eps0_frac * R0)
    r_c, r_f = r_centers_faces(Nr, dr)
    dt = find_stable_dt(f, p, dr, dz, r_c, r_f, 1e-33, W)
    dt *= 0.5
    return int(t_target / dt), dt


if __name__ == "__main__":
    n_steps, dt = n_steps_for_target_t(40.0, 8.0, 1.25, 20.0, t_target=30.0)
    print(f"dt={dt:.3e}, using n_steps_total={n_steps} to reach t~30")
    run_case("conv2_dx1.25_R0-40_lamR0-8", 40.0, 8.0, dx_nm=1.25, W_nm=20.0,
              n_steps_total=n_steps, n_sample=40)
