"""Milestone 16B Section 5: validate the axisymmetric face-projected
operator before any physics benchmark: exact volume conservation,
monotone F, and the exact dissipation identity Fdot_chain=-D_h (to
machine precision, subject only to time-discretization truncation)."""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_face_projected_step, axisym_free_energy, axisym_volume, r_centers_faces,
)


class P:
    def __init__(self, gamma_s=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W


def find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, W, n_check=60):
    dt = 1.0
    for _ in range(300):
        f_try = f.copy()
        stable = True
        for _ in range(n_check):
            f_try, _, _ = axisym_face_projected_step(f_try, p, dr, dz, r_c, r_f, dt, M_s, W)
            if not np.all(np.isfinite(f_try)) or np.max(np.abs(f_try)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


if __name__ == "__main__":
    Nz, Nr, dr, dz = 32, 24, 2.0e-9, 2.0e-9
    r_c, r_f = r_centers_faces(Nr, dr)
    p = P()
    W = 20e-9
    M_s = 1e-33

    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    R0 = 12e-9
    eps0 = 1e-9
    lam = Nz * dz
    Rprofile = R0 + eps0 * np.cos(2 * math.pi * Z / lam)
    f = 0.5 * (1.0 - np.tanh((R - Rprofile) / W))

    dt = find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, W)
    dt *= 0.5
    V0 = axisym_volume(f, r_c, dr, dz)
    F0 = axisym_free_energy(f, p, dr, dz, r_c, r_f)
    print(f"V0={V0:.6e} F0={F0:.6e} dt={dt:.3e}")

    max_dV = 0.0
    max_dF = 0.0
    Fprev = F0
    D_h_neg_count = 0
    fdot_err = []
    for step in range(50):
        f_new, mu, diag = axisym_face_projected_step(f, p, dr, dz, r_c, r_f, dt, M_s, W)
        V_new = axisym_volume(f_new, r_c, dr, dz)
        F_new = axisym_free_energy(f_new, p, dr, dz, r_c, r_f)
        max_dV = max(max_dV, abs(V_new - V0) / abs(V0))
        max_dF = max(max_dF, F_new - Fprev)
        if diag["D_h"] < 0:
            D_h_neg_count += 1
        # exact dissipation identity: Fdot_chain should equal -D_h
        rel = abs(diag["Fdot_chain"] - (-diag["D_h"])) / max(abs(diag["D_h"]), 1e-30)
        fdot_err.append(rel)
        f = f_new
        Fprev = F_new

    print(f"max relative |V-V0|/V0: {max_dV:.3e}")
    print(f"max (F(t+dt)-F(t)) [must be <=0]: {max_dF:.3e}")
    print(f"D_h negative count (must be 0): {D_h_neg_count}")
    print(f"max relative |Fdot_chain - (-D_h)| / D_h: {max(fdot_err):.3e}")
    ok = max_dV < 1e-10 and max_dF <= 1e-20 and D_h_neg_count == 0 and max(fdot_err) < 1e-8
    print(f"PASS={ok}")
