"""Milestone 16B Sections 13-14: Hussein et al. (2022, Appl. Phys. Lett.
121, 141601) one-mode benchmark, corrected initial condition.

M16A did NOT reproduce the paper's actual initial condition -- it
started from a PERFECT cylinder and let the GB's own thermal groove
create the only perturbation. Section 13 requires the ACTUAL one-mode
IC: R(z,0)=R0+epsilon1*cos(2*pi*z/lambda), epsilon2=0, GBs placed AT
THE TROUGH positions, lambda/R_cyl=2*pi, epsilon1_bar~0.2.

To place TWO GBs (matching this project's established two-grain bamboo
convention, GBs at z=L/4, 3L/4 of the total periodic domain -- see
m16a_gb_benchmark.py/m16b_young_herring_test.py) each exactly at a
trough of a single-mode cos(2*pi*z/lambda) profile, the domain must
span TWO full wavelengths: L=2*lambda. Then cos(2*pi*z/lambda) has
minima (troughs) at z=lambda/2=L/4 and z=3*lambda/2=3L/4 -- exactly
the existing GB convention, verified directly below before running any
dynamics.

Tracks (Section 14, the paper's Figure-3-type benchmark):
    R_GB(t)        = R(z=L/4, t)          (at the GB / trough)
    R_grain_max(t) = R(z=0, t)             (mid-grain / crest)
    R_norm(t) = [R_GB(t)/R_grain_max(t)] / [R_GB(0)/R_grain_max(0)]
for psi in {140,120,100} deg (gamma_gb via
gamma_gb=2*gamma_s*cos(psi/2)). Requires R_norm to decrease over time,
and to decrease FASTER (more negative slope) for lower psi (larger
gamma_gb) -- the paper's central qualitative trend.
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
from m16a_gb_benchmark import measure_R_of_z, psi_to_gamma_gb  # noqa: E402
from m16b_young_herring_test import measure_gb_position  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16b_campaign")


class P:
    def __init__(self, gamma_s=1.0, gamma_gb=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W
        self.k_eta = gb_obstacle_coefficients(gamma_gb, W)["k_eta"]


def build_hussein_one_mode(Nz, Nr, dr, dz, R0, W, eps1_bar=0.2):
    """R(z,0)=R0+eps1*cos(2*pi*z/lambda), eps2=0, domain L=2*lambda
    (lambda/R_cyl=2*pi => lambda=2*pi*R0), GBs (via the g(z) ownership
    split) placed at z=L/4, 3L/4 -- verified below to coincide with the
    profile's trough positions exactly."""
    lam = 2 * math.pi * R0
    L_target = 2 * lam
    Nz = max(24, round(L_target / dz))
    if Nz % 4:
        Nz += 4 - (Nz % 4)  # keep L/4, 3L/4 landing exactly on grid rows
    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    L = Nz * dz
    eps1 = eps1_bar * R0
    Rprofile_1d = R0 + eps1 * np.cos(2 * math.pi * z / lam)
    Rprofile = np.broadcast_to(Rprofile_1d[:, None], (Nz, Nr))
    f = 0.5 * (1.0 - np.tanh((R - Rprofile) / W))

    # sanity: troughs at L/4, 3L/4
    j_trough1 = int(round((L / 4.0) / dz))
    j_trough2 = int(round((3.0 * L / 4.0) / dz))
    assert Rprofile_1d[j_trough1] < R0 - 0.9 * eps1, "trough1 not at L/4 as expected"
    assert Rprofile_1d[j_trough2] < R0 - 0.9 * eps1, "trough2 not at 3L/4 as expected"

    g = 0.5 * (1 - np.cos(2 * math.pi * Z / L))
    e2 = f * g
    e1 = f - e2
    return f, e1, e2, L, lam


def find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=200):
    dt = 1.0
    for _ in range(300):
        f_t, e1_t, e2_t = f.copy(), e1.copy(), e2.copy()
        stable = True
        for _ in range(n_check):
            f_t, e1_t, e2_t, _ = axisym_gb_face_projected_step(f_t, e1_t, e2_t, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            if not np.all(np.isfinite(f_t)) or np.max(np.abs(f_t)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def run_case(psi_deg, R0_nm=40.0, W_nm=20.0, dx_nm=2.5, eps1_bar=0.2, t_target=5.0, n_sample=25):
    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    Nr = max(24, round((R0 + 6 * W) / dr))
    Nz0 = 10  # placeholder, recomputed inside build_hussein_one_mode

    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]
    f, e1, e2, L, lam = build_hussein_one_mode(Nz0, Nr, dr, dz, R0, W, eps1_bar=eps1_bar)
    Nz = f.shape[0]
    r_c, r_f = r_centers_faces(Nr, dr)
    z = (np.arange(Nz) + 0.5) * dz
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    R_of_z0 = measure_R_of_z(f, r_c)
    j_gb, j_mid = int(round(L / 4.0 / dz)), 0
    R_GB0, R_max0 = R_of_z0[j_gb], R_of_z0[j_mid]
    ratio0 = R_GB0 / R_max0

    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"blew up at step {step}")
        R_of_z = measure_R_of_z(f, r_c)
        R_GB, R_max = R_of_z[j_gb], R_of_z[j_mid]
        V = axisym_volume(f, r_c, dr, dz)
        ratio = R_GB / R_max if R_max else float("nan")
        rows.append(dict(step=step, t=step * dt, R_GB_nm=R_GB * 1e9, R_max_nm=R_max * 1e9,
                          R_norm=ratio / ratio0, mass_drift=(V - V0) / V0))
        print(f"  [psi={psi_deg:.0f}] step={step} t={rows[-1]['t']:.3e} R_GB={R_GB*1e9:.3f}nm "
              f"R_max={R_max*1e9:.3f}nm R_norm={rows[-1]['R_norm']:.5f} mass_drift={rows[-1]['mass_drift']:.2e}")

    return dict(psi_deg=psi_deg, R0_nm=R0_nm, W_nm=W_nm, dx_nm=dx_nm, lam_nm=lam * 1e9, L_nm=L * 1e9,
                dt=dt, n_steps_total=n_steps_total, R_GB0_nm=R_GB0 * 1e9, R_max0_nm=R_max0 * 1e9, rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "hussein_one_mode.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        results = []
        for psi_deg in (140.0, 120.0, 100.0):
            r = run_case(psi_deg, t_target=5.0, n_sample=25)
            results.append(r)
            with open(out_path, "w") as fh:
                json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 14 summary ---")
    for r in results:
        final = r["rows"][-1]
        slope = (final["R_norm"] - 1.0) / final["t"] if final["t"] > 0 else float("nan")
        print(f"psi={r['psi_deg']:.0f}: R_norm(0)=1.0 -> R_norm(t_final)={final['R_norm']:.5f} "
              f"(t_final={final['t']:.3e}) approx_slope={slope:.4e}")
