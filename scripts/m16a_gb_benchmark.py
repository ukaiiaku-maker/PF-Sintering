"""Milestone 16A Sections 11-13: axisymmetric GB destabilization
benchmark (Hussein et al. 2022-style).

A periodic domain of length lambda=2*pi*R0 (Section 13's canonical
lambda/R_cyl=2*pi test point) containing TWO grains separated by TWO
grain boundaries (at z=0/wrap and z=lambda/2), matching the paper's
periodic bamboo-grain fiber geometry -- NOT an externally-imposed
sinusoidal perturbation; the GB's own thermal groove is the sole
source of instability here (Section 12's physical mechanism). Starts
from a perfectly uniform rod R(z,0)=R0 (no shape perturbation at all)
-- the groove that forms at each GB is entirely a consequence of the
GB/coupling energy imbalance (dihedral-angle relation), not an initial
condition choice.

gamma_GB/gamma_s set via the equilibrium dihedral-angle relation
psi=2*acos(gamma_GB/(2*gamma_s)) for target psi in {100,120,140}deg.

Tracks R_GB(t) (at the GB, z=lambda/2) and R_max(t) (mid-grain, z=
lambda/4) -- the paper's R(t)=R(lambda/2,t)/R(lambda,t) analogue.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_gb_step, axisym_volume, r_centers_faces  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16a_campaign")


class P:
    def __init__(self, gamma_s=1.0, gamma_gb=1.0, W=20e-9, k_eta=None, Wc=None):
        self.gamma_s = gamma_s
        self.gamma_gb = gamma_gb
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W
        # obstacle calibration (gb_obstacle_energy.gb_obstacle_coefficients),
        # reused directly -- same formulas already validated in the
        # Cartesian production path.
        self.k_eta = 4.0 * gamma_gb * W / math.pi ** 2
        self.Wc = 4.0 * gamma_gb / W


def psi_to_gamma_gb(psi_deg, gamma_s=1.0):
    return 2.0 * gamma_s * math.cos(math.radians(psi_deg) / 2.0)


def build_two_grain_rod(Nz, Nr, dr, dz, R0, W):
    """Uniform R(z)=R0, two grains split at z=0(wrap)/z=Nz*dz/2."""
    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    f = 0.5 * (1.0 - np.tanh((R - R0) / W))
    L = Nz * dz
    half = L / 2.0
    # grain-ownership split along z, smooth transition of width ~W at
    # each of the two GB locations (z=0/wrap and z=half)
    d_to_gb1 = np.minimum(np.abs(Z - 0.0), L - np.abs(Z - 0.0))  # distance to z=0 (periodic)
    d_to_gb2 = np.abs(Z - half)
    # signed: positive means "grain 2 side" near gb1, handled via a
    # smooth indicator that is 0 near z=0, 1 near z=half, back to 0 near z=L
    g = 0.5 * (1 - np.cos(2 * math.pi * Z / L))  # 0 at z=0 and z=L(periodic), 1 at z=half/L*... check below
    e2 = f * g
    e1 = f - e2
    return f, e1, e2


def measure_R_of_z(f, r_c):
    Nz, Nr = f.shape
    R_of_z = np.full(Nz, np.nan)
    for j in range(Nz):
        row = f[j]
        idx = np.where((row[:-1] - 0.5) * (row[1:] - 0.5) < 0)[0]
        if len(idx) == 0:
            continue
        i = idx[-1]
        r0v, r1v = r_c[i], r_c[i + 1]
        f0v, f1v = row[i], row[i + 1]
        R_of_z[j] = r0v + (0.5 - f0v) * (r1v - r0v) / (f1v - f0v)
    return R_of_z


def find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=40):
    dt = 1.0
    for _ in range(300):
        f_try, e1_try, e2_try = f.copy(), e1.copy(), e2.copy()
        stable = True
        for _ in range(n_check):
            f_try, e1_try, e2_try, _ = axisym_gb_step(f_try, e1_try, e2_try, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            if not np.all(np.isfinite(f_try)) or np.max(np.abs(f_try)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def run_case(case_id, psi_deg, R0_nm=40.0, W_nm=20.0, dx_nm=2.5, n_steps_total=40000, n_sample=40):
    out_path = os.path.join(CAMPAIGN_DIR, f"{case_id}.json")
    if os.path.exists(out_path):
        print(f"[{case_id}] already done, skipping")
        with open(out_path) as fh:
            return json.load(fh)

    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    lam = 2 * math.pi * R0
    Nz = max(24, round(lam / dz))
    if Nz % 2:
        Nz += 1
    lam = Nz * dz
    Nr = max(24, round((R0 + 6 * W) / dr))

    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    f, e1, e2 = build_two_grain_rod(Nz, Nr, dr, dz, R0, W)
    r_c, r_f = r_centers_faces(Nr, dr)
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))  # rough same-order-of-magnitude choice, no physical claim

    dt = find_stable_dt(f, e1, e2, p, p.Wc, dr, dz, r_c, r_f, M_s, M_eta, W)
    dt *= 0.5

    V0 = axisym_volume(f, r_c, dr, dz)
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))
    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, _ = axisym_gb_step(f, e1, e2, p, p.Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
        R_of_z = measure_R_of_z(f, r_c)
        # GBs are where g=0.5*(1-cos(2*pi*Z/L)) crosses 0.5, i.e. at
        # Z=L/4 and Z=3L/4 (NOT at Z=0/L/2, which are mid-grain: g=0 or
        # g=1 there, full single-grain ownership) -- fixed after finding
        # this index swap empirically (the first version showed the
        # OPPOSITE of the expected groove-deepens-at-GB trend, tracing
        # directly to this).
        j_gb = Nz // 4
        j_mid = 0
        R_GB = R_of_z[j_gb] if np.isfinite(R_of_z[j_gb]) else float("nan")
        R_max_val = R_of_z[j_mid] if np.isfinite(R_of_z[j_mid]) else float("nan")
        V = axisym_volume(f, r_c, dr, dz)
        row = dict(step=step, t=step * dt, R_GB=R_GB, R_mid=R_max_val,
                   ratio=(R_GB / R_max_val) if R_max_val else float("nan"),
                   mass_drift=(V - V0) / V0)
        rows.append(row)
        print(f"  [{case_id}] step={step} t={row['t']:.4e} R_GB={R_GB*1e9:.3f}nm R_mid={R_max_val*1e9:.3f}nm "
              f"ratio={row['ratio']:.4f} mass_drift={row['mass_drift']:.2e}")

    result = dict(case_id=case_id, psi_deg=psi_deg, gamma_gb=gamma_gb, R0_nm=R0_nm, W_nm=W_nm, dx_nm=dx_nm,
                  lam_nm=lam * 1e9, dt=dt, rows=rows)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{case_id}] saved to {out_path}")
    return result


if __name__ == "__main__":
    for psi_deg in (140.0, 100.0):
        run_case(f"gb_psi{psi_deg:g}", psi_deg)
