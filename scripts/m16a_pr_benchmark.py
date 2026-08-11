"""Milestone 16A Sections 7-10: single-crystal Plateau-Rayleigh benchmark.

Axisymmetric rod R(z,0)=R0+eps0*cos(2*pi*z/lambda), periodic z, no eta/
GB/substrate/sink/anisotropy. Measures the perturbation amplitude
eps(t) (half peak-to-trough of the f=0.5 contour R(z,t)) over time,
fits the linear-regime growth/decay rate omega via
log(eps(t)) ~ log(eps0) + omega*t, and compares sign/zero-crossing
against the classical (Mullins/Nichols) surface-diffusion dispersion
relation, whose defining feature is a neutral wavelength at
lambda/R0=2*pi -- decay for lambda/R0<2*pi, growth for lambda/R0>2*pi.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_free_energy, axisym_surface_diffusion_step, axisym_volume, r_centers_faces,
)

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16a_campaign")


class P:
    def __init__(self, gamma_s=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W


def build_rod(Nz, Nr, dr, dz, R0, W, eps0, n_periods=1):
    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    lam = Nz * dz / n_periods
    Rprofile = R0 + eps0 * np.cos(2 * math.pi * Z / lam)
    f = 0.5 * (1.0 - np.tanh((R - Rprofile) / W))
    return f, lam


def measure_R_of_z(f, r_c, dr):
    """f=0.5 crossing radius per z-row, linear interpolation."""
    Nz, Nr = f.shape
    R_of_z = np.full(Nz, np.nan)
    for j in range(Nz):
        row = f[j]
        idx = np.where((row[:-1] - 0.5) * (row[1:] - 0.5) < 0)[0]
        if len(idx) == 0:
            continue
        i = idx[-1]  # outermost crossing (in case of noise near axis)
        r0v, r1v = r_c[i], r_c[i + 1]
        f0v, f1v = row[i], row[i + 1]
        R_of_z[j] = r0v + (0.5 - f0v) * (r1v - r0v) / (f1v - f0v)
    return R_of_z


def amplitude(f, r_c, dr):
    R_of_z = measure_R_of_z(f, r_c, dr)
    valid = R_of_z[np.isfinite(R_of_z)]
    if len(valid) < 4:
        return float("nan"), R_of_z
    return 0.5 * (float(np.max(valid)) - float(np.min(valid))), R_of_z


def find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, W, n_check=60):
    dt = 1.0
    for _ in range(300):
        f_try = f.copy()
        stable = True
        for _ in range(n_check):
            f_try, _, _ = axisym_surface_diffusion_step(f_try, p, dr, dz, r_c, r_f, dt, M_s, W)
            if not np.all(np.isfinite(f_try)) or np.max(np.abs(f_try)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def run_case(case_id, R0_nm, lam_over_R0, dx_nm=2.5, W_nm=20.0, eps0_frac=0.05, n_steps_total=4000,
             n_sample=40, verbose=True):
    out_path = os.path.join(CAMPAIGN_DIR, f"{case_id}.json")
    if os.path.exists(out_path):
        print(f"[{case_id}] already done, skipping")
        with open(out_path) as fh:
            return json.load(fh)

    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    lam = lam_over_R0 * R0
    Nz = max(16, round(lam / dz))
    lam = Nz * dz  # snap exactly
    Nr = max(24, round((R0 + 6 * W) / dr))
    eps0 = eps0_frac * R0

    p = P(gamma_s=1.0, W=W)
    f, lam_actual = build_rod(Nz, Nr, dr, dz, R0, W, eps0, n_periods=1)
    r_c, r_f = r_centers_faces(Nr, dr)
    M_s = 1e-33

    dt = find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, W)
    dt *= 0.5  # safety margin below the empirical instability threshold

    V0 = axisym_volume(f, r_c, dr, dz)
    F0 = axisym_free_energy(f, p, dr, dz, r_c, r_f)
    amp0, _ = amplitude(f, r_c, dr)

    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))
    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, _, _ = axisym_surface_diffusion_step(f, p, dr, dz, r_c, r_f, dt, M_s, W)
            step += 1
        amp, R_of_z = amplitude(f, r_c, dr)
        V = axisym_volume(f, r_c, dr, dz)
        Fe = axisym_free_energy(f, p, dr, dz, r_c, r_f)
        row = dict(step=step, t=step * dt, amplitude=amp, V=V, F=Fe,
                   mass_drift=(V - V0) / V0, R_min=float(np.nanmin(R_of_z)),
                   R_max=float(np.nanmax(R_of_z)))
        rows.append(row)
        if verbose:
            print(f"  [{case_id}] step={step} t={row['t']:.4e} amp={amp*1e9:.4f}nm "
                  f"mass_drift={row['mass_drift']:.2e} F={Fe:.4e}")

    result = dict(case_id=case_id, R0_nm=R0_nm, lam_over_R0=lam_over_R0, lam_nm=lam_actual * 1e9,
                  dx_nm=dx_nm, W_nm=W_nm, eps0_nm=eps0 * 1e9, dt=dt, Nz=Nz, Nr=Nr,
                  amp0=amp0, F0=F0, V0=V0, rows=rows)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{case_id}] saved to {out_path}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--R0-nm", type=float, default=40.0)
    ap.add_argument("--lam-over-R0", type=float, required=True)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--W-nm", type=float, default=20.0)
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--case-id", type=str, required=True)
    args = ap.parse_args()
    run_case(args.case_id, args.R0_nm, args.lam_over_R0, dx_nm=args.dx_nm, W_nm=args.W_nm,
              n_steps_total=args.n_steps)
