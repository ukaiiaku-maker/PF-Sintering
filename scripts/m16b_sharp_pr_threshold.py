"""Milestone 16B Sections 7-8: sharper-interface PR qualification +
threshold convergence.

M16A's own PR benchmark used R0=40nm/W=20nm (W/R0=0.5) -- too diffuse
for a precise threshold measurement (Section 7's own critique). This
builds a better-resolved family, W/R0<=0.15, W/dx>=8 (the milestone's
own example: R0=80nm/W=10nm/dx=1.25nm), repeats the bounded ladder
lambda/R0 in {5.5, 2*pi, 7, 8} with the scalar-mobility operator
(already cross-validated against the face-projected operator for sign/
neutral-location agreement in Section 6), fits omega(lambda/R0), and
locates the zero crossing by linear interpolation between the
straddling points -- requiring it near 2*pi (not exact equality at
finite W, per Section 8's own wording).

Uses the SAME target-physical-time sizing fix (not fixed step count)
established for Section 6 and m16a_stage_convergence2.py.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_surface_diffusion_step, axisym_volume, r_centers_faces  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_pr_benchmark import amplitude, build_rod, find_stable_dt  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16b_campaign")


class P:
    def __init__(self, gamma_s=1.0, W=10e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W


def run_one(R0_nm, lam_over_R0, dx_nm, W_nm, eps0_frac, t_target, n_sample, fit_frac=0.5):
    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    lam = lam_over_R0 * R0
    Nz = max(16, round(lam / dz))
    lam = Nz * dz
    Nr = max(24, round((R0 + 6 * W) / dr))
    eps0 = eps0_frac * R0

    p = P(gamma_s=1.0, W=W)
    f, lam_actual = build_rod(Nz, Nr, dr, dz, R0, W, eps0, n_periods=1)
    r_c, r_f = r_centers_faces(Nr, dr)
    M_s = 1e-33

    dt = find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, W)
    dt *= 0.5
    n_steps_total = max(1, int(t_target / dt))

    V0 = axisym_volume(f, r_c, dr, dz)
    amp0, _ = amplitude(f, r_c, dr)

    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))
    ts, amps = [], []
    step = 0
    for target in sample_steps:
        while step < target:
            f, _, _ = axisym_surface_diffusion_step(f, p, dr, dz, r_c, r_f, dt, M_s, W)
            step += 1
        amp, _ = amplitude(f, r_c, dr)
        ts.append(step * dt)
        amps.append(amp)

    ts_a = np.array(ts)
    amps_a = np.array(amps)
    end = max(3, int(len(ts_a) * fit_frac))
    sel = np.isfinite(amps_a[:end]) & (amps_a[:end] > 0)
    tt = ts_a[:end][sel]
    aa = np.log(amps_a[:end][sel])
    A = np.vstack([tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(A, aa, rcond=None)
    omega = float(coef[0])
    pred = A @ coef
    ss_res = float(np.sum((aa - pred) ** 2))
    ss_tot = float(np.sum((aa - np.mean(aa)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    print(f"  [R0={R0_nm}nm W={W_nm}nm dx={dx_nm}nm] lam/R0={lam_over_R0:.4f} n_steps={n_steps_total} "
          f"t_final={ts[-1]:.3e} omega={omega:.4e} r2={r2:.4f}")
    return dict(R0_nm=R0_nm, W_nm=W_nm, dx_nm=dx_nm, lam_over_R0=lam_over_R0, lam_nm=lam_actual * 1e9,
                dt=dt, n_steps_total=n_steps_total, omega=omega, r2=r2)


def find_crossing(cases):
    """Linear interpolation for the omega=0 crossing between the two
    points straddling the sign change, sorted by lam_over_R0."""
    cs = sorted(cases, key=lambda c: c["lam_over_R0"])
    for i in range(len(cs) - 1):
        o1, o2 = cs[i]["omega"], cs[i + 1]["omega"]
        if (o1 < 0) != (o2 < 0):
            x1, x2 = cs[i]["lam_over_R0"], cs[i + 1]["lam_over_R0"]
            frac = -o1 / (o2 - o1)
            return x1 + frac * (x2 - x1)
    return float("nan")


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "sharp_pr_threshold.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        ratios = [5.5, 2 * math.pi, 7.0, 8.0]
        results = []
        for lam_over_R0 in ratios:
            r = run_one(R0_nm=80.0, lam_over_R0=lam_over_R0, dx_nm=1.25, W_nm=10.0, eps0_frac=0.05,
                        t_target=15.0, n_sample=30)
            results.append(r)
            with open(out_path, "w") as fh:
                json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 7-8 summary ---")
    for r in results:
        print(f"lam/R0={r['lam_over_R0']:.4f} omega={r['omega']:.4e} r2={r['r2']:.4f}")
    xc = find_crossing(results)
    print(f"\nzero-crossing lam_c/R0 = {xc:.4f}  (target 2*pi = {2*math.pi:.4f}, "
          f"rel_err={abs(xc-2*math.pi)/(2*math.pi):.4f})")
    print(f"W/R0 = {10.0/80.0:.4f}  dx/W = {1.25/10.0:.4f}  W/dx = {10.0/1.25:.4f}")
