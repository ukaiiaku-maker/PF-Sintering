"""Milestone 16B Section 19: direct stability test of the M15 neck,
replacing visual guesses with a measured early-time growth/decay rate.

Symmetric perturbations {-2%,-1%,0,+1%,+2%} in neck (contact) radius,
AT FIXED PARTICLE VOLUME, on the Section 17/18 axisymmetric M15-flat-
substrate analogue. Neck size is controlled by `overlap_nm` (more
overlap -> larger neck where the spheroid meets the flat substrate
wall); for each perturbed overlap, particle volume is restored to
EXACTLY the base case's V_particle_analytic by a small compensating
isotropic rescale of (Rz,Rr) (solved by bisection against the closed-
form spheroid-minus-cap volume formula in axisym_m15_flat_volumes --
no simulation needed for this part, it is pure geometry).

Classification: run the (0%) base case and all four perturbed cases
under the IDENTICAL dynamics (axisym_gb_face_projected_step) for the
same short EARLY-time window, and track
    delta_neck(t) = neck_perturbed(t) - neck_base(t)
i.e. the perturbation measured RELATIVE TO the (already-evolving, per
Section 18) base trajectory -- exactly the standard linearized-
stability-around-a-possibly-time-dependent-base-state construction.
Fit log|delta_neck(t)| ~ log|delta_neck(0)| + omega*t in the early
regime: omega>0 classifies unstable (perturbation amplifies relative
to the base evolution), omega<0 stable (perturbation decays, base
trajectory attracts back), |omega| small/not clearly signed neutral.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_gb_face_projected_step, axisym_m15_flat_geometry, axisym_m15_flat_volumes, axisym_reproject,
    axisym_volume, r_centers_faces,
)
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16b_com_no_sink_test import P, find_stable_dt, measure_neck_radius  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16b_campaign")


def solve_scale_for_volume(Rz0_nm, Rr0_nm, overlap_nm, wall_z_frac, Nz, Nr, dz, dr, W, target_V,
                            n_iter=40):
    """Bisect a common isotropic scale s (Rz=s*Rz0, Rr=s*Rr0) so that
    axisym_m15_flat_volumes' V_particle_analytic matches target_V at
    this overlap -- pure closed-form geometry, no simulation."""
    def vol_at(s):
        geom = axisym_m15_flat_geometry(Nz, Nr, dz, dr, W, Rz_nm=s * Rz0_nm, Rr_nm=s * Rr0_nm,
                                         overlap_nm=overlap_nm, wall_z_frac=wall_z_frac)
        r_c, r_f = r_centers_faces(Nr, dr)
        return axisym_m15_flat_volumes(geom, r_c, dr, dz)["V_particle_analytic"]

    lo, hi = 0.5, 1.5
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        v = vol_at(mid)
        if v < target_V:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_perturbation(pct, Rz0_nm, Rr0_nm, base_overlap_nm, target_V, Nz, Nr, dx_nm, W_nm, wall_z_frac,
                      gamma_gb, t_target, n_sample):
    dr = dz = dx_nm * 1e-9
    W = W_nm * 1e-9
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]

    # perturb overlap by a small delta chosen empirically so the
    # resulting neck-radius shift is close to the target pct (a linear
    # proxy; exact pct is measured post-hoc from the actual constructed
    # geometry, not assumed).
    overlap_nm = base_overlap_nm * (1.0 + 3.0 * pct)  # empirical proxy scale, refined below by direct measurement
    scale = solve_scale_for_volume(Rz0_nm, Rr0_nm, overlap_nm, wall_z_frac, Nz, Nr, dz, dr, W, target_V)
    geom = axisym_m15_flat_geometry(Nz, Nr, dz, dr, W, Rz_nm=scale * Rz0_nm, Rr_nm=scale * Rr0_nm,
                                     overlap_nm=overlap_nm, wall_z_frac=wall_z_frac)
    r_c, r_f = r_centers_faces(Nr, dr)
    V_check = axisym_m15_flat_volumes(geom, r_c, dr, dz)["V_particle_analytic"]

    f = geom["f"]
    e1, e2 = axisym_reproject(f, geom["e1_raw"].copy(), geom["e2_raw"].copy())
    z = (np.arange(Nz) + 0.5) * dz
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    neck0 = measure_neck_radius(f, r_c, z, geom["wall_z"], dz)

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    ts, necks = [], []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"blew up at step {step}")
        neck = measure_neck_radius(f, r_c, z, geom["wall_z"], dz)
        ts.append(step * dt)
        necks.append(neck)

    V_final = axisym_volume(f, r_c, dr, dz)
    print(f"  [pct={pct:+.3f}] overlap={overlap_nm:.4f}nm scale={scale:.6f} V_analytic_check_rel_err="
          f"{abs(V_check-target_V)/target_V:.2e} neck0={neck0*1e9:.4f}nm dt={dt:.3e} n_steps={n_steps_total} "
          f"mass_drift={(V_final-V0)/V0:.2e}")

    return dict(pct=pct, overlap_nm=overlap_nm, scale=scale, neck0_nm=neck0 * 1e9, dt=dt,
                n_steps_total=n_steps_total, ts=ts, necks_nm=[n * 1e9 for n in necks],
                mass_drift=(V_final - V0) / V0)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "neck_stability.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        Nz, Nr, dx_nm, W_nm = 200, 120, 2.0, 10.0
        Rz0_nm, Rr0_nm, base_overlap_nm, wall_z_frac = 113.137, 56.569, 20.0, 0.2
        dr = dz = dx_nm * 1e-9
        W = W_nm * 1e-9
        geom0 = axisym_m15_flat_geometry(Nz, Nr, dz, dr, W, Rz_nm=Rz0_nm, Rr_nm=Rr0_nm,
                                          overlap_nm=base_overlap_nm, wall_z_frac=wall_z_frac)
        r_c, r_f = r_centers_faces(Nr, dr)
        target_V = axisym_m15_flat_volumes(geom0, r_c, dr, dz)["V_particle_analytic"]
        print(f"target particle volume: {target_V:.6e}")

        results = []
        for pct in (0.0, 0.01, -0.01, 0.02, -0.02):
            r = run_perturbation(pct, Rz0_nm, Rr0_nm, base_overlap_nm, target_V, Nz, Nr, dx_nm, W_nm,
                                  wall_z_frac, gamma_gb=0.6, t_target=0.4, n_sample=25)
            results.append(r)
            with open(out_path, "w") as fh:
                json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 19 summary ---")
    base = next(r for r in results if r["pct"] == 0.0)
    for r in results:
        if r["pct"] == 0.0:
            continue
        ts = np.array(r["ts"])
        d_neck = np.array(r["necks_nm"]) - np.array(base["necks_nm"][:len(r["necks_nm"])])
        d0 = r["neck0_nm"] - base["neck0_nm"]
        print(f"pct={r['pct']:+.3f}: initial delta_neck={d0:.4f}nm final delta_neck={d_neck[-1]:.4f}nm")
        sel = np.abs(d_neck) > 1e-6
        if np.sum(sel) >= 3:
            aa = np.log(np.abs(d_neck[sel]))
            tt = ts[sel]
            A = np.vstack([tt, np.ones_like(tt)]).T
            coef, *_ = np.linalg.lstsq(A, aa, rcond=None)
            omega = float(coef[0])
            sign = "UNSTABLE" if omega > 1e-3 else ("STABLE" if omega < -1e-3 else "NEUTRAL")
            print(f"    fitted omega={omega:.4e} -> {sign}")
