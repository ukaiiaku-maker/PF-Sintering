"""M16M continuation Sections 3-4: separate W-convergence from curvature-
window convergence.

The first W/dx comparison (W=10/6/4nm) used each resolution's own
"official" r_neck definition (a 1.5*W half-window circle fit), so
changing W simultaneously changed the ABSOLUTE physical fitting window
(15nm -> 9nm -> 6nm) -- confounding two different effects. This script
removes that confound: for each of several diffuse-interface widths W,
run a short SINK-OFF (deterministic, no RBM) reference relaxation and, at
each saved sample, evaluate r_neck/sigma over the SAME FIXED PHYSICAL
half-window ladder (independent of W), plus report window/W for context
(Section 4's reciprocal test).

Also records X_neck (R(z)-minimum-based) at each sample so different W
trajectories can be compared at MATCHED MORPHOLOGY rather than matched
model time (Section 5) -- the first comparison found the t=0 diffuse
construction is not an equivalent relaxed state across W (a fast, large
transient at W=4nm absent at W=10/6nm), so matched-time comparisons
before that transient has relaxed away are not meaningful.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
RATIO = 0.10
FIXED_WINDOWS_NM = (6.0, 9.0, 12.0, 15.0, 18.0, 24.0)

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16m_window_W_convergence")


def run_one(W_nm, dx_nm, t_target, n_samples):
    os.makedirs(OUT_ROOT, exist_ok=True)
    label = f"W{W_nm:g}dx{dx_nm:g}"
    t0 = time.time()

    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_nm, dr_nm=dx_nm, dz_nm=dx_nm, aspect_ratio=1.0)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = W_nm * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    n_steps_total = max(1, int(t_target / dt))
    diag_every = max(1, n_steps_total // n_samples)
    print(f"[{label}] Nz={f.shape[0]} Nr={f.shape[1]} dt={dt:.4e} n_steps_total={n_steps_total}", flush=True)

    rows = []
    V0_mass = axisym_volume(f, r_c, dr, dz)
    for step in range(0, n_steps_total + 1):
        if step > 0:
            f, e1, e2, _ = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                                           bc_z="noflux")
            if not np.all(np.isfinite(f)):
                print(f"[{label}] BLOWUP at step {step}"); break

        if step % diag_every != 0:
            continue
        t = step * dt
        R_of_z = measure_R_of_z(f, r_c)
        ext = find_all_extrema(R_of_z, z)
        # NOTE: "first found" (z-scan order), matching the established
        # convention in scripts/m16j_stage1_short_pf_screen.py, NOT sorted
        # by radius -- multiple candidate minima appear from early times
        # (M16K finding), and sorting by radius silently picks a DIFFERENT
        # candidate once that happens, giving a non-comparable trajectory.
        mins = [(zz, RR) for k, zz, RR in ext if k == "min"]
        if not mins:
            continue
        z_min, a_contact = mins[0]
        X_neck = 2 * a_contact
        windows = neck_curvature_windows(R_of_z, z, z_min, 1e-9, window_widths_in_W=FIXED_WINDOWS_NM)
        V = axisym_volume(f, r_c, dr, dz)
        row = dict(W_nm=W_nm, dx_nm=dx_nm, step=step, time=t, z_min_nm=z_min * 1e9, X_neck_nm=X_neck * 1e9,
                   mass_drift=(V - V0_mass) / V0_mass)
        for w_nm, win in zip(FIXED_WINDOWS_NM, windows):
            r_neck = win["r_neck"]
            if np.isfinite(r_neck) and r_neck > 0:
                sigma, *_ = hussein_eq1b_sigma(r_neck, X_neck, GAMMA_S, GAMMA_GB)
            else:
                sigma = float("nan")
            row[f"r_neck_{w_nm:g}nm"] = r_neck * 1e9 if np.isfinite(r_neck) else float("nan")
            row[f"sigma_{w_nm:g}nm_MPa"] = sigma / 1e6 if np.isfinite(sigma) else float("nan")
            row[f"window_over_W_{w_nm:g}nm"] = w_nm / W_nm
        rows.append(row)
        print(f"[{label}] t={t:.3f} X_neck={row['X_neck_nm']:.3f}nm "
              f"sigma_15nm={row.get('sigma_15nm_MPa', float('nan')):.3f}MPa wall={time.time()-t0:.0f}s", flush=True)

    out_path = os.path.join(OUT_ROOT, f"{label}.csv")
    if rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[{label}] DONE wall={time.time()-t0:.0f}s -- wrote {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-nm", type=float, required=True)
    ap.add_argument("--dx-nm", type=float, required=True)
    ap.add_argument("--t-target", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=40)
    args = ap.parse_args()
    run_one(args.w_nm, args.dx_nm, args.t_target, args.n_samples)
