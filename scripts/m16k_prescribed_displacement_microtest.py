"""M16K continuation, Section 8 (second handoff) / Section 4 (first
handoff): frozen-state prescribed-displacement microtest.

Reproduces the deterministic (sink-inactive, no hazard, no RNG) PF
trajectory up to the exact step where the prior first-event run
activated (step=90490, t=4.4185, sigma_Hussein=45.12 MPa) -- bit-
identical since this segment has zero randomness. Freezes that state,
then applies a SERIES of conservative (mass-preserving) prescribed
rigid-body translations dz to the frozen state (a pure grid-remap,
distinct from the flawed rbm_step/active_sink_transport_step numerics)
and measures the resulting X_neck, r_neck, sigma_Hussein, and mass for
each -- establishing the TRUE mechanical response sigma(delta_event)
independent of any RBM implementation bug.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import shift as ndi_shift

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
W_NM = 10.0
DX_NM = 1.25
TARGET_STEP = 90490

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16k_prescribed_displacement_microtest")
os.makedirs(OUT, exist_ok=True)


def diagnose(f, e1, e2, r_c, z, W, V_ref):
    R_of_z = measure_R_of_z(f, r_c)
    ext = find_all_extrema(R_of_z, z)
    mins = [(zz, RR) for k, zz, RR in ext if k == "min"]
    if not mins:
        return dict(a_contact_nm=float("nan"), X_neck_nm=float("nan"), r_neck_nm=float("nan"),
                     sigma_Hussein_MPa=float("nan"), mass_drift=float("nan"))
    z_gb, a = mins[0]
    X_neck = 2 * a
    win = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5,))[0]
    r_neck = win["r_neck"]
    if np.isfinite(r_neck) and r_neck > 0:
        sigma_H, _, _, _ = hussein_eq1b_sigma(r_neck, X_neck, GAMMA_S, GAMMA_GB)
    else:
        sigma_H = float("nan")
    V = 2 * math.pi * float(np.sum(r_c[None, :] * f)) * (z[1] - z[0]) * (r_c[1] - r_c[0])
    return dict(a_contact_nm=a * 1e9, X_neck_nm=X_neck * 1e9,
                r_neck_nm=r_neck * 1e9 if np.isfinite(r_neck) else float("nan"),
                sigma_Hussein_MPa=sigma_H / 1e6 if np.isfinite(sigma_H) else float("nan"),
                mass_drift=(V - V_ref) / V_ref)


def main():
    t0 = time.time()
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = W_NM * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))
    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4

    print(f"reproducing deterministic trajectory to step={TARGET_STEP} (t={TARGET_STEP*dt:.4f})...", flush=True)
    for step in range(1, TARGET_STEP + 1):
        f, e1, e2, _ = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                                       bc_z="noflux")
        if step % 20000 == 0:
            print(f"  step={step} t={step*dt:.3f} wall={time.time()-t0:.0f}s", flush=True)

    V_ref = axisym_volume(f, r_c, dr, dz)
    baseline = diagnose(f, e1, e2, r_c, z, W, V_ref)
    print(f"frozen state at step={TARGET_STEP}: {baseline}", flush=True)
    np.savez_compressed(os.path.join(OUT, "frozen_state.npz"), f=f, e1=e1, e2=e2, step=TARGET_STEP,
                         dz_grid=dz, dr_grid=dr)

    dz_list_nm = [0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.250]
    rows = []
    for dz_nm_applied in dz_list_nm:
        # conservative prescribed rigid-body shift along z (axis 0), via
        # spline interpolation (order=1, mode='nearest' at the domain
        # ends -- no flux across the boundary) -- a PURE, exact grid
        # remap distinct from rbm_step's own upwind-advection numerics,
        # to isolate "what does a REAL delta_z translation do" from "does
        # the RBM implementation compute that translation correctly".
        shift_cells = -dz_nm_applied * 1e-9 / dz  # negative z = toward substrate, matching this project's convention
        f_shifted = ndi_shift(f, shift=(shift_cells, 0), order=1, mode="nearest")
        e1_shifted = ndi_shift(e1, shift=(shift_cells, 0), order=1, mode="nearest")
        e2_shifted = ndi_shift(e2, shift=(shift_cells, 0), order=1, mode="nearest")
        diag = diagnose(f_shifted, e1_shifted, e2_shifted, r_c, z, W, V_ref)
        e1e2f_resid = float(np.max(np.abs(e1_shifted + e2_shifted - f_shifted)))
        row = dict(dz_applied_nm=dz_nm_applied, **diag, e1e2f_residual=e1e2f_resid)
        rows.append(row)
        print(f"  dz={dz_nm_applied:.3f}nm -> X_neck={diag['X_neck_nm']:.3f}nm r_neck={diag['r_neck_nm']:.3f}nm "
              f"sigma={diag['sigma_Hussein_MPa']:.3f}MPa mass_drift={diag['mass_drift']:.2e} "
              f"e1e2f_resid={e1e2f_resid:.2e}", flush=True)

    fieldnames = list(rows[0].keys())
    with open(os.path.join(OUT, "microtest_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDONE wall={time.time()-t0:.0f}s -- wrote {OUT}/microtest_results.csv")


if __name__ == "__main__":
    main()
