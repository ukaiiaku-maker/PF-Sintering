"""Milestone 16F Sections 8-13: sharper-resolution (W/Rcyl=0.1, W/dx=8)
PF dynamics of the exact Figure-4 geometry, tracking grain-elimination
progress and the best-effort lambda/lambda_c(t) stability ratio (fit
current R(z,t) to the two-mode form, evaluate the Eq.-4 reference
module -- NOT the raw unrestricted Hessian eigenvalue, per this
milestone's explicit instruction).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_eq4_reference import lambda_c_over_R_eq4  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import EPS1, EPS2, LAM_OVER_RCYL, P, build_exact_two_mode, find_stable_dt, grain_volumes, psi_to_gamma_gb  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16f_campaign")


def fit_two_mode_and_stability_ratio(R_of_z, z, R_cyl_nominal, psi_deg):
    """Best-effort two-mode fit of the current profile: R0,e1,e2 solved
    via linear least squares against [1, cos(2*pi*z/lambda),
    cos(pi*z/lambda)] (lambda held fixed at the nominal domain value --
    only the AMPLITUDES are refit, matching Section 11's "fit R0(t),
    e1bar(t), e2bar(t)" at fixed geometry/wavelength), then evaluates
    lambda/lambda_c(t) via the LITERAL printed Eq. 4
    (`lambda_c_over_R_eq4`, closed form -- not the full-area quadrature
    diagnostic) at the CURRENT fitted e1bar(t), e2bar(t)."""
    lam = LAM_OVER_RCYL * R_cyl_nominal
    cos1 = np.cos(2 * math.pi * z / lam)
    cos2 = np.cos(math.pi * z / lam)
    A = np.vstack([np.ones_like(z), cos1, cos2]).T
    coef, *_ = np.linalg.lstsq(A, R_of_z, rcond=None)
    R0_fit, e1_amp, e2_amp = coef
    R_cyl_fit = math.sqrt(R0_fit ** 2 + (e1_amp ** 2 + e2_amp ** 2) / 2.0)
    e1bar = e1_amp / R_cyl_fit
    e2bar = e2_amp / R_cyl_fit
    try:
        lam_c = lambda_c_over_R_eq4(abs(e1bar), abs(e2bar), psi_deg)
        ratio = (lam / R_cyl_fit) / lam_c
    except (ValueError, ZeroDivisionError):
        lam_c, ratio = float("nan"), float("nan")
    return dict(R_cyl_fit=R_cyl_fit, e1bar=e1bar, e2bar=e2bar, lam_c=lam_c, lam_over_lamc=ratio)


def run_case(psi_deg, R_cyl_nm=100.0, W_nm=10.0, dx_nm=1.25, t_target=60.0, n_sample=30):
    R_cyl = R_cyl_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]

    state = build_exact_two_mode(R_cyl, W, dr, dz)
    f, e1, e2 = state["f"], state["e1"], state["e2"]
    Nz, Nr = state["Nz"], state["Nr"]
    r_c, r_f = r_centers_faces(Nr, dr)
    z = state["z"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    V1_0, V2_0 = grain_volumes(e1, e2, r_c, dr, dz)

    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"[psi={psi_deg}] blew up at step {step}")
        R_of_z = measure_R_of_z(f, r_c)
        V1, V2 = grain_volumes(e1, e2, r_c, dr, dz)
        V = axisym_volume(f, r_c, dr, dz)
        F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
        fit = fit_two_mode_and_stability_ratio(R_of_z, z, R_cyl, psi_deg)
        R_crest = float(np.nanmax(R_of_z))
        R_trough = float(np.nanmin(R_of_z))
        R_at_0 = float(R_of_z[0])
        R_at_lam = float(R_of_z[Nz // 2]) if Nz % 2 == 0 else float(R_of_z[Nz // 2])
        rows.append(dict(step=step, t=step * dt, V1_frac=V1 / V1_0, V2_frac=V2 / V2_0,
                          mass_drift=(V - V0) / V0, R_crest_nm=R_crest * 1e9, R_trough_nm=R_trough * 1e9,
                          R_at_0_nm=R_at_0 * 1e9, R_at_lam_nm=R_at_lam * 1e9,
                          e1bar_fit=fit["e1bar"], e2bar_fit=fit["e2bar"], lam_over_lamc=fit["lam_over_lamc"], F=F))
        print(f"  [psi={psi_deg:.0f}] t={rows[-1]['t']:.3e} V1/V1_0={rows[-1]['V1_frac']:.5f} "
              f"R(0)={R_at_0*1e9:.3f}nm R(lam)={R_at_lam*1e9:.3f}nm "
              f"e1bar={fit['e1bar']:.4f} e2bar={fit['e2bar']:.4f} lam/lamc={fit['lam_over_lamc']:.4f} "
              f"mass_drift={rows[-1]['mass_drift']:.2e}")

    return dict(psi_deg=psi_deg, R_cyl_nm=R_cyl_nm, W_nm=W_nm, dx_nm=dx_nm, Nz=Nz, Nr=Nr, dt=dt,
                n_steps_total=n_steps_total, V1_0=V1_0, V2_0=V2_0, rows=rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi", type=float, required=True)
    ap.add_argument("--t-target", type=float, default=60.0)
    ap.add_argument("--n-sample", type=int, default=30)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    out_path = os.path.join(CAMPAIGN_DIR, f"psi{args.psi:g}{args.tag}.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
    else:
        result = run_case(args.psi, t_target=args.t_target, n_sample=args.n_sample)
        with open(out_path, "w") as fh:
            json.dump(result, fh, default=str)
        print(f"saved to {out_path}")
