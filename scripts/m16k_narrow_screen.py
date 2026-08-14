"""Milestone 16K Section 9: narrow physical-neighborhood static screen.

Exact flat substrate only. R_p in {0.5,1.0,2.0}um x X0/(2Rp) in
{0.10,0.11,0.125,0.15} (12 cases), aspect_ratio=1, psi=160 (unchanged).
Static construction only (no time evolution) -- looking for a candidate
whose INITIAL sigma_H sits in the ~10-30 MPa range (Section 10's stated
preference), rather than the M16J winner's ~47 MPa start.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.axisym import axisym_free_energy_gb, axisym_volume  # noqa: E402
from pf_sintering.axisym_interface_metrics import free_surface_area_of_revolution, surface_to_volume_metrics  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import grain_volumes  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16k_geometry_search")
os.makedirs(ROOT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "morphology"), exist_ok=True)

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
W_NM = 10.0
DX_NM = 1.25
AR = 1.0

R_P_UM_VALUES = [0.5, 1.0, 2.0]
X0_OVER_2RP_VALUES = [0.10, 0.11, 0.125, 0.15]


def candidate_id(R_p_um, ratio):
    return f"Rp{R_p_um:g}um_flat_X0over2Rp{ratio:.3f}"


def evaluate_candidate(R_p_um, ratio):
    R_p_nm = R_p_um * 1000.0
    geo = build_candidate_geometry(R_p_nm=R_p_nm, R_s_nm=None, X0_over_2Rp=ratio, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=AR)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = W_NM * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]

    R_of_z = measure_R_of_z(f, r_c)
    ext = find_all_extrema(R_of_z, z)
    n_min = sum(1 for k, *_ in ext if k == "min")
    n_max = sum(1 for k, *_ in ext if k == "max")
    mins = [(zz, RR) for k, zz, RR in ext if k == "min"]

    qc = dict(single_neck=(n_min == 1), no_nan=bool(np.all(np.isfinite(f))),
              f_range_ok=bool(f.min() >= -1e-6 and f.max() <= 1 + 1e-6),
              e1e2_consistent=bool(np.max(np.abs(e1 + e2 - f)) < 1e-6))

    z_gb, a_contact = (mins[0] if mins else (float("nan"), float("nan")))
    X_neck = 2 * a_contact if np.isfinite(a_contact) else float("nan")
    windows = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.0, 1.5, 2.0, 2.5, 3.0)) \
        if np.isfinite(z_gb) else []
    r_neck_1p0W = windows[0]["r_neck"] if len(windows) > 0 else float("nan")
    r_neck_1p5W = windows[1]["r_neck"] if len(windows) > 1 else float("nan")
    r_neck_2p0W = windows[2]["r_neck"] if len(windows) > 2 else float("nan")
    r_neck_2p5W = windows[3]["r_neck"] if len(windows) > 3 else float("nan")
    r_neck_3p0W = windows[4]["r_neck"] if len(windows) > 4 else float("nan")

    fil_p = geo["particle_info"]["fillet"]
    rho1_analytic = fil_p["rho"] * 1e-9
    rho2_analytic = geo["substrate_info"]["rho"] * 1e-9
    r_neck_analytic = 0.5 * (rho1_analytic + rho2_analytic)

    r_neck_for_stress = r_neck_1p5W if np.isfinite(r_neck_1p5W) else r_neck_analytic
    finite_estimates = [x for x in (r_neck_1p0W, r_neck_1p5W, r_neck_2p0W, r_neck_2p5W, r_neck_3p0W)
                         if np.isfinite(x) and x > 0]
    spread_pct = (100.0 * (max(finite_estimates) - min(finite_estimates)) / np.mean(finite_estimates)
                  if len(finite_estimates) >= 2 else float("nan"))

    if np.isfinite(r_neck_for_stress) and np.isfinite(X_neck) and r_neck_for_stress > 0:
        sigma, sigma_curv, sigma_width, C_GB = hussein_eq1b_sigma(r_neck_for_stress, X_neck, GAMMA_S, GAMMA_GB)
        Q = X_neck / (C_GB * r_neck_for_stress) if np.isfinite(C_GB) and C_GB > 0 else float("nan")
    else:
        sigma = sigma_curv = sigma_width = C_GB = Q = float("nan")

    Vp, Vs = grain_volumes(e1, e2, r_c, dr, dz)
    V = axisym_volume(f, r_c, dr, dz)
    A_free = free_surface_area_of_revolution(R_of_z, z)
    A_GB = math.pi * a_contact ** 2 if np.isfinite(a_contact) else float("nan")
    sv = surface_to_volume_metrics(A_free, A_GB, V)
    F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)

    r_neck_over_W = r_neck_for_stress / W if np.isfinite(r_neck_for_stress) else float("nan")
    r_neck_over_dx = r_neck_for_stress / dr if np.isfinite(r_neck_for_stress) else float("nan")
    qc["resolved"] = bool(np.isfinite(r_neck_over_W) and r_neck_over_W >= 1.0 and r_neck_over_dx >= 3.0)
    qc["stress_robust"] = bool(np.isfinite(spread_pct) and spread_pct <= 30.0)  # loose Stage-0 gate; tightened later
    qc_pass = all(qc.values())

    row = dict(
        candidate_id=candidate_id(R_p_um, ratio), R_p_um=R_p_um, X0_over_2Rp=ratio, X0_over_Rp=2 * ratio,
        contact_radius_nm=a_contact * 1e9, X_neck_nm=X_neck * 1e9,
        r_neck_1p0W_nm=r_neck_1p0W * 1e9, r_neck_1p5W_nm=r_neck_1p5W * 1e9, r_neck_2p0W_nm=r_neck_2p0W * 1e9,
        r_neck_2p5W_nm=r_neck_2p5W * 1e9, r_neck_3p0W_nm=r_neck_3p0W * 1e9, r_neck_analytic_nm=r_neck_analytic * 1e9,
        r_neck_spread_pct=spread_pct,
        r_neck_over_Rp=r_neck_for_stress / (R_p_nm * 1e-9), X_neck_over_Rp=X_neck / (R_p_nm * 1e-9) if np.isfinite(X_neck) else float("nan"),
        sigma_hussein_MPa=sigma / 1e6, sigma_curvature_MPa=sigma_curv / 1e6, sigma_width_MPa=sigma_width / 1e6,
        Q=Q, surface_area_nm2=A_free * 1e18, solid_volume_nm3=V * 1e27, A_over_V_1_per_nm=sv["S_over_V"] * 1e-9,
        GB_area_nm2=A_GB * 1e18 if np.isfinite(A_GB) else float("nan"), total_free_energy=F,
        r_neck_over_W=r_neck_over_W, r_neck_over_dx=r_neck_over_dx, n_local_minima=n_min, n_local_maxima=n_max,
        single_neck_pass=qc["single_neck"], no_nan_pass=qc["no_nan"], f_range_pass=qc["f_range_ok"],
        e1e2_consistent_pass=qc["e1e2_consistent"], resolved_pass=qc["resolved"], stress_robust_pass=qc["stress_robust"],
        qc_pass=qc_pass, Nz=geo["Nz"], Nr=geo["Nr"], TJ_slope_target=geo["m_target"],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    Z_nm, R_nm = np.meshgrid(geo["z"], geo["r_c"], indexing="ij")
    grain_id = np.where(e1 >= e2, -1.0, 1.0) * f
    pc = ax.pcolormesh(Z_nm, R_nm, grain_id, cmap="RdBu", vmin=-1, vmax=1, shading="auto")
    ax.contour(Z_nm, R_nm, f, levels=[0.5], colors="k", linewidths=0.8)
    ax.set_xlabel("z (nm)"); ax.set_ylabel("r (nm)")
    ax.set_title(f"{row['candidate_id']}  a={row['contact_radius_nm']:.2f}nm  "
                 f"sigma~{sigma/1e6:.2f}MPa  spread={spread_pct:.1f}%  QC={'PASS' if qc_pass else 'FAIL'}", fontsize=8)
    fig.colorbar(pc, ax=ax, label="grain id * f")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "morphology", f"{row['candidate_id']}.png"), dpi=130)
    plt.close(fig)

    return row


def main():
    rows = []
    for R_p_um in R_P_UM_VALUES:
        for ratio in X0_OVER_2RP_VALUES:
            print(f"evaluating Rp={R_p_um}um ratio={ratio} ...", flush=True)
            try:
                row = evaluate_candidate(R_p_um, ratio)
            except Exception as exc:
                row = dict(candidate_id=candidate_id(R_p_um, ratio), error=str(exc), qc_pass=False)
                print(f"  FAILED: {exc}")
            rows.append(row)
            print(f"  -> sigma={row.get('sigma_hussein_MPa')}  spread={row.get('r_neck_spread_pct')}  "
                  f"qc_pass={row.get('qc_pass')}")

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(ROOT, "narrow_screen_candidates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(ROOT, "narrow_screen_candidates.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"\nwrote {len(rows)} candidates")
    print("\nsorted by |sigma-20MPa| (closest to the preferred 10-30MPa initial-stress midpoint):")
    ok = [r for r in rows if r.get("qc_pass") and np.isfinite(r.get("sigma_hussein_MPa", float("nan")))]
    ok.sort(key=lambda r: abs(r["sigma_hussein_MPa"] - 20.0))
    for r in ok:
        print(f"  {r['candidate_id']}: sigma={r['sigma_hussein_MPa']:.2f}MPa spread={r['r_neck_spread_pct']:.1f}% "
              f"r_neck(1.5W)={r['r_neck_1p5W_nm']:.2f}nm r_neck/W={r['r_neck_over_W']:.2f}")


if __name__ == "__main__":
    main()
