"""Milestone 16J Stage 0: cheap static geometry screen (Section 15).

No time evolution at all -- construction + diagnostics only. For every
candidate in the (R_s/R_p, X0/(2*R_p)) matrix at R_p=1um (AR=1), computes
the full required diagnostic set, saves one morphology PNG, and applies
the QC rejection criteria (single neck, resolved curvature, no NaN/seam
issues). Writes runs/m16j_geometry_search/stage0_candidates.csv.
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

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16j_geometry_search")
os.makedirs(ROOT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "stage0_morphology"), exist_ok=True)

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
W_NM = 10.0
DX_NM = 1.25

R_P_NM = 1000.0
CHI_VALUES = [1.0, 2.0, 5.0, 10.0, None]  # None = exact flat
X0_OVER_2RP_VALUES = [0.025, 0.05, 0.10]
AR = 1.0


def candidate_id(chi, ratio):
    chi_label = "flat" if chi is None else f"chi{chi:g}"
    return f"Rp1000nm_{chi_label}_X0over2Rp{ratio:.3f}"


def evaluate_candidate(chi, ratio):
    R_s_nm = None if chi is None else chi * R_P_NM
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=R_s_nm, X0_over_2Rp=ratio, psi_deg=PSI_DEG,
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

    if not mins:
        z_gb, a_contact = float("nan"), float("nan")
    else:
        z_gb, a_contact = mins[0]

    X_neck = 2 * a_contact if np.isfinite(a_contact) else float("nan")
    windows = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5, 2.5)) if np.isfinite(z_gb) else []
    r_neck_1p5W = windows[0]["r_neck"] if len(windows) > 0 else float("nan")
    r_neck_2p5W = windows[1]["r_neck"] if len(windows) > 1 else float("nan")

    # analytic (construction-exact) fillet radii, independent of grid resolution
    rho1_analytic = geo["particle_info"]["fillet"]["rho"] * 1e-9
    if not geo["flat"]:
        rho2_analytic = geo["substrate_info"]["fillet"]["rho"] * 1e-9
    else:
        rho2_analytic = geo["substrate_info"]["rho"] * 1e-9
    r_neck_analytic = 0.5 * (rho1_analytic + rho2_analytic)

    r_neck_for_stress = r_neck_1p5W if np.isfinite(r_neck_1p5W) else r_neck_analytic
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
    qc_pass = all(qc.values())

    row = dict(
        candidate_id=candidate_id(chi, ratio), R_p_um=R_P_NM / 1000.0,
        R_s_um=(R_s_nm / 1000.0 if R_s_nm is not None else float("inf")),
        R_s_over_R_p=(chi if chi is not None else float("inf")), flat_substrate=(chi is None),
        aspect_ratio=AR, X0_over_Rp=2 * ratio, contact_radius_nm=a_contact * 1e9,
        X_neck_nm=X_neck * 1e9, r_neck_1p5W_nm=r_neck_1p5W * 1e9, r_neck_2p5W_nm=r_neck_2p5W * 1e9,
        r_neck_analytic_nm=r_neck_analytic * 1e9, r_neck_over_Rp=r_neck_for_stress / (R_P_NM * 1e-9),
        X_neck_over_Rp=X_neck / (R_P_NM * 1e-9) if np.isfinite(X_neck) else float("nan"),
        sigma_hussein_MPa=sigma / 1e6, sigma_curvature_MPa=sigma_curv / 1e6, sigma_width_MPa=sigma_width / 1e6,
        Q=Q, surface_area_nm2=A_free * 1e18, solid_volume_nm3=V * 1e27, A_over_V_1_per_nm=sv["S_over_V"] * 1e-9,
        GB_area_nm2=A_GB * 1e18 if np.isfinite(A_GB) else float("nan"), total_free_energy=F,
        r_neck_over_W=r_neck_over_W, r_neck_over_dx=r_neck_over_dx, n_local_minima=n_min, n_local_maxima=n_max,
        single_neck_pass=qc["single_neck"], no_nan_pass=qc["no_nan"], f_range_pass=qc["f_range_ok"],
        e1e2_consistent_pass=qc["e1e2_consistent"], resolved_pass=qc["resolved"], qc_pass=qc_pass,
        Nz=geo["Nz"], Nr=geo["Nr"], TJ_slope_target=geo["m_target"],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    Z_nm, R_nm = np.meshgrid(geo["z"], geo["r_c"], indexing="ij")
    grain_id = np.where(e1 >= e2, -1.0, 1.0) * f
    pc = ax.pcolormesh(Z_nm, R_nm, grain_id, cmap="RdBu", vmin=-1, vmax=1, shading="auto")
    ax.contour(Z_nm, R_nm, f, levels=[0.5], colors="k", linewidths=0.8)
    ax.set_xlabel("z (nm)"); ax.set_ylabel("r (nm)")
    ax.set_title(f"{row['candidate_id']}  a={row['contact_radius_nm']:.2f}nm  "
                 f"r_neck~{r_neck_for_stress*1e9:.3f}nm  QC={'PASS' if qc_pass else 'FAIL'}", fontsize=9)
    fig.colorbar(pc, ax=ax, label="grain id * f")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "stage0_morphology", f"{row['candidate_id']}.png"), dpi=130)
    plt.close(fig)

    return row


def main():
    rows = []
    for chi in CHI_VALUES:
        for ratio in X0_OVER_2RP_VALUES:
            print(f"evaluating chi={chi} ratio={ratio} ...", flush=True)
            try:
                row = evaluate_candidate(chi, ratio)
            except Exception as exc:
                row = dict(candidate_id=candidate_id(chi, ratio), error=str(exc), qc_pass=False)
                print(f"  FAILED: {exc}")
            rows.append(row)
            print(f"  -> {row}")

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(ROOT, "stage0_candidates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(ROOT, "stage0_candidates.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"\nwrote {len(rows)} candidates to stage0_candidates.csv")
    n_pass = sum(1 for r in rows if r.get("qc_pass"))
    print(f"QC pass: {n_pass}/{len(rows)}")


if __name__ == "__main__":
    main()
