"""Milestone 16J Stage 1: short sink-OFF PF screen (Section 16).

For a candidate promoted out of Stage 0, runs pure capillary (sink OFF,
hazard OFF, RBM OFF) axisymmetric PF evolution for a short horizon,
tracking r_neck(t), X_neck(t), the Hussein stress decomposition, Q(t),
S/V(t), energy(t), and mass drift. One candidate per process invocation
(so multiple candidates can run in parallel as separate background jobs).

Usage: python scripts/m16j_stage1_short_pf_screen.py --chi 1.0 --ratio 0.05 --t-target 8.0
       python scripts/m16j_stage1_short_pf_screen.py --flat --ratio 0.10 --t-target 8.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.axisym_interface_metrics import free_surface_area_of_revolution, surface_to_volume_metrics  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt, grain_volumes  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16j_geometry_search", "stage1")
PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
W_NM = 10.0
DX_NM = 1.25
R_P_NM = 1000.0
AR = 1.0


def candidate_dir(chi, ratio):
    chi_label = "flat" if chi is None else f"chi{chi:g}"
    d = os.path.join(ROOT, f"Rp1000nm_{chi_label}_X0over2Rp{ratio:.3f}")
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    os.makedirs(os.path.join(d, "morphology"), exist_ok=True)
    return d


def diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, W, gamma_s, gamma_gb, step, t, V0, F0):
    R_of_z = measure_R_of_z(f, r_c)
    ext = find_all_extrema(R_of_z, z)
    mins = [(zz, RR) for k, zz, RR in ext if k == "min"]
    if not mins:
        z_gb, a = float("nan"), float("nan")
    else:
        z_gb, a = mins[0]
    X_neck = 2 * a if np.isfinite(a) else float("nan")
    windows = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5, 2.5)) if np.isfinite(z_gb) else []
    r_neck = windows[0]["r_neck"] if len(windows) > 0 else float("nan")
    if np.isfinite(r_neck) and np.isfinite(X_neck) and r_neck > 0:
        sigma, sigma_curv, sigma_width, C_GB = hussein_eq1b_sigma(r_neck, X_neck, gamma_s, gamma_gb)
        Q = X_neck / (C_GB * r_neck) if np.isfinite(C_GB) and C_GB > 0 else float("nan")
    else:
        sigma = sigma_curv = sigma_width = Q = float("nan")
    V = axisym_volume(f, r_c, dr, dz)
    A_free = free_surface_area_of_revolution(R_of_z, z)
    A_GB = math.pi * a ** 2 if np.isfinite(a) else float("nan")
    sv = surface_to_volume_metrics(A_free, A_GB, V)
    F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
    return dict(step=step, time=t, r_neck_nm=r_neck * 1e9, X_neck_nm=X_neck * 1e9,
                contact_radius_nm=a * 1e9 if np.isfinite(a) else float("nan"),
                sigma_MPa=sigma / 1e6, sigma_curvature_MPa=sigma_curv / 1e6, sigma_width_MPa=sigma_width / 1e6,
                Q=Q, S_over_V_1_per_nm=sv["S_over_V"] * 1e-9, free_energy=F,
                mass_drift=(V - V0) / V0 if V0 else 0.0, energy_drop=(F0 - F) if F0 is not None else 0.0)


def run_candidate(chi, ratio, t_target, n_samples=40, w_nm=None, dx_nm=None, dir_suffix=""):
    R_s_nm = None if chi is None else chi * R_P_NM
    label = "flat" if chi is None else f"chi{chi:g}"
    w_nm = w_nm if w_nm is not None else W_NM
    dx_nm = dx_nm if dx_nm is not None else DX_NM
    cdir = candidate_dir(chi, ratio) + dir_suffix
    os.makedirs(os.path.join(cdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "morphology"), exist_ok=True)
    t0 = time.time()

    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=R_s_nm, X0_over_2Rp=ratio, psi_deg=PSI_DEG,
                                    W_nm=w_nm, dr_nm=dx_nm, dz_nm=dx_nm, aspect_ratio=AR)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = w_nm * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    n_steps_total = max(1, int(t_target / dt))
    diag_every = max(1, n_steps_total // n_samples)

    V0 = axisym_volume(f, r_c, dr, dz)
    F0 = None
    history_path = os.path.join(cdir, "history.csv")
    rows = []
    print(f"[{label} ratio={ratio}] Nz={geo['Nz']} Nr={geo['Nr']} dt={dt:.4e} n_steps_total={n_steps_total}", flush=True)

    for step in range(0, n_steps_total + 1):
        if step > 0:
            f, e1, e2, _ = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                                           bc_z="noflux")
            if not np.all(np.isfinite(f)):
                print(f"[{label}] BLOWUP at step {step}")
                break
        if step % diag_every == 0 or step == n_steps_total:
            row = diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, W, GAMMA_S, GAMMA_GB, step, step * dt, V0, F0)
            if F0 is None:
                F0 = row["free_energy"]
            rows.append(row)
            print(f"  [{label}] t={row['time']:.3f} r_neck={row['r_neck_nm']:.3f}nm X_neck={row['X_neck_nm']:.3f}nm "
                  f"sigma={row['sigma_MPa']:.4f}MPa Q={row['Q']:.4f} mass_drift={row['mass_drift']:.2e} "
                  f"wall={time.time()-t0:.0f}s", flush=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(history_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rows:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        t = [r["time"] for r in rows]
        axs[0, 0].plot(t, [r["sigma_MPa"] for r in rows], color="#B4530A")
        axs[0, 0].set_ylabel("sigma_s (MPa)")
        axs[0, 1].plot(t, [r["r_neck_nm"] for r in rows], color="#B22222")
        axs[0, 1].set_ylabel("r_neck (nm)")
        axs[1, 0].plot(t, [r["X_neck_nm"] for r in rows], color="#2E8B57")
        axs[1, 0].set_ylabel("X_neck (nm)"); axs[1, 0].set_xlabel("time")
        axs[1, 1].plot(t, [r["Q"] for r in rows], color="0.3")
        axs[1, 1].set_ylabel("Q"); axs[1, 1].set_xlabel("time")
        fig.suptitle(f"{label} X0/(2Rp)={ratio}")
        fig.tight_layout()
        fig.savefig(os.path.join(cdir, "figures", "stage1_summary.png"), dpi=130)
        plt.close(fig)

    meta = dict(chi=chi, ratio=ratio, R_p_nm=R_P_NM, R_s_nm=R_s_nm, t_target=t_target, dt=dt,
                n_steps_total=n_steps_total, Nz=geo["Nz"], Nr=geo["Nr"], wall_time_s=time.time() - t0,
                final_mass_drift=rows[-1]["mass_drift"] if rows else None)
    with open(os.path.join(cdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[{label} ratio={ratio}] DONE wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=float, default=None)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--t-target", type=float, default=8.0)
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--w-nm", type=float, default=None)
    ap.add_argument("--dx-nm", type=float, default=None)
    ap.add_argument("--dir-suffix", type=str, default="")
    args = ap.parse_args()
    chi = None if args.flat else args.chi
    run_candidate(chi, args.ratio, args.t_target, n_samples=args.n_samples,
                  w_nm=args.w_nm, dx_nm=args.dx_nm, dir_suffix=args.dir_suffix)
