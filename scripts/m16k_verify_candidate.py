"""Milestone 16K: short sink-OFF PF verification run for a selected
candidate geometry, using the path-continuous NeckTracker (Section 4-6)
and BOTH stress measures (Section 12) at every diagnostic step. Sink OFF,
hazard OFF, RBM OFF throughout -- deterministic capillary coarsening only.
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
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402
from pf_sintering.m16k_neck_tracking import NeckTracker  # noqa: E402
from pf_sintering.sintering_potential_stress import sigma_potential as sigma_potential_fn  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)


def run(R_p_um, ratio, t_target, W_nm, dx_nm, n_samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "morphology"), exist_ok=True)
    t0 = time.time()
    R_p_nm = R_p_um * 1000.0

    geo = build_candidate_geometry(R_p_nm=R_p_nm, R_s_nm=None, X0_over_2Rp=ratio, psi_deg=PSI_DEG,
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
    print(f"Rp={R_p_um}um ratio={ratio} Nz={geo['Nz']} Nr={geo['Nr']} dt={dt:.4e} n_steps_total={n_steps_total}",
          flush=True)

    tracker = NeckTracker(window_widths_in_W=(1.0, 1.5, 2.0, 2.5, 3.0))
    V0 = axisym_volume(f, r_c, dr, dz)
    F0 = None
    rows = []

    for step in range(0, n_steps_total + 1):
        if step > 0:
            f, e1, e2, _ = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                                           bc_z="noflux")
            if not np.all(np.isfinite(f)):
                print(f"BLOWUP at step {step}"); break
        if step % diag_every != 0 and step != n_steps_total:
            continue

        t = step * dt
        R_of_z = measure_R_of_z(f, r_c)
        tr = tracker.step(R_of_z, z, W, step_index=step, t=t)
        if tr["selected_contact"] is None:
            continue
        z_gb, a_contact = tr["z_gb"], tr["a_contact"]
        X_neck = 2 * a_contact
        r_neck_1p5W = tr["all_candidate_curvatures"].get(1.5, float("nan"))

        if np.isfinite(r_neck_1p5W) and r_neck_1p5W > 0:
            sigma_H, sc, sw, C_GB = hussein_eq1b_sigma(r_neck_1p5W, X_neck, GAMMA_S, GAMMA_GB)
        else:
            sigma_H = sc = sw = float("nan")

        # kappa_meridional (signed) from the same 1.5W fit, for sigma_potential
        from pf_sintering.hussein_neck_stress import neck_curvature_windows
        win = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5,))[0]
        sigma_P, km, ka = sigma_potential_fn(win["kappa_signed"], a_contact, GAMMA_S)

        V = axisym_volume(f, r_c, dr, dz)
        A_free = free_surface_area_of_revolution(R_of_z, z)
        A_GB = math.pi * a_contact ** 2
        sv = surface_to_volume_metrics(A_free, A_GB, V)
        F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
        if F0 is None:
            F0 = F

        row = dict(step=step, time=t, a_contact_nm=a_contact * 1e9, X_neck_nm=X_neck * 1e9,
                   n_candidates=len(tr["all_candidate_contacts"]), switched=tr["switched"],
                   r_neck_spread_pct=tr["r_neck_spread_pct"],
                   sigma_Hussein_MPa=sigma_H / 1e6, sigma_curvature_MPa=sc / 1e6, sigma_width_MPa=sw / 1e6,
                   sigma_potential_MPa=sigma_P / 1e6,
                   S_over_V_1_per_nm=sv["S_over_V"] * 1e-9, free_energy=F, energy_decreasing=(F <= F0 + 1e-20),
                   mass_drift=(V - V0) / V0)
        for w_mult, rn in tr["all_candidate_curvatures"].items():
            row[f"r_neck_{w_mult}W_nm"] = rn * 1e9 if np.isfinite(rn) else float("nan")
        rows.append(row)
        print(f"  t={t:.3f} a={a_contact*1e9:.2f}nm X={X_neck*1e9:.2f}nm r1.5W={r_neck_1p5W*1e9:.3f}nm "
              f"sigmaH={sigma_H/1e6:.3f}MPa sigmaP={sigma_P/1e6:.3f}MPa spread={tr['r_neck_spread_pct']:.1f}% "
              f"n_cand={len(tr['all_candidate_contacts'])} switch={tr['switched']} wall={time.time()-t0:.0f}s",
              flush=True)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(out_dir, "switch_log.json"), "w") as fh:
        json.dump(tracker.switch_log, fh, indent=2, default=str)

    if rows:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        t_arr = [r["time"] for r in rows]
        axs[0, 0].plot(t_arr, [r["sigma_Hussein_MPa"] for r in rows], color="#B4530A", label="sigma_Hussein")
        axs[0, 0].axhline(50, color="red", linestyle="--", linewidth=0.7)
        axs[0, 0].set_ylabel("sigma_Hussein (MPa)")
        axs[0, 1].plot(t_arr, [r["r_neck_1.5W_nm"] for r in rows], color="#B22222")
        axs[0, 1].set_ylabel("r_neck 1.5W (nm)")
        axs[1, 0].plot(t_arr, [r["r_neck_spread_pct"] for r in rows], color="0.3")
        axs[1, 0].set_ylabel("r_neck window spread (%)"); axs[1, 0].set_xlabel("time")
        axs[1, 1].plot(t_arr, [r["n_candidates"] for r in rows], color="#2E8B57", drawstyle="steps-post")
        axs[1, 1].set_ylabel("n candidate contacts"); axs[1, 1].set_xlabel("time")
        fig.suptitle(f"Rp={R_p_um}um X0/(2Rp)={ratio} W={W_nm}nm dx={dx_nm}nm")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "verify_summary.png"), dpi=130)
        plt.close(fig)

    meta = dict(R_p_um=R_p_um, ratio=ratio, W_nm=W_nm, dx_nm=dx_nm, t_target=t_target, dt=dt,
                n_steps_total=n_steps_total, Nz=geo["Nz"], Nr=geo["Nr"], wall_time_s=time.time() - t0,
                n_switches=len(tracker.switch_log))
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"DONE wall={time.time()-t0:.0f}s n_switches={len(tracker.switch_log)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rp-um", type=float, required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--t-target", type=float, default=5.0)
    ap.add_argument("--w-nm", type=float, default=10.0)
    ap.add_argument("--dx-nm", type=float, default=1.25)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    run(args.rp_um, args.ratio, args.t_target, args.w_nm, args.dx_nm, args.n_samples, args.out_dir)
