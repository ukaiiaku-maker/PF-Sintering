"""M16K Section 3/7: diagnose whether the M16J refined run's sharp r_neck
jump (t~1.1->1.2) is a genuine morphological rearrangement or a
neck-diagnostic artifact.

Reproduces the EXACT same deterministic (sink-OFF, no RNG) trajectory as
the M16J refined run (flat, X0/(2Rp)=0.10, R_p=1000nm, W=6nm, dx=1.0nm)
but with much finer diagnostic sampling (Delta_t=0.01) bounding the first
jump, recording ALL local minima/maxima in R(z) (not just the first) and
r_neck at SEVERAL window widths at every sample.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.axisym import axisym_gb_face_projected_step  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import fit_local_circle, hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16k_jump_diagnostic")
os.makedirs(os.path.join(OUT, "morphology"), exist_ok=True)

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
W_NM, DX_NM, R_P_NM, RATIO = 6.0, 1.0, 1000.0, 0.10
T_STOP = 1.35
DT_OUTPUT = 0.01

WINDOW_WIDTHS_IN_W = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # -> 6,9,12,15,18,24 nm at W=6nm


def main():
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
    n_steps_total = int(T_STOP / dt)
    diag_every = max(1, int(DT_OUTPUT / dt))
    print(f"dt={dt:.4e} n_steps_total={n_steps_total} diag_every={diag_every} ({diag_every*dt:.4f} time units)")

    rows = []
    snap_count = 0
    for step in range(0, n_steps_total + 1):
        if step > 0:
            f, e1, e2, _ = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                                           bc_z="noflux")
        if step % diag_every != 0 and step != n_steps_total:
            continue
        t = step * dt
        R_of_z = measure_R_of_z(f, r_c)
        ext = find_all_extrema(R_of_z, z)
        mins = [(zz, RR) for k, zz, RR in ext if k == "min"]
        maxs = [(zz, RR) for k, zz, RR in ext if k == "max"]
        row = dict(step=step, time=t, n_minima=len(mins), n_maxima=len(maxs),
                   all_minima=str([(round(zz * 1e9, 3), round(RR * 1e9, 3)) for zz, RR in mins]))
        if mins:
            z_gb, a = mins[0]
            row["z_gb_nm"] = z_gb * 1e9
            row["a_contact_nm"] = a * 1e9
            windows = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=WINDOW_WIDTHS_IN_W)
            for w_mult, win in zip(WINDOW_WIDTHS_IN_W, windows):
                half_nm = w_mult * W_NM
                row[f"r_neck_{half_nm:g}nm"] = win["r_neck"] * 1e9 if np.isfinite(win["r_neck"]) else float("nan")
                row[f"npts_{half_nm:g}nm"] = win["n_points"]
            r_neck_ref = windows[1]["r_neck"]  # 1.5W, the "authoritative" one
            X_neck = 2 * a
            if np.isfinite(r_neck_ref) and r_neck_ref > 0:
                sigma, sc, sw, C_GB = hussein_eq1b_sigma(r_neck_ref, X_neck, GAMMA_S, GAMMA_GB)
                row["sigma_MPa_1p5W"] = sigma / 1e6
            else:
                row["sigma_MPa_1p5W"] = float("nan")
            row["X_neck_nm"] = X_neck * 1e9
        rows.append(row)
        print(f"  t={t:.4f} n_min={row['n_minima']} a={row.get('a_contact_nm', float('nan')):.3f}nm "
              f"r1.5W={row.get('r_neck_9nm', float('nan')):.3f}nm sigma={row.get('sigma_MPa_1p5W', float('nan')):.3f}MPa",
              flush=True)

        if 1.0 <= t <= 1.3 and (snap_count % 1 == 0):
            fig, ax = plt.subplots(figsize=(6, 8))
            Zg, Rg = np.meshgrid(z * 1e9, r_c * 1e9, indexing="ij")
            grain_id = np.where(e1 >= e2, -1.0, 1.0) * f
            ax.pcolormesh(Rg, Zg, grain_id, cmap="RdBu", vmin=-1, vmax=1, shading="auto")
            ax.contour(Rg, Zg, f, levels=[0.5], colors="k", linewidths=0.8)
            ax.set_xlim(0, 120); ax.set_ylim(-40, 100)
            ax.set_title(f"t={t:.4f}  a={row.get('a_contact_nm', float('nan')):.2f}nm")
            ax.set_xlabel("r (nm)"); ax.set_ylabel("z (nm)")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "morphology", f"t{t:.4f}.png"), dpi=110)
            plt.close(fig)
        snap_count += 1

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(OUT, "high_cadence_history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows to {OUT}/high_cadence_history.csv")


if __name__ == "__main__":
    main()
