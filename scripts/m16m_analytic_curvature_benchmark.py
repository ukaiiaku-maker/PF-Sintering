"""M16M continuation Section 6: static analytic-geometry curvature-
extraction benchmark.

Builds SHARP axisymmetric profiles R(z) with an EXACTLY KNOWN local
radius of curvature `r_true` at the neck (a circular arc of radius
r_true, capped to a constant plateau far from the trough so the local
minimum is isolated), diffuses them into a phase field using the SAME
tanh construction `pf_sintering.m16j_geometry.build_candidate_geometry`
uses (`f = 0.5*(1 - tanh((R_g - R(z))/W))`, i.e. the field's interface
width is set directly by `W`, independent of any PF relaxation), and runs
the SAME extraction pipeline (measure_R_of_z -> find_all_extrema ->
neck_curvature_windows -> hussein_eq1b_sigma) used throughout M16H-M16M.

No PF solver is involved -- this isolates whether the W-dependence found
in the real geometry (Sections 10-12 of the continuation report) is an
EXTRACTION BIAS (present even for a perfectly known, static sharp shape)
or requires real PF relaxation dynamics to appear.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import find_all_extrema  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import psi_to_gamma_gb  # noqa: E402

GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(160.0, GAMMA_S)
FIXED_WINDOWS_NM = (6.0, 9.0, 12.0, 15.0, 18.0, 24.0)
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16m_analytic_curvature_benchmark")


def analytic_R_of_z(z, R_min, r_true, plateau_R):
    """Circular arc of EXACT radius r_true dipping from `plateau_R` down
    to `R_min` at z=0 (meridional radius of curvature = r_true by
    construction), capped at plateau_R away from the trough."""
    arc_half_width = 0.9 * r_true
    inside = np.abs(z) <= arc_half_width
    R = np.where(inside, R_min + r_true - np.sqrt(np.maximum(r_true ** 2 - z ** 2, 0.0)), plateau_R)
    return np.minimum(R, plateau_R)


def build_diffuse_field(R_min, r_true, plateau_R, W_nm, dx_nm, z_half_range_nm):
    W = W_nm * 1e-9
    dz = dx_nm * 1e-9
    Nz = int(round(2 * z_half_range_nm * 1e-9 / dz))
    z = (np.arange(Nz) - Nz // 2 + 0.5) * dz
    r_max_nm = plateau_R + 8 * W_nm
    Nr = int(round(r_max_nm * 1e-9 / dz))
    r_c = (np.arange(Nr) + 0.5) * dz
    R_of_z_true = analytic_R_of_z(z * 1e9, R_min, r_true, plateau_R) * 1e-9
    Rg = r_c[None, :]
    f = 0.5 * (1.0 - np.tanh((Rg - R_of_z_true[:, None]) / W))
    return f, z, r_c


def main():
    os.makedirs(OUT, exist_ok=True)
    R_min_true = 120.0  # nm, order-of-magnitude match to the real a_contact
    plateau_R = 400.0   # nm, far enough that the arc's kink doesn't leak into any tested window
    r_true_list = [10.0, 20.0, 40.0, 80.0]
    W_dx_pairs = [(10.0, 1.25), (6.0, 1.0), (4.0, 0.65)]

    rows = []
    print(f"{'W(nm)':>6} {'r_true':>8} {'a_meas':>8}   " + "  ".join(f"r@{w:g}nm" for w in FIXED_WINDOWS_NM))
    for W_nm, dx_nm in W_dx_pairs:
        for r_true in r_true_list:
            f, z, r_c = build_diffuse_field(R_min_true, r_true, plateau_R, W_nm, dx_nm, z_half_range_nm=3 * r_true + 60)
            R_of_z = measure_R_of_z(f, r_c)
            ext = find_all_extrema(R_of_z, z)
            mins = sorted([(zz, RR) for k, zz, RR in ext if k == "min"], key=lambda c: c[1])
            if not mins:
                print(f"{W_nm:>6.1f} {r_true:>8.1f}  NO MINIMUM FOUND")
                continue
            z_min, a_meas = mins[0]
            X_neck = 2 * a_meas
            windows = neck_curvature_windows(R_of_z, z, z_min, 1e-9, window_widths_in_W=FIXED_WINDOWS_NM)
            row = dict(W_nm=W_nm, dx_nm=dx_nm, r_true_nm=r_true, R_min_true_nm=R_min_true,
                       a_meas_nm=a_meas * 1e9, X_neck_meas_nm=X_neck * 1e9)
            vals_str = []
            for w_nm, win in zip(FIXED_WINDOWS_NM, windows):
                r_neck = win["r_neck"]
                r_neck_nm = r_neck * 1e9 if np.isfinite(r_neck) and r_neck > 0 else float("nan")
                row[f"r_neck_meas_{w_nm:g}nm"] = r_neck_nm
                row[f"r_meas_over_r_true_{w_nm:g}nm"] = r_neck_nm / r_true if np.isfinite(r_neck_nm) else float("nan")
                if np.isfinite(r_neck_nm) and r_neck_nm > 0:
                    sigma, *_ = hussein_eq1b_sigma(r_neck_nm * 1e-9, X_neck, GAMMA_S, GAMMA_GB)
                    row[f"sigma_{w_nm:g}nm_MPa"] = sigma / 1e6
                else:
                    row[f"sigma_{w_nm:g}nm_MPa"] = float("nan")
                vals_str.append(f"{r_neck_nm:7.2f}")
            rows.append(row)
            print(f"{W_nm:>6.1f} {r_true:>8.1f} {a_meas*1e9:>8.2f}   " + "  ".join(vals_str))

    with open(os.path.join(OUT, "analytic_benchmark.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDONE -- wrote {OUT}/analytic_benchmark.csv")


if __name__ == "__main__":
    main()
