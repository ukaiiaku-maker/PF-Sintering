"""M16N Section A: cheap analytic (NO PF) flat-substrate geometry scan.

Uses the exact Young-Herring tangent-fillet solve
(`pf_sintering.m16j_geometry.solve_body_fillet` /
`solve_flat_groove_fillet` / `young_herring_slope`) directly -- no PF
field construction, no PF stepping -- to find a candidate initial
geometry whose TRUE analytic fillet radius `rho` lands in the resolvable
~25-40nm range (Section B target), rather than the current canonical
geometry's rho=6.87nm (X0/(2Rp)=0.10), which M16M's continuation found
is smaller than every W/window tested throughout the project.

Varies `X0/(2Rp)` (the dominant lever) at fixed `R_p=1000nm`, `psi=160deg`,
`gamma_s=1 J/m^2`, flat substrate, aspect_ratio=1 (varying aspect ratio is
explicitly secondary per Section A -- not explored this pass unless the
ratio scan alone cannot reach the target band).
"""
from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, ".")
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma  # noqa: E402
from pf_sintering.m16j_geometry import solve_body_fillet, solve_flat_groove_fillet, young_herring_slope  # noqa: E402

sys.path.insert(0, "scripts")
from m16e_exact_hussein_two_mode import psi_to_gamma_gb  # noqa: E402

R_P_NM = 1000.0
PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16n_analytic_geometry_scan")


def analytic_candidate(ratio_X0_over_2Rp, R_p_nm=R_P_NM, psi_deg=PSI_DEG, gamma_s=GAMMA_S, gamma_gb=GAMMA_GB):
    a = ratio_X0_over_2Rp * R_p_nm
    m = young_herring_slope(psi_deg)
    fil_p = solve_body_fillet(a, m, R_p_nm, R_p_nm, +1.0)
    rho = fil_p["rho"]
    z_T = fil_p["z_tangent"]
    r_T = fil_p["r_tangent"]
    g = solve_flat_groove_fillet(a, m, rho)  # groove fillet = particle fillet (symmetric default)
    X_neck = 2 * a
    sigma, sigma_curv, sigma_width, C_GB = hussein_eq1b_sigma(rho * 1e-9, X_neck * 1e-9, gamma_s, gamma_gb)
    return dict(ratio=ratio_X0_over_2Rp, a_nm=a, rho_nm=rho, z_tangent_nm=z_T, r_tangent_nm=r_T,
                X_neck_nm=X_neck, r_transition_nm=g["r_transition"], h_flat_nm=g["h_flat"],
                sigma_Hussein_sharp_MPa=sigma / 1e6, sigma_curvature_MPa=sigma_curv / 1e6,
                sigma_width_MPa=sigma_width / 1e6, rho_over_a=rho / a)


def main():
    os.makedirs(OUT, exist_ok=True)
    ratios = [r / 1000.0 for r in range(20, 261, 5)]  # X0/(2Rp) = 0.020 .. 0.260
    rows = []
    print(f"{'ratio':>7} {'a(nm)':>8} {'rho(nm)':>9} {'z_T(nm)':>8} {'X_neck(nm)':>11} {'sigma_sharp(MPa)':>17}")
    for ratio in ratios:
        try:
            row = analytic_candidate(ratio)
        except (RuntimeError, ValueError) as exc:
            print(f"{ratio:>7.3f}  FAILED: {exc}")
            continue
        rows.append(row)
        print(f"{ratio:>7.3f} {row['a_nm']:>8.2f} {row['rho_nm']:>9.3f} {row['z_tangent_nm']:>8.3f} "
              f"{row['X_neck_nm']:>11.2f} {row['sigma_Hussein_sharp_MPa']:>17.3f}")

    with open(os.path.join(OUT, "analytic_scan.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # highlight candidates with rho in [25,40]nm (Section B target band)
    band = [r for r in rows if 25.0 <= r["rho_nm"] <= 40.0]
    print(f"\nCandidates with rho in [25,40]nm ({len(band)} found):")
    for r in band:
        print(f"  ratio={r['ratio']:.3f} rho={r['rho_nm']:.3f}nm z_T={r['z_tangent_nm']:.3f}nm "
              f"X_neck={r['X_neck_nm']:.2f}nm sigma_sharp={r['sigma_Hussein_sharp_MPa']:.3f}MPa "
              f"rho/W(W=4nm)={r['rho_nm']/4.0:.2f} rho/W(W=6nm)={r['rho_nm']/6.0:.2f}")
    print(f"\nDONE -- wrote {OUT}/analytic_scan.csv")


if __name__ == "__main__":
    main()
