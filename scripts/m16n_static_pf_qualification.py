"""M16N Section C: static PF representation qualification (NO evolution).

For the top candidates from the Section A analytic scan
(runs/m16n_analytic_geometry_scan/analytic_scan.csv), constructs the PF
field at t=0 ONLY (via build_candidate_geometry) at W=6nm and W=4nm, and
measures the initial curvature with FIXED physical windows -- no PF
stepping. Requires the extracted radius agree with the analytic `rho`
target within ~10% to qualify a (geometry, W) pair. This is a
well-defined benchmark now that rho_true is known analytically for the
EXACT geometry actually being constructed (not a separate synthetic
proxy, per Section C's "This is now a well-defined benchmark because
rho_true is known").
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import psi_to_gamma_gb  # noqa: E402
from m16n_analytic_geometry_scan import analytic_candidate  # noqa: E402

GAMMA_S = 1.0
PSI_DEG = 160.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16n_static_pf_qualification")

CANDIDATE_RATIOS = [0.185, 0.200, 0.215]
W_DX_PAIRS = [(6.0, 0.85), (4.0, 0.55)]
# windows chosen to bracket rho for these candidates (rho~27-38nm):
# small windows should now be BELOW rho (well-resolved per the analytic
# benchmark), larger windows above it (should show the expected breakdown).
FIXED_WINDOWS_NM = (6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 30.0, 36.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print(f"{'ratio':>7} {'W(nm)':>6} {'dx(nm)':>7} {'rho_true':>9}  best-window-agreement")
    for ratio in CANDIDATE_RATIOS:
        target = analytic_candidate(ratio)
        rho_true = target["rho_nm"]
        for W_nm, dx_nm in W_DX_PAIRS:
            geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=ratio, psi_deg=PSI_DEG,
                                            W_nm=W_nm, dr_nm=dx_nm, dz_nm=dx_nm, aspect_ratio=1.0)
            f = geo["f"]
            z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
            R_of_z = measure_R_of_z(f, r_c)
            ext = find_all_extrema(R_of_z, z)
            mins = [(zz, RR) for k, zz, RR in ext if k == "min"]
            if not mins:
                print(f"{ratio:>7.3f} {W_nm:>6.1f} {dx_nm:>7.2f}  NO MINIMUM FOUND")
                continue
            z_min, a_meas = mins[0]
            X_neck = 2 * a_meas
            windows = neck_curvature_windows(R_of_z, z, z_min, 1e-9, window_widths_in_W=FIXED_WINDOWS_NM)
            best_agree_pct, best_w = None, None
            for w_nm, win in zip(FIXED_WINDOWS_NM, windows):
                r_neck = win["r_neck"]
                if not (np.isfinite(r_neck) and r_neck > 0):
                    continue
                r_neck_nm = r_neck * 1e9
                agree_pct = 100.0 * abs(r_neck_nm - rho_true) / rho_true
                sigma, *_ = hussein_eq1b_sigma(r_neck, X_neck, GAMMA_S, GAMMA_GB)
                row = dict(ratio=ratio, W_nm=W_nm, dx_nm=dx_nm, rho_true_nm=rho_true, window_nm=w_nm,
                           window_over_rho=w_nm / rho_true, r_neck_meas_nm=r_neck_nm,
                           agreement_pct=agree_pct, qualified=agree_pct <= 10.0,
                           X_neck_meas_nm=X_neck * 1e9, a_meas_nm=a_meas * 1e9,
                           sigma_meas_MPa=sigma / 1e6, sigma_true_MPa=target["sigma_Hussein_sharp_MPa"])
                rows.append(row)
                if best_agree_pct is None or agree_pct < best_agree_pct:
                    best_agree_pct, best_w = agree_pct, w_nm
            print(f"{ratio:>7.3f} {W_nm:>6.1f} {dx_nm:>7.2f} {rho_true:>9.3f}  "
                  f"best window={best_w}nm agreement={best_agree_pct:.2f}% "
                  f"{'QUALIFIED' if best_agree_pct is not None and best_agree_pct <= 10.0 else 'NOT qualified'}")

    with open(os.path.join(OUT, "static_qualification.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDONE -- wrote {OUT}/static_qualification.csv")


if __name__ == "__main__":
    main()
