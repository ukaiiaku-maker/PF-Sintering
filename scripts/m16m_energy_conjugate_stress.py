"""M16M continuation Section 14: energy-conjugate stress diagnostic.

Independent, GLOBAL check on the local Hussein curvature-based stress:
take a frozen PF state, apply small conservative +/- particle-relative
virtual displacements (same physically-correct construction M16L's
relative-displacement microtest uses -- shift ONLY the particle grain
`e2`, keep the substrate grain `e1` completely fixed, reconstruct
f=clip(e2_shifted+e1,0,1)), evaluate the TOTAL PF free energy
(axisym_free_energy_gb) at each shift, and estimate the axial force
conjugate to that displacement via a central finite difference:

    F_sint = -dF/d(delta)
    sigma_energy = F_sint / A_GB,  A_GB = pi * a_contact^2

using the SAME (path-continuous, R(z)-minimum-based) a_contact the
production driver treats as authoritative for X_neck (Section 8's TJ-vs-
trough audit).

This is DIAGNOSTIC ONLY -- it does not replace sigma_Hussein. Verifies a
linear finite-difference regime by checking F(delta) is symmetric/smooth
across several small delta magnitudes before trusting the derivative
estimate.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.ndimage import shift as ndi_shift

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
RATIO = 0.10
W_NM = 10.0
DX_NM = 1.25


def energy_at_shift(f0, e1_fixed, e2_0, delta_m, dz_grid, p, Wc, dr, dz, r_c, r_f):
    shift_cells = -delta_m / dz_grid
    e2_shifted = ndi_shift(e2_0, shift=(shift_cells, 0), order=1, mode="nearest")
    f_new = np.clip(e2_shifted + e1_fixed, 0.0, 1.0)
    F = axisym_free_energy_gb(f_new, e1_fixed, e2_shifted, p, Wc, dr, dz, r_c, r_f, bc_z="noflux")
    return F


def main():
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    W = W_NM * 1e-9
    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]

    d = np.load("runs/m16k_prescribed_displacement_microtest/frozen_state.npz")
    f0, e1_0, e2_0 = d["f"], d["e1"], d["e2"]  # M16J convention: e1=substrate, e2=particle
    dz_grid = float(d["dz_grid"])
    assert f0.shape == (len(z), len(r_c)), "frozen state grid mismatch"

    # authoritative a_contact via R(z)-minimum (Section 8 finding: use
    # trough, not raw contour-crossing TJ, as the X_neck source)
    R_of_z = measure_R_of_z(f0, r_c)
    ext = find_all_extrema(R_of_z, z)
    mins = sorted([(zz, RR) for k, zz, RR in ext if k == "min"], key=lambda c: c[1])
    z_min, a_contact = mins[0]
    A_GB = np.pi * a_contact ** 2
    print(f"reference state: z_min={z_min*1e9:.4f}nm a_contact={a_contact*1e9:.4f}nm A_GB={A_GB*1e18:.4f}nm^2")

    deltas_nm = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04]
    F_pos, F_neg = {}, {}
    F0 = energy_at_shift(f0, e1_0, e2_0, 0.0, dz_grid, p, Wc, dr, dz, r_c, r_f)
    print(f"F(delta=0) = {F0:.10e} J")

    for dnm in deltas_nm[1:]:
        d_m = dnm * 1e-9
        Fp = energy_at_shift(f0, e1_0, e2_0, +d_m, dz_grid, p, Wc, dr, dz, r_c, r_f)
        Fm = energy_at_shift(f0, e1_0, e2_0, -d_m, dz_grid, p, Wc, dr, dz, r_c, r_f)
        F_pos[dnm], F_neg[dnm] = Fp, Fm
        dFdd_central = (Fp - Fm) / (2 * d_m)
        F_sint = -dFdd_central
        sigma_energy = F_sint / A_GB
        # symmetry/linearity check: (Fp+Fm-2F0)/d^2 should be roughly
        # constant across delta (a 2nd-derivative / curvature-of-F proxy)
        curvature_proxy = (Fp + Fm - 2 * F0) / (d_m ** 2)
        print(f"delta={dnm:6.4f}nm  F+={Fp:.10e}  F-={Fm:.10e}  dF/dd={dFdd_central:.6e} J/m  "
              f"F_sint={F_sint:.6e}N  sigma_energy={sigma_energy/1e6:.4f}MPa  "
              f"curvature_proxy={curvature_proxy:.4e}")


if __name__ == "__main__":
    main()
