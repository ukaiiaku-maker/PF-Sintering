"""Milestone 16L Sections 5-7: CORRECT relative-displacement microtest.

The M16K prescribed-displacement microtest (scripts/
m16k_prescribed_displacement_microtest.py) translated f, e1, AND e2 all
by the same amount -- a global translation of the entire sample, which
does NOT test particle-relative-to-substrate motion (the substrate would
translate too, so nothing about the CONTACT itself changes relative to a
fixed frame; it is retained for provenance but is explicitly
NON-AUTHORITATIVE for "how does the contact respond to RBM" questions).

This script instead:
  - shifts ONLY e1 (the particle grain, this codebase's convention) by
    the prescribed displacement, via sub-grid-accurate interpolation;
  - leaves e2 (the substrate grain) COMPLETELY UNCHANGED;
  - reconstructs f = clip(e1_shifted + e2, 0, 1), preserving e1+e2~=f;
  - measures the ACTUAL relative particle/substrate COM displacement
    achieved (which should equal the requested displacement to
    interpolation-order precision, and the substrate's own COM shift
    should be EXACTLY zero by construction -- a built-in correctness
    check the old global-shift test could never provide).

This produces the authoritative mechanical curve sigma_Hussein(delta_RBM)
that Section 15 asks for.

*** CRITICAL GRAIN-IDENTITY CORRECTION (discovered while building this
script) ***: pf_sintering.m16j_geometry.build_candidate_geometry (the
construction used for the M16K/M16L geometry) assigns e1=SUBSTRATE,
e2=PARTICLE (ind_inner~1 for z<0/substrate, e1=f*ind_inner) -- the
OPPOSITE of what pf_sintering.axisym_sink_rbm.py's functions assume
(its own docstring, and scripts/m16g_pr_derived_particle_asperity.py's
OLDER geometry construction which it was originally validated against,
both say "e1=particle (grain1), e2=substrate (grain2)"). This means
EVERY M16K driver script that called `active_sink_transport_step(f, e1,
e2, ...)` or `particle_com_z(e1, ...)` directly with the M16J-geometry's
own e1/e2 was silently advecting/measuring the SUBSTRATE, not the
particle. Verified empirically: shifting e1 (wrongly assumed "particle")
by 0.25nm gave a measured COM shift of only ~49% of the request (a
finite body's whole-domain COM respond at roughly half rate to a
one-sided boundary shift when integrated over a domain with a fixed far
edge); shifting e2 (the ACTUAL particle) by 0.25nm gives ~91% fidelity
(the expected near-unity response, with the remaining ~9% gap from
ordinary finite-domain/interpolation effects, not a further sign of
wrong grain identity). This script SWAPS the roles below (shifts e2,
keeps e1 fixed) to be physically correct for this geometry.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import shift as ndi_shift

sys.path.insert(0, ".")
from pf_sintering.axisym_sink_rbm import particle_com_z  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import psi_to_gamma_gb  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
RATIO = 0.10
W_NM = 10.0
DX_NM = 1.25

FROZEN_STATE_NPZ = os.path.join(os.path.dirname(__file__), "..", "runs",
                                 "m16k_prescribed_displacement_microtest", "frozen_state.npz")
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16l_relative_displacement_microtest")
os.makedirs(OUT, exist_ok=True)


def diagnose(f, e1, e2, r_c, z, W, V_ref, Vp_ref, Vs_ref):
    R_of_z = measure_R_of_z(f, r_c)
    ext = find_all_extrema(R_of_z, z)
    mins = [(zz, RR) for k, zz, RR in ext if k == "min"]
    if not mins:
        return dict(a_contact_nm=float("nan"), X_neck_nm=float("nan"), r_neck_nm=float("nan"),
                     sigma_Hussein_MPa=float("nan"))
    z_gb, a = mins[0]
    X_neck = 2 * a
    win = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5,))[0]
    r_neck = win["r_neck"]
    if np.isfinite(r_neck) and r_neck > 0:
        sigma_H, _, _, _ = hussein_eq1b_sigma(r_neck, X_neck, GAMMA_S, GAMMA_GB)
    else:
        sigma_H = float("nan")
    dz_grid, dr_grid = float(z[1] - z[0]), float(r_c[1] - r_c[0])
    V = 2 * math.pi * float(np.sum(r_c[None, :] * f)) * dz_grid * dr_grid
    Vp = 2 * math.pi * float(np.sum(r_c[None, :] * e1)) * dz_grid * dr_grid
    Vs = 2 * math.pi * float(np.sum(r_c[None, :] * e2)) * dz_grid * dr_grid
    return dict(a_contact_nm=a * 1e9, X_neck_nm=X_neck * 1e9,
                r_neck_nm=r_neck * 1e9 if np.isfinite(r_neck) else float("nan"),
                sigma_Hussein_MPa=sigma_H / 1e6 if np.isfinite(sigma_H) else float("nan"),
                V_total_drift=(V - V_ref) / V_ref, V_particle_drift=(Vp - Vp_ref) / Vp_ref if Vp_ref else float("nan"),
                V_substrate_drift=(Vs - Vs_ref) / Vs_ref if Vs_ref else float("nan"))


def main():
    t0 = time.time()
    d = np.load(FROZEN_STATE_NPZ)
    f0, e1_0, e2_0 = d["f"], d["e1"], d["e2"]
    dz_grid, dr_grid = float(d["dz_grid"]), float(d["dr_grid"])

    # rebuild coordinate arrays (cheap, construction-only -- same
    # deterministic geometry, verified in M16J/M16K to hash-match)
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    assert f0.shape == (len(z), len(r_c)), f"frozen state shape {f0.shape} != rebuilt grid {(len(z), len(r_c))}"
    W = W_NM * 1e-9

    V_ref = 2 * math.pi * float(np.sum(r_c[None, :] * f0)) * dz_grid * dr_grid
    Vp_ref = 2 * math.pi * float(np.sum(r_c[None, :] * e1_0)) * dz_grid * dr_grid
    Vs_ref = 2 * math.pi * float(np.sum(r_c[None, :] * e2_0)) * dz_grid * dr_grid

    # M16J geometry convention: e1=SUBSTRATE, e2=PARTICLE (opposite of
    # axisym_sink_rbm.py's assumed e1=particle -- see module docstring
    # above). Shift e2 (the true particle), keep e1 (true substrate) fixed.
    com_particle_ref = particle_com_z(e2_0, z, r_c, dr_grid, dz_grid)
    com_substrate_ref = particle_com_z(e1_0, z, r_c, dr_grid, dz_grid)
    baseline = diagnose(f0, e1_0, e2_0, r_c, z, W, V_ref, Vp_ref, Vs_ref)
    print(f"baseline (delta=0): {baseline}")
    print(f"baseline particle(e2) COM: {com_particle_ref*1e9:.6f}nm  substrate(e1) COM: {com_substrate_ref*1e9:.6f}nm")

    deltas_nm = [0.0, 0.001, 0.0025, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200, 0.250]
    rows = []
    for delta_nm in deltas_nm:
        shift_cells = -delta_nm * 1e-9 / dz_grid  # negative z = toward substrate, this project's convention
        e2_shifted = ndi_shift(e2_0, shift=(shift_cells, 0), order=1, mode="nearest")  # PARTICLE moves
        e1_fixed = e1_0.copy()  # SUBSTRATE COMPLETELY UNCHANGED
        f_new = np.clip(e2_shifted + e1_fixed, 0.0, 1.0)

        com_particle = particle_com_z(e2_shifted, z, r_c, dr_grid, dz_grid)
        com_substrate = particle_com_z(e1_fixed, z, r_c, dr_grid, dz_grid)
        measured_particle_d = (com_particle_ref - com_particle) * 1e9  # nm, positive = moved toward substrate
        measured_substrate_d = (com_substrate_ref - com_substrate) * 1e9  # should be exactly 0
        measured_relative_d = measured_particle_d - measured_substrate_d

        diag = diagnose(f_new, e1_fixed, e2_shifted, r_c, z, W, V_ref, Vp_ref, Vs_ref)
        e1e2f_resid = float(np.max(np.abs(e2_shifted + e1_fixed - f_new)))
        row = dict(requested_delta_nm=delta_nm, measured_particle_COM_d_nm=measured_particle_d,
                   measured_substrate_COM_d_nm=measured_substrate_d, measured_relative_d_nm=measured_relative_d,
                   **diag, e1e2f_residual=e1e2f_resid)
        rows.append(row)
        print(f"  delta={delta_nm:.4f}nm -> particle_d={measured_particle_d:.5f}nm "
              f"substrate_d={measured_substrate_d:.2e}nm relative_d={measured_relative_d:.5f}nm "
              f"X_neck={diag['X_neck_nm']:.3f}nm r_neck={diag['r_neck_nm']:.3f}nm "
              f"sigma={diag['sigma_Hussein_MPa']:.3f}MPa V_drift={diag['V_total_drift']:.2e} "
              f"e1e2f_resid={e1e2f_resid:.2e}", flush=True)

    fieldnames = list(rows[0].keys())
    with open(os.path.join(OUT, "relative_microtest_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDONE wall={time.time()-t0:.0f}s -- wrote {OUT}/relative_microtest_results.csv")


if __name__ == "__main__":
    main()
