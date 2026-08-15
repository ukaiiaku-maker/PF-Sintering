"""M16M continuation Section 8: trough-versus-TJ geometry audit.

Explicitly computes and compares two candidate "neck contact" coordinates
that the M16H-M16M lineage has, until now, implicitly conflated:

  A. the free-surface R(z) local minimum ("trough") -- what the
     production driver (NeckTracker) has used as the authoritative
     z_gb/a_contact throughout M16H-M16L.
  B. the true grain-identity (particle/substrate) crossing along the
     traced f=0.5 contour ("TJ") -- pf_sintering.m16k_neck_tracking.
     find_tj_from_contour, sub-grid interpolated.

Run from the repo root: ../.venv/bin/python scripts/m16m_tj_vs_trough_audit.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402
from pf_sintering.m16k_neck_tracking import find_tj_from_contour  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402

R_P_NM, RATIO, W_NM, DX_NM = 1000.0, 0.10, 10.0, 1.25

geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=160.0,
                                W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9


def audit(label, f, e1, e2):
    """f, e1, e2 in M16J convention (e1=substrate, e2=particle)."""
    roles = roles_from_m16j_geometry(e1, e2)
    R_of_z = measure_R_of_z(f, r_c)
    ext = find_all_extrema(R_of_z, z)
    mins = sorted([(zz, RR) for k, zz, RR in ext if k == "min"], key=lambda c: c[1])
    if not mins:
        print(f"{label}: NO MINIMA FOUND")
        return
    z_min, R_min = mins[0]
    tj = find_tj_from_contour(f, roles.particle, roles.substrate, r_c, z, R_of_z, z_min)
    z_tj, R_tj = tj["z_tj"], tj["r_tj"]
    print(f"{label}:")
    print(f"   z_min={z_min*1e9:8.4f}nm  R_min={R_min*1e9:8.4f}nm  X_min=2R_min={2*R_min*1e9:8.4f}nm")
    if np.isfinite(z_tj):
        print(f"   z_TJ ={z_tj*1e9:8.4f}nm  R_TJ ={R_tj*1e9:8.4f}nm  X_TJ =2R_TJ ={2*R_tj*1e9:8.4f}nm")
        print(f"   |z_TJ - z_min| = {abs(z_tj-z_min)*1e9:.4f}nm    "
              f"|X_TJ - X_min|/X_min = {abs(2*R_tj-2*R_min)/(2*R_min)*100:.2f}%")
    else:
        print("   z_TJ: NOT FOUND (no grain-identity sign change in search window)")
    print(f"   n_candidate_minima={len(mins)} all={[(round(zz*1e9,3), round(RR*1e9,3)) for zz,RR in mins]}")
    print()


def main():
    audit("t=0 (initial construction)", geo["f"], geo["e1"], geo["e2"])
    d = np.load("runs/m16k_prescribed_displacement_microtest/frozen_state.npz")
    audit("step=90490 t=4.4185 (~45.1MPa, near activation)", d["f"], d["e1"], d["e2"])


if __name__ == "__main__":
    main()
