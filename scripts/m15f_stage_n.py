"""Milestone 15F Section 11-13: primary neck screen.

Geometries G1 (A=100,lambda=320), G2 (A=120,lambda=320). Overlaps
restricted to what actually resolves at each W (probed directly before
writing this script -- tj_subgrid's Newton-based contact locator is
sensitive to the exact W/overlap combination, not just a simple
overlap<W geometric threshold): W=20nm supports overlap in {5,10,15}nm
for both geometries; W=10nm reliably supports only overlap=10nm.

W in {20,10} x AN in {AN0,AN1,AN2} x orientation A (theta_mis_deg=0)
first (Section 6: "do not perform a dense angular sweep initially").
Short screen (t_target=0.06s, dx matched to W per the ladder). After
this, Section 13's facet diagnostics and Section 12's promotion metric
identify which (W,AN) combinations are worth a second orientation.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

GEOMETRIES = [("G1", 100.0, 320.0), ("G2", 120.0, 320.0)]
AN_LEVELS = dict(AN0=(False, None), AN1=(True, 0.02), AN2=(True, 0.045))
SAMPLE_TIMES = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06]

BASE = dict(R2_nm=80.0, aspect_ratio=2.0, gamma_gb_ratio=1.4, gamma_s=1.0,
            M_GB_scale=30.0, M_s_scale=1.0)


def build_cases(theta_mis_deg=0.0, orientation_label="A"):
    # W=10nm/dx=1.25nm cases are ~10-20x slower per case than W=20nm/dx=2.5nm
    # (empirically: Section 10's t=0.02s W=10 qualification cases took
    # 224-480s wall each). Per Section 11's "do NOT automatically run the
    # full Cartesian matrix" instruction, the expensive W=10 leg is
    # restricted to G2 only (the M15E-winning morphology) x both overlaps
    # that resolve there; the cheap W=20 leg keeps the fuller
    # geometry x overlap matrix.
    cases = []
    for W_nm, dx_nm, geoms_overlaps in (
        (20.0, 2.5, [(gname, A_nm, L_nm, ov) for gname, A_nm, L_nm in GEOMETRIES for ov in (5.0, 10.0, 15.0)]),
        (10.0, 1.25, [("G2", 120.0, 320.0, 10.0)]),
    ):
        for gname, A_nm, L_nm, overlap_nm in geoms_overlaps:
            for an_label, (use_aniso, delta) in AN_LEVELS.items():
                case_id = f"N_{gname}_ov{overlap_nm:g}_W{W_nm:g}_{an_label}_{orientation_label}"
                cases.append(dict(BASE, case_id=case_id, stage="N", A_nm=A_nm, lambda_nm=L_nm,
                                   overlap_nm=overlap_nm, dx_nm=dx_nm, W_nm=W_nm, t_target=0.06,
                                   sample_times=SAMPLE_TIMES, use_aniso_surface=use_aniso,
                                   aniso_delta=delta, theta_mis_deg=theta_mis_deg,
                                   aniso_label=an_label))
    return cases


if __name__ == "__main__":
    cases = build_cases(theta_mis_deg=0.0, orientation_label="A")
    print(f"Stage N (orientation A, theta_mis_deg=0): {len(cases)} cases")
    run_cases(cases)
