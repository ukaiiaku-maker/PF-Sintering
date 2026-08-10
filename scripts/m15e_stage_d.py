"""Milestone 15E Section 7 (Stage D): high-resolution narrow-contact
region. Milestone 15D found overlap<=3nm at dx=2.5nm gave apparent
contact near ~18.7nm but unresolved TJ/arc diagnostics -- important
because 75 MPa needs L_contact<=26.7nm and 100 MPa needs <=20nm (at
gamma_s=1, max alignment). Revisit at dx=1.25nm for the best one or two
Stage-C substrate geometries, narrow overlaps only. Short horizon: first
require robust resolution, then check whether L_contact narrows
post-relaxation rather than broadens (do NOT trust t=0 values).
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15e_campaign_lib import run_cases  # noqa: E402

SAMPLE_TIMES = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15]


def build_cases(morphologies, overlaps, gamma_gb_ratio=1.4, M_GB_scale=30.0, M_s_scale=1.0,
                 dx_nm=1.25, t_target=0.15, aspect_ratio=2.0, R2_nm=80.0):
    cases = []
    for A_nm, lambda_nm in morphologies:
        for overlap_nm in overlaps:
            case_id = f"D_A{A_nm:g}_L{lambda_nm:g}_ov{overlap_nm:g}_dx{dx_nm:g}"
            cases.append(dict(
                case_id=case_id, stage="D", A_nm=float(A_nm), lambda_nm=float(lambda_nm), R2_nm=R2_nm,
                aspect_ratio=aspect_ratio, overlap_nm=overlap_nm, gamma_gb_ratio=gamma_gb_ratio, gamma_s=1.0,
                M_GB_scale=M_GB_scale, M_s_scale=M_s_scale, dx_nm=dx_nm, t_target=t_target,
                sample_times=SAMPLE_TIMES,
            ))
    return cases


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--morphologies", default="120,320", help="semicolon-separated A,lambda pairs in nm")
    ap.add_argument("--overlaps", default="2,3,4,5,7.5,10", help="comma-separated overlaps in nm")
    ap.add_argument("--t-target", type=float, default=0.15)
    args = ap.parse_args()
    morphologies = [tuple(float(x) for x in pair.split(",")) for pair in args.morphologies.split(";")]
    overlaps = [float(x) for x in args.overlaps.split(",")]
    cases = build_cases(morphologies, overlaps, t_target=args.t_target)
    print(f"Stage D: {len(cases)} cases at dx=1.25nm")
    for c in cases:
        print(f"  {c['case_id']}")
    run_cases(cases)
