"""Milestone 15E Stage C: longer dx=2.5nm runs on the top candidates.

Takes the top candidates satisfying Section 5's early-promotion criterion
(or, if none/too few, the top N by A_sigma) from Stage B, reruns to
t=0.4-0.6s with denser sampling.
"""
from __future__ import annotations

import csv
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15e_campaign_lib import MANIFEST_PATH, run_cases  # noqa: E402

SAMPLE_TIMES = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
                0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


def top_stage_b_candidates(n=6, min_a_sigma=1.0):
    rows = []
    with open(MANIFEST_PATH) as fh:
        for row in csv.DictReader(fh):
            if row["stage"] != "B" or row["status"] != "ok":
                continue
            rows.append(row)
    rows.sort(key=lambda r: float(r["A_sigma"]), reverse=True)
    return [r for r in rows if float(r["A_sigma"]) >= min_a_sigma][:n] or rows[:n]


def build_cases(base_rows, t_target=0.5):
    cases = []
    for i, r in enumerate(base_rows):
        case_id = f"C_{r['case_id']}"
        cases.append(dict(
            case_id=case_id, stage="C", A_nm=float(r["A_nm"]), lambda_nm=float(r["lambda_nm"]),
            R2_nm=float(r["R2_nm"]), aspect_ratio=float(r["aspect_ratio"]), overlap_nm=float(r["overlap_nm"]),
            gamma_gb_ratio=float(r["gamma_gb_ratio"]), gamma_s=float(r["gamma_s"]),
            M_GB_scale=float(r["M_GB_scale"]), M_s_scale=float(r["M_s_scale"]),
            dx_nm=2.5, t_target=t_target, sample_times=SAMPLE_TIMES,
        ))
    return cases


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--t-target", type=float, default=0.5)
    args = ap.parse_args()
    base_rows = top_stage_b_candidates(args.n)
    print(f"Stage C: {len(base_rows)} candidates")
    for r in base_rows:
        print(f"  {r['case_id']} A_sigma={r['A_sigma']}")
    cases = build_cases(base_rows, t_target=args.t_target)
    run_cases(cases)
