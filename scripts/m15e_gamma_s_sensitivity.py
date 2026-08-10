"""Milestone 15E Section 10: conditional material-surface-energy
sensitivity on the single BEST mechanically-amplifying trajectory
(picked from Stage C/D once available). Sweeps gamma_s in {1.0,1.5,2.0}
J/m^2 holding gamma_GB/gamma_s fixed and using the SAME physical
mobility scale factors -- NOT re-tuned to force S75/S100. Reports both
absolute sigma_endpoint and A_sigma; does not designate any gamma_s as
"the" material value.
"""
from __future__ import annotations

import csv
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15e_campaign_lib import run_cases  # noqa: E402

GAMMA_S_VALUES = [1.0, 1.5, 2.0]


def build_cases(base_case_id, base_row, t_target=None, sample_times=None):
    t_target = t_target if t_target is not None else float(base_row["t_target"])
    if sample_times is None:
        sample_times = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
                         0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    cases = []
    for gamma_s in GAMMA_S_VALUES:
        case_id = f"G_{base_case_id}_gs{gamma_s:g}"
        cases.append(dict(
            case_id=case_id, stage="G", A_nm=float(base_row["A_nm"]), lambda_nm=float(base_row["lambda_nm"]),
            R2_nm=float(base_row["R2_nm"]), aspect_ratio=float(base_row["aspect_ratio"]),
            overlap_nm=float(base_row["overlap_nm"]), gamma_gb_ratio=float(base_row["gamma_gb_ratio"]),
            gamma_s=gamma_s, M_GB_scale=float(base_row["M_GB_scale"]), M_s_scale=float(base_row["M_s_scale"]),
            dx_nm=float(base_row["dx_nm"]), t_target=t_target, sample_times=sample_times,
        ))
    return cases


if __name__ == "__main__":
    import argparse
    from m15e_campaign_lib import MANIFEST_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True, help="manifest case_id of the best trajectory to sweep")
    ap.add_argument("--t-target", type=float, default=None)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(MANIFEST_PATH)))
    match = [r for r in rows if r["case_id"] == args.case_id]
    if not match:
        raise SystemExit(f"case_id {args.case_id!r} not found in manifest")
    base_row = match[-1]
    cases = build_cases(args.case_id, base_row, t_target=args.t_target)
    print(f"gamma_s sensitivity on {args.case_id}: {len(cases)} cases")
    run_cases(cases)
