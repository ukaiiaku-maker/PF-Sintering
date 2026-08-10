"""Milestone 15E Stage B: adaptive physics refinement.

Takes the best 4 valid Stage-A morphologies (ranked by A_sigma) and
varies ONE physics family at a time: aspect_ratio in {1.5,2.0,3.0},
gamma_GB/gamma_s in {1.0,1.4,1.7} (<2 always), rate competition
(M_GB_scale,M_s_scale) in {(30,1),(100,0.1),(100,1/30)}. t<=0.10s.
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15e_campaign_lib import CAMPAIGN_DIR, MANIFEST_PATH, case_path, run_cases  # noqa: E402

SAMPLE_TIMES = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]


def _endpoint_slope(case_id, n_tail=4):
    """Pa/s slope of sigma_endpoint over the last n_tail samples. Stage A
    (t<=0.06s) turned out to be short enough that every valid morphology
    is still in its initial monotonic-relaxation phase (A_sigma degenerate
    at 1.0 for essentially all cases -- see MILESTONE_15E report Section
    'Stage A'), so A_sigma cannot discriminate morphologies here. The
    least-negative (most-flattened) endpoint slope is the best available
    proxy for which morphology is closest to its true post-transient
    reset, and is used as the Stage-B tiebreak instead."""
    with open(case_path(case_id)) as fh:
        d = json.load(fh)
    rows = d["trajectory"]["rows"]
    sig = [(r["t"], r.get("sigma_sint_app_endpoint_form")) for r in rows
           if r.get("sigma_sint_app_endpoint_form") is not None]
    tail = sig[-n_tail:]
    (t0, s0), (t1, s1) = tail[0], tail[-1]
    return (s1 - s0) / (t1 - t0) if t1 > t0 else float("nan")


def best_stage_a_morphologies(n=4):
    rows = []
    with open(MANIFEST_PATH) as fh:
        for row in csv.DictReader(fh):
            if row["stage"] != "A" or row["status"] != "ok":
                continue
            rows.append(row)
    for r in rows:
        r["_slope"] = _endpoint_slope(r["case_id"])
    # Primary: A_sigma descending (rewards any case that has actually
    # started to turn over). Secondary: least-negative endpoint slope
    # (closest to turnover) when A_sigma is tied/degenerate.
    rows.sort(key=lambda r: (-float(r["A_sigma"]), r["_slope"]))
    seen = set()
    out = []
    for r in rows:
        key = (r["A_nm"], r["lambda_nm"], r["overlap_nm"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


def build_cases(base_rows):
    cases = []
    for base in base_rows:
        A_nm, lambda_nm, overlap_nm = float(base["A_nm"]), float(base["lambda_nm"]), float(base["overlap_nm"])
        tag = f"A{A_nm:g}_L{lambda_nm:g}_ov{overlap_nm:g}"

        for aspect in (1.5, 2.0, 3.0):
            case_id = f"B_{tag}_aspect{aspect:g}"
            cases.append(dict(case_id=case_id, stage="B", A_nm=A_nm, lambda_nm=lambda_nm, R2_nm=80.0,
                               aspect_ratio=aspect, overlap_nm=overlap_nm, gamma_gb_ratio=1.4, gamma_s=1.0,
                               M_GB_scale=100.0, M_s_scale=0.1, dx_nm=2.5, t_target=0.10,
                               sample_times=SAMPLE_TIMES))

        for gb_ratio in (1.0, 1.4, 1.7):
            case_id = f"B_{tag}_gbratio{gb_ratio:g}"
            cases.append(dict(case_id=case_id, stage="B", A_nm=A_nm, lambda_nm=lambda_nm, R2_nm=80.0,
                               aspect_ratio=2.0, overlap_nm=overlap_nm, gamma_gb_ratio=gb_ratio, gamma_s=1.0,
                               M_GB_scale=100.0, M_s_scale=0.1, dx_nm=2.5, t_target=0.10,
                               sample_times=SAMPLE_TIMES))

        for M_GB_scale, M_s_scale in ((30.0, 1.0), (100.0, 0.1), (100.0, 1.0 / 30.0)):
            case_id = f"B_{tag}_MGB{M_GB_scale:g}_Ms{M_s_scale:.4f}"
            cases.append(dict(case_id=case_id, stage="B", A_nm=A_nm, lambda_nm=lambda_nm, R2_nm=80.0,
                               aspect_ratio=2.0, overlap_nm=overlap_nm, gamma_gb_ratio=1.4, gamma_s=1.0,
                               M_GB_scale=M_GB_scale, M_s_scale=M_s_scale, dx_nm=2.5, t_target=0.10,
                               sample_times=SAMPLE_TIMES))
    return cases


if __name__ == "__main__":
    base_rows = best_stage_a_morphologies(4)
    print("Best Stage-A morphologies:")
    for r in base_rows:
        print(f"  A={r['A_nm']} lambda={r['lambda_nm']} overlap={r['overlap_nm']} "
              f"A_sigma={r['A_sigma']} endpoint_slope={r['_slope']:.3e} Pa/s")
    cases = build_cases(base_rows)
    print(f"Stage B: {len(cases)} cases")
    run_cases(cases)
