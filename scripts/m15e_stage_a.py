"""Milestone 15E Stage A: cheap morphology screen.

dx=2.5nm, t=0.06s, gamma_s=1.0, gamma_GB/gamma_s=1.4, M_GB=100*M_GB_ref,
M_s=baseline/10. (A_nm,lambda_nm) x overlap_nm, 6x3=18 cases.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15e_campaign_lib import run_cases  # noqa: E402

MORPHOLOGIES = [(80, 360), (100, 360), (100, 320), (120, 320), (120, 280), (140, 280)]
OVERLAPS = [15.0, 20.0, 30.0]
SAMPLE_TIMES = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06]


def build_cases():
    cases = []
    for A_nm, lambda_nm in MORPHOLOGIES:
        for overlap_nm in OVERLAPS:
            case_id = f"A_A{A_nm}_L{lambda_nm}_ov{overlap_nm:g}"
            cases.append(dict(
                case_id=case_id, stage="A", A_nm=float(A_nm), lambda_nm=float(lambda_nm),
                R2_nm=80.0, aspect_ratio=2.0, overlap_nm=overlap_nm,
                gamma_gb_ratio=1.4, gamma_s=1.0, M_GB_scale=100.0, M_s_scale=0.1,
                dx_nm=2.5, t_target=0.06, sample_times=SAMPLE_TIMES,
            ))
    return cases


if __name__ == "__main__":
    run_cases(build_cases())
