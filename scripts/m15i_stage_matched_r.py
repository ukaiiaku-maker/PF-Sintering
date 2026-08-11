"""Milestone 15I Sections 2-4: matched-R rate families.

Same nominal R=M_GB_scale_mult/M_s_scale_mult but different ABSOLUTE
mobilities, to separate the rate-ratio effect (M15H) from an absolute-
timescale effect. A=125*M_GB_REF... no: A="speed GB" (M_s baseline,
M_GB scaled up), B="slow surface" (M_GB baseline=30, M_s scaled down),
C="slow both at fixed R" (both scaled down from A/baseline).

theta=45deg/AN3 primary condition (A=120nm,lambda=320nm,overlap=5nm,
W=20nm/dx=2.5nm,gamma_GB/gamma_s=1.4), t_target=0.20s, dense sampling
0-0.05s.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
            gamma_gb_ratio=1.4, gamma_s=1.0, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
            theta_mis_deg=45.0, aniso_label="AN3")

SAMPLE_TIMES = ([0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005,
                 0.006, 0.007, 0.008, 0.009, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03,
                 0.035, 0.04, 0.045, 0.05] +
                [0.06, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])

# (M_GB_scale, M_s_scale) triples per R
FAMILIES = {
    2: dict(A=(60.0, 1.0), B=(30.0, 0.5), C=(15.0, 0.25)),
    3: dict(A=(90.0, 1.0), B=(30.0, 1.0 / 3.0), C=(15.0, 1.0 / 6.0)),
}


def build_cases():
    cases = []
    for R, fam in FAMILIES.items():
        for label, (M_GB_scale, M_s_scale) in fam.items():
            case_id = f"MR_R{R}_{label}_MGB{M_GB_scale:g}_Ms{M_s_scale:.4f}"
            cases.append(dict(BASE, case_id=case_id, stage="MR", M_GB_scale=M_GB_scale, M_s_scale=M_s_scale,
                               t_target=0.20, sample_times=SAMPLE_TIMES))
    return cases


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage MR (matched-R families): {len(cases)} cases")
    for c in cases:
        print(f"  {c['case_id']}")
    run_cases(cases)
