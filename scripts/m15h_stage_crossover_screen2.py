"""Milestone 15H Section 7 (extended): the R=1.5-3.0 ladder gave C1
(transient narrowing, ~0.4-0.5nm dip, ~0.015-0.025s duration) at EVERY
R tested, with dip DEPTH growing but duration roughly FLAT or even
shrinking as R increases (0.447nm/0.025s at R=1.5 vs 0.529nm/0.018s at
R=3.0) -- the instantaneous chi(t) at R=3.0 confirms chi=1.04 at
t=0.01 but has already fallen to chi=0.77 by t=0.02, a genuinely
short-lived crossover WINDOW in time, not simply extendable by scaling
R. Extending the ladder to R=4,5,6 to determine whether the dip
duration eventually crosses the 0.1s C2 threshold or saturates.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
            gamma_gb_ratio=1.4, gamma_s=1.0, M_s_scale=1.0)

SAMPLE_TIMES = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02,
                0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

R_LADDER = [4.0, 5.0, 6.0]


def build_cases():
    cases = []
    for R in R_LADDER:
        M_GB_scale = 30.0 * R
        case_id = f"X_th45_AN3_R{R:g}"
        cases.append(dict(BASE, case_id=case_id, stage="X", M_GB_scale=M_GB_scale, t_target=0.20,
                           sample_times=SAMPLE_TIMES, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
                           theta_mis_deg=45.0, aniso_label="AN3"))
    return cases


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage X (extended crossover screen): {len(cases)} cases")
    run_cases(cases)
