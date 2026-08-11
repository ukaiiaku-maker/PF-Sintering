"""Milestone 15I Sections 11-12, 14: promote the C2 "slow both" candidates
to long trajectories, test an even-slower D variant (half of C's
absolute mobilities at the same R=3), and run the isotropic AN0 control
at the best (R3_C) slow condition.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
            gamma_gb_ratio=1.4, gamma_s=1.0)

SAMPLE_TIMES_LONG = ([0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03,
                      0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
SAMPLE_TIMES_SCREEN = ([0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015,
                        0.0175, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])

CASES = [
    dict(case_id="XLI_R2_C_long", stage="XLI", M_GB_scale=15.0, M_s_scale=0.25, t_target=0.5,
         sample_times=SAMPLE_TIMES_LONG, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
         theta_mis_deg=45.0, aniso_label="AN3"),
    dict(case_id="XLI_R3_C_long", stage="XLI", M_GB_scale=15.0, M_s_scale=1.0 / 6.0, t_target=0.5,
         sample_times=SAMPLE_TIMES_LONG, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
         theta_mis_deg=45.0, aniso_label="AN3"),
    dict(case_id="XLI_R3_D_MGB7.5_Ms0.0833", stage="XLI", M_GB_scale=7.5, M_s_scale=1.0 / 12.0, t_target=0.2,
         sample_times=SAMPLE_TIMES_SCREEN, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
         theta_mis_deg=45.0, aniso_label="AN3"),
    dict(case_id="XLI_R3_C_AN0_control", stage="XLI", M_GB_scale=15.0, M_s_scale=1.0 / 6.0, t_target=0.5,
         sample_times=SAMPLE_TIMES_LONG, use_aniso_surface=False, aniso_delta=None,
         theta_mis_deg=0.0, aniso_label="AN0"),
]


def build_cases():
    return [dict(BASE, **c) for c in CASES]


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage XLI: {len(cases)} cases")
    for c in cases:
        print(f"  {c['case_id']}")
    run_cases(cases)
