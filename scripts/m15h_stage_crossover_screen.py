"""Milestone 15H Section 6-7: predicted crossover + bounded full-trajectory
rate screen.

Section 5's linearity test confirmed dL_SURF/dt propto M_f_scale and
dL_GB/dt propto M_eta_scale EXACTLY (linear across 0.5-2.0x at all 3
checkpoints, both configs). Using the measured baseline
chi(M_GB_scale=30,M_s_scale=1) at t=0.01/0.02/0.10s
(AN0: 0.89/1.07/0.50; theta=45/AN3: 0.92/1.12/0.60), the rate ratio
R=M_GB_scale_mult/M_s_scale_mult required for chi>=1.1 at ALL THREE
checkpoints simultaneously (the hardest constraint is always t=0.10s,
where chi is smallest) is:

    AN0:        R = 1.1/0.50 = 2.2   -> M_GB_scale ~= 66
    th45/AN3:   R = 1.1/0.60 = 1.84  -> M_GB_scale ~= 55

(holding M_s_scale=1 and scaling M_GB_scale=30*R, matching Section 7's
own suggested range "M_GB scale ~30-60"). This screen runs a bounded
ladder of R around these predictions for BOTH configs (satisfying
Section 15's isotropic-vs-anisotropic comparison in the same screen),
t_target=0.20s.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
            gamma_gb_ratio=1.4, gamma_s=1.0, M_s_scale=1.0)

SAMPLE_TIMES = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

R_LADDER_AN3 = [1.5, 2.0, 2.5, 3.0]
R_LADDER_AN0 = [1.5, 2.0, 2.5]


def build_cases():
    cases = []
    for R in R_LADDER_AN3:
        M_GB_scale = 30.0 * R
        case_id = f"X_th45_AN3_R{R:g}"
        cases.append(dict(BASE, case_id=case_id, stage="X", M_GB_scale=M_GB_scale, t_target=0.20,
                           sample_times=SAMPLE_TIMES, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
                           theta_mis_deg=45.0, aniso_label="AN3"))
    for R in R_LADDER_AN0:
        M_GB_scale = 30.0 * R
        case_id = f"X_AN0_R{R:g}"
        cases.append(dict(BASE, case_id=case_id, stage="X", M_GB_scale=M_GB_scale, t_target=0.20,
                           sample_times=SAMPLE_TIMES, use_aniso_surface=False, aniso_delta=None,
                           theta_mis_deg=0.0, aniso_label="AN0"))
    return cases


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage X (crossover screen): {len(cases)} cases")
    for c in cases:
        print(f"  {c['case_id']} M_GB_scale={c['M_GB_scale']}")
    run_cases(cases)
