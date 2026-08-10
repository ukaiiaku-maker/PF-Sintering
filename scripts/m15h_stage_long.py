"""Milestone 15H Section 11/12/15: long trajectories at the best rate
ratio found (Section 7-8: R=2.5-3 gives the deepest/longest transient
narrowing dip -- C1, not C2/C3 -- of the whole R=1.5-6 ladder tested).
No C2/C3 candidate exists to "promote" per Section 8's literal
instruction, but Section 12 (S75/S100 gate) and Section 15 (isotropic
control) still need long-horizon data, and confirming no LATER
narrowing recurrence appears past t=0.2s is itself part of Section 11's
intent. Runs the best anisotropic candidate (R=3, theta=45deg/AN3) and
its isotropic control (R=2.5, AN0 -- matching the isotropic screen's
own best dip) to t=0.5s.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
            gamma_gb_ratio=1.4, gamma_s=1.0, M_s_scale=1.0)

SAMPLE_TIMES = [0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03,
                0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

CANDIDATES = [
    dict(case_id="XL_th45_AN3_R3", M_GB_scale=90.0, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
         theta_mis_deg=45.0, aniso_label="AN3"),
    dict(case_id="XL_AN0_R2.5", M_GB_scale=75.0, use_aniso_surface=False, aniso_delta=None,
         theta_mis_deg=0.0, aniso_label="AN0"),
]


def build_cases():
    return [dict(BASE, stage="XL", t_target=0.5, sample_times=SAMPLE_TIMES, **c) for c in CANDIDATES]


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage XL (long crossover trajectories): {len(cases)} cases")
    run_cases(cases)
