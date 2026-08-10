"""Milestone 15G Section 12: long favorable-orientation trajectories.

Promoted from the orientation screen (Section 6-7): all 22 screen cases
stayed H0/H1, but theta_mis_deg=40-45deg with AN3 (delta=0.0647, just
inside the positive-stiffness boundary) at overlap=5nm gave the
strongest A_sigma seen anywhere in the M15 series so far (1.054 at
theta=45, vs 1.000 for the isotropic AN0 control at the SAME
morphology) -- a genuine, quantifiable orientation effect, even though
still solidly H1. Promoting the top 3: theta=45/AN3, theta=40/AN3 (near-
identical, both promoted to check robustness), and theta=45/AN2 (to see
whether the effect strengthens or saturates with anisotropy amplitude)
-- plus the isotropic AN0 control at the same morphology for a fair
long-horizon comparison.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
            gamma_gb_ratio=1.4, gamma_s=1.0, M_GB_scale=30.0, M_s_scale=1.0)

SAMPLE_TIMES = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
                0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

CANDIDATES = [
    dict(case_id="LO_ov5_AN0_control", use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0,
         aniso_label="AN0"),
    dict(case_id="LO_ov5_th45_AN3", use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002, theta_mis_deg=45.0,
         aniso_label="AN3"),
    dict(case_id="LO_ov5_th40_AN3", use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002, theta_mis_deg=40.0,
         aniso_label="AN3"),
    dict(case_id="LO_ov5_th45_AN2", use_aniso_surface=True, aniso_delta=0.045, theta_mis_deg=45.0,
         aniso_label="AN2"),
]


def build_cases():
    return [dict(BASE, stage="LO", t_target=0.5, sample_times=SAMPLE_TIMES, **c) for c in CANDIDATES]


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage LO (long favorable-orientation trajectories): {len(cases)} cases")
    run_cases(cases)
