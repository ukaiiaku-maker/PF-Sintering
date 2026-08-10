"""Milestone 15F Section 14: long trajectories on the promoted candidates.

Selection (from Stage N's 21-case screen, all still H0/H1 at t=0.06s):
  - N_G2_ov5_W20_AN0_A: best overall (A_sigma=1.028, sharpest resolvable
    contact at W=20, ov=5nm -- ov=5 does NOT resolve at W=10, confirmed
    by direct probe).
  - N_G2_ov5_W20_AN1_A: anisotropy comparison on the winning morphology.
  - N_G2_ov10_W20_AN0_A / N_G2_ov10_W10_AN0_A: the width x anisotropy
    interaction pair (Section 18) -- ov=10nm is the only overlap that
    reliably resolves at BOTH W=20 and W=10 for this morphology, so it
    anchors the width comparison even though it isn't the single best
    Stage-N candidate.
t_target=0.5s, dense sampling around the expected stress minimum/turnover.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

SAMPLE_TIMES = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
                0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

BASE = dict(R2_nm=80.0, aspect_ratio=2.0, gamma_gb_ratio=1.4, gamma_s=1.0, M_GB_scale=30.0, M_s_scale=1.0)

CANDIDATES = [
    dict(case_id="L_G2_ov5_W20_AN0", A_nm=120.0, lambda_nm=320.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
         use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0, aniso_label="AN0"),
    dict(case_id="L_G2_ov5_W20_AN1", A_nm=120.0, lambda_nm=320.0, overlap_nm=5.0, dx_nm=2.5, W_nm=20.0,
         use_aniso_surface=True, aniso_delta=0.02, theta_mis_deg=0.0, aniso_label="AN1"),
    dict(case_id="L_G2_ov10_W20_AN0", A_nm=120.0, lambda_nm=320.0, overlap_nm=10.0, dx_nm=2.5, W_nm=20.0,
         use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0, aniso_label="AN0"),
    dict(case_id="L_G2_ov10_W10_AN0", A_nm=120.0, lambda_nm=320.0, overlap_nm=10.0, dx_nm=1.25, W_nm=10.0,
         use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0, aniso_label="AN0"),
]


SAMPLE_TIMES_W10 = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25]


def build_cases():
    cases = []
    for c in CANDIDATES:
        # W=10nm/dx=1.25nm is ~25x slower per unit time than W=20nm/dx=2.5nm
        # (Section 3's ladder: t=0.15s took ~1800s at W=10 vs ~76s at W=20).
        # A full t=0.5s W=10 run would cost ~1.7 hours for one case; capped
        # at t=0.25s instead (still well past the reset/turnover point seen
        # in Section 3's width ladder at the same W), given the width ladder
        # already showed W barely changes sigma/L_contact at t=0.15s -- the
        # marginal information from running W=10 as long as the W=20 cases
        # does not justify the cost.
        if c["W_nm"] == 10.0:
            cases.append(dict(BASE, stage="L", t_target=0.25, sample_times=SAMPLE_TIMES_W10, **c))
        else:
            cases.append(dict(BASE, stage="L", t_target=0.5, sample_times=SAMPLE_TIMES, **c))
    return cases


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage L (long trajectories): {len(cases)} cases")
    for c in cases:
        print(f"  {c['case_id']}")
    run_cases(cases)
