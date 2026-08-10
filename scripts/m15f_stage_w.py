"""Milestone 15F Sections 3+10: width ladder + isolated width x anisotropy
qualification.

Base morphology/physics point: the best trajectory found in Milestone
15E (A=120nm, lambda=320nm, overlap=15nm, gamma_GB/gamma_s=1.4,
M_GB_scale=30, M_s_scale=1.0 -- the only rate-competition point that
showed genuine post-reset upturn there).

Section 3 (width ladder): isotropic only, W=20/dx=2.5, W=10/dx=1.25,
W=7.5/dx=1.25, t_target=0.15s (short-medium horizon to see the initial
trend, matching M15E Stage A/B's screening philosophy).

Section 10 (isolated qualification): W in {20,10} x AN in {AN0,AN1,AN2}
at a short t_target=0.02s, single orientation (theta_mis_deg=0),
checking surface energy/mobility/mass/energy-descent are qualitatively
robust across the combination before the full neck screen.
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=10.0,
            gamma_gb_ratio=1.4, gamma_s=1.0, M_GB_scale=30.0, M_s_scale=1.0)
# overlap dropped from 15nm (M15E's winning value) to 10nm: at W=10/7.5nm the
# subgrid-contact Newton solver fails to converge at overlap=15nm (confirmed
# a numerical-robustness sensitivity of that solver to the specific
# W/overlap combination, not a geometric ill-posedness -- overlap=10nm
# resolves cleanly at ALL THREE width-ladder points, confirmed by direct
# probe before launching this campaign).

AN_LEVELS = dict(AN0=(False, None), AN1=(True, 0.02), AN2=(True, 0.045))

SAMPLE_TIMES_W = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15]
SAMPLE_TIMES_Q = [0, 0.002, 0.005, 0.01, 0.02]


def build_width_ladder_cases():
    cases = []
    for W_nm, dx_nm in ((20.0, 2.5), (10.0, 1.25), (7.5, 1.25)):
        case_id = f"W_W{W_nm:g}_dx{dx_nm:g}"
        cases.append(dict(BASE, case_id=case_id, stage="W", dx_nm=dx_nm, W_nm=W_nm,
                           t_target=0.15, sample_times=SAMPLE_TIMES_W,
                           use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0,
                           aniso_label="AN0"))
    return cases


def build_qualification_cases():
    cases = []
    for W_nm, dx_nm in ((20.0, 2.5), (10.0, 1.25)):
        for label, (use_aniso, delta) in AN_LEVELS.items():
            case_id = f"Q_W{W_nm:g}_{label}"
            cases.append(dict(BASE, case_id=case_id, stage="Q", dx_nm=dx_nm, W_nm=W_nm,
                               t_target=0.02, sample_times=SAMPLE_TIMES_Q,
                               use_aniso_surface=use_aniso, aniso_delta=delta, theta_mis_deg=0.0,
                               aniso_label=label))
    return cases


if __name__ == "__main__":
    print("=== Section 10: isolated width x anisotropy qualification ===")
    run_cases(build_qualification_cases())
    print("=== Section 3: width ladder (isotropic) ===")
    run_cases(build_width_ladder_cases())
