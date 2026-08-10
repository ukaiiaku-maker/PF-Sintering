"""Milestone 15G Section 2/5/6/7: orientation screen.

M15F tested only theta_mis_deg=0 (orientation "A"), and found the
TJ-adjacent particle surface sits ~40-45deg from the anisotropy's
low-energy axis throughout -- near the HIGH-energy orientation for
gamma(psi)=gamma_s[1-delta*cos(4*psi)]. This screen rotates theta_mis_deg
so the easy axis lines up with that same ~40-45deg normal direction
instead.

W=20nm/dx=2.5nm (Section 5: the W question is closed, use the cheap
resolution), G2 (A=120nm,lambda=320nm), overlap in {5,10}nm, theta_mis_deg
in {30,40,45,50,60}deg, AN in {AN0,AN2,AN3} (AN1 omitted per Section 3),
t_target=0.10s. gamma_GB/gamma_s=1.4, M_GB_scale=30, M_s_scale=1
(M15E/15F's winning rate-competition point).

Reuses scripts/m15f_campaign_lib.py's restartable infrastructure and
runs/m15f_campaign/ manifest directly (case_id prefix "O_" avoids any
collision with M15F's own case_ids).
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, dx_nm=2.5, W_nm=20.0,
            gamma_gb_ratio=1.4, gamma_s=1.0, M_GB_scale=30.0, M_s_scale=1.0)

AN_LEVELS = dict(AN0=(False, None), AN2=(True, 0.045), AN3=(True, 1.0 / 15 - 0.002))
THETAS = [30.0, 40.0, 45.0, 50.0, 60.0]
OVERLAPS = [5.0, 10.0]

SAMPLE_TIMES = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]


def build_cases():
    # AN0 (isotropic) does not depend on theta_mis_deg at all -- run it
    # ONCE per overlap (theta_mis_deg=0 as a nominal placeholder), not
    # once per theta, to avoid 5x redundant identical-physics runs.
    cases = []
    for overlap_nm in OVERLAPS:
        case_id = f"O_ov{overlap_nm:g}_AN0"
        cases.append(dict(BASE, case_id=case_id, stage="O", overlap_nm=overlap_nm,
                           t_target=0.10, sample_times=SAMPLE_TIMES,
                           use_aniso_surface=False, aniso_delta=None,
                           theta_mis_deg=0.0, aniso_label="AN0"))
        for theta in THETAS:
            for an_label in ("AN2", "AN3"):
                use_aniso, delta = AN_LEVELS[an_label]
                case_id = f"O_ov{overlap_nm:g}_th{theta:g}_{an_label}"
                cases.append(dict(BASE, case_id=case_id, stage="O", overlap_nm=overlap_nm,
                                   t_target=0.10, sample_times=SAMPLE_TIMES,
                                   use_aniso_surface=use_aniso, aniso_delta=delta,
                                   theta_mis_deg=theta, aniso_label=an_label))
    return cases


if __name__ == "__main__":
    cases = build_cases()
    print(f"Stage O (orientation screen): {len(cases)} cases")
    run_cases(cases)
