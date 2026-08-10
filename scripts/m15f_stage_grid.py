"""Milestone 15F Section 17: grid check of the best anisotropic candidate.

Repeats the best Stage-L candidate at a finer dx while holding W,
gamma(theta)/orientation, M_s, M_GB, and geometry fixed, to verify facet
orientation, L_contact evolution, endpoint generalized force, and
A_sigma are grid-robust (not an artifact of the coarser mesh).
"""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15f_campaign_lib import run_cases  # noqa: E402

BASE = dict(R2_nm=80.0, aspect_ratio=2.0, gamma_gb_ratio=1.4, gamma_s=1.0, M_GB_scale=30.0, M_s_scale=1.0)
SAMPLE_TIMES = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
                0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def build_case(case_id, A_nm, lambda_nm, overlap_nm, W_nm, dx_nm, use_aniso_surface, aniso_delta,
               theta_mis_deg, aniso_label, t_target=0.5):
    return dict(BASE, case_id=case_id, stage="Grid", A_nm=A_nm, lambda_nm=lambda_nm, overlap_nm=overlap_nm,
                W_nm=W_nm, dx_nm=dx_nm, t_target=t_target, sample_times=SAMPLE_TIMES,
                use_aniso_surface=use_aniso_surface, aniso_delta=aniso_delta, theta_mis_deg=theta_mis_deg,
                aniso_label=aniso_label)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--A-nm", type=float, required=True)
    ap.add_argument("--lambda-nm", type=float, required=True)
    ap.add_argument("--overlap-nm", type=float, required=True)
    ap.add_argument("--W-nm", type=float, required=True)
    ap.add_argument("--dx-nm", type=float, required=True)
    ap.add_argument("--use-aniso-surface", action="store_true")
    ap.add_argument("--aniso-delta", type=float, default=None)
    ap.add_argument("--theta-mis-deg", type=float, default=0.0)
    ap.add_argument("--aniso-label", type=str, default="AN0")
    ap.add_argument("--t-target", type=float, default=0.5)
    ap.add_argument("--case-id", type=str, required=True)
    args = ap.parse_args()
    case = build_case(args.case_id, args.A_nm, args.lambda_nm, args.overlap_nm, args.W_nm, args.dx_nm,
                       args.use_aniso_surface, args.aniso_delta, args.theta_mis_deg, args.aniso_label,
                       t_target=args.t_target)
    run_cases([case])
