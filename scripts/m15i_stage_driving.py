"""Milestone 15I Sections 6-7: D_eta(t), D_surf(t), chi(t), L_contact(t)
for each matched-R family member, evolved forward with its OWN
(M_GB_scale, M_s_scale) -- direct test of the GB-reservoir-exhaustion
hypothesis from M15H (does D_eta collapse earlier in PHYSICAL time when
M_GB is pushed higher at fixed M_s? does lowering M_s at fixed baseline
M_GB let D_eta stay active longer while suppressing the opposing SURF
term?).
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y, build_state, sample_state  # noqa: E402
from m15f_campaign_lib import M_GB_REF  # noqa: E402
from m15g_mechanism_lib import operator_audit, trial_evolve  # noqa: E402
from m15i_driving_lib import D_eta, D_surf  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15i_campaign")
CHECKPOINTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

MORPH = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0,
             gamma_gb=1.4, dx_nm=2.5, W_nm=20.0, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
             theta_mis_deg=45.0)

FAMILIES = {
    "R2_A_MGB60_Ms1": (60.0, 1.0),
    "R2_B_MGB30_Ms0.5": (30.0, 0.5),
    "R2_C_MGB15_Ms0.25": (15.0, 0.25),
    "R3_A_MGB90_Ms1": (90.0, 1.0),
    "R3_B_MGB30_Ms0.333": (30.0, 1.0 / 3.0),
    "R3_C_MGB15_Ms0.167": (15.0, 1.0 / 6.0),
}


def run_family(label, M_GB_scale, M_s_scale):
    out_path = os.path.join(CAMPAIGN_DIR, f"drv_{label}.json")
    if os.path.exists(out_path):
        print(f"[{label}] already done, skipping")
        return

    p, f, e1, e2, e3 = build_state(M_GB=M_GB_scale * M_GB_REF, surface_mobility_scale=0.3 * M_s_scale, **MORPH)
    s = Sink(threshold=math.inf)
    dt = p.dt
    step = 0
    rows = []
    for t_ckpt in CHECKPOINTS:
        n_target = round(t_ckpt / dt)
        while step < n_target:
            f, e1, e2, e3 = trial_evolve(f, e1, e2, e3, s, p, "FULL", 1, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        t = step * dt
        de, _ = D_eta(f, e1, e2, e3, s, p, n_steps=1)
        ds, _ = D_surf(f, e1, e2, e3, s, p)
        audit = operator_audit(f, e1, e2, e3, s, p, f"{label}_t{t_ckpt:g}", n_steps=5, verbose=False)
        m = audit["modes"]
        chi = (abs(m["GB"]["Delta_L_contact"]) / m["SURF"]["Delta_L_contact"]
               if m["SURF"]["Delta_L_contact"] not in (None, 0) else float("nan"))
        state = sample_state(f, e1, e2, e3, s, p, step, t, float(f.sum()) * p.dx * p.dx, 0.0, 0.0, 0.0)
        row = dict(t=t, D_eta=de, D_surf=ds, chi=chi, L_contact=state.get("L_contact"),
                   SURF_dL_dt=(m["SURF"]["Delta_L_contact"] / (5 * dt)) if m["SURF"]["Delta_L_contact"] else None,
                   GB_dL_dt=(m["GB"]["Delta_L_contact"] / (5 * dt)) if m["GB"]["Delta_L_contact"] else None,
                   FULL_dL_dt=(m["FULL"]["Delta_L_contact"] / (5 * dt)) if m["FULL"]["Delta_L_contact"] else None)
        rows.append(row)
        print(f"[{label}] t={t:.4f} D_eta={de:.4e} D_surf={ds:.4e} chi={chi:.4f} "
              f"L_contact={row['L_contact']*1e9 if row['L_contact'] else float('nan'):.3f}nm")

    result = dict(label=label, M_GB_scale=M_GB_scale, M_s_scale=M_s_scale, rows=rows)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{label}] saved to {out_path}")


if __name__ == "__main__":
    for label, (M_GB_scale, M_s_scale) in FAMILIES.items():
        run_family(label, M_GB_scale, M_s_scale)
