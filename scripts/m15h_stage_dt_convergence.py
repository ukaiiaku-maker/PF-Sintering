"""Milestone 15H Section 3-4: Delta_t convergence of the operator
decomposition, and the crossover number chi(t).

For AN0 and theta=45deg/AN3, at canonical checkpoints t=0.01, 0.02,
0.10s (same states m15g_stage_mechanism.py used), repeats the SURF/GB/
FULL operator audit at n_steps in {100,50,25,10,5} from the SAME saved
state, and reports Delta(L_contact)/Delta_t, Delta(sigma)/Delta_t,
Delta(F_total)/Delta_t, v_TJ, and R_L/Delta_t (the FULL-vs-SURF+GB
interaction remainder) versus Delta_t, to check whether the
instantaneous-rate decomposition converges as Delta_t->0 (Section 17's
PASS-1) rather than being an artifact of the 100-step window M15G used.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y, build_state  # noqa: E402
from m15f_campaign_lib import M_GB_REF, BASELINE_MS_SCALE  # noqa: E402
from m15g_mechanism_lib import operator_audit, trial_evolve  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15h_campaign")
CHECKPOINTS = [0.01, 0.02, 0.10]
N_STEPS_LADDER = [100, 50, 25, 10, 5]

CONFIGS = [
    dict(label="dtc_AN0", use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0),
    dict(label="dtc_th45_AN3", use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002, theta_mis_deg=45.0),
]

MORPH = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0,
             gamma_gb=1.4, M_GB=30.0 * M_GB_REF, surface_mobility_scale=BASELINE_MS_SCALE * 1.0, dx_nm=2.5,
             W_nm=20.0)


def run_config(cfg):
    out_path = os.path.join(CAMPAIGN_DIR, f"{cfg['label']}.json")
    if os.path.exists(out_path):
        print(f"[{cfg['label']}] already done, skipping")
        with open(out_path) as fh:
            return json.load(fh)

    p, f, e1, e2, e3 = build_state(**MORPH, use_aniso_surface=cfg["use_aniso_surface"],
                                    aniso_delta=cfg["aniso_delta"], theta_mis_deg=cfg["theta_mis_deg"])
    s = Sink(threshold=math.inf)
    dt = p.dt
    results = []
    t = 0.0
    step = 0
    for t_ckpt in CHECKPOINTS:
        n_target = round(t_ckpt / dt)
        while step < n_target:
            f, e1, e2, e3 = trial_evolve(f, e1, e2, e3, s, p, "FULL", 1, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        t = step * dt
        print(f"[{cfg['label']}] checkpoint t={t:.4f}s -- running n_steps ladder {N_STEPS_LADDER}")
        for n_steps in N_STEPS_LADDER:
            audit = operator_audit(f, e1, e2, e3, s, p, f"{cfg['label']}_t{t_ckpt:g}_n{n_steps}",
                                    n_steps=n_steps, verbose=False)
            Delta_t = n_steps * dt
            row = dict(checkpoint_t=t, n_steps=n_steps, Delta_t=Delta_t, modes={})
            for mode in ("SURF", "GB", "FULL"):
                m = audit["modes"][mode]
                row["modes"][mode] = dict(
                    dL_dt=(m["Delta_L_contact"] / Delta_t) if m["Delta_L_contact"] is not None else None,
                    dsigma_dt=(m["Delta_sigma"] / Delta_t) if m["Delta_sigma"] is not None else None,
                    dF_dt=m["Delta_F_total"] / Delta_t,
                    v_TJ_top=m["v_TJ_top"][:2],
                )
            if audit.get("R_coupled_L_contact") is not None:
                row["R_L_over_dt"] = audit["R_coupled_L_contact"] / Delta_t
            results.append(row)
            print(f"  n_steps={n_steps:3d} Delta_t={Delta_t:.3e}s "
                  f"SURF.dL/dt={row['modes']['SURF']['dL_dt']:.4e} "
                  f"GB.dL/dt={row['modes']['GB']['dL_dt']:.4e} "
                  f"FULL.dL/dt={row['modes']['FULL']['dL_dt']:.4e} "
                  f"R_L/dt={row.get('R_L_over_dt')}")

    result = dict(config=cfg, morph=MORPH, results=results)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{cfg['label']}] saved to {out_path}")
    return result


if __name__ == "__main__":
    for cfg in CONFIGS:
        run_config(cfg)
