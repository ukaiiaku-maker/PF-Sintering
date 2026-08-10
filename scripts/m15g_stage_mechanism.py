"""Milestone 15G Sections 8-11: contact-broadening operator audit.

For isotropic AN0 and orientation-45deg AN2/AN3 (the morphology
A=120nm,lambda=320nm,overlap=5nm,gamma_GB/gamma_s=1.4,M_GB_scale=30,
M_s_scale=1, W=20nm/dx=2.5nm -- Section 5's screening point), evolves
the canonical trajectory forward with the SAME step logic
m15f_campaign_lib.run_trajectory_f uses, pausing at three checkpoint
times (chosen from the orientation screen's own short-run data,
Section 6): t=0.01s (initial post-transient), t=0.02s (stress minimum),
t=0.1s (later loading state). At each checkpoint, runs
m15g_mechanism_lib.operator_audit (SURF/GB/FULL non-committed trials,
TJ kinematics, flux budget, energy decomposition) WITHOUT altering the
canonical trajectory, then continues evolving the canonical state to
the next checkpoint.

Results are written to runs/m15g_campaign/mechanism_<label>.json
(restartable: skipped if the output file already exists).
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

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15g_campaign")

CHECKPOINTS = [0.01, 0.02, 0.10]

CONFIGS = [
    dict(label="mech_AN0", use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0),
    dict(label="mech_th45_AN2", use_aniso_surface=True, aniso_delta=0.045, theta_mis_deg=45.0),
    dict(label="mech_th45_AN3", use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002, theta_mis_deg=45.0),
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
    audits = []
    t = 0.0
    step = 0
    for t_ckpt in CHECKPOINTS:
        n_target = round(t_ckpt / dt)
        while step < n_target:
            f, e1, e2, e3 = trial_evolve(f, e1, e2, e3, s, p, "FULL", 1, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        t = step * dt
        print(f"[{cfg['label']}] reached checkpoint t={t:.4f}s (step {step}), running operator audit...")
        audit = operator_audit(f, e1, e2, e3, s, p, f"{cfg['label']}_t{t_ckpt:g}", n_steps=100, verbose=True)
        audit["checkpoint_t"] = t
        audits.append(audit)

    result = dict(config=cfg, morph=MORPH, checkpoints=CHECKPOINTS, audits=audits)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{cfg['label']}] saved to {out_path}")
    return result


if __name__ == "__main__":
    for cfg in CONFIGS:
        run_config(cfg)
