"""Milestone 15H Section 5: local mobility-linearity tests.

At each canonical checkpoint (t=0.01,0.02,0.10s), for AN0 and
theta=45deg/AN3, vary ONE operator's mobility at a time (SURF: M_f
scale in {0.5,0.75,1.0,1.25,1.5} with GB OFF; GB: M_eta scale, expressed
as the equivalent absolute M_GB_scale in {15,22.5,30,37.5,45,60} i.e.
relative-to-30 multiplier in {0.5,0.75,1.0,1.25,1.5,2.0}, with SURF OFF)
using a small n_steps=10 trial (Section 3's ladder showed this is deep
in the converged small-Delta_t regime -- see that stage's own report),
and report dL_contact/dt and dsigma/dt vs mobility scale, to check
whether the response is locally linear.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y, build_state  # noqa: E402
from m15f_campaign_lib import BASELINE_MS_SCALE, M_GB_REF, sample_state_f  # noqa: E402
from m15g_mechanism_lib import trial_evolve  # noqa: E402
from m15h_mechanism_lib import trial_evolve_scaled  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15h_campaign")
CHECKPOINTS = [0.01, 0.02, 0.10]
N_STEPS = 10

M_F_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
M_ETA_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # relative to the current M_GB_scale=30 baseline

CONFIGS = [
    dict(label="lin_AN0", use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0),
    dict(label="lin_th45_AN3", use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002, theta_mis_deg=45.0),
]

MORPH = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0,
             gamma_gb=1.4, M_GB=30.0 * M_GB_REF, surface_mobility_scale=BASELINE_MS_SCALE * 1.0, dx_nm=2.5,
             W_nm=20.0)


def probe(f, e1, e2, e3, s, p, mode, scale_kind, scale, n_steps):
    kwargs = dict(M_f_scale=scale, M_eta_scale=1.0) if scale_kind == "M_f" else dict(M_f_scale=1.0, M_eta_scale=scale)
    mass0 = float(f.sum()) * p.dx * p.dx
    before = sample_state_f(f, e1, e2, e3, s, p, 0, 0.0, mass0, 0.0, 0.0, 0.0)
    f1, e1_1, e2_1, e3_1 = trial_evolve_scaled(f, e1, e2, e3, s, p, mode, n_steps, bc_x=BC_X, bc_y=BC_Y, **kwargs)
    after = sample_state_f(f1, e1_1, e2_1, e3_1, s, p, n_steps, n_steps * p.dt, mass0, 0.0, 0.0, 0.0)
    Delta_t = n_steps * p.dt
    dL = (after.get("L_contact", float("nan")) - before.get("L_contact", float("nan"))) / Delta_t
    sigma_key = "sigma_sint_app_endpoint_form_aniso" if p.use_aniso_surface else "sigma_sint_app_endpoint_form"
    dsig = (after.get(sigma_key, float("nan")) - before.get(sigma_key, float("nan"))) / Delta_t
    return dict(scale=scale, dL_dt=dL, dsigma_dt=dsig)


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
    step = 0
    for t_ckpt in CHECKPOINTS:
        n_target = round(t_ckpt / dt)
        while step < n_target:
            f, e1, e2, e3 = trial_evolve(f, e1, e2, e3, s, p, "FULL", 1, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        t = step * dt
        print(f"[{cfg['label']}] checkpoint t={t:.4f}s")
        surf_rows = [probe(f, e1, e2, e3, s, p, "SURF", "M_f", sc, N_STEPS) for sc in M_F_SCALES]
        gb_rows = [probe(f, e1, e2, e3, s, p, "GB", "M_eta", sc, N_STEPS) for sc in M_ETA_SCALES]
        for r in surf_rows:
            print(f"  SURF M_f_scale={r['scale']:.2f} dL/dt={r['dL_dt']:.4e} dsigma/dt={r['dsigma_dt']:.4e}")
        for r in gb_rows:
            print(f"  GB   M_eta_scale={r['scale']:.2f} (M_GB_scale~{30*r['scale']:.1f}) dL/dt={r['dL_dt']:.4e} "
                  f"dsigma/dt={r['dsigma_dt']:.4e}")
        results.append(dict(checkpoint_t=t, surf=surf_rows, gb=gb_rows))

    result = dict(config=cfg, morph=MORPH, results=results)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{cfg['label']}] saved to {out_path}")
    return result


if __name__ == "__main__":
    for cfg in CONFIGS:
        run_config(cfg)
