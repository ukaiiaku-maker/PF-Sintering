"""Milestone 15H Sections 9-10: conservative control-volume surface mass
balance, and the mu(s)/J_t(s)/-dJ_t/ds driving map.

For the SAME canonical checkpoints (AN0, theta=45deg/AN3 at t=0.01,
0.02, 0.10s), reports:
  - Section 9: control_volume_mass_balance over 4 concentric Euclidean-
    distance shells [0,1W),[1W,2W),[2W,3W),[3W,4W) around each TJ, for a
    SURF-only trial (n_steps=5, in the converged small-Delta_t regime
    from Section 3), cross-checked against the exact flux-divergence
    primitive;
  - Section 10: the full mu(s)/J_tangent(s)/dJ_tangent_ds(s) profile
    from each TJ out to 5W (branch_mu_J_profile, already anisotropy-
    aware via m15g_mechanism_lib._mu_field).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y, build_state  # noqa: E402
from m15f_campaign_lib import BASELINE_MS_SCALE, M_GB_REF  # noqa: E402
from m15g_mechanism_lib import _mu_field, trial_evolve  # noqa: E402
from m15h_mechanism_lib import control_volume_mass_balance  # noqa: E402
from pf_sintering.curvature_extraction import branch_mu_J_profile  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, surface_flux_face_projected  # noqa: E402
from pf_sintering.tj_force import locate_neck_tjs  # noqa: E402
from m15_gb_surface_rate_competition import sample_state  # noqa: E402
from m12b_grid_convergence import classify_branches  # noqa: E402
from pf_sintering.tj_force import compute_tj_force  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15h_campaign")
CHECKPOINTS = [0.01, 0.02, 0.10]
N_STEPS_MB = 5

CONFIGS = [
    dict(label="mb_AN0", use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0),
    dict(label="mb_th45_AN3", use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002, theta_mis_deg=45.0),
]

MORPH = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0,
             gamma_gb=1.4, M_GB=30.0 * M_GB_REF, surface_mobility_scale=BASELINE_MS_SCALE * 1.0, dx_nm=2.5,
             W_nm=20.0)


def branch_dirs(f, e1, e2, e3, s, p, tj_top, tj_bottom):
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx
    top = compute_tj_force(f, e1, e2, tj_top, s, p)
    bottom = compute_tj_force(f, e1, e2, tj_bottom, s, p)
    tpd, _tsd = classify_branches(f, p, tj_top, top.v_s1, top.v_s2, wall_mean)
    bpd, _bsd = classify_branches(f, p, tj_bottom, bottom.v_s1, bottom.v_s2, wall_mean)
    return tpd, bpd


def run_config(cfg):
    out_path = os.path.join(CAMPAIGN_DIR, f"{cfg['label']}.json")
    if os.path.exists(out_path):
        print(f"[{cfg['label']}] already done, skipping")
        return

    p, f, e1, e2, e3 = build_state(**MORPH, use_aniso_surface=cfg["use_aniso_surface"],
                                    aniso_delta=cfg["aniso_delta"], theta_mis_deg=cfg["theta_mis_deg"])
    s = Sink(threshold=math.inf)
    dt = p.dt
    W = p.interface_width
    out = []
    step = 0
    for t_ckpt in CHECKPOINTS:
        n_target = round(t_ckpt / dt)
        while step < n_target:
            f, e1, e2, e3 = trial_evolve(f, e1, e2, e3, s, p, "FULL", 1, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        t = step * dt
        tjs = locate_neck_tjs(f, e1, e2, p)
        if tjs is None:
            print(f"[{cfg['label']}] t={t:.4f}: TJs not resolved, skipping")
            continue
        tj_top, tj_bottom = tjs["tj_top"], tjs["tj_bottom"]
        top_dir, bot_dir = branch_dirs(f, e1, e2, e3, s, p, tj_top, tj_bottom)

        # Section 9: mass balance from a SURF-only trial
        mu_before = _mu_field(f, e1, e2, e3, s, p, BC_X, BC_Y)
        f1, e1_1, e2_1, e3_1 = trial_evolve(f, e1, e2, e3, s, p, "SURF", N_STEPS_MB, bc_x=BC_X, bc_y=BC_Y)
        mb_top = control_volume_mass_balance(f, f1, mu_before, p, tj_top, N_STEPS_MB, dt)
        mb_bot = control_volume_mass_balance(f, f1, mu_before, p, tj_bottom, N_STEPS_MB, dt)
        print(f"[{cfg['label']}] t={t:.4f} mass balance (top TJ, SURF-only, n={N_STEPS_MB}):")
        for k, v in mb_top.items():
            print(f"    {k}: {v}")

        # Section 10: mu(s)/J_tangent(s)/dJ_tangent_ds(s) map to 5W
        M_s = m_s_ref(p.M_f, p.interface_width)
        fp = surface_flux_face_projected(f, mu_before, p.dx, W, M_s, BC_X, BC_Y)
        # cell-centered from face fluxes, matching m15_gb_surface_rate_competition.profile_mu_J exactly
        Jx = 0.5 * (fp["Jx_face"] + np.roll(fp["Jx_face"], 1, axis=1))
        Jy = np.zeros_like(Jx)
        Jy[1:] = 0.5 * (fp["Jy_face"][1:] + fp["Jy_face"][:-1])
        prof_top = branch_mu_J_profile(f, mu_before, Jx, Jy, p, tj_top, top_dir, max_arclength=5.0 * W)
        prof_bot = branch_mu_J_profile(f, mu_before, Jx, Jy, p, tj_bottom, bot_dir, max_arclength=5.0 * W)
        print(f"[{cfg['label']}] t={t:.4f} mu/J profile (top TJ), s in [0,5W]:")
        if prof_top:
            for i in range(0, len(prof_top["s"]), 5):
                print(f"    s={prof_top['s'][i]*1e9:.2f}nm ({prof_top['s'][i]/W:.2f}W) "
                      f"mu={prof_top['mu'][i]:.3e} J_t={prof_top['J_tangent'][i]:.3e} "
                      f"-dJt/ds={-prof_top['dJ_tangent_ds'][i]:.3e}")

        out.append(dict(checkpoint_t=t, mass_balance_top=mb_top, mass_balance_bottom=mb_bot,
                         mu_J_profile_top=prof_top, mu_J_profile_bottom=prof_bot))

    result = dict(config=cfg, morph=MORPH, data=out)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{cfg['label']}] saved to {out_path}")


if __name__ == "__main__":
    for cfg in CONFIGS:
        run_config(cfg)
