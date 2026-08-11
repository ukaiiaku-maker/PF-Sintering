"""Milestone 15I Sections 9-10: corrected-units control-volume mass
balance + shell-integrated -div(J) map for the C2 "slow both" narrowing
trajectories (R2_C, R3_C), at checkpoints spanning before/during/after
the narrowing window.

UNITS CORRECTION (Section 9): `integral f dA` in this 2-D model has
units of m^2 (per unit out-of-plane depth) -- NOT kg or "kg-equivalent"
as M15H's report mislabeled it. The underlying computation
(m15h_mechanism_lib.control_volume_mass_balance) was always correct
(sum(f)*dx^2, a pure area integral); only the report's units label was
wrong. Reported here with the correct label.

Section 10: `control_volume_mass_balance`'s `predicted_delta_from_div`
IS the shell-integrated 2-D -div(J) map (bc_ops.flux_divergence summed
over the shell mask) -- already reconciled against the direct
Delta(sum f) to <0.1% relative error in M15H; reconfirmed here on the
NEW C2 states."""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y, build_state  # noqa: E402
from m15f_campaign_lib import M_GB_REF  # noqa: E402
from m15g_mechanism_lib import _mu_field, trial_evolve  # noqa: E402
from m15h_mechanism_lib import control_volume_mass_balance  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402
from pf_sintering.tj_force import locate_neck_tjs  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15i_campaign")
N_STEPS_MB = 5
CHECKPOINTS = [0.01, 0.05, 0.1, 0.15]

MORPH = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0,
             gamma_gb=1.4, dx_nm=2.5, W_nm=20.0, use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002,
             theta_mis_deg=45.0)

FAMILIES = dict(shell_R2_C=(15.0, 0.25), shell_R3_C=(15.0, 1.0 / 6.0))


def run_family(label, M_GB_scale, M_s_scale):
    out_path = os.path.join(CAMPAIGN_DIR, f"{label}.json")
    if os.path.exists(out_path):
        print(f"[{label}] already done, skipping")
        return
    p, f, e1, e2, e3 = build_state(M_GB=M_GB_scale * M_GB_REF, surface_mobility_scale=0.3 * M_s_scale, **MORPH)
    s = Sink(threshold=math.inf)
    dt = p.dt
    step = 0
    out = []
    for t_ckpt in CHECKPOINTS:
        n_target = round(t_ckpt / dt)
        while step < n_target:
            f, e1, e2, e3 = trial_evolve(f, e1, e2, e3, s, p, "FULL", 1, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        t = step * dt
        tjs = locate_neck_tjs(f, e1, e2, p)
        if tjs is None:
            print(f"[{label}] t={t:.4f}: TJs not resolved")
            continue
        mu_before = _mu_field(f, e1, e2, e3, s, p, BC_X, BC_Y)
        f1, e1_1, e2_1, e3_1 = trial_evolve(f, e1, e2, e3, s, p, "SURF", N_STEPS_MB, bc_x=BC_X, bc_y=BC_Y)
        mb_top = control_volume_mass_balance(f, f1, mu_before, p, tjs["tj_top"], N_STEPS_MB, dt)
        print(f"[{label}] t={t:.4f} shell mass balance (units: m^2, per unit out-of-plane depth):")
        for k, v in mb_top.items():
            if v is None:
                continue
            print(f"    {k}: delta_area={v['delta_mass']:.4e} m^2  "
                  f"predicted_from_div(J)={v['predicted_delta_from_div']:.4e} m^2  "
                  f"rel_check={v['rel_check']:.2e}")
        out.append(dict(checkpoint_t=t, shell_balance_top=mb_top))
    result = dict(label=label, M_GB_scale=M_GB_scale, M_s_scale=M_s_scale, data=out)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{label}] saved to {out_path}")


if __name__ == "__main__":
    for label, (M_GB_scale, M_s_scale) in FAMILIES.items():
        run_family(label, M_GB_scale, M_s_scale)
