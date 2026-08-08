"""Milestone 13B Section 10: active_tol sensitivity for the tangent-cone
constrained eta integrator.

Runs the SAME State-A sinusoidal-substrate contact, full production M_eta
(eta-active, not frozen), dx=2.5nm, for a short common trajectory at
active_tol = 1e-5, 1e-4, 1e-3, and compares V2(t), M_neck_f(t), and TJ
position across the three runs. If they agree to within normal
grid/timestep discretization tolerance, active_tol is confirmed to be a
purely numerical active-set tolerance, not a hidden physical parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from unified_sinusoidal_campaign import BC_X, BC_Y, build_config  # noqa: E402

from pf_sintering.ch_crossover_diagnostics import neck_region_mask
from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update, f_weighted_ownership_volumes
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_state_a(args, dx_nm):
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def run_one(dx_nm, args, active_tol, t_end, sample_every_steps):
    p, f, e1, e2, e3 = build_state_a(args, dx_nm)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    n_steps = round(t_end / p.dt)

    sub0 = compute_subgrid_contact(f, e1, e2, p)
    neck_mask = neck_region_mask(p, (sub0.top.x_sub, sub0.top.y_sub), (sub0.bottom.x_sub, sub0.bottom.y_sub)) \
        if sub0.resolved else np.zeros_like(f, dtype=bool)

    rows = []
    V1_0, V2_0, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    M_neck_0 = float(np.sum(f * neck_mask)) * p.dx * p.dx
    rows.append(dict(step=0, time_s=0.0, V2=V2_0, M_neck_f=M_neck_0,
                      tj_top=(float(sub0.top.x_sub), float(sub0.top.y_sub)) if sub0.resolved else None,
                      safety_fraction=0.0))

    cum_safety = 0.0
    cum_var = 0.0
    for step in range(1, n_steps + 1):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
        e1, e2, e3, ediag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, bc_x=BC_X, bc_y=BC_Y,
                                                                  active_tol=active_tol)
        cum_safety += ediag["safety_correction"]
        cum_var += ediag["variational_change"]

        if step % sample_every_steps == 0 or step == n_steps:
            V1, V2, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
            M_neck = float(np.sum(f * neck_mask)) * p.dx * p.dx
            sub = compute_subgrid_contact(f, e1, e2, p)
            rows.append(dict(step=step, time_s=step * p.dt, V2=V2, M_neck_f=M_neck,
                              tj_top=(float(sub.top.x_sub), float(sub.top.y_sub)) if sub.resolved else None,
                              safety_fraction=cum_safety / (cum_var + 1e-300)))

    return dict(active_tol=active_tol, rows=rows, cum_safety=cum_safety, cum_var=cum_var)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--t-end", type=float, default=0.02)
    ap.add_argument("--sample-every-steps", type=int, default=200)
    ap.add_argument("--tolerances", type=str, default="1e-5,1e-4,1e-3")
    ap.add_argument("--out", type=str, default="runs/m13b_active_tol_sensitivity.json")
    args = ap.parse_args()

    tols = [float(x) for x in args.tolerances.split(",")]
    out = dict(args=vars(args), runs={})
    for tol in tols:
        print(f"\n=== active_tol={tol} ===")
        result = run_one(args.dx_nm, args, tol, args.t_end, args.sample_every_steps)
        out["runs"][str(tol)] = result
        last = result["rows"][-1]
        print(f"  V2(end)={last['V2']:.10e}  M_neck_f(end)={last['M_neck_f']:.10e}  "
              f"tj_top(end)={last['tj_top']}  cumulative safety_fraction={result['cum_safety']/(result['cum_var']+1e-300):.4e}")

    print("\n=== Cross-tolerance comparison (end state) ===")
    base_tol = tols[0]
    base_row = out["runs"][str(base_tol)]["rows"][-1]
    for tol in tols:
        row = out["runs"][str(tol)]["rows"][-1]
        dV2 = abs(row["V2"] - base_row["V2"]) / abs(base_row["V2"])
        dMneck = abs(row["M_neck_f"] - base_row["M_neck_f"]) / abs(base_row["M_neck_f"])
        print(f"  tol={tol}: rel|dV2 vs tol={base_tol}|={dV2:.4e}  rel|dM_neck vs tol={base_tol}|={dMneck:.4e}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
