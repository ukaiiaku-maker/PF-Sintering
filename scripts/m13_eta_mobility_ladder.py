"""Milestone 13 Section 13: eta-mobility mechanism ladder.

From the SAME initial state (State A, dx fixed), run the full unified
transport (conserved-f surface diffusion + the Section 10 tangent-cone
constrained eta update) at three M_eta scales:

    eta_ladder_scale = 0    (identical to the Section 3 frozen-eta baseline
                              -- a consistency cross-check, not re-derived)
    eta_ladder_scale = 0.1
    eta_ladder_scale = 1    (full production M_eta)

This is explicitly a MECHANISM-SEPARATION study (Section 13: "not a
parameter tuning" exercise) -- M_eta is scaled via dataclasses.replace
AFTER build_params (keeping eta_diffusivity_fixed_physical=True's
dx-independent base M_eta, so the ladder factor itself stays dx-
independent too), not by changing eta_mobility_scale in ModelConfig
(which build_params ignores whenever eta_diffusivity_fixed_physical=True,
the convention every M12B/M13 fixed-physics campaign uses).

At every sample, dV2/dt is decomposed into conserved-f transport vs.
eta-ownership-migration contributions (flux_closure.
particle_volume_rate_decomposition) -- Section 13's explicit requirement,
distinct from just tracking V2(t) alone -- alongside dM_neck_f/dt.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from pf_sintering.flux_closure import particle_volume_rate_decomposition
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_state_a(args, dx_nm):
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def run_one_ladder_rung(dx_nm, args, eta_ladder_scale, t_end, sample_every_steps):
    print(f"\n=== eta ladder rung: scale={eta_ladder_scale}, dx={dx_nm}nm ===")
    p0, f, e1, e2, e3 = build_state_a(args, dx_nm)
    p = dataclasses.replace(p0, M_eta=p0.M_eta * eta_ladder_scale)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    n_steps = round(t_end / p.dt)

    sub0 = compute_subgrid_contact(f, e1, e2, p)
    neck_mask = neck_region_mask(p, (sub0.top.x_sub, sub0.top.y_sub), (sub0.bottom.x_sub, sub0.bottom.y_sub)) \
        if sub0.resolved else np.zeros_like(f, dtype=bool)

    V1_0, V2_0, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    M_neck_0 = float(np.sum(f * neck_mask)) * p.dx * p.dx
    rows = [dict(step=0, time_s=0.0, V1=V1_0, V2=V2_0, M_neck_f=M_neck_0,
                  dV2_dt_transport=0.0, dV2_dt_eta_migration=0.0, dV2_dt_total=0.0)]
    print(f"  step=0 V2={V2_0:.8e} M_neck_f={M_neck_0:.6e}")

    cum_transport = 0.0
    cum_eta_migration = 0.0
    for step in range(1, n_steps + 1):
        f_old, e1_old, e2_old, e3_old = f, e1, e2, e3
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
        e1, e2, e3, ediag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, bc_x=BC_X, bc_y=BC_Y)

        r = particle_volume_rate_decomposition(f_old, f, e1_old, e2_old, e3_old, e1, e2, e3, p.dt, p.dx)
        cum_transport += r["dV2_dt_transport"] * p.dt
        cum_eta_migration += r["dV2_dt_eta_migration"] * p.dt

        if step % sample_every_steps == 0 or step == n_steps:
            V1, V2, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
            M_neck = float(np.sum(f * neck_mask)) * p.dx * p.dx
            rows.append(dict(step=step, time_s=step * p.dt, V1=V1, V2=V2, M_neck_f=M_neck,
                              dV2_dt_transport=r["dV2_dt_transport"], dV2_dt_eta_migration=r["dV2_dt_eta_migration"],
                              dV2_dt_total=r["dV2_dt_total"],
                              cum_transport=cum_transport, cum_eta_migration=cum_eta_migration))
            if step % (sample_every_steps * 10) == 0 or step == n_steps:
                print(f"  step={step:6d} t={step*p.dt:.4e}s V2={V2:.8e} M_neck_f={M_neck:.6e} "
                      f"dV2/dt_transport={r['dV2_dt_transport']:.4e} dV2/dt_eta={r['dV2_dt_eta_migration']:.4e} "
                      f"cum_transport={cum_transport:.4e} cum_eta_migration={cum_eta_migration:.4e}")

    return dict(dx_nm=dx_nm, eta_ladder_scale=eta_ladder_scale, M_eta=p.M_eta, p_dt=p.dt, rows=rows)


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
    ap.add_argument("--t-end", type=float, default=0.03)
    ap.add_argument("--sample-every-steps", type=int, default=20)
    ap.add_argument("--rungs", type=str, default="0.0,0.1,1.0")
    ap.add_argument("--out", type=str, default="runs/m13_eta_mobility_ladder.json")
    args = ap.parse_args()

    rungs = [float(x) for x in args.rungs.split(",")]
    out = dict(args=vars(args), rungs={})
    for scale in rungs:
        result = run_one_ladder_rung(args.dx_nm, args, scale, args.t_end, args.sample_every_steps)
        out["rungs"][str(scale)] = result

    print("\n=== Ladder summary ===")
    for scale in rungs:
        rows = out["rungs"][str(scale)]["rows"]
        V2_0, V2_end = rows[0]["V2"], rows[-1]["V2"]
        Mn_0, Mn_end = rows[0]["M_neck_f"], rows[-1]["M_neck_f"]
        print(f"  scale={scale}: V2 {V2_0:.6e} -> {V2_end:.6e} (d={V2_end-V2_0:+.4e})  "
              f"M_neck_f {Mn_0:.6e} -> {Mn_end:.6e} (d={Mn_end-Mn_0:+.4e})")

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
