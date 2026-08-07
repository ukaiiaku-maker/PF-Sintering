#!/usr/bin/env python3
"""Constrained relaxation at fixed (V2, L): G*(V2, L), F_L = -dG*/dL, and the
V2 sweep at fixed L (Sections per the force-balance-conjugate-to-L handoff).

Every relaxation in this script uses `constrained_relaxation.relax_at_fixed_V2_L`
(CH + structural relaxation only; no Ostwald, no hazard, no RBM; V2 and L held
via the explicit corrector in separation_constraint.py). x_neck and psi are
never constrained -- they are read off the converged state.

Given the intrinsic (system_size/reference_length)^4 stiffness of explicit
CH time-stepping, full asymptotic convergence to the TJ force-balance
equilibrium was found (see accompanying report) to require a step budget
beyond what is practical in this session; this script uses a large but fixed
practical step budget for every state so comparisons across V2/L are made on
equal footing, and reports the residual TJ force explicitly rather than
assuming it has vanished.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pf_sintering.constrained_relaxation import relax_at_fixed_V2_L, summarize_constrained_state
from pf_sintering.diagnostics import wall_x0
from pf_sintering.model import ModelConfig, Sink, build_params, center
from pf_sintering.separation_constraint import enforce_separation
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving


def base_params(args):
    return build_params(ModelConfig(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny, dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        t_total=1e-6,
    ))


def relax_and_summarize(p, s, v2_target, L_target, n_steps, init=None, x_neck_seed=None,
                         check_every=20, converge_window=50, converge_rtol=1e-9):
    res = relax_at_fixed_V2_L(
        p, s, v2_target, L_target, init=init, x_neck_seed=x_neck_seed,
        n_steps=n_steps, check_every=check_every, converge_window=converge_window,
        converge_rtol=converge_rtol,
    )
    summ = summarize_constrained_state(res.f, res.e1, res.e2, res.e3, p, s)
    summ["steps_run"] = res.steps_run
    summ["converged_flag"] = res.converged
    summ["dL_final"] = res.dL_final
    summ["dV2_final"] = res.dV2_final
    summ["G_history_head"] = res.G_history[:3]
    summ["G_history_tail"] = res.G_history[-3:]
    return res, summ


def warm_start_translate(res, p, dL, L_new):
    """Translate a converged state by dL (via the same corrector used during
    relaxation) to seed relaxation at a nearby L."""
    w0 = wall_x0(p)
    f, e1, e2, e3 = res.f.copy(), res.e1.copy(), res.e2.copy(), res.e3.copy()
    pre_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f, e1, e2, e3, _ = enforce_separation(f, e1, e2, e3, p, w0, L_new, tol=1e-14)
    e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=pre_targets)
    return f, e1, e2, e3


def warm_start_rescale_V2(res, p, v2_target_new):
    """Adjust a converged state's e2 mass to a new V2 target (e1 unchanged),
    used to seed the V2 sweep from the previous, larger-V2 converged state."""
    f, e1, e2, e3 = res.f.copy(), res.e1.copy(), res.e2.copy(), res.e3.copy()
    v1_now, v2_now, v3_now = eta_masses(e1, e2, e3, p.use_eta3)
    v2_target_raw = v2_target_new / (p.dx * p.dx)
    e1, e2, e3 = project_eta_mass_preserving(
        f, e1, e2, e3, p, target_masses=(v1_now, v2_target_raw, v3_now),
    )
    return f, e1, e2, e3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--L0-nm", type=float, default=100.0)
    ap.add_argument("--x-neck-seed-nm", type=float, default=30.0)
    ap.add_argument("--n-steps", type=int, default=60000)
    ap.add_argument("--n-steps-warm", type=int, default=30000)
    ap.add_argument("--dL-nm", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--v2-fracs", type=float, nargs="+", default=[1.000, 0.995, 0.990, 0.980])
    ap.add_argument("--stationarity-seed2-nm", type=float, default=45.0)
    ap.add_argument("--out", type=Path, default=Path("runs/constrained_G_star_benchmark.json"))
    args = ap.parse_args()

    p = base_params(args)
    s = Sink(threshold=1.0)
    v20 = math.pi * (args.r2_nm * 1e-9) ** 2
    L0 = args.L0_nm * 1e-9
    result = dict(args=vars(args) | dict(out=str(args.out)), v20=v20)

    print(f"=== single well-resolved state @ L0={args.L0_nm}nm, V2=V20={v20:.4e}, "
          f"seed x_neck={args.x_neck_seed_nm}nm, n_steps={args.n_steps} ===")
    res0, summ0 = relax_and_summarize(p, s, v20, L0, args.n_steps, x_neck_seed=args.x_neck_seed_nm * 1e-9)
    print(json.dumps({k: v for k, v in summ0.items() if k not in ("G_history_head", "G_history_tail")}, indent=2, default=str))
    result["primary_state"] = summ0

    print(f"\n=== stationarity check: second seed x_neck={args.stationarity_seed2_nm}nm ===")
    res0b, summ0b = relax_and_summarize(p, s, v20, L0, args.n_steps, x_neck_seed=args.stationarity_seed2_nm * 1e-9)
    print(json.dumps({k: v for k, v in summ0b.items() if k not in ("G_history_head", "G_history_tail")}, indent=2, default=str))
    result["stationarity_check_state"] = summ0b
    print(f"\nG* seed1={summ0['G_interface_J']:.6e}  G* seed2={summ0b['G_interface_J']:.6e}  "
          f"relative diff={(summ0['G_interface_J']-summ0b['G_interface_J'])/summ0['G_interface_J']:+.3e}")
    print(f"x_neck* seed1={summ0['x_neck_m']*1e9:.3f}nm  seed2={summ0b['x_neck_m']*1e9:.3f}nm")

    print("\n=== F_L via centered finite difference ===")
    fl_rows = []
    for dL_nm in args.dL_nm:
        dL = dL_nm * 1e-9
        f_p, e1_p, e2_p, e3_p = warm_start_translate(res0, p, dL, L0 + dL)
        _, summ_p = relax_and_summarize(p, s, v20, L0 + dL, args.n_steps_warm, init=(f_p, e1_p, e2_p, e3_p))
        f_m, e1_m, e2_m, e3_m = warm_start_translate(res0, p, -dL, L0 - dL)
        _, summ_m = relax_and_summarize(p, s, v20, L0 - dL, args.n_steps_warm, init=(f_m, e1_m, e2_m, e3_m))
        F_L = -(summ_p["G_interface_J"] - summ_m["G_interface_J"]) / (2 * dL)
        print(f"  dL={dL_nm}nm  G*(L0+dL)={summ_p['G_interface_J']:.6e}  "
              f"G*(L0-dL)={summ_m['G_interface_J']:.6e}  F_L={F_L:.6e}")
        fl_rows.append(dict(dL_nm=dL_nm, G_plus=summ_p["G_interface_J"], G_minus=summ_m["G_interface_J"],
                             F_L=F_L, state_plus=summ_p, state_minus=summ_m))
    result["F_L_finite_difference"] = fl_rows

    print("\n=== V2 sweep at fixed L0 ===")
    v2_rows = []
    prev_res = res0
    for vf in args.v2_fracs:
        v2t = vf * v20
        if vf == 1.000:
            summ_v = summ0
            F_L_here = fl_rows[0]["F_L"] if fl_rows else math.nan
        else:
            fw, e1w, e2w, e3w = warm_start_rescale_V2(prev_res, p, v2t)
            res_v, summ_v = relax_and_summarize(p, s, v2t, L0, args.n_steps_warm, init=(fw, e1w, e2w, e3w))
            prev_res = res_v
            # F_L at this V2 via the same finite-difference recipe, smallest dL only (cheaper).
            dL = args.dL_nm[0] * 1e-9
            f_p, e1_p, e2_p, e3_p = warm_start_translate(res_v, p, dL, L0 + dL)
            _, s_p = relax_and_summarize(p, s, v2t, L0 + dL, args.n_steps_warm, init=(f_p, e1_p, e2_p, e3_p))
            f_m, e1_m, e2_m, e3_m = warm_start_translate(res_v, p, -dL, L0 - dL)
            _, s_m = relax_and_summarize(p, s, v2t, L0 - dL, args.n_steps_warm, init=(f_m, e1_m, e2_m, e3_m))
            F_L_here = -(s_p["G_interface_J"] - s_m["G_interface_J"]) / (2 * dL)
        row = dict(V2_over_V20=vf, V2=summ_v["V2"], G_star=summ_v["G_interface_J"],
                   x_neck_star_nm=summ_v["x_neck_m"] * 1e9, A_GB_star_m2=summ_v["A_GB_m2"],
                   psi_top_deg=summ_v["psi_top_deg"], psi_bottom_deg=summ_v["psi_bottom_deg"],
                   F_L=F_L_here, F_TJ_top_mag=summ_v["F_TJ_top_mag"], F_TJ_bottom_mag=summ_v["F_TJ_bottom_mag"])
        print("  " + json.dumps(row, default=str))
        v2_rows.append(row)
    result["V2_sweep"] = v2_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
