"""Milestone 8 -- fixed-physics discretization and paired operator causal audit.

DIAGNOSTIC ONLY. Produces the data behind
MILESTONE_8_FIXED_PHYSICS_PAIRED_OPERATOR_AUDIT.md:

1. true fixed-physics (fixed W, fixed eta diffusivity) paired C0/C1 grid
   comparison at dx=5nm and dx=2.5nm, dt-converged at each grid;
2. compact H1 confirmation (protected vs unprotected C1) using the paired
   sub-grid metric;
3. paired operator ledger on the primary case (operator-resolved attribution
   of delta_L_coarsening itself, not either branch's absolute trajectory);
4. CH mu/curvature/flux causal chain -- what Ostwald changes near each TJ,
   correlated with the next CH update's contribution to delta_L_coarsening;
5. O0/O1/O2 (normal/removal-only/addition-only) Ostwald mechanism-isolation
   decomposition using the qualified sub-grid physical contact metric.

Does not modify production physics.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.ch_exact_energy import exact_free_energy_isotropic
from pf_sintering.ch_flux_diagnostics import evolve_f_diagnostic, surface_flux_near_tj
from pf_sintering.differential_coarsening import (
    PAIRED_KEYS,
    full_state_sample,
    paired_delta,
    paired_delta_series,
    run_paired_trajectory,
    run_single_trajectory,
)
from pf_sintering.model import (
    ModelConfig,
    Sink,
    build_params,
    compute_stress,
    evolve_eta,
    evolve_f,
    initialize_fields,
    ostwald_substrate,
)
from pf_sintering.operator_ledger import OPERATORS
from pf_sintering.ostwald_diagnostics import ostwald_addition_only, ostwald_removal_only
from pf_sintering.paired_operator_ledger import (
    run_paired_operator_ledger_trajectory,
    summarize_paired_ledger,
)
from pf_sintering.signed_curvature import signed_curvature_at
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_config(rate, args, dx_nm=None, w_nm=None, fixed_eta=False, dt_override=None,
                  overlap_nm=None, reservoir_neck_unprotected=False, ostwald_fn_label=None):
    cfg = ModelConfig(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny,
        dx=(dx_nm if dx_nm is not None else args.dx_nm) * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=(overlap_nm if overlap_nm is not None else args.overlap_nm) * 1e-9,
        t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=rate, surface_mobility_scale=args.surface_mobility_scale,
        eta_mobility_scale=args.eta_mobility_scale, dt_override=dt_override,
        reservoir_neck_unprotected=reservoir_neck_unprotected,
    )
    if w_nm is not None:
        cfg.interface_width_override = w_nm * 1e-9
        cfg.eta_diffusivity_fixed_physical = fixed_eta
    return cfg


def _reduced(row):
    return {k: row.get(k) for k in ("step", "time_s", "V2") + PAIRED_KEYS}


# ---------------------------------------------------------------------------
# Section 6: true fixed-physics paired grid test
# ---------------------------------------------------------------------------

def fixed_physics_grid_test(args):
    print("\n=== Section 6: true fixed-physics (fixed W, fixed eta-diffusivity) paired grid test ===")
    out = {}
    for dx_nm in (5.0, 2.5):
        p_natural = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm,
                                               w_nm=args.fixed_w_nm, fixed_eta=True))
        dt_natural = p_natural.dt
        print(f"  dx={dx_nm}nm: W={p_natural.interface_width*1e9:.2f}nm dt_natural={dt_natural:.4e}s")

        f0, e1_0, e2_0, e3_0 = initialize_fields(p_natural)
        # quick dt-sensitivity: dt, dt/2, dt/4 on the paired differential quantity
        sens = {}
        for frac_name, frac in (("dt", 1.0), ("dt/2", 0.5), ("dt/4", 0.25)):
            dt_i = dt_natural * frac
            n_i = round(args.dt_bench_steps / frac)
            p0 = build_params(build_config(0.0, args, dx_nm=dx_nm, w_nm=args.fixed_w_nm,
                                            fixed_eta=True, dt_override=dt_i))
            p1 = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, w_nm=args.fixed_w_nm,
                                            fixed_eta=True, dt_override=dt_i))
            r0, _, _ = run_single_trajectory(p0, f0, e1_0, e2_0, e3_0, n_i)
            r1, _, _ = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, n_i)
            d = paired_delta(r0[-1], r1[-1])
            print(f"    dt={frac_name}: n_steps={n_i} delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm")
            sens[frac_name] = dict(dt=dt_i, n_steps=n_i, delta_L_coarsening=d["L_contact_TJ_sub"])
        chosen_frac = args.dx5_dt_frac if dx_nm == 5.0 else args.dx25_dt_frac
        dt_chosen = dt_natural * chosen_frac

        p0 = build_params(build_config(0.0, args, dx_nm=dx_nm, w_nm=args.fixed_w_nm,
                                        fixed_eta=True, dt_override=dt_chosen))
        p1 = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, w_nm=args.fixed_w_nm,
                                        fixed_eta=True, dt_override=dt_chosen))
        rows_c0, rows_c1, reason = run_paired_trajectory(
            p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps,
        )
        v20 = rows_c0[0]["V2"]
        d = paired_delta_series(rows_c0, rows_c1)[-1]
        dv2 = (rows_c1[-1]["V2"] - v20) / v20
        print(f"    chosen dt={chosen_frac}x natural ({dt_chosen:.4e}s): n_steps={len(rows_c1)-1} "
              f"dV2/V20={dv2:+.4e} delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm "
              f"delta_L_GB_geom_sub={d['L_GB_geom_sub']*1e9:+.6f}nm delta_sigma={d['sigma_Pa']/1e6:+.6f}MPa")
        out[f"{dx_nm:g}"] = dict(
            dt_natural=dt_natural, dt_chosen=dt_chosen, dt_chosen_frac=chosen_frac,
            sensitivity=sens, n_steps=len(rows_c1) - 1, dV2_over_V20=dv2, delta_final=d,
            c0_final=_reduced(rows_c0[-1]), c1_final=_reduced(rows_c1[-1]),
        )
    return out


# ---------------------------------------------------------------------------
# Section 7: compact H1 confirmation using the paired sub-grid metric
# ---------------------------------------------------------------------------

def h1_confirmation(args):
    print("\n=== Section 7: compact H1 confirmation (protected vs unprotected reservoir) ===")
    p0 = build_params(build_config(0.0, args))
    p1_protected = build_params(build_config(args.coarsening_rate_scale, args, reservoir_neck_unprotected=False))
    p1_unprotected = build_params(build_config(args.coarsening_rate_scale, args, reservoir_neck_unprotected=True))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)

    out = {}
    for label, p1 in (("protected", p1_protected), ("unprotected", p1_unprotected)):
        rows_c0, rows_c1, reason = run_paired_trajectory(
            p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps,
        )
        v20 = rows_c0[0]["V2"]
        d = paired_delta_series(rows_c0, rows_c1)[-1]
        dv2 = (rows_c1[-1]["V2"] - v20) / v20
        print(f"  {label}: n_steps={len(rows_c1)-1} dV2/V20={dv2:+.4e} "
              f"delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm")
        out[label] = dict(n_steps=len(rows_c1) - 1, dV2_over_V20=dv2, delta_final=d,
                           c0_final=_reduced(rows_c0[-1]), c1_final=_reduced(rows_c1[-1]))
    return out


# ---------------------------------------------------------------------------
# Section 8/9: paired operator ledger + P1/P2/P3 classification data
# ---------------------------------------------------------------------------

def paired_operator_ledger_run(args):
    print("\n=== Section 8/9: paired operator ledger (primary case) ===")
    p0 = build_params(build_config(0.0, args))
    p1 = build_params(build_config(args.coarsening_rate_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)
    steps, l0, l1 = run_paired_operator_ledger_trajectory(
        p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps,
    )
    summ = summarize_paired_ledger(steps)
    print(f"  n_steps={summ['n_steps']}")
    print(f"  total_observed delta_L_coarsening = {summ['total_observed']['L_contact_TJ_sub']*1e9:+.6f}nm")
    for op in OPERATORS:
        print(f"  {op:22s} d(delta_L_contact_TJ_sub) total = "
              f"{summ['totals'][op]['L_contact_TJ_sub']*1e9:+.7f}nm   "
              f"d(delta_L_GB_geom_sub) total = {summ['totals'][op]['L_GB_geom_sub']*1e9:+.7f}nm")
    closure = {k: v for k, v in summ["closure_error"].items() if v not in (0.0, None)}
    print(f"  closure errors (nonzero only): {closure}")

    # split into first-half vs second-half of the trajectory to check whether
    # CH's contribution grows relative to Ostwald's over time (P2 signature).
    n = summ["n_steps"]
    half = n // 2
    first_half_ch = sum(s.paired_operator_deltas["CH"]["L_contact_TJ_sub"] for s in steps[:half])
    second_half_ch = sum(s.paired_operator_deltas["CH"]["L_contact_TJ_sub"] for s in steps[half:n])
    first_half_ost = sum(s.paired_operator_deltas["Ostwald"]["L_contact_TJ_sub"] for s in steps[:half])
    second_half_ost = sum(s.paired_operator_deltas["Ostwald"]["L_contact_TJ_sub"] for s in steps[half:n])
    print(f"  CH contribution: first-half={first_half_ch*1e9:+.6f}nm  second-half={second_half_ch*1e9:+.6f}nm")
    print(f"  Ostwald contribution: first-half={first_half_ost*1e9:+.6f}nm  second-half={second_half_ost*1e9:+.6f}nm")

    per_step = [dict(step=s.step,
                      CH=s.paired_operator_deltas["CH"]["L_contact_TJ_sub"],
                      post_CH_projection=s.paired_operator_deltas["post_CH_projection"]["L_contact_TJ_sub"],
                      Ostwald=s.paired_operator_deltas["Ostwald"]["L_contact_TJ_sub"],
                      eta_raw=s.paired_operator_deltas["eta_raw"]["L_contact_TJ_sub"],
                      post_eta_projection=s.paired_operator_deltas["post_eta_projection"]["L_contact_TJ_sub"])
                for s in steps]
    return dict(n_steps=summ["n_steps"], totals=summ["totals"], total_observed=summ["total_observed"],
                closure_error=summ["closure_error"],
                first_half_ch=first_half_ch, second_half_ch=second_half_ch,
                first_half_ostwald=first_half_ost, second_half_ostwald=second_half_ost,
                per_step=per_step)


# ---------------------------------------------------------------------------
# Section 10: CH mu/curvature/flux causal chain
# ---------------------------------------------------------------------------

def _step_with_ch_diag(f, e1, e2, e3, s, p):
    """One production step, but returns the pre-CH state, the CH diagnostics
    (mu, Jx, Jy), and the post-CH (pre-projection) f alongside the final
    post-step state -- for Section 10's near-TJ causal-chain analysis."""
    ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f_new, diag = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    e1a, e2a, e3a = project_eta_mass_preserving(f_new, e1, e2, e3, p, target_masses=ch_targets)

    f2, e1b, e2b, e3b = ostwald_substrate(f_new, e1a, e2a, e3a, p)

    ac_targets = eta_masses(e1b, e2b, e3b, p.use_eta3)
    e1c, e2c, e3c = evolve_eta(e1b, e2b, e3b, p)
    e1d, e2d, e3d = project_eta_mass_preserving(f2, e1c, e2c, e3c, p, target_masses=ac_targets)
    return (f2, e1d, e2d, e3d), diag, f_new


def ch_causal_chain(args, n_steps=10):
    print(f"\n=== Section 10: CH mu/curvature/flux causal chain near TJs (first {n_steps} steps) ===")
    p0 = build_params(build_config(0.0, args))
    p1 = build_params(build_config(args.coarsening_rate_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)

    f_c0, e1_c0, e2_c0, e3_c0 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    f_c1, e1_c1, e2_c1, e3_c1 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s0 = Sink(threshold=math.inf)
    s1 = Sink(threshold=math.inf)

    rows = []
    for step in range(1, n_steps + 1):
        sub_c0_before = compute_subgrid_contact(f_c0, e1_c0, e2_c0, p0)
        sub_c1_before = compute_subgrid_contact(f_c1, e1_c1, e2_c1, p1)

        (f_c0n, e1_c0n, e2_c0n, e3_c0n), diag0, f_c0_postCH = _step_with_ch_diag(f_c0, e1_c0, e2_c0, e3_c0, s0, p0)
        (f_c1n, e1_c1n, e2_c1n, e3_c1n), diag1, f_c1_postCH = _step_with_ch_diag(f_c1, e1_c1, e2_c1, e3_c1, s1, p1)

        sub_c0_postCH = compute_subgrid_contact(f_c0_postCH, e1_c0, e2_c0, p0)
        sub_c1_postCH = compute_subgrid_contact(f_c1_postCH, e1_c1, e2_c1, p1)

        row = dict(step=step)
        if (sub_c0_before.resolved and sub_c1_before.resolved
                and sub_c0_postCH.resolved and sub_c1_postCH.resolved):
            dL_CH_c0 = sub_c0_postCH.L_contact_TJ_sub - sub_c0_before.L_contact_TJ_sub
            dL_CH_c1 = sub_c1_postCH.L_contact_TJ_sub - sub_c1_before.L_contact_TJ_sub
            row["d_delta_L_coarsening_CH"] = dL_CH_c1 - dL_CH_c0

            for name, tj0, tj1 in (("top", sub_c0_before.top, sub_c1_before.top),
                                    ("bottom", sub_c0_before.bottom, sub_c1_before.bottom)):
                if tj0.resolved and tj1.resolved:
                    xy0 = (tj0.x_sub, tj0.y_sub)
                    xy1 = (tj1.x_sub, tj1.y_sub)
                    from pf_sintering.tj_force import _sample_bilinear
                    mu_at_0 = float(_sample_bilinear(diag0.mu, [xy0[0]], [xy0[1]], p0)[0])
                    mu_at_1 = float(_sample_bilinear(diag1.mu, [xy1[0]], [xy1[1]], p1)[0])
                    kappa0 = signed_curvature_at(f_c0, xy0, p0)
                    kappa1 = signed_curvature_at(f_c1, xy1, p1)
                    flux0 = surface_flux_near_tj(f_c0, diag0.Jx, diag0.Jy, xy0, p0)
                    flux1 = surface_flux_near_tj(f_c1, diag1.Jx, diag1.Jy, xy1, p1)
                    row[f"delta_mu_{name}"] = mu_at_1 - mu_at_0
                    row[f"delta_kappa_{name}"] = (kappa1 - kappa0) if (math.isfinite(kappa0) and math.isfinite(kappa1)) else math.nan
                    if flux0 and flux1:
                        row[f"delta_J_tangent_{name}"] = flux1["J_tangent"] - flux0["J_tangent"]
                        row[f"delta_J_normal_{name}"] = flux1["J_normal"] - flux0["J_normal"]
                        row[f"J_tangent_{name}_c0"] = flux0["J_tangent"]
                        row[f"J_tangent_{name}_c1"] = flux1["J_tangent"]
        rows.append(row)
        print(f"  step={step}: " + ", ".join(f"{k}={v:+.4e}" for k, v in row.items() if k != "step" and isinstance(v, float)))

        f_c0, e1_c0, e2_c0, e3_c0 = f_c0n, e1_c0n, e2_c0n, e3_c0n
        f_c1, e1_c1, e2_c1, e3_c1 = f_c1n, e1_c1n, e2_c1n, e3_c1n
    return rows


# ---------------------------------------------------------------------------
# Section 11: O0/O1/O2 decomposition with the sub-grid physical metric
# ---------------------------------------------------------------------------

def o0_o1_o2_subgrid(args, n_ref):
    print(f"\n=== Section 11: O0/O1/O2 Ostwald decomposition, sub-grid metric, N_ref={n_ref} ===")
    p0 = build_params(build_config(0.0, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)
    rows_c0, _, _ = run_single_trajectory(p0, f0, e1_0, e2_0, e3_0, n_ref)

    out = {}
    variants = (("O0_normal", ostwald_substrate), ("O1_removal_only", ostwald_removal_only),
                ("O2_addition_only", ostwald_addition_only))
    for label, fn in variants:
        p1 = build_params(build_config(args.coarsening_rate_scale, args))
        rows_c1, stop, reason = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, n_ref, ostwald_fn=fn)
        d = paired_delta(rows_c0[-1], rows_c1[-1])
        v2_c1 = rows_c1[-1]["V2"]
        v20 = rows_c0[0]["V2"]
        print(f"  {label}: dV2/V20={((v2_c1-v20)/v20):+.4e} "
              f"delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm "
              f"delta_L_GB_geom_sub={d['L_GB_geom_sub']*1e9:+.6f}nm stop={stop}")
        out[label] = dict(dV2_over_V20=(v2_c1 - v20) / v20, delta_final=d, stop=stop, reason=reason)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--coarsening-rate-scale", type=float, default=3.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--eta-mobility-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--fixed-w-nm", type=float, default=20.0)
    ap.add_argument("--dt-bench-steps", type=int, default=100)
    ap.add_argument("--dx5-dt-frac", type=float, default=1.0)
    ap.add_argument("--dx25-dt-frac", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="runs/fixed_physics_paired_operator_audit.json")
    args = ap.parse_args()

    out = dict(args=vars(args))
    out["fixed_physics_grid_test"] = fixed_physics_grid_test(args)
    out["h1_confirmation"] = h1_confirmation(args)
    ledger_result = paired_operator_ledger_run(args)
    out["paired_operator_ledger"] = ledger_result
    out["ch_causal_chain"] = ch_causal_chain(args, n_steps=10)
    out["o0_o1_o2"] = o0_o1_o2_subgrid(args, n_ref=ledger_result["n_steps"])

    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
