#!/usr/bin/env python3
"""Milestone 6C: continuous sub-grid TJ tracking and resolved contact evolution.

Runs, for a given geometry/rate case:

1. The v3 operator ledger (sub-grid + legacy contact metrics together).
2. CH flux (mu, J) at the initial state, related to the measured TJ velocity
   under the raw-CH stage specifically (v_TJ = dr_TJ/dt during that stage).
3. (primary case only) an extended trajectory to a larger |dV2|/V20, to see
   a smooth slope rather than a short-horizon delta.
4. (primary case only) one finer-grid (~half dx) directional check.

DIAGNOSTIC ONLY. hazard_step and rbm are never called. No production
physics is modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pf_sintering.ch_flux_diagnostics import evolve_f_diagnostic, surface_flux_near_tj
from pf_sintering.model import ModelConfig, Sink, build_params, initialize_fields
from pf_sintering.operator_ledger import OPERATORS, run_ledger_trajectory_v3, summarize_ledger_v3
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_config(overlap_nm, coarsening_rate_scale, surface_mobility_scale, eta_mobility_scale, args, dx_nm=None):
    return ModelConfig(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny,
        dx=(dx_nm if dx_nm is not None else args.dx_nm) * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=coarsening_rate_scale, surface_mobility_scale=surface_mobility_scale,
        eta_mobility_scale=eta_mobility_scale,
    )


def run_ledger_case(p, f0, e1_0, e2_0, e3_0, target_dv2_frac, max_steps):
    steps = run_ledger_trajectory_v3(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=target_dv2_frac, max_steps=max_steps)
    summ = summarize_ledger_v3(steps)
    return steps, summ


def print_ledger_table(tag, summ):
    print(f"\n=== {tag}: n_steps={summ['n_steps']}, dV2/V20={summ['dV2_over_V20']:+.4e} ===")
    tot = summ["total_observed"]
    print(f"  TOTAL: x_neck_eta_nm={tot['x_neck_m']*1e9:+.6f}  L_contact_TJ_nm={tot['L_contact_TJ']*1e9:+.6f}  "
          f"L_contact_TJ_sub_nm={tot['L_contact_TJ_sub']*1e9:+.6f}  L_GB_geom_sub_nm={tot['L_GB_geom_sub']*1e9:+.6f}")
    for op in OPERATORS:
        v = summ["totals"][op]
        print(f"  {op:22s} dL_contact_TJ_sub_nm={v['L_contact_TJ_sub']*1e9:+.7f}  "
              f"dL_GB_geom_sub_nm={v['L_GB_geom_sub']*1e9:+.7f}  dx_neck_eta_nm={v['x_neck_m']*1e9:+.7f}")
    closure_nonzero = {k: v for k, v in summ["closure_error"].items() if v not in (0.0, None)}
    print(f"  closure errors (nonzero only): {closure_nonzero}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--coarsening-rate-scale", type=float, default=3.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--eta-mobility-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primary-overlap-nm", type=float, default=20.0)
    ap.add_argument("--baseline-overlap-nm", type=float, default=5.0)
    ap.add_argument("--target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--extended-dv2-frac", type=float, default=1.5e-3)
    ap.add_argument("--extended-dv2-frac-2", type=float, default=3e-3)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--fine-dx-nm", type=float, default=2.5)
    ap.add_argument("--out", type=Path, default=Path("runs/subgrid_contact_audit.json"))
    args = ap.parse_args()

    result = dict(args=vars(args) | dict(out=str(args.out)))

    # --- Primary case: 3e-4 ledger table ---
    p_primary = build_params(build_config(args.primary_overlap_nm, args.coarsening_rate_scale,
                                           args.surface_mobility_scale, args.eta_mobility_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p_primary)
    print(f"primary geometry: Nx={p_primary.Nx} Ny={p_primary.Ny} dx={p_primary.dx*1e9:.2f}nm "
          f"overlap={args.primary_overlap_nm}nm dt={p_primary.dt:.4e}s")

    steps_primary, summ_primary = run_ledger_case(p_primary, f0, e1_0, e2_0, e3_0,
                                                    args.target_dv2_frac, args.max_steps)
    print_ledger_table(f"PRIMARY ({args.primary_overlap_nm}nm) @ {args.target_dv2_frac}", summ_primary)
    result["primary_3e4"] = summ_primary

    # --- CH flux vs measured TJ velocity (first few steps of the primary case) ---
    print("\n=== CH flux vs measured TJ velocity (primary case, first 5 steps) ===")
    s_dummy = Sink(threshold=math.inf)
    f, e1, e2, e3 = f0.copy(), e1_0.copy(), e2_0.copy(), e3_0.copy()
    flux_vs_velocity = []
    from pf_sintering.model import evolve_f
    from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving
    from pf_sintering.model import ostwald_substrate, evolve_eta
    prev_top = None
    prev_bot = None
    for step in range(1, 6):
        sub0 = compute_subgrid_contact(f, e1, e2, p_primary)
        _, flux_diag = evolve_f_diagnostic(f, e1, e2, e3, s_dummy, p_primary)
        flux_top = None
        if sub0.top.resolved:
            flux_top = surface_flux_near_tj(f, flux_diag.Jx, flux_diag.Jy, (sub0.top.x_sub, sub0.top.y_sub), p_primary)
        flux_bot = None
        if sub0.bottom.resolved:
            flux_bot = surface_flux_near_tj(f, flux_diag.Jx, flux_diag.Jy, (sub0.bottom.x_sub, sub0.bottom.y_sub), p_primary)

        ch_targets = eta_masses(e1, e2, e3, p_primary.use_eta3)
        f1 = evolve_f(f, e1, e2, e3, s_dummy, Sink(), p_primary)
        e1a, e2a, e3a = project_eta_mass_preserving(f1, e1, e2, e3, p_primary, target_masses=ch_targets)
        sub1 = compute_subgrid_contact(f1, e1a, e2a, p_primary)

        row = dict(step=step, dt=p_primary.dt)
        if sub0.top.resolved and sub1.top.resolved:
            row["v_top_x"] = (sub1.top.x_sub - sub0.top.x_sub) / p_primary.dt
            row["v_top_y"] = (sub1.top.y_sub - sub0.top.y_sub) / p_primary.dt
        if sub0.bottom.resolved and sub1.bottom.resolved:
            row["v_bottom_x"] = (sub1.bottom.x_sub - sub0.bottom.x_sub) / p_primary.dt
            row["v_bottom_y"] = (sub1.bottom.y_sub - sub0.bottom.y_sub) / p_primary.dt
        row["J_tangent_top"] = flux_top["J_tangent"] if flux_top else None
        row["J_normal_top"] = flux_top["J_normal"] if flux_top else None
        row["J_tangent_bottom"] = flux_bot["J_tangent"] if flux_bot else None
        row["J_normal_bottom"] = flux_bot["J_normal"] if flux_bot else None
        row["L_contact_TJ_sub_before"] = sub0.L_contact_TJ_sub if sub0.resolved else None
        row["L_contact_TJ_sub_after_CH"] = sub1.L_contact_TJ_sub if sub1.resolved else None
        flux_vs_velocity.append(row)
        print(f"  step={step} v_top=({row.get('v_top_x')}, {row.get('v_top_y')}) "
              f"J_top=(tan={row['J_tangent_top']}, norm={row['J_normal_top']}) "
              f"dL_contact_TJ_sub_CH_nm={(row['L_contact_TJ_sub_after_CH']-row['L_contact_TJ_sub_before'])*1e9 if row['L_contact_TJ_sub_after_CH'] and row['L_contact_TJ_sub_before'] else None}")

        # advance the full step (Ostwald + eta + projections) to continue the trajectory
        f2, e1b, e2b, e3b = ostwald_substrate(f1, e1a, e2a, e3a, p_primary)
        ac_targets = eta_masses(e1b, e2b, e3b, p_primary.use_eta3)
        e1c, e2c, e3c = evolve_eta(e1b, e2b, e3b, p_primary)
        e1d, e2d, e3d = project_eta_mass_preserving(f2, e1c, e2c, e3c, p_primary, target_masses=ac_targets)
        f, e1, e2, e3 = f2, e1d, e2d, e3d

    result["ch_flux_vs_tj_velocity"] = flux_vs_velocity

    # --- Extended trajectory (primary geometry only) ---
    print(f"\n=== PRIMARY (20nm) extended to {args.extended_dv2_frac} ===")
    steps_ext, summ_ext = run_ledger_case(p_primary, f0, e1_0, e2_0, e3_0, args.extended_dv2_frac, args.max_steps)
    print_ledger_table(f"PRIMARY (20nm) @ {args.extended_dv2_frac}", summ_ext)
    result["primary_extended"] = summ_ext

    # slope: L_contact_TJ_sub vs V2/V20 sampled along the trajectory
    slope_rows = []
    for ledger in steps_ext:
        s0 = ledger.stage_samples["start"]
        slope_rows.append(dict(step=ledger.step, V2_ratio=s0["V2_ratio"], L_contact_TJ_sub=s0["L_contact_TJ_sub"],
                                L_GB_geom_sub=s0["L_GB_geom_sub"], x_neck_eta=s0["x_neck_m"], sigma_Pa=s0["sigma_Pa"]))
    last_ledger = steps_ext[-1].stage_samples["end"]
    slope_rows.append(dict(step=steps_ext[-1].step, V2_ratio=last_ledger["V2_ratio"],
                            L_contact_TJ_sub=last_ledger["L_contact_TJ_sub"], L_GB_geom_sub=last_ledger["L_GB_geom_sub"],
                            x_neck_eta=last_ledger["x_neck_m"], sigma_Pa=last_ledger["sigma_Pa"]))
    result["primary_extended_trajectory"] = slope_rows

    # optional further extension
    if args.extended_dv2_frac_2 and args.extended_dv2_frac_2 > args.extended_dv2_frac:
        print(f"\n=== PRIMARY (20nm) extended further to {args.extended_dv2_frac_2} ===")
        try:
            steps_ext2, summ_ext2 = run_ledger_case(p_primary, f0, e1_0, e2_0, e3_0,
                                                      args.extended_dv2_frac_2, args.max_steps)
            print_ledger_table(f"PRIMARY (20nm) @ {args.extended_dv2_frac_2}", summ_ext2)
            result["primary_extended_2"] = summ_ext2
        except Exception as exc:  # noqa: BLE001
            print(f"  extended-2 run failed: {exc}")
            result["primary_extended_2_error"] = str(exc)

    # --- Finer-grid directional check ---
    print(f"\n=== finer grid dx={args.fine_dx_nm}nm directional check ===")
    p_fine = build_params(build_config(args.primary_overlap_nm, args.coarsening_rate_scale,
                                        args.surface_mobility_scale, args.eta_mobility_scale, args,
                                        dx_nm=args.fine_dx_nm))
    f0f, e1_0f, e2_0f, e3_0f = initialize_fields(p_fine)
    print(f"  Nx={p_fine.Nx} Ny={p_fine.Ny} dx={p_fine.dx*1e9:.2f}nm dt={p_fine.dt:.4e}s")
    steps_fine, summ_fine = run_ledger_case(p_fine, f0f, e1_0f, e2_0f, e3_0f, args.target_dv2_frac, args.max_steps)
    print_ledger_table(f"FINE GRID (dx={args.fine_dx_nm}nm) @ {args.target_dv2_frac}", summ_fine)
    result["fine_grid"] = summ_fine

    # --- Baseline 5nm cross-check ---
    print("\n=== BASELINE (5nm overlap) cross-check ===")
    p_baseline = build_params(build_config(args.baseline_overlap_nm, args.coarsening_rate_scale,
                                            args.surface_mobility_scale, args.eta_mobility_scale, args))
    f0b, e1_0b, e2_0b, e3_0b = initialize_fields(p_baseline)
    steps_baseline, summ_baseline = run_ledger_case(p_baseline, f0b, e1_0b, e2_0b, e3_0b,
                                                      args.target_dv2_frac, args.max_steps)
    print_ledger_table(f"BASELINE ({args.baseline_overlap_nm}nm) @ {args.target_dv2_frac}", summ_baseline)
    result["baseline_3e4"] = summ_baseline

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
