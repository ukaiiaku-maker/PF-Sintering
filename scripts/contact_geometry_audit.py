#!/usr/bin/env python3
"""Milestone 6B: physical contact-geometry audit.

Runs the Milestone-6-extended operator ledger (operator_ledger.run_ledger_trajectory_v2)
for the exact Milestone-6 primary case (overlap=20nm, coarsening=3x,
surface=0.3x, eta=1x, target |dV2|/V20~3e-4) and the baseline-geometry
cross-check (overlap=5nm, same rates), reporting the new physical
contact-geometry metrics (contact_geometry.py) alongside the legacy
eta-only ones, plus an explicit grid-quantization ("jump") analysis of
L_contact_TJ/L_GB_geom, since both are anchored to the discrete-grid TJ
locations from tj_force.locate_neck_tjs and can only change in units of
~dx. See MILESTONE_6B_PHYSICAL_CONTACT_GEOMETRY_AUDIT.md for the writeup.

DIAGNOSTIC ONLY. hazard_step and rbm are never called. No production
physics is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.operator_ledger import OPERATORS, run_ledger_trajectory_v2, summarize_ledger_v2


def jump_events(steps, dx, keys=("L_contact_TJ", "L_GB_geom"), threshold_frac=0.5):
    """Per-operator, per-step deltas whose magnitude exceeds threshold_frac*dx
    -- i.e. grid-quantization jumps in the discrete-TJ-anchored metrics."""
    threshold = threshold_frac * dx
    events = []
    for ledger in steps:
        for op in OPERATORS:
            d = ledger.operator_deltas.get(op)
            if d is None:
                continue
            for key in keys:
                v = d.get(key)
                if v is not None and abs(v) >= threshold:
                    events.append(dict(step=ledger.step, operator=op, key=key, value=v))
    return events


def dequantized_totals(steps, dx, keys=("L_contact_TJ", "L_GB_geom"), threshold_frac=0.5):
    """Per-operator totals for `keys`, excluding any single-step delta that
    is itself a jump (>= threshold_frac*dx) -- isolates the continuous,
    sub-grid-resolution component of the signal from the discrete jumps."""
    threshold = threshold_frac * dx
    totals_raw = {op: {k: 0.0 for k in keys} for op in OPERATORS}
    totals_dequant = {op: {k: 0.0 for k in keys} for op in OPERATORS}
    n_excluded = {op: {k: 0 for k in keys} for op in OPERATORS}
    for ledger in steps:
        for op in OPERATORS:
            d = ledger.operator_deltas.get(op)
            if d is None:
                continue
            for key in keys:
                v = d.get(key)
                if v is None:
                    continue
                totals_raw[op][key] += v
                if abs(v) < threshold:
                    totals_dequant[op][key] += v
                else:
                    n_excluded[op][key] += 1
    return dict(raw=totals_raw, dequantized=totals_dequant, n_excluded=n_excluded)


def run_case(overlap_nm, args, tag):
    p = build_params(ModelConfig(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny, dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=args.coarsening_rate_scale, surface_mobility_scale=args.surface_mobility_scale,
        eta_mobility_scale=args.eta_mobility_scale,
    ))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p)
    steps = run_ledger_trajectory_v2(p, f0, e1_0, e2_0, e3_0,
                                      target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps)
    summ = summarize_ledger_v2(steps)
    jumps = jump_events(steps, p.dx)
    dequant = dequantized_totals(steps, p.dx)

    print(f"\n=== {tag}: overlap={overlap_nm}nm, n_steps={summ['n_steps']}, "
          f"dV2/V20={summ['dV2_over_V20']:+.4e} ===")
    tot = summ["total_observed"]
    print(f"  total x_neck_eta_nm={tot['x_neck_m']*1e9:+.6f}  A_GB_eta_nm2={tot['A_GB_m2']*1e18:+.4f}  "
          f"L_contact_TJ_nm={tot['L_contact_TJ']*1e9:+.6f}  L_GB_geom_nm={tot['L_GB_geom']*1e9:+.6f}")
    print(f"  L_f_farfield_nm={tot['L_f_farfield']*1e9:+.6f}  L_eta_centroid_nm={tot['separation_m']*1e9:+.6f}  "
          f"V2_eta={tot['V2']:+.4e}  V2_f_eta={tot['V2_f_eta']:+.4e}")
    print(f"  jump events (>=0.5dx in L_contact_TJ or L_GB_geom): {len(jumps)}")
    for j in jumps:
        print(f"    step={j['step']:4d} op={j['operator']:20s} key={j['key']:14s} value_nm={j['value']*1e9:+.4f}")
    print("  RAW vs DEQUANTIZED per-operator totals:")
    for op in OPERATORS:
        r = dequant["raw"][op]
        dq = dequant["dequantized"][op]
        e = summ["totals"][op]
        print(f"    {op:22s} L_contact_TJ raw={r['L_contact_TJ']*1e9:+.6f} dequant={dq['L_contact_TJ']*1e9:+.6f}  "
              f"L_GB_geom raw={r['L_GB_geom']*1e9:+.6f} dequant={dq['L_GB_geom']*1e9:+.6f}  "
              f"x_neck_eta={e['x_neck_m']*1e9:+.6f}")

    closure_nonzero = {k: v for k, v in summ["closure_error"].items() if v not in (0.0, None)}
    print(f"  closure errors (nonzero only): {closure_nonzero}")

    return dict(overlap_nm=overlap_nm, dt_s=p.dt, ledger_summary=summ, jumps=jumps, dequantized=dequant)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--coarsening-rate-scale", type=float, default=3.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--eta-mobility-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("runs/contact_geometry_audit.json"))
    args = ap.parse_args()

    primary = run_case(20.0, args, "PRIMARY (wider)")
    baseline = run_case(5.0, args, "CROSS-CHECK (baseline)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(dict(args=vars(args) | dict(out=str(args.out)), primary=primary, baseline=baseline),
                   fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
