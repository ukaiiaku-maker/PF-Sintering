#!/usr/bin/env python3
"""Operator-resolved audit of sink-off natural-coarsening neck widening.

Runs, for a given geometry/rate case:

1. The full operator-resolved ledger (CH / post-CH projection / Ostwald /
   eta / post-eta projection), accumulated over a short trajectory.
2. Spatial Ostwald removal/addition diagnostics (near-neck fractions,
   centroids) at a few representative snapshots, plus saved PNG maps.
3. The O0 (normal) / O1 (removal-only) / O2 (addition-only) mechanism
   comparison, all three run for the same number of steps (O0's step count
   to reach the target), so they are on an equal physical-time footing.
4. CH chemical-potential/flux diagnostics near both TJs at the same
   snapshots.

DIAGNOSTIC ONLY. hazard_step and rbm are never called anywhere in this
script. No production physics is modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pf_sintering.ch_flux_diagnostics import evolve_f_diagnostic, surface_flux_near_tj
from pf_sintering.model import (
    ModelConfig,
    Sink,
    build_params,
    compute_stress,
    contact_width,
    evolve_eta,
    evolve_f,
    initialize_fields,
    ostwald_substrate,
)
from pf_sintering.operator_ledger import OPERATORS, run_ledger_trajectory, summarize_ledger
from pf_sintering.ostwald_diagnostics import (
    centroid,
    near_neck_fractions,
    ostwald_addition_only,
    ostwald_removal_only,
    ostwald_substrate_diagnostic,
)
from pf_sintering.diagnostics import contact_area
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving
from pf_sintering.tj_force import locate_neck_tjs


def build_config(overlap_nm, coarsening_rate_scale, surface_mobility_scale, eta_mobility_scale, args):
    return ModelConfig(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny, dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=coarsening_rate_scale, surface_mobility_scale=surface_mobility_scale,
        eta_mobility_scale=eta_mobility_scale,
    )


# ---------------------------------------------------------------------------
# Part 1: operator ledger
# ---------------------------------------------------------------------------

def run_ledger_and_table(p, f0, e1_0, e2_0, e3_0, target_dv2_frac, max_steps):
    steps = run_ledger_trajectory(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=target_dv2_frac, max_steps=max_steps)
    summ = summarize_ledger(steps)
    return steps, summ


# ---------------------------------------------------------------------------
# Part 2: spatial Ostwald snapshots
# ---------------------------------------------------------------------------

def light_step(f, e1, e2, e3, p, s):
    ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f1 = evolve_f(f, e1, e2, e3, s, Sink(), p)
    e1a, e2a, e3a = project_eta_mass_preserving(f1, e1, e2, e3, p, target_masses=ch_targets)
    f2, e1b, e2b, e3b = ostwald_substrate(f1, e1a, e2a, e3a, p)
    ac_targets = eta_masses(e1b, e2b, e3b, p.use_eta3)
    e1c, e2c, e3c = evolve_eta(e1b, e2b, e3b, p)
    e1d, e2d, e3d = project_eta_mass_preserving(f2, e1c, e2c, e3c, p, target_masses=ac_targets)
    return f2, e1d, e2d, e3d


def spatial_snapshots(p, f0, e1_0, e2_0, e3_0, n_snapshots, total_steps, out_dir, tag):
    s = Sink(threshold=math.inf)
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    checkpoints = sorted(set(int(round(x)) for x in np.linspace(0, total_steps, n_snapshots)))
    results = []
    step = 0
    for target_step in checkpoints:
        while step < target_step:
            f, e1, e2, e3 = light_step(f, e1, e2, e3, p, s)
            step += 1
        tjs = locate_neck_tjs(f, e1, e2, p)
        _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
        entry = dict(step=step)
        if tjs is not None:
            fr_remove = near_neck_fractions(diag.remove, tjs["tj_top"], tjs["tj_bottom"], p)
            fr_add = near_neck_fractions(diag.add, tjs["tj_top"], tjs["tj_bottom"], p)
            c_remove = centroid(diag.remove, p)
            c_add = centroid(diag.add, p)
            entry.update(
                tot_removed=diag.actual_removed, tot_added=diag.actual_added,
                conservation_residual=diag.actual_removed - diag.actual_added,
                frac_removed_near_neck=fr_remove, frac_added_near_neck=fr_add,
                centroid_removal=c_remove, centroid_addition=c_add,
                tj_top=tuple(float(x) for x in tjs["tj_top"]),
                tj_bottom=tuple(float(x) for x in tjs["tj_bottom"]),
            )
            _save_spatial_plot(f, e1, e2, diag, tjs, p, out_dir, f"{tag}_step{step:05d}")
        results.append(entry)
    return results, (f, e1, e2, e3)


def _save_spatial_plot(f, e1, e2, diag, tjs, p, out_dir, name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    extent = [0, p.Nx * p.dx * 1e9, p.Ny * p.dx * 1e9, 0]
    panels = [
        ("f", f, axes[0, 0]),
        ("eta1 (substrate)", e1, axes[0, 1]),
        ("eta2 (particle)", e2, axes[0, 2]),
        ("removal_candidate (pre-conv)", diag.removal_candidate, axes[1, 0]),
        ("remove (applied)", diag.remove, axes[1, 1]),
        ("add (applied)", diag.add, axes[1, 2]),
    ]
    for title, arr, ax in panels:
        im = ax.imshow(arr, extent=extent, origin="upper", aspect="auto", cmap="viridis")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        tjx = [tjs["tj_top"][0] * 1e9, tjs["tj_bottom"][0] * 1e9]
        tjy = [tjs["tj_top"][1] * 1e9, tjs["tj_bottom"][1] * 1e9]
        ax.plot(tjx, tjy, "r+", markersize=12, mew=2)
    fig.suptitle(name)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 3: O0 / O1 / O2 mechanism comparison
# ---------------------------------------------------------------------------

def run_variant_trajectory(p, f0, e1_0, e2_0, e3_0, variant, n_steps):
    s = Sink(threshold=math.inf)
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    v20 = float(e2.sum() * p.dx * p.dx)
    external_reservoir = 0.0
    artificial_added = 0.0
    x0, _ = contact_width(e1, e2, p)
    a_gb0 = contact_area(e1, e2, p)

    for _ in range(n_steps):
        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, Sink(), p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)

        if variant == "O0":
            f, e1, e2, e3 = ostwald_substrate(f, e1, e2, e3, p)
        elif variant == "O1":
            f, e1, e2, e3, removed = ostwald_removal_only(f, e1, e2, e3, p)
            external_reservoir += removed
        elif variant == "O2":
            f, e1, e2, e3, added = ostwald_addition_only(f, e1, e2, e3, p)
            artificial_added += added
        else:
            raise ValueError(variant)

        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)

    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    x1, _ = contact_width(e1, e2, p)
    a_gb1 = contact_area(e1, e2, p)
    v2 = float(e2.sum() * p.dx * p.dx)
    return dict(
        variant=variant, n_steps=n_steps, stop=stop, reason=reason,
        x_neck0_nm=x0 * 1e9, x_neck1_nm=x1 * 1e9, d_x_neck_nm=(x1 - x0) * 1e9,
        A_GB0_nm2=a_gb0 * 1e18, A_GB1_nm2=a_gb1 * 1e18, d_A_GB_nm2=(a_gb1 - a_gb0) * 1e18,
        V2_ratio=v2 / v20, external_reservoir=external_reservoir, artificial_added=artificial_added,
        sigma_Pa=st.sigma if not stop else None,
    )


def run_o0_o1_o2(p, f0, e1_0, e2_0, e3_0, target_dv2_frac, max_steps):
    # First find how many steps O0 needs to reach the target (physical-time anchor).
    s = Sink(threshold=math.inf)
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    v20 = float(e2.sum() * p.dx * p.dx)
    n = 0
    for step in range(1, max_steps + 1):
        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, Sink(), p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)
        f, e1, e2, e3 = ostwald_substrate(f, e1, e2, e3, p)
        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)
        v2 = float(e2.sum() * p.dx * p.dx)
        if abs(v2 - v20) / v20 >= target_dv2_frac:
            n = step
            break
    if n == 0:
        n = max_steps

    results = {}
    for variant in ("O0", "O1", "O2"):
        results[variant] = run_variant_trajectory(p, f0, e1_0, e2_0, e3_0, variant, n)
    return n, results


# ---------------------------------------------------------------------------
# Part 4: CH flux near TJs at representative snapshots
# ---------------------------------------------------------------------------

def ch_flux_at_snapshot(f, e1, e2, e3, p):
    s = Sink(threshold=math.inf)
    _, diag = evolve_f_diagnostic(f, e1, e2, e3, s, p)
    tjs = locate_neck_tjs(f, e1, e2, p)
    out = dict()
    if tjs is not None:
        out["top"] = surface_flux_near_tj(f, diag.Jx, diag.Jy, tjs["tj_top"], p)
        out["bottom"] = surface_flux_near_tj(f, diag.Jx, diag.Jy, tjs["tj_bottom"], p)
    out["mu_mean"] = float(np.mean(diag.mu))
    out["mu_std"] = float(np.std(diag.mu))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--coarsening-rate-scale", type=float, default=3.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--eta-mobility-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ledger-target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--ledger-max-steps", type=int, default=5000)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--compact", action="store_true", help="ledger table only (cross-check runs)")
    ap.add_argument("--out", type=Path, default=Path("runs/operator_audit_primary.json"))
    ap.add_argument("--plot-dir", type=Path, default=Path("runs/operator_audit_plots"))
    args = ap.parse_args()

    p = build_params(build_config(args.overlap_nm, args.coarsening_rate_scale,
                                   args.surface_mobility_scale, args.eta_mobility_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p)
    print(f"case: overlap={args.overlap_nm}nm coarsening={args.coarsening_rate_scale}x "
          f"surface={args.surface_mobility_scale}x eta={args.eta_mobility_scale}x dt={p.dt:.4e}s")

    print("\n=== operator ledger ===")
    steps, summ = run_ledger_and_table(p, f0, e1_0, e2_0, e3_0, args.ledger_target_dv2_frac, args.ledger_max_steps)
    print(f"n_steps={summ['n_steps']} dV2/V20={summ['dV2_over_V20']:+.4e}")
    for op in OPERATORS:
        v = summ["totals"][op]
        tot_xneck = summ["total_observed"]["x_neck_m"]
        frac = v["x_neck_m"] / tot_xneck * 100 if tot_xneck else float("nan")
        print(f"  {op:22s} dx_neck_nm={v['x_neck_m']*1e9:+.6f} ({frac:+.1f}%)  "
              f"dA_GB_nm2={v['A_GB_m2']*1e18:+.4f}  dV2={v['V2']:+.3e}  dsigma_MPa={v['sigma_Pa']/1e6:+.5f}")
    print("closure errors:", {k: v for k, v in summ["closure_error"].items() if v not in (0.0, None)})

    result = dict(args=vars(args) | dict(out=str(args.out)), ledger_summary=summ)

    if not args.compact:
        print("\n=== spatial Ostwald snapshots ===")
        snaps, final_state = spatial_snapshots(
            p, f0, e1_0, e2_0, e3_0, args.snapshots, summ["n_steps"], args.plot_dir, "primary",
        )
        for sN in snaps:
            print(f"  step={sN['step']:5d} tot_removed={sN.get('tot_removed')} "
                  f"frac_removed_near_neck={sN.get('frac_removed_near_neck')} "
                  f"frac_added_near_neck={sN.get('frac_added_near_neck')} "
                  f"centroid_removal={sN.get('centroid_removal')} centroid_addition={sN.get('centroid_addition')}")
        result["spatial_snapshots"] = snaps

        print("\n=== O0 / O1 / O2 mechanism comparison ===")
        n_o, o_results = run_o0_o1_o2(p, f0, e1_0, e2_0, e3_0, args.ledger_target_dv2_frac, args.ledger_max_steps)
        print(f"n_steps used for all three variants: {n_o}")
        for variant, r in o_results.items():
            print(f"  [{variant}] d_x_neck_nm={r['d_x_neck_nm']:+.4f} d_A_GB_nm2={r['d_A_GB_nm2']:+.4f} "
                  f"V2_ratio={r['V2_ratio']:.6f} sigma_Pa={r['sigma_Pa']}")
        result["o0_o1_o2"] = dict(n_steps=n_o, results=o_results)

        print("\n=== CH flux near TJs (initial and final ledger state) ===")
        flux_initial = ch_flux_at_snapshot(f0, e1_0, e2_0, e3_0, p)
        flux_final = ch_flux_at_snapshot(*final_state, p)
        print("  initial:", flux_initial)
        print("  final:  ", flux_final)
        result["ch_flux"] = dict(initial=flux_initial, final=flux_final)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
