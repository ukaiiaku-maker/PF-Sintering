#!/usr/bin/env python3
"""Rate-competition campaign: let the model's OWN coarsening physics run
(Ostwald + CH + structural relaxation), sink and RBM off, and test whether
the sign of the neck-width response depends on the competition between the
Ostwald/coarsening rate, the free-surface/CH shape-relaxation rate, and (in a
follow-up matrix) the GB/TJ structural-relaxation rate.

No V2 trajectory is prescribed anywhere in this script -- V2(t) emerges from
the existing Ostwald kernel. No shape is frozen or globally minimized. dt is
fixed across the whole matrix (via --dt-override, computed once from the
1x/1x baseline) purely so that every case in the matrix integrates the same
physical time per step -- see the accompanying report for why this is
necessary for a fair rate-ratio comparison (surface_mobility_scale otherwise
gets silently canceled by build_params' own CFL-derived dt in some regimes).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.rate_competition import run_sinkoff_trajectory

_DELTA_KEYS = (
    "V2", "V2_ratio", "separation_m", "strain", "x_neck_m", "A_GB_m2",
    "sigma_Pa", "sigma_lt_Pa", "sigma_curv_Pa", "E_surf_J", "E_gb_J", "G_interface_J",
)
_RICH_DELTA_KEYS = (
    "psi_deg_top", "psi_deg_bottom", "F_TJ_mag_top", "F_TJ_mag_bottom",
    "F_TJ_x_top", "F_TJ_x_bottom", "kappa_top_1pm", "kappa_bottom_1pm", "F_drive",
)


def summarize_rich(samples):
    if len(samples) < 2:
        raise ValueError("need >=2 samples")
    first, last = samples[0], samples[-1]
    out = dict(n_samples=len(samples), step0=first["step"], step1=last["step"])
    for k in _DELTA_KEYS:
        out[f"d_{k}"] = last[k] - first[k]
    for k in _RICH_DELTA_KEYS:
        a, b = first.get(k), last.get(k)
        out[f"d_{k}"] = (b - a) if (a is not None and b is not None and math.isfinite(a) and math.isfinite(b)) else None
    dv2 = last["V2"] - first["V2"]
    out["V20"] = first["V2"]
    out["dV2_over_V20"] = dv2 / first["V2"] if first["V2"] else math.nan
    out["first"] = first
    out["last"] = last
    return out


def base_config(args, **overrides):
    kw = dict(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny, dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=overrides.pop("initial_overlap_nm", args.overlap_nm) * 1e-9,
        t_total=1e-6, seed=args.seed,
    )
    kw.update(overrides)
    return ModelConfig(**kw)


def run_case(args, geometry_name, overlap_nm, coarsening_rate_scale, surface_mobility_scale,
             eta_mobility_scale, dt_fixed):
    cfg = base_config(
        args, initial_overlap_nm=overlap_nm,
        coarsening_rate_scale=coarsening_rate_scale,
        surface_mobility_scale=surface_mobility_scale,
        eta_mobility_scale=eta_mobility_scale,
        dt_override=dt_fixed,
    )
    p = build_params(cfg)
    f0, e1_0, e2_0, e3_0 = initialize_fields(p)
    samples = run_sinkoff_trajectory(
        p, f0, e1_0, e2_0, e3_0,
        target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps,
        sample_every=args.sample_every, seed=args.seed,
    )
    summ = summarize_rich(samples)
    return dict(
        geometry=geometry_name, overlap_nm=overlap_nm,
        coarsening_rate_scale=coarsening_rate_scale,
        surface_mobility_scale=surface_mobility_scale,
        eta_mobility_scale=eta_mobility_scale,
        dt_s=p.dt, steps_run=summ["step1"], summary=summ,
    )


def print_row(row):
    s = row["summary"]
    print(
        f"[{row['geometry']:>8s} ov={row['overlap_nm']:5.1f}nm "
        f"coarse={row['coarsening_rate_scale']:.1f}x surf={row['surface_mobility_scale']:.1f}x "
        f"eta={row['eta_mobility_scale']:.1f}x] steps={row['steps_run']:5d} "
        f"dV2/V20={s['dV2_over_V20']:+.3e} "
        f"dx_neck_nm={s['d_x_neck_m']*1e9:+.4f} "
        f"dA_GB_nm2={s['d_A_GB_m2']*1e18:+.4f} "
        f"dsigma_MPa={s['d_sigma_Pa']/1e6:+.5f} "
        f"dpsi_top={s['d_psi_deg_top']} dpsi_bot={s['d_psi_deg_bottom']} "
        f"dF_TJ_top={s['d_F_TJ_mag_top']} dF_TJ_bot={s['d_F_TJ_mag_bottom']} "
        f"dG_int_J={s['d_G_interface_J']:+.4e}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=5.0, help="baseline (narrower) geometry overlap")
    ap.add_argument("--overlap-wide-nm", type=float, default=20.0, help="wider-neck geometry overlap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-dv2-frac", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--coarsening-scales", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--surface-scales", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--eta-scales", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--out", type=Path, default=Path("runs/rate_competition_benchmark.json"))
    args = ap.parse_args()

    # Fixed dt for the whole campaign, from the 1x/1x/1x baseline on the
    # narrower geometry -- see module docstring for why this must be fixed
    # rather than left to build_params' own CFL formula.
    base_p = build_params(base_config(args))
    dt_fixed = base_p.dt
    print(f"base geometry: Nx={base_p.Nx} Ny={base_p.Ny} dx={base_p.dx*1e9:.2f}nm R2={base_p.R2*1e9:.1f}nm "
          f"GS1={base_p.GS1*1e9:.1f}nm dt_fixed={dt_fixed:.4e}s tau_ripening_base={base_p.tau_ripening:.3f} "
          f"M_f_base={base_p.M_f:.4e} M_eta_base={base_p.M_eta:.4e}")

    geometries = [("baseline", args.overlap_nm), ("wider", args.overlap_wide_nm)]

    results = []
    print("\n=== 3x3 coarsening-rate x surface-rate matrix (eta_mobility_scale=1.0) ===")
    for geom_name, ov in geometries:
        for cscale in args.coarsening_scales:
            for sscale in args.surface_scales:
                row = run_case(args, geom_name, ov, cscale, sscale, 1.0, dt_fixed)
                print_row(row)
                results.append(row)

    print("\n=== eta/GB-mobility variants on representative cases (coarsening=1x, surface=1x) ===")
    eta_results = []
    for geom_name, ov in geometries:
        for escale in args.eta_scales:
            if escale == 1.0:
                continue  # already covered by the 1x/1x cell above
            row = run_case(args, geom_name, ov, 1.0, 1.0, escale, dt_fixed)
            print_row(row)
            eta_results.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(dict(args=vars(args) | dict(out=str(args.out)), dt_fixed=dt_fixed,
                        matrix=results, eta_variants=eta_results), fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
