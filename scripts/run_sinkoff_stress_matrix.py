#!/usr/bin/env python3
"""Milestone 2 sink-off 2x2 directional benchmark
(CODEX_PHYSICS_SEQUENCE.md Milestone 2 / handoff Section 9).

Runs the SAME starting geometry through four cases that independently toggle
the two leading-hypothesis mechanism controls:

  H1 (reservoir neck protection): Params.reservoir_neck_unprotected
  H2 (GB/eta mobility):           ModelConfig.eta_mobility_scale

Hazard integration and RBM are both suspended completely for this benchmark
(not merely prevented from firing): the loop below never calls hazard_step or
rbm. This is the literal Milestone-2 instruction ("Disable hazard and RBM
completely") and is also the only way to guarantee, by construction rather
than by tuning a random threshold, that zero rigid-body densification occurs.
One documented side effect: the dynamic GB-excess-energy state (Sink.g_ex),
which is normally updated inside hazard_step, stays pinned at its initial
value (0) for the whole benchmark, so effective_gamma(s, p) == gamma_gb_ref
throughout. That is a known simplification of this isolation benchmark, not a
bug; it is revisited when the hazard is reconnected in Milestone 5.

Each step reproduces the production per-operator sequence from
pf_sintering/runner.py (CH -> mass-preserving eta projection -> Ostwald ->
structural relaxation -> mass-preserving eta projection) so the qualified
mass-accounting contract in README.md is preserved during the benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pf_sintering.diagnostics import sample, summarize, wall_x0
from pf_sintering.model import (
    ModelConfig,
    Sink,
    build_params,
    center,
    compute_stress,
    evolve_eta,
    evolve_f,
    initialize_fields,
    ostwald_substrate,
)
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving

CASES = {
    "A_current_reservoir_current_Meta": dict(reservoir_neck_unprotected=False, eta_mobility_scale=1.0),
    "B_unprotected_reservoir_current_Meta": dict(reservoir_neck_unprotected=True, eta_mobility_scale=1.0),
    "C_current_reservoir_zero_Meta": dict(reservoir_neck_unprotected=False, eta_mobility_scale=0.0),
    "D_unprotected_reservoir_zero_Meta": dict(reservoir_neck_unprotected=True, eta_mobility_scale=0.0),
}


def base_config(args) -> ModelConfig:
    return ModelConfig(
        preset="dev", geometry="substrate",
        nx=args.nx, ny=args.ny, dx=args.dx_nm * 1e-9, r2=args.r2_nm * 1e-9,
        aspect_ratio=args.aspect_ratio, contact_orientation=args.contact_orientation,
        initial_overlap=args.overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
    )


def run_case(case_name, overrides, f0, e1_0, e2_0, e3_0, args):
    cfg = base_config(args)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    p = build_params(cfg)

    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)  # never touched: hazard integration is suspended below

    v20 = float(e2.sum() * p.dx * p.dx)
    st0, stop0, reason0 = compute_stress(f, e1, e2, e3, s, p)
    if stop0:
        raise RuntimeError(f"case {case_name}: initial state already stopped ({reason0})")

    samples = [sample(f, e1, e2, e3, s, st0, p, step=0, time_s=0.0, v20=v20)]
    step = 0
    while step < args.max_steps:
        step += 1

        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, Sink(), p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)

        f, e1, e2, e3 = ostwald_substrate(f, e1, e2, e3, p)

        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)

        st, stop, reason = compute_stress(f, e1, e2, e3, s, p)

        if step % args.sample_every == 0 or stop:
            samples.append(sample(f, e1, e2, e3, s, st, p, step=step, time_s=step * p.dt, v20=v20))

        v2 = float(e2.sum() * p.dx * p.dx)
        if stop:
            print(f"[{case_name}] stopped at step {step}: {reason}")
            break
        if abs(v2 - v20) / v20 >= args.target_dv2_frac:
            break

    final_mass_f = float(f.sum() * p.dx * p.dx)
    return dict(
        case=case_name, overrides=overrides, params=dict(dt=p.dt, Nx=p.Nx, Ny=p.Ny, dx=p.dx, GS1=p.GS1),
        samples=samples, summary=summarize(samples), steps_run=step,
        final_mass_f=final_mass_f,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--contact-orientation", choices=["short_plane", "long_plane"], default="short_plane")
    ap.add_argument("--overlap-nm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("runs/sinkoff_stress_matrix.json"))
    args = ap.parse_args()

    base_p = build_params(base_config(args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(base_p)
    initial_mass_f = float(f0.sum() * base_p.dx * base_p.dx)
    L0 = float(center(e2_0, base_p) - wall_x0(base_p))

    print(f"base geometry: Nx={base_p.Nx} Ny={base_p.Ny} dx={base_p.dx*1e9:.2f}nm "
          f"R2={base_p.R2*1e9:.1f}nm aspect_ratio={base_p.aspect_ratio} "
          f"contact_orientation={base_p.contact_orientation} overlap={base_p.initial_overlap*1e9:.2f}nm "
          f"GS1={base_p.GS1*1e9:.1f}nm dt={base_p.dt:.4e}s")
    print(f"initial separation L0={L0*1e9:.4f} nm, initial mass(f)={initial_mass_f:.6e}")

    results = {}
    for name, overrides in CASES.items():
        results[name] = run_case(name, overrides, f0, e1_0, e2_0, e3_0, args)
        summ = results[name]["summary"]
        final_mass_f = results[name]["final_mass_f"]
        print(
            f"[{name}] steps={results[name]['steps_run']} "
            f"dV2/V20={summ['dV2_over_V20']:+.3e} "
            f"dL_nm={summ['d_separation_m']*1e9:+.5f} "
            f"dstrain={summ['d_strain']:+.3e} "
            f"dx_neck_nm={summ['d_x_neck_m']*1e9:+.4f} "
            f"dA_GB_nm2={summ['d_A_GB_m2']*1e18:+.4f} "
            f"dsigma_MPa={summ['d_sigma_Pa']/1e6:+.5f} "
            f"dE_surf_J={summ['d_E_surf_J']:+.4e} "
            f"dE_gb_J={summ['d_E_gb_J']:+.4e} "
            f"dG_interface_J={summ['d_G_interface_J']:+.4e} "
            f"mass_f_drift={(final_mass_f-initial_mass_f)/initial_mass_f:+.3e}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(dict(args=vars(args) | dict(out=str(args.out)), L0_m=L0,
                        initial_mass_f=initial_mass_f, results=results), fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
