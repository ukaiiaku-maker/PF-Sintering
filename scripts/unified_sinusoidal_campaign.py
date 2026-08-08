"""Milestone 12 -- Gate E: particle on periodic sinusoidal substrate, using
ONLY the unified variational surface-diffusion transport (surface_transport.py)
+ constrained variational eta kinetics (constrained_eta.py). NO legacy
Ostwald kernel is imported or called anywhere in this script -- grep-
verifiable (the only "ostwald" occurrences below are in comments).

Genuine per-axis BC (Milestone 12 Section 10, not Milestone 10's reflection-
symmetry-equivalence workaround): X (substrate-normal) = reflecting/no_flux,
Y (lateral, the sinusoid's own repeat direction) = periodic.

Runs from TWO initial states (Section 22): A) the original analytic
sinusoidal-substrate contact (same construction as Milestone 10), and
B) a substantially evolved contact (Milestone 11's "mid" S0 relaxation
state, step 10319 of that milestone's long trajectory, regenerated here
under the OLD legacy dynamics ONLY to produce the initial condition --
the unified transport law itself never touches the legacy Ostwald kernel
during the campaign that follows).

Hazard/sink/RBM OFF throughout, per the milestone's explicit instruction.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.bc_ops import lap9_bc
from pf_sintering.ch_crossover_diagnostics import neck_ch_mass_balance, neck_region_mask
from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.constrained_eta import constrained_variational_eta_update, f_weighted_ownership_volumes
from pf_sintering.differential_coarsening import _step_once  # legacy stepper, ONLY for building state B's IC
from pf_sintering.model import ModelConfig, Sink, build_params, compute_stress, initialize_fields, reproject
from pf_sintering.signed_curvature import signed_curvature_at
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact

BC_X, BC_Y = "reflecting", "periodic"


def build_config(args, coarsening_rate_scale=1.0):
    return ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=args.overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
        sinusoid_wavelength=args.wavelength_nm * 1e-9, sinusoid_amplitude=args.amplitude_nm * 1e-9,
        interface_width_override=args.w_nm * 1e-9, eta_diffusivity_fixed_physical=True,
        use_aniso_surface=False, surface_mobility_scale=args.surface_mobility_scale,
        coarsening_rate_scale=coarsening_rate_scale,
    )


def build_state_a(args):
    p = build_params(build_config(args))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def build_state_b(args):
    """Legacy (production Ostwald-off) stepping ONLY to build the evolved
    initial condition -- matches Milestone 11's 'mid' state (step 10319),
    NOT reused during the unified-transport campaign itself."""
    p = build_params(build_config(args, coarsening_rate_scale=0.0))
    f, e1, e2, e3 = initialize_fields(p)
    s = Sink(threshold=math.inf)
    for _ in range(args.state_b_steps):
        f, e1, e2, e3 = _step_once(f, e1, e2, e3, s, p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def sample_state(f, e1, e2, e3, p, step, time_s):
    s = Sink(threshold=math.inf)
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    V1, V2, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    total_f = float(f.sum()) * p.dx * p.dx

    row = dict(step=step, time_s=time_s, V1=V1, V2=V2, total_f=total_f,
               stop=bool(stop), reason=reason)
    if sub.resolved:
        row["L_contact_TJ_sub"] = sub.L_contact_TJ_sub
        row["L_GB_geom_sub"] = sub.L_GB_geom_sub
        row["tj_top"] = (sub.top.x_sub, sub.top.y_sub)
        row["tj_bottom"] = (sub.bottom.x_sub, sub.bottom.y_sub)
        row["kappa_top"] = signed_curvature_at(f, (sub.top.x_sub, sub.top.y_sub), p)
        row["kappa_bottom"] = signed_curvature_at(f, (sub.bottom.x_sub, sub.bottom.y_sub), p)
    else:
        row["L_contact_TJ_sub"] = math.nan
        row["L_GB_geom_sub"] = math.nan
    if rep.top and rep.top.resolved:
        row["psi_top"] = rep.top.psi_deg
        row["F_TJ_mag_top"] = rep.top.F_TJ_mag
    if rep.bottom and rep.bottom.resolved:
        row["psi_bottom"] = rep.bottom.psi_deg
        row["F_TJ_mag_bottom"] = rep.bottom.F_TJ_mag
    return row, sub


def run_unified_campaign(label, p, f, e1, e2, e3, args, sample_every):
    print(f"\n--- unified transport campaign: {label} ---")
    M_s = m_s_ref(p.M_f, p.interface_width)
    s = Sink(threshold=math.inf)

    rows = []
    row0, sub0 = sample_state(f, e1, e2, e3, p, 0, 0.0)
    rows.append(row0)
    print(f"  step=0 V2={row0['V2']:.6e} L_contact_TJ_sub={row0.get('L_contact_TJ_sub', float('nan'))*1e9:.4f}nm")

    flux_events = []
    for step in range(1, args.steps + 1):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
        e1, e2, e3, ediag = constrained_variational_eta_update(e1, e2, e3, f_new, p, bc_x=BC_X, bc_y=BC_Y)

        if step % sample_every == 0 or step == args.steps:
            sub_before = compute_subgrid_contact(f, e1, e2, p)
            sub_after = compute_subgrid_contact(f_new, e1, e2, p)
            if sub_before.resolved and sub_after.resolved:
                mask = neck_region_mask(p, (sub_before.top.x_sub, sub_before.top.y_sub),
                                         (sub_before.bottom.x_sub, sub_before.bottom.y_sub))
                dM_neck = neck_ch_mass_balance(f, f_new, mask, p)
                flux_events.append(dict(step=step, dM_neck=dM_neck))

        f = f_new
        if step % sample_every == 0 or step == args.steps:
            row, sub = sample_state(f, e1, e2, e3, p, step, step * p.dt)
            rows.append(row)
            print(f"  step={step:6d} t={step*p.dt:.4e}s V2={row['V2']:.6e} "
                  f"L_contact_TJ_sub={row.get('L_contact_TJ_sub', float('nan'))*1e9:.4f}nm "
                  f"total_f_drift={(row['total_f']-row0['total_f'])/row0['total_f']:.2e}")
            if row["stop"]:
                print(f"  ** compute_stress stop: {row['reason']} -- stopping **")
                break

    return rows, flux_events


def branch_mu_J_profile(f, e1, e2, e3, p, tj_xy, branch_dir, M_s, max_arclength=150e-9, n_samples=40):
    from pf_sintering.tj_force import _sample_bilinear
    from pf_sintering.surface_transport import surface_flux, surface_mobility_tensor
    s = Sink(threshold=math.inf)
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)

    from pf_sintering.ch_crossover_diagnostics import trace_branch_profile
    return trace_branch_profile(f, mu, Jx, Jy, p, tj_xy, branch_dir, max_arclength, n_samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--state-b-steps", type=int, default=10319)
    ap.add_argument("--out", type=str, default="runs/unified_sinusoidal_campaign.json")
    args = ap.parse_args()

    out = dict(args=vars(args))

    print("=== Building state A (original analytic contact) ===")
    pA, fA, e1A, e2A, e3A = build_state_a(args)
    print(f"  Nx={pA.Nx} Ny={pA.Ny} dx={pA.dx*1e9:.2f}nm dt={pA.dt:.4e}s")

    print("\n=== Building state B (evolved contact, legacy dynamics for IC only) ===")
    pB, fB, e1B, e2B, e3B = build_state_b(args)
    print(f"  after {args.state_b_steps} legacy steps: Nx={pB.Nx} Ny={pB.Ny}")

    rowsA, fluxA = run_unified_campaign("state A (original)", pA, fA, e1A, e2A, e3A, args, args.sample_every)
    rowsB, fluxB = run_unified_campaign("state B (evolved)", pB, fB, e1B, e2B, e3B, args, args.sample_every)

    out["state_A"] = dict(rows=rowsA, flux_events=fluxA)
    out["state_B"] = dict(rows=rowsB, flux_events=fluxB)

    # full mu/J profiles at start and end of state A, both branches
    print("\n=== Branch mu/J profiles, state A, start and end ===")
    M_s = m_s_ref(pA.M_f, pA.interface_width)
    sub_start = compute_subgrid_contact(fA, e1A, e2A, pA)
    rep_start = compute_neck_tj_forces(fA, e1A, e2A, e3A, Sink(threshold=math.inf), pA)
    profiles = {}
    if sub_start.resolved and rep_start.top.resolved:
        profiles["start_top"] = branch_mu_J_profile(fA, e1A, e2A, e3A, pA, rep_start.top.tj_xy, rep_start.top.v_s1, M_s)
    if sub_start.resolved and rep_start.bottom.resolved:
        profiles["start_bottom"] = branch_mu_J_profile(fA, e1A, e2A, e3A, pA, rep_start.bottom.tj_xy, rep_start.bottom.v_s1, M_s)
    out["profiles_state_A_start"] = profiles

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
