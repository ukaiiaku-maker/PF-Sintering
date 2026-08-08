"""Milestone 12B Sections 13-17: definitive control-volume flux closure and
full mu(s)/J_s(s) surface trace at a representative mid-campaign State-A
snapshot, at dx=5nm and dx=2.5nm.

Evolves state A with the SAME unified isotropic transport as
unified_sinusoidal_campaign.py / m12b_grid_convergence.py to a target
physical time, then for the next single step reports:
  - particle_volume_rate_decomposition (Section 15): dV2/dt split into
    f-transport vs. eta-ownership-migration contributions.
  - neck_control_volume_flux_balance (Section 16): exact mass-balance
    split of the neck control volume into particle-facing/substrate-facing
    halves.
  - flux_divergence_shape_change_cross_check (Section 17).
  - branch_mu_J_profile (Sections 13-14) on all four branches (particle/
    substrate, top/bottom TJ).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from unified_sinusoidal_campaign import BC_X, BC_Y, build_config  # noqa: E402

from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.ch_crossover_diagnostics import neck_region_mask
from pf_sintering.constrained_eta import constrained_variational_eta_update
from pf_sintering.curvature_extraction import branch_mu_J_profile
from pf_sintering.flux_closure import (
    flux_divergence_shape_change_cross_check,
    neck_control_volume_flux_balance,
    particle_volume_rate_decomposition,
)
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update, surface_flux, surface_mobility_tensor
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact


def substrate_baseline_x(Y, p, wall_mean):
    return wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)


def classify_branches(f, p, tj_xy, v1, v2, wall_mean, probe_len=None):
    from pf_sintering.curvature_extraction import _walk_branch
    probe_len = probe_len or 2.0 * p.interface_width
    devs = []
    for v in (v1, v2):
        path, s_cum = _walk_branch(f, p, tj_xy, v, probe_len)
        if path is None:
            devs.append(-math.inf)
            continue
        baseline_x = substrate_baseline_x(path[:, 1], p, wall_mean)
        devs.append(float(np.mean(path[:, 0] - baseline_x)))
    if devs[0] >= devs[1]:
        return v1, v2
    return v2, v1


def run(dx_nm, args, t_mid):
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    # Milestone 13 Section 8 fix: this file's coordinate arrays are all the
    # UNCENTERED convention (x=(arange(1,Nx+1))*dx); the correct wall
    # position there is wall_frac*Nx*dx, not (wall_frac-0.5)*Nx*dx (which
    # is model.initialize_fields' CENTERED-coordinate formula) -- see
    # scripts/m12b_grid_convergence.py's run_campaign for the full
    # explanation of how this bug was found.
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx

    n_mid = round(t_mid / p.dt)
    for _ in range(n_mid):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
        e1, e2, e3, ediag = constrained_variational_eta_update(e1, e2, e3, f_new, s, p, bc_x=BC_X, bc_y=BC_Y)
        f = f_new

    # one more diagnostic step, keeping all intermediate fields
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
    e1n, e2n, e3n, ediag = constrained_variational_eta_update(e1, e2, e3, f_new, s, p, bc_x=BC_X, bc_y=BC_Y)

    out = dict(dx_nm=dx_nm, t_mid=n_mid * p.dt, dt=p.dt)

    # Section 15
    r15 = particle_volume_rate_decomposition(f, f_new, e1, e2, e3, e1n, e2n, e3n, p.dt, p.dx)
    out["particle_volume_rate"] = r15
    print(f"\n[dx={dx_nm}nm t={out['t_mid']:.4e}s] Section 15 -- particle (V2) rate decomposition:")
    print(f"    dV2/dt total={r15['dV2_dt_total']:.6e}  transport={r15['dV2_dt_transport']:.6e}  "
          f"eta_migration={r15['dV2_dt_eta_migration']:.6e}  closure_residual={r15['closure_residual']:.3e}")

    # Section 16
    sub = compute_subgrid_contact(f, e1, e2, p)
    out["neck_balance"] = {}
    if sub.resolved:
        mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
        r16 = neck_control_volume_flux_balance(diag["Jx"], diag["Jy"], mask, p, wall_mean, BC_X, BC_Y)
        out["neck_balance"] = r16
        print(f"    Section 16 -- neck control volume: dM/dt_particle_half={r16['dM_neck_dt_particle_half']:.6e}  "
              f"dM/dt_substrate_half={r16['dM_neck_dt_substrate_half']:.6e}  total={r16['dM_neck_dt_total']:.6e}")

    # Section 17
    r17 = flux_divergence_shape_change_cross_check(f, f_new, diag["Jx"], diag["Jy"], p.dt, p, BC_X, BC_Y)
    out["shape_change_cross_check"] = r17
    print(f"    Section 17 -- flux-divergence/shape-change sign agreement: {r17['sign_agreement_fraction']:.6f} "
          f"(band_fraction={r17['band_fraction']:.4f})")

    # Sections 13-14: mu(s)/J_s(s) on all four branches (particle/substrate x top/bottom)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)
    out["branch_profiles"] = {}
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            continue
        pd, sdir = classify_branches(f, p, tj.tj_xy, tj.v_s1, tj.v_s2, wall_mean)
        prof_p = branch_mu_J_profile(f, mu, Jx, Jy, p, tj.tj_xy, pd, 150e-9, n_samples=30)
        prof_s = branch_mu_J_profile(f, mu, Jx, Jy, p, tj.tj_xy, sdir, 150e-9, n_samples=30)
        out["branch_profiles"][label] = dict(particle=prof_p, substrate=prof_s)
        if prof_p and prof_s:
            print(f"    [{label} TJ] particle branch: mu(s=0)={prof_p['mu'][0]:.4e} mu(s=far)={prof_p['mu'][-1]:.4e} "
                  f"J_tangent(s=0)={prof_p['J_tangent'][0]:.4e} J_tangent(s=far)={prof_p['J_tangent'][-1]:.4e}")
            print(f"    [{label} TJ] substrate branch: mu(s=0)={prof_s['mu'][0]:.4e} mu(s=far)={prof_s['mu'][-1]:.4e} "
                  f"J_tangent(s=0)={prof_s['J_tangent'][0]:.4e} J_tangent(s=far)={prof_s['J_tangent'][-1]:.4e}")

    return out


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
    ap.add_argument("--dx-list-nm", type=str, default="5.0,2.5")
    ap.add_argument("--t-mid", type=float, default=0.015)
    ap.add_argument("--out", type=str, default="runs/m12b_flux_diagnostics.json")
    args = ap.parse_args()

    dx_list = [float(x) for x in args.dx_list_nm.split(",")]
    out = {}
    for dx_nm in dx_list:
        out[str(dx_nm)] = run(dx_nm, args, args.t_mid)

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
