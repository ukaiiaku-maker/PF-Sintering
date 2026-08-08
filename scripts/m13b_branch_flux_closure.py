"""Milestone 13B: branch-resolved surface-flux closure.

Runs the SAME frozen-eta State-A sinusoidal-substrate trajectory as
Milestone 13's scripts/m13_frozen_eta_campaign.py (dx=2.5nm), and at each
of the 9 requested checkpoint times performs:

  - TJ-core radius characterization (Section 6 investigation): the exact
    Cartesian boundary flux (pf_sintering.branch_flux.
    cartesian_total_boundary_flux) of a disk centered at each TJ, at a
    sequence of small radii, to locate WHERE the net flux accumulation is
    actually concentrated.
  - Three-way closure test (A: finite-difference of the disk's own f-mass
    over one extra short step; B: the same Cartesian boundary flux; C: the
    naive two-branch-cut estimate, pf_sintering.branch_flux.
    branch_flux_balance_at_tj) at s_cut = 1.5W, 2W, 2.5W, 3W.
  - Q_p(t), Q_s(t), R_flux(t) = Q_s/Q_p at each tested s_cut (Section 7).
  - Geometric particle-lobe f-mass (Section 8-9): a half-plane region
    bounded by the instantaneous TJ pair's shared neck column (X > x_neck,
    the same physical GB/TJ-connection Section 9 asks for -- in this
    geometry the two TJs share essentially one X coordinate, so the
    "chord connecting the TJs" is simply the vertical line X=x_neck), with
    its own independent boundary-flux closure (this mask's only internal
    boundary is that single line, so cartesian_total_boundary_flux applies
    directly and unambiguously).

Preserves Milestone 13's explicit constraints: eta frozen (never updated,
but kept in mu_f), sinusoidal geometry/R2/wavelength/amplitude/overlap/W/
surface-mobility-scale unchanged, sink/hazard/RBM off throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from unified_sinusoidal_campaign import BC_X, BC_Y, build_config  # noqa: E402

from pf_sintering.branch_flux import (
    branch_flux_balance_at_tj,
    cartesian_total_boundary_flux,
    disk_control_volume_mask,
)
from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update, surface_flux, surface_mobility_tensor
from pf_sintering.tj_force import compute_neck_tj_forces


def build_state_a(args, dx_nm):
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def step_f_only(f, e1, e2, e3, s, p, M_s):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
    return f_new, diag


def current_flux_fields(f, e1, e2, e3, s, p, M_s):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)
    return Jx, Jy


def tj_core_radius_scan(f, Jx, Jy, p, tj_xy, radii_in_W):
    W = p.interface_width
    out = {}
    for rf in radii_in_W:
        r = rf * W
        mask = disk_control_volume_mask(p, tj_xy, r)
        B = cartesian_total_boundary_flux(Jx, Jy, mask, p, BC_X, BC_Y)
        out[str(rf)] = dict(radius_m=r, B=B, n_cells=int(mask.sum()))
    return out


def three_way_closure(f, e1, e2, e3, s, p, M_s, tj_xy, particle_dir, substrate_dir, s_cuts_in_W):
    W = p.interface_width
    Jx, Jy = current_flux_fields(f, e1, e2, e3, s, p, M_s)
    f_after, _ = step_f_only(f, e1, e2, e3, s, p, M_s)

    out = {}
    for sf in s_cuts_in_W:
        s_cut = sf * W
        mask = disk_control_volume_mask(p, tj_xy, s_cut)
        A = float(((f_after - f) * mask).sum()) * p.dx * p.dx / p.dt
        B = cartesian_total_boundary_flux(Jx, Jy, mask, p, BC_X, BC_Y)
        halfwidths = [0.5 * W, 1.0 * W, 1.5 * W]
        res = branch_flux_balance_at_tj(f, Jx, Jy, p, tj_xy, particle_dir, substrate_dir, s_cut, halfwidths)
        C = None if res is None else res["net_by_halfwidth"]
        out[str(sf)] = dict(s_cut=s_cut, A=A, B=B, C_by_halfwidth=C,
                             Q_p=None if res is None else res["Q_p"],
                             Q_s=None if res is None else res["Q_s"])
    return out


def particle_lobe_mass_and_closure(f, e1, e2, e3, s, p, M_s, x_neck):
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    mask = X > x_neck

    Jx, Jy = current_flux_fields(f, e1, e2, e3, s, p, M_s)
    f_after, _ = step_f_only(f, e1, e2, e3, s, p, M_s)

    M0 = float((f * mask).sum()) * p.dx * p.dx
    M1 = float((f_after * mask).sum()) * p.dx * p.dx
    A = (M1 - M0) / p.dt
    B = cartesian_total_boundary_flux(Jx, Jy, mask, p, BC_X, BC_Y)
    return dict(x_neck=x_neck, M_particle_geom=M0, A_finite_diff=A, B_cartesian=B,
                closure_residual=A - B, n_cells=int(mask.sum()))


def sample_at_current_state(f, e1, e2, e3, s, p, M_s, step, time_s):
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    W = p.interface_width
    row = dict(step=step, time_s=time_s)

    Jx, Jy = current_flux_fields(f, e1, e2, e3, s, p, M_s)

    tj_results = {}
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            tj_results[label] = None
            continue
        core_scan = tj_core_radius_scan(f, Jx, Jy, p, tj.tj_xy, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
        closure = three_way_closure(f, e1, e2, e3, s, p, M_s, tj.tj_xy, tj.v_s1, tj.v_s2, [1.5, 2.0, 2.5, 3.0])
        tj_results[label] = dict(tj_xy=tuple(float(v) for v in tj.tj_xy),
                                  core_scan=core_scan, closure=closure)
    row["tj"] = tj_results

    if rep.top and rep.top.resolved and rep.bottom and rep.bottom.resolved:
        x_neck = 0.5 * (rep.top.tj_xy[0] + rep.bottom.tj_xy[0])
        row["particle_lobe"] = particle_lobe_mass_and_closure(f, e1, e2, e3, s, p, M_s, x_neck)
    else:
        row["particle_lobe"] = None

    return row


def run(dx_nm, args, target_times):
    p, f, e1, e2, e3 = build_state_a(args, dx_nm)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    print(f"=== Milestone 13B branch-flux closure, dx={dx_nm}nm ===")
    print(f"  Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s W={p.interface_width*1e9:.1f}nm")

    rows = []
    step = 0
    row0 = sample_at_current_state(f, e1, e2, e3, s, p, M_s, step, 0.0)
    rows.append(row0)
    print(f"  t=0.000s sampled")

    for t_target in target_times:
        if t_target <= 0.0:
            continue
        n_target = round(t_target / p.dt)
        while step < n_target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        row = sample_at_current_state(f, e1, e2, e3, s, p, M_s, step, step * p.dt)
        rows.append(row)
        print(f"  t={step*p.dt:.4e}s sampled (step={step})")

    return dict(dx_nm=dx_nm, p_dt=p.dt, rows=rows)


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
    ap.add_argument("--times", type=str, default="0,0.015,0.03,0.06,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--out", type=str, default="runs/m13b_branch_flux_closure.json")
    args = ap.parse_args()

    target_times = [float(x) for x in args.times.split(",")]
    result = run(args.dx_nm, args, target_times)

    with open(args.out, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
