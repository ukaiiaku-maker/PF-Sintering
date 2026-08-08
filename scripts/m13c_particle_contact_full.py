"""Milestone 13C Sections 4-5, 10-12, 15-16: particle-contact repeat with
discrete tangentiality, substrate-shape, substrate-flux, and a corrected
(fixed-Eulerian) particle-lobe control volume, tracked continuously over
a long finite-time trajectory (not endpoints).

Same frozen-eta State-A sinusoidal-substrate contact as Milestones 13/13B
(dx=2.5nm primary). At each of the requested times:

  - cell- and face-centered J.n tangentiality (pf_sintering.
    discrete_tangentiality), globally and within 1W/3W of each TJ
    (Sections 4-5).
  - substrate free-surface shape AWAY from the particle-contact
    perturbation (Y rows further than a margin from either TJ), via the
    same Fourier-harmonic decomposition as the isolated-substrate test
    (Section 11).
  - substrate-branch mu(s)/J_s(s) profiles from each TJ toward the far
    substrate (Section 12).
  - a FIXED EULERIAN particle-lobe control volume (Section 15 option A):
    the boundary X=x_neck is fixed at its t=0 value for the whole
    trajectory (not recomputed as the TJ moves), so its own flux closure
    needs no Reynolds boundary-motion correction (v_b=0 identically) --
    M_fixed(t) and its exact flux closure are reported directly.
  - M_neck_f(t) (unchanged from Milestone 13/13B).
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
from m13b_branch_flux_closure import current_flux_fields, step_f_only  # noqa: E402

from pf_sintering.branch_flux import cartesian_total_boundary_flux
from pf_sintering.ch_crossover_diagnostics import neck_region_mask
from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.curvature_extraction import branch_mu_J_profile
from pf_sintering.discrete_tangentiality import tangentiality_summary
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_state_a(args, dx_nm):
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def substrate_x_crossing_away_from_tj(f, p, tj_ys, margin):
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    x_cross = np.full(p.Ny, np.nan)
    for i in range(p.Ny):
        row = f[i, :]
        sign = row - 0.5
        idx = np.where(np.diff(np.sign(sign)) != 0)[0]
        if len(idx) == 0:
            continue
        j = idx[-1]  # substrate crossing is the LAST (largest-X) one when a particle is also present
        v0, v1 = sign[j], sign[j + 1]
        t = -v0 / (v1 - v0) if (v1 - v0) != 0 else 0.0
        x_cross[i] = x[j] + t * (x[j + 1] - x[j])
    Ly = p.Ny * p.dx
    away = np.ones(p.Ny, dtype=bool)
    for ty in tj_ys:
        d = np.abs(y - ty)
        d = np.minimum(d, Ly - d)
        away &= d > margin
    return y, x_cross, away


def substrate_fourier_harmonics(f, p, wavelength, tj_ys, margin, n_harmonics=3):
    y, x_cross, away = substrate_x_crossing_away_from_tj(f, p, tj_ys, margin)
    valid = np.isfinite(x_cross) & away
    if valid.sum() < 6:
        return [math.nan] * n_harmonics
    cols = [np.ones(valid.sum())]
    for n in range(1, n_harmonics + 1):
        cols.append(np.cos(n * 2 * math.pi * y[valid] / wavelength))
        cols.append(np.sin(n * 2 * math.pi * y[valid] / wavelength))
    A = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(A, x_cross[valid], rcond=None)
    return [float(math.hypot(coeffs[1 + 2 * n], coeffs[2 + 2 * n])) for n in range(n_harmonics)]


def sample_state(f, e1, e2, e3, s, p, M_s, step, t, x_neck_fixed_mask, wavelength):
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    tj_list = []
    tj_ys = []
    if rep.top and rep.top.resolved:
        tj_list.append(rep.top.tj_xy)
        tj_ys.append(rep.top.tj_xy[1])
    if rep.bottom and rep.bottom.resolved:
        tj_list.append(rep.bottom.tj_xy)
        tj_ys.append(rep.bottom.tj_xy[1])

    Jx, Jy = current_flux_fields(f, e1, e2, e3, s, p, M_s)
    tang = tangentiality_summary(f, Jx, Jy, p, BC_X, BC_Y, tj_list) if tj_list else None

    W = p.interface_width
    harmonics = substrate_fourier_harmonics(f, p, wavelength, tj_ys, margin=6 * W) if tj_ys else [math.nan] * 3

    branch_profiles = {}
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            continue
        from m12b_grid_convergence import classify_branches
        wall_mean = p.substrate_wall_frac * p.Nx * p.dx
        _, substrate_dir = classify_branches(f, p, tj.tj_xy, tj.v_s1, tj.v_s2, wall_mean)
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        prof = branch_mu_J_profile(f, mu, Jx, Jy, p, tj.tj_xy, substrate_dir, 150e-9, n_samples=20)
        branch_profiles[label] = prof

    f_after, _ = step_f_only(f, e1, e2, e3, s, p, M_s)
    M_fixed_before = float((f * x_neck_fixed_mask).sum()) * p.dx * p.dx
    M_fixed_after = float((f_after * x_neck_fixed_mask).sum()) * p.dx * p.dx
    A_fixed = (M_fixed_after - M_fixed_before) / p.dt
    B_fixed = cartesian_total_boundary_flux(Jx, Jy, x_neck_fixed_mask, p, BC_X, BC_Y)

    M_neck = None
    if sub.resolved:
        neck_mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
        M_neck = float((f * neck_mask).sum()) * p.dx * p.dx

    return dict(
        step=step, time_s=t,
        tangentiality=tang,
        substrate_harmonics=dict(A1=harmonics[0], A2=harmonics[1], A3=harmonics[2]),
        branch_profiles=branch_profiles,
        M_fixed=M_fixed_before, M_fixed_A=A_fixed, M_fixed_B=B_fixed,
        M_neck_f=M_neck,
        L_contact_TJ_sub=sub.L_contact_TJ_sub if sub.resolved else math.nan,
    )


def run(dx_nm, args, target_times):
    p, f, e1, e2, e3 = build_state_a(args, dx_nm)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    print(f"=== particle-contact full trajectory, dx={dx_nm}nm: Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s ===")

    sub0 = compute_subgrid_contact(f, e1, e2, p)
    x_neck0 = 0.5 * (sub0.top.x_sub + sub0.bottom.x_sub) if sub0.resolved else None
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    x_neck_fixed_mask = X > x_neck0

    rows = []
    step = 0
    row0 = sample_state(f, e1, e2, e3, s, p, M_s, step, 0.0, x_neck_fixed_mask, p.sinusoid_wavelength)
    rows.append(row0)
    print(f"  t=0.000s M_fixed={row0['M_fixed']:.6e} M_neck_f={row0['M_neck_f']:.6e} "
          f"A1_substrate={row0['substrate_harmonics']['A1']*1e9:.3f}nm")

    for t_target in target_times:
        if t_target <= 0.0:
            continue
        n_target = round(t_target / p.dt)
        while step < n_target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        row = sample_state(f, e1, e2, e3, s, p, M_s, step, step * p.dt, x_neck_fixed_mask, p.sinusoid_wavelength)
        rows.append(row)
        tg = row["tangentiality"]
        print(f"  t={step*p.dt:.4e}s M_fixed={row['M_fixed']:.6e} A={row['M_fixed_A']:.3e} B={row['M_fixed_B']:.3e}  "
              f"M_neck_f={row['M_neck_f']:.6e}  A1_substrate={row['substrate_harmonics']['A1']*1e9:.3f}nm  "
              f"face_x_rms(3W)={tg['face_x']['within_3W']['rms']:.4f} face_x_rms(far)={tg['face_x']['far_from_tj']['rms']:.4f}")

    return dict(dx_nm=dx_nm, p_dt=p.dt, x_neck0=x_neck0, wavelength=p.sinusoid_wavelength, rows=rows)


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
    ap.add_argument("--times", type=str, default="0,0.03,0.06,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--out", type=str, default="runs/m13c_particle_contact_full.json")
    args = ap.parse_args()

    target_times = [float(x) for x in args.times.split(",")]
    result = run(args.dx_nm, args, target_times)

    with open(args.out, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
