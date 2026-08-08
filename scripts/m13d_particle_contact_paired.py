"""Milestone 13D Sections 14-19: particle-contact paired legacy/new
comparison.

From the IDENTICAL frozen-eta State-A sinusoidal-substrate initial state
(dx=2.5nm primary, dx=5/1.25nm grid checks), run BOTH `face_flux_mode`s
("cell_average_legacy" and "face_projected") and compare:

  - M_neck_f(t), dM_neck_f/dt (Section 15's primary mass metric -- NOT
    eta-weighted V2, which stays frozen/unused throughout since eta is
    frozen for this entire milestone, Section 20).
  - fixed-Eulerian particle-side mass M_fixed(t) and its exact flux rate.
  - contact width L_contact_TJ_sub(t), TJ positions.
  - the COMPLETE substrate contour x_cross(y) (not just its Fourier
    projection -- Section 17's h(y,t)-h(y,0) spatially resolved response,
    to separate local TJ-induced deformation from the underlying
    sinusoidal flattening tendency).
  - substrate Fourier A1/A2/A3 away from the TJs.
  - substrate-branch mu(s)/J_s(s) profiles.
  - face-normal tangentiality error, mode-appropriate (legacy: the
    Milestone 13C interpolated-face diagnostic; face_projected: the
    Milestone 13D authoritative face vectors directly).
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

from pf_sintering.branch_flux import cartesian_total_boundary_flux, cartesian_total_boundary_flux_from_faces
from pf_sintering.ch_crossover_diagnostics import neck_region_mask
from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.curvature_extraction import branch_mu_J_profile
from pf_sintering.discrete_tangentiality import _region_masks, tangentiality_summary
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import (
    face_projected_tangentiality,
    m_s_ref,
    surface_divergence_update,
    surface_flux,
    surface_flux_face_projected,
    surface_mobility_tensor,
)
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_state_a(args, dx_nm):
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def step_f(f, e1, e2, e3, s, p, M_s, mode):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y,
                                             face_flux_mode=mode)
    return f_new, diag


def flux_fields(f, e1, e2, e3, s, p, M_s, mode):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    if mode == "cell_average_legacy":
        Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
        Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)
        return Jx, Jy
    else:
        fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
        # cell-centered proxy for diagnostics that need a single field (mu(s)/J_s(s)
        # tracing, particle-lobe boundary flux): average the two face families onto
        # cell centers -- NOT the authoritative flux (that stays face-resident and
        # is what the conservative update actually used), only a sampling convenience.
        from pf_sintering.bc_ops import _shift
        Jx_c = 0.5 * (fp["xface"]["Jx"] + _shift(fp["xface"]["Jx"], 1, axis=1, bc=BC_X))
        Jy_c = 0.5 * (fp["yface"]["Jy"] + _shift(fp["yface"]["Jy"], 1, axis=0, bc=BC_Y))
        return Jx_c, Jy_c


def tangentiality_for_mode(f, mu, p, M_s, tj_list, mode):
    if mode == "cell_average_legacy":
        Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
        Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)
        return tangentiality_summary(f, Jx, Jy, p, BC_X, BC_Y, tj_list) if tj_list else None

    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    tang = face_projected_tangentiality(fp)
    if not tj_list:
        return None
    masks = _region_masks(p, tj_list, [1.0, 3.0])
    near1, near3 = masks[1.0], masks[3.0]
    far = ~near3

    def stats(ratio, mag, region=None):
        floor = 1e-6 * np.nanmax(mag)
        valid = np.isfinite(ratio) & (mag > floor)
        if region is not None:
            valid = valid & region
        if not np.any(valid):
            return dict(max=math.nan, rms=math.nan, n=0)
        v = ratio[valid]
        return dict(max=float(np.max(v)), rms=float(np.sqrt(np.mean(v ** 2))), n=int(valid.sum()))

    out = {}
    for axis, rk, mk in (("face_x", "ratio_x", "mag_x"), ("face_y", "ratio_y", "mag_y")):
        r, m = tang[rk], tang[mk]
        out[axis] = dict(global_=stats(r, m), within_1W=stats(r, m, near1),
                          within_3W=stats(r, m, near3), far_from_tj=stats(r, m, far))
    return out


def substrate_x_crossing(f, p):
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    x_cross = np.full(p.Ny, np.nan)
    for i in range(p.Ny):
        row = f[i, :]
        sign = row - 0.5
        idx = np.where(np.diff(np.sign(sign)) != 0)[0]
        if len(idx) == 0:
            continue
        j = idx[-1]  # substrate crossing is the LAST (largest-X) one with a particle also present
        v0, v1 = sign[j], sign[j + 1]
        t = -v0 / (v1 - v0) if (v1 - v0) != 0 else 0.0
        x_cross[i] = x[j] + t * (x[j + 1] - x[j])
    return y, x_cross


def substrate_fourier_away_from_tj(y, x_cross, wavelength, tj_ys, margin, p, n_harmonics=3):
    Ly = p.Ny * p.dx
    away = np.ones(p.Ny, dtype=bool)
    for ty in tj_ys:
        d = np.abs(y - ty)
        d = np.minimum(d, Ly - d)
        away &= d > margin
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


def sample_state(f, e1, e2, e3, s, p, M_s, mode, step, t, x_neck_fixed_mask, wavelength, h0):
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    tj_list, tj_ys = [], []
    if rep.top and rep.top.resolved:
        tj_list.append(rep.top.tj_xy)
        tj_ys.append(rep.top.tj_xy[1])
    if rep.bottom and rep.bottom.resolved:
        tj_list.append(rep.bottom.tj_xy)
        tj_ys.append(rep.bottom.tj_xy[1])

    mu = mu_isotropic(f, e1, e2, e3, s, p)
    tang = tangentiality_for_mode(f, mu, p, M_s, tj_list, mode)
    Jx, Jy = flux_fields(f, e1, e2, e3, s, p, M_s, mode)

    y, x_cross = substrate_x_crossing(f, p)
    h_displacement = None if h0 is None else (x_cross - h0).tolist()
    harmonics = substrate_fourier_away_from_tj(y, x_cross, wavelength, tj_ys, 6 * p.interface_width, p) if tj_ys else [math.nan] * 3

    branch_profiles = {}
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            continue
        from m12b_grid_convergence import classify_branches
        wall_mean = p.substrate_wall_frac * p.Nx * p.dx
        _, substrate_dir = classify_branches(f, p, tj.tj_xy, tj.v_s1, tj.v_s2, wall_mean)
        prof = branch_mu_J_profile(f, mu, Jx, Jy, p, tj.tj_xy, substrate_dir, 150e-9, n_samples=20)
        branch_profiles[label] = prof

    f_after, _ = step_f(f, e1, e2, e3, s, p, M_s, mode)
    M_fixed_before = float((f * x_neck_fixed_mask).sum()) * p.dx * p.dx
    M_fixed_after = float((f_after * x_neck_fixed_mask).sum()) * p.dx * p.dx
    A_fixed = (M_fixed_after - M_fixed_before) / p.dt
    if mode == "cell_average_legacy":
        B_fixed = cartesian_total_boundary_flux(Jx, Jy, x_neck_fixed_mask, p, BC_X, BC_Y)
    else:
        # Jx, Jy above are a cell-centered PROXY (for mu(s)/J_s(s) tracing
        # only) -- re-derive the TRUE face-resident flux here so B_fixed
        # matches what the face_projected update actually integrated
        # (cartesian_total_boundary_flux's own face_average re-interpolation
        # of the proxy does not reproduce it exactly, see branch_flux.py).
        fp_true = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
        B_fixed = cartesian_total_boundary_flux_from_faces(fp_true["Jx_face"], fp_true["Jy_face"],
                                                             x_neck_fixed_mask, p, BC_X, BC_Y)

    M_neck = None
    if sub.resolved:
        neck_mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
        M_neck = float((f * neck_mask).sum()) * p.dx * p.dx

    return dict(
        step=step, time_s=t, tangentiality=tang,
        substrate_harmonics=dict(A1=harmonics[0], A2=harmonics[1], A3=harmonics[2]),
        substrate_h_y=y.tolist(), substrate_h_displacement=h_displacement,
        branch_profiles=branch_profiles,
        M_fixed=M_fixed_before, M_fixed_A=A_fixed, M_fixed_B=B_fixed,
        M_neck_f=M_neck, L_contact_TJ_sub=sub.L_contact_TJ_sub if sub.resolved else math.nan,
        tj_top=tuple(float(v) for v in rep.top.tj_xy) if rep.top and rep.top.resolved else None,
    ), x_cross


def run(dx_nm, args, target_times, mode):
    p, f, e1, e2, e3 = build_state_a(args, dx_nm)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    print(f"=== particle-contact paired, dx={dx_nm}nm mode={mode}: Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s ===")

    sub0 = compute_subgrid_contact(f, e1, e2, p)
    x_neck0 = 0.5 * (sub0.top.x_sub + sub0.bottom.x_sub) if sub0.resolved else None
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    x_neck_fixed_mask = X > x_neck0

    rows = []
    step = 0
    row0, h0 = sample_state(f, e1, e2, e3, s, p, M_s, mode, step, 0.0, x_neck_fixed_mask, p.sinusoid_wavelength, None)
    rows.append(row0)
    print(f"  t=0.000s M_fixed={row0['M_fixed']:.6e} M_neck_f={row0['M_neck_f']:.6e} "
          f"A1_substrate={row0['substrate_harmonics']['A1']*1e9:.3f}nm")

    for t_target in target_times:
        if t_target <= 0.0:
            continue
        n_target = round(t_target / p.dt)
        while step < n_target:
            f, diag = step_f(f, e1, e2, e3, s, p, M_s, mode)
            step += 1
        row, _ = sample_state(f, e1, e2, e3, s, p, M_s, mode, step, step * p.dt, x_neck_fixed_mask,
                               p.sinusoid_wavelength, h0)
        rows.append(row)
        tg = row["tangentiality"]
        tg_str = "" if tg is None else f" face_x_rms(3W)={tg['face_x']['within_3W']['rms']:.4e}"
        print(f"  t={step*p.dt:.4e}s M_fixed={row['M_fixed']:.6e} A={row['M_fixed_A']:.3e} B={row['M_fixed_B']:.3e}  "
              f"M_neck_f={row['M_neck_f']:.6e}  A1_substrate={row['substrate_harmonics']['A1']*1e9:.3f}nm  "
              f"L_contact={row['L_contact_TJ_sub']*1e9:.3f}nm{tg_str}")

    return dict(dx_nm=dx_nm, mode=mode, p_dt=p.dt, x_neck0=x_neck0, wavelength=p.sinusoid_wavelength, rows=rows)


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
    ap.add_argument("--modes", type=str, default="cell_average_legacy,face_projected")
    ap.add_argument("--out", type=str, default="runs/m13d_particle_contact_paired.json")
    args = ap.parse_args()

    target_times = [float(x) for x in args.times.split(",")]
    modes = args.modes.split(",")
    out = {}
    for mode in modes:
        result = run(args.dx_nm, args, target_times, mode)
        out[mode] = result

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
