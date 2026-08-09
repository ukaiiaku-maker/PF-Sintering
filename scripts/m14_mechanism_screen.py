"""Milestone 14: bounded geometry / dihedral / mobility mechanism screen.

Single-mode (face_projected, via the Milestone 13E-promoted canonical
`variational_surface_diffusion_step`) campaign for one physical condition
(amplitude, gamma_gb override, surface_mobility_scale, dx). Eta kinetics,
sink, hazard, RBM, anisotropy, imposed stress/strain all stay OFF -- eta
fields are only used, frozen, for their thermodynamic coupling into mu
(Wc*eta2 groove term) and the TJ/dihedral diagnostic.

Tracks, at each sample time: M_neck_f, L_contact_TJ_sub, fixed-Eulerian
particle-side mass M_fixed (+ its exact conservative boundary-flux rate
B_fixed), measured TJ dihedral angle psi (top/bottom + mean), TJ positions,
substrate contour (x_cross(y)), substrate Fourier A1/A2/A3 away from the
TJs, authoritative face-tangentiality summary, total free energy F, exact
discrete dissipation D_h, and total conserved f mass (for drift audit).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m12b_grid_convergence import classify_branches  # noqa: E402
from m13d_particle_contact_paired import (  # noqa: E402
    substrate_fourier_away_from_tj,
    substrate_x_crossing,
)

from pf_sintering.branch_flux import cartesian_total_boundary_flux_from_faces
from pf_sintering.ch_crossover_diagnostics import neck_region_mask
from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.curvature_extraction import branch_mu_J_profile
from pf_sintering.discrete_tangentiality import _region_masks
from pf_sintering.model import ModelConfig, Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import (
    exact_dissipation_face_projected,
    face_projected_tangentiality,
    m_s_ref,
    surface_flux_face_projected,
    variational_surface_diffusion_step,
)
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact

BC_X, BC_Y = "reflecting", "periodic"


def build_config(args):
    return ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=args.overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
        sinusoid_wavelength=args.wavelength_nm * 1e-9, sinusoid_amplitude=args.amplitude_nm * 1e-9,
        interface_width_override=args.w_nm * 1e-9, eta_diffusivity_fixed_physical=True,
        use_aniso_surface=False, surface_mobility_scale=args.surface_mobility_scale,
        gamma_gb_override=args.gamma_gb_override, dt_override=args.dt_override,
    )


def build_state(args):
    p = build_params(build_config(args))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def step_f(f, e1, e2, e3, s, p, M_s):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = variational_surface_diffusion_step(f, mu, p.dx, p.dt, p.interface_width, M_s,
                                                       bc_x=BC_X, bc_y=BC_Y)
    return f_new, diag


def tangentiality_stats(f, mu, p, M_s, tj_list):
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    tang = face_projected_tangentiality(fp)
    if not tj_list:
        return None, fp
    masks = _region_masks(p, tj_list, [1.0, 3.0])
    near3 = masks[3.0]

    def stats(ratio, mag, region):
        floor = 1e-6 * np.nanmax(mag)
        valid = np.isfinite(ratio) & (mag > floor) & region
        if not np.any(valid):
            return dict(max=math.nan, rms=math.nan, n=0)
        v = ratio[valid]
        return dict(max=float(np.max(v)), rms=float(np.sqrt(np.mean(v ** 2))), n=int(valid.sum()))

    out = {}
    for axis, rk, mk in (("face_x", "ratio_x", "mag_x"), ("face_y", "ratio_y", "mag_y")):
        out[axis] = dict(within_3W=stats(tang[rk], tang[mk], near3))
    return out, fp


def sample_state(f, e1, e2, e3, s, p, M_s, step, t, x_neck_fixed_mask, h0):
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    tj_list, tj_ys, psis = [], [], []
    for tj in (rep.top, rep.bottom):
        if tj is not None and tj.resolved:
            tj_list.append(tj.tj_xy)
            tj_ys.append(tj.tj_xy[1])
            if math.isfinite(tj.psi_deg):
                psis.append(tj.psi_deg)

    mu = mu_isotropic(f, e1, e2, e3, s, p)
    tang, fp = tangentiality_stats(f, mu, p, M_s, tj_list)

    y, x_cross = substrate_x_crossing(f, p)
    h_displacement = None if h0 is None else (x_cross - h0).tolist()
    harmonics = substrate_fourier_away_from_tj(y, x_cross, p.sinusoid_wavelength, tj_ys,
                                                6 * p.interface_width, p) if tj_ys else [math.nan] * 3

    branch_profiles = {}
    Jx_c = 0.5 * (fp["xface"]["Jx"] + np.roll(fp["xface"]["Jx"], 1, axis=1))
    Jy_c = 0.5 * (fp["yface"]["Jy"] + np.roll(fp["yface"]["Jy"], 1, axis=0))
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            continue
        _, substrate_dir = classify_branches(f, p, tj.tj_xy, tj.v_s1, tj.v_s2, wall_mean)
        prof = branch_mu_J_profile(f, mu, Jx_c, Jy_c, p, tj.tj_xy, substrate_dir, 150e-9, n_samples=20)
        if prof is not None:
            branch_profiles[label] = dict(mu_mean=float(np.mean(prof["mu"])),
                                           J_tangent_mean=float(np.mean(prof["J_tangent"])))

    f_after, diag = step_f(f, e1, e2, e3, s, p, M_s)
    M_fixed_before = float((f * x_neck_fixed_mask).sum()) * p.dx * p.dx
    M_fixed_after = float((f_after * x_neck_fixed_mask).sum()) * p.dx * p.dx
    A_fixed = (M_fixed_after - M_fixed_before) / p.dt
    B_fixed = cartesian_total_boundary_flux_from_faces(fp["Jx_face"], fp["Jy_face"], x_neck_fixed_mask,
                                                         p, BC_X, BC_Y)

    M_neck = None
    if sub.resolved:
        neck_mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
        M_neck = float((f * neck_mask).sum()) * p.dx * p.dx

    F = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    Dx, Dy = exact_dissipation_face_projected(fp)
    D_h = (p.dx * p.dx) * (float(np.sum(Dx)) + float(np.sum(Dy)))
    total_mass = float(f.sum()) * p.dx * p.dx

    return dict(
        step=step, time_s=t,
        psi_top_deg=rep.top.psi_deg if rep.top and rep.top.resolved else math.nan,
        psi_bottom_deg=rep.bottom.psi_deg if rep.bottom and rep.bottom.resolved else math.nan,
        psi_mean_deg=float(np.mean(psis)) if psis else math.nan,
        tangentiality=tang,
        substrate_harmonics=dict(A1=harmonics[0], A2=harmonics[1], A3=harmonics[2]),
        substrate_h_y=y.tolist(), substrate_h_displacement=h_displacement,
        branch_profiles=branch_profiles,
        M_fixed=M_fixed_before, M_fixed_A=A_fixed, M_fixed_B=B_fixed,
        M_neck_f=M_neck, L_contact_TJ_sub=sub.L_contact_TJ_sub if sub.resolved else math.nan,
        tj_top=tuple(float(v) for v in rep.top.tj_xy) if rep.top and rep.top.resolved else None,
        tj_bottom=tuple(float(v) for v in rep.bottom.tj_xy) if rep.bottom and rep.bottom.resolved else None,
        F=F, D_h=D_h, total_mass=total_mass,
    ), x_cross


def run(args, target_times):
    p, f, e1, e2, e3 = build_state(args)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    print(f"=== m14 {args.label}: dx={args.dx_nm}nm A={args.amplitude_nm}nm "
          f"gamma_gb={p.gamma_gb:.6f} gamma_gb_ref={p.gamma_gb_ref:.6f} gamma_s={p.gamma_s:.4f} "
          f"surface_mobility_scale={args.surface_mobility_scale} M_f={p.M_f:.4e} M_s={M_s:.4e} "
          f"Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s ===")

    sub0 = compute_subgrid_contact(f, e1, e2, p)
    x_neck0 = 0.5 * (sub0.top.x_sub + sub0.bottom.x_sub) if sub0.resolved else None
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    x_neck_fixed_mask = X > x_neck0

    rows = []
    step = 0
    row0, h0 = sample_state(f, e1, e2, e3, s, p, M_s, step, 0.0, x_neck_fixed_mask, None)
    rows.append(row0)
    print(f"  t=0.000s M_fixed={row0['M_fixed']:.6e} M_neck_f={row0['M_neck_f']:.6e} "
          f"A1_substrate={row0['substrate_harmonics']['A1']*1e9:.3f}nm "
          f"psi_mean={row0['psi_mean_deg']:.2f}deg F={row0['F']:.6e} mass={row0['total_mass']:.6e}")

    for t_target in target_times:
        if t_target <= 0.0:
            continue
        n_target = round(t_target / p.dt)
        while step < n_target:
            f, diag = step_f(f, e1, e2, e3, s, p, M_s)
            step += 1
        row, _ = sample_state(f, e1, e2, e3, s, p, M_s, step, step * p.dt, x_neck_fixed_mask, h0)
        rows.append(row)
        tg = row["tangentiality"]
        tg_str = "" if tg is None else f" face_x_rms(3W)={tg['face_x']['within_3W']['rms']:.4e}"
        print(f"  t={step*p.dt:.4e}s M_fixed={row['M_fixed']:.6e} A={row['M_fixed_A']:.3e} "
              f"B={row['M_fixed_B']:.3e}  M_neck_f={row['M_neck_f']:.6e}  "
              f"A1_substrate={row['substrate_harmonics']['A1']*1e9:.3f}nm  "
              f"L_contact={row['L_contact_TJ_sub']*1e9:.3f}nm psi_mean={row['psi_mean_deg']:.2f}deg "
              f"F={row['F']:.6e} D_h={row['D_h']:.4e} mass_drift={(row['total_mass']-row0['total_mass'])/row0['total_mass']:.2e}{tg_str}")

    return dict(label=args.label, dx_nm=args.dx_nm, amplitude_nm=args.amplitude_nm,
                gamma_gb_override=args.gamma_gb_override, gamma_gb=p.gamma_gb, gamma_gb_ref=p.gamma_gb_ref,
                gamma_s=p.gamma_s, surface_mobility_scale=args.surface_mobility_scale,
                M_f=p.M_f, M_s=M_s, p_dt=p.dt, x_neck0=x_neck0, wavelength=p.sinusoid_wavelength, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", type=str, default="baseline")
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--gamma-gb-override", type=float, default=None)
    ap.add_argument("--dt-override", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--times", type=str, default="0,0.01,0.03,0.06,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--out", type=str, default="runs/m14_mechanism_screen.json")
    args = ap.parse_args()

    target_times = [float(x) for x in args.times.split(",")]
    result = run(args, target_times)

    with open(args.out, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
