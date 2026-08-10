"""Milestone 15: coupled GB migration / surface-reconstruction rate competition.

The first actual COUPLED neck calculation under Milestone 14G's physically
normalized GB energy and calibrated physical GB mobility: f (conserved
surface transport, `variational_surface_diffusion_step`, face_projected)
and eta (GB migration, `constrained_eta.constrained_tangent_cone_eta_update`)
evolve TOGETHER under ONE unified `p` (single free energy: `p.k_eta`,
`p.W_cpl_f` come from `gb_obstacle_coefficients`, same `p.gamma_gb_ref` feeds
both `mu`'s Wc-groove term and eta's structural force -- no p/p_eta
decoupling, no g_external bias). sink, hazard, RBM, anisotropy, imposed
stress/strain stay OFF. No independent M_TJ.

`gb_mobility_m4_J_s` is the physical GB-migration control (Milestone 14G
Section 13): `p.M_eta = pi^2*M_GB/(4*W_GB)`, `M_GB` in m^4/(J*s).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m12b_grid_convergence import classify_branches, substrate_baseline_x  # noqa: E402

from pf_sintering.capillary_stress import (
    apparent_sintering_stress,
    capillary_force_curvature_form,
    capillary_force_endpoint_form,
    force_validation_relative_error,
    oriented_contact_normal,
    substrate_curvature_windows,
    trace_particle_arc,
    window_mean_kappa,
    window_mean_kappa_from_end,
)
from pf_sintering.bc_ops import lap9_bc
from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update, local_wc
from pf_sintering.curvature_extraction import branch_mu_J_profile
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta
from pf_sintering.model import ModelConfig, Sink, build_params, initialize_fields, lap9, reproject
from pf_sintering.surface_transport import m_s_ref, surface_flux_face_projected, variational_surface_diffusion_step
from pf_sintering.tj_force import compute_tj_force, locate_neck_tjs
from pf_sintering.tj_subgrid import compute_subgrid_contact

BC_X, BC_Y = "reflecting", "periodic"

# The Milestone 14E-14G "historical reference" eta mobility, mapped through
# the Milestone 14G obstacle calibration's M_GB=4*M_eta*W_GB/pi^2 to a
# calibrated reference PHYSICAL GB mobility (Section 5). NOT a claimed
# experimental material mobility.
M_ETA_HISTORICAL_REF = 4.266666666666666e-09


def build_config(dx_nm, W_nm=20.0, gamma_gb=1.0, M_GB=None, surface_mobility_scale=0.3,
                  A_nm=100.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=20.0,
                  seed=42, dt_override=None, use_aniso_surface=False, aniso_delta=None,
                  theta_mis_deg=0.0):
    # use_aniso_surface/aniso_delta/theta_mis_deg: Milestone 15F Section 4-6.
    # Defaults (False/None/0.0) reproduce this function's exact pre-15F
    # behavior for every existing (M15/M15E) call site. theta_mis_deg is
    # ONLY the crystal-orientation reference angle here (p.theta_grain[1])
    # since gamma_gb is always passed as an explicit override -- it does
    # NOT also select gamma_gb via the empirical misorientation curve the
    # way it would with gamma_gb_override left at None.
    return ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", dx=dx_nm * 1e-9, r2=R2_nm * 1e-9,
        aspect_ratio=aspect_ratio, contact_orientation="short_plane", initial_overlap=overlap_nm * 1e-9,
        t_total=1e-6, sinusoid_wavelength=lambda_nm * 1e-9, sinusoid_amplitude=A_nm * 1e-9,
        interface_width_override=W_nm * 1e-9, use_aniso_surface=use_aniso_surface, gamma_gb_override=gamma_gb,
        gb_mobility_m4_J_s=M_GB, surface_mobility_scale=surface_mobility_scale, seed=seed,
        dt_override=dt_override, aniso_delta=aniso_delta, theta_mis_deg=theta_mis_deg,
    )


def build_state(**kwargs):
    p = build_params(build_config(**kwargs))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def energy_ledger(f, e1, e2, e3, s, p):
    """E_surface (pure f/CH double-well+gradient), E_coupling (Wc*eta2*
    (f^2/2-f), the bulk term coupling f-ownership to eta), E_GB (pure
    (k_eta/2)*sum|grad(eta_i)|^2, via the lap9-self-adjoint form -- the
    discrete variational conjugate `structural_thermodynamic_force`
    actually uses, NOT a naive central-difference |grad|^2, which is not
    the discrete adjoint of lap9 and would misstate this term by ~47%
    (Milestone 12B finding)). F_total = sum of the three, reproducing
    ch_exact_energy.exact_free_energy_isotropic (E_surface+E_coupling)
    plus the E_GB term that function omits."""
    fb = np.clip(f, 0.0, 1.0)
    Wc = local_wc(f, e1, e2, e3, s, p)
    eta2 = e1 * e1 + e2 * e2 + e3 * e3
    E_surface = float(np.sum(0.5 * p.W_f * f * f * (1 - f) ** 2 - 0.5 * p.k_f * f * lap9(f, p.dx))) * p.dx * p.dx
    E_coupling = float(np.sum(Wc * eta2 * (0.5 * fb * fb - fb))) * p.dx * p.dx
    grad_term = e1 * lap9(e1, p.dx) + e2 * lap9(e2, p.dx)
    if p.use_eta3:
        grad_term = grad_term + e3 * lap9(e3, p.dx)
    E_GB = float(np.sum(-0.5 * p.k_eta * grad_term)) * p.dx * p.dx
    return dict(E_surface=E_surface, E_coupling=E_coupling, E_GB=E_GB, F_total=E_surface + E_coupling + E_GB)


def gb_excess_energy(f, e1, e2, e3, s, p, bc_x=BC_X, bc_y=BC_Y):
    """Milestone 15B Section 11: the PHYSICAL GB interfacial excess energy
    (obstacle-potential excess + eta-gradient energy), background-
    subtracted against the LOCAL single-grain-at-the-same-f reference (not
    just the raw energy_ledger's E_GB+E_coupling, which include a nonzero
    single-grain "background" -- e.g. the coupling term's own bulk value
    at f=1, single grain, is -0.5*Wc*f^2, not zero).

    Derivation: with e_i=f (only grain i present, no GB), eta2=sum(e_i^2)=
    f^2, so the coupling-term background density is Wc*f^2*(0.5f^2-f).
    Subtracting this from the general two-grain coupling density
    Wc*(e1^2+e2^2)*(0.5f^2-f), and using e1+e2=f (Milestone 15B Section 2's
    now-exact identity) to eliminate f^2-(e1^2+e2^2)=2*e1*e2, gives the
    EXACT excess coupling density Wc*e1*e2*f*(2-f) -- reducing exactly to
    Wc*phi*(1-phi) at f=1 (phi=e2), matching gb_obstacle_energy's own
    planar F_GB formula. The gradient term's background (single grain,
    e_i=f) is -0.5*k_eta*f*lap9_bc(f) (zero wherever f is locally uniform,
    but nonzero at a free surface where f itself has a gradient, even with
    no second grain present at all -- this term correctly removes that
    spurious contribution). Uses `lap9_bc` (not the periodic-only `lap9`
    energy_ledger uses for the DYNAMICS' own -- correct, since production
    eta evolution literally uses periodic-X `lap9` -- coefficients) with
    the SAME bc_x/bc_y the coupled trajectory's surface/eta steps use, so
    this diagnostic's boundary treatment matches what the field actually
    saw; using plain periodic `lap9` here is WRONG whenever the physical
    domain is not periodic in that direction (confirmed by hand: it
    diverges as dx->0 on a one-sided planar test, ~4x too large at
    dx=2.5nm growing to ~80x at dx=0.1nm, because a periodic Laplacian
    assumes phi wraps back to matching values at the domain edges)."""
    Wc = local_wc(f, e1, e2, e3, s, p)
    excess_coupling = Wc * e1 * e2 * f * (2.0 - f)
    grad_term = e1 * lap9_bc(e1, p.dx, bc_x=bc_x, bc_y=bc_y) + e2 * lap9_bc(e2, p.dx, bc_x=bc_x, bc_y=bc_y)
    bg_grad_term = f * lap9_bc(f, p.dx, bc_x=bc_x, bc_y=bc_y)
    excess_grad = -0.5 * p.k_eta * (grad_term - bg_grad_term)
    return float(np.sum(excess_grad + excess_coupling)) * p.dx * p.dx


def sample_state(f, e1, e2, e3, s, p, step, t, mass0, F0, cum_safety_correction, cum_variational_change):
    out = dict(step=step, t=t, mass=float(f.sum()) * p.dx * p.dx)
    out["mass_drift"] = (out["mass"] - mass0) / mass0
    led = energy_ledger(f, e1, e2, e3, s, p)
    out.update(led)
    out["dF"] = led["F_total"] - F0
    out["cum_safety_correction"] = cum_safety_correction
    out["cum_variational_change"] = cum_variational_change
    out["cum_safety_ratio"] = cum_safety_correction / (cum_variational_change + 1e-300)

    tjs = locate_neck_tjs(f, e1, e2, p)
    out["tj_resolved"] = tjs is not None
    if tjs is None:
        return out
    tj_top, tj_bottom = tjs["tj_top"], tjs["tj_bottom"]
    out["tj_top_xy"] = [float(v) for v in tj_top]
    out["tj_bottom_xy"] = [float(v) for v in tj_bottom]

    top = compute_tj_force(f, e1, e2, tj_top, s, p)
    bottom = compute_tj_force(f, e1, e2, tj_bottom, s, p)
    out["top_resolved"], out["bottom_resolved"] = bool(top.resolved), bool(bottom.resolved)
    if top.resolved:
        out["psi_top_deg"], out["F_TJ_mag_top"] = top.psi_deg, top.F_TJ_mag
        out["F_TJ_vec_top"] = [float(v) for v in top.F_TJ]
    if bottom.resolved:
        out["psi_bottom_deg"], out["F_TJ_mag_bottom"] = bottom.psi_deg, bottom.F_TJ_mag
        out["F_TJ_vec_bottom"] = [float(v) for v in bottom.F_TJ]

    sub = compute_subgrid_contact(f, e1, e2, p)
    out["subgrid_resolved"] = bool(sub.resolved)
    if sub.resolved:
        out["L_contact"] = sub.L_contact_TJ_sub
        out["L_GB"] = sub.L_GB_geom_sub
        out["gamma_GB_L_GB"] = p.gamma_gb * sub.L_GB_geom_sub
        out["tj_top_xy_sub"] = [sub.top.x_sub, sub.top.y_sub]
        out["tj_bottom_xy_sub"] = [sub.bottom.x_sub, sub.bottom.y_sub]

    if not (top.resolved and bottom.resolved):
        return out
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx
    top_particle_dir, top_substrate_dir = classify_branches(f, p, tj_top, top.v_s1, top.v_s2, wall_mean)
    bot_particle_dir, bot_substrate_dir = classify_branches(f, p, tj_bottom, bottom.v_s1, bottom.v_s2, wall_mean)
    out["top_particle_tangent"] = [float(v) for v in top_particle_dir]
    out["bot_particle_tangent"] = [float(v) for v in bot_particle_dir]

    arc = trace_particle_arc(f, p, tj_top, tj_bottom, top_particle_dir, bot_particle_dir)
    out["arc_resolved"] = bool(arc.get("resolved", False))
    L_surface = None
    if arc["resolved"]:
        L_surface = arc["arc_length"]
        out["arc_length"] = arc["arc_length"]
        F_curv = capillary_force_curvature_form(arc, p.gamma_s)
        F_ep = capillary_force_endpoint_form(top_particle_dir, bot_particle_dir, p.gamma_s)
        out["F_cap_curvature_form"] = list(F_curv)
        out["F_cap_endpoint_form"] = list(F_ep)
        out["force_rel_err"] = force_validation_relative_error(F_curv, F_ep)
        out["kappa_arc_max_abs"] = float(np.max(np.abs(arc["kappa"])))

        if sub.resolved and math.isfinite(sub.L_contact_TJ_sub):
            n_GB = oriented_contact_normal(sub.n_GB_sub)
            sigma_curv, Fn_curv = apparent_sintering_stress(F_curv, n_GB, sub.L_contact_TJ_sub)
            sigma_ep, Fn_ep = apparent_sintering_stress(F_ep, n_GB, sub.L_contact_TJ_sub)
            out["sigma_sint_app_curvature_form"] = sigma_curv
            out["sigma_sint_app_endpoint_form"] = sigma_ep
            out["F_cap_n_curvature_form"], out["F_cap_n_endpoint_form"] = Fn_curv, Fn_ep
            # Milestone 15D Section 2: exact isotropic upper bound.
            # F_cap_endpoint=-gamma_s*(t_top+t_bottom), |t_i|=1, so
            # |F_cap_endpoint|<=2*gamma_s (equality only at t_top=t_bottom,
            # i.e. psi=180deg, a perfectly flat/open neck) and
            # |F_cap_n|<=|F_cap_endpoint|<=2*gamma_s always -- an exact,
            # geometry-dependent ceiling on sigma_sint_endpoint, independent
            # of any dynamics.
            sigma_ep_max = 2.0 * p.gamma_s / sub.L_contact_TJ_sub
            out["sigma_sint_endpoint_max"] = sigma_ep_max
            out["sigma_sint_endpoint_over_max"] = abs(sigma_ep) / sigma_ep_max if sigma_ep_max > 0 else float("nan")

        W = p.interface_width
        p_gamma = {}
        for lo, hi in ((1.5, 3.0), (2.0, 4.0), (3.0, 5.0)):
            key = f"{lo}W-{hi}W"
            kt = window_mean_kappa(arc, lo * W, hi * W)
            kb = window_mean_kappa_from_end(arc, lo * W, hi * W)
            p_gamma[key] = dict(kappa_top=kt, kappa_bottom=kb,
                                 p_gamma_top=p.gamma_s * kt, p_gamma_bottom=p.gamma_s * kb)
        out["p_gamma_particle_windows"] = p_gamma

    sub_curv = {}
    for label, tj_xy, branch_dir in (("top", tj_top, top_substrate_dir), ("bottom", tj_bottom, bot_substrate_dir)):
        sub_curv[label] = substrate_curvature_windows(f, e1, e2, e3, s, p, tj_xy, branch_dir,
                                                        windows=((1.5, 3.0), (3.0, 5.0)))
    out["substrate_curvature_windows"] = sub_curv

    if L_surface is not None:
        out["gamma_s_L_surface"] = p.gamma_s * L_surface

    return out


def profile_mu_J(f, e1, e2, e3, s, p, tj_top, tj_bottom, top_particle_dir, bot_particle_dir):
    """Section 10: mu(s), J_tangent(s), J_normal(s), kappa(s) along the
    particle-side branch from each TJ, using the SAME mu/flux the
    production f-step actually consumes."""
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    M_s = getattr(p, "_M_s_cache", None)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    Jx = 0.5 * (fp["Jx_face"] + np.roll(fp["Jx_face"], 1, axis=1))
    Jy = np.zeros_like(Jx)
    Jy[1:] = 0.5 * (fp["Jy_face"][1:] + fp["Jy_face"][:-1])
    out = {}
    for label, tj_xy, branch_dir in (("top", tj_top, top_particle_dir), ("bottom", tj_bottom, bot_particle_dir)):
        out[label] = branch_mu_J_profile(f, mu, Jx, Jy, p, tj_xy, branch_dir,
                                          max_arclength=6.0 * p.interface_width)
    return out


def run_trajectory(dx_nm, M_GB, surface_mobility_scale, t_target, sample_times, label,
                    W_nm=20.0, gamma_gb=1.0, A_nm=100.0, lambda_nm=320.0, R2_nm=80.0,
                    aspect_ratio=2.0, overlap_nm=20.0, save_profile_at=None, verbose=True,
                    dt_override=None, use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0):
    p, f, e1, e2, e3 = build_state(dx_nm=dx_nm, W_nm=W_nm, gamma_gb=gamma_gb, M_GB=M_GB,
                                    surface_mobility_scale=surface_mobility_scale, A_nm=A_nm,
                                    lambda_nm=lambda_nm, R2_nm=R2_nm, aspect_ratio=aspect_ratio,
                                    overlap_nm=overlap_nm, dt_override=dt_override,
                                    use_aniso_surface=use_aniso_surface, aniso_delta=aniso_delta,
                                    theta_mis_deg=theta_mis_deg)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    p._M_s_cache = M_s
    dt = p.dt
    n_target = round(t_target / dt)
    sample_steps = sorted(set(round(ts / dt) for ts in sample_times if ts <= t_target) | {0, n_target})

    mass0 = float(f.sum()) * p.dx * p.dx
    F0 = energy_ledger(f, e1, e2, e3, s, p)["F_total"]
    if verbose:
        print(f"=== {label}: dx={dx_nm}nm M_GB={M_GB} M_s={M_s:.4e} dt={dt:.4e}s "
              f"n_target={n_target} Nx={p.Nx} Ny={p.Ny} M_eta={p.M_eta:.4e} ===")

    rows = []
    cum_safety = 0.0
    cum_variational = 0.0
    max_instant_safety_frac = 0.0
    saved_profiles = {}
    t0_wall = time.time()

    def do_sample(step, t):
        row = sample_state(f, e1, e2, e3, s, p, step, t, mass0, F0, cum_safety, cum_variational)
        rows.append(row)
        if verbose:
            psi_t = row.get("psi_top_deg", math.nan)
            sig = row.get("sigma_sint_app_curvature_form", math.nan)
            lgb = row.get("L_GB", math.nan)
            print(f"  step={step} t={t:.4e}s F={row['F_total']:.6e} dF={row['dF']:.3e} "
                  f"mass_drift={row['mass_drift']:.2e} psi_top={psi_t:.2f} L_GB={lgb*1e9 if math.isfinite(lgb) else float('nan'):.2f}nm "
                  f"sigma_app={sig:.4e} cum_safety_ratio={row['cum_safety_ratio']:.3e}")
        if save_profile_at is not None and step in save_profile_at and row.get("tj_resolved") and row.get("top_resolved") and row.get("bottom_resolved"):
            try:
                tjs = locate_neck_tjs(f, e1, e2, p)
                top = compute_tj_force(f, e1, e2, tjs["tj_top"], s, p)
                bottom = compute_tj_force(f, e1, e2, tjs["tj_bottom"], s, p)
                wall_mean = p.substrate_wall_frac * p.Nx * p.dx
                tpd, tsd = classify_branches(f, p, tjs["tj_top"], top.v_s1, top.v_s2, wall_mean)
                bpd, bsd = classify_branches(f, p, tjs["tj_bottom"], bottom.v_s1, bottom.v_s2, wall_mean)
                saved_profiles[step] = dict(
                    f=f.copy(), e1=e1.copy(), e2=e2.copy(), e3=e3.copy(),
                    mu_J=profile_mu_J(f, e1, e2, e3, s, p, tjs["tj_top"], tjs["tj_bottom"], tpd, bpd),
                )
            except Exception as exc:  # pragma: no cover -- diagnostic-only save path
                if verbose:
                    print(f"    (profile save at step {step} failed: {exc})")

    do_sample(0, 0.0)
    step = 0
    for target in sample_steps:
        if target == 0:
            continue
        while step < target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, fdiag = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s,
                                                            bc_x=BC_X, bc_y=BC_Y)
            e1, e2, e3, ediag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                                      bc_x=BC_X, bc_y=BC_Y)
            cum_safety += ediag["safety_correction"]
            cum_variational += ediag["variational_change"]
            max_instant_safety_frac = max(max_instant_safety_frac, ediag["safety_fraction"])
            step += 1
        do_sample(step, step * dt)

    wall = time.time() - t0_wall
    if verbose:
        print(f"  ({label} done in {wall:.1f}s wall)")

    return dict(label=label, dx_nm=dx_nm, M_GB=M_GB, M_s=M_s, dt=dt, M_eta=p.M_eta, Nx=p.Nx, Ny=p.Ny,
                n_target=n_target, mass0=mass0, F0=F0, rows=rows,
                max_instant_safety_frac=max_instant_safety_frac,
                final_cum_safety_ratio=cum_safety / (cum_variational + 1e-300),
                wall_seconds=wall), saved_profiles


def _strip_for_json(result):
    out = dict(result)
    out["rows"] = [{k: v for k, v in r.items()} for r in result["rows"]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--t-target", type=float, default=0.25)
    ap.add_argument("--sample-times", type=str,
                     default="0,0.002,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25")
    ap.add_argument("--cases", type=str, default="frozen,slow,ref,fast")
    ap.add_argument("--M-GB-scales", type=str, default="0,0.1,1.0,10.0")
    args = ap.parse_args()

    sample_times = [float(x) for x in args.sample_times.split(",")]
    case_names = args.cases.split(",")
    scales = [float(x) for x in args.M_GB_scales.split(",")]
    M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
    print(f"M_GB_ref = {M_GB_ref:.6e} m^4/(J*s)")

    out = {"M_GB_ref": M_GB_ref}
    for name, scale in zip(case_names, scales):
        M_GB = 0.0 if scale == 0 else scale * M_GB_ref  # 0.0 (not None) so gb_mobility_m4_J_s actually freezes M_eta=0
        result, _profiles = run_trajectory(args.dx_nm, M_GB, args.surface_mobility_scale, args.t_target,
                                            sample_times, name)
        out[name] = _strip_for_json(result)

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
