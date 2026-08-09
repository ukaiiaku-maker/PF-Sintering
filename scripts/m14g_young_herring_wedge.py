"""Milestone 14G Section 14: static Young-Herring qualification, single
unified free energy.

Unlike Milestone 14F's script (which used two different `Params` --
`p`/`p_eta` -- with two different `gamma_gb_ref` for the f-transport step
vs. the eta step, explicitly rejected by this milestone's Section 5), this
script uses ONE `Params` object throughout: `gamma_gb_override` sets
`p.gamma_gb`/`p.gamma_gb_ref`, and Milestone 14G's `gb_obstacle_coefficients`
already makes `p.k_eta`/`p.W_cpl_f` (eta's own energy) and `p.gamma_gb_ref`
(f's Wc=4*gamma_gb_ref/W groove coupling) consistent by construction --
there is nothing left to decouple.

A SYNTHETIC wedge (not the sintering-neck geometry) is built directly:
a triple junction at the origin where a vertical grain boundary (grain 1 at
x<0, grain 2 at x>0, solid for y<~0) meets two free-surface branches with
EXACTLY the prescribed local slope at the TJ (tangent (-+sin(alpha),cos
(alpha)), dihedral psi=2*alpha) that saturate to a FLAT plateau far from
the TJ, h(x) = lambda*cot(alpha)*(1-exp(-|x|/lambda)) with lambda a few
interface widths -- NOT infinite straight lines to the domain edge. This
matters for the dynamic relaxation test (Step 2 below): infinite straight
branches terminating at a finite reflecting box edge make the box's total
free-surface arc length itself depend strongly on alpha (arc length
~ h_box/cos(alpha) for a fixed box height), which is a spurious,
box-size-scale energy gradient that swamps the local TJ force balance and
drives psi in the WRONG direction (confirmed empirically: an earlier
straight-line version of this script relaxed AWAY from psi_eq). The
saturating profile confines the alpha-dependent part of the surface energy
to a small O(lambda) neighborhood of the TJ -- matching real grain-
boundary-grooving physics, where the far-field surface is flat (zero local
curvature, no driving force, free to translate at zero cost) and only the
near-TJ region determines the equilibrium angle:

    v_gb = (0, -1)                         (GB tangent, away from TJ)
    v_s1 = (-sin(alpha), cos(alpha))       (left free-surface tangent, at TJ)
    v_s2 = (+sin(alpha), cos(alpha))       (right free-surface tangent, at TJ)

Step 1: build the state at the ANALYTIC Young-Herring angle
alpha_eq=acos(gamma_gb/(2*gamma_s)) (psi_eq=2*alpha_eq) and confirm
tj_force.compute_tj_force reports F_TJ_mag/gamma_s small (the constructed
equilibrium wedge is already a near-force-balance state, up to
diffuse-interface/discretization corrections -- this is a DIRECT geometric
check, no relaxation).

Step 2: perturb by building the SAME wedge at a deliberately wrong angle
(alpha != alpha_eq, symmetric so the x=0 mirror symmetry is preserved) and
relax f and eta TOGETHER under the SAME unified `p` (production
`variational_surface_diffusion_step` + `constrained_tangent_cone_eta_update`),
with M_eta and M_s both scaled up so neither kinetic process is
rate-limiting (Section 14's explicit requirement -- this test is about
THERMODYNAMIC qualification, not the rate competition). Requires
psi(t) -> psi_eq and F_TJ_mag/gamma_s -> small.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update
from pf_sintering.gb_obstacle_energy import obstacle_ell, obstacle_profile
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step
from pf_sintering.tj_force import compute_tj_force

BC_X, BC_Y = "reflecting", "reflecting"


def build_wedge(dx_nm, W_nm, gamma_gb, alpha, Nx, Ny, tj_row_frac=0.35, surface_mobility_scale=1.0,
                 lambda_over_W=15.0, saturating=False):
    p = build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, gamma_gb_override=gamma_gb, use_aniso_surface=False,
        surface_mobility_scale=surface_mobility_scale,
    ))
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    x0 = x.mean()
    y0 = tj_row_frac * (y[-1] - y[0]) + y[0]
    X, Y = np.meshgrid(x - x0, y - y0)

    if saturating:
        # h(x) matches the prescribed local TJ slope cot(alpha) at x=0
        # exactly (dh/dx|_{x=0+}=cot(alpha)) and saturates to a flat
        # plateau for |x|>>lambda, instead of an infinite straight line --
        # needed for the DYNAMIC relaxation test (see module docstring):
        # infinite branches terminating at a finite reflecting box edge
        # make the box's total free-surface arc length itself depend
        # strongly on alpha, a spurious box-scale energy gradient that
        # swamps the local TJ force balance. Introduces a small, known
        # measurement bias in the STATIC local-tangent extraction (the
        # measurement circle, radius ~2.5*W, sits close enough to the TJ
        # that the profile's curvature away from the pure cot(alpha)
        # asymptote is not fully negligible) -- acceptable for tracking
        # the SIGN and approach of psi(t), not used for the precise
        # equilibrium-check gate.
        lam = lambda_over_W * p.interface_width
        cot_a = math.cos(alpha) / math.sin(alpha)
        h = lam * cot_a * (1.0 - np.exp(-np.abs(X) / lam))
        d_surf = Y - h  # >0 vapor, <0 solid
    else:
        # Exact infinite-line construction: matches the prescribed
        # tangent EVERYWHERE, not just at the TJ, giving the most accurate
        # possible static local-tangent measurement (used for the Step-1
        # equilibrium-check gate, which does no dynamics and so is not
        # exposed to the box-truncation artifact above).
        sin_a, cos_a = math.sin(alpha), math.cos(alpha)
        d_left = cos_a * X + sin_a * Y
        d_right = sin_a * Y - cos_a * X
        d_surf = np.where(X < 0, d_left, d_right)
    f = 0.5 * (1.0 - np.tanh(d_surf / p.interface_width))

    ell = obstacle_ell(p.k_eta, p.W_cpl_f)
    phi = obstacle_profile(X, ell)  # X itself is the exact distance to the vertical GB (x=x0)
    e2 = phi * f
    e1 = (1.0 - phi) * f
    e3 = np.zeros_like(f)
    tj_xy = np.array([x0, y0])
    return p, f, e1, e2, e3, tj_xy


def measure(f, e1, e2, e3, s, p, tj_xy):
    tj = compute_tj_force(f, e1, e2, tj_xy, s, p)
    F = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    return tj, F


def run_equilibrium_check(dx_nm, W_nm, gamma_gb, Nx, Ny, label):
    p_local = build_params(ModelConfig(preset="dev", dx=dx_nm * 1e-9, r2=80e-9, aspect_ratio=2.0,
                                        contact_orientation="short_plane", initial_overlap=20e-9,
                                        t_total=1e-6, interface_width_override=W_nm * 1e-9,
                                        gamma_gb_override=gamma_gb, use_aniso_surface=False))
    alpha_eq = math.acos(min(1.0, gamma_gb / (2.0 * p_local.gamma_s)))
    psi_eq_deg = math.degrees(2 * alpha_eq)
    p, f, e1, e2, e3, tj_xy = build_wedge(dx_nm, W_nm, gamma_gb, alpha_eq, Nx, Ny)
    s = Sink(threshold=math.inf)
    tj, F = measure(f, e1, e2, e3, s, p, tj_xy)
    ratio = tj.F_TJ_mag / p.gamma_s if tj.resolved else float("nan")
    print(f"{label} [equilibrium check]: gamma_gb={gamma_gb:.4f} psi_eq={psi_eq_deg:.2f}deg "
          f"resolved={tj.resolved} psi_measured={tj.psi_deg:.3f}deg F_TJ_mag/gamma_s={ratio:.4e}")
    return dict(label=label, gamma_gb=gamma_gb, psi_eq_deg=psi_eq_deg, resolved=tj.resolved,
                psi_measured_deg=tj.psi_deg, F_TJ_mag=tj.F_TJ_mag, ratio=ratio)


def run_perturbed_relaxation(dx_nm, W_nm, gamma_gb, Nx, Ny, alpha0_frac, M_eta, surface_mobility_scale,
                              dt_frac, n_steps, sample_every, label):
    p_local = build_params(ModelConfig(preset="dev", dx=dx_nm * 1e-9, r2=80e-9, aspect_ratio=2.0,
                                        contact_orientation="short_plane", initial_overlap=20e-9,
                                        t_total=1e-6, interface_width_override=W_nm * 1e-9,
                                        gamma_gb_override=gamma_gb, use_aniso_surface=False))
    alpha_eq = math.acos(min(1.0, gamma_gb / (2.0 * p_local.gamma_s)))
    psi_eq_deg = math.degrees(2 * alpha_eq)
    alpha0 = alpha0_frac * alpha_eq  # deliberately wrong initial half-angle

    p, f, e1, e2, e3, tj_xy = build_wedge(dx_nm, W_nm, gamma_gb, alpha0, Nx, Ny,
                                           surface_mobility_scale=surface_mobility_scale, saturating=True)
    s = Sink(threshold=math.inf)
    p.M_eta = M_eta
    # p.dt (build_params' own CH-biharmonic-stability limit, already consistent
    # with p.M_f/surface_mobility_scale) governs the f-step; the eta step's own
    # diffusive (parabolic) stability scale can be tighter or looser depending
    # on how far M_eta is scaled up, so take whichever timestep is smaller.
    M_s = m_s_ref(p.M_f, p.interface_width)
    dt = min(p.dt, dt_frac * (p.dx ** 2) / (p.M_eta * p.k_eta))

    mass0 = float(f.sum()) * p.dx * p.dx
    rows = []

    def sample(step, t):
        tj, F = measure(f, e1, e2, e3, s, p, tj_xy)
        mass = float(f.sum()) * p.dx * p.dx
        ratio = tj.F_TJ_mag / p.gamma_s if tj.resolved else float("nan")
        return dict(step=step, t=t, resolved=tj.resolved, psi_deg=tj.psi_deg, F_TJ_mag=tj.F_TJ_mag,
                    ratio=ratio, F=F, mass_drift=(mass - mass0) / mass0)

    rows.append(sample(0, 0.0))
    print(f"{label} [perturbed relax]: gamma_gb={gamma_gb:.4f} psi_eq={psi_eq_deg:.2f}deg "
          f"psi0={math.degrees(2*alpha0):.2f}deg M_eta={p.M_eta:.4e} M_s={M_s:.4e} dt={dt:.4e}s "
          f"n_steps={n_steps}")
    for step in range(1, n_steps + 1):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f, _ = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s,
                                                   bc_x=BC_X, bc_y=BC_Y)
        e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                              bc_x=BC_X, bc_y=BC_Y)
        if step % sample_every == 0:
            row = sample(step, step * dt)
            rows.append(row)
            print(f"  t={step*dt:.4e}s psi={row['psi_deg']:.3f}deg F_TJ_mag/gamma_s={row['ratio']:.4e} "
                  f"mass_drift={row['mass_drift']:.2e}")

    return dict(label=label, gamma_gb=gamma_gb, psi_eq_deg=psi_eq_deg, psi0_deg=math.degrees(2 * alpha0),
                M_eta=p.M_eta, M_s=M_s, dt=dt, n_steps=n_steps, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--W-nm", type=float, default=20.0)
    ap.add_argument("--Nx", type=int, default=160)
    ap.add_argument("--Ny", type=int, default=160)
    ap.add_argument("--gamma-gb-ratio-list", type=str, default="0.5,1.0,1.4")
    ap.add_argument("--gamma-s", type=float, default=1.0)
    ap.add_argument("--alpha0-frac", type=float, default=0.6)
    ap.add_argument("--M-eta", type=float, default=4.266666666666666e-08)
    ap.add_argument("--surface-mobility-scale", type=float, default=10.0)
    ap.add_argument("--dt-frac", type=float, default=0.1)
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=100)
    args = ap.parse_args()

    out = {}
    for ratio_str in args.gamma_gb_ratio_list.split(","):
        ratio = float(ratio_str)
        gamma_gb = ratio * args.gamma_s
        label = f"ratio{ratio}"
        eq = run_equilibrium_check(args.dx_nm, args.W_nm, gamma_gb, args.Nx, args.Ny, label)
        pert = run_perturbed_relaxation(args.dx_nm, args.W_nm, gamma_gb, args.Nx, args.Ny,
                                         args.alpha0_frac, args.M_eta, args.surface_mobility_scale,
                                         args.dt_frac, args.n_steps, args.sample_every, label)
        out[label] = dict(equilibrium_check=eq, perturbed_relaxation=pert)

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
