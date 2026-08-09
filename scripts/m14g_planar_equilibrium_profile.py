"""Milestone 14G Section 9: planar GB equilibrium-profile qualification.

Single flat GB (f=1 everywhere, no free surface, no TJ), bc_x="reflecting"
so there is exactly one boundary in the domain (no periodic-image
ambiguity; the domain is wide enough that the reflecting edges sit deep in
bulk grain and never see a nonzero gradient). Initializes eta2 with one of
five profiles -- tanh at the interface_width scale, a broad tanh, a narrow
tanh, a tanh plus a small smooth perturbation, and the analytic obstacle
(compact-support sine) profile itself -- and relaxes ALL of them under the
SAME production integrator, constrained_eta.constrained_tangent_cone_eta_update
(no bias, no diagnostics bypassing it).

For each run, measures:
  - gamma_measured: the ACTUAL excess free energy of the relaxed field,
    integrated directly from the discretized eta fields via the exact
    obstacle-reduction density k_eta*(dphi/dx)^2 + Wc*phi*(1-phi) (Milestone
    14G Section 1), divided by the domain's y-extent (line energy per unit
    GB area) -- NOT the closed-form gamma_obstacle_gb(k_eta,Wc) formula,
    which is a property of the PARAMETERS alone and would pass trivially.
  - width_fit: a nonlinear least-squares fit of the relaxed profile to the
    analytic compact-support sine shape (gb_obstacle_energy.obstacle_profile),
    with only (ell, x0) free -- i.e. does the SIMULATED, discretized
    equilibrium profile actually have the predicted sine shape and length
    scale, not just approximately the right energy.

Both are compared against the analytic target (obstacle_gamma_gb,
obstacle_compact_width) computed from the run's own p.k_eta/p.W_cpl_f.
Requires gamma_measured/gamma_target -> 1 and width_fit/width_target -> 1,
with the disagreement shrinking under grid refinement (dx = 5, 2.5, 1.25 nm).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import curve_fit

from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update, local_wc
from pf_sintering.gb_obstacle_energy import (
    obstacle_compact_width,
    obstacle_ell,
    obstacle_gamma_gb,
    obstacle_profile,
)
from pf_sintering.model import ModelConfig, Sink, build_params

BC_X, BC_Y = "reflecting", "reflecting"

IC_KINDS = ("tanh", "broad_tanh", "narrow_tanh", "smooth_perturbation", "obstacle")


def build_state(dx_nm, W_nm, gamma_gb, Lx_nm, Ny, ic):
    Nx = max(64, round(Lx_nm / dx_nm))
    p = build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, gamma_gb_override=gamma_gb, use_aniso_surface=False,
    ))
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    x0 = 0.5 * (x[0] + x[-1])
    ell_target = obstacle_ell(p.k_eta, p.W_cpl_f)

    if ic == "obstacle":
        e2 = obstacle_profile(X, ell_target, x0)
    elif ic == "smooth_perturbation":
        width_m = p.interface_width
        e2 = 0.5 * (1 + np.tanh((X - x0) / width_m))
        noise = 0.02 * np.sin(2 * math.pi * (X - x0) / (5 * width_m))
        e2 = np.clip(e2 + noise, 0.0, 1.0)
    else:
        width_scale = dict(tanh=1.0, broad_tanh=3.0, narrow_tanh=1.0 / 3.0)[ic]
        width_m = width_scale * p.interface_width
        e2 = 0.5 * (1 + np.tanh((X - x0) / width_m))
    e1 = 1.0 - e2
    f = np.ones_like(e1)
    e3 = np.zeros_like(e1)
    return p, f, e1, e2, e3, x, x0, ell_target


def excess_energy_and_row(e1, e2, f, p, s):
    Wc = local_wc(f, e1, e2, np.zeros_like(f), s, p)
    phi = e2  # e1 = 1-e2 to projection tolerance under this f=1, two-grain update
    dphidx = np.gradient(phi, p.dx, axis=1)
    density = p.k_eta * dphidx ** 2 + Wc * phi * (1.0 - phi)
    Ly = p.Ny * p.dx
    gamma_measured = float(np.sum(density) * p.dx * p.dx) / Ly
    mid_row = p.Ny // 2
    return gamma_measured, phi[mid_row, :]


def fit_ell(x, phi_row, x0_guess, ell_guess):
    def model_fn(x_, ell, x0):
        return obstacle_profile(x_, ell, x0)
    try:
        popt, _ = curve_fit(model_fn, x, phi_row, p0=[ell_guess, x0_guess],
                             bounds=([ell_guess * 0.1, x0_guess - 20 * ell_guess],
                                      [ell_guess * 10.0, x0_guess + 20 * ell_guess]))
        return float(popt[0]), float(popt[1])
    except Exception:
        return float("nan"), float("nan")


def run_one(dx_nm, W_nm, gamma_gb, Lx_nm, Ny, ic, dt_frac, relax_factor, label):
    p, f, e1, e2, e3, x, x0, ell_target = build_state(dx_nm, W_nm, gamma_gb, Lx_nm, Ny, ic)
    s = Sink(threshold=math.inf)
    # p.dt (CH-biharmonic-stability-limited, ~dx^4) is far more conservative
    # than the pure-eta relaxation needs; use the eta-PDE's own diffusive
    # (parabolic, ~dx^2) stability scale instead, tied to the physical
    # relaxation timescale tau=ell_target^2/(M_eta*k_eta) so total physical
    # relaxation time (relax_factor*tau) and step-count safety margin
    # (dt_frac) are both grid-independent -- n_steps then grows as 1/dx^2,
    # the correct explicit-diffusion CFL scaling, instead of a fixed count.
    tau = ell_target ** 2 / (p.M_eta * p.k_eta)
    dt = dt_frac * (p.dx ** 2) / (p.M_eta * p.k_eta)
    n_steps = max(1, int(math.ceil(relax_factor * tau / dt)))
    gamma_target = obstacle_gamma_gb(p.k_eta, p.W_cpl_f)
    width_target = obstacle_compact_width(p.k_eta, p.W_cpl_f)

    max_safety_frac = 0.0
    for _ in range(n_steps):
        e1, e2, e3, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                                 bc_x=BC_X, bc_y=BC_Y)
        max_safety_frac = max(max_safety_frac, diag["safety_fraction"])

    gamma_measured, phi_row = excess_energy_and_row(e1, e2, f, p, s)
    ell_fit, x0_fit = fit_ell(x, phi_row, x0, ell_target)
    width_fit = math.pi * ell_fit if math.isfinite(ell_fit) else float("nan")

    gamma_ratio = gamma_measured / gamma_target
    width_ratio = width_fit / width_target if math.isfinite(width_fit) else float("nan")
    print(f"{label}: dx={dx_nm}nm ic={ic:<20s} n_steps={n_steps} gamma_meas/target={gamma_ratio:.5f} "
          f"width_fit/target={width_ratio:.5f} max_safety_frac={max_safety_frac:.3e}")

    return dict(label=label, dx_nm=dx_nm, W_nm=W_nm, ic=ic, n_steps=n_steps, dt=dt,
                gamma_measured=gamma_measured, gamma_target=gamma_target, gamma_ratio=gamma_ratio,
                width_fit=width_fit, width_target=width_target, width_ratio=width_ratio,
                ell_fit=ell_fit, ell_target=ell_target, max_safety_frac=max_safety_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dx-nm-list", type=str, default="5,2.5,1.25")
    ap.add_argument("--ic-list", type=str, default=",".join(IC_KINDS))
    ap.add_argument("--W-nm", type=float, default=20.0)
    ap.add_argument("--Lx-nm", type=float, default=400.0)
    ap.add_argument("--Ny", type=int, default=8)
    ap.add_argument("--gamma-gb", type=float, default=1.0)
    ap.add_argument("--dt-frac", type=float, default=0.2, help="dt as a fraction of dx^2/(M_eta*k_eta)")
    ap.add_argument("--relax-factor", type=float, default=60.0,
                     help="total relaxation time as a multiple of ell_target^2/(M_eta*k_eta)")
    args = ap.parse_args()

    dx_list = [float(v) for v in args.dx_nm_list.split(",")]
    ic_list = [v for v in args.ic_list.split(",")]

    out = {}
    for dx_nm in dx_list:
        for ic in ic_list:
            label = f"dx{dx_nm}_{ic}"
            result = run_one(dx_nm, args.W_nm, args.gamma_gb, args.Lx_nm, args.Ny, ic,
                              args.dt_frac, args.relax_factor, label)
            out[label] = result

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
