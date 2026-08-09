"""Milestone 14G Section 11: circular-grain M_GB calibration (obstacle IC).

Rebuilds Milestone 14E's circular-grain curvature-migration calibration
(scripts/m14e_circular_grain_calibration.py) with two Milestone 14G
corrections:

  1. The radial GB is initialized with the CALIBRATED OBSTACLE PROFILE
     (gb_obstacle_energy.obstacle_profile), not a tanh -- Section 2 showed
     tanh is not the equilibrium profile of this constrained free energy,
     so a tanh IC would spend part of the trajectory relaxing shape before
     any meaningful curvature-driven shrinkage could be measured cleanly.
  2. M_gb_eff is compared against the ANALYTIC mobility mapping (Section
     10), M_GB_analytic = 4*M_eta*W_GB/pi^2 = m_gb_from_m_eta(M_eta, W_GB),
     not just an M_eta-proportionality trend.

Still: f=1 everywhere (no free surface, no TJ), periodic BC both axes,
production constrained_tangent_cone_eta_update, R(t) measured via bisection
on the eta2=0.5 level set on 72 rays from the grain center, isotropic
curvature-driven shrinkage R^2(t) = R0^2 - 2*M_gb*gamma_gb*t so
M_gb_eff = -(1/(2*gamma_gb)) * d(R^2)/dt (linear least-squares fit).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update, local_wc
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta, obstacle_ell, obstacle_profile
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.tj_force import _sample_bilinear

BC_X, BC_Y = "periodic", "periodic"


def build_state(dx_nm, W_nm, R0_nm, gamma_gb, domain_R_factor=3.2):
    N = max(64, 2 * round(domain_R_factor * R0_nm / dx_nm / 2))
    p = build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=N, ny=N, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, gamma_gb_override=gamma_gb, use_aniso_surface=False,
    ))
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    cx, cy = x.mean(), y.mean()
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - cx, Y - cy)
    R0 = R0_nm * 1e-9
    ell = obstacle_ell(p.k_eta, p.W_cpl_f)
    # obstacle_profile is 0->1 with increasing argument; the grain-2
    # inclusion should be 1 inside (r<R0) and 0 outside, i.e. use -(r-R0).
    e2 = obstacle_profile(-(r - R0), ell)
    e1 = 1 - e2
    f = np.ones_like(e1)
    e3 = np.zeros_like(e1)
    return p, f, e1, e2, e3, (cx, cy)


def measure_R(e2, p, center, R_guess, n_rays=72):
    cx, cy = center
    thetas = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    def sample_at(rr, th):
        xq = cx + rr * np.cos(th)
        yq = cy + rr * np.sin(th)
        return _sample_bilinear(e2, [xq], [yq], p)[0]

    Rs = []
    for th in thetas:
        lo, hi = 0.3 * R_guess, 2.2 * R_guess
        flo, fhi = sample_at(lo, th) - 0.5, sample_at(hi, th) - 0.5
        if flo * fhi > 0:
            continue
        for _ in range(40):
            mid = (lo + hi) / 2
            fm = sample_at(mid, th) - 0.5
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        Rs.append((lo + hi) / 2)
    if not Rs:
        return float("nan"), float("nan"), 0
    return float(np.mean(Rs)), float(np.std(Rs)), len(Rs)


def structural_free_energy(e1, e2, f, p, s):
    Wc = local_wc(f, e1, e2, np.zeros_like(f), s, p)
    eta2 = e1 * e1 + e2 * e2
    fb = np.clip(f, 0.0, 1.0)
    e_bulk = Wc * eta2 * (0.5 * fb * fb - fb)
    gx1, gy1 = np.gradient(e1, p.dx, axis=(1, 0))
    gx2, gy2 = np.gradient(e2, p.dx, axis=(1, 0))
    e_grad = 0.5 * p.k_eta * (gx1 * gx1 + gy1 * gy1 + gx2 * gx2 + gy2 * gy2)
    return float(np.sum(e_bulk + e_grad)) * p.dx * p.dx


def run_one(dx_nm, W_nm, R0_nm, gamma_gb, M_eta, dt_budget, n_steps, sample_every, label):
    p, f, e1, e2, e3, center = build_state(dx_nm, W_nm, R0_nm, gamma_gb)
    s = Sink(threshold=math.inf)
    p.M_eta = M_eta
    dt = dt_budget / M_eta
    mass0 = float(f.sum()) * p.dx * p.dx
    F0 = structural_free_energy(e1, e2, f, p, s)

    rows = []
    R, std, n = measure_R(e2, p, center, R0_nm * 1e-9)
    rows.append(dict(step=0, t=0.0, R=R, std=std, n=n))
    Rcur = R
    max_safety_frac = 0.0
    for step in range(1, n_steps + 1):
        e1, e2, e3, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                                 bc_x=BC_X, bc_y=BC_Y)
        max_safety_frac = max(max_safety_frac, diag["safety_fraction"])
        if step % sample_every == 0:
            R, std, n = measure_R(e2, p, center, Rcur)
            if math.isfinite(R):
                Rcur = R
            rows.append(dict(step=step, t=step * dt, R=R, std=std, n=n))

    mass1 = float(f.sum()) * p.dx * p.dx
    F1 = structural_free_energy(e1, e2, f, p, s)

    valid = [r for r in rows if math.isfinite(r["R"])]
    ts = np.array([r["t"] for r in valid])
    R2s = np.array([r["R"] ** 2 for r in valid])
    A = np.c_[ts, np.ones_like(ts)]
    slope, intercept = np.linalg.lstsq(A, R2s, rcond=None)[0]
    resid = R2s - (A @ [slope, intercept])
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((R2s - R2s.mean()) ** 2))
    r2_of_fit = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    M_gb_eff = -slope / (2 * gamma_gb)
    M_gb_analytic = m_gb_from_m_eta(M_eta, W_nm * 1e-9)

    print(f"{label}: dx={dx_nm}nm M_eta={M_eta:.4e} R0={R0_nm}nm N={p.Nx} dt={dt:.4e}s n_steps={n_steps} "
          f"M_gb_eff={M_gb_eff:.6e} M_gb_analytic={M_gb_analytic:.6e} ratio={M_gb_eff/M_gb_analytic:.4f} "
          f"fit_R2={r2_of_fit:.6f} mass_drift={(mass1-mass0)/mass0:.3e} max_safety_frac={max_safety_frac:.3e}")

    return dict(label=label, dx_nm=dx_nm, W_nm=W_nm, R0_nm=R0_nm, gamma_gb=gamma_gb, M_eta=M_eta, dt=dt,
                n_steps=n_steps, rows=rows, slope=slope, intercept=intercept, r2_of_fit=r2_of_fit,
                M_gb_eff=M_gb_eff, M_gb_analytic=M_gb_analytic, ratio=M_gb_eff / M_gb_analytic,
                mass0=mass0, mass1=mass1, F0=F0, F1=F1, max_safety_frac=max_safety_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dx-nm-list", type=str, default="2.5")
    ap.add_argument("--M-eta-base", type=float, default=4.266666666666666e-09)
    ap.add_argument("--W-nm", type=float, default=20.0)
    ap.add_argument("--R0-nm", type=float, default=100.0)
    ap.add_argument("--gamma-gb", type=float, default=1.0)
    ap.add_argument("--dt-budget", type=float, default=8e-14)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=20)
    args = ap.parse_args()

    dx_list = [float(x) for x in args.dx_nm_list.split(",")]

    out = {}
    for dx_nm in dx_list:
        label = f"dx{dx_nm}"
        result = run_one(dx_nm, args.W_nm, args.R0_nm, args.gamma_gb, args.M_eta_base, args.dt_budget,
                          args.n_steps, args.sample_every, label)
        out[label] = result

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
