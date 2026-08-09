"""Milestone 14E Section 3: circular-grain curvature-migration calibration.

Constructs f=1 everywhere with a circular grain-2 inclusion in a grain-1
matrix (no free surface, no TJ, periodic BC both axes), runs the
production constrained tangent-cone eta integrator
(constrained_eta.constrained_tangent_cone_eta_update) at fixed f, and
measures R(t) via bisection on the eta2=0.5 level set sampled on 72 rays
from the grain center. For isotropic curvature-driven migration
R^2(t) = R0^2 - 2*M_gb*gamma_gb*t, so

    M_gb_eff = -(1/(2*gamma_gb)) * d(R^2)/dt

(linear least-squares fit over the run). Reports M_gb_eff vs. the
requested M_eta (proportionality check) and vs. dx (grid convergence),
plus total-f/energy checks each run.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update, structural_thermodynamic_force, local_wc
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
    e2 = 0.5 * (1 - np.tanh((r - R0) / p.interface_width))
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
    e_bulk = Wc * eta2 * (0.5 * f * f - f)
    gx1, gy1 = np.gradient(e1, p.dx, axis=(1, 0))
    gx2, gy2 = np.gradient(e2, p.dx, axis=(1, 0))
    e_grad = 0.5 * p.k_eta * (gx1 * gx1 + gy1 * gy1 + gx2 * gx2 + gy2 * gy2)
    return float(np.sum(e_bulk + e_grad)) * p.dx * p.dx


def run_one(dx_nm, W_nm, R0_nm, gamma_gb, M_eta, dt_mult, n_steps, sample_every, label, dt_budget=None):
    p, f, e1, e2, e3, center = build_state(dx_nm, W_nm, R0_nm, gamma_gb)
    s = Sink(threshold=math.inf)
    p.M_eta = M_eta
    # dt_budget keeps M_eta*dt fixed across an M_eta ladder (the empirical
    # stability parameter controlling the tangent-cone safety correction,
    # Section 11) so every scale reaches a comparable R^2 shrinkage
    # fraction in the same n_steps while staying in the safety_frac~
    # roundoff regime -- overrides dt_mult when given.
    dt = (dt_budget / M_eta) if dt_budget is not None else p.dt * dt_mult
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

    print(f"{label}: dx={dx_nm}nm M_eta={M_eta:.4e} R0={R0_nm}nm N={p.Nx} dt={dt:.4e}s n_steps={n_steps} "
          f"M_gb_eff={M_gb_eff:.6e} fit_R2={r2_of_fit:.6f} mass_drift={(mass1-mass0)/mass0:.3e} "
          f"dF={(F1-F0):.4e} max_safety_frac={max_safety_frac:.3e}")

    return dict(label=label, dx_nm=dx_nm, W_nm=W_nm, R0_nm=R0_nm, gamma_gb=gamma_gb, M_eta=M_eta, dt=dt,
                n_steps=n_steps, rows=rows, slope=slope, intercept=intercept, r2_of_fit=r2_of_fit,
                M_gb_eff=M_gb_eff, mass0=mass0, mass1=mass1, F0=F0, F1=F1, max_safety_frac=max_safety_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dx-nm-list", type=str, default="2.5")
    ap.add_argument("--M-eta-scale-list", type=str, default="1,5,20")
    ap.add_argument("--M-eta-base", type=float, default=4.266666666666666e-09)
    ap.add_argument("--W-nm", type=float, default=20.0)
    ap.add_argument("--R0-nm", type=float, default=100.0)
    ap.add_argument("--gamma-gb", type=float, default=1.0)
    ap.add_argument("--dt-mult", type=float, default=20.0)
    ap.add_argument("--dt-budget", type=float, default=None, help="fixed M_eta*dt product; overrides dt-mult")
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=20)
    args = ap.parse_args()

    dx_list = [float(x) for x in args.dx_nm_list.split(",")]
    scale_list = [float(x) for x in args.M_eta_scale_list.split(",")]

    out = {}
    for dx_nm in dx_list:
        for scale in scale_list:
            label = f"dx{dx_nm}_scale{scale}"
            M_eta = args.M_eta_base * scale
            result = run_one(dx_nm, args.W_nm, args.R0_nm, args.gamma_gb, M_eta, args.dt_mult,
                              args.n_steps, args.sample_every, label, dt_budget=args.dt_budget)
            out[label] = result

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
