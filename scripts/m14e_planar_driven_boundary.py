"""Milestone 14E Section 4: planar driven-boundary cross-check.

Two flat GBs in a periodic-X domain (grain2 a stripe in the middle of a
grain1 background, so the domain closes periodically with two symmetric
flat boundaries), f=1 everywhere, no curvature. A small, DIAGNOSTIC-ONLY
bulk free-energy bias `Delta_g` [J/m^3] is added directly in this script
(never wired into pf_sintering.constrained_eta) by adding `Delta_g` to
g2's structural thermodynamic force before the tangent-cone projection --
equivalent to adding `Delta_g*eta2` to the free energy, so grain 2 has a
higher bulk free energy per unit ownership than grain 1 and the boundary
advances into grain 2. Expected: v_GB = M_gb*Delta_g. Measures v for
several small Delta_g at fixed M_eta and compares M_gb_eff=v/Delta_g
against the circular-grain calibration.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.constrained_eta import local_wc, structural_thermodynamic_force, tangent_cone_projected_velocity
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.tj_force import _sample_bilinear

BC_X, BC_Y = "periodic", "periodic"


def build_state(dx_nm, W_nm, Lx_nm, gamma_gb, Ny=8):
    Nx = max(64, round(Lx_nm / dx_nm))
    p = build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, gamma_gb_override=gamma_gb, use_aniso_surface=False,
    ))
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    Lx = p.Nx * p.dx
    x1, x2 = 0.25 * Lx, 0.75 * Lx  # grain2 stripe occupies (x1, x2)
    e2 = 0.5 * (np.tanh((X - x1) / p.interface_width) - np.tanh((X - x2) / p.interface_width))
    e1 = 1 - e2
    f = np.ones_like(e1)
    return p, f, e1, e2, x, x1, x2


def measure_front_x(e2, p, x, x_guess, row=0):
    # bisection for the eta2=0.5 crossing near x_guess along one row
    y0 = p.dx  # first row's physical y (index+1)*dx convention, row index 0
    def sample(xx):
        return _sample_bilinear(e2, [xx], [y0], p)[0]
    lo, hi = x_guess - 3 * p.interface_width, x_guess + 3 * p.interface_width
    flo, fhi = sample(lo) - 0.5, sample(hi) - 0.5
    if flo * fhi > 0:
        return float("nan")
    for _ in range(50):
        mid = (lo + hi) / 2
        fm = sample(mid) - 0.5
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def biased_velocity(e1, e2, f, Wc, p, dt, Delta_g):
    g1 = structural_thermodynamic_force(e1, f, Wc, p.dx, p.k_eta, BC_X, BC_Y)
    g2 = structural_thermodynamic_force(e2, f, Wc, p.dx, p.k_eta, BC_X, BC_Y) + Delta_g
    v0_1 = -p.M_eta * g1
    v0_2 = -p.M_eta * g2
    v1, v2 = tangent_cone_projected_velocity([e1, e2], [v0_1, v0_2], active_tol=1e-4)
    return v1, v2


def run_one(dx_nm, W_nm, Lx_nm, gamma_gb, M_eta, Delta_g, dt, n_steps, sample_every, label):
    p, f, e1, e2, x, x1, x2 = build_state(dx_nm, W_nm, Lx_nm, gamma_gb)
    s = Sink(threshold=math.inf)
    p.M_eta = M_eta
    mass0 = float(f.sum()) * p.dx * p.dx

    rows = []
    xf = measure_front_x(e2, p, x, x2)
    rows.append(dict(step=0, t=0.0, x_front=xf))
    for step in range(1, n_steps + 1):
        Wc = local_wc(f, e1, e2, np.zeros_like(f), s, p)
        v1, v2 = biased_velocity(e1, e2, f, Wc, p, dt, Delta_g)
        e1 = np.clip(e1 + dt * v1, 0.0, 1.0)
        e2 = np.clip(e2 + dt * v2, 0.0, 1.0)
        tot = e1 + e2
        bad = tot > 1e-30
        e1 = np.where(bad, e1 * f / np.where(bad, tot, 1.0), 0.0)
        e2 = np.where(bad, e2 * f / np.where(bad, tot, 1.0), 0.0)
        if step % sample_every == 0:
            xf = measure_front_x(e2, p, x, xf if math.isfinite(xf) else x2)
            rows.append(dict(step=step, t=step * dt, x_front=xf))

    mass1 = float(f.sum()) * p.dx * p.dx
    valid = [r for r in rows if math.isfinite(r["x_front"])]
    ts = np.array([r["t"] for r in valid])
    xs = np.array([r["x_front"] for r in valid])
    A = np.c_[ts, np.ones_like(ts)]
    slope, intercept = np.linalg.lstsq(A, xs, rcond=None)[0]
    resid = xs - (A @ [slope, intercept])
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((xs - xs.mean()) ** 2))
    r2_of_fit = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"{label}: dx={dx_nm}nm M_eta={M_eta:.4e} Delta_g={Delta_g:.4e} v={slope:.6e} fit_R2={r2_of_fit:.6f} "
          f"mass_drift={(mass1-mass0)/mass0:.3e}")
    return dict(label=label, dx_nm=dx_nm, M_eta=M_eta, Delta_g=Delta_g, dt=dt, n_steps=n_steps, rows=rows,
                v=slope, r2_of_fit=r2_of_fit, mass0=mass0, mass1=mass1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--W-nm", type=float, default=20.0)
    ap.add_argument("--Lx-nm", type=float, default=400.0)
    ap.add_argument("--gamma-gb", type=float, default=1.0)
    ap.add_argument("--M-eta", type=float, default=4.266666666666666e-09)
    ap.add_argument("--M-eta-scale", type=float, default=20.0)
    ap.add_argument("--delta-g-list", type=str, default="1e4,2e4,5e4,1e5")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=100)
    args = ap.parse_args()

    M_eta = args.M_eta * args.M_eta_scale
    dt = args.dt if args.dt is not None else 8e-14 / M_eta

    out = {}
    for dg_str in args.delta_g_list.split(","):
        dg = float(dg_str)
        label = f"dg{dg:.2e}"
        result = run_one(args.dx_nm, args.W_nm, args.Lx_nm, args.gamma_gb, M_eta, dg, dt,
                          args.n_steps, args.sample_every, label)
        out[label] = result

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
