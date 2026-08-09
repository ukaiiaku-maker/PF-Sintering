"""Milestone 14G Section 12: planar driven-boundary mobility calibration.

Rebuilds Milestone 14E's ad hoc planar-bias script
(scripts/m14e_planar_driven_boundary.py, which used its own
`eta += dt*v; clip; renormalize` loop, bypassing the actual production
tangent-cone machinery -- explicitly rejected by Section 12) using the
`g_external` hook added to `constrained_eta.constrained_tangent_cone_eta_update`
in this milestone instead.

Single flat GB (f=1, bc_x="reflecting" so there is exactly one boundary,
no periodic-image ambiguity), grain 1 on the left / grain 2 on the right.
Two stages:

  1. PRE-RELAX at Delta_g=0 (g_external=None) starting from the calibrated
     obstacle profile, for the same physical relaxation time used in the
     Section 9 qualification, so the run starts from the actual discretized
     equilibrium at this dx -- not from an arbitrary non-equilibrium shape.
  2. Apply a small, constant bulk free-energy bias Delta_g [J/m^3] to grain
     2 via g_external=[None, Delta_g, None] (equivalent to adding
     Delta_g*eta2 to the free energy) THROUGH THE SAME PRODUCTION UPDATE,
     and track the eta2=0.5 front position via bisection. Expect
     v_GB = M_GB*Delta_g (linear in Delta_g), extract M_gb_eff = v/Delta_g,
     and compare against the analytic mapping M_gb_from_m_eta(M_eta,W_GB)
     and the Section 11 circular-grain calibration.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta, obstacle_ell, obstacle_profile
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.tj_force import _sample_bilinear

BC_X, BC_Y = "reflecting", "reflecting"


def build_state(dx_nm, W_nm, gamma_gb, Lx_nm, Ny=8):
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
    ell = obstacle_ell(p.k_eta, p.W_cpl_f)
    e2 = obstacle_profile(X, ell, x0)
    e1 = 1.0 - e2
    f = np.ones_like(e1)
    e3 = np.zeros_like(e1)
    return p, f, e1, e2, e3, x, x0, ell


def measure_front_x(e2, p, x, x_guess, row=None):
    y0 = (row if row is not None else p.Ny // 2) * p.dx + p.dx

    def sample(xx):
        return _sample_bilinear(e2, [xx], [y0], p)[0]

    lo, hi = x_guess - 6 * p.interface_width, x_guess + 6 * p.interface_width
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


def run_one(dx_nm, W_nm, gamma_gb, Lx_nm, Ny, M_eta, delta_g, dt_frac, prerelax_factor,
            n_steps, sample_every, label):
    p, f, e1, e2, e3, x, x0, ell = build_state(dx_nm, W_nm, gamma_gb, Lx_nm, Ny)
    s = Sink(threshold=math.inf)
    p.M_eta = M_eta
    dt = dt_frac * (p.dx ** 2) / (p.M_eta * p.k_eta)
    tau = ell ** 2 / (p.M_eta * p.k_eta)
    n_prerelax = max(1, int(math.ceil(prerelax_factor * tau / dt)))

    for _ in range(n_prerelax):
        e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                              bc_x=BC_X, bc_y=BC_Y)

    bias = delta_g * np.ones_like(e2)
    rows = []
    xf = measure_front_x(e2, p, x, x0)
    rows.append(dict(step=0, t=0.0, x_front=xf))
    max_safety_frac = 0.0
    for step in range(1, n_steps + 1):
        e1, e2, e3, diag = constrained_tangent_cone_eta_update(
            e1, e2, e3, f, s, p, dt=dt, use_eta3=False, bc_x=BC_X, bc_y=BC_Y,
            g_external=[None, bias, None])
        max_safety_frac = max(max_safety_frac, diag["safety_fraction"])
        if step % sample_every == 0:
            xf = measure_front_x(e2, p, x, xf if math.isfinite(xf) else x0)
            rows.append(dict(step=step, t=step * dt, x_front=xf))

    valid = [r for r in rows if math.isfinite(r["x_front"])]
    ts = np.array([r["t"] for r in valid])
    xs = np.array([r["x_front"] for r in valid])
    A = np.c_[ts, np.ones_like(ts)]
    slope, intercept = np.linalg.lstsq(A, xs, rcond=None)[0]
    resid = xs - (A @ [slope, intercept])
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((xs - xs.mean()) ** 2))
    r2_of_fit = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    M_gb_eff = slope / delta_g
    M_gb_analytic = m_gb_from_m_eta(M_eta, W_nm * 1e-9)

    print(f"{label}: dx={dx_nm}nm M_eta={M_eta:.4e} delta_g={delta_g:.4e} v={slope:.6e} "
          f"M_gb_eff={M_gb_eff:.6e} M_gb_analytic={M_gb_analytic:.6e} ratio={M_gb_eff/M_gb_analytic:.4f} "
          f"fit_R2={r2_of_fit:.6f} max_safety_frac={max_safety_frac:.3e}")

    return dict(label=label, dx_nm=dx_nm, W_nm=W_nm, M_eta=M_eta, delta_g=delta_g, dt=dt,
                n_prerelax=n_prerelax, n_steps=n_steps, rows=rows, v=slope, r2_of_fit=r2_of_fit,
                M_gb_eff=M_gb_eff, M_gb_analytic=M_gb_analytic, ratio=M_gb_eff / M_gb_analytic,
                max_safety_frac=max_safety_frac)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dx-nm-list", type=str, default="2.5")
    ap.add_argument("--W-nm", type=float, default=20.0)
    ap.add_argument("--Lx-nm", type=float, default=400.0)
    ap.add_argument("--Ny", type=int, default=8)
    ap.add_argument("--gamma-gb", type=float, default=1.0)
    ap.add_argument("--M-eta", type=float, default=4.266666666666666e-09)
    ap.add_argument("--delta-g-list", type=str, default="1e4,2e4,5e4")
    ap.add_argument("--dt-frac", type=float, default=0.2)
    ap.add_argument("--prerelax-factor", type=float, default=60.0)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=20)
    args = ap.parse_args()

    dx_list = [float(v) for v in args.dx_nm_list.split(",")]

    out = {}
    for dx_nm in dx_list:
        for dg_str in args.delta_g_list.split(","):
            dg = float(dg_str)
            label = f"dx{dx_nm}_dg{dg:.2e}"
            result = run_one(dx_nm, args.W_nm, args.gamma_gb, args.Lx_nm, args.Ny, args.M_eta, dg,
                              args.dt_frac, args.prerelax_factor, args.n_steps, args.sample_every, label)
            out[label] = result

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
