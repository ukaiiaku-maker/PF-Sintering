"""Milestone 13C Section 6: manufactured curved-interface tangentiality
convergence test.

Unlike Milestone 12B's straight-interface/uniform-J synthetic test (which
only validated the flux-integration machinery, not the difficult part of
the operator), this constructs a CURVED diffuse interface -- a circle and
a sinusoid -- with a controlled chemical-potential variation along
arclength, runs the PRODUCTION P_t / cell flux / face reconstruction /
conservative divergence for a modest physical interval (not just one
step), and tracks pf_sintering.discrete_tangentiality's cell- and
face-centered J.n ratios throughout, at dx=5, 2.5, and 1.25nm.

Requirement (Section 6): J_cell.n_cell ~ 0 always (by construction,
already confirmed on the real geometry); J_face.n_face -> 0 with spatial
refinement, and this convergence must hold throughout the evolved
trajectory, not just at t=0.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.discrete_tangentiality import tangentiality_summary
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import (
    m_s_ref,
    surface_divergence_update,
    surface_mobility_tensor,
)

BC_X, BC_Y = "reflecting", "periodic"


def build_isotropic_params(dx_nm, W_nm, nx, ny, surface_mobility_scale=0.3):
    return build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=nx, ny=ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, use_aniso_surface=False,
        surface_mobility_scale=surface_mobility_scale,
    ))


def circular_interface(p, R, mu_amplitude_frac=0.3, n_modes=3):
    """A circular diffuse interface (solid inside), with an artificial
    but SMOOTH, arclength-periodic chemical potential imposed directly
    (not self-consistently derived from mu_isotropic -- this test isolates
    the TRANSPORT OPERATOR's tangentiality, not the free energy), so the
    flux field has genuine spatial structure (not a trivial uniform
    rotation) while remaining perfectly smooth and controllable."""
    W = p.interface_width
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    cx, cy = x.mean(), y.mean()
    r = np.hypot(X - cx, Y - cy)
    theta = np.arctan2(Y - cy, X - cx)
    f = 0.5 * (1 - np.tanh((r - R) / W))
    mu = mu_amplitude_frac * np.sin(n_modes * theta)
    return f, mu, dict(cx=cx, cy=cy, R=R, theta=theta)


def sinusoidal_interface(p, wavelength, amplitude, mu_amplitude_frac=0.3):
    W = p.interface_width
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    xc = x.mean()
    h = xc + amplitude * np.cos(2 * math.pi * Y / wavelength)
    f = 0.5 * (1 - np.tanh((X - h) / W))
    mu = mu_amplitude_frac * np.cos(2 * math.pi * Y / wavelength)
    return f, mu


def run_case(label, dx_nm, args, build_fn, t_end, sample_every_steps):
    W = args.w_nm
    if label == "circle":
        Nx = Ny = max(80, round(4 * args.R_nm / dx_nm))
        p = build_isotropic_params(dx_nm, W, Nx, Ny, args.surface_mobility_scale)
        f, mu0, meta = circular_interface(p, args.R_nm * 1e-9)
        tj_probe = [(meta["cx"], meta["cy"])]
    else:
        Ny = max(80, round(args.wavelength_nm / dx_nm))
        Nx = max(80, round(3 * args.wavelength_nm / dx_nm))
        p = build_isotropic_params(dx_nm, W, Nx, Ny, args.surface_mobility_scale)
        f, mu0 = sinusoidal_interface(p, args.wavelength_nm * 1e-9, args.sin_amplitude_nm * 1e-9)
        tj_probe = [(x_ := (np.arange(1, p.Nx + 1) * p.dx).mean(), (np.arange(1, p.Ny + 1) * p.dx).mean())]

    M_s = m_s_ref(p.M_f, p.interface_width)
    n_steps = round(t_end / p.dt)
    mass0 = float(f.sum())

    rows = []
    for step in range(0, n_steps + 1):
        # mu is held at a FIXED functional shape (re-evaluated on the current
        # f's own geometry each step via the same construction) so the
        # imposed chemical-potential structure persists as the interface
        # itself evolves, rather than decaying to a trivial constant.
        if label == "circle":
            f_now = f
            _, mu, meta = circular_interface(p, args.R_nm * 1e-9)
            mu = mu  # same fixed field; interface itself moves under it
        else:
            _, mu = sinusoidal_interface(p, args.wavelength_nm * 1e-9, args.sin_amplitude_nm * 1e-9)

        Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y,
                                                 eps_n=1e-6 / p.interface_width)
        from pf_sintering.surface_transport import surface_flux
        Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)

        if step % sample_every_steps == 0 or step == n_steps:
            summ = tangentiality_summary(f, Jx, Jy, p, BC_X, BC_Y, tj_probe)
            mass_drift = abs(float(f.sum()) - mass0) / mass0
            rows.append(dict(step=step, time_s=step * p.dt, mass_drift=mass_drift,
                              cell_max=summ["cell"]["global_"]["max"], cell_rms=summ["cell"]["global_"]["rms"],
                              face_x_max=summ["face_x"]["global_"]["max"], face_x_rms=summ["face_x"]["global_"]["rms"],
                              face_y_max=summ["face_y"]["global_"]["max"], face_y_rms=summ["face_y"]["global_"]["rms"]))

        if step < n_steps:
            f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
            f = f_new

    return dict(dx_nm=dx_nm, Nx=p.Nx, Ny=p.Ny, dt=p.dt, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--r-nm", dest="R_nm", type=float, default=150.0)
    ap.add_argument("--wavelength-nm", type=float, default=200.0)
    ap.add_argument("--sin-amplitude-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--dx-list-nm", type=str, default="5.0,2.5,1.25")
    ap.add_argument("--t-end", type=float, default=2e-4)
    ap.add_argument("--sample-every-steps", type=int, default=5)
    ap.add_argument("--out", type=str, default="runs/m13c_manufactured_interface.json")
    args = ap.parse_args()

    dx_list = [float(x) for x in args.dx_list_nm.split(",")]
    out = dict(args=vars(args), circle={}, sinusoid={})
    for dx_nm in dx_list:
        print(f"\n=== circle, dx={dx_nm}nm ===")
        r = run_case("circle", dx_nm, args, circular_interface, args.t_end, args.sample_every_steps)
        out["circle"][str(dx_nm)] = r
        for row in r["rows"]:
            print(f"  t={row['time_s']:.4e}s cell_max={row['cell_max']:.3e} cell_rms={row['cell_rms']:.3e} "
                  f"face_x_max={row['face_x_max']:.4f} face_x_rms={row['face_x_rms']:.4f} "
                  f"face_y_max={row['face_y_max']:.4f} face_y_rms={row['face_y_rms']:.4f} "
                  f"mass_drift={row['mass_drift']:.2e}")

        print(f"\n=== sinusoid, dx={dx_nm}nm ===")
        r2 = run_case("sinusoid", dx_nm, args, sinusoidal_interface, args.t_end, args.sample_every_steps)
        out["sinusoid"][str(dx_nm)] = r2
        for row in r2["rows"]:
            print(f"  t={row['time_s']:.4e}s cell_max={row['cell_max']:.3e} cell_rms={row['cell_rms']:.3e} "
                  f"face_x_max={row['face_x_max']:.4f} face_x_rms={row['face_x_rms']:.4f} "
                  f"face_y_max={row['face_y_max']:.4f} face_y_rms={row['face_y_rms']:.4f} "
                  f"mass_drift={row['mass_drift']:.2e}")

    print("\n=== Convergence summary (final-time face RMS, both shapes) ===")
    for shape in ("circle", "sinusoid"):
        for dx_nm in dx_list:
            last = out[shape][str(dx_nm)]["rows"][-1]
            print(f"  {shape} dx={dx_nm}nm: face_x_rms={last['face_x_rms']:.4e} face_y_rms={last['face_y_rms']:.4e}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
