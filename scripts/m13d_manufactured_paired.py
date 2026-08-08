"""Milestone 13D Sections 10-11: manufactured sinusoid/circle, evolved
trajectory, legacy ("cell_average_legacy") vs new ("face_projected")
face-flux mode, paired from identical initial states.

Unlike Milestone 13C's near-static 2e-4s manufactured-interface check,
this evolves the SELF-CONSISTENT Mullins sinusoid (mu = mu_isotropic, the
actual production chemical potential -- not an artificially imposed one)
long enough for a measurable (target ~2-10%) amplitude change, using a
short wavelength (120nm, matching Milestone 12B's own qualified Mullins
benchmark) rather than an artificially altered mobility, per Section 10's
explicit instruction. The circle case retains an imposed mu (a decaying
physical mu would just be another Mullins-type problem) so its curvature
stays controlled, but is also evolved over a comparably long interval.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import (
    dissipation_density_face_projected,
    face_projected_tangentiality,
    m_s_ref,
    surface_flux_face_projected,
    surface_divergence_update,
)
from pf_sintering.discrete_tangentiality import tangentiality_summary

BC_X, BC_Y = "reflecting", "periodic"


def build_isotropic_params(dx_nm, W_nm, nx, ny, surface_mobility_scale=0.3):
    return build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=nx, ny=ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, use_aniso_surface=False,
        surface_mobility_scale=surface_mobility_scale,
    ))


def sinusoid_state(dx_nm, wavelength_nm, amplitude_frac, W_nm, surface_mobility_scale):
    dx = dx_nm * 1e-9
    wavelength = wavelength_nm * 1e-9
    Ny = round(wavelength / dx)
    wavelength = Ny * dx
    amplitude0 = amplitude_frac * wavelength
    Nx = max(60, round(6 * W_nm * 1e-9 / dx) + round(3 * amplitude0 / dx))
    p = build_isotropic_params(dx_nm, W_nm, Nx, Ny, surface_mobility_scale)
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    W = p.interface_width
    h = x.mean() + amplitude0 * np.cos(2 * math.pi * Y / wavelength)
    f = 0.5 * (1 + np.tanh((X - h) / W))
    return p, f, wavelength, amplitude0


def fit_amplitude(f, p, wavelength):
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    x_cross = []
    for i in range(p.Ny):
        row = f[i, :]
        sign = row - 0.5
        idx = np.where(np.diff(np.sign(sign)) != 0)[0]
        if len(idx) == 0:
            x_cross.append(np.nan)
            continue
        j = idx[0]
        v0, v1 = sign[j], sign[j + 1]
        t = -v0 / (v1 - v0) if (v1 - v0) != 0 else 0.0
        x_cross.append(x[j] + t * (x[j + 1] - x[j]))
    x_cross = np.array(x_cross)
    valid = np.isfinite(x_cross)
    if valid.sum() < 6:
        return math.nan
    A = np.c_[np.ones(valid.sum()), np.cos(2 * math.pi * y[valid] / wavelength),
              np.sin(2 * math.pi * y[valid] / wavelength)]
    coeffs, *_ = np.linalg.lstsq(A, x_cross[valid], rcond=None)
    return float(math.hypot(coeffs[1], coeffs[2]))


def circle_state(dx_nm, R_nm, W_nm, surface_mobility_scale, n_modes=3, mu_amplitude_frac=0.3):
    dx = dx_nm * 1e-9
    Nx = Ny = max(80, round(4 * R_nm * 1e-9 / dx))
    p = build_isotropic_params(dx_nm, W_nm, Nx, Ny, surface_mobility_scale)
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    cx, cy = x.mean(), y.mean()
    r = np.hypot(X - cx, Y - cy)
    theta = np.arctan2(Y - cy, X - cx)
    W = p.interface_width
    f = 0.5 * (1 - np.tanh((r - R_nm * 1e-9) / W))
    mu = mu_amplitude_frac * np.sin(n_modes * theta)
    return p, f, mu, (cx, cy)


def tangentiality_stats(fp, mag_floor_frac=1e-6):
    tang = face_projected_tangentiality(fp)
    out = {}
    for axis, ratio_key, mag_key in (("x", "ratio_x", "mag_x"), ("y", "ratio_y", "mag_y")):
        ratio, mag = tang[ratio_key], tang[mag_key]
        floor = mag_floor_frac * np.nanmax(mag)
        valid = np.isfinite(ratio) & (mag > floor)
        out[axis] = dict(max=float(np.max(ratio[valid])) if np.any(valid) else math.nan,
                          rms=float(np.sqrt(np.mean(ratio[valid] ** 2))) if np.any(valid) else math.nan,
                          n=int(valid.sum()))
    return out


def run_sinusoid(dx_nm, args, mode):
    p, f, wavelength, amplitude0 = sinusoid_state(dx_nm, args.wavelength_nm, args.amplitude_frac,
                                                    args.w_nm, args.surface_mobility_scale)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    e1 = e2 = e3 = np.zeros_like(f)
    n_steps = round(args.t_end / p.dt)
    sample_every = max(1, n_steps // args.n_samples)
    mass0 = float(f.sum())

    rows = []
    A0 = fit_amplitude(f, p, wavelength)
    F0 = None
    rows.append(dict(step=0, time_s=0.0, A=A0, mass_drift=0.0))
    for step in range(1, n_steps + 1):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s,
                                                  bc_x=BC_X, bc_y=BC_Y, face_flux_mode=mode)
        f = f_new
        if step % sample_every == 0 or step == n_steps:
            A = fit_amplitude(f, p, wavelength)
            mass_drift = (float(f.sum()) - mass0) / mass0
            row = dict(step=step, time_s=step * p.dt, A=A, mass_drift=mass_drift)
            if mode == "face_projected":
                fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
                row["tangentiality"] = tangentiality_stats(fp)
            rows.append(row)

    return dict(dx_nm=dx_nm, mode=mode, wavelength=wavelength, amplitude0=amplitude0, p_dt=p.dt, rows=rows)


def run_circle(dx_nm, args, mode):
    p, f, mu, center = circle_state(dx_nm, args.r_nm, args.w_nm, args.surface_mobility_scale)
    M_s = m_s_ref(p.M_f, p.interface_width)
    n_steps = round(args.circle_t_end / p.dt)
    sample_every = max(1, n_steps // args.n_samples)
    mass0 = float(f.sum())

    rows = []
    for step in range(0, n_steps + 1):
        if step % sample_every == 0 or step == n_steps:
            mass_drift = (float(f.sum()) - mass0) / mass0
            row = dict(step=step, time_s=step * p.dt, mass_drift=mass_drift)
            if mode == "face_projected":
                fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
                row["tangentiality"] = tangentiality_stats(fp)
            else:
                tang = tangentiality_summary(f, *_legacy_flux(f, mu, p, M_s), p, BC_X, BC_Y, [center])
                row["tangentiality_legacy"] = tang
            rows.append(row)
        if step < n_steps:
            f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s,
                                                      bc_x=BC_X, bc_y=BC_Y, face_flux_mode=mode)
            f = f_new

    return dict(dx_nm=dx_nm, mode=mode, R_nm=args.r_nm, p_dt=p.dt, rows=rows)


def _legacy_flux(f, mu, p, M_s):
    from pf_sintering.surface_transport import surface_flux, surface_mobility_tensor
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
    return surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=120.0)
    ap.add_argument("--amplitude-frac", type=float, default=0.02)
    ap.add_argument("--r-nm", type=float, default=150.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--dx-list-nm", type=str, default="5.0,2.5,1.25")
    ap.add_argument("--t-end", type=float, default=0.25)
    ap.add_argument("--circle-t-end", type=float, default=2e-4)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--out", type=str, default="runs/m13d_manufactured_paired.json")
    args = ap.parse_args()

    dx_list = [float(x) for x in args.dx_list_nm.split(",")]
    out = dict(args=vars(args), sinusoid={}, circle={})
    for dx_nm in dx_list:
        for mode in ("cell_average_legacy", "face_projected"):
            print(f"\n=== sinusoid dx={dx_nm}nm mode={mode} ===")
            r = run_sinusoid(dx_nm, args, mode)
            out["sinusoid"][f"{dx_nm}_{mode}"] = r
            for row in r["rows"]:
                tg = row.get("tangentiality")
                tg_str = "" if tg is None else f" tang_x_rms={tg['x']['rms']:.3e} tang_y_rms={tg['y']['rms']:.3e}"
                print(f"  t={row['time_s']:.4e}s A={row['A']*1e9:.4f}nm mass_drift={row['mass_drift']:.2e}{tg_str}")

            print(f"\n=== circle dx={dx_nm}nm mode={mode} ===")
            rc = run_circle(dx_nm, args, mode)
            out["circle"][f"{dx_nm}_{mode}"] = rc
            for row in rc["rows"]:
                tg = row.get("tangentiality")
                tg_str = "" if tg is None else f" tang_x_rms={tg['x']['rms']:.3e} tang_x_max={tg['x']['max']:.3e}"
                print(f"  t={row['time_s']:.4e}s mass_drift={row['mass_drift']:.2e}{tg_str}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
