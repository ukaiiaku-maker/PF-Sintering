"""Milestone 13D Sections 12-13: isolated sinusoidal substrate regression
+ paired legacy/new face-flux comparison.

Repeats Milestone 13C's exact qualified isolated-substrate benchmark
(wavelength=480nm, amplitude=24nm, W=20nm, particle removed, X
reflecting, Y periodic, isotropic, unified conserved surface diffusion
only), from the SAME initial state, under BOTH `face_flux_mode`s
("cell_average_legacy", Milestone 13C's own path, and "face_projected",
Milestone 13D's repaired operator) -- the new discretization must not
destroy the already-qualified macroscopic physics (A1 monotonic decrease,
mass conservation, energy decrease, away-from-crest flux, k^4-consistent
decay rate), and the two modes' physical trajectories should converge
toward one another with dx while the new operator additionally eliminates
the non-tangential face flux (tracked here too, face_projected mode only).
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.curvature_extraction import _kasa_fit_curvature
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import (
    face_projected_tangentiality,
    m_s_ref,
    surface_divergence_update,
    surface_flux,
    surface_flux_face_projected,
    surface_mobility_tensor,
)

BC_X, BC_Y = "reflecting", "periodic"


def build_substrate_only(dx_nm, wavelength_nm, amplitude_nm, w_nm, surface_mobility_scale=0.3, margin_w=6.0):
    dx = dx_nm * 1e-9
    wavelength = wavelength_nm * 1e-9
    amplitude = amplitude_nm * 1e-9
    W = w_nm * 1e-9
    Ny = round(wavelength / dx)
    wavelength = Ny * dx  # snap exactly, genuine periodicity
    Nx = max(80, round((margin_w * W + 2 * amplitude) / dx))

    p = build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=w_nm * 1e-9, use_aniso_surface=False,
        surface_mobility_scale=surface_mobility_scale,
    ))
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    xc = x.mean()
    x_s = xc + amplitude * np.cos(2 * math.pi * Y / wavelength)
    f = 0.5 * (1 - np.tanh((X - x_s) / W))
    return p, f, wavelength, amplitude


def x_crossing_profile(f, p):
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    x_cross = np.full(p.Ny, np.nan)
    for i in range(p.Ny):
        row = f[i, :]
        sign = row - 0.5
        idx = np.where(np.diff(np.sign(sign)) != 0)[0]
        if len(idx) == 0:
            continue
        j = idx[0]
        v0, v1 = sign[j], sign[j + 1]
        t = -v0 / (v1 - v0) if (v1 - v0) != 0 else 0.0
        x_cross[i] = x[j] + t * (x[j + 1] - x[j])
    return y, x_cross


def fourier_harmonics(f, p, wavelength, n_harmonics=3):
    y, x_cross = x_crossing_profile(f, p)
    valid = np.isfinite(x_cross)
    if valid.sum() < 6:
        return [math.nan] * n_harmonics, math.nan
    cols = [np.ones(valid.sum())]
    for n in range(1, n_harmonics + 1):
        cols.append(np.cos(n * 2 * math.pi * y[valid] / wavelength))
        cols.append(np.sin(n * 2 * math.pi * y[valid] / wavelength))
    A = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(A, x_cross[valid], rcond=None)
    x0 = coeffs[0]
    amps = [float(math.hypot(coeffs[1 + 2 * n], coeffs[2 + 2 * n])) for n in range(n_harmonics)]
    return amps, float(x0)


def crest_trough(f, p):
    y, x_cross = x_crossing_profile(f, p)
    valid = np.isfinite(x_cross)
    yv, xv = y[valid], x_cross[valid]
    i_crest = int(np.argmax(xv))
    i_trough = int(np.argmin(xv))
    crest_xy = (float(xv[i_crest]), float(yv[i_crest]))
    trough_xy = (float(xv[i_trough]), float(yv[i_trough]))
    return crest_xy, trough_xy


def local_curvature(f, p, xy, window_pts=15):
    y, x_cross = x_crossing_profile(f, p)
    valid = np.isfinite(x_cross)
    yv, xv = y[valid], x_cross[valid]
    d = np.abs(yv - xy[1])
    d = np.minimum(d, (p.Ny * p.dx) - d)  # periodic distance in Y
    order = np.argsort(d)[:window_pts]
    pts = np.column_stack([xv[order], yv[order]])
    return _kasa_fit_curvature(pts, f, p, min_points=6)


def away_from_crest_flux(f, Jx, Jy, p, crest_xy, probe_distance):
    """J_s (tangential, arclength oriented AWAY from the crest) sampled at
    +/- probe_distance in Y from the crest, along the local free surface."""
    from pf_sintering.tj_force import _sample_bilinear
    y, x_cross = x_crossing_profile(f, p)
    valid = np.isfinite(x_cross)
    yv, xv = y[valid], x_cross[valid]
    Ly = p.Ny * p.dx

    def sample_side(sign):
        y_target = (crest_xy[1] + sign * probe_distance) % Ly
        idx = int(np.argmin(np.minimum(np.abs(yv - y_target), Ly - np.abs(yv - y_target))))
        lo, hi = max(0, idx - 2), min(len(yv), idx + 3)
        ys, xs = yv[lo:hi], xv[lo:hi]
        order = np.argsort(ys)
        ys, xs = ys[order], xs[order]
        dxdy = np.gradient(xs, ys) if len(ys) > 1 else np.array([0.0])
        tnorm = math.hypot(1.0, dxdy[len(dxdy) // 2])
        tx, ty = dxdy[len(dxdy) // 2] / tnorm, 1.0 / tnorm  # tangent, +y-leaning orientation
        if sign < 0:
            tx, ty = -tx, -ty  # orient AWAY from crest on this side
        xq, yq = xs[len(xs) // 2], ys[len(ys) // 2]
        Jx_s = float(_sample_bilinear(Jx, [xq], [yq], p)[0])
        Jy_s = float(_sample_bilinear(Jy, [xq], [yq], p)[0])
        return Jx_s * tx + Jy_s * ty

    return sample_side(+1), sample_side(-1)


def run(dx_nm, args, target_times, face_flux_mode="cell_average_legacy"):
    p, f, wavelength, amplitude0 = build_substrate_only(dx_nm, args.wavelength_nm, args.amplitude_nm,
                                                          args.w_nm, args.surface_mobility_scale)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    print(f"=== isolated substrate, dx={dx_nm}nm mode={face_flux_mode}: Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s "
          f"wavelength={wavelength*1e9:.2f}nm ===")

    e1 = e2 = e3 = np.zeros_like(f)
    mass0 = float(f.sum()) * p.dx * p.dx

    rows = []
    step = 0

    def sample(step, t):
        amps, x0 = fourier_harmonics(f, p, wavelength)
        crest_xy, trough_xy = crest_trough(f, p)
        kappa_crest = local_curvature(f, p, crest_xy)
        kappa_trough = local_curvature(f, p, trough_xy)
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y,
                                                 eps_n=1e-6 / p.interface_width)
        Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)
        j_right, j_left = away_from_crest_flux(f, Jx, Jy, p, crest_xy, 2 * p.interface_width)
        F = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
        mass = float(f.sum()) * p.dx * p.dx
        row = dict(step=step, time_s=t, A1=amps[0], A2=amps[1], A3=amps[2],
                   crest_xy=crest_xy, trough_xy=trough_xy,
                   kappa_crest=kappa_crest, kappa_trough=kappa_trough,
                   J_away_right=j_right, J_away_left=j_left,
                   F=F, mass=mass, mass_drift=(mass - mass0) / mass0)
        if face_flux_mode == "face_projected":
            fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
            tang = face_projected_tangentiality(fp)
            face_of = {"x": fp["xface"], "y": fp["yface"]}
            for axis, ratio_key, mag_key in (("x", "ratio_x", "mag_x"), ("y", "ratio_y", "mag_y")):
                ratio, mag = tang[ratio_key], tang[mag_key]
                # Milestone 13E Section 9: J.n/|J| blows up to an O(1) ratio
                # wherever the RAW (unregularized) |grad f| at that face is
                # weak compared to eps_n -- i.e. off the interface entirely
                # (e.g. the reflecting-wall tail deep in bulk, f~1-1e-5),
                # where n_face = grad(f)/sqrt(|grad f|^2+eps_n^2) is no
                # longer a meaningful unit normal and the tangential
                # projection P=I-n n^T stops suppressing J.n (1-|n|^2 is
                # O(1), not ~eps_n^2/|grad f|^2 << 1). The face's OWN
                # q_face (already returned by surface_flux_face_projected,
                # the authoritative interface-localization weight at that
                # exact face) is what distinguishes real interface faces
                # (q_face ~ its own max, ratio ~1e-7) from these physically
                # off-interface faces (q_face << its max, ratio ~1e-2) --
                # both the mag-floor (roundoff) and this q_face-floor
                # (interface presence) are needed to report an authoritative
                # tangentiality statistic, matching the near-roundoff values
                # seen on manufactured/particle-contact face_projected runs.
                q_face = face_of[axis]["q_face"]
                mag_floor = 1e-6 * np.nanmax(mag)
                q_floor = 1e-3 * np.nanmax(q_face)
                valid = np.isfinite(ratio) & (mag > mag_floor) & (q_face > q_floor)
                row[f"tang_{axis}_rms"] = float(np.sqrt(np.mean(ratio[valid] ** 2))) if np.any(valid) else math.nan
                row[f"tang_{axis}_max"] = float(np.max(ratio[valid])) if np.any(valid) else math.nan
        return row

    row0 = sample(0, 0.0)
    rows.append(row0)
    print(f"  t=0.000s A1={row0['A1']*1e9:.4f}nm A2={row0['A2']*1e9:.4f}nm A3={row0['A3']*1e9:.4f}nm "
          f"kappa_crest={row0['kappa_crest']:.3e} J_away={row0['J_away_right']:.3e},{row0['J_away_left']:.3e}")

    for t_target in target_times:
        if t_target <= 0.0:
            continue
        n_target = round(t_target / p.dt)
        while step < n_target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y,
                                                 face_flux_mode=face_flux_mode)
            step += 1
        row = sample(step, step * p.dt)
        rows.append(row)
        tg = "" if face_flux_mode != "face_projected" else f" tang_x_rms={row['tang_x_rms']:.3e}"
        print(f"  t={step*p.dt:.4e}s A1={row['A1']*1e9:.4f}nm A2={row['A2']*1e9:.4f}nm A3={row['A3']*1e9:.4f}nm "
              f"kappa_crest={row['kappa_crest']:.3e} kappa_trough={row['kappa_trough']:.3e} "
              f"J_away_R={row['J_away_right']:.3e} J_away_L={row['J_away_left']:.3e} "
              f"F={row['F']:.6e} mass_drift={row['mass_drift']:.2e}{tg}")

    return dict(dx_nm=dx_nm, mode=face_flux_mode, wavelength=wavelength, amplitude0=amplitude0, p_dt=p.dt, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--times", type=str, default="0,0.01,0.03,0.06,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--modes", type=str, default="cell_average_legacy,face_projected")
    ap.add_argument("--out", type=str, default="runs/m13d_isolated_substrate_paired.json")
    args = ap.parse_args()

    target_times = [float(x) for x in args.times.split(",")]
    modes = args.modes.split(",")
    out = {}
    for mode in modes:
        result = run(args.dx_nm, args, target_times, face_flux_mode=mode)
        out[mode] = result

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
