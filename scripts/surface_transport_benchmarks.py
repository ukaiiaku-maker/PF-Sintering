"""Milestone 12 -- Gate A/B benchmarks for the unified variational
surface-diffusion transport law (pf_sintering/surface_transport.py).

1. Planar free surface: J~0, df/dt~0, mass conserved, energy stationary.
2. Mullins sinusoidal decay: k^4 scaling of the linear decay rate.
3. Energy dissipation: exact discrete isotropic F decreases for small dt.
4. dt convergence at fixed dx.

Single-phase (e1=e2=e3=0 identically) isotropic mu throughout -- no
particle/substrate/eta complexity yet (that is Gates C-E, later scripts).
Does not modify production physics.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update


def _zero_eta(p):
    return np.zeros((p.Ny, p.Nx)), np.zeros((p.Ny, p.Nx))


def build_isotropic_params(dx_nm, W_nm, nx, ny, surface_mobility_scale=0.3):
    return build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=nx, ny=ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, use_aniso_surface=False,
        surface_mobility_scale=surface_mobility_scale,
    ))


def planar_benchmark(args):
    print("\n=== Benchmark 17: planar free surface ===")
    W = args.w_nm * 1e-9
    dx = args.dx_nm * 1e-9
    Nx, Ny = 60, 20
    p = build_isotropic_params(args.dx_nm, args.w_nm, Nx, Ny)
    x = (np.arange(1, Nx + 1)) * dx
    f = np.tile(0.5 * (1 + np.tanh((x - x.mean()) / W)), (Ny, 1))
    e1 = np.zeros_like(f); e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    mu = mu_isotropic(f, e1, e1, e3, s, p)  # e2=e1=0 dummy (isotropic, no eta coupling)
    M_s = m_s_ref(p.M_f, W)
    f_new, diag = surface_divergence_update(f, mu, dx, dt=p.dt, W=W, M_s=M_s,
                                             bc_x="reflecting", bc_y="periodic")
    max_J = max(np.max(np.abs(diag["Jx"])), np.max(np.abs(diag["Jy"])))
    max_df = np.max(np.abs(f_new - f))
    mass_before = float(f.sum()) * dx * dx
    mass_after = float(f_new.sum()) * dx * dx
    F_before = exact_free_energy_isotropic(f, e1, e1, e3, s, p)
    F_after = exact_free_energy_isotropic(f_new, e1, e1, e3, s, p)
    print(f"  max|J|={max_J:.4e}  max|df|={max_df:.4e}  mass_before={mass_before:.6e}  "
          f"mass_after={mass_after:.6e}  F_before={F_before:.8e}  F_after={F_after:.8e}")
    return dict(max_J=max_J, max_df=max_df, mass_before=mass_before, mass_after=mass_after,
                F_before=F_before, F_after=F_after)


def _fit_fourier_amplitude(f, p, wavelength):
    """Fit the fundamental cos-amplitude of the X-crossing (f=0.5) as a
    function of Y, for a genuinely periodic domain (Y periodic, one full
    wavelength = Ny*dx)."""
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


def mullins_decay_run(args, wavelength_nm, n_steps, sample_every):
    W = args.w_nm * 1e-9
    dx = args.dx_nm * 1e-9
    wavelength = wavelength_nm * 1e-9
    Ny = round(wavelength / dx)
    wavelength = Ny * dx  # snap exactly, genuine periodicity (no symmetry trick needed)
    amplitude0 = args.amplitude_frac * wavelength
    Nx = max(60, round(6 * W / dx) + round(3 * amplitude0 / dx))

    p = build_isotropic_params(args.dx_nm, args.w_nm, Nx, Ny)
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    h = x.mean() + amplitude0 * np.cos(2 * math.pi * Y / wavelength)
    f = 0.5 * (1 + np.tanh((X - h) / W))
    e1 = np.zeros_like(f); e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, W)

    t_series, A_series = [0.0], [_fit_fourier_amplitude(f, p, wavelength)]
    mass0 = float(f.sum())
    for step in range(1, n_steps + 1):
        mu = mu_isotropic(f, e1, e1, e3, s, p)
        f, diag = surface_divergence_update(f, mu, dx, p.dt, W, M_s, bc_x="reflecting", bc_y="periodic")
        if step % sample_every == 0:
            t_series.append(step * p.dt)
            A_series.append(_fit_fourier_amplitude(f, p, wavelength))
    mass_drift = abs(float(f.sum()) - mass0) / mass0
    return dict(wavelength=wavelength, k=2 * math.pi / wavelength, t=t_series, A=A_series,
                mass_drift=mass_drift, dt=p.dt, Nx=Nx, Ny=Ny)


def mullins_benchmark(args):
    print("\n=== Benchmark 18: Mullins sinusoidal decay, k^4 scaling ===")
    results = {}
    for wl_nm in args.wavelengths_nm:
        r = mullins_decay_run(args, wl_nm, args.mullins_steps, args.mullins_sample_every)
        A = np.array(r["A"]); t = np.array(r["t"])
        valid = np.isfinite(A) & (A > 0)
        if valid.sum() < 3:
            print(f"  wavelength={wl_nm}nm: could not fit (insufficient valid samples)")
            results[f"{wl_nm:g}"] = r
            continue
        slope, intercept = np.polyfit(t[valid], np.log(A[valid]), 1)
        tau = -1.0 / slope if slope < 0 else math.inf
        print(f"  wavelength={wl_nm}nm k={r['k']:.4e}/m  A0={A[0]*1e9:.4f}nm A_end={A[valid][-1]*1e9:.4f}nm "
              f"decay_rate={-slope:.4e}/s  tau={tau:.4e}s  mass_drift={r['mass_drift']:.2e}  Nx,Ny=({r['Nx']},{r['Ny']})")
        r["decay_rate"] = -slope
        r["tau"] = tau
        results[f"{wl_nm:g}"] = r

    # k^4 scaling check between consecutive pairs
    wls = sorted(args.wavelengths_nm)
    print("  k^4 scaling check (decay_rate ratio vs (k1/k2)^4):")
    for i in range(len(wls) - 1):
        r1, r2 = results[f"{wls[i]:g}"], results[f"{wls[i+1]:g}"]
        if "decay_rate" not in r1 or "decay_rate" not in r2:
            continue
        ratio_rate = r1["decay_rate"] / r2["decay_rate"]
        ratio_k4 = (r1["k"] / r2["k"]) ** 4
        print(f"    wavelength {wls[i]}nm vs {wls[i+1]}nm: decay_rate ratio={ratio_rate:.4f}  (k1/k2)^4={ratio_k4:.4f}")

    ks = [results[f"{wl:g}"]["k"] for wl in wls if "decay_rate" in results[f"{wl:g}"]]
    rates = [results[f"{wl:g}"]["decay_rate"] for wl in wls if "decay_rate" in results[f"{wl:g}"]]
    if len(ks) >= 3:
        slope, intercept = np.polyfit(np.log(ks), np.log(rates), 1)
        pred = np.exp(intercept) * np.array(ks) ** slope
        ss_res = np.sum((np.array(rates) - pred) ** 2)
        ss_tot = np.sum((np.array(rates) - np.mean(rates)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  log-log power-law fit: decay_rate ~ k^{slope:.4f}  (R^2={r2:.6f}, expect exponent~4)")
        results["power_law_exponent"] = slope
        results["power_law_r2"] = r2
    return results


def energy_dissipation_benchmark(args):
    print("\n=== Energy dissipation test (exact discrete isotropic F) ===")
    W = args.w_nm * 1e-9
    dx = args.dx_nm * 1e-9
    Nx, Ny = 60, 30
    p = build_isotropic_params(args.dx_nm, args.w_nm, Nx, Ny)
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    x = (np.arange(1, Nx + 1)) * dx
    f = 0.5 + 0.15 * gaussian_filter(rng.normal(size=(Ny, Nx)), sigma=2.0)
    f = np.clip(f, 0.02, 0.98)
    e1 = np.zeros_like(f); e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, W)

    out = {}
    for frac in (1.0, 0.5, 0.25, 0.125):
        dt = p.dt * frac
        mu = mu_isotropic(f, e1, e1, e3, s, p)
        f_new, _ = surface_divergence_update(f, mu, dx, dt, W, M_s, bc_x="reflecting", bc_y="periodic")
        F0 = exact_free_energy_isotropic(f, e1, e1, e3, s, p)
        F1 = exact_free_energy_isotropic(f_new, e1, e1, e3, s, p)
        print(f"  dt_frac={frac}: dt={dt:.4e}s F0={F0:.10e} F1={F1:.10e} F1<=F0: {F1<=F0}  (F1-F0)/dt={(F1-F0)/dt:.4e}")
        out[str(frac)] = dict(dt=dt, F0=F0, F1=F1, decreased=bool(F1 <= F0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--wavelengths-nm", type=float, nargs="+", default=[120.0, 160.0, 240.0])
    ap.add_argument("--amplitude-frac", type=float, default=0.02)
    ap.add_argument("--mullins-steps", type=int, default=4000)
    ap.add_argument("--mullins-sample-every", type=int, default=40)
    ap.add_argument("--out", type=str, default="runs/surface_transport_benchmarks.json")
    args = ap.parse_args()

    out = dict(args=vars(args))
    out["planar"] = planar_benchmark(args)
    out["energy_dissipation"] = energy_dissipation_benchmark(args)
    out["mullins"] = mullins_benchmark(args)

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
