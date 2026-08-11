"""Milestone 16A Section 3: planar Cartesian control.

Simplest possible single-phase planar ligament: NO eta/GB/substrate/
sink/anisotropy -- a slab of solid (f=1) between two free surfaces at
x = x_mid +- [W0 + eps0*cos(2*pi*z/lambda)]/2, periodic in z (mapped to
the grid's "y" axis, matching this project's BC_Y=periodic convention),
reflecting in x. Evolved with the PRODUCTION Cartesian pathway
(ch_exact_energy.mu_isotropic + surface_transport.
variational_surface_diffusion_step) -- the exact per-unit-depth
variational problem documented in Section 2, no axisymmetric weighting
at all. Since a per-unit-depth Cartesian ligament has NO analogue of
the r-weighted curvature term, plain surface diffusion should SMOOTH
(decay) the perturbation at every wavelength -- there is no Cartesian
analogue of the Plateau-Rayleigh instability. This is the explicit
control the milestone calls for, not a failure mode.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.ch_exact_energy import mu_isotropic  # noqa: E402
from pf_sintering.model import ModelConfig, Sink, build_params  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16a_campaign")
BC_X, BC_Y = "reflecting", "periodic"


def build_ligament(Nx, Ny, dx, W0, W_iface, eps0, x_mid):
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    lam = Ny * dx
    half_width = 0.5 * (W0 + eps0 * np.cos(2 * math.pi * Y / lam))
    x1 = x_mid - half_width
    x2 = x_mid + half_width
    f = 0.5 * (1.0 - np.tanh((X - x2) / W_iface)) * 0.5 * (1.0 + np.tanh((X - x1) / W_iface))
    return f


def amplitude_ligament(f, dx, x_mid):
    """half peak-to-trough of the ligament half-width(z), measured from
    the two f=0.5 crossings per row."""
    Ny, Nx = f.shape
    widths = np.full(Ny, np.nan)
    xs = (np.arange(1, Nx + 1)) * dx
    for j in range(Ny):
        row = f[j]
        idx = np.where((row[:-1] - 0.5) * (row[1:] - 0.5) < 0)[0]
        if len(idx) < 2:
            continue
        i_lo, i_hi = idx[0], idx[-1]
        x_lo = xs[i_lo] + (0.5 - row[i_lo]) * dx / (row[i_lo + 1] - row[i_lo])
        x_hi = xs[i_hi] + (0.5 - row[i_hi]) * dx / (row[i_hi + 1] - row[i_hi])
        widths[j] = x_hi - x_lo
    valid = widths[np.isfinite(widths)]
    if len(valid) < 4:
        return float("nan")
    return 0.5 * (float(np.max(valid)) - float(np.min(valid)))


def run(case_id, lam_nm, W0_nm=200.0, W_iface_nm=20.0, dx_nm=2.5, eps0_frac=0.05, n_steps_total=40000,
        n_sample=30):
    out_path = os.path.join(CAMPAIGN_DIR, f"{case_id}.json")
    if os.path.exists(out_path):
        print(f"[{case_id}] already done, skipping")
        return

    Ny = max(16, round(lam_nm / dx_nm))
    Nx = max(24, round((W0_nm + 6 * W_iface_nm) / dx_nm))
    p = build_params(ModelConfig(preset="dev", geometry="substrate", nx=Nx, ny=Ny, dx=dx_nm * 1e-9,
                                  r2=40e-9, t_total=1e-6, interface_width_override=W_iface_nm * 1e-9,
                                  use_aniso_surface=False))
    x_mid = 0.5 * Nx * p.dx
    eps0 = eps0_frac * W0_nm * 1e-9
    f = build_ligament(Nx, Ny, p.dx, W0_nm * 1e-9, W_iface_nm * 1e-9, eps0, x_mid)
    e1 = f.copy()
    e2 = np.zeros_like(f)
    e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    dt = p.dt

    mass0 = float(f.sum()) * p.dx * p.dx
    amp0 = amplitude_ligament(f, p.dx, x_mid)
    print(f"[{case_id}] Nx={Nx} Ny={Ny} dt={dt:.3e} amp0={amp0*1e9:.4f}nm")

    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))
    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, _ = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        amp = amplitude_ligament(f, p.dx, x_mid)
        mass = float(f.sum()) * p.dx * p.dx
        row = dict(step=step, t=step * dt, amplitude=amp, mass_drift=(mass - mass0) / mass0)
        rows.append(row)
        print(f"  [{case_id}] step={step} t={row['t']:.4e} amp={amp*1e9:.4f}nm mass_drift={row['mass_drift']:.2e}")

    result = dict(case_id=case_id, lam_nm=lam_nm, W0_nm=W0_nm, W_iface_nm=W_iface_nm, dx_nm=dx_nm,
                  amp0=amp0, rows=rows)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"[{case_id}] saved to {out_path}")


if __name__ == "__main__":
    for lam_nm in (400.0, 630.0, 800.0):
        run(f"planar_control_lam{lam_nm:g}", lam_nm)
