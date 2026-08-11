"""Milestone 16A Sections 4-6: validate the axisymmetric solver's core
variational structure before using it for any physics benchmark:
  (a) mu is the exact variational derivative of F (finite-difference
      directional-derivative test, matching ch_exact_energy.py's own
      validation pattern);
  (b) V_f=2*pi*integral(r*f)dr dz is exactly conserved under the
      conservative step;
  (c) dF/dt<=0 along a real trial evolution, and the chain-rule Fdot
      matches an independent finite-difference [F(t+dt)-F(t)]/dt.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_free_energy, axisym_mu, axisym_surface_diffusion_step, axisym_volume, r_centers_faces,
)


class P:
    def __init__(self, gamma_s=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W


def random_smooth_field(Nz, Nr, rng):
    z = np.linspace(0, 2 * math.pi, Nz, endpoint=False)
    r = np.linspace(0, 2 * math.pi, Nr, endpoint=False)
    Z, R = np.meshgrid(z, r, indexing="ij")
    f = 0.5 + 0.3 * np.sin(Z + 0.3) * np.cos(0.7 * R + 0.5) + 0.05 * rng.standard_normal((Nz, Nr))
    return np.clip(f, 0.02, 0.98)


def test_variational_derivative():
    print("(a) mu is the exact variational derivative of F:")
    rng = np.random.default_rng(0)
    Nz, Nr, dr, dz = 24, 20, 2.0e-9, 2.0e-9
    r_c, r_f = r_centers_faces(Nr, dr)
    p = P()
    f = random_smooth_field(Nz, Nr, rng)
    q = random_smooth_field(Nz, Nr, rng) - 0.5  # perturbation direction

    mu = axisym_mu(f, p, dr, dz, r_c, r_f)
    lhs_pred = 2 * math.pi * float(np.sum(r_c[None, :] * mu * q)) * dr * dz

    for eps in (1e-3, 1e-4, 1e-5):
        Fp = axisym_free_energy(f + eps * q, p, dr, dz, r_c, r_f)
        Fm = axisym_free_energy(f - eps * q, p, dr, dz, r_c, r_f)
        fd = (Fp - Fm) / (2 * eps)
        rel_err = abs(fd - lhs_pred) / max(abs(lhs_pred), 1e-30)
        print(f"  eps={eps:.0e}  finite-diff={fd:.8e}  predicted(2pi*sum(r*mu*q))={lhs_pred:.8e}  rel_err={rel_err:.3e}")
    print()


def test_volume_conservation_and_dissipation():
    print("(b)+(c) volume conservation and energy dissipation under real evolution:")
    Nz, Nr, dr, dz = 32, 24, 2.0e-9, 2.0e-9
    r_c, r_f = r_centers_faces(Nr, dr)
    p = P()
    W = 20e-9
    M_s = 1e-33

    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    R0 = 12e-9
    eps0 = 1e-9
    lam = Nz * dz
    Rprofile = R0 + eps0 * np.cos(2 * math.pi * Z / lam)
    f = 0.5 * (1.0 - np.tanh((R - Rprofile) / W))

    V0 = axisym_volume(f, r_c, dr, dz)
    F0 = axisym_free_energy(f, p, dr, dz, r_c, r_f)
    # conservative empirical dt: bisect down from a generous guess until
    # 10 CONSECUTIVE trial steps stay bounded and non-growing, rather
    # than trusting an analytic CFL estimate for this new discretization.
    dt = 1.0
    for _ in range(200):
        f_try = f.copy()
        stable = True
        for _ in range(60):
            f_try, _, _ = axisym_surface_diffusion_step(f_try, p, dr, dz, r_c, r_f, dt, M_s, W)
            if not np.all(np.isfinite(f_try)) or np.max(np.abs(f_try)) > 2.0:
                stable = False
                break
        if stable:
            break
        dt *= 0.5
    print(f"  V0={V0:.6e}  F0={F0:.6e}  dt={dt:.3e}")

    max_dV_rel = 0.0
    max_dF = 0.0
    Fdot_chain_vs_fd_err = []
    Fprev = F0
    for step in range(50):
        f_new, mu, diag = axisym_surface_diffusion_step(f, p, dr, dz, r_c, r_f, dt, M_s, W)
        V_new = axisym_volume(f_new, r_c, dr, dz)
        F_new = axisym_free_energy(f_new, p, dr, dz, r_c, r_f)
        dV_rel = abs(V_new - V0) / abs(V0)
        max_dV_rel = max(max_dV_rel, dV_rel)
        dF = F_new - Fprev
        max_dF = max(max_dF, dF)
        fd_rate = (F_new - Fprev) / dt
        rel = abs(fd_rate - diag["Fdot_chain"]) / max(abs(diag["Fdot_chain"]), 1e-30)
        Fdot_chain_vs_fd_err.append(rel)
        f = f_new
        Fprev = F_new

    print(f"  max relative |V(t)-V0|/V0 over 50 steps: {max_dV_rel:.3e}")
    print(f"  max (F(t+dt)-F(t)) over 50 steps (must be <=0): {max_dF:.3e}")
    print(f"  max relative error Fdot_chain vs finite-diff rate: {max(Fdot_chain_vs_fd_err):.3e}")
    print()
    return max_dV_rel < 1e-10, max_dF <= 1e-20, max(Fdot_chain_vs_fd_err) < 1e-6


if __name__ == "__main__":
    test_variational_derivative()
    ok_v, ok_f, ok_chain = test_volume_conservation_and_dissipation()
    print(f"PASS: volume_conserved={ok_v} energy_descends={ok_f} chain_rule_consistent={ok_chain}")
