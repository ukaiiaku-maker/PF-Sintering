"""Milestone 16C Section 8/8b: validate the sharp-interface constrained-
Hessian eigenmode machinery (pf_sintering/sharp_interface_stability.py)
against (a) the classical single-crystal Rayleigh-Plateau threshold
lambda_c/R0=2*pi, and (b) the Hussein et al. GB-destabilization trend
(adding a GB should REDUCE the critical wavelength for instability,
more strongly for larger gamma_gb / smaller dihedral angle).
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.sharp_interface_stability import (  # noqa: E402
    relax_to_equilibrium, volume_constrained_eigenmodes,
)

R0 = 40e-9
GAMMA_S = 1.0
NZ = 40


def lowest_eig_at(lam_over_R0, gamma_gb, gb_indices, n_relax=2000):
    L = lam_over_R0 * R0
    dz = L / NZ
    R = np.full(NZ, R0)
    if gamma_gb > 0:
        R = relax_to_equilibrium(R, dz, GAMMA_S, gamma_gb, gb_indices, "periodic",
                                  n_steps=n_relax, step_scale=1.0)
    evals, evecs, lam_lagrange, gnorm = volume_constrained_eigenmodes(
        R, dz, GAMMA_S, gamma_gb, gb_indices, "periodic", n_modes=1)
    return evals[0]


def find_crossing(gamma_gb, gb_indices, lo=4.5, hi=6.3, n=8):
    xs = np.linspace(lo, hi, n)
    ys = [lowest_eig_at(x, gamma_gb, gb_indices) for x in xs]
    for i in range(len(xs) - 1):
        if (ys[i] < 0) != (ys[i + 1] < 0):
            frac = -ys[i] / (ys[i + 1] - ys[i])
            return xs[i] + frac * (xs[i + 1] - xs[i]), list(zip(xs, ys))
    return float("nan"), list(zip(xs, ys))


def psi_to_gamma_gb(psi_deg, gamma_s=1.0):
    return 2.0 * gamma_s * math.cos(math.radians(psi_deg) / 2.0)


if __name__ == "__main__":
    print("--- Section 8: single-crystal (gamma_gb=0) validation ---")
    xc0, pts0 = find_crossing(0.0, [])
    for x, y in pts0:
        print(f"  lam/R0={x:.4f} lowest_eig={y:.4e}")
    print(f"crossing lam_c/R0 = {xc0:.5f} vs 2*pi={2*math.pi:.5f} rel_err={abs(xc0-2*math.pi)/(2*math.pi):.5f}\n")

    print("--- Section 8b: GB destabilization trend ---")
    results = []
    for psi_deg in (180.0, 140.0, 120.0, 100.0):
        gamma_gb = psi_to_gamma_gb(psi_deg) if psi_deg < 180 else 0.0
        xc, pts = find_crossing(gamma_gb, [0], lo=3.5, hi=6.3, n=8)
        results.append((psi_deg, gamma_gb, xc))
        print(f"psi={psi_deg:.0f} gamma_gb={gamma_gb:.4f}: crossing lam_c/R0={xc:.4f}")
        for x, y in pts:
            print(f"    lam/R0={x:.4f} lowest_eig={y:.4e}")

    print("\n--- summary ---")
    for psi_deg, gamma_gb, xc in results:
        print(f"psi={psi_deg:.0f} gamma_gb={gamma_gb:.4f} lam_c/R0={xc:.4f}")
    monotone = all(results[i][2] >= results[i + 1][2] for i in range(len(results) - 1))
    print(f"\nlam_c/R0 monotonically DECREASES as psi decreases (gamma_gb increases): {monotone}")
