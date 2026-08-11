"""Milestone 16F correction, item 9: the raw Figure-4 initial profile is
not guaranteed to be a stationary state of the constrained Lagrangian, so
the Hessian eigenvalue of mode 1 (the candidate inner-grain-shrinkage
mode, see scripts/m16f_eigenmode_decomposition.py) is local CURVATURE
information only -- it does not by itself say which direction the state
actually evolves from a non-stationary starting point. This script
computes the FIRST directional derivative of the raw sharp-interface
energy F along mode 1, in both signed directions, for a small-eps ladder,
to determine which orientation is actually downhill (energy-decreasing).

Mode 1 is oriented so that "+eps*v1" means grain 1 (inner grain) SHRINKS
(dV1<0) -- checked explicitly and flipped if the solver's raw eigenvector
sign came out the other way.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.hussein_eq4_reference import psi_to_gamma_ratio  # noqa: E402
from pf_sintering.sharp_interface_stability import energy, volume_constrained_eigenmodes  # noqa: E402

sys.path.insert(0, "scripts")
from m16f_eigenmode_decomposition import build_grid  # noqa: E402


def directional_derivative_scan(psi_deg, eps_list=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)):
    z, R, dz, lam, L, i1, i2 = build_grid()
    Nz = len(z)
    gamma_ratio = psi_to_gamma_ratio(psi_deg)
    evals, evecs, lam_lagrange, gnorm = volume_constrained_eigenmodes(
        R, dz, 1.0, gamma_ratio, [i1, i2], "periodic", n_modes=8)

    idx = np.arange(Nz)
    inner_mask = (idx > i1) & (idx < i2)

    v1 = evecs[:, 1] / np.linalg.norm(evecs[:, 1])
    dV1 = 2 * math.pi * float(np.sum(R[inner_mask] * v1[inner_mask])) * dz
    if dV1 > 0:
        v1 = -v1
        dV1 = -dV1
    assert dV1 < 0, "orientation check failed: +v1 must shrink grain 1"

    F0 = energy(R, dz, 1.0, gamma_ratio, [i1, i2], "periodic")

    rows = []
    for eps in eps_list:
        Fp = energy(R + eps * v1, dz, 1.0, gamma_ratio, [i1, i2], "periodic")
        Fm = energy(R - eps * v1, dz, 1.0, gamma_ratio, [i1, i2], "periodic")
        dFp = Fp - F0
        dFm = Fm - F0
        central_slope = (Fp - Fm) / (2 * eps)  # -> dF/deps|0 as eps->0
        rows.append(dict(eps=eps, dF_plus=dFp, dF_minus=dFm, central_slope=central_slope))

    return dict(psi_deg=psi_deg, eig_mode1=float(evals[1]), F0=F0, rows=rows)


if __name__ == "__main__":
    for psi in (160.0, 80.0):
        result = directional_derivative_scan(psi)
        print(f"\n=== psi={psi} (mode 1 eig={result['eig_mode1']:+.4e}, F0={result['F0']:.6f}) ===")
        print("+eps*v1 := grain 1 (inner) shrinks")
        for r in result["rows"]:
            downhill = "+v1 (shrink)" if r["dF_plus"] < r["dF_minus"] else "-v1 (grow)"
            print(f"  eps={r['eps']:.1e}  dF(+eps*v1)={r['dF_plus']:+.6e}  "
                  f"dF(-eps*v1)={r['dF_minus']:+.6e}  central_slope={r['central_slope']:+.6e}  "
                  f"downhill={downhill}")
