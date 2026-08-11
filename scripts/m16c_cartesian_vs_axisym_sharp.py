"""Milestone 16C Section 14: run the EXACT SAME sharp-interface
particle-between-two-substrates geometry in both the axisymmetric and
Cartesian (per-unit-depth) reductions, matched on domain length L and
initial profile shape/scale, to directly identify the source of the
Cartesian-vs-axisymmetric sign reversal M16B found on its flat-
substrate spheroid (Cartesian narrows, axisymmetric broadens).

Since volume (axisymmetric, ~length^3) and cross-sectional area
(Cartesian, ~length^2) are dimensionally different constraints, the two
reductions cannot be matched on identical volume/area; instead they
are matched on identical MAXIMUM PROFILE EXTENT (R_REF=40nm for both R
and h) and identical L, gamma_s, gamma_gb, and initial barrel-shape
family -- the natural, physically meaningful matching for isolating
the geometric-reduction effect itself, not a dimensional-scaling
artifact.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.sharp_interface_stability import (  # noqa: E402
    cartesian_area, cartesian_constrained_eigenmodes, cartesian_relax_to_equilibrium,
    relax_to_equilibrium, total_volume, volume_constrained_eigenmodes,
)

GAMMA_S = 1.0
GAMMA_GB = 0.0
R_REF = 40e-9
NZ = 48


def axisym_classify(L_nm, n_relax=4000):
    L = L_nm * 1e-9
    dz = L / (NZ - 1)
    z = np.arange(NZ) * dz
    a_guess, Rbody_guess = 0.4 * R_REF, 1.3 * R_REF
    R = a_guess + (Rbody_guess - a_guess) * np.sin(math.pi * z / L)
    V_target = (4.0 / 3.0) * math.pi * R_REF ** 3
    R = R * math.sqrt(V_target / total_volume(R, dz, "capped"))
    Req = relax_to_equilibrium(R, dz, GAMMA_S, GAMMA_GB, [0, NZ - 1], "capped", n_steps=n_relax, step_scale=1.0)
    evals, evecs, lam, gnorm = volume_constrained_eigenmodes(Req, dz, GAMMA_S, GAMMA_GB, [0, NZ - 1],
                                                               "capped", n_modes=2)
    return dict(L_nm=L_nm, neck_nm=float(Req[0] * 1e9), body_nm=float(np.max(Req) * 1e9),
                lowest_eig=float(evals[0]))


def cartesian_classify(L_nm, n_relax=4000):
    """A_target is a FIXED reference (independent of L, matching the
    axisymmetric case's fixed V_target philosophy) -- an earlier
    version recomputed it from the L-dependent barrel guess each call,
    which made it scale ~proportionally with L, silently forcing the
    SAME uniform neck value (h=A/(2L), constant when A~L) at every L
    -- a real bug, not a physical finding (confirmed directly: the
    'cart_neck' column was IDENTICAL, 26.396nm, at all 9 tested L
    values, which is the smoking gun -- a fixed reference should NOT
    do that)."""
    L = L_nm * 1e-9
    dz = L / (NZ - 1)
    z = np.arange(NZ) * dz
    a_guess, hbody_guess = 0.4 * R_REF, 1.3 * R_REF
    h = a_guess + (hbody_guess - a_guess) * np.sin(math.pi * z / L)
    L_REF_FOR_AREA = 100e-9  # fixed reference length used ONLY to define A_target, independent of the swept L
    z_ref = np.linspace(0, L_REF_FOR_AREA, NZ)
    h_ref = a_guess + (hbody_guess - a_guess) * np.sin(math.pi * z_ref / L_REF_FOR_AREA)
    A_target = cartesian_area(h_ref, L_REF_FOR_AREA / (NZ - 1), "capped")
    h = h * math.sqrt(A_target / cartesian_area(h, dz, "capped"))
    heq = cartesian_relax_to_equilibrium(h, dz, GAMMA_S, GAMMA_GB, [0, NZ - 1], "capped", n_steps=n_relax,
                                          step_scale=1.0)
    evals, evecs, lam, gnorm = cartesian_constrained_eigenmodes(heq, dz, GAMMA_S, GAMMA_GB, [0, NZ - 1],
                                                                  "capped", n_modes=2)
    return dict(L_nm=L_nm, neck_nm=float(heq[0] * 1e9), body_nm=float(np.max(heq) * 1e9),
                lowest_eig=float(evals[0]))


if __name__ == "__main__":
    print(f"{'L(nm)':>8s} {'axisym_neck':>12s} {'axisym_eig':>12s} | {'cart_neck':>10s} {'cart_eig':>10s}")
    for L_nm in [40, 60, 80, 100, 120, 140, 160, 180, 200]:
        a = axisym_classify(L_nm, n_relax=4000)
        c = cartesian_classify(L_nm, n_relax=4000)
        a_cls = "UNSTABLE" if a["lowest_eig"] < -1e-3 else ("STABLE" if a["lowest_eig"] > 1e-3 else "NEUTRAL")
        c_cls = "UNSTABLE" if c["lowest_eig"] < -1e-3 else ("STABLE" if c["lowest_eig"] > 1e-3 else "NEUTRAL")
        print(f"{L_nm:8.0f} {a['neck_nm']:12.3f} {a['lowest_eig']:12.4e}({a_cls:>8s}) | "
              f"{c['neck_nm']:10.3f} {c['lowest_eig']:10.4e}({c_cls:>8s})")
