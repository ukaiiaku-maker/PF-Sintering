"""Milestone 16I: axisymmetric free-surface-area / surface-to-volume
diagnostics, computed directly from the measured f=0.5 contour R(z)
(NOT the analytic two-mode formula -- these functions operate on
whatever R(z) the current PF state actually has, at any point in a
run, via scripts.m16a_gb_benchmark.measure_R_of_z).

    A_free = 2*pi * integral[ R(z)*sqrt(1+(dR/dz)^2) ] dz

computed by simple trapezoidal quadrature on the already-discretized
z-grid, using central finite differences for dR/dz and skipping any
non-finite (contour-not-found) points -- DIAGNOSTIC ONLY, does not feed
back into any evolution equation, and does NOT include the solid-solid
GB disk area (A_GB=pi*a_contact^2 is tracked separately, exactly as
Milestone 16I Section 10 requires).
"""
from __future__ import annotations

import math

import numpy as np


def free_surface_area_of_revolution(R_of_z, z):
    """A_free (m^2) via trapezoidal quadrature of the surface-of-
    revolution integrand over the finite (contour-resolved) portion of
    the domain. Returns nan if fewer than 2 finite points exist."""
    R_of_z = np.asarray(R_of_z, dtype=float)
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(R_of_z)
    if int(np.sum(finite)) < 2:
        return float("nan")
    Rf = R_of_z[finite]
    zf = z[finite]
    dRdz = np.gradient(Rf, zf)
    integrand = 2 * math.pi * Rf * np.sqrt(1.0 + dRdz ** 2)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz -> trapezoid
    return float(trapezoid(integrand, zf))


def surface_to_volume_metrics(A_free, A_GB, V_solid):
    """Returns dict(S_over_V, GB_over_V, total_interface_over_V), all in
    1/length units matching A_free/A_GB/V_solid's own units. nan-safe
    (returns nan rather than raising if V_solid<=0)."""
    if not (np.isfinite(V_solid) and V_solid > 0):
        return dict(S_over_V=float("nan"), GB_over_V=float("nan"), total_interface_over_V=float("nan"))
    return dict(
        S_over_V=A_free / V_solid if np.isfinite(A_free) else float("nan"),
        GB_over_V=A_GB / V_solid if np.isfinite(A_GB) else float("nan"),
        total_interface_over_V=(A_free + A_GB) / V_solid if np.isfinite(A_free) and np.isfinite(A_GB) else float("nan"),
    )
