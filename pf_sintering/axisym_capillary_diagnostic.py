"""Milestone 16G: a genuinely AXISYMMETRIC local Young-Laplace capillary
PRESSURE diagnostic, built directly from the axisymmetric free-surface
profile R(z,t) -- NOT a port or reuse of capillary_stress.py's Cartesian
contour-tracing/Cahn-Hoffman-vector construction.

RETIRED, per explicit correction, from any "sintering stress" role: this
module's `local_young_laplace_pressure_at_index` is a distinct capillary
PRESSURE quantity, not the paper-based sintering-STRESS coordinate used
for this milestone's quantitative gates (see
`pf_sintering/hussein_neck_stress.py` for the literal, signed Hussein et
al. Eq.-1b implementation, which is the primary stress diagnostic). The
two are never mixed in this milestone's report.

For a surface of revolution r=R(z), the two principal curvatures at a
point are the standard differential-geometry results for a meridian
curve revolved about the z-axis:

    kappa_meridional(z) = -R''(z) / (1+R'(z)^2)^1.5
    kappa_azimuthal(z)  =  1 / (R(z)*sqrt(1+R'(z)^2))

(sign convention: a NECK, R''>0 at a trough, gives kappa_meridional<0 --
locally saddle-shaped). The mean curvature enters the Young-Laplace
capillary pressure jump directly:

    Delta_p(z) = gamma_s * (kappa_meridional(z) + kappa_azimuthal(z))

This is the local capillary pressure at the free surface nearest the
GB/contact, computed here directly from the diffuse-interface R(z,t)
measurement with no Cartesian machinery involved.

DIAGNOSTIC ONLY, as with the Cartesian capillary_stress.py -- does not
feed back into any evolution equation.
"""
from __future__ import annotations

import numpy as np


def local_curvatures(R_of_z, dz, idx):
    """R_of_z: 1-D array (same units as dz). Central finite differences
    at index idx (must not be the first/last element). Returns
    (kappa_meridional, kappa_azimuthal), same length units as 1/R_of_z."""
    Rp = (R_of_z[idx + 1] - R_of_z[idx - 1]) / (2.0 * dz)
    Rpp = (R_of_z[idx + 1] - 2.0 * R_of_z[idx] + R_of_z[idx - 1]) / (dz * dz)
    denom = (1.0 + Rp ** 2) ** 1.5
    kappa_m = -Rpp / denom
    kappa_a = 1.0 / (R_of_z[idx] * np.sqrt(1.0 + Rp ** 2))
    return float(kappa_m), float(kappa_a)


def young_laplace_pressure(kappa_m, kappa_a, gamma_s=1.0):
    return gamma_s * (kappa_m + kappa_a)


def local_young_laplace_pressure_at_index(R_of_z, dz, idx, gamma_s=1.0):
    """Capillary PRESSURE magnitude |Delta_p| at the given index
    (typically the current GB/contact trough location) -- NOT a
    sintering-stress diagnostic (see module docstring). Returns
    (pressure_magnitude, kappa_m, kappa_a, delta_p_signed)."""
    km, ka = local_curvatures(R_of_z, dz, idx)
    dp = young_laplace_pressure(km, ka, gamma_s=gamma_s)
    return abs(dp), km, ka, dp
