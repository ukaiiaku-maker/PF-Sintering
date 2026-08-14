"""Milestone 16K Section 12: a SECOND, independent neck-stress estimate,
named distinctly (`sigma_potential`) and never mixed with `sigma_Hussein`
(pf_sintering.hussein_neck_stress.hussein_eq1b_sigma).

`sigma_Hussein` is the literal Hussein et al. Eq. 1b form:
    sigma_Hussein = gamma_s*[1/r_neck - C_GB/X_neck]
(a 1D neck-curvature-minus-neck-width difference of terms).

`sigma_potential` here is the full axisymmetric two-principal-curvature
Young-Laplace capillary pressure at the neck -- a genuinely different
physical construction already present in this codebase
(`hussein_neck_stress.local_young_laplace_pressure`, explicitly retired
there as a "sintering stress" name specifically so it would not be
conflated with sigma_Hussein, and reused here for exactly its intended
purpose: an independent cross-check):

    sigma_potential = gamma_s*(kappa_meridional + kappa_azimuthal)

kappa_meridional: the SIGNED local curvature of the free-surface contour
    in the (r,z) meridian plane at the neck (negative/concave at a
    groove, from the same circle fit used for sigma_Hussein's r_neck).
kappa_azimuthal: the hoop curvature of a surface of revolution at radius
    a_contact, kappa_azimuthal = +1/a_contact (positive/convex -- ANY
    point at finite radius on a body of revolution has this sign,
    independent of the meridional groove shape).

At a real neck these two principal curvatures have OPPOSITE signs (a
saddle point, negative Gaussian curvature) so sigma_potential is also a
difference/partial-cancellation of two terms, structurally analogous to
sigma_Hussein's own two-term form -- but built from entirely independent
inputs (a genuine second physical estimate, not a rearrangement of the
first).
"""
from __future__ import annotations

import math

from .hussein_neck_stress import local_young_laplace_pressure


def sigma_potential(kappa_meridional_signed: float, a_contact: float, gamma_s: float) -> tuple:
    """Returns (sigma_potential, kappa_meridional_signed, kappa_azimuthal).
    `kappa_meridional_signed` should be the SIGNED curvature from
    `signed_curvature_from_fit` (negative at a groove); `a_contact` is the
    contact/neck radius (X_neck/2), used for the azimuthal term."""
    if not (math.isfinite(a_contact) and a_contact > 0):
        return float("nan"), kappa_meridional_signed, float("nan")
    kappa_azimuthal = 1.0 / a_contact
    sigma = local_young_laplace_pressure(kappa_meridional_signed, kappa_azimuthal, gamma_s)
    return sigma, kappa_meridional_signed, kappa_azimuthal
