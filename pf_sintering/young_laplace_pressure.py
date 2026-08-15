"""Milestone 16K Section 8 (naming correction): a SECOND, independent
neck capillary-pressure estimate.

CORRECTION: this module was originally named `sintering_potential_stress`
and its function `sigma_potential`, incorrectly implying it was the
experimental particle-on-substrate "sintering potential / GB-area"
stress formulation from the literature. It is NOT that. It is the full
axisymmetric two-principal-curvature YOUNG-LAPLACE CAPILLARY PRESSURE at
the neck (already present in this codebase as
`hussein_neck_stress.local_young_laplace_pressure`, itself explicitly
retired from any "sintering stress" name in M16G for the same reason).
Renamed here to `young_laplace_pressure` to name it honestly. The actual
experimental sintering-potential / GB-area formulation is NOT
implemented in this codebase (no literature source was available to this
session) -- if it is added later it must be named separately and never
conflated with either this quantity or `sigma_Hussein`.

`sigma_Hussein` (pf_sintering.hussein_neck_stress.hussein_eq1b_sigma) is
the literal Hussein et al. Eq. 1b form:
    sigma_Hussein = gamma_s*[1/r_neck - C_GB/X_neck]

`young_laplace_pressure` here:
    p_YL = gamma_s*(kappa_meridional + kappa_azimuthal)

kappa_meridional: the SIGNED local curvature of the free-surface contour
    in the (r,z) meridian plane at the neck (negative/concave at a
    groove, from the same circle fit used for sigma_Hussein's r_neck).
kappa_azimuthal: the hoop curvature of a surface of revolution at radius
    a_contact, kappa_azimuthal = +1/a_contact (positive/convex -- ANY
    point at finite radius on a body of revolution has this sign,
    independent of the meridional groove shape).

At a real neck these two principal curvatures have OPPOSITE signs (a
saddle point, negative Gaussian curvature), so p_YL is also a
difference/partial-cancellation of two terms, structurally analogous to
sigma_Hussein's own two-term form -- but built from entirely independent
inputs (a genuine second physical estimate, not a rearrangement of the
first). The two are NOT expected to agree in sign or magnitude (see
M16K's report: this was already found and is consistent with M16G's own
prior finding when this quantity was first retired from the
"sintering stress" role).
"""
from __future__ import annotations

import math

from .hussein_neck_stress import local_young_laplace_pressure


def young_laplace_pressure(kappa_meridional_signed: float, a_contact: float, gamma_s: float) -> tuple:
    """Returns (p_YL, kappa_meridional_signed, kappa_azimuthal).
    `kappa_meridional_signed` should be the SIGNED curvature from
    `signed_curvature_from_fit` (negative at a groove); `a_contact` is the
    contact/neck radius (X_neck/2), used for the azimuthal term."""
    if not (math.isfinite(a_contact) and a_contact > 0):
        return float("nan"), kappa_meridional_signed, float("nan")
    kappa_azimuthal = 1.0 / a_contact
    p_YL = local_young_laplace_pressure(kappa_meridional_signed, kappa_azimuthal, gamma_s)
    return p_YL, kappa_meridional_signed, kappa_azimuthal
