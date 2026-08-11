"""Milestone 16F (corrected): reference implementations of the
Hussein/Abdeljawad two-mode geometry, in two DISTINCT, NOT-to-be-conflated
forms:

(A) `beta1`/`beta2`/`lambda_c_over_R_eq4` -- a LITERAL transcription of the
    printed closed-form Eqs. 4-5, taking `e1bar`, `e2bar`, and the dihedral
    angle `psi` (degrees) directly and returning `lambda_critical/R_cyl` in
    closed form (no quadrature, no energy functional at all). This is THE
    printed Eq.-4 criterion and is what "PASS/FAIL against Figure 4" checks
    must use.

(B) `delta_E_full_area`/`lambda_c_full_area` -- a full numerical
    surface-of-revolution quadrature of the Eq.-2-style energy functional
    `Epert = gamma_s*A_surface[R(z)] + gamma_gb*2*pi*R(z1)^2` compared
    against a reference cylinder, with `lambda_critical` located by root
    find on `DeltaE=0`. This is a SEPARATE analytical/numerical energy
    treatment, evaluated independently of (A). It is NOT the printed Eq.-4
    threshold and its root must never be reported as such. Comparing (A)
    against (B) quantifies the gap between the closed-form Eq.-4/5 result
    and a direct full-area energy quadrature of the same nominal physical
    model -- both are legitimate, but they are different calculations, and
    disagreement between them is reported as exactly that (a difference
    between two analytical/numerical energy treatments), not attributed to
    an unresolved/unavailable-source normalization mystery.

Shared geometry (both treatments):

    R(z)/R_cyl = R0/R_cyl + e1bar*cos(2*pi*z/lambda) + e2bar*cos(pi*z/lambda)
    R0/R_cyl = sqrt(1 - (e1bar^2+e2bar^2)/2)   (volume-matching to a
               plain cylinder of radius R_cyl over one period 2*lambda)
    domain 0 <= z < 2*lambda
    z1/lambda = (1/pi)*acos(-e2bar/(4*e1bar)),  z2/lambda = 2 - z1/lambda
       (the two GB trough positions, identical minimum radius)
    Epert_GB = 2*pi*gamma_GB*R(z_min)^2   (both GB disks, per the source)

Neither (A) nor (B) calls any PF or Hessian routine from the rest of the
project.
"""
from __future__ import annotations

import math

import numpy as np


def z1_over_lambda(e1bar: float, e2bar: float) -> float:
    return (1.0 / math.pi) * math.acos(-e2bar / (4.0 * e1bar))


def R0_over_Rcyl(e1bar: float, e2bar: float) -> float:
    return math.sqrt(1.0 - (e1bar ** 2 + e2bar ** 2) / 2.0)


def R_over_Rcyl(u, e1bar: float, e2bar: float):
    """u = z/lambda, dimensionless axial coordinate."""
    R0 = R0_over_Rcyl(e1bar, e2bar)
    return R0 + e1bar * np.cos(2 * math.pi * u) + e2bar * np.cos(math.pi * u)


def psi_to_gamma_ratio(psi_deg: float) -> float:
    return 2.0 * math.cos(math.radians(psi_deg) / 2.0)


# ---------------------------------------------------------------------------
# (A) Literal printed Eqs. 4-5 (closed form, no quadrature).
# ---------------------------------------------------------------------------

def beta1(e1bar: float, e2bar: float) -> float:
    e1, e2 = e1bar, e2bar
    num = (e1 ** 2 - e2 ** 2 / 2.0 - 4.0 * e1 - e2 ** 2 / (2.0 * e1) + e2 ** 4 / (32.0 * e1 ** 2))
    den = e1 ** 2 + e2 ** 2
    return num / den


def beta2(e1bar: float, e2bar: float) -> float:
    e1, e2 = e1bar, e2bar
    num = 4.0 * e1 ** 2 + e2 ** 2 + 3.0 * e1 * e2 ** 2
    den = e1 ** 2 + e2 ** 2
    return num / den


def lambda_c_over_R_eq4(e1bar: float, e2bar: float, psi_deg: float) -> float:
    """Literal transcription of the printed Eq. 4:

        lambda_c/R = beta1*cos(psi/2) + sqrt(beta1^2*cos(psi/2)^2 + pi^2*beta2)

    Verified exactly against the required checks for e1bar=e2bar=0.4
    (beta1=-5.359375, beta2=3.1, psi=160 -> 4.6784429106, psi=80 ->
    2.7829537965) and e1bar=e2bar=0.02 (psi=160 -> 0.6304147176, psi=80 ->
    0.1450922158)."""
    b1 = beta1(e1bar, e2bar)
    b2 = beta2(e1bar, e2bar)
    c = math.cos(math.radians(psi_deg) / 2.0)
    return b1 * c + math.sqrt(b1 ** 2 * c ** 2 + math.pi ** 2 * b2)


def lambda_over_lambda_c_eq4(lam_over_Rcyl: float, e1bar: float, e2bar: float, psi_deg: float) -> float:
    return lam_over_Rcyl / lambda_c_over_R_eq4(e1bar, e2bar, psi_deg)


# ---------------------------------------------------------------------------
# (B) Full direct Eq.-2-style surface-of-revolution quadrature. A SEPARATE
# energy treatment -- its root is `lambda_c_full_area`, never "the Eq.-4
# threshold".
# ---------------------------------------------------------------------------

def surface_area_over_Rcyl2(lam_over_Rcyl: float, e1bar: float, e2bar: float, n_quad: int = 20000) -> float:
    """Exact axisymmetric surface-of-revolution area (dimensionless,
    divided by R_cyl^2), via fine numerical quadrature (trapezoidal on
    a fine grid; verified converged by doubling n_quad) of
    `2*pi*integral R(z)*sqrt(1+(dR/dz)^2) dz` over the full period
    `0<=z<2*lambda`."""
    Lam = lam_over_Rcyl
    u = np.linspace(0.0, 2.0, n_quad, endpoint=False)
    du = u[1] - u[0]
    R = R_over_Rcyl(u, e1bar, e2bar)
    dRdu = -2 * math.pi * e1bar * np.sin(2 * math.pi * u) - math.pi * e2bar * np.sin(math.pi * u)
    dRdz = dRdu / Lam
    integrand = R * np.sqrt(1.0 + dRdz ** 2)
    area = 2 * math.pi * float(np.sum(integrand) * du * Lam)
    return area


def R_gb_over_Rcyl(e1bar: float, e2bar: float) -> float:
    z1u = z1_over_lambda(e1bar, e2bar)
    return float(R_over_Rcyl(np.array([z1u]), e1bar, e2bar)[0])


def E_pert_full_area(lam_over_Rcyl: float, e1bar: float, e2bar: float, gamma_ratio: float,
                      n_quad: int = 20000) -> float:
    """(Epert)/(gamma_s*R_cyl^2), gamma_ratio=gamma_gb/gamma_s. Uses the
    source's stated Epert_GB=2*pi*gamma_GB*R(zmin)^2 for the two GB disks."""
    A = surface_area_over_Rcyl2(lam_over_Rcyl, e1bar, e2bar, n_quad=n_quad)
    Rgb = R_gb_over_Rcyl(e1bar, e2bar)
    return A + gamma_ratio * 2 * math.pi * Rgb ** 2


def E_cyl_full_area(lam_over_Rcyl: float, gamma_ratio: float) -> float:
    """(Ecyl)/(gamma_s*R_cyl^2)."""
    L_over_Rcyl = 2.0 * lam_over_Rcyl
    A = 2 * math.pi * L_over_Rcyl
    return A + gamma_ratio * 2 * math.pi * 1.0 ** 2


def delta_E_full_area(lam_over_Rcyl: float, e1bar: float, e2bar: float, gamma_ratio: float,
                       n_quad: int = 20000) -> float:
    return (E_pert_full_area(lam_over_Rcyl, e1bar, e2bar, gamma_ratio, n_quad=n_quad)
            - E_cyl_full_area(lam_over_Rcyl, gamma_ratio))


def lambda_c_full_area(e1bar: float, e2bar: float, gamma_ratio: float, lo: float = 0.05, hi: float = 8.0,
                        n_quad: int = 20000, tol: float = 1e-10) -> float:
    """Bisection root-find of delta_E_full_area(lambda/Rcyl)=0. This is
    the full-area-quadrature energy treatment's own threshold -- a
    diagnostic quantity distinct from `lambda_c_over_R_eq4`, NOT to be
    reported as "the Eq.-4 threshold". Requires a sign change to be
    bracketed in [lo,hi] -- raises if not found."""
    f_lo = delta_E_full_area(lo, e1bar, e2bar, gamma_ratio, n_quad=n_quad)
    f_hi = delta_E_full_area(hi, e1bar, e2bar, gamma_ratio, n_quad=n_quad)
    if (f_lo < 0) == (f_hi < 0):
        raise RuntimeError(f"no sign change bracketed in [{lo},{hi}]: f_lo={f_lo:.6e} f_hi={f_hi:.6e}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = delta_E_full_area(mid, e1bar, e2bar, gamma_ratio, n_quad=n_quad)
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return mid
        if (f_mid < 0) == (f_lo < 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)
