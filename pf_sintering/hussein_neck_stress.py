"""Milestone 16G revision, Sections 9-15: literal implementation of the
Hussein et al. (ACS Appl. Nano Mater. 2021, 4, 8039-8049,
DOI 10.1021/acsanm.1c01322) Eq. 1b neck-stress formulation --
`mu_N=-Omega*sigma`, `sigma = gamma_s*[1/r - sqrt(1-(gamma_gb/(2*gamma_s))^2)/X]`
-- implemented literally, SIGNED (no absolute value), replacing the
earlier `sigma_local=|gamma_s*(kappa_meridional+kappa_azimuthal)|`
diagnostic used in the first draft of this milestone (retained, if at
all, only under the explicitly non-"sintering-stress" name
`local_young_laplace_pressure`).

Definitions used here (Sections 11-13 of the handoff):

  X_neck = 2*a_contact  (the paper's X is the FULL particle-neck width;
      our axisymmetric a_contact is the radial distance from the axis to
      the TJ/contact edge, i.e. a HALF-width -- X is twice that, not
      a_contact itself).

  r_neck: the LOCAL RADIUS OF CURVATURE OF THE PARTICLE FREE SURFACE at
      the neck region, extracted from a local circular-arc fit to the
      resolved f=0.5 contour over a finite physical window (NOT a
      single-cell second derivative) -- following the paper's own
      practice of fitting local geometry to noisy resolved contours
      rather than differentiating a raw grid field directly.

Sign convention for r_neck (documented explicitly, per Section 13's
requirement): the fit returns a magnitude r_fit>0 plus the fitted
circle's center. If the center lies on the VAPOR side of the local
contour point (center's R > local R, i.e. the surface curves away from
the solid -- the classical CONCAVE neck/groove shape, R''>0 in this
project's z-parametrized convention), the signed meridional curvature is
NEGATIVE (matching this project's existing kappa_meridional=-R''/(1+R'^2)
^1.5 convention used elsewhere, e.g. axisym_capillary_diagnostic.py) and
r_neck = 1/kappa_meridional is likewise NEGATIVE. If the center lies on
the SOLID side (a locally convex/bulging particle surface), r_neck is
POSITIVE. This mapping is applied once, here, and `1/r_neck` is used
directly in Eq. 1b with no further sign handling -- a neck (the expected
case at the retained contact) contributes a NEGATIVE `1/r_neck` term,
consistent with the classical result that sintering necks are in
tension.
"""
from __future__ import annotations

import math

import numpy as np


def fit_local_circle(z_pts, R_pts):
    """Kasa algebraic least-squares circle fit in the (z,R) meridional
    plane: minimizes sum((z-zc)^2+(R-Rc)^2-r^2)^2 via the standard
    linearization zi^2+Ri^2 = 2*zc*zi + 2*Rc*Ri + (r^2-zc^2-Rc^2).
    Returns (zc, Rc, r_fit) with r_fit>0. Requires >=3 points."""
    z_pts = np.asarray(z_pts, dtype=float)
    R_pts = np.asarray(R_pts, dtype=float)
    if len(z_pts) < 3:
        raise ValueError("need at least 3 points for a circle fit")
    A = np.column_stack([2 * z_pts, 2 * R_pts, np.ones_like(z_pts)])
    b = z_pts ** 2 + R_pts ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    zc, Rc, D = sol
    r2 = D + zc ** 2 + Rc ** 2
    if r2 <= 0:
        raise ValueError("degenerate circle fit (near-collinear points)")
    return float(zc), float(Rc), float(math.sqrt(r2))


def signed_curvature_from_fit(zc, Rc, z_local, R_local):
    """Maps a circle fit to a SIGNED meridional curvature at the local
    contour point (z_local,R_local), using this project's existing
    kappa_meridional sign convention (negative for a concave neck/groove,
    positive for a convex/bulging surface) -- see module docstring."""
    r_fit = math.hypot(z_local - zc, R_local - Rc)
    if r_fit <= 0:
        return float("nan")
    concave = Rc > R_local  # center on the vapor side => neck/groove
    return (-1.0 if concave else 1.0) / r_fit


def neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5, 2.5)):
    """Fits the local free-surface contour over each window (half-width
    = window_widths_in_W[k]*W, centered on z_gb) and returns a list of
    dicts (window_W, n_points, kappa_signed, r_neck_signed). Skips a
    window if it does not contain >=3 finite points."""
    results = []
    for half_w_in_W in window_widths_in_W:
        half_window = half_w_in_W * W
        mask = np.abs(z - z_gb) <= half_window
        idx = np.where(mask & np.isfinite(R_of_z))[0]
        if len(idx) < 3:
            results.append(dict(window_W=half_w_in_W, n_points=len(idx),
                                 kappa_signed=float("nan"), r_neck_signed=float("nan")))
            continue
        z_pts, R_pts = z[idx], R_of_z[idx]
        try:
            zc, Rc, r_fit = fit_local_circle(z_pts, R_pts)
            j_gb = idx[np.argmin(np.abs(z[idx] - z_gb))]
            kappa = signed_curvature_from_fit(zc, Rc, z[j_gb], R_of_z[j_gb])
            r_neck = 1.0 / kappa if (np.isfinite(kappa) and kappa != 0) else float("nan")
        except (ValueError, np.linalg.LinAlgError):
            kappa, r_neck = float("nan"), float("nan")
        results.append(dict(window_W=half_w_in_W, n_points=len(idx),
                             kappa_signed=kappa, r_neck_signed=r_neck))
    return results


def local_young_laplace_pressure(kappa_meridional, kappa_azimuthal, gamma_s=1.0):
    """RETIRED as a sintering-stress diagnostic (Section 9 of the
    handoff) -- kept only as a distinctly-named Young-Laplace capillary
    PRESSURE diagnostic, not to be reported as "sintering stress"."""
    return gamma_s * (kappa_meridional + kappa_azimuthal)


def hussein_eq1b_sigma(r_neck_signed, X_neck, gamma_s, gamma_gb):
    """Literal Eq. 1b: sigma = gamma_s*[1/r - C_GB/X], C_GB=sqrt(1-(gamma_gb/
    (2*gamma_s))^2). SIGNED, no absolute value. Returns (sigma,
    sigma_curvature, sigma_contact_gb, C_GB)."""
    ratio = gamma_gb / (2.0 * gamma_s)
    if abs(ratio) > 1.0:
        C_GB = float("nan")
    else:
        C_GB = math.sqrt(1.0 - ratio ** 2)
    sigma_curvature = gamma_s / r_neck_signed if np.isfinite(r_neck_signed) and r_neck_signed != 0 else float("nan")
    sigma_contact_gb = -gamma_s * C_GB / X_neck if np.isfinite(C_GB) else float("nan")
    sigma = sigma_curvature + sigma_contact_gb if np.isfinite(sigma_curvature) and np.isfinite(sigma_contact_gb) else float("nan")
    return sigma, sigma_curvature, sigma_contact_gb, C_GB


def mu_N_from_sigma(sigma, Omega):
    """Eq. 1b's mu_N=-Omega*sigma. Diagnostic only -- never feeds back
    into the PF evolution law."""
    return -Omega * sigma if np.isfinite(sigma) else float("nan")


def mu_S_far_field(a_particle, gamma_s, Omega):
    """Eq. 1a (diagnostic only, Section 15): mu_S=-Omega*gamma_s*(2/a_particle)
    for a far-field particle of effective radius a_particle."""
    return -Omega * gamma_s * (2.0 / a_particle)
