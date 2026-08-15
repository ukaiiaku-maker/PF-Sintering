"""Milestone 16M Sections 17-19: robust TJ/curvature-consensus stress
measurement, completing M16L's single largest flagged scope gap (Section
14 "Remaining limitations": "the full curvature-consensus-gate
architecture ... was NOT built this session").

Combines THREE INDEPENDENT curvature estimators evaluated from the SAME
PF state (same traced free-surface contour / same field, Section 19 --
never a stale tracker position against a fresh morphology):

  1. Circle-fit family (`pf_sintering.hussein_neck_stress.neck_curvature_windows`,
     Kasa algebraic least-squares) at FIXED PHYSICAL half-window widths
     {6, 9, 12, 18, 24} nm (Section 17) -- consensus value = median of the
     finite window results.
  2. Local quadratic-polynomial fit to the (z, R(z)) contour in a fixed
     12 nm half-window, analytic curvature from the fit's own R', R''
     (a genuinely different fitting basis from the circle fit -- a
     Cartesian polynomial in z rather than an algebraic circle in the
     (z,R) plane).
  3. Diffuse-interface curvature computed DIRECTLY from the phase field
     f's own gradient (`-div(grad f/|grad f|)`, axisymmetric form), NOT
     from the extracted R(z) contour at all -- the only one of the three
     that does not depend on contour tracing/extrema-finding.

The TJ location itself (z_gb, a_contact, hence X_neck=2*a_contact) is
fixed ONCE per call from `find_tj_from_contour` (path-continuous,
sub-grid f=0.5/grain-identity intersection) and used identically by all
three curvature estimators and downstream sigma_Hussein evaluations, so
disagreement reflects genuine estimator disagreement, not different TJ
definitions.

`stress_valid` (Section 18): True only if >=2 of the three estimators
produce a finite, positive r_neck AND their relative spread
((max-min)/mean of the resulting sigma_Hussein_MPa values) is
<=15% (SPREAD_TOL_INITIAL). When False, the caller must NOT use this
sample to drive nucleation or RBM (Section 18) -- geometry keeps
evolving, but the kinetic update is paused for that sample.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from pf_sintering.hussein_neck_stress import fit_local_circle, hussein_eq1b_sigma, neck_curvature_windows
from pf_sintering.m16k_neck_tracking import find_tj_from_contour

SPREAD_TOL_INITIAL = 0.15
SPREAD_TOL_PREFERRED = 0.10
FIXED_WINDOWS_NM = (6.0, 9.0, 12.0, 18.0, 24.0)
# Diagnostic testing against the M16K frozen state (near the 45MPa
# activation point) showed the local quadratic-polynomial fit is
# numerically unstable (non-monotonic, order-of-magnitude swings) at
# windows >=9nm for this geometry's neck profile -- a small window (most
# locally accurate Taylor expansion) is used instead of the 12nm window
# originally planned.
POLY_HALF_WINDOW_NM = 6.0
DIFFUSE_WINDOW_NM = 9.0


def circle_fit_consensus_r_neck(R_of_z, z, z_gb):
    """Method 1: median r_neck over fixed physical half-windows
    {6,9,12,18,24}nm (Section 17). `neck_curvature_windows`'s
    `window_widths_in_W` multiplies by W -- passing W=1nm makes the
    multipliers themselves the absolute half-window widths in nm."""
    windows = neck_curvature_windows(R_of_z, z, z_gb, 1e-9, window_widths_in_W=FIXED_WINDOWS_NM)
    r_vals = [w["r_neck"] for w in windows if np.isfinite(w["r_neck"]) and w["r_neck"] > 0]
    if not r_vals:
        return float("nan"), windows
    return float(np.median(r_vals)), windows


def polynomial_curvature_r_neck(R_of_z, z, z_gb, half_window_nm=POLY_HALF_WINDOW_NM):
    """Method 2: local quadratic-polynomial fit to R(z) in a fixed
    12nm half-window, analytic curvature at z_gb from the fit
    coefficients -- independent fitting basis from the circle fit."""
    half_window = half_window_nm * 1e-9
    mask = np.abs(z - z_gb) <= half_window
    idx = np.where(mask & np.isfinite(R_of_z))[0]
    if len(idx) < 4:
        return float("nan")
    z_pts, R_pts = z[idx], R_of_z[idx]
    try:
        coeffs = np.polyfit(z_pts - z_gb, R_pts, deg=2)  # R(z) ~= c2*(z-z_gb)^2 + c1*(z-z_gb) + c0
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    c2, c1, _c0 = coeffs
    Rp = c1  # dR/dz at z=z_gb (since expansion is about z_gb)
    Rpp = 2.0 * c2
    denom = (1.0 + Rp ** 2) ** 1.5
    if denom <= 0:
        return float("nan")
    kappa_m = -Rpp / denom
    if not np.isfinite(kappa_m) or kappa_m == 0:
        return float("nan")
    return float(1.0 / abs(kappa_m))


def diffuse_interface_r_neck(f, r_c, z, z_gb, r_gb, window_nm=DIFFUSE_WINDOW_NM, smoothing_sigma_cells=2.0):
    """Method 3: diffuse-interface curvature computed directly from the
    phase field's own gradient, axisymmetric divergence of the unit
    normal n=grad(f)/|grad(f)|:
        kappa_meridional_diffuse = -(d(n_z)/dz + d(n_r)/dr)
    evaluated at the grid point nearest (z_gb, r_gb) via a finite
    difference of the normal vector taken over a `window_nm` physical
    offset (NOT single-grid-cell spacing -- diagnostic testing against
    the M16K frozen state showed a 1-2 cell offset is dominated by the
    diffuse interface's OWN intrinsic thickness, order dx, rather than
    the mesoscale neck curvature; a window comparable to the other two
    estimators' windows recovers a consistent order-of-magnitude
    result), on a lightly Gaussian-smoothed copy of f (does NOT use the
    traced R(z) contour at all -- an independent data source from
    methods 1 and 2)."""
    Nz, Nr = f.shape
    dz = float(z[1] - z[0])
    dr = float(r_c[1] - r_c[0])
    offset_cells = max(1, int(round(window_nm * 1e-9 / dz)))
    if Nz < 2 * offset_cells + 3 or Nr < 2 * offset_cells + 3:
        return float("nan")
    fs = gaussian_filter(f, sigma=smoothing_sigma_cells, mode="nearest")
    j_z = int(np.argmin(np.abs(z - z_gb)))
    j_r = int(np.argmin(np.abs(r_c - r_gb)))
    j_z = min(max(j_z, offset_cells + 1), Nz - offset_cells - 2)
    j_r = min(max(j_r, offset_cells + 1), Nr - offset_cells - 2)

    def normal_at(jz, jr):
        Fz = (fs[jz + 1, jr] - fs[jz - 1, jr]) / (2.0 * dz)
        Fr = (fs[jz, jr + 1] - fs[jz, jr - 1]) / (2.0 * dr)
        mag = np.hypot(Fz, Fr)
        if mag < 1e-8:
            return 0.0, 0.0
        return Fz / mag, Fr / mag

    nz_p, _ = normal_at(j_z + offset_cells, j_r)
    nz_m, _ = normal_at(j_z - offset_cells, j_r)
    _, nr_p = normal_at(j_z, j_r + offset_cells)
    _, nr_m = normal_at(j_z, j_r - offset_cells)
    dnz_dz = (nz_p - nz_m) / (2.0 * offset_cells * dz)
    dnr_dr = (nr_p - nr_m) / (2.0 * offset_cells * dr)

    kappa_meridional_diffuse = -(dnz_dz + dnr_dr)
    if not np.isfinite(kappa_meridional_diffuse) or kappa_meridional_diffuse == 0:
        return float("nan")
    return float(1.0 / abs(kappa_meridional_diffuse))


def stress_consensus(f, particle, substrate, r_c, z, R_of_z, z_gb, a_contact, gamma_s, gamma_gb,
                      spread_tol=SPREAD_TOL_INITIAL):
    """Section 17-19: single entry point. `z_gb`/`a_contact` MUST be
    supplied by the caller from a PATH-CONTINUOUS tracker
    (`pf_sintering.m16k_neck_tracking.NeckTracker`, the same one M16L
    validated for "track one connected free-surface branch" continuity)
    evaluated against the SAME (f, particle, substrate) state passed here
    -- never a stale position from a previous PF state (Section 19). All
    three curvature estimators, and X_neck, are evaluated against this
    SAME (z_gb, a_contact), so disagreement reflects genuine estimator
    disagreement, not different TJ definitions.

    `find_tj_from_contour` is still run as an independent CROSS-CHECK
    (its z_tj/agreement are reported diagnostically) but, per M16L's own
    validated convention, does NOT itself set X_neck -- a direct
    grain-identity contour crossing can land on the internal planar GB
    reference rather than the curved neck when the two are close in z,
    as observed empirically in this milestone; the R(z)-extremum-based
    tracker position remains the authoritative TJ/X_neck source.

    Returns dict with: z_gb, a_contact_nm, X_neck_nm, z_tj_crosscheck_nm,
    tj_agreement_nm, r_neck_circle_nm, r_neck_poly_nm, r_neck_diffuse_nm,
    sigma_circle_MPa, sigma_poly_MPa, sigma_diffuse_MPa,
    sigma_consensus_MPa (mean of finite estimates), n_finite_estimates,
    spread_pct, stress_valid, spread_tol_used."""
    tj = find_tj_from_contour(f, particle, substrate, r_c, z, R_of_z, z_gb)
    if not (np.isfinite(z_gb) and np.isfinite(a_contact) and a_contact > 0):
        return dict(z_gb=float("nan"), a_contact_nm=float("nan"), X_neck_nm=float("nan"),
                    z_tj_crosscheck_nm=float("nan"), tj_agreement_nm=float("nan"),
                    r_neck_circle_nm=float("nan"), r_neck_poly_nm=float("nan"), r_neck_diffuse_nm=float("nan"),
                    sigma_circle_MPa=float("nan"), sigma_poly_MPa=float("nan"), sigma_diffuse_MPa=float("nan"),
                    sigma_consensus_MPa=float("nan"), n_finite_estimates=0, spread_pct=float("nan"),
                    stress_valid=False, spread_tol_used=spread_tol)

    X_neck = 2.0 * a_contact
    r_circle, _ = circle_fit_consensus_r_neck(R_of_z, z, z_gb)
    r_poly = polynomial_curvature_r_neck(R_of_z, z, z_gb)
    r_diffuse = diffuse_interface_r_neck(f, r_c, z, z_gb, a_contact)

    def sigma_or_nan(r_neck):
        if not (np.isfinite(r_neck) and r_neck > 0):
            return float("nan")
        sigma, *_ = hussein_eq1b_sigma(r_neck, X_neck, gamma_s, gamma_gb)
        return sigma / 1e6 if np.isfinite(sigma) else float("nan")

    sigma_circle = sigma_or_nan(r_circle)
    sigma_poly = sigma_or_nan(r_poly)
    sigma_diffuse = sigma_or_nan(r_diffuse)

    finite_sigmas = [s for s in (sigma_circle, sigma_poly, sigma_diffuse) if np.isfinite(s)]
    n_finite = len(finite_sigmas)
    if n_finite >= 2:
        mean_s = float(np.mean(finite_sigmas))
        spread_pct = (100.0 * (max(finite_sigmas) - min(finite_sigmas)) / abs(mean_s)) if mean_s != 0 else float("inf")
        stress_valid = spread_pct <= (spread_tol * 100.0)
        sigma_consensus = mean_s
    else:
        spread_pct = float("nan")
        stress_valid = False
        sigma_consensus = finite_sigmas[0] if finite_sigmas else float("nan")

    return dict(z_gb=z_gb, a_contact_nm=a_contact * 1e9, X_neck_nm=X_neck * 1e9,
                z_tj_crosscheck_nm=tj["z_tj"] * 1e9 if np.isfinite(tj["z_tj"]) else float("nan"),
                tj_agreement_nm=tj["agreement_nm"],
                r_neck_circle_nm=r_circle * 1e9 if np.isfinite(r_circle) else float("nan"),
                r_neck_poly_nm=r_poly * 1e9 if np.isfinite(r_poly) else float("nan"),
                r_neck_diffuse_nm=r_diffuse * 1e9 if np.isfinite(r_diffuse) else float("nan"),
                sigma_circle_MPa=sigma_circle, sigma_poly_MPa=sigma_poly, sigma_diffuse_MPa=sigma_diffuse,
                sigma_consensus_MPa=sigma_consensus, n_finite_estimates=n_finite, spread_pct=spread_pct,
                stress_valid=stress_valid, spread_tol_used=spread_tol)
