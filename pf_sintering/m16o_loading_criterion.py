"""Milestone 16O Sections 2-3: analytic Hussein Eq. 1b loading criterion.

sigma_s = gamma_s * (1/r - C/X),  C = sqrt(1 - (gamma_gb/(2*gamma_s))^2)

    d sigma/dt = gamma_s * [ -(dr/dt)/r^2 + C*(dX/dt)/X^2 ]

Decomposed into a curvature-loading term and a contact-width term:

    L_r = -(dr/dt)/r^2       (curvature-loading term)
    L_X = C*(dX/dt)/X^2      (contact-width term)
    L_total = L_r + L_X      (proportional to d sigma/dt via the gamma_s factor)

For intervals where dr/dt<0 and dX/dt<0 (both neck radius and contact
width shrinking), defining k_r=-dln(r)/dt>0, k_X=-dln(X)/dt>0:

    sigma increases  <=>  k_r/k_X > C*r/X

(direct algebraic rearrangement of L_total>0 under the shrinking-both
assumption -- NOT assumed to hold outside that regime, where L_total's
sign should be used directly instead).

This module implements ONLY the diagnostic decomposition; it does not
feed back into any PF evolution law.
"""
from __future__ import annotations

import numpy as np


def loading_terms_dt(t, r, X, gamma_s, C_GB):
    """t, r, X: 1-D arrays (same length, sorted by time). Returns a dict
    of arrays (length = len(t)-1, one per interval, associated with the
    interval midpoint) with:
        dt, dr_dt, dX_dt, L_r, L_X, L_total, predicted_sign_dsigma_dt,
        k_r, k_X, loading_ratio_lhs (k_r/k_X), loading_ratio_rhs (C*r/X),
        both_shrinking (bool: dr_dt<0 and dX_dt<0)."""
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    X = np.asarray(X, dtype=float)
    dt = np.diff(t)
    dr = np.diff(r)
    dX = np.diff(X)
    r_mid = 0.5 * (r[:-1] + r[1:])
    X_mid = 0.5 * (X[:-1] + X[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        dr_dt = dr / dt
        dX_dt = dX / dt
        L_r = -dr_dt / r_mid ** 2
        L_X = C_GB * dX_dt / X_mid ** 2
        L_total = L_r + L_X
        k_r = -dr_dt / r_mid
        k_X = -dX_dt / X_mid
        loading_ratio_lhs = k_r / k_X
        loading_ratio_rhs = C_GB * r_mid / X_mid
    predicted_sign = np.sign(L_total)
    both_shrinking = (dr_dt < 0) & (dX_dt < 0)
    return dict(t_mid=0.5 * (t[:-1] + t[1:]), dt=dt, r_mid=r_mid, X_mid=X_mid, dr_dt=dr_dt, dX_dt=dX_dt,
                L_r=gamma_s * L_r, L_X=gamma_s * L_X, L_total=gamma_s * L_total,
                predicted_sign_dsigma_dt=predicted_sign, k_r=k_r, k_X=k_X,
                loading_ratio_lhs=loading_ratio_lhs, loading_ratio_rhs=loading_ratio_rhs,
                both_shrinking=both_shrinking)


def prediction_agreement(t, r, X, sigma, gamma_s, C_GB):
    """Compares the analytic predicted_sign(d sigma/dt) (from r,X alone)
    against the MEASURED sign of d sigma/dt (from the sigma array
    directly). Returns the loading_terms_dt dict augmented with
    `measured_sign_dsigma_dt` and `agrees` (bool array)."""
    terms = loading_terms_dt(t, r, X, gamma_s, C_GB)
    sigma = np.asarray(sigma, dtype=float)
    dsigma = np.diff(sigma)
    dt = terms["dt"]
    with np.errstate(divide="ignore", invalid="ignore"):
        measured_dsigma_dt = dsigma / dt
    measured_sign = np.sign(measured_dsigma_dt)
    terms["measured_dsigma_dt"] = measured_dsigma_dt
    terms["measured_sign_dsigma_dt"] = measured_sign
    terms["agrees"] = (terms["predicted_sign_dsigma_dt"] == measured_sign) | (measured_sign == 0)
    return terms
