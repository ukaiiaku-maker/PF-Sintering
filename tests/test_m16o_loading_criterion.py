"""Milestone 16O Section 2-3: tests for the analytic Hussein loading
criterion decomposition."""
from __future__ import annotations

import math

import numpy as np

from pf_sintering.m16o_loading_criterion import loading_terms_dt, prediction_agreement

GAMMA_S = 1.0
GAMMA_GB = 2.0 * GAMMA_S * math.cos(math.radians(160.0 / 2.0))
C_GB = math.sqrt(1.0 - (GAMMA_GB / (2.0 * GAMMA_S)) ** 2)


def test_pure_curvature_loading_predicted_positive():
    """r shrinking, X constant -> sigma must be predicted to increase."""
    t = np.array([0.0, 1.0, 2.0])
    r = np.array([40e-9, 30e-9, 20e-9])
    X = np.array([200e-9, 200e-9, 200e-9])
    res = loading_terms_dt(t, r, X, GAMMA_S, C_GB)
    assert np.all(res["predicted_sign_dsigma_dt"] > 0)
    assert np.all(res["L_X"] == 0.0)


def test_pure_contact_growth_relaxation_predicted_negative():
    """r constant, X shrinking (C*dX/dt term negative) -> predicted
    decrease (matches the -C/X term of Eq 1b: shrinking X makes -C/X
    MORE negative, i.e. reduces sigma)."""
    t = np.array([0.0, 1.0, 2.0])
    r = np.array([30e-9, 30e-9, 30e-9])
    X = np.array([200e-9, 190e-9, 180e-9])
    res = loading_terms_dt(t, r, X, GAMMA_S, C_GB)
    assert np.all(res["predicted_sign_dsigma_dt"] < 0)
    assert np.all(res["L_r"] == 0.0)


def test_prediction_matches_measurement_on_analytic_trajectory():
    """Construct r(t), X(t) and the EXACT resulting sigma(t) from Eq 1b
    directly -- predicted and measured signs must agree at every
    interval (this is an algebraic identity, not an approximation, in
    the limit of small dt)."""
    from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma

    t = np.linspace(0.0, 1.0, 200)
    r = 40e-9 - 15e-9 * t  # shrinking
    X = 200e-9 + 5e-9 * t  # growing
    sigma = np.array([hussein_eq1b_sigma(ri, Xi, GAMMA_S, GAMMA_GB)[0] for ri, Xi in zip(r, X)])
    res = prediction_agreement(t, r, X, sigma, GAMMA_S, C_GB)
    assert np.all(res["agrees"])


def test_loading_ratio_criterion_consistent_with_L_total_when_both_shrinking():
    """When both r and X are shrinking, k_r/k_X > C*r/X must coincide
    exactly with L_total>0 (algebraic identity)."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    r = np.array([40e-9, 35e-9, 28e-9, 24e-9])
    X = np.array([200e-9, 190e-9, 175e-9, 165e-9])
    res = loading_terms_dt(t, r, X, GAMMA_S, C_GB)
    assert np.all(res["both_shrinking"])
    ratio_predicts_loading = res["loading_ratio_lhs"] > res["loading_ratio_rhs"]
    L_total_predicts_loading = res["L_total"] > 0
    assert np.array_equal(ratio_predicts_loading, L_total_predicts_loading)
