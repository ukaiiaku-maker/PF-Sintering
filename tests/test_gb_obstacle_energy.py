import math

import numpy as np

from pf_sintering.gb_obstacle_energy import (
    TANH_PROFILE_EXCESS_FACTOR,
    gb_obstacle_coefficients,
    m_eta_from_m_gb,
    m_gb_from_m_eta,
    obstacle_compact_width,
    obstacle_ell,
    obstacle_gamma_gb,
    obstacle_grad_sq_integral,
    obstacle_profile,
)


def test_obstacle_profile_satisfies_euler_lagrange_equation_inside_active_interval():
    # -2*k_eta*phi'' + Wc*(1-2*phi) = 0 inside the active interval.
    k_eta, Wc = 6.0e-8, 1.8e9
    ell = obstacle_ell(k_eta, Wc)
    x = np.linspace(-0.4 * math.pi * ell, 0.4 * math.pi * ell, 4001)
    dx = x[1] - x[0]
    phi = obstacle_profile(x, ell)
    phi_pp = np.gradient(np.gradient(phi, dx), dx)
    residual = -2 * k_eta * phi_pp + Wc * (1 - 2 * phi)
    # exclude the few points nearest each end where the second finite
    # difference straddles the free boundary
    interior = slice(20, -20)
    assert np.max(np.abs(residual[interior])) < 1e-3 * Wc


def test_obstacle_profile_reaches_0_and_1_with_zero_derivative_at_free_boundary():
    k_eta, Wc = 6.0e-8, 1.8e9
    ell = obstacle_ell(k_eta, Wc)
    width = obstacle_compact_width(k_eta, Wc)
    assert math.isclose(width, math.pi * ell, rel_tol=1e-12)
    x_edge = width / 2
    assert math.isclose(float(obstacle_profile(np.array([x_edge]), ell)[0]), 1.0, abs_tol=1e-9)
    assert math.isclose(float(obstacle_profile(np.array([-x_edge]), ell)[0]), 0.0, abs_tol=1e-9)
    assert float(obstacle_profile(np.array([x_edge + ell]), ell)[0]) == 1.0
    assert float(obstacle_profile(np.array([-x_edge - ell]), ell)[0]) == 0.0


def test_grad_sq_integral_and_gamma_gb_match_direct_numerical_integration():
    k_eta, Wc = 6.0e-8, 1.8e9
    ell = obstacle_ell(k_eta, Wc)
    width = obstacle_compact_width(k_eta, Wc)
    x = np.linspace(-0.5001 * width, 0.5001 * width, 200001)
    phi = obstacle_profile(x, ell)
    dphi_dx = np.gradient(phi, x)

    grad_sq_numeric = np.trapezoid(dphi_dx ** 2, x)
    assert math.isclose(grad_sq_numeric, obstacle_grad_sq_integral(ell), rel_tol=1e-4)

    density = k_eta * dphi_dx ** 2 + Wc * phi * (1 - phi)
    gamma_numeric = np.trapezoid(density, x)
    assert math.isclose(gamma_numeric, obstacle_gamma_gb(k_eta, Wc), rel_tol=1e-4)


def test_gb_obstacle_coefficients_round_trip():
    for gamma_gb_target in (0.5, 1.0, 1.4, 1.6):
        for W_GB in (10e-9, 20e-9, 40e-9):
            c = gb_obstacle_coefficients(gamma_gb_target, W_GB)
            assert math.isclose(obstacle_compact_width(c["k_eta"], c["Wc"]), W_GB, rel_tol=1e-12)
            assert math.isclose(obstacle_gamma_gb(c["k_eta"], c["Wc"]), gamma_gb_target, rel_tol=1e-12)
            assert c["W_cpl_f"] == c["Wc"]


def test_old_tanh_derived_coefficients_reinterpreted_as_obstacle_profile():
    # Milestone 14G Section 3: the OLD k_eta=3*gamma*W, Wc=36*gamma/W
    # formula, reinterpreted through the CORRECT obstacle-equilibrium
    # relations (not the tanh-profile energy), gives
    # delta_GB~=0.9069*W, gamma_GB_PF~=8.1621*gamma -- neither of which
    # is what was previously assumed (a tanh profile with the DECLARED
    # gamma directly, nor even the tanh's own 19x excess energy).
    gamma_declared = 1.0
    W = 20e-9
    k_eta = 3 * gamma_declared * W
    Wc = 36 * gamma_declared / W
    width = obstacle_compact_width(k_eta, Wc)
    gamma_pf = obstacle_gamma_gb(k_eta, Wc)
    assert math.isclose(width / W, math.pi / math.sqrt(12), rel_tol=1e-9)
    assert math.isclose(gamma_pf / gamma_declared, 3 * math.pi * math.sqrt(3) / 2, rel_tol=1e-9)
    assert math.isclose(width / W, 0.9069, rel_tol=2e-4)
    assert math.isclose(gamma_pf / gamma_declared, 8.1621, rel_tol=2e-4)


def test_tanh_profile_excess_factor_still_19_but_is_not_physical_calibration():
    # The tanh-profile excess energy (Milestone 14E/14F's finding) remains
    # numerically correct as a fact about the TANH profile specifically,
    # evaluated with the OLD coefficients -- it is simply not the
    # equilibrium profile, so it is not the physical GB energy.
    W = 20e-9
    gamma_declared = 1.0
    k_eta = 3 * gamma_declared * W
    Wc = 36 * gamma_declared / W
    x = np.linspace(-500e-9, 500e-9, 200001)
    eta2 = 0.5 * (1 + np.tanh(x / W))
    eta1 = 1 - eta2
    bulk = Wc * (eta1 ** 2 + eta2 ** 2) * (0.5 - 1.0)
    ge1 = np.gradient(eta1, x)
    ge2 = np.gradient(eta2, x)
    grad_e = 0.5 * k_eta * (ge1 ** 2 + ge2 ** 2)
    background = -0.5 * Wc
    gamma_tanh = np.trapezoid((bulk + grad_e) - background, x)
    assert math.isclose(gamma_tanh, TANH_PROFILE_EXCESS_FACTOR * gamma_declared, rel_tol=1e-6)
    # and it does NOT match the true obstacle-equilibrium energy for the
    # SAME coefficients (confirming tanh is not the equilibrium profile).
    gamma_equilibrium = obstacle_gamma_gb(k_eta, Wc)
    assert not math.isclose(gamma_tanh, gamma_equilibrium, rel_tol=0.1)


def test_mobility_mapping_round_trip():
    for M_gb in (1e-16, 1e-15, 5e-15):
        for W_GB in (10e-9, 20e-9, 40e-9):
            M_eta = m_eta_from_m_gb(M_gb, W_GB)
            assert math.isclose(m_gb_from_m_eta(M_eta, W_GB), M_gb, rel_tol=1e-12)


def test_mobility_mapping_matches_explicit_formula():
    M_eta, W_GB = 4.2667e-9, 20e-9
    M_gb = m_gb_from_m_eta(M_eta, W_GB)
    assert math.isclose(M_gb, 4 * M_eta * W_GB / math.pi ** 2, rel_tol=1e-12)
