"""Production high-barrier run, Sections 4 and 6: regression tests for
the EXP nucleation barrier -- the activation-volume identity at 75 MPa
and the hard DeltaG/Gamma reference values at 1200 K."""
import math

import numpy as np

from pf_sintering.exp_barrier_nucleation import (
    CreepExpFloorHazardParams,
    CreepExpFloorParams,
    CreepFloorFirstPassageClock,
    CompleteExpFloorParams,
    ExpBarrierParams,
    activation_volume_creep_floor_m3,
    activation_volume_m3,
    delta_G_creep_floor_eV,
    delta_G_complete_exp_floor_eV,
    delta_G_eV,
    creep_equivalent_stress_pa,
    creep_floor_first_passage_step,
    creep_floor_site_count,
    gamma_birth_creep_floor_per_s,
    gamma_complete_exp_floor_per_model_time,
    gamma_pref,
    gamma_birth,
    reset_creep_floor_clock,
    tj_site_count,
)

P = ExpBarrierParams()  # defaults match the prescribed G0/a_exp/sigma_c/n/nu0/GS/b/T


def test_activation_volume_at_75MPa_matches_12p5_b_cubed():
    b = 2.5e-10
    target = 12.5 * b ** 3
    v_star = activation_volume_m3(75e6, P)
    rel_err = abs(v_star - target) / target
    assert rel_err < 1e-2, f"v*(75MPa)={v_star:.6e} vs target={target:.6e} rel_err={rel_err:.4e}"


def test_gamma_pref_matches_prescribed_1903():
    assert math.isclose(gamma_pref(P), 1903.0, rel_tol=2e-3)


def test_deltaG_reference_values_at_1200K():
    refs = {0.0: 0.64233, 25e6: 0.60726, 50e6: 0.57410, 75e6: 0.54275, 100e6: 0.51311}
    for sigma_pa, expected in refs.items():
        got = delta_G_eV(sigma_pa, P)
        assert math.isclose(got, expected, rel_tol=1e-3), f"sigma={sigma_pa/1e6}MPa got={got} expected={expected}"


def test_gamma_reference_values_at_1200K():
    refs = {0.0: 3.82, 25e6: 5.36, 50e6: 7.38, 75e6: 10.00, 100e6: 13.32}
    for sigma_pa, expected in refs.items():
        got = gamma_birth(sigma_pa, P)
        assert math.isclose(got, expected, rel_tol=1e-2), f"sigma={sigma_pa/1e6}MPa got={got} expected={expected}"


def test_no_negative_stress_enhancement():
    """DeltaG(sigma<=0) must equal DeltaG(0) exactly, not exceed it."""
    dG0 = delta_G_eV(0.0, P)
    for sigma_pa in (-1e6, -50e6, -1e9):
        assert delta_G_eV(sigma_pa, P) == dG0


def test_deltaG_equivalent_implementation_coefficient():
    """DeltaG_eV = 0.642329 * exp(-2.24605765e-9 * sigma_Pa) -- the
    'equivalent implementation' form given directly in the instructions,
    checked against the closed-form ExpBarrierParams evaluation."""
    for sigma_pa in (0.0, 25e6, 50e6, 75e6, 100e6):
        direct = 0.642329 * math.exp(-2.24605765e-9 * sigma_pa)
        via_params = delta_G_eV(sigma_pa, P)
        assert math.isclose(direct, via_params, rel_tol=1e-6)


def test_authoritative_creep_floor_has_exact_limits_and_required_fit_values():
    p = CreepExpFloorParams(
        G0_eV=2.0, sigma_star_pa=100e6, floor_fraction=0.2, n=3.0)
    assert delta_G_creep_floor_eV(0.0, p) == 2.0
    assert delta_G_creep_floor_eV(-1.0, p) == 2.0
    assert delta_G_creep_floor_eV(1e15, p) == 0.4
    assert activation_volume_creep_floor_m3(-1.0, p) == 0.0


def test_authoritative_creep_floor_activation_volume_is_same_law_derivative():
    p = CreepExpFloorParams(
        G0_eV=1.7, sigma_star_pa=260e6, floor_fraction=0.13, n=2.4)
    sigma = 91e6
    ds = 10.0
    numerical = -(
        delta_G_creep_floor_eV(sigma + ds, p)
        - delta_G_creep_floor_eV(sigma - ds, p)) * 1.602176634e-19 / (2.0 * ds)
    analytic = activation_volume_creep_floor_m3(sigma, p)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-8)


def test_creep_equivalent_stress_uses_source_volume_jacobian():
    assert creep_equivalent_stress_pa(1.4e-6, 7.0e-15) == 200e6
    with np.testing.assert_raises(ValueError):
        creep_equivalent_stress_pa(1.0, 0.0)


def test_creep_floor_rate_and_first_passage_clock_use_required_physical_inputs():
    barrier = CreepExpFloorParams(
        G0_eV=1.2, sigma_star_pa=300e6, floor_fraction=0.15, n=3.0)
    p = CreepExpFloorHazardParams(
        barrier=barrier, attempt_frequency_per_s=2e9,
        b_m=0.25e-9, temperature_K=1200.0)
    r_tj = 46e-9
    sigma = 190e6
    expected_sites = 2.0 * math.pi * r_tj / p.b_m
    assert math.isclose(creep_floor_site_count(r_tj, p), expected_sites)
    expected_rate = (
        expected_sites * p.attempt_frequency_per_s
        * math.exp(-delta_G_creep_floor_eV(sigma, barrier)
                   / (1.380649e-23 * p.temperature_K / 1.602176634e-19)))
    assert math.isclose(
        gamma_birth_creep_floor_per_s(sigma, r_tj, p), expected_rate)

    class FixedThreshold:
        def exponential(self):
            return 0.75

    rng = FixedThreshold()
    clock = CreepFloorFirstPassageClock()
    rate = gamma_birth_creep_floor_per_s(sigma, r_tj, p)
    assert not creep_floor_first_passage_step(
        clock, sigma, r_tj, 0.5 / rate, p, rng)
    assert creep_floor_first_passage_step(
        clock, sigma, r_tj, 0.3 / rate, p, rng)
    frozen_hazard = clock.hazard
    assert not creep_floor_first_passage_step(
        clock, sigma, r_tj, 1.0 / rate, p, rng)
    assert clock.hazard == frozen_hazard
    reset_creep_floor_clock(clock, rng)
    assert not clock.triggered and clock.hazard == 0.0 and clock.threshold == 0.75


def test_complete_exp_floor_keeps_floor_a_and_fixed_tj_site_formula():
    barrier = CompleteExpFloorParams(
        G0_eV=1.2, G_floor_eV=0.2, a=1.7,
        sigma_hat_pa=200e6, n=3.0)
    assert delta_G_complete_exp_floor_eV(0.0, barrier) == 1.2
    assert delta_G_complete_exp_floor_eV(-10e6, barrier) == 1.2
    assert delta_G_complete_exp_floor_eV(1e15, barrier) == 0.2
    sigma = 195e6
    radius = 46e-9
    b_m = 0.25e-9
    scale = 0.31
    sites = 2.0 * math.pi * radius / b_m
    assert math.isclose(tj_site_count(radius, b_m), sites)
    expected = (
        sites * scale
        * math.exp(-delta_G_complete_exp_floor_eV(sigma, barrier)
                   / (1.380649e-23 * 1200.0 / 1.602176634e-19)))
    assert math.isclose(gamma_complete_exp_floor_per_model_time(
        sigma, radius, barrier=barrier,
        clock_scale_per_model_time=scale, b_m=b_m,
        temperature_K=1200.0), expected)


if __name__ == "__main__":
    test_activation_volume_at_75MPa_matches_12p5_b_cubed()
    test_gamma_pref_matches_prescribed_1903()
    test_deltaG_reference_values_at_1200K()
    test_gamma_reference_values_at_1200K()
    test_no_negative_stress_enhancement()
    test_deltaG_equivalent_implementation_coefficient()
    print("ALL PASSED")
