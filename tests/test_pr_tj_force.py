from __future__ import annotations

import math

import numpy as np

from pf_sintering.pr_tj_force import equilibrium_reference, instantaneous_tj_resultant


GAMMA_S = 1.0
PSI_EQ_DEG = 160.0
GAMMA_GB = 2.0 * GAMMA_S * math.cos(math.radians(PSI_EQ_DEG) / 2.0)


def _wedge(psi_deg: float, dz: float = 1e-9):
    z = np.arange(-40, 41, dtype=float) * dz
    slope = 1.0 / math.tan(math.radians(psi_deg) / 2.0)
    R = 50e-9 + slope * np.abs(z)
    return R, z


def test_exact_equilibrium_is_zero_reference_only():
    ref = equilibrium_reference(PSI_EQ_DEG, GAMMA_S, GAMMA_GB)
    assert abs(ref["Phi_TJ_J_m2"]) < 1e-14
    assert ref["F_TJ_magnitude_J_m2"] < 1e-14
    assert ref["equilibrium_enforced_in_dynamics"] is False


def test_instantaneous_wedge_recovers_equilibrium_without_enforcing_it():
    R, z = _wedge(PSI_EQ_DEG)
    result = instantaneous_tj_resultant(
        R, z, 0.0, GAMMA_S, GAMMA_GB, fit_length_m=12e-9)
    assert abs(result.measured_psi_deg - PSI_EQ_DEG) < 1e-10
    assert abs(result.Phi_TJ_J_m2) < 1e-12
    assert result.equilibrium_enforced is False
    assert result.whole_ring_factor_applied is False


def test_nonzero_dynamic_resultant_is_preserved_with_expected_sign():
    values = []
    for psi_deg in (150.0, 160.0, 170.0):
        R, z = _wedge(psi_deg)
        values.append(instantaneous_tj_resultant(
            R, z, 0.0, GAMMA_S, GAMMA_GB, fit_length_m=12e-9).Phi_TJ_J_m2)
    # Inward climb projection: gamma_gb - 2 gamma_s cos(psi/2).
    assert values[0] < 0.0
    assert abs(values[1]) < 1e-12
    assert values[2] > 0.0
    assert values[0] < values[1] < values[2]


def test_result_is_local_per_unit_circumference_not_whole_ring_force():
    R, z = _wedge(170.0)
    first = instantaneous_tj_resultant(
        R, z, 0.0, GAMMA_S, GAMMA_GB, fit_length_m=8e-9)
    translated = instantaneous_tj_resultant(
        R + 3e-6, z, 0.0, GAMMA_S, GAMMA_GB, fit_length_m=8e-9)
    assert math.isclose(first.Phi_TJ_J_m2, translated.Phi_TJ_J_m2, abs_tol=1e-13)
    assert first.whole_ring_factor_applied is False


def test_tangent_fit_uses_fixed_physical_length_across_grids():
    coarse_R, coarse_z = _wedge(155.0, dz=1.25e-9)
    fine_R, fine_z = _wedge(155.0, dz=1.0e-9)
    coarse = instantaneous_tj_resultant(
        coarse_R, coarse_z, 0.0, GAMMA_S, GAMMA_GB, fit_length_m=10e-9)
    fine = instantaneous_tj_resultant(
        fine_R, fine_z, 0.0, GAMMA_S, GAMMA_GB, fit_length_m=10e-9)
    assert math.isclose(coarse.Phi_TJ_J_m2, fine.Phi_TJ_J_m2, abs_tol=1e-12)
