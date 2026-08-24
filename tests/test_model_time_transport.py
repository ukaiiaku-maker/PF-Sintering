import math

import numpy as np
import pytest

from pf_sintering.model_time_transport import (
    KINETIC_TIME_BASIS,
    ModelTimeGBTransport,
    predeclared_timescale_matrix,
    surface_mobility_for_target_time,
    surface_relaxation_time_model,
)


def _transport(reference_tau=0.03, tau_ex=0.003):
    return ModelTimeGBTransport.from_reference_target(
        x_d_m=25e-9, kB_J_per_K=1.380649e-23, temperature_K=1200.0,
        atomic_volume_m3=1e-29, b_m=0.25e-9,
        reference_affinity_Pa=8e7, reference_tau_gb_model=reference_tau,
        tau_ex_model=tau_ex)


def test_reference_target_is_recovered_without_seconds_conversion():
    transport = _transport()
    np.testing.assert_allclose(transport.tau_gb_model(8e7), 0.03, rtol=2e-15)
    np.testing.assert_allclose(transport.qdot_m_per_model_time(8e7), 0.25e-9 / 0.03)
    diag = transport.diagnostics(8e7)
    assert diag["kinetic_time_basis"] == KINETIC_TIME_BASIS
    assert diag["physical_seconds_conversion"] is None
    assert diag["material_calibration"] is None


def test_instantaneous_affinity_controls_rate_and_nonpositive_drive_stops():
    transport = _transport(tau_ex=0.0)
    np.testing.assert_allclose(
        transport.tau_gb_model(4e7), 2.0 * transport.tau_gb_model(8e7))
    assert transport.qdot_m_per_model_time(0.0) == 0.0
    assert transport.qdot_m_per_model_time(-1.0) == 0.0
    assert math.isinf(transport.tau_gb_model(-1.0))


def test_surface_target_and_predeclared_matrix_close_exactly():
    ell = 15e-9
    gamma = 1.0
    target = 0.005
    mobility = surface_mobility_for_target_time(ell, gamma, target)
    np.testing.assert_allclose(
        surface_relaxation_time_model(ell, mobility, gamma), target,
        rtol=2e-15)
    matrix = predeclared_timescale_matrix()
    assert len(matrix) == 9
    assert {(row["tau_gb_over_tau_load"], row["tau_surface_over_tau_gb"])
            for row in matrix} == {
                (a, b) for a in (0.01, 0.03, 0.10) for b in (0.1, 0.3, 1.0)}
    baseline = next(
        row for row in matrix
        if row["tau_gb_over_tau_load"] == 0.03
        and row["tau_surface_over_tau_gb"] == 0.1)
    assert baseline["tau_gb_model"] == 0.03
    assert baseline["tau_surface_model"] == pytest.approx(0.003)


def test_invalid_reference_excess_time_is_rejected():
    with pytest.raises(ValueError, match="must exceed"):
        _transport(reference_tau=0.03, tau_ex=0.03)
