import math

import numpy as np

from pf_sintering.axisym_branch_boundary_flux import AxisymmetricSurfaceBranch
from pf_sintering.model_time_transport import ModelTimeGBTransport
from pf_sintering.tj_transport_node import (
    solve_model_time_tj_node,
    solve_prescribed_rate_tj_node,
)


def _branch(side, radius, mu1):
    return AxisymmetricSurfaceBranch(
        side=side, row_indices=np.arange(4),
        s_centers_m=np.arange(4) + 0.5,
        s_faces_m=np.arange(5.0),
        r_centers_m=np.full(4, radius),
        r_faces_m=np.full(5, radius),
        mu_Pa=np.array([mu1, mu1 - 0.1, mu1 - 0.2, mu1 - 0.3]))


def _transport(tau_ex=0.0):
    # K_GB=1 Pa*model-time and L_GB=A*b/K_GB=4 for A=4, b=1.
    return ModelTimeGBTransport(
        x_d_m=1.0, kB_J_per_K=1.0, temperature_K=1.0,
        atomic_volume_m3=1.0, b_m=1.0,
        D_gb_m2_per_model_time=1.0, tau_ex_model=tau_ex)


def test_manufactured_asymmetric_prescribed_rate_recovers_exact_node():
    branches = (_branch("positive", 2.0, 3.0), _branch("negative", 1.0, 1.0))
    mobility = 1.0 / (4.0 * math.pi)  # C_plus=2, C_minus=1.
    node = solve_prescribed_rate_tj_node(branches, mobility, 5.0)
    np.testing.assert_allclose(node["mu_TJ_Pa"], 4.0)
    np.testing.assert_allclose(
        node["branch_volume_rates_m3_per_model_time"]["positive"], 2.0)
    np.testing.assert_allclose(
        node["branch_volume_rates_m3_per_model_time"]["negative"], 3.0)
    assert node["branch_rate_over_net_GB"] == {
        "positive": 0.4, "negative": 0.6}
    assert abs(node["zero_storage_closure_relative"]) < 1e-15
    assert node["imposed_half_partition"] is False


def test_prescribed_tiny_rate_closes_after_node_potential_roundoff():
    branches = (
        _branch("positive", 5.0e-8, 4.0e7),
        _branch("negative", 4.8e-8, 4.0e7 - 2.0e3))
    incoming = 2.6178241302607806e-26
    node = solve_prescribed_rate_tj_node(branches, 2.5e-30, incoming)
    rates = node["branch_volume_rates_m3_per_model_time"]
    assert sum(rates.values()) == incoming
    assert node["zero_storage_closure_m3_per_model_time"] == 0.0
    assert node["prescribed_rate_roundoff_closure_corrected"] is True


def test_manufactured_linear_gb_coupling_recovers_analytical_node():
    branches = (_branch("positive", 2.0, 3.0), _branch("negative", 1.0, 1.0))
    mobility = 1.0 / (4.0 * math.pi)
    node = solve_model_time_tj_node(
        branches, mobility, mu_GB_Pa=10.0, contact_area_m2=4.0,
        transport=_transport())
    expected_mu = 47.0 / 7.0
    np.testing.assert_allclose(node["mu_TJ_Pa"], expected_mu)
    np.testing.assert_allclose(node["Vdot_GB_m3_per_model_time"],
                               4.0 * (10.0 - expected_mu))
    np.testing.assert_allclose(
        sum(node["branch_volume_rates_m3_per_model_time"].values()),
        node["Vdot_GB_m3_per_model_time"])
    assert node["branch_rate_over_net_GB"]["positive"] != 0.5
    assert abs(node["zero_storage_closure_relative"]) < 1e-15


def test_sink_off_node_allows_equal_opposite_asymmetric_branch_exchange():
    branches = (_branch("positive", 2.0, 3.0), _branch("negative", 1.0, 1.0))
    node = solve_model_time_tj_node(
        branches, 1.0 / (4.0 * math.pi), mu_GB_Pa=10.0,
        contact_area_m2=4.0, transport=_transport(), gb_path_active=False)
    np.testing.assert_allclose(node["mu_TJ_Pa"], 7.0 / 3.0)
    rates = node["branch_volume_rates_m3_per_model_time"]
    np.testing.assert_allclose(
        rates["positive"] + rates["negative"], 0.0, atol=1e-15)
    assert rates["positive"] < 0.0 < rates["negative"]
    assert node["Vdot_GB_m3_per_model_time"] == 0.0


def test_nonlinear_tau_ex_node_closes_same_scalar_conservation_law():
    branches = (_branch("positive", 2.0, 3.0), _branch("negative", 1.0, 1.0))
    node = solve_model_time_tj_node(
        branches, 1.0 / (4.0 * math.pi), mu_GB_Pa=10.0,
        contact_area_m2=4.0, transport=_transport(tau_ex=0.2))
    assert 7.0 / 3.0 < node["mu_TJ_Pa"] < 10.0
    np.testing.assert_allclose(
        sum(node["branch_volume_rates_m3_per_model_time"].values()),
        node["Vdot_GB_m3_per_model_time"], rtol=2e-14)
    assert node["nonlinear_tau_ex_solve"] is True
