import math

import numpy as np

from pf_sintering.axisym_branch_boundary_flux import (
    AxisymmetricSurfaceBranch,
    apply_axisymmetric_branch_boundary_flux_step,
    axisymmetric_branch_flux_fv,
    extrapolate_outer_tj_branch_limit,
    extract_axisymmetric_surface_branches,
    linearly_implicit_mullins_branch_increment,
    symmetric_outer_tj_limit,
)
from pf_sintering.interface_attachment import planar_tanh_profile


def _straight_branch(side, rows, r0, mu):
    n = len(rows)
    ds = 1.0
    return AxisymmetricSurfaceBranch(
        side=side,
        row_indices=np.asarray(rows),
        s_centers_m=(np.arange(n) + 0.5) * ds,
        s_faces_m=np.arange(n + 1) * ds,
        r_centers_m=np.full(n, r0),
        r_faces_m=np.full(n + 1, r0),
        mu_Pa=np.asarray(mu, dtype=float))


def test_axisymmetric_fv_telescopes_and_uses_circumference_factor():
    branch = _straight_branch("positive", [3, 4, 5], 2.0, [3.0, 2.0, 1.0])
    result = axisymmetric_branch_flux_fv(
        branch, surface_flux_mobility_m6_per_J_model_time=0.25,
        incoming_volume_rate_m3_per_model_time=4.0 * math.pi)
    np.testing.assert_allclose(
        result["Q_face_m2_per_model_time"], [1.0, 0.25, 0.25, 0.0])
    np.testing.assert_allclose(
        result["circumference_volume_rate_faces_m3_per_model_time"],
        [4.0 * math.pi, math.pi, math.pi, 0.0])
    np.testing.assert_allclose(
        np.sum(result["cell_volume_rate_m3_per_model_time"]), 4.0 * math.pi)
    assert abs(result["integrated_closure_m3_per_model_time"]) < 1e-14


def test_extract_branches_samples_mu_on_half_contour_and_orients_outward():
    dz = dr = 1.0
    z = (np.arange(12) + 0.5) * dz - 6.0
    r_c = (np.arange(10) + 0.5) * dr
    radius = 4.0 + 0.1 * np.abs(z)
    f = planar_tanh_profile(r_c[None, :] - radius[:, None], W=1.2)
    mu = np.broadcast_to((10.0 + 2.0 * np.abs(z))[:, None], f.shape).copy()
    positive, negative = extract_axisymmetric_surface_branches(
        f, mu, r_c, z, z_tj=0.0, r_tj=4.0)
    assert np.all(np.diff(positive.s_centers_m) > 0.0)
    assert np.all(np.diff(negative.s_centers_m) > 0.0)
    assert np.all(np.diff(z[positive.row_indices]) > 0.0)
    assert np.all(np.diff(z[negative.row_indices]) < 0.0)
    np.testing.assert_allclose(positive.mu_Pa, 10.0 + 2.0 * z[positive.row_indices])
    np.testing.assert_allclose(negative.mu_Pa, 10.0 - 2.0 * z[negative.row_indices])
    assert positive.s_faces_m[0] == negative.s_faces_m[0] == 0.0


def test_boundary_flux_phase_update_closes_mass_and_separates_ownership():
    dz = dr = 1.0
    z = (np.arange(12) + 0.5) * dz - 6.0
    r_c = (np.arange(10) + 0.5) * dr
    f = np.broadcast_to(
        planar_tanh_profile(r_c[None, :] - 4.0, W=1.2), (12, 10)).copy()
    positive_mask = z[:, None] > 0.0
    grain1 = np.where(positive_mask, f, 0.0)
    grain2 = f - grain1
    positive = _straight_branch("positive", [6, 7, 8, 9, 10, 11], 4.0, np.zeros(6))
    negative = _straight_branch("negative", [5, 4, 3, 2, 1, 0], 4.0, np.zeros(6))
    rate = 1e-3
    dt = 0.2
    out = apply_axisymmetric_branch_boundary_flux_step(
        f, grain1, grain2, r_c, dr, dz, W=1.2,
        branches=(positive, negative),
        surface_flux_mobility_m6_per_J_model_time=0.0,
        incoming_volume_rate_m3_per_model_time=rate,
        dt_model=dt)
    f_new, g1_new, g2_new, diag = out
    np.testing.assert_allclose(g1_new + g2_new, f_new, atol=2e-14)
    np.testing.assert_allclose(diag["actual_added_volume_m3"], rate * dt, rtol=5e-9)
    assert diag["branch"]["positive"]["owner_grain"] == 1
    assert diag["branch"]["negative"]["owner_grain"] == 2
    assert np.max(np.abs((g2_new - grain2)[z > 0.0, :])) == 0.0
    assert np.max(np.abs((g1_new - grain1)[z < 0.0, :])) == 0.0
    assert diag["endpoint_row_count"] is None
    assert diag["endpoint_width_m"] is None
    assert not diag["ordinary_M_s_step_applied"]


def test_node_rates_allow_one_branch_to_feed_the_junction():
    dz = dr = 1.0
    z = (np.arange(12) + 0.5) - 6.0
    r_c = np.arange(10) + 0.5
    f = np.broadcast_to(
        planar_tanh_profile(r_c[None, :] - 4.0, W=1.2), (12, 10)).copy()
    grain1 = np.where(z[:, None] > 0.0, f, 0.0)
    grain2 = f - grain1
    branches = (
        _straight_branch("positive", [6, 7, 8, 9, 10, 11], 4.0, np.zeros(6)),
        _straight_branch("negative", [5, 4, 3, 2, 1, 0], 4.0, np.zeros(6)))
    total_rate = 1e-3
    rates = {"positive": -1e-3, "negative": 2e-3}
    f_new, grain1_new, grain2_new, diag = (
        apply_axisymmetric_branch_boundary_flux_step(
            f, grain1, grain2, r_c, dr, dz, W=1.2, branches=branches,
            surface_flux_mobility_m6_per_J_model_time=0.0,
            incoming_volume_rate_m3_per_model_time=total_rate,
            dt_model=0.1,
            branch_volume_rates_m3_per_model_time=rates))
    assert diag["branch_partition_mode"] == "zero-storage TJ node"
    assert diag["branch_fractions"] is None
    assert diag["branch_rate_over_net_GB"] == {
        "positive": -1.0, "negative": 2.0}
    assert diag["branch"]["positive"]["added_branch_volume_m3"] < 0.0
    assert diag["branch"]["negative"]["added_branch_volume_m3"] > 0.0
    np.testing.assert_allclose(
        diag["actual_added_volume_m3"], 0.1 * total_rate, rtol=1e-8)
    np.testing.assert_allclose(
        diag["grain_flux_volume_change_m3"][1],
        diag["branch"]["positive"]["added_branch_volume_m3"], rtol=1e-8)
    np.testing.assert_allclose(
        diag["grain_flux_volume_change_m3"][2],
        diag["branch"]["negative"]["added_branch_volume_m3"], rtol=1e-8)
    np.testing.assert_allclose(grain1_new + grain2_new, f_new, atol=2e-14)


def test_sink_off_node_exchange_closes_each_grain_and_zero_net_mass():
    z = (np.arange(12) + 0.5) - 6.0
    r_c = np.arange(10) + 0.5
    f = np.broadcast_to(
        planar_tanh_profile(r_c[None, :] - 4.0, W=1.2), (12, 10)).copy()
    grain1 = np.where(z[:, None] > 0.0, f, 0.0)
    grain2 = f - grain1
    branches = (
        _straight_branch("positive", [6, 7, 8, 9, 10, 11], 4.0, np.zeros(6)),
        _straight_branch("negative", [5, 4, 3, 2, 1, 0], 4.0, np.zeros(6)))
    _, _, _, diag = apply_axisymmetric_branch_boundary_flux_step(
        f, grain1, grain2, r_c, 1.0, 1.0, W=1.2, branches=branches,
        surface_flux_mobility_m6_per_J_model_time=0.0,
        incoming_volume_rate_m3_per_model_time=0.0, dt_model=0.1,
        branch_volume_rates_m3_per_model_time={
            "positive": -1e-3, "negative": 1e-3})
    assert diag["branch_rate_over_net_GB"] == {
        "positive": None, "negative": None}
    np.testing.assert_allclose(diag["actual_added_volume_m3"], 0.0, atol=1e-12)
    assert diag["grain_flux_volume_change_m3"][1] < 0.0
    assert diag["grain_flux_volume_change_m3"][2] > 0.0
    assert max(abs(value) for value in diag["grain_flux_closure_relative"].values()) < 1e-8


def test_normal_reconstruction_uses_physical_tanh_resolution_floor():
    z = (np.arange(12) + 0.5) - 6.0
    r_c = np.arange(10) + 0.5
    W = 1.2
    f = np.broadcast_to(
        planar_tanh_profile(r_c[None, :] - 4.0, W=W), (12, 10)).copy()
    grain1 = np.where(z[:, None] > 0.0, f, 0.0)
    grain2 = f - grain1
    branches = (
        _straight_branch("positive", [6, 7, 8, 9, 10, 11], 4.0, np.zeros(6)),
        _straight_branch("negative", [5, 4, 3, 2, 1, 0], 4.0, np.zeros(6)))
    _, _, _, diag = apply_axisymmetric_branch_boundary_flux_step(
        f, grain1, grain2, r_c, 1.0, 1.0, W=W, branches=branches,
        surface_flux_mobility_m6_per_J_model_time=0.0,
        incoming_volume_rate_m3_per_model_time=1e-3, dt_model=0.1,
        normal_displacement_diffusion_B_m4_per_model_time=1e-8)
    expected_floor = 2.0 * math.atanh(0.9) * W
    for side in ("negative", "positive"):
        branch = diag["branch"][side]
        np.testing.assert_allclose(
            branch["normal_displacement_reconstruction_length_m"],
            expected_floor)
        assert branch["redistribution_length_control"] == (
            "diffuse_interface_5_95_resolution")
        assert branch["packet_surface_diffusion_length_m"] < expected_floor


def test_outer_branch_limit_excludes_core_and_does_not_use_endpoint_cells():
    s = np.arange(0.5, 10.0, 1.0)
    mu = 7.0 + 2.0 * s + 0.25 * s ** 2
    positive = AxisymmetricSurfaceBranch(
        side="positive", row_indices=np.arange(len(s)),
        s_centers_m=s, s_faces_m=np.arange(11.0),
        r_centers_m=np.full(len(s), 4.0), r_faces_m=np.full(11, 4.0),
        mu_Pa=mu)
    negative = AxisymmetricSurfaceBranch(
        side="negative", row_indices=np.arange(len(s)),
        s_centers_m=s, s_faces_m=np.arange(11.0),
        r_centers_m=np.full(len(s), 4.0), r_faces_m=np.full(11, 4.0),
        mu_Pa=mu)
    limit = extrapolate_outer_tj_branch_limit(
        positive, core_exclusion_m=2.0, outer_fit_limit_m=8.0,
        polynomial_degree=2)
    np.testing.assert_allclose(limit["mu_TJ_outer_limit_Pa"], 7.0)
    assert limit["endpoint_row_count"] is None
    assert limit["diffuse_TJ_core_sampled"] is False
    symmetric = symmetric_outer_tj_limit(
        (positive, negative), 2.0, 8.0, polynomial_degree=2)
    assert symmetric["branch_relative_disagreement"] == 0.0
    assert symmetric["force_balance_constraint_applied"] is False


def test_implicit_mullins_increment_closes_mass_and_has_explicit_small_dt_limit():
    branch = _straight_branch(
        "positive", np.arange(8), 4.0,
        [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0])
    flux = axisymmetric_branch_flux_fv(
        branch, surface_flux_mobility_m6_per_J_model_time=0.2,
        incoming_volume_rate_m3_per_model_time=0.5)
    small_dt = 1e-10
    small = linearly_implicit_mullins_branch_increment(
        branch, flux, small_dt, B_m4_per_model_time=0.3)
    np.testing.assert_allclose(
        small["cell_volume_changes_m3"],
        small_dt * flux["cell_volume_rate_m3_per_model_time"], rtol=2e-8)
    large = linearly_implicit_mullins_branch_increment(
        branch, flux, dt_model=2.0, B_m4_per_model_time=0.3)
    np.testing.assert_allclose(
        np.sum(large["cell_volume_changes_m3"]),
        2.0 * np.sum(flux["cell_volume_rate_m3_per_model_time"]),
        rtol=2e-14, atol=2e-14)
    explicit = 2.0 * flux["normal_velocity_m_per_model_time"]
    assert np.linalg.norm(np.diff(large["normal_displacements_m"], 2)) < np.linalg.norm(
        np.diff(explicit, 2))
    assert large["endpoint_regularization_used"] is False
