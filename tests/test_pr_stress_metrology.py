import math
from pathlib import Path

import numpy as np

from pf_sintering.pr_stress_metrology import (
    analytic_m16g_reference,
    branch_patch_3d_stress,
    circular_contact_terms,
    convex_mean_width,
    integral_turning,
    local_3d_contact_stress,
    mirror_axisymmetric_particle_branch,
    normal_support,
)


def test_exact_analytic_m16g_reference():
    r = analytic_m16g_reference()
    assert math.isclose(r["lambda_over_Rcyl"], 1.14 * math.pi, rel_tol=1e-15)
    assert math.isclose(r["z1_over_lambda"], 0.5804306232551663, rel_tol=1e-15)
    assert math.isclose(r["r_neck"] * 1e9, 46.6515138991, rel_tol=2e-12)
    assert math.isclose(r["X_neck"] * 1e9, 93.3030277982, rel_tol=2e-12)
    assert abs(r["Rprime_at_z1"]) < 1e-14
    assert math.isclose(r["Rpp_at_z1_per_m"], 1.1542012927e7, rel_tol=2e-10)
    assert math.isclose(r["kappa_meridional_per_m"], -1.1542012927e7, rel_tol=2e-10)
    assert math.isclose(r["kappa2_neck"], 2.1109877702e7, rel_tol=2e-10)
    assert math.isclose(r["s_local_cap"], 3.2651890629e7, rel_tol=2e-10)
    assert math.isclose(r["s_line_3D"], 4.2219755404e7, rel_tol=2e-10)
    assert math.isclose(r["s_3D_local"], 7.4871646032e7, rel_tol=2e-10)
    assert math.isclose(r["sigma_3D_local_MPa"], 74.8716460, rel_tol=1e-9)


def test_line_force_and_dimensions_and_concave_sign():
    radius = 50e-9
    psi = math.radians(160.0)
    terms = circular_contact_terms(radius, psi)
    assert math.isclose(terms["s_line_3D"], 2 * math.sin(psi / 2) / radius)
    result = local_3d_contact_stress(-1e7, -1e7, radius, psi, gamma_s=1.7)
    assert result["s_local_cap"] > 0
    assert math.isclose(result["sigma_3D_local_Pa"], 1.7 * result["s_3D_local"])


def test_circle_global_estimators():
    radius = 3.25
    theta = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
    points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    turn = integral_turning(points, closed=True)
    assert math.isclose(abs(turn["turning_angle"]), 2 * math.pi, rel_tol=5e-4)
    assert math.isclose(abs(turn["k_integral"]), 1 / radius, rel_tol=5e-4)
    mw = convex_mean_width(points)
    assert math.isclose(mw["mean_width_2D"], 2 * radius, rel_tol=5e-6)
    assert math.isclose(mw["k_mean_width"], 1 / radius, rel_tol=5e-6)
    support = normal_support(points, (0, 1))
    assert math.isclose(support["normal_support_width"], 2 * radius, rel_tol=5e-6)
    assert math.isclose(support["k_normal_support"], 1 / radius, rel_tol=5e-6)


def test_axisymmetric_mirror_reconstruction():
    r = np.array([2.0, 2.5, 1.0])
    z = np.array([0.0, 1.0, 2.0])
    contour = mirror_axisymmetric_particle_branch(r, z, z_contact=0.0)
    assert contour.shape == (6, 2)
    np.testing.assert_allclose(contour[:3, 0], [2.0, 2.5, 1.0])
    np.testing.assert_allclose(contour[3:, 0], [-1.0, -2.5, -2.0])
    assert contour[0, 1] == contour[-1, 1] == 0.0


def test_uniform_similarity_scaling():
    base = analytic_m16g_reference(R_cyl=100e-9)
    scale = 0.8
    scaled = analytic_m16g_reference(R_cyl=scale * 100e-9)
    assert math.isclose(scaled["s_3D_local"], base["s_3D_local"] / scale, rel_tol=1e-14)


def test_branch_patch_turning_has_symmetric_signed_curvature():
    z = np.linspace(-40e-9, 40e-9, 161)
    radius = 50e-9
    curvature_scale = 8e6
    R = radius + 0.5 * curvature_scale * z * z
    out = branch_patch_3d_stress(
        R, z, 0.0, patch_length_m=20e-9, tangent_span_m=5e-9)
    assert out["kappa_patch_side1"] < 0.0
    assert out["kappa_patch_side2"] < 0.0
    np.testing.assert_allclose(
        out["kappa_patch_side1"], out["kappa_patch_side2"], rtol=2e-2)


def test_pr_fast_solver_and_energy_use_same_noflux_bc():
    source = Path("scripts/pr_stress_metrology_sinkoff.py").read_text()
    assert 'axisym_free_energy_gb(f, particle, neighbor, p, Wc, dr, dz, rc, rf, bc_z="noflux")' in source
    assert 'axisym_free_energy_gb_components(' in source
    assert 'bc_z="noflux")' in source
    assert "Meta, p.W, scratch" in source
