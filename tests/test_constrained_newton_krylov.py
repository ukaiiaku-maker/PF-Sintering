import numpy as np

from pf_sintering.constrained_densification_relaxation import (
    constraint_values, multigrain_energy, volume_targets)
from pf_sintering.constrained_newton_krylov import (
    active_interface_mask, hessian_finite_difference_check,
    projected_hessian_eigenpairs, solve_constrained_stationary)
from test_constrained_densification_relaxation import _case


def test_analytic_hessian_action_matches_fixed_ownership_gradient_difference():
    f, ownership, g = _case()
    rng = np.random.default_rng(17)
    direction = rng.normal(size=f.shape)
    direction /= np.max(np.abs(direction))
    check = hessian_finite_difference_check(
        f, ownership, g, direction, epsilon=2e-7)
    assert check["relative_Linf"] < 2e-8


def test_reduced_solver_preserves_constraints_bounds_and_monotonic_energy():
    f, ownership, g = _case()
    target = constraint_values(
        f, ownership, g["z"], g["r_c"], g["config"].outer_radius)
    perturbed = f+2e-4*np.exp(-((g["z"][:, None]/2e-9)**2
                                +(g["r_c"][None, :]/2e-9)**2))
    initial = multigrain_energy(perturbed, ownership, g)
    result = solve_constrained_stationary(
        perturbed, ownership, g, target, halo_cells=2,
        lbfgs_max_iterations=8, newton_max_iterations=2,
        kkt_tolerance=1e-12)
    energies = [row["energy_J"] for row in result.history]
    assert multigrain_energy(result.f, ownership, g) <= initial
    assert all(b <= a+1e-30 for a, b in zip(energies, energies[1:]))
    actual = constraint_values(
        result.f, ownership, g["z"], g["r_c"], g["config"].outer_radius)
    np.testing.assert_allclose(actual, target, rtol=2e-12, atol=1e-10)
    assert result.f.min() >= -2e-14
    assert result.f.max() <= 1+2e-14


def test_volume_only_solver_preserves_grain_volumes_without_fixing_moments():
    f, ownership, g = _case()
    target = volume_targets(
        f, ownership, g["z"], g["r_c"], g["config"].outer_radius)
    perturbation = 2e-4*np.exp(-((g["z"][:, None]/2e-9)**2
                                  +(g["r_c"][None, :]/2e-9)**2))
    result = solve_constrained_stationary(
        f+perturbation, ownership, g, target, halo_cells=2,
        lbfgs_max_iterations=8, newton_max_iterations=2,
        kkt_tolerance=1e-12, include_moments=False)
    actual = constraint_values(
        result.f, ownership, g["z"], g["r_c"],
        g["config"].outer_radius, include_moments=False)
    np.testing.assert_allclose(actual, target, rtol=2e-12, atol=1e-10)
    assert result.f.min() >= -2e-14
    assert result.f.max() <= 1+2e-14


def test_projected_hessian_modes_are_volume_tangent():
    f, ownership, g = _case()
    active = active_interface_mask(
        f, threshold=0.02, tail_threshold=1e-4, halo_cells=2)
    spectrum = projected_hessian_eigenpairs(
        f, ownership, g, active_mask=active, include_moments=False,
        eigenpair_count=2, tolerance=1e-6)
    assert spectrum["eigenvalues"].shape == (2,)
    assert spectrum["modes"].shape == (2,)+f.shape
    assert np.max(spectrum["tangent_residuals"]) < 1e-8
