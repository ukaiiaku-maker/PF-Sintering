import numpy as np

from pf_sintering.constrained_densification_relaxation import (
    multigrain_energy, volume_targets)
from pf_sintering.constrained_envelope_force import stationary_envelope_force
from pf_sintering.constrained_newton_krylov import solve_constrained_stationary
from test_constrained_densification_relaxation import _case


def test_stationary_envelope_matches_small_full_domain_branch_difference():
    f0, _, g = _case()
    z = g["z"]

    def ownership(u):
        right = np.broadcast_to(
            0.5*(1+np.tanh((z[:, None]+0.5*u)/1e-9)), f0.shape).copy()
        return np.stack([1-right, right])

    target = volume_targets(
        f0, ownership(0.0), z, g["r_c"], g["config"].outer_radius)
    active = np.ones_like(f0, dtype=bool)
    u0 = 0.25e-9
    center = solve_constrained_stationary(
        f0, ownership(u0), g, target, active_mask=active,
        lbfgs_max_iterations=4000, newton_max_iterations=1000,
        include_moments=False, kkt_tolerance=1e-8)
    assert center.converged
    du = 0.025e-9
    minus = ownership(u0-du)
    plus = ownership(u0+du)
    envelope = stationary_envelope_force(
        center.f, ownership(u0), minus, plus, g, du,
        active_mask=active, include_moments=False)
    lower = solve_constrained_stationary(
        center.f, minus, g, target, active_mask=active,
        lbfgs_max_iterations=2000, newton_max_iterations=1000,
        include_moments=False, kkt_tolerance=1e-8)
    upper = solve_constrained_stationary(
        center.f, plus, g, target, active_mask=active,
        lbfgs_max_iterations=2000, newton_max_iterations=1000,
        include_moments=False, kkt_tolerance=1e-8)
    assert lower.converged and upper.converged
    branch_force = -(multigrain_energy(upper.f, plus, g)
                     - multigrain_energy(lower.f, minus, g))/(2*du)
    np.testing.assert_allclose(
        envelope["envelope_force_N"], branch_force, rtol=1e-5, atol=1e-18)
    assert envelope["KKT_Linf"] < 1e-8
