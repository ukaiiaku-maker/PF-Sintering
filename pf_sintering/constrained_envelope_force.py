"""Work-conjugate force from a stationary constrained Lagrangian.

The parameter derivative is taken through the prescribed ownership field at
fixed total-solid field.  This avoids subtracting neighboring minimized total
energies while retaining both the explicit objective derivative and the
explicit derivative of the volume-constraint basis.
"""
from __future__ import annotations

import math

import numpy as np

from .constrained_densification_relaxation import (
    constraint_basis, constraint_values, fixed_ownership_mu,
    multigrain_energy,
)
from .constrained_newton_krylov import (
    LinearConstraintProjector, _feasible_reduced_gradient,
)
from .three_particle_phase_a import FrozenPhysics


def recover_kkt_multipliers(
        f, ownership, g, *, active_mask, include_moments=False,
        box_minimum_step=1e-2):
    """Recover physical KKT multipliers on the solver's free active set."""
    field = np.asarray(f, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != field.shape:
        raise ValueError("active_mask shape does not match field")
    scale = g["config"].outer_radius
    basis = constraint_basis(
        ownership, g["z"], scale, include_moments=include_moments)
    weight = np.broadcast_to(g["r_c"][None, :]/scale, field.shape)
    projector = LinearConstraintProjector(basis, weight, active)
    ids = np.flatnonzero(active)
    Wf = 12.0*FrozenPhysics().gamma_s/g["config"].width
    raw = (weight*fixed_ownership_mu(field, ownership, g)/Wf).ravel()[ids]
    projected, free = _feasible_reduced_gradient(
        raw, field.ravel()[ids], projector.A,
        minimum_step=box_minimum_step)
    jacobian = (basis*weight[None, :, :]).reshape(len(basis), -1)
    jacobian = jacobian[:, ids][:, free]
    beta = -np.linalg.solve(jacobian@jacobian.T, jacobian@raw[free])
    energy_scale = (2.0*math.pi*g["dr"]*g["dz"]*scale*Wf)
    multipliers = energy_scale*beta
    raw_residual = raw[free]+jacobian.T@beta
    return dict(
        multipliers=multipliers,
        KKT_Linf=float(np.max(np.abs(raw_residual))),
        projected_KKT_Linf=float(np.max(np.abs(projected))),
        active_cells=int(len(ids)), free_cells=int(np.count_nonzero(free)),
        gram_condition=projector.gram_condition)


def centered_ownership_parameter_derivative(
        f, ownership_minus, ownership_plus, g, delta_u,
        *, include_moments=False):
    """Return explicit ``G_u`` and ``C_u`` at fixed ``f``."""
    du = float(delta_u)
    if du <= 0.0:
        raise ValueError("delta_u must be positive")
    Gu = (multigrain_energy(f, ownership_plus, g)
          - multigrain_energy(f, ownership_minus, g))/(2.0*du)
    scale = g["config"].outer_radius
    Cplus = constraint_values(
        f, ownership_plus, g["z"], g["r_c"], scale,
        include_moments=include_moments)
    Cminus = constraint_values(
        f, ownership_minus, g["z"], g["r_c"], scale,
        include_moments=include_moments)
    return dict(G_u=float(Gu), C_u=(Cplus-Cminus)/(2.0*du))


def stationary_envelope_force(
        f, ownership, ownership_minus, ownership_plus, g, delta_u,
        *, active_mask, include_moments=False, box_minimum_step=1e-2):
    """Evaluate ``-(G_u + lambda.T C_u)`` at one stationary field."""
    kkt = recover_kkt_multipliers(
        f, ownership, g, active_mask=active_mask,
        include_moments=include_moments,
        box_minimum_step=box_minimum_step)
    derivative = centered_ownership_parameter_derivative(
        f, ownership_minus, ownership_plus, g, delta_u,
        include_moments=include_moments)
    constraint_term = float(np.dot(kkt["multipliers"], derivative["C_u"]))
    force = -(derivative["G_u"]+constraint_term)
    return dict(**kkt, **derivative,
                constraint_term_N=constraint_term,
                envelope_force_N=float(force))
