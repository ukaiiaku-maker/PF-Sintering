"""Discarded-copy constrained relaxation for a prescribed displacement.

The normalized ownership field is fixed during each minimization.  Its
selected GB midplane therefore remains at the prescribed kinematic coordinate.
The constraint helpers support either per-grain volumes alone or the earlier
volume-plus-first-moment manifold.  The volume-only family lets surface
redistribution move the material centroid while the ownership/GB frame retains
the prescribed mechanical displacement.

The optimizer is a mass/moment projected, interface-localized energy descent.
It is used to locate a constrained stationary field, not to assign physical
time.  A stationary result is compatible with conservative surface diffusion
because its chemical potential has no component in the admissible tangent
space.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .axisym import axisym_laplacian
from .corrected_interfacial_energy import (
    _axisym_gate_derivative_of_weighted_gradient,
    _axisym_weighted_gradient_energy,
)
from .gb_obstacle_energy import gb_obstacle_coefficients
from .three_particle_phase_a import FrozenPhysics


def constraint_basis(ownership, z, length_scale, *, include_moments=True):
    """Return volume basis fields and, optionally, scaled first moments."""
    phi = np.asarray(ownership, dtype=float)
    if not include_moments:
        return phi.copy()
    return np.concatenate([phi, phi*np.asarray(z)[None, :, None]
                           /float(length_scale)], axis=0)


def constraint_values(f, ownership, z, r_c, length_scale, *, include_moments=True):
    basis = constraint_basis(ownership, z, length_scale,
                             include_moments=include_moments)
    weight = np.asarray(r_c)[None, :]/float(length_scale)
    return np.einsum("azr,zr,zr->a", basis, f, weight, optimize=True)


def rigid_translation_targets(f, ownership, z, r_c, length_scale,
                              moving_grain, displacement_m):
    """Targets that preserve every grain volume and translate one centroid."""
    target = constraint_values(f, ownership, z, r_c, length_scale)
    n = len(ownership)
    target[n+moving_grain] -= (
        float(displacement_m)/float(length_scale)*target[moving_grain])
    return target


def volume_targets(f, ownership, z, r_c, length_scale):
    """Per-grain material volumes with no centroid/first-moment constraint."""
    return constraint_values(f, ownership, z, r_c, length_scale,
                             include_moments=False)


def multigrain_energy(f, ownership, g, physics=FrozenPhysics()):
    """Exact fixed-ownership Phase-A energy using positive face ledgers."""
    phi = np.asarray(ownership, dtype=float)
    W = g["config"].width
    Wf = 12.0*physics.gamma_s/W
    kf = 3.0*physics.gamma_s*W
    coef = gb_obstacle_coefficients(physics.gamma_gb, W)
    factor = 2.0*math.pi*g["dr"]*g["dz"]
    rc = g["r_c"][None, :]
    bulk = factor*float(np.sum(rc*0.5*Wf*f*f*(1.0-f)**2))
    surface_gradient = _axisym_weighted_gradient_energy(
        f, np.ones_like(f), 0.5*kf, g["dr"], g["dz"], g["r_c"], g["r_f"])
    pair = sum(phi[i]*phi[j] for i in range(len(phi)) for j in range(i+1, len(phi)))
    gb_bulk = factor*float(np.sum(rc*f*coef["Wc"]*pair))
    gb_gradient = sum(_axisym_weighted_gradient_energy(
        p, f, 0.5*coef["k_eta"], g["dr"], g["dz"], g["r_c"], g["r_f"])
        for p in phi)
    return bulk+surface_gradient+gb_bulk+gb_gradient


def fixed_ownership_mu(f, ownership, g, physics=FrozenPhysics()):
    phi = np.asarray(ownership, dtype=float)
    W = g["config"].width
    Wf = 12.0*physics.gamma_s/W
    kf = 3.0*physics.gamma_s*W
    coef = gb_obstacle_coefficients(physics.gamma_gb, W)
    gb_density = coef["Wc"]*sum(
        phi[i]*phi[j] for i in range(len(phi)) for j in range(i+1, len(phi)))
    for p in phi:
        gb_density += _axisym_gate_derivative_of_weighted_gradient(
            p, 0.5*coef["k_eta"], g["dr"], g["dz"], g["r_c"], g["r_f"])
    return (Wf*f*(1.0-f)*(1.0-2.0*f)
            - kf*axisym_laplacian(f, g["dr"], g["dz"],
                                  g["r_c"], g["r_f"], bc_z="noflux")
            + gb_density)


def restore_constraints(f, ownership, g, targets, *, tolerance=1e-10,
                        max_iterations=100, include_moments=True):
    """Move an initial guess onto common linear constraints without clipping."""
    field = np.asarray(f, dtype=float).copy()
    scale = g["config"].outer_radius
    basis = constraint_basis(ownership, g["z"], scale,
                             include_moments=include_moments)
    weight = g["r_c"][None, :]/scale
    for iteration in range(max_iterations):
        current = np.einsum("azr,zr,zr->a", basis, field, weight, optimize=True)
        residual = np.asarray(targets)-current
        if float(np.max(np.abs(residual))) <= tolerance:
            return field, dict(iterations=iteration,
                absolute_constraint_residual=float(np.max(np.abs(residual))))
        mobility = np.maximum(field*(1.0-field), 1e-30)
        gram = np.einsum("azr,bzr,zr,zr->ab", basis, basis, mobility,
                         weight, optimize=True)
        coefficients = np.linalg.solve(gram, residual)
        direction = mobility*np.einsum(
            "a,azr->zr", coefficients, basis, optimize=True)
        step = 1.0
        while step > 1e-12:
            trial = field+step*direction
            if trial.min() >= -2e-14 and trial.max() <= 1.0+2e-14:
                field = trial
                break
            step *= 0.5
        else:
            raise RuntimeError("constraint restoration reached its bound-safe step floor")
    raise RuntimeError("constraint restoration did not converge")


@dataclass(frozen=True)
class RelaxationResult:
    f: np.ndarray
    history: tuple[dict, ...]
    converged: bool
    reason: str
    projected_residual: float
    constraint_residual: float


def relax_constrained(f, ownership, g, targets, *, max_iterations=500,
                      initial_step=0.05, maximum_step=0.1,
                      projected_tolerance=1e-7,
                      energy_window_tolerance_J=1e-20,
                      record_every=25):
    """Energy descent on the fixed-displacement volume/moment manifold."""
    field = np.asarray(f, dtype=float).copy()
    phi = np.asarray(ownership, dtype=float)
    scale = g["config"].outer_radius
    basis = constraint_basis(phi, g["z"], scale)
    weight = g["r_c"][None, :]/scale
    Wf = 12.0*FrozenPhysics().gamma_s/g["config"].width
    energy = multigrain_energy(field, phi, g)
    history = []
    recent = [energy]
    step = float(initial_step)
    residual = math.inf
    for iteration in range(1, max_iterations+1):
        mu = fixed_ownership_mu(field, phi, g)
        mobility = np.maximum(field*(1.0-field), 0.0)
        gram = np.einsum("azr,bzr,zr,zr->ab", basis, basis, mobility,
                         weight, optimize=True)
        rhs = np.einsum("azr,zr,zr,zr->a", basis, mobility, mu/Wf,
                        weight, optimize=True)
        multipliers = np.linalg.solve(gram, rhs)
        direction = -mobility*(mu/Wf-np.einsum(
            "a,azr->zr", multipliers, basis, optimize=True))
        residual = float(np.max(np.abs(direction)))
        trial_step = step
        for _ in range(30):
            trial = field+trial_step*direction
            if trial.min() >= -2e-14 and trial.max() <= 1.0+2e-14:
                trial_energy = multigrain_energy(trial, phi, g)
                if trial_energy <= energy:
                    break
            trial_step *= 0.5
        else:
            return RelaxationResult(field, tuple(history), False,
                "line_search_floor", residual, math.inf)
        field = trial
        energy = trial_energy
        step = min(float(maximum_step), 1.2*trial_step)
        recent.append(energy)
        if len(recent) > 101:
            recent.pop(0)
        current = np.einsum("azr,zr,zr->a", basis, field, weight, optimize=True)
        constraint_residual = float(np.max(np.abs(current-np.asarray(targets))))
        if iteration == 1 or iteration % record_every == 0:
            history.append(dict(iteration=iteration, energy_J=energy,
                projected_residual=residual, accepted_step=trial_step,
                absolute_constraint_residual=constraint_residual,
                energy_drop_window_J=(recent[0]-recent[-1])))
        if (residual <= projected_tolerance and len(recent) == 101
                and recent[0]-recent[-1] <= energy_window_tolerance_J):
            return RelaxationResult(field, tuple(history), True,
                "stationary", residual, constraint_residual)
    return RelaxationResult(field, tuple(history), False,
        "iteration_limit", residual, constraint_residual)
