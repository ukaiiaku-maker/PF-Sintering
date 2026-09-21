"""Independent axial configurational stress for the stationary PF functional.

This module evaluates the canonical (Eshelby/Korteweg) stress directly from
the implemented continuum energy density.  It does not differentiate an
ownership parameter and does not call the stationary-envelope machinery.
The volume-constraint multipliers enter as bulk pressure terms in the
augmented stationary Lagrangian.
"""
from __future__ import annotations

import math

import numpy as np

from .constrained_densification_relaxation import constraint_basis, fixed_ownership_mu
from .constrained_newton_krylov import LinearConstraintProjector, _feasible_reduced_gradient
from .corrected_interfacial_energy import _axisym_divergence_f_grad_phi
from .gb_obstacle_energy import gb_obstacle_coefficients
from .three_particle_phase_a import FrozenPhysics


def multigrain_variational_derivatives(
        f, ownership, g, physics=FrozenPhysics()):
    """Exact discrete derivatives of ``multigrain_energy`` in ``(f, phi)``.

    The ownership fields are differentiated before enforcing their local
    partition constraint.  Contracting these derivatives with any admissible
    direction satisfying ``sum(phi_i,u)=0`` gives the constrained directional
    derivative without an arbitrary ownership gauge.
    """
    field = np.asarray(f, dtype=float)
    phi = np.asarray(ownership, dtype=float)
    mu_f = fixed_ownership_mu(field, phi, g, physics)
    width = float(g["config"].width)
    coeff = gb_obstacle_coefficients(physics.gamma_gb, width)
    g_phi = []
    for i in range(len(phi)):
        others = sum(phi[j] for j in range(len(phi)) if j != i)
        g_phi.append(
            field*coeff["Wc"]*others
            - coeff["k_eta"]*_axisym_divergence_f_grad_phi(
                field, phi[i], g["dr"], g["dz"],
                g["r_c"], g["r_f"]))
    return mu_f, np.asarray(g_phi)


def dynamic_axial_configurational_balance(
        f, ownership, g, physics=FrozenPhysics()):
    """Axial canonical resultant and Euler--Lagrange bulk-force density.

    No stationarity or volume multiplier is assumed.  With the canonical
    convention used by :func:`axial_configurational_resultant`, continuum
    balance is ``dP_z/dz = integral_A sum_a EL_a a_,z dA`` when radial-boundary
    flux is negligible.
    """
    field = np.asarray(f, dtype=float)
    phi = np.asarray(ownership, dtype=float)
    result = axial_configurational_resultant(
        field, phi, g, np.zeros(len(phi)), physics)
    mu_f, g_phi = multigrain_variational_derivatives(field, phi, g, physics)
    f_z = np.gradient(field, float(g["dz"]), axis=0, edge_order=2)
    phi_z = np.gradient(phi, float(g["dz"]), axis=1, edge_order=2)
    local = mu_f*f_z+np.sum(g_phi*phi_z, axis=0)
    radial_weight = (
        2.0*math.pi*float(g["dr"])*np.asarray(g["r_c"])[None, :])
    bulk_force_per_length = np.sum(radial_weight*local, axis=1)
    return dict(
        z_m=np.asarray(g["z"], dtype=float).copy(),
        resultant_N=np.asarray(result["resultant_N"], dtype=float),
        components_N=result["components_N"],
        bulk_force_per_length_N_per_m=bulk_force_per_length,
        mu_f=mu_f, g_phi=g_phi)


def recover_volume_multipliers_for_configurational_stress(
        f, ownership, g, active_mask, physics=FrozenPhysics(),
        box_minimum_step=1e-2):
    """Recover volume KKT multipliers without the envelope-force module."""
    field = np.asarray(f, dtype=float)
    phi = np.asarray(ownership, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    scale = float(g["config"].outer_radius)
    basis = constraint_basis(phi, g["z"], scale, include_moments=False)
    weight = np.broadcast_to(g["r_c"][None, :]/scale, field.shape)
    projector = LinearConstraintProjector(basis, weight, active)
    ids = np.flatnonzero(active)
    Wf = 12.0*physics.gamma_s/g["config"].width
    raw = (weight*fixed_ownership_mu(field, phi, g, physics)/Wf).ravel()[ids]
    projected, free = _feasible_reduced_gradient(
        raw, field.ravel()[ids], projector.A,
        minimum_step=box_minimum_step)
    jacobian = (basis*weight[None, :, :]).reshape(len(basis), -1)
    jacobian = jacobian[:, ids][:, free]
    beta = -np.linalg.solve(jacobian@jacobian.T, jacobian@raw[free])
    factor = 2.0*math.pi*g["dr"]*g["dz"]*scale*Wf
    return dict(
        multipliers_J=factor*beta,
        KKT_Linf=float(np.max(np.abs(raw[free]+jacobian.T@beta))),
        projected_KKT_Linf=float(np.max(np.abs(projected))),
        free_cells=int(np.count_nonzero(free)))


def ownership_configurational_force(
        f, ownership, ownership_minus, ownership_plus, g, delta_u_m,
        volume_multipliers_J, physics=FrozenPhysics()):
    """Evaluate the ownership shape force from analytic PF variations.

    This is the volume integral of ``-(delta L/delta phi_i) phi_i,u``.
    It differentiates the implemented GB functional analytically and is
    independent of both minimized-energy differences and the envelope-force
    implementation.
    """
    field = np.asarray(f, dtype=float)
    phi = np.asarray(ownership, dtype=float)
    phi_u = ((np.asarray(ownership_plus, dtype=float)
              - np.asarray(ownership_minus, dtype=float))/(2.0*float(delta_u_m)))
    multipliers = np.asarray(volume_multipliers_J, dtype=float)
    if phi.shape != phi_u.shape or phi.shape[1:] != field.shape:
        raise ValueError("ownership fields have inconsistent shapes")
    width = float(g["config"].width)
    coeff = gb_obstacle_coefficients(physics.gamma_gb, width)
    volume_normalization = (
        2.0*math.pi*g["dr"]*g["dz"]*g["config"].outer_radius)
    multiplier_density = multipliers/volume_normalization

    energy_density_derivatives = []
    for i in range(len(phi)):
        others = sum(phi[j] for j in range(len(phi)) if j != i)
        energy_density_derivatives.append(
            field*coeff["Wc"]*others
            - coeff["k_eta"]*_axisym_divergence_f_grad_phi(
                field, phi[i], g["dr"], g["dz"], g["r_c"], g["r_f"]))
    energy_density_derivatives = np.asarray(energy_density_derivatives)
    constraint_density_derivatives = multiplier_density[:, None, None]*field
    volume_weights = (
        2.0*math.pi*g["dr"]*g["dz"]*np.asarray(g["r_c"])[None, None, :])
    energy_derivative = float(np.sum(
        volume_weights*energy_density_derivatives*phi_u))
    constraint_derivative = float(np.sum(
        volume_weights*constraint_density_derivatives*phi_u))
    return dict(
        energy_force_N=-energy_derivative,
        constraint_force_N=-constraint_derivative,
        configurational_force_N=-(energy_derivative+constraint_derivative),
        energy_derivative_N=energy_derivative,
        constraint_derivative_N=constraint_derivative)


def axial_configurational_resultant(
        f, ownership, g, volume_multipliers_J,
        physics=FrozenPhysics()):
    """Return the axial canonical-stress resultant on every z cross-section.

    For fields ``a=(f, phi_i)`` and augmented density ``ell``, the canonical
    stress used here is

    ``P_zz = ell - sum_a (d ell/d a_z) a_z``.

    The returned resultant is ``integral P_zz 2*pi*r dr``.  Cell-centered
    gradients provide a continuum-stress diagnostic on the archived discrete
    state; the stationary envelope calculation uses a separate discrete
    derivative and is therefore an independent implementation.
    """
    field = np.asarray(f, dtype=float)
    phi = np.asarray(ownership, dtype=float)
    multipliers = np.asarray(volume_multipliers_J, dtype=float)
    if phi.ndim != 3 or phi.shape[1:] != field.shape:
        raise ValueError("ownership must have shape (ngrains, nz, nr)")
    if multipliers.shape != (len(phi),):
        raise ValueError("one volume multiplier is required per grain")

    dr, dz = float(g["dr"]), float(g["dz"])
    width = float(g["config"].width)
    scale = float(g["config"].outer_radius)
    Wf = 12.0*physics.gamma_s/width
    kf = 3.0*physics.gamma_s*width
    coeff = gb_obstacle_coefficients(physics.gamma_gb, width)
    multiplier_density = multipliers/(2.0*math.pi*dr*dz*scale)

    fz = np.gradient(field, dz, axis=0, edge_order=2)
    fr = np.gradient(field, dr, axis=1, edge_order=2)
    phiz = np.gradient(phi, dz, axis=1, edge_order=2)
    phir = np.gradient(phi, dr, axis=2, edge_order=2)
    pair = sum(phi[i]*phi[j] for i in range(len(phi))
               for j in range(i+1, len(phi)))

    densities = {
        "surface_bulk": 0.5*Wf*field*field*(1.0-field)**2,
        "surface_gradient": 0.5*kf*(fr*fr-fz*fz),
        "gb_bulk": field*coeff["Wc"]*pair,
        "gb_gradient": 0.5*coeff["k_eta"]*field*sum(
            phir[i]*phir[i]-phiz[i]*phiz[i] for i in range(len(phi))),
        "constraint": field*np.tensordot(multiplier_density, phi, axes=(0, 0)),
    }
    radial_weight = 2.0*math.pi*dr*np.asarray(g["r_c"], dtype=float)[None, :]
    components = {
        name: np.sum(radial_weight*density, axis=1)
        for name, density in densities.items()
    }
    total = sum(components.values())
    return dict(
        z_m=np.asarray(g["z"], dtype=float).copy(),
        resultant_N=total,
        components_N=components,
        multiplier_density_Pa=multiplier_density,
        pressure_Pa=-multiplier_density)


def plateau_average(result, z_gb_m, width_m, *, side, lo_widths=3.0,
                    hi_widths=5.0):
    """Average a configurational resultant over a declared bulk-side window."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    z = np.asarray(result["z_m"])
    distance = side*(z-float(z_gb_m))/float(width_m)
    selected = (distance >= float(lo_widths)) & (distance <= float(hi_widths))
    if np.count_nonzero(selected) < 3:
        raise ValueError("configurational-stress plateau window is under-resolved")
    values = np.asarray(result["resultant_N"])[selected]
    row = dict(
        side=side, lo_widths=float(lo_widths), hi_widths=float(hi_widths),
        cells=int(np.count_nonzero(selected)), mean_N=float(np.mean(values)),
        standard_deviation_N=float(np.std(values)),
        minimum_N=float(np.min(values)), maximum_N=float(np.max(values)))
    for name, component in result["components_N"].items():
        row[f"{name}_mean_N"] = float(np.mean(np.asarray(component)[selected]))
    return row
