"""Parallel corrected interfacial free-energy formulations.

This module is intentionally not connected to any production evolution
operator.  The promoted static candidate represents ownership by the
normalized fraction ``phi=eta1/f`` and gates its obstacle GB energy linearly
by the solid field ``f``:

    psi = psi_surface(f)
        + f [Wc phi(1-phi) + k_eta |grad phi|^2].

It gives exactly the calibrated pure-f surface for a single grain, exactly
the calibrated obstacle GB at ``f=1``, and no homogeneous bulk offset.  The
linear ``f`` gate also uses the natural Gibbs dividing surface of the
symmetric signed-distance profile, removing the leading artificial diffuse
TJ line excess.

The earlier algebraic ownership-background subtraction is retained under an
explicit diagnostic name.  Its density is

    psi = (W_f/2) f^2 (1-f)^2 + (k_f/2) |grad f|^2
        + Wc eta1 eta2 f (2-f)
        + (k_eta/2) (|grad eta1|^2 + |grad eta2|^2 - |grad f|^2).

Although it passes the two binary interfaces, it leaves a resolved finite-TJ
shift in the spherical-cap audit and is not the promoted candidate.
"""
from __future__ import annotations

import math

import numpy as np

from .axisym import axisym_laplacian


def axisym_background_subtracted_interfacial_energy_components(
    f: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    p,
    Wc: float,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
    *,
    bc_z: str = "periodic",
) -> dict[str, float]:
    """Evaluate the rejected algebraic-background diagnostic candidate."""
    f = np.asarray(f, dtype=float)
    eta1 = np.asarray(eta1, dtype=float)
    eta2 = np.asarray(eta2, dtype=float)
    fb = np.clip(f, 0.0, 1.0)
    lap_f = axisym_laplacian(f, dr, dz, r_c, r_f, bc_z=bc_z)
    lap_eta1 = axisym_laplacian(eta1, dr, dz, r_c, r_f, bc_z=bc_z)
    lap_eta2 = axisym_laplacian(eta2, dr, dz, r_c, r_f, bc_z=bc_z)
    densities = {
        "F_surface_bulk": 0.5 * p.W_f * f*f * (1.0 - f)**2,
        "F_surface_gradient": -0.5 * p.k_f * f * lap_f,
        "F_GB_coupling_excess": Wc * eta1 * eta2 * fb * (2.0 - fb),
        "F_GB_gradient_excess": -0.5 * p.k_eta * (
            eta1 * lap_eta1 + eta2 * lap_eta2 - f * lap_f),
    }
    factor = 2.0 * math.pi * dr * dz
    weights = np.asarray(r_c, dtype=float)[None, :]
    result = {
        name: factor * float(np.sum(weights * density))
        for name, density in densities.items()
    }
    result["F_surface"] = (
        result["F_surface_bulk"] + result["F_surface_gradient"])
    result["F_GB_excess"] = (
        result["F_GB_coupling_excess"]
        + result["F_GB_gradient_excess"])
    result["F_total_corrected"] = result["F_surface"] + result["F_GB_excess"]
    return result


def _normalized_ownership(eta1: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Bounded ``phi=eta1/f`` with a harmless convention in empty vacuum."""
    phi = np.zeros_like(f, dtype=float)
    np.divide(eta1, f, out=phi, where=np.abs(f) > 1.0e-14)
    return np.clip(phi, 0.0, 1.0)


def _axisym_weighted_gradient_energy(
    phi: np.ndarray,
    gate: np.ndarray,
    coefficient: float,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
) -> float:
    """Positive face-based integral of ``coefficient*gate*|grad phi|^2``."""
    # Radial internal faces. No-flux at the axis and outer boundary.
    delta_r = phi[:, 1:] - phi[:, :-1]
    gate_r = 0.5 * (gate[:, 1:] + gate[:, :-1])
    radial = (
        2.0 * math.pi * coefficient * dz / dr
        * float(np.sum(r_f[None, 1:-1] * gate_r * delta_r*delta_r)))

    # Axial internal faces. Static audit domains use no-flux end boundaries.
    delta_z = phi[1:, :] - phi[:-1, :]
    gate_z = 0.5 * (gate[1:, :] + gate[:-1, :])
    axial = (
        2.0 * math.pi * coefficient * dr / dz
        * float(np.sum(r_c[None, :] * gate_z * delta_z*delta_z)))
    return radial + axial


def axisym_corrected_interfacial_energy_components(
    f: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    p,
    Wc: float,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
    *,
    bc_z: str = "noflux",
) -> dict[str, float]:
    """Evaluate the promoted normalized-ownership static formulation.

    ``bc_z`` is currently restricted to ``noflux`` because the positive
    face-based weighted-gradient ledger deliberately has no wrap face.  No
    evolution operator uses this function.
    """
    if bc_z != "noflux":
        raise ValueError("normalized-ownership static energy requires noflux z")
    f = np.asarray(f, dtype=float)
    eta1 = np.asarray(eta1, dtype=float)
    eta2 = np.asarray(eta2, dtype=float)
    closure = float(np.max(np.abs(eta1 + eta2 - f)))
    if closure > 1.0e-10:
        raise ValueError(f"eta1+eta2=f closure violated: {closure:.3e}")
    fb = np.clip(f, 0.0, 1.0)
    phi = _normalized_ownership(eta1, f)
    lap_f = axisym_laplacian(f, dr, dz, r_c, r_f, bc_z=bc_z)
    factor = 2.0 * math.pi * dr * dz
    weights = np.asarray(r_c, dtype=float)[None, :]
    surface_bulk = factor * float(np.sum(
        weights * (0.5 * p.W_f * f*f * (1.0-f)**2)))
    surface_gradient = factor * float(np.sum(
        weights * (-0.5 * p.k_f * f * lap_f)))
    gb_coupling = factor * float(np.sum(
        weights * (Wc * fb * phi * (1.0-phi))))
    gb_gradient = _axisym_weighted_gradient_energy(
        phi, fb, p.k_eta, dr, dz, r_c, r_f)
    return dict(
        F_surface_bulk=surface_bulk,
        F_surface_gradient=surface_gradient,
        F_GB_coupling=gb_coupling,
        F_GB_gradient=gb_gradient,
        F_surface=surface_bulk + surface_gradient,
        F_GB=gb_coupling + gb_gradient,
        F_total_corrected=(surface_bulk + surface_gradient
                           + gb_coupling + gb_gradient),
        partition_closure_max=closure,
    )


def _axisym_gate_derivative_of_weighted_gradient(
    phi: np.ndarray,
    coefficient: float,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
) -> np.ndarray:
    """Cell derivative of the face energy ``coefficient*f*|grad(phi)|^2``.

    The gate is arithmetic-averaged to each internal face by
    :func:`_axisym_weighted_gradient_energy`.  Splitting every face derivative
    equally between its two adjacent cells gives the expression below after
    division by the axisymmetric cell volume.  This is consequently the exact
    discrete derivative, not a separately discretized cell-gradient norm.
    """
    phi = np.asarray(phi, dtype=float)
    out = np.zeros_like(phi)
    delta_r = phi[:, 1:] - phi[:, :-1]
    radial_numerator = (
        coefficient * r_f[None, 1:-1] * delta_r * delta_r
        / (2.0 * dr * dr))
    out[:, :-1] += radial_numerator / r_c[None, :-1]
    out[:, 1:] += radial_numerator / r_c[None, 1:]
    delta_z = phi[1:, :] - phi[:-1, :]
    axial_face = coefficient * delta_z * delta_z / (2.0 * dz * dz)
    out[:-1, :] += axial_face
    out[1:, :] += axial_face
    return out


def _axisym_divergence_f_grad_phi(
    f: np.ndarray,
    phi: np.ndarray,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
) -> np.ndarray:
    """No-flux FV divergence of arithmetic-face-averaged ``f grad(phi)``."""
    f = np.asarray(f, dtype=float)
    phi = np.asarray(phi, dtype=float)
    radial = np.zeros((f.shape[0], f.shape[1] + 1), dtype=float)
    radial[:, 1:-1] = (
        0.5 * (f[:, :-1] + f[:, 1:])
        * (phi[:, 1:] - phi[:, :-1]) / dr)
    axial = np.zeros_like(f)
    axial[:-1, :] = (
        0.5 * (f[:-1, :] + f[1:, :])
        * (phi[1:, :] - phi[:-1, :]) / dz)
    div_r = (
        r_f[None, 1:] * radial[:, 1:]
        - r_f[None, :-1] * radial[:, :-1]
    ) / (r_c[None, :] * dr)
    div_z = np.empty_like(f)
    div_z[0, :] = axial[0, :] / dz
    div_z[1:, :] = (axial[1:, :] - axial[:-1, :]) / dz
    return div_r + div_z


def normalized_ownership_variational_derivatives_axisym(
    f: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    p,
    Wc: float,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
    *,
    bc_z: str = "noflux",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact discrete ``(delta G/delta f|phi, delta G/delta phi)``.

    ``f`` and normalized ownership ``phi`` are the independent variables.
    Both weighted-gradient terms use precisely the same internal-face ledger
    as the promoted static energy.  External radial and axial faces carry
    zero flux; no equilibrium contact angle enters either derivative.
    """
    if bc_z != "noflux":
        raise ValueError("normalized-ownership derivatives require noflux z")
    f = np.asarray(f, dtype=float)
    eta1 = np.asarray(eta1, dtype=float)
    eta2 = np.asarray(eta2, dtype=float)
    closure = float(np.max(np.abs(eta1 + eta2 - f)))
    if closure > 1.0e-10:
        raise ValueError(f"eta1+eta2=f closure violated: {closure:.3e}")
    phi = _normalized_ownership(eta1, f)
    lap_f = axisym_laplacian(f, dr, dz, r_c, r_f, bc_z="noflux")
    mu_f = (
        p.W_f * f * (1.0 - f) * (1.0 - 2.0 * f)
        - p.k_f * lap_f
        + Wc * phi * (1.0 - phi)
        + _axisym_gate_derivative_of_weighted_gradient(
            phi, p.k_eta, dr, dz, r_c, r_f))
    g_phi = (
        f * Wc * (1.0 - 2.0 * phi)
        - 2.0 * p.k_eta * _axisym_divergence_f_grad_phi(
            f, phi, dr, dz, r_c, r_f))
    return mu_f, g_phi


def background_subtracted_constrained_variational_derivatives_axisym(
    f: np.ndarray,
    eta1: np.ndarray,
    p,
    Wc: float,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
    *,
    bc_z: str = "periodic",
) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the rejected algebraic-background candidate.

    ``eta2=f-eta1`` is eliminated before variation.  These arrays document
    that diagnostic formulation; this function performs no time integration.
    """
    f = np.asarray(f, dtype=float)
    u = np.asarray(eta1, dtype=float)
    lap_f = axisym_laplacian(f, dr, dz, r_c, r_f, bc_z=bc_z)
    lap_u = axisym_laplacian(u, dr, dz, r_c, r_f, bc_z=bc_z)
    surface_bulk_derivative = p.W_f * f * (1.0 - f) * (1.0 - 2.0*f)
    coupling_df = Wc * u * (
        (2.0*f - f*f) + (f-u) * (2.0 - 2.0*f))
    coupling_du = Wc * (f - 2.0*u) * f * (2.0 - f)
    delta_f = (
        surface_bulk_derivative + coupling_df
        - p.k_f * lap_f + p.k_eta * lap_u)
    delta_u = coupling_du - 2.0 * p.k_eta * lap_u + p.k_eta * lap_f
    return delta_f, delta_u


def homogeneous_energy_densities() -> dict[str, float]:
    """The corrected vacuum and one-grain solid reference densities."""
    return {"vacuum_J_per_m3": 0.0, "one_grain_solid_J_per_m3": 0.0}
