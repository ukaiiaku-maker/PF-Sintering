"""Dimensionally explicit triple-junction thermodynamic observables.

The conserved phase-field chemical potential is ``delta F / delta f``.  Since
``f`` is dimensionless, its units are ``J/m^3 = Pa``.  This module keeps that
volumetric quantity distinct from the TJ line excess (``Pa m = N/m``) and
from the creep-equivalent activation stress obtained by dividing the line
excess by an explicitly physical TJ width.

No fitted spatial width is used.  The TJ is the intrinsic intersection of the
resolved GB indicator ``eta1*eta2`` and the free-surface indicator
``16*f^2*(1-f)^2``.  The numerical diffuse width therefore remains observable
in a subsequent W-convergence audit rather than being hidden in a chosen
Gaussian or cutoff.
"""
from __future__ import annotations

import numpy as np

from .axisym import axisym_mu_f_gb


def tj_geometric_indicators(f, eta1, eta2):
    """Return non-negative GB, free-surface, and intrinsic TJ indicators."""
    f_clip = np.clip(np.asarray(f, dtype=float), 0.0, 1.0)
    gb = np.maximum(np.asarray(eta1, dtype=float) * np.asarray(eta2, dtype=float), 0.0)
    free_surface = 16.0 * f_clip * f_clip * (1.0 - f_clip) ** 2
    return gb, free_surface, gb * free_surface


def tj_functionals_from_mu(mu_Pa, f, eta1, eta2, dr, dz, r_c):
    """Evaluate raw TJ-local volumetric and line functionals.

    ``mu_PF_TJ_Pa`` is an axisymmetrically normalized mean over the intrinsic
    TJ intersection.  To form the line functional, ``mu`` is first averaged
    through the GB-normal (z) direction at every radial cell.  That profile is
    then integrated in the radial direction through the geometrically selected
    free-surface/GB intersection.  The latter operation is per unit length of
    the axisymmetric TJ ring and therefore has units ``Pa*m = N/m``; it does
    not include a ``2*pi*r_TJ`` factor.
    """
    mu = np.asarray(mu_Pa, dtype=float)
    f = np.asarray(f, dtype=float)
    eta1 = np.asarray(eta1, dtype=float)
    eta2 = np.asarray(eta2, dtype=float)
    r_c = np.asarray(r_c, dtype=float)
    if mu.shape != f.shape or eta1.shape != f.shape or eta2.shape != f.shape:
        raise ValueError("mu, f, eta1, and eta2 must have identical shapes")
    if f.ndim != 2 or f.shape[1] != r_c.size:
        raise ValueError("fields must be (z,r) arrays matching r_c")
    if dr <= 0.0 or dz <= 0.0:
        raise ValueError("dr and dz must be positive")

    gb, free_surface, tj = tj_geometric_indicators(f, eta1, eta2)
    radial_weight = r_c[None, :]
    tj_axisym_norm = float(np.sum(radial_weight * tj))
    if not np.isfinite(tj_axisym_norm) or tj_axisym_norm <= 0.0:
        raise ValueError("intrinsic TJ support has zero or non-finite measure")
    mu_tj = float(np.sum(radial_weight * tj * mu) / tj_axisym_norm)

    gb_column_norm = np.sum(gb, axis=0)
    valid = gb_column_norm > 1e-300
    mu_gb_profile = np.zeros_like(r_c)
    surface_intersection_profile = np.zeros_like(r_c)
    mu_gb_profile[valid] = (
        np.sum(mu * gb, axis=0)[valid] / gb_column_norm[valid])
    surface_intersection_profile[valid] = (
        np.sum(gb * free_surface, axis=0)[valid] / gb_column_norm[valid])
    line_support_length = float(np.sum(surface_intersection_profile) * dr)
    if not np.isfinite(line_support_length) or line_support_length <= 0.0:
        raise ValueError("TJ line support has zero or non-finite length")
    lambda_raw = float(
        np.sum(mu_gb_profile * surface_intersection_profile) * dr)

    return dict(
        mu_PF_TJ_Pa=mu_tj,
        lambda_TJ_raw_N_per_m=lambda_raw,
        line_support_length_m=line_support_length,
        line_weighted_mu_Pa=lambda_raw / line_support_length,
        tj_axisym_support_norm=tj_axisym_norm,
        support_definition=(
            "intrinsic (eta1*eta2)*16*f^2*(1-f)^2 intersection; no fitted width"),
        line_definition=(
            "GB-normal-averaged mu integrated radially through intrinsic "
            "free-surface/GB intersection; per unit TJ-ring length"),
        circumference_multiplier_applied=False,
    )


def tj_thermodynamic_state(
        f, eta1, eta2, pf_params, Wc, dr, dz, r_c, r_f, *, bc_z="noflux"):
    """Evaluate the TJ functionals from the authoritative PF chemical potential."""
    mu = axisym_mu_f_gb(
        f, eta1, eta2, pf_params, Wc, dr, dz, r_c, r_f, bc_z=bc_z)
    return tj_functionals_from_mu(mu, f, eta1, eta2, dr, dz, r_c)


def add_equilibrium_referenced_tj_stresses(state, equilibrium, physical_width_m):
    """Reference TJ measures to equilibrium and map line excess through width.

    The line excess is gauge-safe for a constant PF-chemical-potential shift:
    the equilibrium line-weighted mean is subtracted under the *current*
    geometrical line support before integration.  Evaluating the equilibrium
    state against itself is therefore exactly zero apart from roundoff even if
    its represented support length differs from an evolving state.
    """
    if physical_width_m <= 0.0:
        raise ValueError("physical_width_m must be positive")
    mu_eq = float(equilibrium["mu_PF_TJ_Pa"])
    lambda_eq = float(equilibrium["lambda_TJ_raw_N_per_m"])
    length_eq = float(equilibrium["line_support_length_m"])
    if length_eq <= 0.0:
        raise ValueError("equilibrium line support length must be positive")
    line_mu_eq = lambda_eq / length_eq
    lambda_excess = (
        float(state["lambda_TJ_raw_N_per_m"])
        - line_mu_eq * float(state["line_support_length_m"]))
    delta_mu = float(state["mu_PF_TJ_Pa"]) - mu_eq
    out = dict(state)
    out.update(
        mu_PF_TJ_eq_Pa=mu_eq,
        delta_mu_PF_TJ_Pa=delta_mu,
        sigma_act_TJ_mu_Pa=delta_mu,
        lambda_TJ_eq_raw_N_per_m=lambda_eq,
        line_weighted_mu_eq_Pa=line_mu_eq,
        lambda_TJ_excess_N_per_m=lambda_excess,
        physical_TJ_width_m=float(physical_width_m),
        physical_TJ_width_definition="w_TJ=b (physical, not diffuse W)",
        sigma_act_TJ_lambda_Pa=lambda_excess / physical_width_m,
        dimensional_nomenclature=dict(
            mu_PF="J/m^3 = N/m^2 = Pa",
            lambda_TJ="J/m^2 = N/m",
            sigma_act_TJ="Pa",
        ),
    )
    return out
