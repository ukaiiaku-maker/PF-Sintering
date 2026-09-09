"""Instantaneous thermodynamic observables for corrected PR production.

No reference/equilibrium contact angle appears here.  The continuous field
intersection supplies the current TJ; all chemical potentials follow from the
normalized-ownership functional at the current fields.
"""
from __future__ import annotations

import math

import numpy as np

from .axisym_branch_boundary_flux import extract_axisymmetric_surface_branches
from .corrected_interfacial_energy import (
    axisym_corrected_interfacial_energy_components,
    normalized_ownership_variational_derivatives_axisym,
)


def corrected_mu_f(state, setup) -> np.ndarray:
    """Return ``delta G/delta f`` at fixed instantaneous ownership fraction."""
    mu, _ = normalized_ownership_variational_derivatives_axisym(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], bc_z="noflux")
    return mu


def corrected_g_phi(state, setup) -> np.ndarray:
    """Return the current normalized-ownership variational derivative."""
    _, derivative = normalized_ownership_variational_derivatives_axisym(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], bc_z="noflux")
    return derivative


def corrected_total_energy(state, setup) -> float:
    return float(axisym_corrected_interfacial_energy_components(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], bc_z="noflux")[
            "F_total_corrected"])


def _axisym_weighted_mean(values, mask, r_c) -> float:
    weights = np.asarray(r_c, dtype=float)[None, :] * np.asarray(mask, bool)
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("thermodynamic sampling support is empty")
    return float(np.sum(weights * values) / denominator)


def _gb_plane_values(field, z, z_tj):
    """Linear interpolation of a cell field to the current flat GB plane."""
    upper = int(np.searchsorted(z, z_tj, side="right"))
    upper = min(max(upper, 1), len(z) - 1)
    lower = upper - 1
    fraction = float((z_tj - z[lower]) / (z[upper] - z[lower]))
    return (1.0 - fraction) * field[lower, :] + fraction * field[upper, :]


def _radial_area_weights_to_cutoff(r_f, cutoff_m):
    """Exact axisymmetric cell-area weights below a continuous radial cutoff.

    Treating the cell-centered chemical potential as piecewise constant gives
    a weight proportional to ``integral(r dr)`` over the portion of each cell
    below the cutoff.  Unlike a Boolean center mask, the support changes
    continuously as the field-defined TJ crosses a radial grid location.
    The common factor ``2*pi`` cancels in a weighted mean.
    """
    faces = np.asarray(r_f, dtype=float)
    if faces.ndim != 1 or len(faces) < 2 or np.any(np.diff(faces) <= 0.0):
        raise ValueError("radial faces must be a strictly increasing vector")
    cutoff = float(cutoff_m)
    upper = np.minimum(faces[1:], cutoff)
    lower = faces[:-1]
    return 0.5 * np.maximum(upper * upper - lower * lower, 0.0)


def make_corrected_state_evaluator(setup, geom, tj_tracker):
    """Build the event/node evaluator for the corrected functional.

    ``mu_GB_source`` is a radial-area-weighted sample on the ownership GB
    center plane, ending three interface widths before the TJ.  It is not the
    legacy diffuse broad-contact average.  ``mu_TJ_local`` independently
    averages the two current free-surface branches over 1--3 W from the TJ.
    The zero-storage node is solved downstream and remains only a kinetic
    partition variable.
    """
    z = np.asarray(setup["z"], dtype=float)
    r_c = np.asarray(setup["r_c"], dtype=float)
    r_f = np.asarray(setup["r_f"], dtype=float)
    width = float(setup["W"])

    def evaluate(f, particle, substrate):
        state = (np.asarray(f), np.asarray(particle), np.asarray(substrate))
        tracking = tj_tracker.locate(*state)
        z_tj = float(tracking["z_TJ_m"])
        r_tj = float(tracking["r_TJ_m"])
        mu = corrected_mu_f(state, setup)

        phi = np.zeros_like(f, dtype=float)
        np.divide(particle, f, out=phi, where=np.abs(f) > 1.0e-14)
        pure_dense = (
            (f >= 0.99)
            & ((phi <= 0.02) | (phi >= 0.98)))
        mu_bulk = _axisym_weighted_mean(mu, pure_dense, r_c)

        mu_on_gb = _gb_plane_values(mu, z, z_tj)
        gb_cutoff = r_tj - 3.0 * width
        gb_weights = _radial_area_weights_to_cutoff(r_f, gb_cutoff)
        if np.count_nonzero(gb_weights) < 3:
            gb_cutoff = r_tj - width
            gb_weights = _radial_area_weights_to_cutoff(r_f, gb_cutoff)
        if np.count_nonzero(gb_weights) < 2:
            raise ValueError("corrected GB-centerline support is under-resolved")
        mu_gb_plane_continuous = float(
            np.sum(gb_weights * mu_on_gb) / np.sum(gb_weights))
        # Diffuse-core average retained as a diagnostic.  It is exceptionally
        # insensitive to sub-cell translation, but it is not the chemical
        # potential on the GB centre plane and changes the qualified slow
        # clock by about 34 percent.  It therefore must not feed transport.
        gb_core_weight = (
            np.asarray(f, dtype=float) * 4.0 * phi * (1.0 - phi)
            * gb_weights[None, :])
        gb_core_denominator = float(np.sum(gb_core_weight))
        if gb_core_denominator <= 0.0:
            raise ValueError("diffuse ownership-GB sampling support is empty")
        mu_gb_core = float(np.sum(gb_core_weight * mu) / gb_core_denominator)
        # Retain the former grid-identity-changing center mask as a diagnostic
        # only.  It no longer feeds affinity, transport, or physical time.
        legacy_support = r_c <= r_tj - 3.0 * width
        if np.count_nonzero(legacy_support) < 3:
            legacy_support = r_c <= r_tj - width
        legacy_weights = r_c[legacy_support]
        mu_gb_legacy = float(
            np.sum(legacy_weights * mu_on_gb[legacy_support])
            / np.sum(legacy_weights))

        branches = extract_axisymmetric_surface_branches(
            f, mu, r_c, z, z_tj=z_tj, r_tj=r_tj)
        side_values = {}
        for branch in branches:
            support = ((branch.s_centers_m >= width)
                       & (branch.s_centers_m <= 3.0 * width))
            if np.count_nonzero(support) < 2:
                support = branch.s_centers_m <= 3.0 * width
            if not np.any(support):
                raise ValueError("corrected local-TJ support is under-resolved")
            side_values[branch.side] = float(np.mean(branch.mu_Pa[support]))
        mu_tj_local = 0.5 * (
            side_values["negative"] + side_values["positive"])
        return dict(
            mu_field_Pa=mu,
            mu_bulk_Pa=mu_bulk,
            # Production definition.  The axial plane follows the continuous
            # field TJ and the radial partial-cell weights remove the former
            # cell-identity jump at rTJ-3W.
            mu_GB_source_Pa=mu_gb_plane_continuous,
            mu_GB_Pa=mu_gb_plane_continuous,
            mu_GB_source_plane_continuous_Pa=mu_gb_plane_continuous,
            mu_GB_source_diffuse_core_diagnostic_Pa=mu_gb_core,
            mu_GB_source_legacy_discrete_Pa=mu_gb_legacy,
            mu_GB_radial_cutoff_m=gb_cutoff,
            mu_TJ_local_Pa=mu_tj_local,
            mu_TJ_local_negative_Pa=side_values["negative"],
            mu_TJ_local_positive_Pa=side_values["positive"],
            delta_mu_GB_minus_TJ_local_Pa=(
                mu_gb_plane_continuous - mu_tj_local),
            contact_area_m2=math.pi * r_tj * r_tj,
            z_TJ_m=z_tj,
            r_TJ_m=r_tj,
            tj_tracking=tracking,
            chemical_potential_formulation=(
                "delta_G_corrected_normalized_ownership/delta_f_at_fixed_phi"),
            gb_sampling_definition=(
                "field-TJ GB center plane with continuous partial-cell radial "
                "area, 0<=r<=rTJ-3W"),
            tj_local_sampling_definition=(
                "mean of side-resolved surface mu over 1W<=s<=3W"),
            equilibrium_angle_correction_applied=False)

    return evaluate
