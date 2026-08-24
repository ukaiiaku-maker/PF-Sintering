"""Candidate activation coordinates for the scientific 100 nm PR topology.

These functions deliberately contain no mobility, diffusivity, timestep,
hazard, or event-packet inputs.  They evaluate state coordinates only.  In
particular, the localized embryo chemical-potential residual below is distinct
from the broader GB-to-TJ affinity used by post-nucleation transport.
"""
from __future__ import annotations

import math

import numpy as np

from pf_sintering.axisym import axisym_mu_f_gb


def local_capillary_residual(kappa_meridional_1, kappa_meridional_2,
                             r_neck, psi_rad, gamma_s=1.0):
    """Candidate A: local capillary residual without the line-force term.

    The meridional curvature is positive for a convex spherical cap.  The
    result is ``gamma_s*(-mean(kappa_m) + sin(psi/2)/r_neck)`` in Pa.
    """
    if r_neck <= 0.0:
        raise ValueError("r_neck must be positive")
    kappa_mean = 0.5 * (float(kappa_meridional_1)
                        + float(kappa_meridional_2))
    projection = math.sin(0.5 * psi_rad) / float(r_neck)
    return float(gamma_s * (-kappa_mean + projection))


def winterbottom_spherical_cap_reference(R, psi_rad, gamma_s=1.0,
                                          gamma_gb=None):
    """Exact isotropic flat-contact spherical-cap equilibrium benchmark.

    The Young/Winterbottom relation is
    ``gamma_gb/(2*gamma_s) = cos(psi/2)``.  At the spherical-cap contact,
    ``kappa_m=1/R`` and ``r_neck=R*sin(psi/2)``.  Candidate A and the
    equilibrium chemical-potential residual are therefore exactly zero,
    whereas the legacy Hussein quantity and complete 3-D contact traction are
    non-zero absolute tractions.
    """
    if R <= 0.0 or gamma_s <= 0.0:
        raise ValueError("R and gamma_s must be positive")
    young_ratio = math.cos(0.5 * psi_rad)
    gamma_gb_equilibrium = 2.0 * gamma_s * young_ratio
    if gamma_gb is None:
        gamma_gb = gamma_gb_equilibrium
    young_residual = gamma_gb / (2.0 * gamma_s) - young_ratio
    r_neck = R * math.sin(0.5 * psi_rad)
    kappa_m = 1.0 / R
    x_a = local_capillary_residual(
        kappa_m, kappa_m, r_neck, psi_rad, gamma_s)
    c_gb = math.sqrt(max(0.0, 1.0 - (gamma_gb / (2.0 * gamma_s)) ** 2))
    x_neck = 2.0 * r_neck
    sigma_h = gamma_s * (1.0 / R - c_gb / x_neck)
    line_traction = 2.0 * gamma_s * math.sin(0.5 * psi_rad) / r_neck
    return dict(
        topology="isotropic flat-contact spherical-cap/Winterbottom",
        R=R, psi_rad=psi_rad, gamma_s=gamma_s, gamma_gb=gamma_gb,
        gamma_gb_equilibrium=gamma_gb_equilibrium,
        young_relation_residual=young_residual,
        r_neck=r_neck, X_neck=x_neck, kappa_meridional=kappa_m,
        X_A_local_capillary_residual_Pa=x_a,
        X_B_equilibrium_mu_residual_Pa=0.0,
        F_s_constrained_equilibrium=0.0,
        sigma_H_legacy_absolute_Pa=sigma_h,
        Sigma_contact_3D_absolute_Pa=x_a + line_traction,
        candidate_C_status=(
            "undefined without a physically specified disconnection embryo; "
            "an arbitrary deposition path is not an equilibrium coordinate"))


def _normalized_axisym_support(raw, r_c):
    raw = np.maximum(np.asarray(raw, dtype=float), 0.0)
    norm = float(np.sum(np.asarray(r_c)[None, :] * raw))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("localized activation support has zero weight")
    return raw / norm


def localized_embryo_mu_residual(
        f, particle, neighbor, pf_params, Wc, dr, dz, r_c, r_f, z,
        z_gb, r_neck, W, source_width_factor, sink_width_factor):
    """Candidate B: a predeclared TJ-embryo chemical-potential residual.

    The source is the diffuse GB overlap localized radially and axially about
    the TJ.  The sink is the free-surface indicator localized about the same
    point.  Their independently declared Gaussian widths are multiples of
    ``W``.  This is intentionally *not* the broad contact-wide GB source used
    by the post-nucleation transport affinity.

    ``axisym_mu_f_gb`` has units J/m^3 = Pa, so the returned difference is
    directly the stress-equivalent ``X_B``.  Multiplication by ``Omega`` would
    give the corresponding per-atom energy.
    """
    if source_width_factor <= 0.0 or sink_width_factor <= 0.0:
        raise ValueError("localization width factors must be positive")
    mu = axisym_mu_f_gb(
        f, particle, neighbor, pf_params, Wc, dr, dz, r_c, r_f,
        bc_z="noflux")
    Z = np.asarray(z, dtype=float)[:, None]
    RC = np.asarray(r_c, dtype=float)[None, :]
    d2 = (Z - float(z_gb)) ** 2 + (RC - float(r_neck)) ** 2
    source_sigma = source_width_factor * W
    sink_sigma = sink_width_factor * W
    source_raw = (np.maximum(np.asarray(particle) * np.asarray(neighbor), 0.0)
                  * np.exp(-0.5 * d2 / source_sigma ** 2))
    fb = np.clip(np.asarray(f), 0.0, 1.0)
    surface = 16.0 * fb ** 2 * (1.0 - fb) ** 2
    sink_raw = surface * np.exp(-0.5 * d2 / sink_sigma ** 2)
    source_support = _normalized_axisym_support(source_raw, r_c)
    sink_support = _normalized_axisym_support(sink_raw, r_c)
    weights = np.asarray(r_c)[None, :]
    mu_source = float(np.sum(weights * source_support * mu))
    mu_sink = float(np.sum(weights * sink_support * mu))
    return dict(
        source_width_W=float(source_width_factor),
        sink_width_W=float(sink_width_factor),
        mu_embryo_source_Pa=mu_source,
        mu_embryo_sink_Pa=mu_sink,
        X_B_embryo_mu_residual_Pa=mu_source - mu_sink,
        source_support_weighted_norm=float(np.sum(weights * source_support)),
        sink_support_weighted_norm=float(np.sum(weights * sink_support)),
        definition=(
            "local TJ embryo mu_GB - mu_surface in Pa; distinct from "
            "contact-wide post-nucleation transport affinity"))


def localized_embryo_mu_matrix(
        f, particle, neighbor, pf_params, Wc, dr, dz, r_c, r_f, z,
        z_gb, r_neck, W, source_widths=(1.0, 2.0, 3.0),
        sink_widths=(0.5, 1.0, 1.5, 2.0)):
    """Evaluate the predeclared Candidate-B localization matrix."""
    out = {}
    for source_width in source_widths:
        for sink_width in sink_widths:
            key = f"source_{source_width:g}W_sink_{sink_width:g}W"
            out[key] = localized_embryo_mu_residual(
                f, particle, neighbor, pf_params, Wc, dr, dz, r_c, r_f,
                z, z_gb, r_neck, W, source_width, sink_width)
    return out
