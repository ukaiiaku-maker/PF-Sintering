"""Equilibrium-referenced whole-contact observables for scientific PR states.

The GB-contact support is the complete diffuse contact, ``eta1*eta2``, under
the axisymmetric measure.  It has no Gaussian envelope, radial cutoff, embryo
radius, or fit width.  Chemical-potential values from the evolving state are
interpreted only relative to a grid-matched diffuse Winterbottom reference
evaluated with the identical functional.
"""

from __future__ import annotations

import math

import numpy as np

from .axisym import axisym_mu_f_gb


def spherical_major_cap_volume(radius: float, psi_rad: float) -> float:
    """Volume of one spherical grain cap meeting the GB at angle ``psi/2``.

    The branch tangent directed away from the contact is
    ``(cos(psi/2), sin(psi/2))`` in ``(r,z)``.  The corresponding cap height
    is ``radius*(1+cos(psi/2))``.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    c = math.cos(0.5 * psi_rad)
    height = radius * (1.0 + c)
    return math.pi * height * height * (radius - height / 3.0)


def sharp_radius_for_particle_volume(volume: float, psi_rad: float) -> float:
    """Closed-form sharp-interface radius for a prescribed cap volume."""
    if volume <= 0.0:
        raise ValueError("volume must be positive")
    unit_volume = spherical_major_cap_volume(1.0, psi_rad)
    return (volume / unit_volume) ** (1.0 / 3.0)


def _axisym_integral(field: np.ndarray, r_c: np.ndarray, dr: float, dz: float) -> float:
    return 2.0 * math.pi * float(np.sum(np.asarray(r_c)[None, :] * field)) * dr * dz


def _two_cap_fields(
    radius: float,
    psi_rad: float,
    W: float,
    z: np.ndarray,
    r_c: np.ndarray,
    z_contact: float,
    ownership_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diffuse symmetric two-cap Winterbottom/Young geometry.

    The two equal-curvature caps share one planar GB and give two free-surface
    branches at the contact.  This is the one-GB/two-grain local topology used
    by the PR neck observables.  The contact is placed far from both no-flux
    boundaries; absolute placement has no role in the reference values.
    """
    c = math.cos(0.5 * psi_rad)
    Z, R = np.meshgrid(np.asarray(z), np.asarray(r_c), indexing="ij")
    center_upper = z_contact + radius * c
    center_lower = z_contact - radius * c
    distance_upper = np.sqrt((Z - center_upper) ** 2 + R ** 2) - radius
    distance_lower = np.sqrt((Z - center_lower) ** 2 + R ** 2) - radius
    f_upper = 0.5 * (1.0 - np.tanh(distance_upper / W))
    f_lower = 0.5 * (1.0 - np.tanh(distance_lower / W))
    f = np.where(Z >= z_contact, f_upper, f_lower)
    ownership_upper = 0.5 * (1.0 + np.tanh((Z - z_contact) / ownership_width))
    e1 = f * ownership_upper
    e2 = f - e1
    return f, e1, e2


def build_volume_matched_winterbottom(
    target_particle_volume: float,
    psi_rad: float,
    W: float,
    z: np.ndarray,
    r_c: np.ndarray,
    dr: float,
    dz: float,
    z_contact: float,
    *,
    ownership_width: float | None = None,
    relative_volume_tolerance: float = 1e-11,
) -> dict:
    """Build a grid-specific diffuse reference matching particle volume.

    ``radius_sharp_m`` is fixed by the analytical cap-volume constraint.
    ``radius_diffuse_m`` is then adjusted only to remove the finite-W/grid
    volume offset in the identically represented ``eta1`` field.
    """
    if ownership_width is None:
        ownership_width = 1.5 * dz
    radius_sharp = sharp_radius_for_particle_volume(target_particle_volume, psi_rad)

    def make(radius):
        fields = _two_cap_fields(
            radius, psi_rad, W, z, r_c, z_contact, ownership_width)
        return fields, _axisym_integral(fields[1], r_c, dr, dz)

    lower, upper = 0.75 * radius_sharp, 1.25 * radius_sharp
    fields_lower, volume_lower = make(lower)
    fields_upper, volume_upper = make(upper)
    if not volume_lower < target_particle_volume < volume_upper:
        raise ValueError("grid/domain does not bracket the target cap volume")
    fields = fields_lower
    volume = volume_lower
    radius = lower
    for _ in range(80):
        radius = 0.5 * (lower + upper)
        fields, volume = make(radius)
        error = (volume - target_particle_volume) / target_particle_volume
        if abs(error) <= relative_volume_tolerance:
            break
        if volume < target_particle_volume:
            lower = radius
        else:
            upper = radius
    return dict(
        f=fields[0], e1=fields[1], e2=fields[2],
        radius_sharp_m=radius_sharp,
        radius_diffuse_m=radius,
        contact_radius_sharp_m=radius * math.sin(0.5 * psi_rad),
        z_contact_m=float(z_contact),
        particle_volume_target_m3=float(target_particle_volume),
        particle_volume_diffuse_m3=float(volume),
        particle_volume_relative_error=float(
            (volume - target_particle_volume) / target_particle_volume),
        ownership_width_m=float(ownership_width),
        topology="symmetric equal-curvature two-cap local Winterbottom reference",
    )


def whole_contact_chemical_potential(
    f: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    p,
    Wc: float,
    dr: float,
    dz: float,
    r_c: np.ndarray,
    r_f: np.ndarray,
    *,
    bc_z: str = "noflux",
) -> dict:
    """Whole-contact and whole-free-surface means of the same PF ``mu``.

    The factor ``2*pi*dr*dz`` cancels between numerator and denominator,
    but the radial Jacobian is retained explicitly.  The free-surface support
    is the standard diffuse indicator over the entire exterior interface.
    """
    mu = axisym_mu_f_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f, bc_z=bc_z)
    contact_raw = np.maximum(np.asarray(e1) * np.asarray(e2), 0.0)
    fb = np.clip(np.asarray(f), 0.0, 1.0)
    surface_raw = 16.0 * fb * fb * (1.0 - fb) ** 2
    weights = np.asarray(r_c)[None, :]
    contact_norm = float(np.sum(weights * contact_raw))
    surface_norm = float(np.sum(weights * surface_raw))
    if contact_norm <= 0.0 or surface_norm <= 0.0:
        raise ValueError("contact or free-surface support has zero measure")
    mu_contact = float(np.sum(weights * contact_raw * mu) / contact_norm)
    mu_surface = float(np.sum(weights * surface_raw * mu) / surface_norm)
    return dict(
        mu_GB_contact_Pa=mu_contact,
        mu_free_surface_Pa=mu_surface,
        mu_contact_minus_surface_Pa=mu_contact - mu_surface,
        contact_support_axisym_norm=contact_norm,
        free_surface_support_axisym_norm=surface_norm,
        contact_support_definition="eta1*eta2 over the entire resolved GB contact",
        free_surface_support_definition="16*f^2*(1-f)^2 over the entire free surface",
        radial_or_Gaussian_localization=False,
    )


def project_original_pr_modes(
    R_of_z: np.ndarray,
    z: np.ndarray,
    lam: float,
    R_cyl: float,
) -> dict:
    """Project ``R(z)/R_cyl`` onto the fixed original two PR modes."""
    R_of_z = np.asarray(R_of_z, dtype=float)
    z = np.asarray(z, dtype=float)
    keep = np.isfinite(R_of_z) & (z >= 0.0) & (z <= lam)
    if np.count_nonzero(keep) < 6:
        raise ValueError("too few contour samples in the original PR interval")
    phase = z[keep] / lam
    design = np.column_stack([
        np.ones(np.count_nonzero(keep)),
        np.cos(2.0 * math.pi * phase),
        np.cos(math.pi * phase),
    ])
    coefficients, *_ = np.linalg.lstsq(design, R_of_z[keep] / R_cyl, rcond=None)
    fitted = design @ coefficients
    residual = R_of_z[keep] / R_cyl - fitted
    return dict(
        mean_radius_over_Rcyl=float(coefficients[0]),
        mode_2pi_amplitude=float(coefficients[1]),
        mode_pi_amplitude=float(coefficients[2]),
        A_PR=float(math.hypot(coefficients[1], coefficients[2])),
        fit_rms_over_Rcyl=float(np.sqrt(np.mean(residual ** 2))),
        projection_interval="0 <= z <= lambda; fixed original PR basis",
    )
