"""Plateau-Rayleigh sintering-stress metrology.

The local capillary projection and the distinct circular-contact line force
are kept as separate terms.  Reduced stresses have units 1/m; multiplying by
``gamma_s`` (J/m^2) gives Pa.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull


EPS1 = 0.4
EPS2 = 0.4
R0_OVER_RCYL = math.sqrt(1.0 - 0.5 * (EPS1 ** 2 + EPS2 ** 2))
LAMBDA_OVER_RCYL = 1.14 * math.pi
Z1_OVER_LAMBDA = math.acos(-EPS2 / (4.0 * EPS1)) / math.pi


def circular_contact_terms(r_neck, psi_rad):
    """Second principal curvature and distinct 3-D line-force term."""
    projection = math.sin(0.5 * psi_rad) / r_neck
    return dict(kappa2_neck=projection, s_line_3D=2.0 * projection)


def local_3d_contact_stress(kappa_meridional_1, kappa_meridional_2,
                            r_neck, psi_rad, gamma_s):
    """Complete two-endpoint local 3-D estimator with signed curvature."""
    contact = circular_contact_terms(r_neck, psi_rad)
    s1 = -kappa_meridional_1 + contact["kappa2_neck"]
    s2 = -kappa_meridional_2 + contact["kappa2_neck"]
    s_cap = 0.5 * (s1 + s2)
    s_total = s_cap + contact["s_line_3D"]
    return dict(s_endpoint_1=s1, s_endpoint_2=s2, s_local_cap=s_cap,
                **contact, s_3D_local=s_total,
                sigma_3D_local_Pa=gamma_s * s_total,
                sigma_3D_local_MPa=gamma_s * s_total / 1e6)


def analytic_m16g_reference(R_cyl=100e-9, psi_deg=160.0, gamma_s=1.0):
    """Exact sharp-profile reference at the retained M16G trough."""
    lam = LAMBDA_OVER_RCYL * R_cyl
    z1 = Z1_OVER_LAMBDA * lam
    w1, w2 = 2.0 * math.pi / lam, math.pi / lam
    R = R_cyl * (R0_OVER_RCYL + EPS1 * math.cos(w1 * z1) + EPS2 * math.cos(w2 * z1))
    Rp = R_cyl * (-EPS1 * w1 * math.sin(w1 * z1) - EPS2 * w2 * math.sin(w2 * z1))
    Rpp = R_cyl * (-EPS1 * w1 ** 2 * math.cos(w1 * z1) - EPS2 * w2 ** 2 * math.cos(w2 * z1))
    kappa_m = -Rpp / (1.0 + Rp * Rp) ** 1.5
    local = local_3d_contact_stress(kappa_m, kappa_m, R, math.radians(psi_deg), gamma_s)
    return dict(R_cyl=R_cyl, lambda_over_Rcyl=LAMBDA_OVER_RCYL,
                z1_over_lambda=Z1_OVER_LAMBDA, lam=lam, z1=z1,
                r_neck=R, X_neck=2.0 * R, Rprime_at_z1=Rp,
                Rpp_at_z1_per_m=Rpp, kappa_meridional_per_m=kappa_m,
                gamma_s=gamma_s, psi_deg=psi_deg, **local)


def polyline_arc_length(points):
    points = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def integral_turning(points, closed=False):
    """Signed tangent turning and arc length using first differences only."""
    pts = np.asarray(points, dtype=float)
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    d = np.diff(pts, axis=0)
    keep = np.linalg.norm(d, axis=1) > 0
    d = d[keep]
    angles = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    if closed:
        angles = np.unwrap(np.r_[angles, angles[0] + 2.0 * math.pi])
    turning = float(angles[-1] - angles[0]) if len(angles) > 1 else 0.0
    return dict(turning_angle=turning, arc_length=polyline_arc_length(pts),
                k_integral=(turning / polyline_arc_length(pts) if polyline_arc_length(pts) else float("nan")))


def mirror_axisymmetric_particle_branch(r, z, z_contact):
    """Reflect an r>=0 particle free-surface branch and close on the GB."""
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    order = np.argsort(z)
    r, z = r[order], z[order]
    positive = np.column_stack([r, z])
    negative = np.column_stack([-r[::-1], z[::-1]])
    contour = np.vstack([positive, negative])
    # The implicit final segment closes the two contact endpoints at z_contact.
    if not np.isclose(contour[0, 1], z_contact) or not np.isclose(contour[-1, 1], z_contact):
        raise ValueError("branch endpoints must lie on the contact plane")
    return contour


def convex_mean_width(points):
    pts = np.asarray(points, dtype=float)
    hull = ConvexHull(pts)
    vertices = pts[hull.vertices]
    perimeter = polyline_arc_length(np.vstack([vertices, vertices[0]]))
    mean_width = perimeter / math.pi
    return dict(convex_hull_perimeter=perimeter, mean_width_2D=mean_width,
                k_mean_width=2.0 / mean_width)


def normal_support(points, direction=(0.0, 1.0)):
    pts = np.asarray(points, dtype=float)
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    projection = pts @ unit
    width = float(np.max(projection) - np.min(projection))
    return dict(normal_support_width=width, k_normal_support=2.0 / width)


def add_contact_and_line_terms(k_meridional, r_neck, psi_rad, gamma_s, label):
    contact = circular_contact_terms(r_neck, psi_rad)
    s_cap = -k_meridional + contact["kappa2_neck"]
    s_total = s_cap + contact["s_line_3D"]
    return {f"k_{label}": k_meridional, f"s_cap_{label}": s_cap,
            f"s_3D_{label}": s_total, f"sigma_3D_{label}_MPa": gamma_s * s_total / 1e6,
            **contact}


def _quadratic_signed_curvature(z_pts, r_pts, z_eval):
    coeff = np.polyfit(z_pts, r_pts, 2)
    rp = float(np.polyval(np.polyder(coeff, 1), z_eval))
    rpp = float(np.polyval(np.polyder(coeff, 2), z_eval))
    return -rpp / (1.0 + rp * rp) ** 1.5, rp


def _branch_patch_average_curvature(R, z, neck_index, sign, patch_length_m,
                                    tangent_span_m):
    """Signed mean meridional curvature from finite-patch tangent turning."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    indices = (np.arange(neck_index, len(z)) if sign > 0
               else np.arange(neck_index, -1, -1))
    indices = indices[np.isfinite(R[indices])]
    if len(indices) < 4:
        raise ValueError("insufficient branch points")
    zz, rr = z[indices], R[indices]
    ds = np.hypot(np.diff(zz), np.diff(rr))
    s = np.r_[0.0, np.cumsum(ds)]
    if s[-1] < patch_length_m:
        raise ValueError("branch is shorter than requested physical patch")
    endpoint = int(np.searchsorted(s, patch_length_m))

    def tangent_slope(center_s, one_sided_at_neck=False):
        if one_sided_at_neck:
            mask = s <= tangent_span_m
        else:
            mask = np.abs(s - center_s) <= 0.5 * tangent_span_m
            if np.sum(mask) < 3:
                mask = np.abs(s - center_s) <= tangent_span_m
        if np.sum(mask) < 3:
            raise ValueError("insufficient points for patch tangent")
        return float(np.polyfit(zz[mask], rr[mask], 1)[0])

    slope_neck = tangent_slope(0.0, one_sided_at_neck=True)
    slope_end = tangent_slope(patch_length_m)
    theta_neck = math.atan(slope_neck)
    theta_end = math.atan(slope_end)
    turning = math.atan2(math.sin(theta_end - theta_neck),
                         math.cos(theta_end - theta_neck))
    kappa_average = -sign * turning / patch_length_m
    return dict(kappa_average=kappa_average, turning_angle=turning,
                patch_length_m=patch_length_m, tangent_span_m=tangent_span_m,
                slope_neck=slope_neck, slope_end=slope_end,
                n_points=endpoint + 1)


def branch_patch_3d_stress(R_of_z, z, z_gb, patch_length_m=20e-9,
                           tangent_span_m=5e-9, psi_reference_deg=160.0,
                           gamma_s=1.0):
    """Branch-local stress using fixed-physical-patch tangent turning.

    Unlike a point-curvature polynomial, this observable is the signed tangent
    rotation over a predeclared physical arclength on each side of the TJ.  It
    is intended for spatial-resolution tests, not as an automatically selected
    replacement activation coordinate.
    """
    R = np.asarray(R_of_z, dtype=float)
    z = np.asarray(z, dtype=float)
    j0 = int(np.nanargmin(np.where(
        np.isfinite(R) & (np.abs(z - z_gb) <= patch_length_m), R, np.nan)))
    minus = _branch_patch_average_curvature(
        R, z, j0, -1, patch_length_m, tangent_span_m)
    plus = _branch_patch_average_curvature(
        R, z, j0, +1, patch_length_m, tangent_span_m)
    stress = local_3d_contact_stress(
        minus["kappa_average"], plus["kappa_average"], float(R[j0]),
        math.radians(psi_reference_deg), gamma_s)
    return dict(z_neck=float(z[j0]), r_neck=float(R[j0]),
                kappa_patch_side1=minus["kappa_average"],
                kappa_patch_side2=plus["kappa_average"],
                side1=minus, side2=plus, **stress)


def pf_contour_estimators(R_of_z, z, z_gb, gamma_s=1.0, psi_reference_deg=160.0,
                          local_window_m=15e-9, profile_length_m=100e-9):
    """Four geometric estimators from an axisymmetric PF ``f=0.5`` contour.

    The local fit uses the smooth contour separately on either side of the
    contact.  Global particle estimators use only the particle-side branch,
    reflect it through r=0, and close it with the contact segment.
    """
    R = np.asarray(R_of_z, dtype=float)
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(R)
    if np.sum(finite) < 8:
        raise ValueError("insufficient contour points")
    j0 = int(np.nanargmin(np.where(np.abs(z - z_gb) <= local_window_m, R, np.nan)))
    r_neck = float(R[j0])
    z_neck = float(z[j0])

    def side_fit(sign):
        mask = finite & (sign * (z - z_neck) >= 0) & (np.abs(z - z_neck) <= local_window_m)
        idx = np.where(mask)[0]
        if len(idx) < 5:
            raise ValueError("insufficient local branch points")
        return _quadratic_signed_curvature(z[idx], R[idx], z_neck), len(idx)

    (k_plus, rp_plus), n_plus = side_fit(+1)
    (k_minus, rp_minus), n_minus = side_fit(-1)
    # Each free-surface/GB angle is atan2(axial tangent, radial tangent);
    # twice its acute magnitude gives the dihedral diagnostic.
    psi_measured = 2.0 * math.atan2(1.0, 0.5 * (abs(rp_plus) + abs(rp_minus)))
    local_measured = local_3d_contact_stress(k_plus, k_minus, r_neck, psi_measured, gamma_s)
    local_reference = local_3d_contact_stress(
        k_plus, k_minus, r_neck, math.radians(psi_reference_deg), gamma_s)

    particle = finite & (z >= z_neck)
    zp, rp = z[particle], R[particle]
    order = np.argsort(zp)
    zp, rp = zp[order], rp[order]
    free_arc = np.vstack([np.column_stack([rp, zp]), np.column_stack([-rp[::-1], zp[::-1]])])
    full_contour = mirror_axisymmetric_particle_branch(rp, zp, z_neck)
    turning = integral_turning(free_arc, closed=False)
    mw = convex_mean_width(full_contour)
    support = normal_support(full_contour, (0.0, 1.0))
    reference_rad = math.radians(psi_reference_deg)
    global_integral = add_contact_and_line_terms(
        turning["k_integral"], r_neck, reference_rad, gamma_s, "integral")
    global_mw = add_contact_and_line_terms(
        mw["k_mean_width"], r_neck, reference_rad, gamma_s, "mean_width")
    global_support = add_contact_and_line_terms(
        support["k_normal_support"], r_neck, reference_rad, gamma_s, "normal_support")

    branch = finite & (z >= z_neck) & (z <= z_neck + profile_length_m)
    zb, rb = z[branch], R[branch]
    if len(zb) >= 5:
        dr_dz = np.gradient(rb, zb)
        d2r = np.gradient(dr_dz, zb)
        k_profile = -d2r / np.maximum((1.0 + dr_dz ** 2) ** 1.5, 1e-300)
        ds = np.hypot(np.diff(zb), np.diff(rb))
        s_profile = np.r_[0.0, np.cumsum(ds)]
    else:
        s_profile = np.array([])
        k_profile = np.array([])
    out = dict(z_GB=z_neck, r_neck=r_neck, X_neck=2 * r_neck,
               contact_area=math.pi * r_neck ** 2,
               psi_measured_deg=math.degrees(psi_measured),
               psi_reference_deg=psi_reference_deg,
               kappa_local_side1=k_minus, kappa_local_side2=k_plus,
               local_fit_window_m=local_window_m,
               local_points_side1=n_minus, local_points_side2=n_plus,
               local_measured=local_measured, local_reference=local_reference,
               turning_angle_free_surface=turning["turning_angle"],
               free_surface_arc_length=turning["arc_length"],
               curvature_s_m=s_profile, curvature_per_m=k_profile)
    out.update(global_integral)
    out.update(mw)
    out.update(global_mw)
    out.update(support)
    out.update(global_support)
    return out
