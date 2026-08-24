"""Experimental particle geometry for the PR-renewal completion campaign.

The source file is a MATLAB v7.3 (HDF5) archive.  This module deliberately
reads its primitive contour objects with :mod:`h5py`; the experimental
particle is never reconstructed from a screenshot or from an already-combined
stress column.

The measured contact chord is oblique to the particle principal axis.  Its
half *length* therefore fixes the circular axisymmetric contact radius, while
the particle radial profile is obtained independently from transverse chord
widths in area-principal-axis coordinates.  This is the key geometric
distinction requested for the experiment-derived surrogate.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from .axisym import axisym_volume, r_centers_faces
from .pr_stress_metrology import convex_mean_width, integral_turning


def _matlab_array(dataset):
    """Return a numeric MATLAB array from its v7.3 HDF5 storage order."""
    value = np.asarray(dataset)
    return value.T.copy() if value.ndim == 2 else value.copy()


def _deref(file, reference):
    return file[reference]


def polygon_area_principal_geometry(points):
    """Area centroid and principal axes of a closed simple polygon.

    The covariance is the exact uniform-area second moment, not a covariance
    of the nonuniformly sampled boundary points.
    """
    points = np.asarray(points, dtype=float)
    if np.linalg.norm(points[0] - points[-1]) <= 1e-12:
        points = points[:-1]
    closed = np.vstack([points, points[0]])
    x, y = closed[:-1, 0], closed[:-1, 1]
    xn, yn = closed[1:, 0], closed[1:, 1]
    cross = x * yn - xn * y
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) <= 1e-30:
        raise ValueError("experimental particle polygon has zero area")
    centroid = np.array([
        np.sum((x + xn) * cross) / (6.0 * signed_area),
        np.sum((y + yn) * cross) / (6.0 * signed_area),
    ])
    ex2 = np.sum((x*x + x*xn + xn*xn) * cross) / (12.0*signed_area)
    ey2 = np.sum((y*y + y*yn + yn*yn) * cross) / (12.0*signed_area)
    exy = np.sum(
        (2*x*y + x*yn + xn*y + 2*xn*yn) * cross
    ) / (24.0*signed_area)
    covariance = np.array([
        [ex2 - centroid[0]**2, exy - centroid[0]*centroid[1]],
        [exy - centroid[0]*centroid[1], ey2 - centroid[1]**2],
    ])
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    return dict(
        area_px2=abs(signed_area), centroid_px=centroid,
        covariance_px2=covariance, eigenvalues_px2=values,
        major_axis=vectors[:, 0], minor_axis=vectors[:, 1])


def load_initial_experimental_geometry(path):
    """Read snapshot zero and its analysis metadata from MATLAB HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as file:
        pixel = float(file["cfg/pixelSize_m"][0, 0])
        particle = _matlab_array(_deref(
            file, file["snapshots/particleXY_px"][0, 0]))
        neck = _matlab_array(_deref(
            file, file["snapshots/neckXY_px"][0, 0]))
        substrate_cell = _deref(
            file, file["snapshots/substrateSegments_px"][0, 0])
        substrate = [
            _matlab_array(_deref(file, reference))
            for reference in np.asarray(substrate_cell).ravel(order="F")
        ]
        stored_centroid = np.asarray(
            file["cfg/init/particleCentroid"]).ravel(order="F")
        metadata = dict(
            angle_fit_gap_px=float(file["cfg/angle/fitGapPx"][0, 0]),
            angle_fit_span_px=float(file["cfg/angle/fitSpanPx"][0, 0]),
            angle_poly_order=int(file["cfg/angle/polyOrder"][0, 0]),
            curvature_fit_gap_px=float(
                file["cfg/curvature/fitGapPx"][0, 0]),
            curvature_fit_spans_px=np.asarray(
                file["cfg/curvature/fitSpansPx"]).ravel(order="F").tolist(),
            curvature_peak_half_widths_px=np.asarray(
                file["cfg/curvature/peakHalfWidthsPx"]
            ).ravel(order="F").tolist(),
            curvature_poly_order=int(
                file["cfg/curvature/polyOrder"][0, 0]),
            segmentation_resample_spacing_px=float(
                file["cfg/seg/resampleSpacingPx"][0, 0]),
            turning_resample_px=float(
                file["cfg/global/turningResamplePx"][0, 0]),
            turning_smooth_span_px=float(
                file["cfg/global/turningSmoothSpanPx"][0, 0]),
            n_width_angles=int(file["cfg/global/nWidthAngles"][0, 0]),
            gamma_J_per_m2=float(file["cfg/physics/gamma_J_m2"][0, 0]),
            source_matlab_created=str(file.attrs.get("MATLAB_fields", "")),
        )
    if np.linalg.norm(particle[0] - particle[-1]) > 1e-8:
        raise ValueError("snapshot-zero particle contour is not closed")
    area = polygon_area_principal_geometry(particle)
    major = area["major_axis"].copy()
    contact_midpoint = np.mean(neck, axis=0)
    if np.dot(major, area["centroid_px"] - contact_midpoint) < 0.0:
        major *= -1.0
    minor = np.array([-major[1], major[0]])
    uv = np.column_stack([
        (particle - contact_midpoint) @ major,
        (particle - contact_midpoint) @ minor,
    ])
    neck_uv = np.column_stack([
        (neck - contact_midpoint) @ major,
        (neck - contact_midpoint) @ minor,
    ])
    neck_length_px = float(np.linalg.norm(neck[1] - neck[0]))
    return dict(
        path=str(path.resolve()), pixel_size_m=pixel,
        particle_xy_px=particle, neck_xy_px=neck,
        substrate_segments_px=substrate, stored_centroid_px=stored_centroid,
        area_geometry=area, contact_midpoint_px=contact_midpoint,
        major_axis=major, minor_axis=minor, particle_uv_px=uv,
        neck_uv_px=neck_uv, neck_chord_px=neck_length_px,
        neck_chord_m=neck_length_px*pixel,
        neck_radius_m=0.5*neck_length_px*pixel, metadata=metadata)


def _monotone_branch(branch_uv):
    """Return a unique axial parameterization of one measured side."""
    order = np.argsort(branch_uv[:, 0])
    u = branch_uv[order, 0]
    v = branch_uv[order, 1]
    unique, inverse = np.unique(np.round(u, 10), return_inverse=True)
    averaged = np.zeros_like(unique)
    count = np.zeros_like(unique)
    np.add.at(averaged, inverse, v)
    np.add.at(count, inverse, 1.0)
    return unique, averaged/count


def measured_particle_profile(record, sample_spacing_px=0.20):
    """Symmetrize transverse chord widths in principal-axis coordinates."""
    uv = np.asarray(record["particle_uv_px"][:-1])
    apex = int(np.argmax(uv[:, 0]))
    first = uv[:apex+1]
    second = uv[apex:]
    u1, v1 = _monotone_branch(first)
    u2, v2 = _monotone_branch(second)
    common_lo = max(float(np.min(u1)), float(np.min(u2)))
    common_hi = min(float(np.max(u1)), float(np.max(u2)))
    u = np.arange(common_lo, common_hi, sample_spacing_px)
    if u[-1] < common_hi:
        u = np.r_[u, common_hi]
    side1 = PchipInterpolator(u1, v1)(u)
    side2 = PchipInterpolator(u2, v2)(u)
    radius = 0.5*np.abs(side2 - side1)
    center_offset = 0.5*(side2 + side1)
    return dict(
        u_px=u, radius_px=radius, center_offset_px=center_offset,
        common_u_min_px=common_lo, common_u_max_px=common_hi,
        branch1_u_px=u1, branch1_v_px=v1,
        branch2_u_px=u2, branch2_v_px=v2)


def _endpoint_tangent_slope(uv, at_start, span_px):
    # Select by contour arclength, not Euclidean distance.  The two measured
    # neck endpoints are only about ten pixels apart, so a Euclidean ball at
    # one endpoint otherwise includes the *opposite* particle branch and
    # corrupts the tangent fit.
    ordered = uv if at_start else uv[::-1]
    arclength = np.r_[0.0, np.cumsum(np.linalg.norm(
        np.diff(ordered, axis=0), axis=1))]
    points = ordered[arclength <= span_px]
    if len(points) < 5:
        points = ordered[:5]
    values, vectors = np.linalg.eigh(np.cov(points.T))
    tangent = vectors[:, np.argmax(values)]
    return float(abs(tangent[1]/tangent[0]))


def contact_tangent_metadata(record):
    """Measured particle/substrate tangent magnitudes at both neck ends."""
    uv = np.asarray(record["particle_uv_px"][:-1])
    span = float(record["metadata"]["angle_fit_span_px"])
    particle_slopes = [
        _endpoint_tangent_slope(uv, True, span),
        _endpoint_tangent_slope(uv, False, span),
    ]
    substrate_slopes = []
    for index, segment in enumerate(record["substrate_segments_px"]):
        transformed = np.column_stack([
            (segment - record["contact_midpoint_px"]) @ record["major_axis"],
            (segment - record["contact_midpoint_px"]) @ record["minor_axis"],
        ])
        endpoint_is_start = index == 1
        substrate_slopes.append(
            _endpoint_tangent_slope(transformed, endpoint_is_start, span))
    return dict(
        particle_slope_side1=particle_slopes[0],
        particle_slope_side2=particle_slopes[1],
        particle_slope_average=float(np.mean(particle_slopes)),
        substrate_slope_side1=substrate_slopes[0],
        substrate_slope_side2=substrate_slopes[1],
        substrate_slope_average=float(np.mean(substrate_slopes)),
        particle_angle_axis_side1_deg=math.degrees(math.atan(particle_slopes[0])),
        particle_angle_axis_side2_deg=math.degrees(math.atan(particle_slopes[1])),
        substrate_angle_axis_side1_deg=math.degrees(math.atan(substrate_slopes[0])),
        substrate_angle_axis_side2_deg=math.degrees(math.atan(substrate_slopes[1])),
        fit_span_px=span)


def _cubic_hermite(x, x0, x1, y0, y1, m0, m1):
    scale = x1 - x0
    t = np.clip((x - x0)/scale, 0.0, 1.0)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00*y0 + h10*scale*m0 + h01*y1 + h11*scale*m1


def surrogate_profile(record, scale_factor=1.0):
    """Build the sharp axisymmetric particle and cropped substrate profile."""
    pixel = record["pixel_size_m"]*scale_factor
    measured = measured_particle_profile(record)
    tangents = contact_tangent_metadata(record)
    u_raw = measured["u_px"]
    radius_raw = measured["radius_px"]
    # The contact chord is oblique.  Start the revolution curve at the first
    # axial plane for which both measured sides exist, retain the chord half-
    # length (not its transverse projection) as r_n, and bridge only to the
    # first well-resolved two-sided width using the measured tangent.
    contact_u = float(u_raw[0])
    join_u = max(10.0, float(u_raw[0]) + 4.0)
    join_index = int(np.argmin(np.abs(u_raw - join_u)))
    join_u = float(u_raw[join_index])
    derivative = np.gradient(radius_raw, u_raw)
    join_radius = float(radius_raw[join_index])
    join_slope = float(derivative[join_index])
    contact_radius_px = 0.5*record["neck_chord_px"]
    u_particle_raw = np.r_[
        np.linspace(contact_u, join_u,
                    max(20, int(round((join_u-contact_u)/0.2))),
                    endpoint=False),
        u_raw[join_index:],
    ]
    r_bridge = _cubic_hermite(
        u_particle_raw[u_particle_raw < join_u], contact_u, join_u,
        contact_radius_px, join_radius,
        tangents["particle_slope_average"], join_slope)
    r_particle = np.r_[r_bridge, radius_raw[join_index:]]
    z_particle_px = u_particle_raw-contact_u
    # Preserve the measured full principal-axis support.  Obliquity removes
    # about one contact-chord projection from the common two-sided profile;
    # restore that length only in the terminal cap with a C1 smooth stretch,
    # leaving the neck and body profile untouched.
    measured_length_px = float(np.ptp(record["particle_uv_px"][:, 0]))
    missing_length_px = measured_length_px-z_particle_px[-1]
    cap_start = max(0.80*z_particle_px[-1], z_particle_px[-1]-25.0)
    cap_coordinate = np.clip(
        (z_particle_px-cap_start)/(z_particle_px[-1]-cap_start), 0.0, 1.0)
    smoothstep = cap_coordinate*cap_coordinate*(3.0-2.0*cap_coordinate)
    z_particle_px = z_particle_px+missing_length_px*smoothstep
    # Enforce a clean single-valued tip while changing only the final
    # subpixel-scale profile segment.
    r_particle[-1] = 0.0
    z_particle_m = z_particle_px*pixel
    r_particle_m = np.maximum(r_particle*pixel, 0.0)

    rn = 0.5*record["neck_chord_px"]*pixel
    particle_max = float(np.max(r_particle_m))
    substrate_radius = max(particle_max, rn + 8.0e-8*scale_factor)
    substrate_length = 2.40e-7*scale_factor
    z_substrate_m = np.linspace(-substrate_length, 0.0, 321)
    # Cropped compact substrate: zero slope at the no-flux cut and the
    # measured local contact slope at the TJ.
    r_substrate_m = _cubic_hermite(
        z_substrate_m, -substrate_length, 0.0,
        substrate_radius, rn, 0.0,
        -tangents["substrate_slope_average"])
    if np.any(np.diff(r_substrate_m) > 1e-15):
        raise ValueError("cropped substrate profile is not monotone to contact")
    return dict(
        z_particle_m=z_particle_m, r_particle_m=r_particle_m,
        z_substrate_m=z_substrate_m, r_substrate_m=r_substrate_m,
        r_neck_m=rn, particle_axial_length_m=float(z_particle_m[-1]),
        particle_max_radius_m=particle_max,
        substrate_length_m=substrate_length,
        substrate_cut_radius_m=substrate_radius,
        scale_factor=float(scale_factor), measured_profile=measured,
        tangents=tangents, contact_u_px=contact_u,
        join_u_px=join_u, join_z_px=join_u-contact_u,
        terminal_cap_extension_px=missing_length_px)


def build_phase_field_surrogate(record, W=10e-9, spacing=1.25e-9,
                                scale_factor=1.0, margin_W=6.0):
    """Rasterize the sharp profile as a signed-distance diffuse field."""
    profile = surrogate_profile(record, scale_factor=scale_factor)
    z_min = -profile["substrate_length_m"]
    z_max = profile["particle_axial_length_m"] + margin_W*W
    Nz = int(math.ceil((z_max-z_min)/spacing))
    dz = (z_max-z_min)/Nz
    z = z_min + (np.arange(Nz)+0.5)*dz
    r_max = max(
        profile["substrate_cut_radius_m"],
        profile["particle_max_radius_m"])+margin_W*W
    Nr = int(math.ceil(r_max/spacing))
    dr = r_max/Nr
    r_c, r_f = r_centers_faces(Nr, dr)
    particle_interp = PchipInterpolator(
        profile["z_particle_m"], profile["r_particle_m"], extrapolate=False)
    substrate_interp = PchipInterpolator(
        profile["z_substrate_m"], profile["r_substrate_m"], extrapolate=False)
    sharp_radius = np.where(
        z < 0.0, substrate_interp(np.minimum(z, 0.0)),
        particle_interp(np.maximum(z, 0.0)))
    sharp_radius = np.nan_to_num(sharp_radius, nan=-10.0*W)
    inside = r_c[None, :] <= sharp_radius[:, None]
    outside_distance = ndimage.distance_transform_edt(
        ~inside, sampling=(dz, dr))
    inside_distance = ndimage.distance_transform_edt(
        inside, sampling=(dz, dr))
    signed_distance = outside_distance-inside_distance
    f = 0.5*(1.0-np.tanh(signed_distance/W))
    identity_width = 1.5*dz
    particle_fraction = 0.5*(1.0+np.tanh(z/identity_width))[:, None]
    particle = f*particle_fraction
    substrate = f-particle
    result = dict(
        f=f, e1=particle, e2=substrate, z=z, r_c=r_c, r_f=r_f,
        Nz=Nz, Nr=Nr, dz=dz, dr=dr, z1=0.0,
        R_z1=profile["r_neck_m"], lam=profile["particle_axial_length_m"],
        R_cyl=profile["particle_max_radius_m"], profile=profile,
        sharp_radius_m=sharp_radius, z_min_m=z_min, z_max_m=z_max,
        handoff="experimental_particle_axisymmetric_surrogate")
    result["diffuse_volume_m3"] = float(axisym_volume(f, r_c, dr, dz))
    return result


def sharp_profile_metrics(profile):
    """Metrics of the sharp surrogate particle, excluding the substrate."""
    z = np.asarray(profile["z_particle_m"])
    r = np.asarray(profile["r_particle_m"])
    area_2d = float(2.0*np.trapezoid(r, z))
    volume = float(math.pi*np.trapezoid(r*r, z))
    surface = float(2.0*math.pi*np.trapezoid(
        r*np.sqrt(1.0+np.gradient(r, z)**2), z))
    points = np.vstack([
        np.column_stack([z, r]),
        np.column_stack([z[::-1], -r[::-1]])])
    mw = convex_mean_width(points)
    free = np.vstack([
        np.column_stack([z, r]),
        np.column_stack([z[::-1], -r[::-1]])])
    turning = integral_turning(free, closed=False)
    return dict(
        axial_length_m=float(z[-1]-z[0]), width_m=float(2*np.max(r)),
        area_2D_m2=area_2d, axisymmetric_volume_m3=volume,
        axisymmetric_free_surface_area_m2=surface,
        P_h_m=float(mw["convex_hull_perimeter"]),
        mean_width_m=float(mw["mean_width_2D"]),
        normal_support_width_m=float(2.0*np.max(r)),
        free_surface_length_m=float(turning["arc_length"]),
        free_turning_rad=float(turning["turning_angle"]))


def experimental_dimensional_metrics(record):
    particle = np.asarray(record["particle_xy_px"])
    uv = np.asarray(record["particle_uv_px"])
    pixel = record["pixel_size_m"]
    measured = measured_particle_profile(record)
    z = measured["u_px"]*pixel
    r = measured["radius_px"]*pixel
    axisym_volume = float(math.pi*np.trapezoid(r*r, z))
    points_m = particle*pixel
    mw = convex_mean_width(points_m[:-1])
    return dict(
        pixel_size_m=pixel,
        particle_length_m=float(np.ptp(uv[:, 0])*pixel),
        particle_width_m=float(np.ptp(uv[:, 1])*pixel),
        particle_area_m2=float(record["area_geometry"]["area_px2"]*pixel**2),
        axisymmetrized_volume_m3=axisym_volume,
        neck_chord_m=record["neck_chord_m"],
        neck_radius_m=record["neck_radius_m"],
        convex_perimeter_m=float(mw["convex_hull_perimeter"]),
        mean_width_m=float(mw["mean_width_2D"]),
        # This is the minor-axis dimensional check requested for the raw
        # particle (about 0.34 micrometre), not the experimental normal-
        # support estimator.  The latter is imported independently from the
        # experimental analysis table and is about 1.02 micrometre.
        particle_transverse_width_m=float(np.ptp(uv[:, 1])*pixel),
        centroid_error_px=float(np.linalg.norm(
            record["area_geometry"]["centroid_px"]-
            record["stored_centroid_px"])))
