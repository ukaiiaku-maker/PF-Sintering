"""Experimental-comparison geometry records for the PR cyclic run.

The online quantities in this module are deliberately geometric primitives.
They do not feed the nucleation stress or any evolution operator.  The raw
``f=0.5`` meridional contour is retained so the same measurements can later
be repeated with the experimental image-analysis implementation.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from .axisym import axisym_free_energy_gb, axisym_volume
from .pr_stress_metrology import convex_mean_width, integral_turning


LOCAL_CURVATURE_SIGN_CONVENTION = (
    "kappa_m=-d2r/dz2/(1+(dr/dz)^2)^(3/2); positive where r(z) is "
    "locally concave toward the solid at the neck"
)
ANGLE_CONVENTION = (
    "each side-equivalent theta is twice the acute angle between that "
    "free-surface tangent and the radial GB plane; theta_avg is the "
    "arithmetic mean of the two side-equivalent values"
)


def _radius_at_half_level(f, r_c):
    """Linearly interpolate the outermost radial ``f=0.5`` crossing."""
    radius = np.full(f.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(f):
        crossings = np.where((row[:-1] - 0.5) * (row[1:] - 0.5) < 0.0)[0]
        if len(crossings) == 0:
            continue
        i = int(crossings[-1])
        radius[row_index] = (
            r_c[i]
            + (0.5 - row[i]) * (r_c[i + 1] - r_c[i])
            / (row[i + 1] - row[i]))
    return radius


def _branch(radius, z, e1, e2, r_c, z_tj, r_tj, sign, f=None):
    finite = np.isfinite(radius)
    if sign < 0:
        indices = np.where(finite & (z < z_tj))[0][::-1]
    else:
        indices = np.where(finite & (z > z_tj))[0]
    if len(indices) < 4:
        raise ValueError("insufficient f=0.5 contour points on a free-surface branch")
    radii = radius[indices]
    labels = np.empty(len(indices), dtype=np.int8)
    for j, index in enumerate(indices):
        radial_index = int(np.argmin(np.abs(r_c - radii[j])))
        labels[j] = 1 if e1[index, radial_index] >= e2[index, radial_index] else 2
    # Both arrays start at the common TJ and then run outward.  Label zero is
    # reserved for the TJ itself; branch points retain their grain identity.
    z_branch = np.r_[float(z_tj), z[indices]].astype(float)
    r_branch = np.r_[float(r_tj), radii].astype(float)
    grain_label = np.r_[np.int8(0), labels].astype(np.int8)
    if sign > 0 and f is not None:
        # A radial crossing cannot represent the smooth axial tip: once the
        # first radial cell falls below 0.5, the row contains no radial level
        # crossing.  Close the particle branch with the independently
        # interpolated axial f=0.5 crossing at r=0.
        axial = np.asarray(f[:, 0], dtype=float)-0.5
        crossings = np.where(axial[:-1]*axial[1:] < 0.0)[0]
        crossings = crossings[z[crossings] > z_tj]
        if len(crossings):
            index = int(crossings[-1])
            z_tip = float(
                z[index]-axial[index]*(z[index+1]-z[index])
                /(axial[index+1]-axial[index]))
            if z_tip > z_branch[-1]:
                z_branch = np.r_[z_branch, z_tip]
                r_branch = np.r_[r_branch, 0.0]
                label = (1 if e1[index, 0] >= e2[index, 0] else 2)
                grain_label = np.r_[grain_label, np.int8(label)]
    return dict(
        z_m=z_branch, r_m=r_branch, grain_label=grain_label,
        side=int(sign))


def _local_fit(branch, z_tj, window_m):
    z = branch["z_m"]
    r = branch["r_m"]
    mask = np.abs(z - z_tj) <= window_m
    if np.sum(mask) < 5:
        raise ValueError("insufficient branch points in local curvature window")
    coeff = np.polyfit(z[mask], r[mask], 2)
    slope = float(np.polyval(np.polyder(coeff, 1), z_tj))
    second = float(np.polyval(np.polyder(coeff, 2), z_tj))
    curvature = -second / (1.0 + slope * slope) ** 1.5
    half_angle = math.atan2(1.0, abs(slope))
    return dict(
        kappa_m_per_m=float(curvature),
        slope_dr_dz=float(slope),
        theta_side_equiv_rad=float(2.0 * half_angle),
        fit_z_min_m=float(np.min(z[mask])),
        fit_z_max_m=float(np.max(z[mask])),
        fit_points=int(np.sum(mask)))


def _branch_integrals(branch):
    points = np.column_stack([branch["z_m"], branch["r_m"]])
    value = integral_turning(points, closed=False)
    return dict(
        L_f_m=float(value["arc_length"]),
        delta_phi_f_rad=float(value["turning_angle"]))


def _axisymmetric_surface_area(branch):
    z = branch["z_m"]
    r = branch["r_m"]
    ds = np.hypot(np.diff(z), np.diff(r))
    return float(2.0 * math.pi * np.sum(0.5 * (r[:-1] + r[1:]) * ds))


def _weighted_grain_geometry(field, z, r_c, dr, dz):
    weights = np.maximum(field, 0.0) * r_c[None, :]
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("grain has zero axisymmetric weight")
    volume = 2.0 * math.pi * dr * dz * denominator
    z_centroid = float(np.sum(weights * z[:, None]) / denominator)
    return volume, z_centroid


def experimental_geometry_record(state, setup, *, z_tj_m, r_tj_m,
                                 local_window_m=15e-9):
    """Return one scalar primitive record and two ordered raw branches."""
    f, e1, e2 = state
    z = np.asarray(setup["z"], dtype=float)
    r_c = np.asarray(setup["r_c"], dtype=float)
    radius = _radius_at_half_level(f, r_c)
    negative = _branch(radius, z, e1, e2, r_c, z_tj_m, r_tj_m, -1, f)
    positive = _branch(radius, z, e1, e2, r_c, z_tj_m, r_tj_m, +1, f)
    fit_negative = _local_fit(negative, z_tj_m, local_window_m)
    fit_positive = _local_fit(positive, z_tj_m, local_window_m)
    int_negative = _branch_integrals(negative)
    int_positive = _branch_integrals(positive)

    theta_negative = fit_negative["theta_side_equiv_rad"]
    theta_positive = fit_positive["theta_side_equiv_rad"]
    theta_average = 0.5 * (theta_negative + theta_positive)

    # Closed 2-D silhouette of the complete two-grain f=0.5 contour.
    finite = np.isfinite(radius)
    z_outer = z[finite]
    r_outer = radius[finite]
    points = np.vstack([
        np.column_stack([z_outer, r_outer]),
        np.column_stack([z_outer[::-1], -r_outer[::-1]])])
    mean_width = convex_mean_width(points)
    normal_width = float(2.0 * np.max(r_outer))

    V1, zc1 = _weighted_grain_geometry(
        e1, z, r_c, setup["dr"], setup["dz"])
    V2, zc2 = _weighted_grain_geometry(
        e2, z, r_c, setup["dr"], setup["dz"])
    Vsolid = float(axisym_volume(f, r_c, setup["dr"], setup["dz"]))
    A_free_negative = _axisymmetric_surface_area(negative)
    A_free_positive = _axisymmetric_surface_area(positive)
    A_free = A_free_negative + A_free_positive
    A_GB = math.pi * r_tj_m * r_tj_m
    kappa_2_neck = math.sin(0.5 * theta_average) / r_tj_m
    s_line = 2.0 * r_tj_m * math.sin(0.5 * theta_average)
    G_phasefield = float(axisym_free_energy_gb(
        f, e1, e2, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        r_c, setup["r_f"], bc_z="noflux"))
    G_gamma = float(setup["gamma_s"] * A_free + setup["gamma_gb"] * A_GB)

    scalar = dict(
        d_center_m=abs(zc2 - zc1),
        r_n_m=float(r_tj_m),
        A_GB_m2=float(A_GB),
        L_TJ_m=float(2.0 * math.pi * r_tj_m),
        z_TJ_m=float(z_tj_m),
        r_TJ_m=float(r_tj_m),
        theta_negative_rad=float(theta_negative),
        theta_positive_rad=float(theta_positive),
        theta_average_rad=float(theta_average),
        theta_negative_deg=math.degrees(theta_negative),
        theta_positive_deg=math.degrees(theta_positive),
        theta_average_deg=math.degrees(theta_average),
        kappa1_negative_per_m=fit_negative["kappa_m_per_m"],
        kappa1_positive_per_m=fit_positive["kappa_m_per_m"],
        kappa1_average_per_m=0.5 * (
            fit_negative["kappa_m_per_m"] + fit_positive["kappa_m_per_m"]),
        kappa2_negative_per_m=float(
            math.sin(0.5 * theta_negative) / r_tj_m),
        kappa2_positive_per_m=float(
            math.sin(0.5 * theta_positive) / r_tj_m),
        kappa2_neck_per_m=float(kappa_2_neck),
        local_fit_window_m=float(local_window_m),
        local_fit_negative_z_min_m=fit_negative["fit_z_min_m"],
        local_fit_negative_z_max_m=fit_negative["fit_z_max_m"],
        local_fit_positive_z_min_m=fit_positive["fit_z_min_m"],
        local_fit_positive_z_max_m=fit_positive["fit_z_max_m"],
        local_fit_negative_points=float(fit_negative["fit_points"]),
        local_fit_positive_points=float(fit_positive["fit_points"]),
        L_f_negative_m=int_negative["L_f_m"],
        L_f_positive_m=int_positive["L_f_m"],
        L_f_combined_m=int_negative["L_f_m"] + int_positive["L_f_m"],
        delta_phi_f_negative_rad=int_negative["delta_phi_f_rad"],
        delta_phi_f_positive_rad=int_positive["delta_phi_f_rad"],
        delta_phi_f_combined_rad=(
            int_negative["delta_phi_f_rad"] + int_positive["delta_phi_f_rad"]),
        P_h_m=float(mean_width["convex_hull_perimeter"]),
        w_bar_2D_m=float(mean_width["mean_width_2D"]),
        w_N_m=normal_width,
        s_line_m=float(s_line),
        gamma_J_per_m2=float(setup["gamma_s"]),
        V1_m3=float(V1), V2_m3=float(V2), V_solid_m3=Vsolid,
        A_free_m2=float(A_free),
        A_free_negative_m2=float(A_free_negative),
        A_free_positive_m2=float(A_free_positive),
        z_c1_m=float(zc1), z_c2_m=float(zc2),
        equivalent_radius_1_m=float((3.0 * V1 / (4.0 * math.pi)) ** (1.0 / 3.0)),
        equivalent_radius_2_m=float((3.0 * V2 / (4.0 * math.pi)) ** (1.0 / 3.0)),
        equivalent_particle_radius_m=float(
            0.5 * ((3.0 * V1 / (4.0 * math.pi)) ** (1.0 / 3.0)
                   + (3.0 * V2 / (4.0 * math.pi)) ** (1.0 / 3.0))),
        equivalent_volume_radius_m=float(
            (3.0 * Vsolid / (4.0 * math.pi)) ** (1.0 / 3.0)),
        maximum_radial_extent_m=float(np.max(r_outer)),
        G_gamma_J=G_gamma,
        G_phasefield_J=G_phasefield)
    if not all(np.isfinite(float(value)) for value in scalar.values()):
        bad = [key for key, value in scalar.items() if not np.isfinite(float(value))]
        raise ValueError(f"non-finite experimental geometry fields: {bad}")
    return scalar, dict(negative=negative, positive=positive)


def contour_reconstruction(branches):
    """Recompute the four I/O sanity primitives from stored branches."""
    negative = branches["negative"]
    positive = branches["positive"]
    rn = 0.5 * (negative["r_m"][0] + positive["r_m"][0])
    L_negative = _branch_integrals(negative)["L_f_m"]
    L_positive = _branch_integrals(positive)["L_f_m"]
    # P_h and w_N are particle/free-surface primitives, so reconstruct them
    # from the positive (particle) branch and its axisymmetric mirror.  The
    # negative branch is the cropped substrate and must not enter either
    # particle support measure.
    z_all = np.asarray(positive["z_m"])
    r_all = np.asarray(positive["r_m"])
    points = np.vstack([
        np.column_stack([z_all, r_all]),
        np.column_stack([z_all[::-1], -r_all[::-1]])])
    mean_width = convex_mean_width(points)
    return dict(
        r_n_m=float(rn),
        L_f_combined_m=float(L_negative + L_positive),
        P_h_m=float(mean_width["convex_hull_perimeter"]),
        w_N_m=float(2.0 * np.max(r_all)))


class GeometryHistoryWriter:
    """Atomic scalar CSV plus packed variable-length compressed contours."""

    def __init__(self, csv_path, contour_path, base_fields):
        self.csv_path = csv_path
        self.contour_path = contour_path
        self.base_fields = tuple(base_fields)
        self.rows = []
        self.contours = []

    @classmethod
    def reopen(cls, csv_path, contour_path):
        """Reopen an atomically flushed history without pickle."""
        with open(csv_path, newline="") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            raw_rows = list(reader)
        writer = cls(csv_path, contour_path, fields)
        for raw in raw_rows:
            row = {"phase": raw["phase"]}
            for name, value in raw.items():
                if name == "phase":
                    continue
                row[name] = int(float(value)) if name == "sample_id" else float(value)
            writer.rows.append(row)
        with np.load(contour_path, allow_pickle=False) as saved:
            ids = saved["contour_sample_id"].copy()
            sides = saved["branch_code"].copy()
            offsets = saved["offsets"].copy()
            z_all = saved["z_m"].copy()
            r_all = saved["r_m"].copy()
            labels_all = saved["grain_label"].copy()
            for index in range(len(ids)):
                lo, hi = int(offsets[index]), int(offsets[index + 1])
                writer.contours.append(dict(
                    sample_id=int(ids[index]), branch_code=int(sides[index]),
                    z_m=z_all[lo:hi].copy(),
                    r_m=r_all[lo:hi].copy(),
                    grain_label=labels_all[lo:hi].copy()))
        if len(writer.contours) != 2 * len(writer.rows):
            raise ValueError("scalar/contour history length mismatch")
        return writer

    def append(self, scalar, branches):
        row = dict(scalar)
        sample_id = int(row.get("sample_id", len(self.rows)))
        row["sample_id"] = sample_id
        if self.rows and sample_id <= int(self.rows[-1]["sample_id"]):
            raise ValueError("sample_id must increase strictly")
        missing = [name for name in self.base_fields if name not in row]
        if missing:
            raise ValueError(f"missing scalar history fields: {missing}")
        bad = [name for name in self.base_fields
               if name != "phase" and not np.isfinite(float(row[name]))]
        if bad:
            raise ValueError(f"non-finite scalar history fields: {bad}")
        self.rows.append(row)
        for name, code in (("negative", -1), ("positive", 1)):
            branch = branches[name]
            self.contours.append(dict(
                sample_id=sample_id, branch_code=code,
                z_m=np.asarray(branch["z_m"], dtype=float),
                r_m=np.asarray(branch["r_m"], dtype=float),
                grain_label=np.asarray(branch["grain_label"], dtype=np.int8)))

    def flush(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        csv_tmp = self.csv_path + ".writing"
        with open(csv_tmp, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.base_fields)
            writer.writeheader()
            writer.writerows({name: row[name] for name in self.base_fields}
                             for row in self.rows)
        os.replace(csv_tmp, self.csv_path)

        offsets = [0]
        z_parts, r_parts, label_parts = [], [], []
        for contour in self.contours:
            z_parts.append(contour["z_m"])
            r_parts.append(contour["r_m"])
            label_parts.append(contour["grain_label"])
            offsets.append(offsets[-1] + len(contour["z_m"]))
        npz_tmp = self.contour_path + ".writing"
        metadata = json.dumps(dict(
            level_set="f=0.5 outermost radial crossing",
            ordering="each branch starts at TJ and proceeds outward",
            branch_code={"-1": "negative-z branch", "1": "positive-z branch"},
            grain_label={"0": "TJ", "1": "eta1/e1", "2": "eta2/e2"},
            local_curvature_sign_convention=LOCAL_CURVATURE_SIGN_CONVENTION,
            angle_convention=ANGLE_CONVENTION,
            positive_axis_closure=(
                "particle branch ends at axial f=0.5 crossing at r=0")))
        with open(npz_tmp, "wb") as stream:
            np.savez_compressed(
                stream,
                contour_sample_id=np.asarray(
                    [c["sample_id"] for c in self.contours], dtype=np.int64),
                branch_code=np.asarray(
                    [c["branch_code"] for c in self.contours], dtype=np.int8),
                offsets=np.asarray(offsets, dtype=np.int64),
                z_m=np.concatenate(z_parts) if z_parts else np.empty(0),
                r_m=np.concatenate(r_parts) if r_parts else np.empty(0),
                grain_label=(np.concatenate(label_parts) if label_parts
                             else np.empty(0, dtype=np.int8)),
                metadata_json=np.array(metadata))
        os.replace(npz_tmp, self.contour_path)


def load_sample_contours(path, sample_id):
    """Load both packed branches for ``sample_id`` without pickle."""
    out = {}
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["contour_sample_id"]
        sides = saved["branch_code"]
        offsets = saved["offsets"]
        for index in np.where(ids == int(sample_id))[0]:
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            name = "negative" if sides[index] < 0 else "positive"
            out[name] = dict(
                z_m=saved["z_m"][lo:hi].copy(),
                r_m=saved["r_m"][lo:hi].copy(),
                grain_label=saved["grain_label"][lo:hi].copy(),
                side=int(sides[index]))
    if set(out) != {"negative", "positive"}:
        raise KeyError(f"sample {sample_id} does not have both contour branches")
    return out
