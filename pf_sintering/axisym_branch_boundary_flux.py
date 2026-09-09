"""Axisymmetric free-surface branch boundary flux in model time.

This module is the width-free replacement for depositing a finite packet over
an arbitrary number of triple-junction endpoint rows.  Each free-surface
branch is a one-dimensional finite-volume mesh in meridional arclength ``s``.
For outward-positive surface volume flux ``Q`` the axisymmetric conservation
law is

``v_n = -(1/r) d(r Q)/ds``.

The TJ supplies a boundary flux at ``s=0``.  Interior fluxes follow the
instantaneous phase-field chemical potential sampled on the ``f=0.5``
contour.  The resulting control-volume changes are mapped back to the diffuse
field with exact signed tanh-phase shifts, one contour row at a time.  There
is no endpoint-row count, Gaussian source, or fitted TJ width.

All rates are expressed per *model time*.  Physical seconds and material
diffusivities are intentionally outside this demonstration operator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.sparse import linalg as sparse_linalg


@dataclass(frozen=True)
class AxisymmetricSurfaceBranch:
    """Cell-centered meridional finite-volume representation of one branch."""

    side: str
    row_indices: np.ndarray
    s_centers_m: np.ndarray
    s_faces_m: np.ndarray
    r_centers_m: np.ndarray
    r_faces_m: np.ndarray
    mu_Pa: np.ndarray

    def __post_init__(self):
        n = len(self.row_indices)
        if self.side not in ("positive", "negative"):
            raise ValueError("side must be 'positive' or 'negative'")
        if n < 2:
            raise ValueError("a surface branch needs at least two cells")
        for name in ("s_centers_m", "r_centers_m", "mu_Pa"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length must match row_indices")
        if len(self.s_faces_m) != n + 1 or len(self.r_faces_m) != n + 1:
            raise ValueError("branch face arrays must have n+1 entries")
        if self.s_faces_m[0] != 0.0 or np.any(np.diff(self.s_faces_m) <= 0.0):
            raise ValueError("s faces must increase strictly from zero")
        if np.any(self.r_centers_m <= 0.0) or np.any(self.r_faces_m <= 0.0):
            raise ValueError("axisymmetric branch radii must be positive")
        if not all(np.all(np.isfinite(getattr(self, name))) for name in (
                "s_centers_m", "s_faces_m", "r_centers_m", "r_faces_m", "mu_Pa")):
            raise ValueError("branch geometry and chemical potential must be finite")

    @property
    def cell_lengths_m(self):
        return np.diff(self.s_faces_m)


def _outer_half_contour(f, sampled, r_c):
    """Return outer ``f=0.5`` radius and a linearly sampled row field."""
    values = np.asarray(f, dtype=float)
    field = np.asarray(sampled, dtype=float)
    radii = np.asarray(r_c, dtype=float)
    if values.shape != field.shape or values.ndim != 2:
        raise ValueError("f and sampled field must be matching two-dimensional arrays")
    if values.shape[1] != len(radii):
        raise ValueError("r_c does not match the radial field dimension")
    contour = np.full(values.shape[0], np.nan)
    contour_field = np.full(values.shape[0], np.nan)
    for row in range(values.shape[0]):
        crossing = np.where(
            (values[row, :-1] >= 0.5) & (values[row, 1:] < 0.5))[0]
        if len(crossing) == 0:
            continue
        j = int(crossing[-1])
        denominator = values[row, j + 1] - values[row, j]
        if denominator == 0.0:
            continue
        fraction = (0.5 - values[row, j]) / denominator
        contour[row] = radii[j] + fraction * (radii[j + 1] - radii[j])
        contour_field[row] = field[row, j] + fraction * (
            field[row, j + 1] - field[row, j])
    return contour, contour_field


def _junction_radius(z, contour_radius, z_tj):
    valid = np.flatnonzero(np.isfinite(contour_radius))
    if len(valid) < 2:
        raise ValueError("not enough resolved contour rows to locate the TJ radius")
    order = valid[np.argsort(np.abs(np.asarray(z)[valid] - z_tj))]
    chosen = order[:min(4, len(order))]
    coefficients = np.polyfit(np.asarray(z)[chosen] - z_tj, contour_radius[chosen], 1)
    radius = float(coefficients[1])
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("extrapolated TJ radius is not positive and finite")
    return radius


def _make_branch(side, indices, z, radius, sampled_mu, z_tj, r_tj):
    indices = np.asarray(indices, dtype=int)
    if side == "positive":
        indices = indices[np.argsort(np.asarray(z)[indices])]
    else:
        indices = indices[np.argsort(-np.asarray(z)[indices])]
    points = np.c_[np.asarray(z)[indices], radius[indices]]
    first = np.array([z_tj, r_tj], dtype=float)
    segment_lengths = np.linalg.norm(
        np.diff(np.vstack([first, points]), axis=0), axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("branch contour contains coincident points")
    s_centers = np.cumsum(segment_lengths)
    s_faces = np.empty(len(indices) + 1)
    s_faces[0] = 0.0
    s_faces[1:-1] = 0.5 * (s_centers[:-1] + s_centers[1:])
    s_faces[-1] = s_centers[-1] + 0.5 * (
        s_centers[-1] - s_centers[-2])
    r_faces = np.empty(len(indices) + 1)
    r_faces[0] = r_tj
    r_faces[1:-1] = 0.5 * (radius[indices[:-1]] + radius[indices[1:]])
    r_faces[-1] = radius[indices[-1]]
    return AxisymmetricSurfaceBranch(
        side=side,
        row_indices=indices,
        s_centers_m=s_centers,
        s_faces_m=s_faces,
        r_centers_m=radius[indices],
        r_faces_m=r_faces,
        mu_Pa=sampled_mu[indices])


def extract_axisymmetric_surface_branches(
        f, mu_Pa, r_c, z, z_tj, r_tj=None, minimum_cells=3):
    """Extract the two outward-oriented free-surface branches.

    Every resolved contour row on each side of ``z_tj`` is a finite-volume
    cell.  The nearest face is the geometric TJ at ``s=0`` and the far face
    is no-flux.  Refinement changes the control-volume mesh naturally; there
    is no selected count of endpoint cells.
    """
    z_values = np.asarray(z, dtype=float)
    radius, sampled_mu = _outer_half_contour(f, mu_Pa, r_c)
    if len(z_values) != len(radius):
        raise ValueError("z does not match the axial field dimension")
    if r_tj is None:
        r_tj = _junction_radius(z_values, radius, z_tj)
    valid = np.isfinite(radius) & np.isfinite(sampled_mu)
    positive = np.flatnonzero(valid & (z_values > z_tj))
    negative = np.flatnonzero(valid & (z_values < z_tj))
    if len(positive) < minimum_cells or len(negative) < minimum_cells:
        raise ValueError("both TJ surface branches must be resolved")
    return (
        _make_branch(
            "positive", positive, z_values, radius, sampled_mu, z_tj, float(r_tj)),
        _make_branch(
            "negative", negative, z_values, radius, sampled_mu, z_tj, float(r_tj)))


def extrapolate_outer_tj_branch_limit(
        branch: AxisymmetricSurfaceBranch, core_exclusion_m: float,
        outer_fit_limit_m: float, polynomial_degree: int = 1):
    """Extrapolate the free-surface ``mu`` outside the diffuse TJ core to s=0.

    Both fit limits are physical/geometric distances.  The diagnostic never
    selects a number of endpoint cells and is separate from the boundary-flux
    discretization.  The returned value is an outer branch limit, not the
    diffuse three-interface-core value.
    """
    if core_exclusion_m <= 0.0 or outer_fit_limit_m <= core_exclusion_m:
        raise ValueError("fit limits must satisfy 0 < core exclusion < outer limit")
    if polynomial_degree not in (1, 2):
        raise ValueError("polynomial_degree must be 1 or 2")
    selected = ((branch.s_centers_m >= core_exclusion_m)
                & (branch.s_centers_m <= outer_fit_limit_m))
    minimum = polynomial_degree + 2
    if np.count_nonzero(selected) < minimum:
        raise ValueError("outer branch-limit interval is not sufficiently resolved")
    s_fit = branch.s_centers_m[selected]
    mu_fit = branch.mu_Pa[selected]
    coefficients = np.polyfit(s_fit, mu_fit, polynomial_degree)
    predicted = np.polyval(coefficients, s_fit)
    residual_rms = float(np.sqrt(np.mean((predicted - mu_fit) ** 2)))
    return dict(
        side=branch.side,
        mu_TJ_outer_limit_Pa=float(np.polyval(coefficients, 0.0)),
        core_exclusion_m=float(core_exclusion_m),
        outer_fit_limit_m=float(outer_fit_limit_m),
        polynomial_degree=polynomial_degree,
        n_fit_points=int(np.count_nonzero(selected)),
        fit_residual_rms_Pa=residual_rms,
        fit_coefficients=[float(value) for value in coefficients],
        endpoint_row_count=None,
        diffuse_TJ_core_sampled=False)


def symmetric_outer_tj_limit(
        branches, core_exclusion_m: float, outer_fit_limit_m: float,
        polynomial_degree: int = 1):
    """Return branch limits, their symmetric mean, and their disagreement."""
    if len(branches) != 2 or {branch.side for branch in branches} != {
            "positive", "negative"}:
        raise ValueError("one positive and one negative branch are required")
    limits = {
        branch.side: extrapolate_outer_tj_branch_limit(
            branch, core_exclusion_m, outer_fit_limit_m, polynomial_degree)
        for branch in branches}
    positive = limits["positive"]["mu_TJ_outer_limit_Pa"]
    negative = limits["negative"]["mu_TJ_outer_limit_Pa"]
    scale = max(abs(positive), abs(negative), 1e-300)
    return dict(
        branch=limits,
        symmetric_mean_mu_TJ_outer_limit_Pa=0.5 * (positive + negative),
        branch_difference_Pa=positive - negative,
        branch_relative_disagreement=abs(positive - negative) / scale,
        definition="linear/quadratic s->0 extrapolation outside diffuse TJ core",
        force_balance_constraint_applied=False)


def axisymmetric_branch_flux_fv(
        branch: AxisymmetricSurfaceBranch,
        surface_flux_mobility_m6_per_J_model_time,
        incoming_volume_rate_m3_per_model_time,
        outer_volume_rate_m3_per_model_time=0.0):
    """Evaluate conservative flux divergence on one surface branch.

    ``Q_face[0]`` is imposed by the supplied TJ volume rate.  Interior faces
    use ``Q=-K dmu/ds`` and the last face has the requested outer rate
    (normally zero).  Total circumference-resolved rate is
    ``F=2*pi*r*Q``.  Hence cell rates telescope exactly to incoming minus
    outgoing volume rate.
    """
    mobility = float(surface_flux_mobility_m6_per_J_model_time)
    incoming = float(incoming_volume_rate_m3_per_model_time)
    outgoing = float(outer_volume_rate_m3_per_model_time)
    if mobility < 0.0 or not all(math.isfinite(value) for value in (
            mobility, incoming, outgoing)):
        raise ValueError("mobility must be non-negative and boundary rates finite")
    n = len(branch.row_indices)
    Q_face = np.zeros(n + 1)
    total_rate_face = np.zeros(n + 1)
    total_rate_face[0] = incoming
    Q_face[0] = incoming / (2.0 * math.pi * branch.r_faces_m[0])
    if n > 1:
        center_spacing = np.diff(branch.s_centers_m)
        Q_face[1:-1] = -mobility * np.diff(branch.mu_Pa) / center_spacing
        total_rate_face[1:-1] = (
            2.0 * math.pi * branch.r_faces_m[1:-1] * Q_face[1:-1])
    total_rate_face[-1] = outgoing
    Q_face[-1] = outgoing / (2.0 * math.pi * branch.r_faces_m[-1])
    cell_volume_rate = total_rate_face[:-1] - total_rate_face[1:]
    cell_area = 2.0 * math.pi * branch.r_centers_m * branch.cell_lengths_m
    normal_velocity = cell_volume_rate / cell_area
    closure = float(np.sum(cell_volume_rate) - (incoming - outgoing))
    return dict(
        Q_face_m2_per_model_time=Q_face,
        circumference_volume_rate_faces_m3_per_model_time=total_rate_face,
        cell_volume_rate_m3_per_model_time=cell_volume_rate,
        normal_velocity_m_per_model_time=normal_velocity,
        cell_area_m2=cell_area,
        integrated_closure_m3_per_model_time=closure,
        incoming_Q_m2_per_model_time=Q_face[0],
        outer_Q_m2_per_model_time=Q_face[-1],
        conservation_law="v_n=-(1/r)d(rQ)/ds",
        kinetic_time_basis="model_time_demonstration")


def _nonuniform_second_derivative_matrix(s_centers_m):
    """Three-point second derivative with constant and linear null modes."""
    s = np.asarray(s_centers_m, dtype=float)
    n = len(s)
    if n < 3 or np.any(np.diff(s) <= 0.0):
        raise ValueError("at least three increasing branch centers are required")
    rows = []
    columns = []
    values = []
    for i in range(n):
        indices = np.array([0, 1, 2]) if i == 0 else (
            np.array([n - 3, n - 2, n - 1]) if i == n - 1
            else np.array([i - 1, i, i + 1]))
        offsets = s[indices] - s[i]
        moments = np.vstack([np.ones(3), offsets, offsets ** 2])
        weights = np.linalg.solve(moments, np.array([0.0, 0.0, 2.0]))
        rows.extend([i] * 3)
        columns.extend(indices.tolist())
        values.extend(weights.tolist())
    return sparse.csr_matrix((values, (rows, columns)), shape=(n, n))


def linearly_implicit_mullins_branch_increment(
        branch: AxisymmetricSurfaceBranch, flux_result: dict,
        dt_model: float, B_m4_per_model_time: float):
    """Linearly implicit, mass-conservative fourth-order branch increment.

    The explicit instantaneous PF-chemical-potential divergence is retained on
    the right-hand side.  A Mullins Jacobian approximation stabilizes only the
    contour displacement increment ``h``:

    ``(A + dt B D2.T A D2) h = dt g``.

    Here ``A`` is the axisymmetric surface-cell area and ``g`` is the explicit
    cell volume rate.  Since ``D2*1=0``, summing ``A*h`` preserves the imposed
    boundary influx exactly.  The update reduces to explicit Euler as
    ``dt*B/ds^4 -> 0``.
    """
    dt = float(dt_model)
    B = float(B_m4_per_model_time)
    if dt <= 0.0 or B <= 0.0 or not math.isfinite(dt) or not math.isfinite(B):
        raise ValueError("implicit Mullins dt and B must be positive and finite")
    area = np.asarray(flux_result["cell_area_m2"], dtype=float)
    rate = np.asarray(
        flux_result["cell_volume_rate_m3_per_model_time"], dtype=float)
    if len(area) != len(branch.row_indices) or rate.shape != area.shape:
        raise ValueError("flux result does not match branch geometry")
    D2 = _nonuniform_second_derivative_matrix(branch.s_centers_m)
    area_matrix = sparse.diags(area, format="csc")
    stiffness = (D2.T @ area_matrix @ D2).tocsc()
    system = area_matrix + dt * B * stiffness
    normal_displacement = sparse_linalg.spsolve(system, dt * rate)
    if not np.all(np.isfinite(normal_displacement)):
        raise RuntimeError("implicit Mullins branch solve produced non-finite displacement")
    volume_changes = area * normal_displacement
    expected = dt * float(np.sum(rate))
    correction = expected - float(np.sum(volume_changes))
    # A uniform normal displacement is in the D2 nullspace.  Applying the
    # roundoff correction through that mode preserves the implicit equation's
    # physical nullspace rather than modifying a TJ endpoint cell.
    volume_changes += correction * area / float(np.sum(area))
    normal_displacement = volume_changes / area
    return dict(
        cell_volume_changes_m3=volume_changes,
        normal_displacements_m=normal_displacement,
        expected_integrated_volume_change_m3=expected,
        integrated_closure_m3=float(np.sum(volume_changes) - expected),
        roundoff_null_mode_correction_m3=correction,
        B_m4_per_model_time=B,
        time_integrator="linearly implicit Mullins Rosenbrock step",
        explicit_small_step_limit=True,
        endpoint_regularization_used=False)


def signed_tanh_phase_shift(f, radial_displacement_m, W):
    """Apply an exact signed tanh-phase translation to one radial profile."""
    values = np.asarray(f, dtype=float)
    displacement = float(radial_displacement_m)
    if W <= 0.0 or not np.isfinite(displacement):
        raise ValueError("W must be positive and displacement finite")
    out = values.copy()
    active = (values > 0.0) & (values < 1.0)
    if displacement == 0.0 or not np.any(active):
        return out
    clipped = np.clip(values[active], np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    phase = np.arctanh(1.0 - 2.0 * clipped)
    out[active] = 0.5 * (1.0 - np.tanh(phase - displacement / W))
    return out


def _solve_row_volume_shift(f_row, r_c, target_weighted, W, relative_tolerance):
    values = np.asarray(f_row, dtype=float)
    radii = np.asarray(r_c, dtype=float)
    target = float(target_weighted)
    absolute_tolerance = max(
        relative_tolerance * abs(target),
        0.1 * np.finfo(float).eps * float(np.sum(np.abs(radii))))
    if target == 0.0:
        return values.copy(), 0.0, 0.0

    def evaluate(shift):
        shifted = signed_tanh_phase_shift(values, shift, W)
        added = float(np.sum(
            np.asarray(radii, dtype=np.longdouble)
            * (np.asarray(shifted, dtype=np.longdouble)
               - np.asarray(values, dtype=np.longdouble)),
            dtype=np.longdouble))
        return shifted, added

    derivative = float(np.sum(
        radii * (2.0 / W) * np.clip(values, 0.0, 1.0)
        * (1.0 - np.clip(values, 0.0, 1.0))))
    if derivative <= 0.0:
        raise ValueError("contour row contains no resolved translation mode")
    estimate = target / derivative
    if target > 0.0:
        lo, hi = 0.0, max(estimate, np.finfo(float).tiny)
        _, added = evaluate(hi)
        for _ in range(80):
            if added >= target:
                break
            hi *= 2.0
            _, added = evaluate(hi)
        else:
            raise RuntimeError("positive row-volume shift cannot be bracketed")
    else:
        lo, hi = min(estimate, -np.finfo(float).tiny), 0.0
        _, added = evaluate(lo)
        for _ in range(80):
            if added <= target:
                break
            lo *= 2.0
            _, added = evaluate(lo)
        else:
            raise RuntimeError("negative row-volume shift cannot be bracketed")

    shift = min(max(estimate, lo), hi)
    for _ in range(80):
        shifted, added = evaluate(shift)
        residual = added - target
        if abs(residual) <= absolute_tolerance:
            return shifted, shift, added
        if residual < 0.0:
            lo = shift
        else:
            hi = shift
        slope = float(np.sum(
            radii * (2.0 / W) * shifted * (1.0 - shifted)))
        newton = shift - residual / slope if slope > 0.0 else math.nan
        shift = newton if lo < newton < hi else 0.5 * (lo + hi)
    raise RuntimeError("signed row-volume solve did not converge")


def _apply_branch_normal_phase_displacement(
        f, branch, r_c, dr, dz, normal_displacements_m, target_volume_m3,
        axisym_cell_factor_m2, W, relative_tolerance):
    """Translate the diffuse interface in its local normal phase coordinate.

    ``atanh(1-2f)`` is the dimensionless tanh phase ``d/W`` for a resolved
    signed-distance interface.  Subtracting ``h/W`` therefore advances the
    interface by the signed normal distance ``h``; it is not a radial motion
    of the contour intersection in an axial grid row.

    The finite-volume displacements already contain the desired spatial
    distribution.  A single uniform normal null-mode correction is solved for
    each branch to remove the finite-shift quadrature error while preserving
    that distribution and the exact integrated branch volume.
    """
    values = np.asarray(f, dtype=float)
    rows = np.asarray(branch.row_indices, dtype=int)
    h_base = np.asarray(normal_displacements_m, dtype=float)
    radii = np.asarray(r_c, dtype=float)
    target = float(target_volume_m3)
    factor = float(axisym_cell_factor_m2)
    if values.ndim != 2 or len(rows) != len(h_base):
        raise ValueError("normal displacement must match the branch rows")
    if not np.all(np.isfinite(h_base)):
        raise ValueError("normal displacements must be finite")

    base_rows = values[rows].copy()
    active = (base_rows > 0.0) & (base_rows < 1.0)
    # Extend each interfacial v_n off the contour along its nearest normal.
    # A constant value on an axial row is not a normal extension when the
    # free surface is sloped, especially in the first few W from the TJ.
    contour_points = np.column_stack([
        rows.astype(float) * float(dz), branch.r_centers_m])
    active_local_row, active_radial = np.nonzero(active)
    active_points = np.column_stack([
        rows[active_local_row].astype(float) * float(dz),
        radii[active_radial]])
    nearest = cKDTree(contour_points).query(active_points, k=1)[1]
    extended_h = np.zeros_like(base_rows)
    extended_h[active_local_row, active_radial] = h_base[nearest]
    weighted_r = np.asarray(radii, dtype=np.longdouble)[None, :]

    def evaluate(correction_m):
        shifts = extended_h[active] + float(correction_m)
        shifted_rows = base_rows.copy()
        clipped_active = np.clip(
            base_rows[active], np.finfo(float).eps,
            1.0 - np.finfo(float).eps)
        phase = np.arctanh(1.0 - 2.0 * clipped_active)
        shifted_rows[active] = 0.5 * (
            1.0 - np.tanh(phase - shifts / W))
        added = factor * float(np.sum(
            weighted_r
            * (np.asarray(shifted_rows, dtype=np.longdouble)
               - np.asarray(base_rows, dtype=np.longdouble)),
            dtype=np.longdouble))
        return shifted_rows, added

    shifted, added = evaluate(0.0)
    closure_scale = max(abs(target), float(np.sum(np.abs(
        np.asarray(normal_displacements_m)
        * (2.0 * math.pi * radii.mean())))), 1e-300)
    absolute_tolerance = max(
        relative_tolerance * max(abs(target), 1e-300),
        0.25 * np.finfo(float).eps * closure_scale)
    residual = added - target
    if abs(residual) <= absolute_tolerance:
        correction = 0.0
    else:
        clipped = np.clip(base_rows, 0.0, 1.0)
        derivative = factor * float(np.sum(
            radii[None, :] * (2.0 / W) * clipped * (1.0 - clipped)))
        if derivative <= 0.0:
            raise ValueError("branch contains no resolved normal translation mode")
        estimate = -residual / derivative
        lo = min(0.0, estimate)
        hi = max(0.0, estimate)
        _, added_lo = evaluate(lo)
        _, added_hi = evaluate(hi)
        step = max(abs(estimate), np.finfo(float).eps * W)
        for _ in range(80):
            if added_lo <= target <= added_hi:
                break
            if target < added_lo:
                hi, added_hi = lo, added_lo
                lo -= step
                _, added_lo = evaluate(lo)
            else:
                lo, added_lo = hi, added_hi
                hi += step
                _, added_hi = evaluate(hi)
            step *= 2.0
        else:
            raise RuntimeError("normal phase-volume correction cannot be bracketed")

        correction = min(max(estimate, lo), hi)
        for _ in range(80):
            shifted, added = evaluate(correction)
            residual = added - target
            if abs(residual) <= absolute_tolerance:
                break
            if residual < 0.0:
                lo = correction
            else:
                hi = correction
            shifted_clipped = np.clip(shifted, 0.0, 1.0)
            slope = factor * float(np.sum(
                radii[None, :] * (2.0 / W)
                * shifted_clipped * (1.0 - shifted_clipped)))
            newton = correction - residual / slope if slope > 0.0 else math.nan
            correction = newton if lo < newton < hi else 0.5 * (lo + hi)
        else:
            raise RuntimeError("normal phase-volume correction did not converge")
    shifted, added = evaluate(correction)
    row_changes = factor * np.asarray([
        float(np.sum(
            np.asarray(radii, dtype=np.longdouble)
            * (np.asarray(new, dtype=np.longdouble)
               - np.asarray(old, dtype=np.longdouble)),
            dtype=np.longdouble))
        for new, old in zip(shifted, base_rows)])
    return shifted, dict(
        requested_normal_displacements_m=h_base,
        applied_normal_displacements_m=h_base + correction,
        uniform_normal_closure_correction_m=float(correction),
        added_cell_volume_changes_m3=row_changes,
        requested_branch_volume_m3=target,
        added_branch_volume_m3=float(added),
        closure_error_m3=float(added - target),
        geometry_realization=(
            "signed tanh phase-coordinate translation with nearest-contour "
            "normal-ray velocity extension"),
        normal_extension_mode="nearest-contour normal coordinate",
        radial_row_displacement_used=False)


def apply_axisymmetric_branch_boundary_flux_step(
        f, grain1, grain2, r_c, dr, dz, W,
        branches, surface_flux_mobility_m6_per_J_model_time,
        incoming_volume_rate_m3_per_model_time, dt_model,
        branch_fractions=(0.5, 0.5), positive_branch_grain=1,
        relative_tolerance=2e-10,
        implicit_mullins_B_m4_per_model_time=None,
        branch_volume_rates_m3_per_model_time=None,
        mass_closure_relative_tolerance=1e-8,
        chemical_potential_filter_length_m=None,
        normal_displacement_diffusion_B_m4_per_model_time=None):
    """Advance two branches over one model-time finite-volume step.

    The scientific path supplies the two signed rates from a zero-storage TJ
    node; ``branch_fractions`` remains available for manufactured tests.
    Conservative ``-d(rQ)/ds`` supplies a volume increment for every contour
    cell.  Dividing by its true axisymmetric surface area gives its normal
    displacement, which translates the local signed tanh phase coordinate.
    The old independent radial-row volume solve is not used.  One branch may
    feed the node while the other carries that exchange plus the net GB
    delivery.
    Ordinary volumetric ``M_s`` evolution must not also be applied over this
    interval; this operator already represents event-time surface transport.
    """
    fields = tuple(np.asarray(a, dtype=float) for a in (f, grain1, grain2))
    if any(a.shape != fields[0].shape for a in fields):
        raise ValueError("f and grain fields must have matching shapes")
    if len(branches) != 2 or {branch.side for branch in branches} != {
            "positive", "negative"}:
        raise ValueError("exactly one positive and one negative branch are required")
    if branch_volume_rates_m3_per_model_time is None:
        fractions = tuple(float(x) for x in branch_fractions)
        if len(fractions) != 2 or any(x < 0.0 for x in fractions) or not math.isclose(
                sum(fractions), 1.0, rel_tol=0.0, abs_tol=2e-15):
            raise ValueError("branch fractions must be non-negative and sum to one")
        branch_rates = {
            "positive": incoming_volume_rate_m3_per_model_time * fractions[0],
            "negative": incoming_volume_rate_m3_per_model_time * fractions[1]}
        normalized_signed_rates = None
        partition_mode = "prescribed fractions"
    else:
        if set(branch_volume_rates_m3_per_model_time) != {"positive", "negative"}:
            raise ValueError("node branch rates require positive and negative entries")
        branch_rates = {
            side: float(branch_volume_rates_m3_per_model_time[side])
            for side in ("positive", "negative")}
        if not all(math.isfinite(rate) for rate in branch_rates.values()):
            raise ValueError("node branch rates must be finite")
        rate_scale = max(
            abs(incoming_volume_rate_m3_per_model_time),
            sum(abs(rate) for rate in branch_rates.values()), 1e-300)
        if not math.isclose(
                sum(branch_rates.values()), incoming_volume_rate_m3_per_model_time,
                rel_tol=2e-12, abs_tol=2e-12 * rate_scale):
            raise ValueError("node branch rates do not sum to total GB input")
        fractions = None
        normalized_signed_rates = (
            {side: branch_rates[side] / incoming_volume_rate_m3_per_model_time
             for side in ("positive", "negative")}
            if incoming_volume_rate_m3_per_model_time > 0.0 else
            {"positive": None, "negative": None})
        partition_mode = "zero-storage TJ node"
    if positive_branch_grain not in (1, 2):
        raise ValueError("positive_branch_grain must be 1 or 2")
    if dt_model <= 0.0 or incoming_volume_rate_m3_per_model_time < 0.0:
        raise ValueError("dt and incoming volume rate must be positive/non-negative")
    if mass_closure_relative_tolerance <= 0.0:
        raise ValueError("mass closure relative tolerance must be positive")
    ordered = sorted(branches, key=lambda branch: branch.side, reverse=True)
    if set(ordered[0].row_indices).intersection(set(ordered[1].row_indices)):
        raise ValueError("positive and negative branch rows must be disjoint")

    f_new, g1_new, g2_new = (a.copy() for a in fields)
    factor = 2.0 * math.pi * float(dr) * float(dz)
    branch_diags = {}
    for branch in ordered:
        fraction = (None if fractions is None else
                    fractions[0] if branch.side == "positive" else fractions[1])
        owner = (positive_branch_grain if branch.side == "positive"
                 else 3 - positive_branch_grain)
        flux_branch = branch
        if chemical_potential_filter_length_m is not None:
            filter_length = float(chemical_potential_filter_length_m)
            spacing = float(np.median(np.diff(branch.s_centers_m)))
            if not math.isfinite(filter_length) or filter_length <= 0.0:
                raise ValueError("chemical-potential filter length must be positive")
            flux_branch = AxisymmetricSurfaceBranch(
                side=branch.side, row_indices=branch.row_indices,
                s_centers_m=branch.s_centers_m, s_faces_m=branch.s_faces_m,
                r_centers_m=branch.r_centers_m, r_faces_m=branch.r_faces_m,
                mu_Pa=gaussian_filter1d(
                    branch.mu_Pa, sigma=filter_length/spacing, mode="nearest"))
        flux = axisymmetric_branch_flux_fv(
            flux_branch, surface_flux_mobility_m6_per_J_model_time,
            branch_rates[branch.side])
        implicit = None
        if implicit_mullins_B_m4_per_model_time is None:
            targets_m3 = flux["cell_volume_rate_m3_per_model_time"] * dt_model
        else:
            implicit = linearly_implicit_mullins_branch_increment(
                flux_branch, flux, dt_model, implicit_mullins_B_m4_per_model_time)
            targets_m3 = implicit["cell_volume_changes_m3"]
        if implicit is None:
            normal_displacements = (
                targets_m3 / flux["cell_area_m2"])
        else:
            normal_displacements = np.asarray(
                implicit["normal_displacements_m"], dtype=float)
        # The FV divergence is cell conservative, but the diffuse interface
        # cannot realize alternating normal motions on a scale below W as
        # independent sharp corners.  Project the displacement (not the
        # source or boundary flux) onto the resolved W-scale interface
        # coordinate, then restore its exact axisymmetric volume moment.
        spacing = float(np.median(np.diff(branch.s_centers_m)))
        # The tanh profile f=(1-tanh(d/W))/2 spans 5--95% over
        # 2*atanh(0.9)*W.  Normal-displacement detail below that physical
        # diffuse thickness is not representable as a smooth PF interface.
        diffuse_resolution_length_m = 2.0 * math.atanh(0.9) * float(W)
        if normal_displacement_diffusion_B_m4_per_model_time is None:
            ell_SD_m = 0.0
        else:
            B_event = float(normal_displacement_diffusion_B_m4_per_model_time)
            if not math.isfinite(B_event) or B_event <= 0.0:
                raise ValueError("normal-displacement diffusion B must be positive")
            ell_SD_m = (B_event * float(dt_model)) ** 0.25
        reconstruction_length_m = max(
            ell_SD_m, diffuse_resolution_length_m)
        reconstruction_sigma_cells = reconstruction_length_m / spacing
        resolved_normal_displacements = gaussian_filter1d(
            normal_displacements, sigma=reconstruction_sigma_cells,
            mode="nearest")
        area = np.asarray(flux["cell_area_m2"], dtype=float)
        resolved_normal_displacements += (
            float(np.sum(targets_m3))
            - float(np.sum(area * resolved_normal_displacements))
        ) / float(np.sum(area))
        try:
            shifted_rows, normal_diag = _apply_branch_normal_phase_displacement(
                f_new, branch, r_c, dr, dz, resolved_normal_displacements,
                float(np.sum(targets_m3)), factor, W, relative_tolerance)
        except RuntimeError as error:
            raise RuntimeError(
                f"{branch.side} branch normal phase update: {error}") from error
        delta = shifted_rows - f_new[branch.row_indices]
        f_new[branch.row_indices] = shifted_rows
        if owner == 1:
            g1_new[branch.row_indices] += delta
        else:
            g2_new[branch.row_indices] += delta
        applied_normal = np.asarray(
            normal_diag["applied_normal_displacements_m"], dtype=float)
        added_rows_m3 = np.asarray(
            normal_diag["added_cell_volume_changes_m3"], dtype=float)
        branch_diags[branch.side] = dict(
            owner_grain=owner,
            fraction=fraction,
            row_indices=branch.row_indices.tolist(),
            normal_phase_displacements_m=applied_normal.tolist(),
            raw_fv_normal_displacements_m=(
                np.asarray(normal_displacements, dtype=float).tolist()),
            normal_displacement_reconstruction_length_m=reconstruction_length_m,
            packet_surface_diffusion_length_m=ell_SD_m,
            diffuse_interface_5_95_resolution_length_m=(
                diffuse_resolution_length_m),
            redistribution_length_control=(
                "packet_surface_diffusion" if ell_SD_m >= diffuse_resolution_length_m
                else "diffuse_interface_5_95_resolution"),
            normal_displacement_reconstruction=(
                "conservative max(ell_SD, tanh 5-95% thickness) projection"),
            radial_phase_shifts_m=None,
            radial_row_displacement_used=False,
            uniform_normal_closure_correction_m=normal_diag[
                "uniform_normal_closure_correction_m"],
            requested_cell_volume_changes_m3=targets_m3.tolist(),
            added_cell_volume_changes_m3=added_rows_m3.tolist(),
            fv=flux,
            chemical_potential_filter_length_m=(
                None if chemical_potential_filter_length_m is None
                else float(chemical_potential_filter_length_m)),
            requested_branch_volume_m3=float(np.sum(targets_m3)),
            added_branch_volume_m3=float(np.sum(added_rows_m3)),
            geometry_realization=normal_diag["geometry_realization"])
        branch_diags[branch.side]["time_integration"] = (
            dict(time_integrator="explicit Euler", endpoint_regularization_used=False)
            if implicit is None else implicit)

    expected = incoming_volume_rate_m3_per_model_time * dt_model
    gross_branch_volume = dt_model * sum(abs(rate) for rate in branch_rates.values())
    closure_scale = max(abs(expected), gross_branch_volume, 1e-300)
    # The explicit-reference path uses extremely small model-time packets.
    # Form the diagnostic difference in extended precision so subtractive
    # cancellation in ``f_new-f_old`` is not mistaken for a conservation
    # failure at those small doses.
    actual = factor * float(np.sum(
        np.asarray(r_c, dtype=np.longdouble)[None, :]
        * (np.asarray(f_new, dtype=np.longdouble)
           - np.asarray(fields[0], dtype=np.longdouble)),
        dtype=np.longdouble))
    owner_actual = {
        owner: factor * float(np.sum(
            np.asarray(r_c, dtype=np.longdouble)[None, :]
            * (np.asarray(updated, dtype=np.longdouble)
               - np.asarray(fields[owner], dtype=np.longdouble)),
            dtype=np.longdouble))
        for owner, updated in ((1, g1_new), (2, g2_new))}
    owner_expected = {
        owner: sum(
            row["added_branch_volume_m3"]
            for row in branch_diags.values() if row["owner_grain"] == owner)
        for owner in (1, 2)}
    owner_closure = {
        owner: owner_actual[owner] - owner_expected[owner]
        for owner in (1, 2)}
    owner_scale = max(
        *(abs(value) for value in owner_expected.values()),
        gross_branch_volume, 1e-300)
    partition = float(np.max(np.abs(g1_new + g2_new - f_new)))
    if not np.isclose(
            actual, expected, rtol=mass_closure_relative_tolerance,
            atol=mass_closure_relative_tolerance * closure_scale):
        raise RuntimeError(
            "branch boundary-flux update did not close total volume: "
            f"actual={actual:.9e} expected={expected:.9e} "
            f"relative={(actual - expected) / closure_scale:.9e}")
    if partition > 1e-10:
        raise RuntimeError("branch boundary-flux update violated grain partition")
    if any(abs(value) > mass_closure_relative_tolerance * owner_scale
           for value in owner_closure.values()):
        raise RuntimeError(
            "branch boundary-flux update did not close grain ownership volumes: "
            f"grain1={owner_closure[1]:.9e} grain2={owner_closure[2]:.9e}")
    return f_new, g1_new, g2_new, dict(
        kinetic_time_basis="model_time_demonstration",
        incoming_volume_rate_m3_per_model_time=incoming_volume_rate_m3_per_model_time,
        dt_model=dt_model,
        expected_added_volume_m3=expected,
        gross_signed_branch_volume_m3=gross_branch_volume,
        actual_added_volume_m3=actual,
        closure_error_m3=actual - expected,
        closure_relative=(actual - expected) / closure_scale,
        closure_relative_scale_m3=closure_scale,
        mass_closure_relative_tolerance=mass_closure_relative_tolerance,
        branch_fractions=fractions,
        branch_rate_over_net_GB=normalized_signed_rates,
        branch_volume_rates_m3_per_model_time=branch_rates,
        branch_partition_mode=partition_mode,
        branch=branch_diags,
        grain_flux_volume_change_m3=owner_actual,
        expected_grain_flux_volume_change_m3=owner_expected,
        grain_flux_closure_m3=owner_closure,
        grain_flux_closure_relative={
            owner: owner_closure[owner] / owner_scale for owner in (1, 2)},
        partition_residual=partition,
        endpoint_row_count=None,
        endpoint_width_m=None,
        ordinary_M_s_step_applied=False,
        branch_time_integrator=(
            "explicit Euler" if implicit_mullins_B_m4_per_model_time is None
            else "linearly implicit Mullins Rosenbrock step"),
        surface_operator=(
            "axisymmetric 1-D FV TJ boundary flux with conservative normal "
            "signed-phase displacement"))
