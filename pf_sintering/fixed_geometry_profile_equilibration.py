"""Zero-model-time diffuse-profile equilibration at fixed sharp geometry.

The operator rebuilds the equilibrium normal surface and obstacle ownership
profiles about the *existing* 0.5 level sets.  It is a representation map,
not a kinetic PF step: no mobility, diffusivity, or physical/model time enters.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import find_contours

from .gb_obstacle_energy import obstacle_ell, obstacle_profile
from .corrected_interfacial_energy import (
    axisym_corrected_interfacial_energy_components,
    normalized_ownership_variational_derivatives_axisym,
)


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    pieces = []
    for p0, p1 in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(p1-p0))
        count = max(1, int(math.ceil(length/spacing)))
        pieces.append(p0[None, :] + np.arange(count)[:, None]/count*(p1-p0))
    pieces.append(points[-1:])
    return np.vstack(pieces)


def _surface_contour_points(f, setup: dict) -> np.ndarray:
    """Closed physical 0.5 contour obtained by reflecting across the axis."""
    f = np.asarray(f, dtype=float)
    full = np.concatenate([f[:, ::-1], f], axis=1)
    candidates = find_contours(full, 0.5, fully_connected="high")
    if not candidates:
        raise ValueError("solid f=0.5 contour is absent")
    curve = max(candidates, key=len)
    nr = f.shape[1]
    z = setup["z"][0] + curve[:, 0]*setup["dz"]
    r = (curve[:, 1]-nr+0.5)*setup["dr"]
    return np.column_stack([z, r])


def _gb_contour_points(f, phi, setup: dict, z_tj: float) -> np.ndarray:
    """Current ownership 0.5 locus, sampled independently at each radius."""
    f = np.asarray(f, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(setup["z"])
    points = []
    for i, radius in enumerate(setup["r_c"]):
        crossings = np.flatnonzero((phi[:-1, i]-0.5)*(phi[1:, i]-0.5) <= 0.0)
        choices = []
        for j in crossings:
            denominator = phi[j+1, i]-phi[j, i]
            if abs(denominator) < 1.0e-14:
                continue
            fraction = (0.5-phi[j, i])/denominator
            z_cross = z[j] + fraction*(z[j+1]-z[j])
            f_cross = f[j, i] + fraction*(f[j+1, i]-f[j, i])
            if f_cross >= 0.45:
                choices.append((abs(z_cross-z_tj), z_cross))
        if choices:
            points.append((min(choices)[1], float(radius)))
    if len(points) < 4:
        raise ValueError("ownership 0.5 locus is under-resolved")
    points = np.asarray(points)
    # The physical GB starts at the axis.  Linear extrapolation avoids making
    # the first radial cell center an artificial endpoint.
    z_axis = points[0, 0]
    if len(points) >= 2:
        z_axis = points[0, 0] - points[0, 1]*(
            (points[1, 0]-points[0, 0])/(points[1, 1]-points[0, 1]))
    return np.vstack([[z_axis, 0.0], points])


def _signed_distance(points: np.ndarray, inside: np.ndarray,
                     setup: dict) -> np.ndarray:
    # Exact projection to nearby contour segments avoids the small sawtooth
    # in |grad(d)| produced by nearest resampled points.  Original marching-
    # squares segments are grid-scale and nearly uniform, so the eight nearest
    # segment midpoints safely contain the nearest segment.
    points = np.asarray(points, dtype=float)
    start, vector = points[:-1], points[1:]-points[:-1]
    length2 = np.sum(vector*vector, axis=1)
    valid = length2 > 0.0
    start, vector, length2 = start[valid], vector[valid], length2[valid]
    midpoints = start+0.5*vector
    tree = cKDTree(midpoints)
    Z, R = np.meshgrid(setup["z"], setup["r_c"], indexing="ij")
    query = np.column_stack([Z.ravel(), R.ravel()])
    distance = np.empty(len(query), dtype=float)
    chunk = 50000
    k = min(8, len(midpoints))
    for begin in range(0, len(query), chunk):
        end = min(begin+chunk, len(query))
        q = query[begin:end]
        indices = tree.query(q, k=k, workers=-1)[1]
        if k == 1: indices = indices[:, None]
        p0 = start[indices]
        direction = vector[indices]
        relative = q[:, None, :]-p0
        parameter = np.sum(relative*direction, axis=2)/length2[indices]
        parameter = np.clip(parameter, 0.0, 1.0)
        nearest = p0+parameter[:, :, None]*direction
        distance[begin:end] = np.sqrt(np.min(np.sum(
            (q[:, None, :]-nearest)**2, axis=2), axis=1))
    distance = distance.reshape(Z.shape)
    return np.where(inside, -distance, distance)


def _axisym_integral(field, setup: dict) -> float:
    weights = (2.0*math.pi*setup["r_c"][None, :]
               * setup["dr"]*setup["dz"])
    return float(np.sum(weights*np.asarray(field, dtype=float)))


def _outer_radius(field: np.ndarray, r_c: np.ndarray) -> np.ndarray:
    result = np.full(field.shape[0], np.nan)
    for j, row in enumerate(field):
        crossings = np.flatnonzero((row[:-1]-0.5)*(row[1:]-0.5) <= 0.0)
        usable = [int(i) for i in crossings if row[i+1] != row[i]
                  and min(row[i], row[i+1]) <= 0.5 <= max(row[i], row[i+1])]
        if usable:
            i = usable[-1]
            fraction = (0.5-row[i])/(row[i+1]-row[i])
            result[j] = r_c[i]+fraction*(r_c[i+1]-r_c[i])
    return result


def _align_radial_half_level(rebuilt: np.ndarray, reference: np.ndarray,
                             setup: dict) -> tuple[np.ndarray, dict]:
    """Make every resolved radial 0.5 intersection equal to the reference."""
    result = np.asarray(rebuilt, dtype=float).copy()
    target = _outer_radius(reference, setup["r_c"])
    r_c = np.asarray(setup["r_c"])
    shifts = []
    for j, radius in enumerate(target):
        if not math.isfinite(radius):
            continue
        upper = int(np.searchsorted(r_c, radius, side="right"))
        upper = min(max(upper, 1), len(r_c)-1)
        lower = upper-1
        fraction = (radius-r_c[lower])/(r_c[upper]-r_c[lower])
        base = np.clip(result[j], np.finfo(float).eps,
                       1.0-np.finfo(float).eps)
        phase = np.arctanh(1.0-2.0*base)
        def residual(shift):
            shifted = 0.5*(1.0-np.tanh(phase-shift/setup["W"]))
            value = (1.0-fraction)*shifted[lower]+fraction*shifted[upper]
            return shifted, value-0.5
        lo, hi = -setup["W"], setup["W"]
        _, rlo = residual(lo); _, rhi = residual(hi)
        for _ in range(20):
            if rlo <= 0.0 <= rhi: break
            lo *= 2.0; hi *= 2.0
            _, rlo = residual(lo); _, rhi = residual(hi)
        else:
            raise RuntimeError("row half-level alignment cannot be bracketed")
        shift = 0.0
        for _ in range(70):
            shift = 0.5*(lo+hi)
            shifted, value = residual(shift)
            if abs(value) <= 2.0e-15:
                break
            if value < 0.0: lo = shift
            else: hi = shift
        result[j] = shifted
        shifts.append(shift)
    changed_support = int(np.count_nonzero(
        np.isfinite(_outer_radius(result, r_c)) != np.isfinite(target)))
    if changed_support:
        raise RuntimeError("profile reconstruction changed axial contour support")
    return result, dict(
        radial_half_level_alignment_rows=len(shifts),
        maximum_radial_phase_alignment_nm=(
            max(abs(value) for value in shifts)*1e9 if shifts else 0.0),
        rms_radial_phase_alignment_nm=(
            float(np.sqrt(np.mean(np.asarray(shifts)**2)))*1e9
            if shifts else 0.0))


def _preserve_grain_amounts_inside_solid(f_new, phi, target_particle,
                                         target_substrate, setup: dict):
    """Correct both grain amounts using only f>0.75 interior capacity."""
    f_new = np.asarray(f_new, dtype=float)
    phi = np.asarray(phi, dtype=float)
    weights = (2.0*math.pi*setup["r_c"][None, :]
               * setup["dr"]*setup["dz"])
    current = np.array([
        float(np.sum(weights*f_new*phi)),
        float(np.sum(weights*f_new*(1.0-phi)))])
    target = np.array([target_particle, target_substrate])
    delta = target-current
    bases = []
    for amount, ownership in zip(delta, (phi, 1.0-phi)):
        if amount >= 0.0:
            capacity = np.maximum(1.0-f_new, 0.0)
        else:
            capacity = np.maximum(f_new-0.75, 0.0)
        bases.append(capacity*ownership**4*(f_new > 0.75))
    matrix = np.array([
        [float(np.sum(weights*phi*bases[0])),
         float(np.sum(weights*phi*bases[1]))],
        [float(np.sum(weights*(1.0-phi)*bases[0])),
         float(np.sum(weights*(1.0-phi)*bases[1]))]])
    coefficients = np.linalg.solve(matrix, delta)
    corrected = f_new+coefficients[0]*bases[0]+coefficients[1]*bases[1]
    if np.min(corrected) < -1e-14 or np.max(corrected) > 1.0+1e-14:
        raise RuntimeError("interior grain-amount correction exceeded capacity")
    corrected = np.clip(corrected, 0.0, 1.0)
    final_particle = float(np.sum(weights*corrected*phi))
    final_substrate = float(np.sum(weights*corrected*(1.0-phi)))
    return corrected, dict(
        interior_particle_basis_coefficient=float(coefficients[0]),
        interior_substrate_basis_coefficient=float(coefficients[1]),
        f_interior_particle_amount_relative_error=(
            final_particle/target_particle-1.0),
        f_interior_substrate_amount_relative_error=(
            final_substrate/target_substrate-1.0),
        interior_correction_minimum_f=float(np.min(
            f_new[(bases[0] != 0.0) | (bases[1] != 0.0)])))


def _ownership_crossing_z(f, phi, setup: dict) -> np.ndarray:
    z = np.asarray(setup["z"])
    result = np.full(phi.shape[1], np.nan)
    for i in range(phi.shape[1]):
        crossings = np.flatnonzero((phi[:-1, i]-0.5)*(phi[1:, i]-0.5) <= 0.0)
        usable = []
        for j in crossings:
            if phi[j+1, i] == phi[j, i]: continue
            fraction = (0.5-phi[j, i])/(phi[j+1, i]-phi[j, i])
            f_cross = f[j, i]+fraction*(f[j+1, i]-f[j, i])
            # Extend the constraint through the diffuse free-surface tail so
            # its intersection with f=0.5 (the field TJ) is preserved too.
            if f_cross >= 0.05:
                usable.append((j, fraction))
        if usable:
            # The solid ownership interface is unique in qualified states.
            j, fraction = usable[0]
            result[i] = z[j]+fraction*(z[j+1]-z[j])
    return result


def _align_ownership_half_level(phi, reference_phi, f, setup: dict):
    """Preserve the complete resolved ownership 0.5 locus column by column."""
    result = np.asarray(phi, dtype=float).copy()
    target = _ownership_crossing_z(f, reference_phi, setup)
    z = np.asarray(setup["z"])
    shifts = []
    for i, location in enumerate(target):
        if not math.isfinite(location): continue
        upper = int(np.searchsorted(z, location, side="right"))
        upper = min(max(upper, 1), len(z)-1); lower = upper-1
        fraction = (location-z[lower])/(z[upper]-z[lower])
        base = np.clip(result[:, i], np.finfo(float).eps,
                       1.0-np.finfo(float).eps)
        logit = np.log(base/(1.0-base))
        def residual(shift):
            shifted = 1.0/(1.0+np.exp(-(logit+shift)))
            value = (1.0-fraction)*shifted[lower]+fraction*shifted[upper]
            return shifted, value-0.5
        lo, hi = -2.0, 2.0
        _, rlo = residual(lo); _, rhi = residual(hi)
        for _ in range(20):
            if rlo <= 0.0 <= rhi: break
            lo *= 2.0; hi *= 2.0
            _, rlo = residual(lo); _, rhi = residual(hi)
        else: raise RuntimeError("ownership half-level alignment cannot be bracketed")
        shift = 0.0
        for _ in range(70):
            shift = 0.5*(lo+hi); shifted, value = residual(shift)
            if abs(value) <= 2.0e-15: break
            if value < 0.0: lo = shift
            else: hi = shift
        result[:, i] = shifted; shifts.append(shift)
    return result, dict(
        ownership_half_level_alignment_columns=len(shifts),
        maximum_ownership_logit_alignment=max(
            (abs(value) for value in shifts), default=0.0))


def _project_ownership_crossing_constraints(value, crossings, setup: dict,
                                            target_value: float):
    """Orthogonally project each crossing bracket to its linear constraint."""
    result = np.asarray(value, dtype=float).copy()
    z = np.asarray(setup["z"])
    for i, location in enumerate(crossings):
        if not math.isfinite(location): continue
        upper = int(np.searchsorted(z, location, side="right"))
        upper = min(max(upper, 1), len(z)-1); lower = upper-1
        fraction = (location-z[lower])/(z[upper]-z[lower])
        a, b = 1.0-fraction, fraction
        residual = a*result[lower, i]+b*result[upper, i]-target_value
        denominator = a*a+b*b
        result[lower, i] -= residual*a/denominator
        result[upper, i] -= residual*b/denominator
    return result


def _fixed_half_level_mass_correction(profile, target_integral: float,
                                      setup: dict) -> tuple[np.ndarray, float]:
    """Enforce one integral without changing 0, 0.5, or 1 level values."""
    profile = np.asarray(profile, dtype=float)
    basis = 16.0*profile*(1.0-profile)*(profile-0.5)**2

    def evaluate(coefficient):
        value = np.clip(profile + coefficient*basis, 0.0, 1.0)
        return value, _axisym_integral(value, setup)-target_integral

    initial, residual = evaluate(0.0)
    tolerance = max(2.0e-15*abs(target_integral), 1.0e-36)
    if abs(residual) <= tolerance:
        return initial, 0.0
    lo, hi = -1.0, 1.0
    vlo, rlo = evaluate(lo)
    vhi, rhi = evaluate(hi)
    for _ in range(20):
        if rlo <= 0.0 <= rhi:
            break
        if residual > 0.0:
            lo *= 2.0; vlo, rlo = evaluate(lo)
        else:
            hi *= 2.0; vhi, rhi = evaluate(hi)
    else:
        raise RuntimeError("fixed-level mass correction cannot be bracketed")
    for _ in range(80):
        mid = 0.5*(lo+hi)
        value, rmid = evaluate(mid)
        if abs(rmid) <= tolerance:
            return value, mid
        if rmid < 0.0:
            lo = mid
        else:
            hi = mid
    raise RuntimeError("fixed-level mass correction did not converge")


def equilibrate_fixed_geometry_profiles(state, setup: dict, *, z_tj_m: float,
                                        equilibrate_f: bool = True,
                                        equilibrate_phi: bool = True):
    """Return a zero-time equilibrium-profile representation and diagnostics."""
    f, particle, substrate = (np.asarray(field, dtype=float) for field in state)
    phi = np.divide(particle, f, out=np.zeros_like(f), where=f > 1.0e-14)
    phi = np.clip(phi, 0.0, 1.0)
    target_volume = _axisym_integral(f, setup)
    target_particle = _axisym_integral(particle, setup)
    target_substrate = _axisym_integral(substrate, setup)
    surface_points = _surface_contour_points(f, setup)
    gb_points = _gb_contour_points(f, phi, setup, z_tj_m)

    f_new = f.copy()
    f_coefficient = 0.0
    if equilibrate_f:
        distance = _signed_distance(surface_points, f >= 0.5, setup)
        raw = 0.5*(1.0-np.tanh(distance/setup["W"]))
        f_new = raw
        cumulative_mass_coefficient = 0.0
        for constraint_iteration in range(1, 31):
            f_new, coefficient = _fixed_half_level_mass_correction(
                f_new, target_volume, setup)
            cumulative_mass_coefficient += coefficient
            f_new, alignment = _align_radial_half_level(f_new, f, setup)
            residual = _axisym_integral(f_new, setup)/target_volume-1.0
            if abs(residual) <= 2.0e-14:
                break
        else:
            raise RuntimeError("smooth mass/contour constraint iteration stalled")
        f_coefficient = cumulative_mass_coefficient
        interior = dict(
            f_constraint_iterations=constraint_iteration,
            f_post_alignment_volume_relative_error=residual,
            interior_particle_basis_coefficient=0.0,
            interior_substrate_basis_coefficient=0.0,
            f_interior_particle_amount_relative_error=math.nan,
            f_interior_substrate_amount_relative_error=math.nan,
            interior_correction_minimum_f=math.nan)
    else:
        alignment = dict(radial_half_level_alignment_rows=0,
                         maximum_radial_phase_alignment_nm=0.0,
                         rms_radial_phase_alignment_nm=0.0)
        interior = dict(interior_particle_basis_coefficient=0.0,
                        interior_substrate_basis_coefficient=0.0,
                        f_constraint_iterations=0,
                        f_post_alignment_volume_relative_error=0.0,
                        f_interior_particle_amount_relative_error=0.0,
                        f_interior_substrate_amount_relative_error=0.0,
                        interior_correction_minimum_f=1.0)

    raw = phi.copy()
    phi_coefficient = 0.0
    if equilibrate_phi:
        # obstacle_profile rises from 0 to 1 with positive signed distance;
        # the particle side is the existing phi>=0.5 side.
        distance = -_signed_distance(gb_points, phi >= 0.5, setup)
        raw = obstacle_profile(distance, obstacle_ell(
            setup["p"].k_eta, setup["Wc"]))
    # Grain amount is integral(f*phi), not integral(phi).  Enforce it even
    # when only f is rebuilt, so a surface-only representation map does not
    # silently transfer ownership between grains.
    basis = 16.0*raw*(1.0-raw)*(raw-0.5)**2
    weights = (2.0*math.pi*setup["r_c"][None, :]
               * setup["dr"]*setup["dz"])
    def residual(coefficient):
        candidate = np.clip(raw+coefficient*basis, 0.0, 1.0)
        return candidate, float(np.sum(weights*f_new*candidate))-target_particle
    candidate, value = residual(0.0)
    tolerance = max(2.0e-15*abs(target_particle), 1.0e-36)
    if abs(value) > tolerance:
        lo, hi = -1.0, 1.0
        _, rlo = residual(lo); _, rhi = residual(hi)
        for _ in range(20):
            if rlo <= 0.0 <= rhi:
                break
            if value > 0.0:
                lo *= 2.0; _, rlo = residual(lo)
            else:
                hi *= 2.0; _, rhi = residual(hi)
        else:
            raise RuntimeError("ownership amount correction cannot be bracketed")
        for _ in range(80):
            phi_coefficient = 0.5*(lo+hi)
            candidate, rmid = residual(phi_coefficient)
            if abs(rmid) <= tolerance:
                break
            if rmid < 0.0: lo = phi_coefficient
            else: hi = phi_coefficient
        else:
            raise RuntimeError("ownership amount correction did not converge")
    phi_new = candidate

    particle_new = f_new*phi_new
    substrate_new = f_new*(1.0-phi_new)
    return (f_new, particle_new, substrate_new), dict(
        zero_model_time=True,
        surface_profile_equilibrated=bool(equilibrate_f),
        ownership_profile_equilibrated=bool(equilibrate_phi),
        f_fixed_half_level_mass_coefficient=float(f_coefficient),
        phi_fixed_half_level_amount_coefficient=float(phi_coefficient),
        volume_relative_error=_axisym_integral(f_new, setup)/target_volume-1.0,
        particle_amount_relative_error=(
            _axisym_integral(particle_new, setup)/target_particle-1.0),
        surface_contour_point_count=len(surface_points),
        ownership_contour_point_count=len(gb_points),
        **alignment, **interior)


def equilibrate_ownership_variational_fixed_geometry(
        state, setup: dict, *, maximum_iterations: int = 400,
        maximum_trial_change: float = 0.02,
        relative_energy_tolerance: float = 1.0e-11):
    """Minimize ownership energy at fixed ``f``, amounts, and 0.5 topology.

    This is a backtracked numerical minimization index, not model time.  The
    original side of the ownership 0.5 locus is imposed cellwise, the descent
    mobility vanishes at 0/0.5/1, and an exact scalar correction preserves the
    particle amount after every accepted trial.
    """
    f, particle, _ = (np.asarray(field, dtype=float) for field in state)
    phi = np.divide(particle, f, out=np.zeros_like(f), where=f > 1.0e-14)
    phi = np.clip(phi, 0.0, 1.0)
    reference_phi = phi.copy()
    reference_crossings = _ownership_crossing_z(f, reference_phi, setup)
    side_low = phi < 0.5
    side_high = phi > 0.5
    weights = (2.0*math.pi*setup["r_c"][None, :]
               * setup["dr"]*setup["dz"])
    target = float(np.sum(weights*f*phi))

    def ownership_energy(value):
        local_state = (f, f*value, f*(1.0-value))
        terms = axisym_corrected_interfacial_energy_components(
            *local_state, setup["p"], setup["Wc"], setup["dr"],
            setup["dz"], setup["r_c"], setup["r_f"], bc_z="noflux")
        return terms["F_GB"], terms

    def enforce_amount(value):
        value = np.asarray(value, dtype=float)
        basis = 16.0*value*(1.0-value)*(value-0.5)**2
        basis = _project_ownership_crossing_constraints(
            basis, reference_crossings, setup, 0.0)
        current_residual = float(np.sum(weights*f*value))-target
        derivative = float(np.sum(weights*f*basis))
        tolerance = max(2.0e-15*abs(target), 1.0e-36)
        if abs(current_residual) <= tolerance:
            return value.copy(), 0.0
        if derivative != 0.0:
            direct_coefficient = -current_residual/derivative
            direct = value+direct_coefficient*basis
            direct_residual = float(np.sum(weights*f*direct))-target
            if (np.min(direct) >= 0.0 and np.max(direct) <= 1.0
                    and abs(direct_residual) <= tolerance):
                return direct, direct_coefficient
        def evaluate(coefficient):
            trial = value+coefficient*basis
            trial[side_low] = np.minimum(trial[side_low], 0.5)
            trial[side_high] = np.maximum(trial[side_high], 0.5)
            trial = _project_ownership_crossing_constraints(
                trial, reference_crossings, setup, 0.5)
            trial = np.clip(trial, 0.0, 1.0)
            return trial, float(np.sum(weights*f*trial))-target
        trial, r0 = evaluate(0.0)
        if abs(r0) <= tolerance:
            return trial, 0.0
        lo, hi = -1.0, 1.0
        _, rlo = evaluate(lo); _, rhi = evaluate(hi)
        for _ in range(30):
            if rlo <= 0.0 <= rhi: break
            if r0 > 0.0:
                lo *= 2.0; _, rlo = evaluate(lo)
            else:
                hi *= 2.0; _, rhi = evaluate(hi)
        else:
            raise RuntimeError("variational ownership amount cannot be bracketed")
        coefficient = 0.0
        for _ in range(80):
            coefficient = 0.5*(lo+hi)
            trial, residual = evaluate(coefficient)
            if abs(residual) <= tolerance: return trial, coefficient
            if residual < 0.0: lo = coefficient
            else: hi = coefficient
        raise RuntimeError("variational ownership amount did not converge")

    initial_energy, initial_terms = ownership_energy(phi)
    energy = initial_energy
    accepted = 0
    maximum_half_level_cell_change = 0.0
    last_correction = 0.0
    alignment = dict(
        ownership_half_level_alignment_columns=int(np.count_nonzero(
            np.isfinite(reference_crossings))),
        maximum_ownership_logit_alignment=0.0,
        pinned_ownership_half_level_cells=0,
        ownership_crossing_constraint=(
            "linear 0.5 interpolation projected exactly at every resolved GB column"))
    for iteration in range(1, maximum_iterations+1):
        _, gphi = normalized_ownership_variational_derivatives_axisym(
            f, f*phi, f*(1.0-phi), setup["p"], setup["Wc"],
            setup["dr"], setup["dz"], setup["r_c"], setup["r_f"],
            bc_z="noflux")
        mobility = 16.0*phi*(1.0-phi)*(phi-0.5)**2
        denominator = float(np.sum(weights*f*mobility))
        if denominator <= 0.0: break
        lagrange = float(np.sum(weights*f*mobility*gphi))/denominator
        direction = -mobility*(gphi-lagrange)
        scale = float(np.max(np.abs(direction)))
        if not math.isfinite(scale) or scale == 0.0: break
        alpha = maximum_trial_change/scale
        accepted_trial = False
        for _ in range(30):
            trial = np.clip(phi+alpha*direction, 0.0, 1.0)
            trial[side_low] = np.minimum(trial[side_low], 0.5)
            trial[side_high] = np.maximum(trial[side_high], 0.5)
            trial = _project_ownership_crossing_constraints(
                trial, reference_crossings, setup, 0.5)
            trial, correction = enforce_amount(trial)
            trial_energy, _ = ownership_energy(trial)
            if trial_energy < energy:
                improvement = energy-trial_energy
                maximum_half_level_cell_change = max(
                    maximum_half_level_cell_change,
                    float(np.max(np.abs(trial-phi))))
                phi = trial
                energy = trial_energy
                last_correction = correction
                accepted += 1
                accepted_trial = True
                break
            alpha *= 0.5
        if not accepted_trial:
            break
        if improvement <= relative_energy_tolerance*max(abs(energy), 1e-300):
            break
    final_state = (f, f*phi, f*(1.0-phi))
    final_energy, final_terms = ownership_energy(phi)
    return final_state, dict(
        zero_model_time=True,
        variational_ownership_equilibration=True,
        attempted_iterations=iteration,
        accepted_iterations=accepted,
        initial_ownership_energy_J=initial_energy,
        final_ownership_energy_J=final_energy,
        ownership_energy_change_J=final_energy-initial_energy,
        final_amount_correction_coefficient=last_correction,
        particle_amount_relative_error=(
            float(np.sum(weights*final_state[1]))/target-1.0),
        maximum_accepted_phi_increment=maximum_half_level_cell_change,
        initial_G_phi_pot_J=initial_terms["F_GB_coupling"],
        initial_G_phi_grad_J=initial_terms["F_GB_gradient"],
        final_G_phi_pot_J=final_terms["F_GB_coupling"],
        final_G_phi_grad_J=final_terms["F_GB_gradient"],
        **alignment)
