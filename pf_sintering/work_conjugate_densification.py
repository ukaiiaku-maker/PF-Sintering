"""Work-conjugate one-contact densification coordinate.

``u`` is the rigid translation of one kinematic body toward a stationary
neighbor.  It is distinct from the historical material-transfer quota.  The
temporary bodies are made from the total-solid field, so translating one body
and taking their union removes the swept GB volume exactly once.  The same
translation moves the diffuse ownership boundary by ``u/2``: the mid-plane
between a stationary body and a body translated by ``u``.

The lost union volume is placed at the selected triple junction.  This makes
the diagnostic path exactly volume conservative without clipping.  Ordinary
surface diffusion remains a separate operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _weighted_sum(field, r_c):
    return float(np.sum(np.asarray(field, dtype=np.longdouble)
                        * np.asarray(r_c, dtype=np.longdouble)[None, :],
                        dtype=np.longdouble))


def axisymmetric_volume(field, r_c, dr, dz):
    return 2.0 * math.pi * float(dr) * float(dz) * _weighted_sum(field, r_c)


def conservative_shift_toward_minus_z(field, displacement_m, dz):
    """First-order finite-volume translation toward -z for one subcell."""
    alpha = float(displacement_m) / float(dz)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"subcell shift requires 0 <= u/dz <= 1; got {alpha}")
    src = np.asarray(field, dtype=float)
    out = np.empty_like(src)
    out[0] = src[0] + alpha * src[1]
    out[1:-1] = (1.0-alpha) * src[1:-1] + alpha * src[2:]
    out[-1] = (1.0-alpha) * src[-1]
    return out


def bounded_shift_toward_minus_z(field, displacement_m, dz):
    """Translate an intensive field with constant boundary extrapolation."""
    alpha = float(displacement_m) / float(dz)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"subcell shift requires 0 <= u/dz <= 1; got {alpha}")
    src = np.asarray(field, dtype=float)
    out = np.empty_like(src)
    out[:-1] = (1.0-alpha)*src[:-1] + alpha*src[1:]
    out[-1] = src[-1]
    return out


def _normalized_support(raw, r_c):
    support = np.maximum(np.asarray(raw, dtype=float), 0.0)
    norm = _weighted_sum(support, r_c)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("source support has zero or non-finite volume weight")
    return support / norm


def triple_junction_support(f, z, r_c, *, z_tj, r_tj, width):
    """Normalized selected-contact arrival support from the reference field."""
    bounded = np.minimum(1.0, np.maximum(0.0, np.asarray(f, dtype=float)))
    interface = 16.0 * bounded**2 * (1.0-bounded)**2
    distance2 = ((np.asarray(z)[:, None]-float(z_tj))**2
                 + (np.asarray(r_c)[None, :]-float(r_tj))**2)
    return _normalized_support(
        interface * np.exp(-0.5*distance2/float(width)**2), r_c)


@dataclass(frozen=True)
class DensificationResult:
    f: np.ndarray
    ownership: np.ndarray
    swept_volume_m3: float
    work_area_m2: float
    total_volume_before_m3: float
    total_volume_after_m3: float
    mass_relative_error: float
    partition_residual: float
    f_bounds: tuple[float, float]
    moving_body_displacement_m: float
    ownership_midplane_displacement_m: float
    source_max_increment: float


def one_contact_state(
        f, ownership, z, r_c, *, dr, dz, u_m, gb_z_m, tj_r_m, width_m,
        moving_grain, neighbor_grain, source_support=None):
    """Return the state at physical one-contact displacement ``u_m``.

    The moving grain must be on the +z side of the selected boundary and
    therefore translates toward -z.  Other ownership fields remain the
    stationary complement.  The returned state is constructed directly from
    the reference state; packetization cannot change its endpoint.
    """
    f0 = np.asarray(f, dtype=float)
    phi0 = np.asarray(ownership, dtype=float)
    if phi0.ndim != 3 or phi0.shape[1:] != f0.shape:
        raise ValueError("ownership must have shape (ngrains, nz, nr)")
    if moving_grain == neighbor_grain:
        raise ValueError("moving and neighbor grains must differ")
    if np.max(np.abs(phi0.sum(axis=0)-1.0)) > 2e-12:
        raise ValueError("reference normalized ownership does not close to one")
    u = float(u_m)
    if not 0.0 <= u <= float(dz):
        raise ValueError("one-contact coordinate is restricted to one subcell")

    moving_side = np.asarray(z)[:, None] >= float(gb_z_m)
    body_moving = np.where(moving_side, f0, 0.0)
    body_stationary = f0-body_moving
    moved = conservative_shift_toward_minus_z(body_moving, u, dz)
    overlap = np.where((moved > 0.0) & (body_stationary > 0.0),
                       np.minimum(moved, body_stationary), 0.0)
    f_union = moved+body_stationary-overlap
    swept_weighted = _weighted_sum(f0, r_c)-_weighted_sum(f_union, r_c)
    roundoff = 2e-13*max(abs(_weighted_sum(f0, r_c)), 1e-300)
    if swept_weighted < -roundoff:
        raise RuntimeError("translated body union gained material")
    swept_weighted = max(0.0, swept_weighted)

    # Relative ownership within the selected pair moves with the geometrical
    # GB mid-plane.  No fitted coefficient enters: a relative approach u
    # moves the plane equidistant from the two bodies by u/2.
    pair0 = phi0[moving_grain]+phi0[neighbor_grain]
    chi0 = np.divide(phi0[moving_grain], pair0,
                     out=np.zeros_like(f0), where=pair0 > 1e-30)
    chi = bounded_shift_toward_minus_z(chi0, 0.5*u, dz)
    other = phi0.sum(axis=0)-pair0
    pair = 1.0-other
    phi = phi0.copy()
    phi[moving_grain] = pair*chi
    phi[neighbor_grain] = pair-phi[moving_grain]
    if np.min(phi) < -2e-14 or np.max(phi) > 1.0+2e-14:
        raise RuntimeError("advected ownership left the simplex")

    if source_support is None:
        source_support = triple_junction_support(
            f0, z, r_c, z_tj=gb_z_m, r_tj=tj_r_m, width=width_m)
    support = np.asarray(source_support, dtype=float)
    if support.shape != f0.shape:
        raise ValueError("source support shape differs from the state")
    if not np.isclose(_weighted_sum(support, r_c), 1.0,
                      rtol=2e-13, atol=2e-15):
        raise ValueError("source support is not axisymmetrically normalized")
    added = swept_weighted*support
    f1 = f_union+added
    if np.min(f1) < -2e-12 or np.max(f1) > 1.0+2e-12:
        raise RuntimeError("physical event exceeds total-solid bounds")
    eta1 = phi*f1[None, :]
    phi1 = np.divide(eta1, f1[None, :], out=phi.copy(),
                     where=np.abs(f1[None, :]) > 1e-30)

    v0 = axisymmetric_volume(f0, r_c, dr, dz)
    v1 = axisymmetric_volume(f1, r_c, dr, dz)
    swept = 2.0*math.pi*float(dr)*float(dz)*swept_weighted
    return DensificationResult(
        f=f1, ownership=phi1, swept_volume_m3=swept,
        work_area_m2=(swept/u if u > 0.0 else float("nan")),
        total_volume_before_m3=v0, total_volume_after_m3=v1,
        mass_relative_error=abs(v1-v0)/max(abs(v0), 1e-300),
        partition_residual=float(np.max(np.abs(phi1.sum(axis=0)-1.0))),
        f_bounds=(float(np.min(f1)), float(np.max(f1))),
        moving_body_displacement_m=u,
        ownership_midplane_displacement_m=0.5*u,
        source_max_increment=float(np.max(added)))


def central_virtual_work(energy, state_at_u, u_m, delta_u_m):
    """Return ``-dG/du`` from discarded copies of one physical path."""
    u = float(u_m); du = float(delta_u_m)
    if du <= 0.0 or u-du < 0.0:
        raise ValueError("central virtual-work probe requires 0 < du <= u")
    gp = float(energy(state_at_u(u+du)))
    gm = float(energy(state_at_u(u-du)))
    return -(gp-gm)/(2.0*du)

@dataclass(frozen=True)
class RigidFrameState:
    """Discarded state generated by an explicit signed rigid-frame motion.

    ``approach_m`` is positive for contact closure.  For a moving body on the
    +z side, its laboratory-frame translation is therefore ``-approach_m``
    and the midpoint ownership plane translates by ``-approach_m/2``.
    """
    f: np.ndarray
    ownership: np.ndarray
    approach_m: float
    moving_frame_translation_z_m: float
    ownership_plane_translation_z_m: float
    swept_volume_m3: float
    mass_relative_error: float
    partition_residual: float
    f_bounds: tuple[float, float]
    source_max_increment: float


@dataclass(frozen=True)
class OneSidedDirectionalComponents:
    """Literal positive event direction split before energy contraction."""
    state: RigidFrameState
    f_swept_u: np.ndarray
    f_deposit_u: np.ndarray
    ownership_u: np.ndarray
    swept_volume_rate_m2: float


def conservative_signed_shift_toward_minus_z(field, displacement_m, dz):
    """Conservative subcell translation; signed positive direction is ``-z``."""
    signed = float(displacement_m)/float(dz)
    if abs(signed) > 1.0:
        raise ValueError(f"subcell shift requires abs(u/dz) <= 1; got {signed}")
    src = np.asarray(field, dtype=float)
    if signed >= 0.0:
        return conservative_shift_toward_minus_z(src, signed*dz, dz)
    alpha = -signed
    out = np.empty_like(src)
    out[0] = (1.0-alpha)*src[0]
    out[1:-1] = (1.0-alpha)*src[1:-1] + alpha*src[:-2]
    out[-1] = src[-1] + alpha*src[-2]
    return out


def bounded_signed_shift_toward_minus_z(field, displacement_m, dz):
    """Signed bounded interpolation for an intensive (nonconserved) field."""
    signed = float(displacement_m)/float(dz)
    if abs(signed) > 1.0:
        raise ValueError(f"subcell shift requires abs(u/dz) <= 1; got {signed}")
    src = np.asarray(field, dtype=float)
    if signed >= 0.0:
        return bounded_shift_toward_minus_z(src, signed*dz, dz)
    alpha = -signed
    out = np.empty_like(src)
    out[0] = src[0]
    out[1:] = (1.0-alpha)*src[1:] + alpha*src[:-1]
    return out


def signed_one_contact_frame_state(
        f, ownership, z, r_c, *, dr, dz, approach_m, gb_z_m, tj_r_m,
        width_m, moving_grain, neighbor_grain, source_support=None):
    """Translate only the +z rigid frame while all other frames remain fixed.

    Positive ``approach_m`` moves the selected +z body toward ``-z``.  Negative
    values provide the discarded separation state used to test whether a
    two-sided instantaneous virtual-work derivative exists.  The solid-body split is geometric and is not
    inferred from material centroids.
    """
    f0 = np.asarray(f, dtype=float)
    phi0 = np.asarray(ownership, dtype=float)
    if phi0.ndim != 3 or phi0.shape[1:] != f0.shape:
        raise ValueError("ownership must have shape (ngrains, nz, nr)")
    if moving_grain == neighbor_grain:
        raise ValueError("moving and neighbor grains must differ")
    if np.max(np.abs(phi0.sum(axis=0)-1.0)) > 2e-12:
        raise ValueError("reference normalized ownership does not close to one")
    u = float(approach_m)
    if abs(u) > float(dz):
        raise ValueError("signed one-contact coordinate is restricted to one subcell")
    if u == 0.0:
        return RigidFrameState(
            f=f0.copy(), ownership=phi0.copy(), approach_m=0.0,
            moving_frame_translation_z_m=0.0,
            ownership_plane_translation_z_m=0.0, swept_volume_m3=0.0,
            mass_relative_error=0.0, partition_residual=float(np.max(np.abs(phi0.sum(axis=0)-1.0))),
            f_bounds=(float(f0.min()), float(f0.max())), source_max_increment=0.0)

    moving_side = np.asarray(z)[:, None] >= float(gb_z_m)
    body_moving = np.where(moving_side, f0, 0.0)
    body_stationary = f0-body_moving
    moved = conservative_signed_shift_toward_minus_z(body_moving, u, dz)
    overlap = np.minimum(np.maximum(moved, 0.0), np.maximum(body_stationary, 0.0))
    f_union = moved+body_stationary-overlap
    swept_weighted = _weighted_sum(f0, r_c)-_weighted_sum(f_union, r_c)
    roundoff = 2e-13*max(abs(_weighted_sum(f0, r_c)), 1e-300)
    if swept_weighted < -roundoff:
        raise RuntimeError("translated body union gained material")
    swept_weighted = max(0.0, swept_weighted)

    pair0 = phi0[moving_grain]+phi0[neighbor_grain]
    chi0 = np.divide(phi0[moving_grain], pair0,
                     out=np.zeros_like(f0), where=pair0 > 1e-30)
    chi = bounded_signed_shift_toward_minus_z(chi0, 0.5*u, dz)
    other = phi0.sum(axis=0)-pair0
    pair = 1.0-other
    phi = phi0.copy()
    phi[moving_grain] = pair*chi
    phi[neighbor_grain] = pair-phi[moving_grain]
    if np.min(phi) < -2e-14 or np.max(phi) > 1.0+2e-14:
        raise RuntimeError("translated ownership left the simplex")

    if source_support is None:
        source_support = triple_junction_support(
            f0, z, r_c, z_tj=gb_z_m, r_tj=tj_r_m, width=width_m)
    support = np.asarray(source_support, dtype=float)
    if support.shape != f0.shape:
        raise ValueError("source support shape differs from the state")
    if not np.isclose(_weighted_sum(support, r_c), 1.0, rtol=2e-13, atol=2e-15):
        raise ValueError("source support is not axisymmetrically normalized")
    added = swept_weighted*support
    f1 = f_union+added
    if np.min(f1) < -2e-12 or np.max(f1) > 1.0+2e-12:
        raise RuntimeError("signed rigid-frame state exceeds total-solid bounds")
    eta = phi*f1[None, :]
    phi1 = np.divide(eta, f1[None, :], out=phi.copy(),
                     where=np.abs(f1[None, :]) > 1e-30)
    v0 = axisymmetric_volume(f0, r_c, dr, dz)
    v1 = axisymmetric_volume(f1, r_c, dr, dz)
    return RigidFrameState(
        f=f1, ownership=phi1, approach_m=u,
        moving_frame_translation_z_m=-u,
        ownership_plane_translation_z_m=-0.5*u,
        swept_volume_m3=2.0*math.pi*float(dr)*float(dz)*swept_weighted,
        mass_relative_error=abs(v1-v0)/max(abs(v0), 1e-300),
        partition_residual=float(np.max(np.abs(phi1.sum(axis=0)-1.0))),
        f_bounds=(float(np.min(f1)), float(np.max(f1))),
        source_max_increment=float(np.max(added)))


def positive_one_contact_direction_components(
        f, ownership, z, r_c, *, dr, dz, delta_u_m, gb_z_m, tj_r_m,
        width_m, moving_grain, neighbor_grain, source_support=None):
    """Return the literal ``0+`` event direction and its material split.

    The total-solid increment is decomposed exactly into the translated-body
    union (``swept``) and the conservative source addition (``deposit``).
    This routine calls the same positive rigid-frame constructor used for the
    discarded event state.  It introduces no negative/contact-opening path.
    """
    du = float(delta_u_m)
    if du <= 0.0:
        raise ValueError("one-sided directional step must be positive")
    f0 = np.asarray(f, dtype=float)
    phi0 = np.asarray(ownership, dtype=float)
    state = signed_one_contact_frame_state(
        f0, phi0, z, r_c, dr=dr, dz=dz, approach_m=du,
        gb_z_m=gb_z_m, tj_r_m=tj_r_m, width_m=width_m,
        moving_grain=moving_grain, neighbor_grain=neighbor_grain,
        source_support=source_support)
    moving_side = np.asarray(z)[:, None] >= float(gb_z_m)
    body_moving = np.where(moving_side, f0, 0.0)
    body_stationary = f0-body_moving
    moved = conservative_signed_shift_toward_minus_z(body_moving, du, dz)
    overlap = np.minimum(np.maximum(moved, 0.0),
                         np.maximum(body_stationary, 0.0))
    f_union = moved+body_stationary-overlap
    f_swept_u = (f_union-f0)/du
    f_deposit_u = (state.f-f_union)/du
    if not np.allclose(
            f_swept_u+f_deposit_u, (state.f-f0)/du,
            rtol=2e-13, atol=1e-18/max(du, 1e-300)):
        raise RuntimeError("directional sweep/deposit decomposition failed")
    return OneSidedDirectionalComponents(
        state=state, f_swept_u=f_swept_u,
        f_deposit_u=f_deposit_u,
        ownership_u=(state.ownership-phi0)/du,
        swept_volume_rate_m2=state.swept_volume_m3/du)
