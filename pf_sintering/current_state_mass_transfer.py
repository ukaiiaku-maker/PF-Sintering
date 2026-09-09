"""Current-state conservative material-transfer event primitives.

This module deliberately does not contain a prescribed ``q -> geometry``
map.  ``q`` records only the amount delivered by the GB transport law.  Each
increment rebuilds its donor and receiver supports from the current phase
fields and moves material with a bounded source/sink update under the exact
axisymmetric volume measure.

The source update has the form

``f' = f + lambda_r w_r (1-f) - lambda_d w_d f``.

Receiver and donor supports are disjoint.  The two multipliers are obtained
from their exact discrete capacities, so neither clipping nor a posteriori
mass repair is used.  The current normalized ownership fraction is retained
pointwise when reconstructing the two grain fields.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TransferMaskParameters:
    """Physical support lengths for one current-state transfer increment."""

    receiver_sigma_over_W: float = 1.25
    receiver_cutoff_over_W: float = 3.0
    donor_center_over_W: float = 6.0
    donor_sigma_over_W: float = 1.5
    donor_inner_cutoff_over_W: float = 3.0
    donor_outer_cutoff_over_W: float = 10.0


def axisymmetric_weights(r_c, dr: float, dz: float) -> np.ndarray:
    """Return cell volumes for the axisymmetric ``(z,r)`` grid."""
    radii = np.asarray(r_c, dtype=float)
    if radii.ndim != 1 or np.any(radii <= 0.0):
        raise ValueError("r_c must be a positive one-dimensional array")
    if dr <= 0.0 or dz <= 0.0:
        raise ValueError("dr and dz must be positive")
    return 2.0 * math.pi * radii[None, :] * float(dr) * float(dz)


def axisymmetric_integral(field, weights) -> float:
    return float(np.sum(
        np.asarray(field, dtype=np.longdouble)
        * np.asarray(weights, dtype=np.longdouble), dtype=np.longdouble))


def build_current_state_masks(
        f, z, r_c, *, z_TJ_m: float, r_TJ_m: float, W_m: float,
        parameters: TransferMaskParameters = TransferMaskParameters()):
    """Build disjoint smooth receiver/donor masks from the current interface.

    Near the TJ, Euclidean distance is an adequate smooth proxy for branch arc
    length and avoids fitting through the junction.  Both supports are gated
    by the current diffuse free surface ``4 f (1-f)``.  The receiver spans the
    first ``3W`` of both branches; the donor is the same two current branches
    over ``3W--10W``.  The masks are recomputed after every accepted state.
    """
    f0 = np.asarray(f, dtype=float)
    if f0.ndim != 2 or f0.shape != (len(z), len(r_c)):
        raise ValueError("f shape must match z and r_c")
    if W_m <= 0.0 or r_TJ_m <= 0.0:
        raise ValueError("W and r_TJ must be positive")
    p = parameters
    lengths = (
        p.receiver_sigma_over_W, p.receiver_cutoff_over_W,
        p.donor_center_over_W, p.donor_sigma_over_W,
        p.donor_inner_cutoff_over_W, p.donor_outer_cutoff_over_W)
    if any(not math.isfinite(x) or x <= 0.0 for x in lengths):
        raise ValueError("all mask length parameters must be positive")
    if not (p.donor_inner_cutoff_over_W
            < p.donor_center_over_W
            < p.donor_outer_cutoff_over_W):
        raise ValueError("donor center must lie within donor cutoffs")

    Z = np.asarray(z, dtype=float)[:, None]
    R = np.asarray(r_c, dtype=float)[None, :]
    distance = np.hypot(Z - float(z_TJ_m), R - float(r_TJ_m))
    distance_over_W = distance / float(W_m)
    bounded = np.clip(f0, 0.0, 1.0)
    interface = 4.0 * bounded * (1.0 - bounded)

    receiver = interface * np.exp(
        -0.5 * (distance_over_W / p.receiver_sigma_over_W) ** 2)
    receiver *= distance_over_W <= p.receiver_cutoff_over_W

    donor = interface * np.exp(
        -0.5 * ((distance_over_W - p.donor_center_over_W)
                / p.donor_sigma_over_W) ** 2)
    donor *= ((distance_over_W >= p.donor_inner_cutoff_over_W)
              & (distance_over_W <= p.donor_outer_cutoff_over_W))

    # Make disjointness exact even at the shared cutoff.  This makes the
    # analytic capacity calculation below exact rather than iterative.
    overlap = (receiver > 0.0) & (donor > 0.0)
    donor[overlap] = 0.0
    if not np.any(receiver > 0.0) or not np.any(donor > 0.0):
        raise RuntimeError("current geometry has no donor or receiver support")
    return receiver, donor, dict(
        mask_geometry_source="current accepted phase field and field TJ",
        interface_gate="4*f*(1-f)",
        distance_coordinate="Euclidean near-TJ proxy for branch arc length",
        receiver_extent_over_W=p.receiver_cutoff_over_W,
        donor_extent_over_W=(p.donor_inner_cutoff_over_W,
                             p.donor_outer_cutoff_over_W),
        supports_disjoint=not bool(np.any((receiver > 0.0) & (donor > 0.0))))


def bounded_conservative_transfer(
        state, receiver_mask, donor_mask, *, transfer_volume_m3: float,
        r_c, dr: float, dz: float, relative_tolerance: float = 2.0e-13):
    """Transfer an exact volume without clipping and preserve ownership.

    The receiver uses its local vacancy capacity and the donor its local solid
    capacity.  A step is rejected if either requested multiplier exceeds one;
    callers may then reduce ``Delta q``.  The returned fields satisfy
    ``eta1+eta2=f`` algebraically.
    """
    f, eta1, eta2 = (np.asarray(x, dtype=float) for x in state)
    receiver = np.maximum(np.asarray(receiver_mask, dtype=float), 0.0)
    donor = np.maximum(np.asarray(donor_mask, dtype=float), 0.0)
    if not (f.shape == eta1.shape == eta2.shape == receiver.shape == donor.shape):
        raise ValueError("state and masks must have identical shapes")
    requested = float(transfer_volume_m3)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("transfer_volume_m3 must be positive and finite")
    if np.any((receiver > 0.0) & (donor > 0.0)):
        raise ValueError("receiver and donor supports must be disjoint")
    weights = axisymmetric_weights(r_c, dr, dz)
    bounded = np.clip(f, 0.0, 1.0)
    receiver_basis = receiver * (1.0 - bounded)
    donor_basis = donor * bounded
    receiver_capacity = axisymmetric_integral(receiver_basis, weights)
    donor_capacity = axisymmetric_integral(donor_basis, weights)
    if receiver_capacity <= 0.0 or donor_capacity <= 0.0:
        raise RuntimeError("bounded transfer support has zero capacity")
    lambda_receiver = requested / receiver_capacity
    lambda_donor = requested / donor_capacity
    if lambda_receiver > 1.0 + 1.0e-14 or lambda_donor > 1.0 + 1.0e-14:
        raise RuntimeError(
            "requested transfer exceeds current support capacity: "
            f"lambda_receiver={lambda_receiver:.6g}, "
            f"lambda_donor={lambda_donor:.6g}")

    addition = lambda_receiver * receiver_basis
    removal = lambda_donor * donor_basis
    f_new = f + addition - removal
    bound_error = max(float(-np.min(f_new)), float(np.max(f_new)-1.0), 0.0)
    if bound_error > 2.0e-14:
        raise RuntimeError(f"bounded source update violated phase bounds: {bound_error}")

    owner_sum = eta1 + eta2
    phi = np.divide(
        eta1, owner_sum, out=np.full_like(f, 0.5),
        where=np.abs(owner_sum) > 1.0e-30)
    phi = np.clip(phi, 0.0, 1.0)
    eta1_new = f_new * phi
    eta2_new = f_new * (1.0 - phi)

    added = axisymmetric_integral(addition, weights)
    removed = axisymmetric_integral(removal, weights)
    # ``added-removed`` is the conservative source operator's discrete
    # balance.  Summing ``f_new-f`` after storing f_new in float64 suffers
    # cancellation against O(1) phase values when a transfer is O(1e-6) of a
    # cell.  Retain that latter quantity as a representation diagnostic, but
    # gate conservation on the source balance and on total solid volume.
    net = added - removed
    realized_net = axisymmetric_integral(f_new-f, weights)
    volume_before = axisymmetric_integral(f, weights)
    volume_after = axisymmetric_integral(f_new, weights)
    scale = max(abs(requested), 1.0e-300)
    closure = max(abs(added-requested), abs(removed-requested), abs(net)) / scale
    total_volume_relative_error = abs(volume_after-volume_before) / max(
        abs(volume_before), 1.0e-300)
    if closure > relative_tolerance or total_volume_relative_error > 2.0e-14:
        raise RuntimeError(
            "current-state transfer mass closure failed: "
            f"source={closure:.6e}, total={total_volume_relative_error:.6e}")
    partition = float(np.max(np.abs(f_new-eta1_new-eta2_new)))
    if partition > 5.0e-15:
        raise RuntimeError(
            f"current-state transfer ownership closure failed: {partition:.6e}")
    return (f_new, eta1_new, eta2_new), dict(
        requested_transfer_volume_m3=requested,
        added_volume_m3=added,
        removed_volume_m3=removed,
        net_volume_change_m3=net,
        realized_net_volume_change_m3=realized_net,
        total_volume_relative_error=total_volume_relative_error,
        relative_mass_closure=closure,
        lambda_receiver=lambda_receiver,
        lambda_donor=lambda_donor,
        f_min=float(np.min(f_new)),
        f_max=float(np.max(f_new)),
        partition_closure=partition,
        live_state_clipped=False,
        immutable_parent_geometry_used=False,
        ownership_rule="preserve current normalized ownership pointwise")


def transport_clock_increment(
        transfer_volume_m3: float, volume_rate_start_m3_per_model_time: float,
        volume_rate_end_m3_per_model_time: float) -> float:
    """Trapezoidal reciprocal-rate time for one accepted transfer."""
    volume = float(transfer_volume_m3)
    rate0 = float(volume_rate_start_m3_per_model_time)
    rate1 = float(volume_rate_end_m3_per_model_time)
    if volume <= 0.0 or rate0 <= 0.0 or rate1 <= 0.0:
        raise ValueError("positive transfer volume and endpoint rates are required")
    return 0.5 * volume * (1.0/rate0 + 1.0/rate1)
