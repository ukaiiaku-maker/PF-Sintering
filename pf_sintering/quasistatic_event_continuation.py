"""Utilities for event-driven quasi-static continuation in the GB coordinate.

The phase-field P--R amplitude is a slow physical coordinate and is therefore
not globally minimized away.  These helpers identify convergence of the fast
local surface/TJ response and integrate the slow GB clock along the resulting
``q`` continuation path.
"""
from __future__ import annotations

import math

import numpy as np

from .axisym_branch_boundary_flux import (
    apply_axisymmetric_branch_boundary_flux_step,
)
from .model_time_pr_event import _node_state
from .rigid_rbm_deposition import representation_corrected_union_transfer


def default_q_schedule() -> np.ndarray:
    """Return the requested coarse continuation schedule on ``0 <= q/b <= 1``."""
    tail = np.round(np.arange(0.04, 1.0000001, 0.02), 12)
    return np.unique(np.round(np.r_[0.0, 0.005, 0.01, 0.02, tail, 1.0], 12))


def fast_manifold_increment(previous: dict, current: dict) -> dict:
    """Return absolute changes in the diagnostics used by the fast-state gate."""
    return {
        "affinity_MPa": abs(
            float(current["transport_affinity_Pa"])
            - float(previous["transport_affinity_Pa"])) * 1.0e-6,
        "sigma_local_MPa": abs(
            float(current["sigma_local_Pa"])
            - float(previous["sigma_local_Pa"])) * 1.0e-6,
        "sigma_integral_MPa": abs(
            float(current["sigma_integral_Pa"])
            - float(previous["sigma_integral_Pa"])) * 1.0e-6,
        "A1_nm": abs(
            float(current["A1_magnitude_m"])
            - float(previous["A1_magnitude_m"])) * 1.0e9,
        "neck_nm": abs(
            float(current["r_neck_smooth_m"])
            - float(previous["r_neck_smooth_m"])) * 1.0e9,
        "energy_relative": abs(
            float(current["G_phasefield_J"])
            - float(previous["G_phasefield_J"])) / max(
                abs(float(previous["G_phasefield_J"])), 1.0e-300),
    }


def fast_manifold_converged(increment: dict, tolerances: dict) -> bool:
    """Whether every monitored fast-coordinate increment is within tolerance."""
    missing = set(tolerances) - set(increment)
    if missing:
        raise ValueError(f"increment omitted convergence keys: {sorted(missing)}")
    return all(
        math.isfinite(float(increment[key]))
        and float(increment[key]) <= float(limit)
        for key, limit in tolerances.items())


def integrate_slow_clock(q_over_b, affinity_pa, transport):
    """Integrate ``dt=dq/qdot`` using a trapezoid in reciprocal rate.

    A nonpositive affinity makes the clock infinite at that point and all
    subsequent cumulative times remain infinite.  Both model time and the
    interval contributions are returned.
    """
    q = np.asarray(q_over_b, dtype=float)
    affinity = np.asarray(affinity_pa, dtype=float)
    if q.ndim != 1 or affinity.shape != q.shape or len(q) == 0:
        raise ValueError("q and affinity must be nonempty one-dimensional arrays")
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(affinity)):
        raise ValueError("q and affinity must be finite")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q must be strictly increasing")
    cumulative = np.zeros_like(q)
    intervals = np.zeros_like(q)
    blocked = False
    for index in range(1, len(q)):
        if blocked or affinity[index - 1] <= 0.0 or affinity[index] <= 0.0:
            intervals[index] = math.inf
            cumulative[index:] = math.inf
            blocked = True
            continue
        rate0 = float(transport.qdot_m_per_model_time(affinity[index - 1]))
        rate1 = float(transport.qdot_m_per_model_time(affinity[index]))
        if rate0 <= 0.0 or rate1 <= 0.0:
            intervals[index] = math.inf
            cumulative[index:] = math.inf
            blocked = True
            continue
        dq_m = (q[index] - q[index - 1]) * float(transport.b_m)
        intervals[index] = 0.5 * dq_m * (1.0 / rate0 + 1.0 / rate1)
        cumulative[index] = cumulative[index - 1] + intervals[index]
    return cumulative, intervals


def direct_prescribed_q_trial(
        state, *, event_base_state, union_previous,
        cumulative_source_weighted: float, q_start_over_b: float,
        q_end_over_b: float, setup: dict, geom: dict, state_evaluator,
        transport, node_surface_mobility_m6_per_J_model_time: float,
        positive_branch_grain: int = 1):
    """Construct one trial continuation state without a slow-time packet loop.

    The rigid body union is evaluated directly at ``q_end`` from the immutable
    parent-event state.  Its newly exposed source volume is deposited through
    the qualified branch-normal phase-coordinate map.  Only the zero-storage
    node's *partition* is retained: no physical slow time and no interior
    surface-diffusion packet are advanced here.  The caller subsequently
    relaxes the fast local PF/TJ coordinates at fixed q and may accept or roll
    back this entirely self-contained trial.
    """
    q0 = float(q_start_over_b)
    q1 = float(q_end_over_b)
    if not (0.0 <= q0 < q1 <= 1.0 + 1.0e-14):
        raise ValueError("trial q interval must satisfy 0 <= q0 < q1 <= 1")
    fields = tuple(np.asarray(field, dtype=float) for field in state)
    base = tuple(np.asarray(field, dtype=float) for field in event_base_state)
    previous_union = tuple(
        np.asarray(field, dtype=float) for field in union_previous)
    before = state_evaluator(*fields)
    branches, node, coordinates = _node_state(
        before, fields[0], setup["r_c"], setup["z"], transport,
        node_surface_mobility_m6_per_J_model_time, active=True)
    affinity = float(node["transport_affinity_Pa"])
    if not math.isfinite(affinity) or affinity <= 0.0:
        raise RuntimeError("direct q trial requires positive transport affinity")

    q_end_m = min(q1, 1.0) * float(transport.b_m)
    transfer = representation_corrected_union_transfer(
        *base, q_end_m, setup["dz"], setup["r_c"], setup["z"],
        geom["z1"])
    union_next = tuple(np.asarray(field, dtype=float) for field in transfer[:3])
    source_next = float(transfer[3]["V_source_weighted"])
    source_increment_weighted = source_next - float(cumulative_source_weighted)
    factor = 2.0 * math.pi * float(setup["dr"]) * float(setup["dz"])
    source_increment_m3 = source_increment_weighted * factor
    if source_increment_m3 <= 0.0:
        raise RuntimeError("cumulative body-union source did not increase")

    state_after_union = tuple(
        current + next_union - old_union
        for current, next_union, old_union
        in zip(fields, union_next, previous_union))
    volume_rate = float(node["Vdot_GB_m3_per_model_time"])
    if volume_rate <= 0.0:
        raise RuntimeError("positive affinity produced no GB delivery rate")
    branch_rates = {
        side: source_increment_m3 * float(
            node["branch_volume_rates_m3_per_model_time"][side]) / volume_rate
        for side in ("positive", "negative")}
    # A unit pseudo-step makes each branch rate numerically equal to its trial
    # volume.  Mobility is zero because slow physical time is integrated only
    # after fast-manifold acceptance; the following map realizes the boundary
    # dose and does not hide the rejected packet-resolved event underneath.
    attached = apply_axisymmetric_branch_boundary_flux_step(
        *state_after_union, setup["r_c"], setup["dr"], setup["dz"],
        setup["W"], branches,
        surface_flux_mobility_m6_per_J_model_time=0.0,
        incoming_volume_rate_m3_per_model_time=source_increment_m3,
        dt_model=1.0, positive_branch_grain=positive_branch_grain,
        branch_volume_rates_m3_per_model_time=branch_rates,
        # The tanh-row inversion closes each tiny dose to O(1e-6) relative;
        # the subsequent conservative projector enforces the global event
        # invariant.  This is the same qualified mapping tolerance as the
        # production packet event, not a relaxation of the final mass gate.
        mass_closure_relative_tolerance=1.0e-5,
        normal_displacement_diffusion_B_m4_per_model_time=None)
    candidate = tuple(np.asarray(field, dtype=float) for field in attached[:3])
    return candidate, dict(
        q_start_over_b=q0, q_end_over_b=q1,
        affinity_start_Pa=affinity,
        source_increment_weighted=source_increment_weighted,
        source_increment_m3=source_increment_m3,
        source_cumulative_weighted=source_next,
        union_next=union_next,
        branch_partition_over_net={
            side: branch_rates[side] / source_increment_m3
            for side in ("positive", "negative")},
        node_zero_storage_closure_relative=float(
            node["zero_storage_closure_relative"]),
        boundary_flux=attached[3],
        coordinates_start=coordinates,
        slow_physical_time_advanced=False,
        interior_packet_surface_diffusion_advanced=False,
        state_committed=False)
