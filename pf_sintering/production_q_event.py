"""Restartable production event propagator on the prescribed GB coordinate.

The propagator advances an immutable-parent representation family in adaptive
q increments, relaxes only the configured fast local manifold, and integrates
the physical GB clock on accepted states.  It never calls the packet-resolved
slow-time event and rejected trials have no externally visible side effects.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .quasistatic_event_continuation import direct_prescribed_q_trial
from .rigid_rbm_deposition import (
    _axisym_weighted_sum,
    representation_corrected_union_transfer,
)


DEFAULT_STEP_LIMITS = dict(
    transport_affinity_MPa=0.50,
    sigma_local_MPa=0.050,
    sigma_integral_MPa=0.100,
    r_neck_nm=0.050,
    r_TJ_nm=0.050,
    A1_nm=0.010,
    z_TJ_nm=0.050,
    kappa1_negative_per_m=5.0e4,
    kappa1_positive_per_m=5.0e4,
    reciprocal_qdot_fraction=0.02,
)


class FixedQFastManifoldConvergenceError(RuntimeError):
    """The fast coordinates failed at fixed q; dq refinement cannot fix it."""


def _changes(previous, current, transport):
    out = {}
    for key in (
            "transport_affinity_MPa", "sigma_local_MPa",
            "sigma_integral_MPa", "r_neck_nm", "r_TJ_nm", "A1_nm"):
        out[key] = abs(float(current[key]) - float(previous[key]))
    out["z_TJ_nm"] = abs(
        float(current["z_TJ_m"])-float(previous["z_TJ_m"])) * 1.0e9
    for key in ("kappa1_negative_per_m", "kappa1_positive_per_m"):
        out[key] = abs(float(current[key]) - float(previous[key]))
    rate0 = transport.qdot_m_per_model_time(
        float(previous["transport_affinity_Pa"]))
    rate1 = transport.qdot_m_per_model_time(
        float(current["transport_affinity_Pa"]))
    # A nonpositive-affinity trial legitimately has zero transport rate and
    # must be rejected by the adaptive controller.  Treat its reciprocal-rate
    # change as infinite instead of raising before the rollback/step-halving
    # logic can inspect it.
    if rate1 == 0.0:
        out["reciprocal_qdot_fraction"] = (
            0.0 if rate0 == 0.0 else math.inf)
    else:
        out["reciprocal_qdot_fraction"] = abs(rate0/rate1-1.0)
    return out


def _precondition_restarted_fast_manifold(
        state, q_over_b, *, fast_relax_fn: Callable,
        state_metrics_fn: Callable, transport, reciprocal_fraction_target: float,
        required_consecutive: int = 3, maximum_calls: int = 256,
        allow_nonpositive_initial: bool = False):
    """Settle fast coordinates at fixed ``q`` without advancing slow time.

    A checkpoint is written only every few accepted continuation states.  Near
    a small positive transport affinity, the residual change made by another
    nominal fast-manifold solve can exceed the much tighter *relative* qdot
    quadrature gate even as ``dq -> 0``.  Halving ``dq`` cannot resolve that
    mismatch.  This routine converges the already-defined fast manifold at the
    checkpoint's fixed reaction coordinate before q-continuation resumes.

    No slow/event time, source volume, RNG state, or rigid displacement is
    advanced here.  The returned state is simply a more tightly converged
    representation of the same q state.
    """
    target = float(reciprocal_fraction_target)
    if not (math.isfinite(target) and target > 0.0):
        raise ValueError("restart precondition target must be positive")
    required = int(required_consecutive)
    limit = int(maximum_calls)
    if required < 1 or limit < required:
        raise ValueError("invalid restart precondition iteration limits")

    current = tuple(np.asarray(field).copy() for field in state)
    previous = state_metrics_fn(current, q_over_b)
    initial_affinity = float(previous["transport_affinity_Pa"])
    if initial_affinity <= 0.0 and not allow_nonpositive_initial:
        return current, previous, dict(
            enabled=True, converged=False, calls=0,
            total_fast_relax_iterations=0,
            total_fast_relax_blocks=0,
            required_consecutive=required,
            reciprocal_fraction_target=target,
            initial_affinity_Pa=initial_affinity,
            final_affinity_Pa=initial_affinity,
            last_reciprocal_qdot_fraction=math.inf,
            maximum_reciprocal_qdot_fraction=math.inf,
            skipped_reason="nonpositive transport affinity")

    # Judge the *whole* final convergence window, not each call in
    # isolation.  Three individually small same-sign changes can otherwise
    # pass while their cumulative reciprocal-rate drift is still large
    # enough to contaminate the slow-clock endpoint.  Keep the initial rate
    # in the window so ``required_consecutive`` means that many complete
    # fast-relax calls at fixed q.
    rate_window = []
    initial_rate = transport.qdot_m_per_model_time(initial_affinity)
    if initial_rate > 0.0:
        rate_window.append(float(initial_rate))
    maximum_fraction = 0.0
    last_fraction = math.inf
    last_single_call_fraction = math.inf
    total_fast_relax_iterations = 0
    total_fast_relax_blocks = 0
    for calls in range(1, limit + 1):
        candidate, row, fast = fast_relax_fn(current, q_over_b)
        total_fast_relax_iterations += int(fast.get("iterations", 0))
        total_fast_relax_blocks += int(fast.get("blocks", 0))
        if not bool(fast["converged"]):
            raise FixedQFastManifoldConvergenceError(
                "restart fast-manifold preconditioner did not converge on "
                f"call {calls}")
        changes = _changes(previous, row, transport)
        last_single_call_fraction = float(
            changes["reciprocal_qdot_fraction"])
        current = tuple(np.asarray(field).copy() for field in candidate)
        previous = row
        current_rate = transport.qdot_m_per_model_time(
            float(previous["transport_affinity_Pa"]))
        if current_rate > 0.0:
            rate_window.append(float(current_rate))
        else:
            rate_window.clear()
        if len(rate_window) > required + 1:
            rate_window.pop(0)
        if len(rate_window) == required + 1:
            last_fraction = abs(rate_window[0] / rate_window[-1] - 1.0)
            maximum_fraction = max(maximum_fraction, last_fraction)
        else:
            last_fraction = math.inf
        if (len(rate_window) == required + 1
                and math.isfinite(last_fraction) and last_fraction <= target):
            return current, previous, dict(
                enabled=True, converged=True, calls=calls,
                total_fast_relax_iterations=total_fast_relax_iterations,
                total_fast_relax_blocks=total_fast_relax_blocks,
                required_consecutive=required,
                reciprocal_fraction_target=target,
                initial_affinity_Pa=initial_affinity,
                final_affinity_Pa=float(row["transport_affinity_Pa"]),
                last_reciprocal_qdot_fraction=last_fraction,
                last_single_call_reciprocal_qdot_fraction=(
                    last_single_call_fraction),
                convergence_window_calls=required,
                maximum_reciprocal_qdot_fraction=maximum_fraction,
                skipped_reason=None)
    raise FixedQFastManifoldConvergenceError(
        "fixed_q_fast_manifold reciprocal_qdot_fraction failed to reach "
        f"{target:.6g} in {limit} calls; last={last_fraction:.6g}")


def prescribed_q_event(
        state, setup, geom, state_evaluator, transport, quota_fraction_b,
        *, fast_relax_fn: Callable, state_metrics_fn: Callable,
        node_surface_mobility_m6_per_J_model_time: float,
        event_restart=None, accepted_state_callback=None,
        accepted_progress_callback=None,
        initial_step_over_b: float = 0.02,
        minimum_step_over_b: float = 0.00125,
        maximum_step_over_b: float = 0.02,
        precondition_restart_fast_manifold: bool = False,
        restart_precondition_fraction_of_qdot_limit: float = 0.5,
        restart_precondition_required_consecutive: int = 3,
        restart_precondition_maximum_calls: int = 256,
        step_limits=None, **_ignored_packet_options):
    """Advance to ``quota_fraction_b`` with exactly-once restart semantics."""
    target_quota = float(quota_fraction_b)
    if not 0.0 < target_quota <= 1.0 + 1.0e-14:
        raise ValueError("event quota must lie in (0,1]")
    limits = DEFAULT_STEP_LIMITS if step_limits is None else dict(step_limits)
    if event_restart is None:
        base = tuple(np.asarray(field).copy() for field in state)
        current = tuple(np.asarray(field).copy() for field in state)
        q = 0.0
        elapsed_model = 0.0
        accepted_total = 0
        trial_rejections_total = 0
        union0 = representation_corrected_union_transfer(
            *base, 0.0, setup["dz"], setup["r_c"], setup["z"], geom["z1"])
        mapped_previous = tuple(np.asarray(field).copy() for field in union0[:3])
        cumulative_source = float(union0[3]["V_source_weighted"])
        mass_initial = _axisym_weighted_sum(base[0], setup["r_c"])
        previous = state_metrics_fn(current, q)
    else:
        base = tuple(np.asarray(field).copy()
                     for field in event_restart["base_fields"])
        current = tuple(np.asarray(field).copy() for field in state)
        mapped_previous = tuple(np.asarray(field).copy()
                                for field in event_restart["union_previous_fields"])
        q = float(event_restart["cumulative_q_m"])/float(transport.b_m)
        elapsed_model = float(event_restart["event_time_model"])
        accepted_total = int(event_restart["accepted_steps_total"])
        trial_rejections_total = int(event_restart.get(
            "quasistatic_trial_rejections_total", 0))
        quadrature_refinements_total = int(event_restart.get(
            "slow_clock_quadrature_refinements_total", 0))
        cumulative_source = float(event_restart["cumulative_source_weighted"])
        mass_initial = float(event_restart["mass_initial_weighted"])
        previous = state_metrics_fn(current, q)
    if event_restart is None:
        quadrature_refinements_total = 0
    precondition_calls_before = int(
        0 if event_restart is None else event_restart.get(
            "restart_fast_precondition_calls_total", 0))
    precondition = dict(
        enabled=False, converged=True, calls=0,
        required_consecutive=int(restart_precondition_required_consecutive),
        reciprocal_fraction_target=(
            float(limits["reciprocal_qdot_fraction"])
            * float(restart_precondition_fraction_of_qdot_limit)),
        initial_affinity_Pa=float(previous["transport_affinity_Pa"]),
        final_affinity_Pa=float(previous["transport_affinity_Pa"]),
        last_reciprocal_qdot_fraction=0.0,
        maximum_reciprocal_qdot_fraction=0.0,
        total_fast_relax_iterations=0,
        total_fast_relax_blocks=0,
        skipped_reason="disabled")
    if event_restart is not None and precondition_restart_fast_manifold:
        current, previous, precondition = _precondition_restarted_fast_manifold(
            current, q, fast_relax_fn=fast_relax_fn,
            state_metrics_fn=state_metrics_fn, transport=transport,
            reciprocal_fraction_target=(
                float(limits["reciprocal_qdot_fraction"])
                * float(restart_precondition_fraction_of_qdot_limit)),
            required_consecutive=restart_precondition_required_consecutive,
            maximum_calls=restart_precondition_maximum_calls)
        # Reconstruct the auxiliary absolute mapped reference at the exact
        # checkpoint q.  Older recovery checkpoints carried a reference that
        # could differ from a fresh immutable-parent map even though its
        # cumulative source scalar was correct; subtracting that stale field
        # made an infinitesimal next dose look like an O(1) interface jump.
        # Rebasing this auxiliary field does not change the physical current
        # state, q, event time, source volume, or stochastic bookkeeping.
        if q > 0.0:
            zero_union = representation_corrected_union_transfer(
                *base, 0.0, setup["dz"], setup["r_c"], setup["z"],
                geom["z1"])
            mapped_rebased, rebase_mapping = direct_prescribed_q_trial(
                base, event_base_state=base,
                union_previous=zero_union[:3],
                cumulative_source_weighted=float(
                    zero_union[3]["V_source_weighted"]),
                q_start_over_b=0.0, q_end_over_b=q,
                setup=setup, geom=geom, state_evaluator=state_evaluator,
                transport=transport,
                node_surface_mobility_m6_per_J_model_time=(
                    node_surface_mobility_m6_per_J_model_time))
            source_rebased = float(
                rebase_mapping["source_cumulative_weighted"])
            source_scale = max(abs(cumulative_source), 1.0e-300)
            source_relative = abs(
                source_rebased - cumulative_source) / source_scale
            if source_relative > 1.0e-12:
                raise RuntimeError(
                    "restart mapped-reference rebase changed cumulative "
                    f"source volume by {source_relative:.3e}")
            precondition["mapped_reference_rebased"] = True
            precondition["mapped_reference_max_abs_correction"] = max(
                float(np.max(np.abs(old - new)))
                for old, new in zip(mapped_previous, mapped_rebased))
            precondition["mapped_reference_source_relative_difference"] = (
                source_relative)
            mapped_previous = tuple(
                np.asarray(field).copy() for field in mapped_rebased)
        else:
            precondition["mapped_reference_rebased"] = False
            precondition["mapped_reference_max_abs_correction"] = 0.0
            precondition["mapped_reference_source_relative_difference"] = 0.0
    precondition_calls_total = precondition_calls_before + int(
        precondition["calls"])
    if target_quota < q-1.0e-13:
        raise ValueError("event quota precedes committed restart q")

    active_minimum_step = float(minimum_step_over_b)
    if event_restart is not None:
        active_minimum_step = min(active_minimum_step, float(
            event_restart.get("active_minimum_step_over_b",
                              active_minimum_step)))
    step = min(maximum_step_over_b, max(initial_step_over_b,
                                        active_minimum_step))
    packets = []
    clean = 0
    rejection_reason_counts = {}
    last_rejection_reasons = []
    ordinary_fixed_q_calls_total = int(
        0 if event_restart is None else event_restart.get(
            "ordinary_fixed_q_calls_total", 0))
    # Qualify the entry endpoint once.  Thereafter every accepted candidate
    # has already passed this same fixed-q gate and is therefore the qualified
    # start endpoint of the next interval.  Re-running the fixed-q solve at
    # the top of every loop would relax each q state twice, advance fast-field
    # coordinates at zero slow time, and make the adaptive q controller react
    # to that duplicate relaxation rather than to q quadrature.
    if event_restart is not None and precondition_restart_fast_manifold:
        current_fixed_q = precondition
    else:
        current, previous, current_fixed_q = (
            _precondition_restarted_fast_manifold(
                current, q, fast_relax_fn=fast_relax_fn,
                state_metrics_fn=state_metrics_fn, transport=transport,
                reciprocal_fraction_target=(
                    float(limits["reciprocal_qdot_fraction"])
                    * float(restart_precondition_fraction_of_qdot_limit)),
                required_consecutive=restart_precondition_required_consecutive,
                maximum_calls=restart_precondition_maximum_calls,
                allow_nonpositive_initial=False))
        ordinary_fixed_q_calls_total += int(current_fixed_q["calls"])
    while q < target_quota-1.0e-14:
        q_trial = min(q+step, target_quota)
        fixed_q = None
        union0 = representation_corrected_union_transfer(
            *base, 0.0, setup["dz"], setup["r_c"], setup["z"], geom["z1"])
        try:
            mapped_next, mapping = direct_prescribed_q_trial(
                base, event_base_state=base, union_previous=union0[:3],
                cumulative_source_weighted=float(
                    union0[3]["V_source_weighted"]),
                q_start_over_b=0.0, q_end_over_b=q_trial,
                setup=setup, geom=geom, state_evaluator=state_evaluator,
                transport=transport,
                node_surface_mobility_m6_per_J_model_time=(
                    node_surface_mobility_m6_per_J_model_time))
            candidate = tuple(
                now + mapped - old_mapped
                for now, mapped, old_mapped
                in zip(current, mapped_next, mapped_previous))
            # Every slow-clock endpoint must first lie on the same converged
            # fast manifold.  A single absolute-tolerance relaxation call is
            # insufficient near small affinity because its residual drift can
            # dominate the reciprocal-qdot quadrature error even as dq -> 0.
            # Hold q exactly fixed until successive fast solves satisfy the
            # clock-compatible relative-rate criterion; only then compare the
            # two converged q endpoints and consider accepting the q step.
            candidate, row, fixed_q = _precondition_restarted_fast_manifold(
                candidate, q_trial, fast_relax_fn=fast_relax_fn,
                state_metrics_fn=state_metrics_fn, transport=transport,
                reciprocal_fraction_target=(
                    float(limits["reciprocal_qdot_fraction"])
                    * float(restart_precondition_fraction_of_qdot_limit)),
                required_consecutive=restart_precondition_required_consecutive,
                maximum_calls=restart_precondition_maximum_calls,
                allow_nonpositive_initial=True)
            ordinary_fixed_q_calls_total += int(fixed_q["calls"])
            fast = dict(
                converged=bool(fixed_q["converged"]),
                iterations=int(fixed_q["total_fast_relax_iterations"]),
                blocks=int(fixed_q.get("total_fast_relax_blocks", 0)))
            changes = _changes(previous, row, transport)
            reasons = [
                ("slow_clock_reciprocal_qdot_quadrature"
                 if key == "reciprocal_qdot_fraction" else key)
                for key, value in changes.items()
                if value > float(limits[key])]
            if not bool(fast["converged"]):
                reasons.append("fast_manifold")
            if float(row["transport_affinity_Pa"]) <= 0.0:
                reasons.append("nonpositive_affinity")
        except FixedQFastManifoldConvergenceError:
            # A q-independent fast-manifold failure must propagate to the
            # exact-checkpoint recovery path.  Treating it as a q trial error
            # would incorrectly halve dq while the endpoint is unconverged.
            raise
        except (RuntimeError, ValueError) as error:
            reasons = [f"trial_exception:{error}"]
        if reasons:
            trial_rejections_total += 1
            last_rejection_reasons = list(reasons)
            for reason in reasons:
                rejection_reason_counts[reason] = (
                    int(rejection_reason_counts.get(reason, 0)) + 1)
            clean = 0
            if q_trial-q <= active_minimum_step+1.0e-15:
                # Once both endpoints have passed fixed-q convergence, a
                # remaining reciprocal-rate failure belongs to the slow-clock
                # quadrature itself.  Refine that interval in place from the
                # unchanged committed state; do not pay for a process restart
                # or another restart-only fast-manifold solve.  No other gate
                # is permitted to lower dq here.
                quadrature_only = (
                    reasons == ["slow_clock_reciprocal_qdot_quadrature"]
                    and fixed_q is not None
                    and bool(fixed_q.get("converged", False)))
                if quadrature_only and active_minimum_step > 1.0e-12:
                    active_minimum_step *= 0.5
                    quadrature_refinements_total += 1
                    step = active_minimum_step
                    continue
                affinity_limited = "nonpositive_affinity" in reasons
                restart = dict(
                    base_fields=base, union_previous_fields=mapped_previous,
                    cumulative_q_m=q*transport.b_m,
                    cumulative_source_weighted=cumulative_source,
                    event_time_model=elapsed_model,
                    accepted_steps_total=accepted_total,
                    mass_initial_weighted=mass_initial,
                    cumulative_grain_flux_volume_m3={1: 0.0, 2: 0.0},
                    quasistatic_trial_rejections_total=trial_rejections_total,
                    restart_fast_precondition_calls_total=(
                        precondition_calls_total),
                    slow_clock_quadrature_refinements_total=(
                        quadrature_refinements_total),
                    active_minimum_step_over_b=active_minimum_step)
                return *current, False, dict(
                    completed=False, event_progress_over_b=q,
                    event_progress_m=q*transport.b_m,
                    event_time_model=elapsed_model, packets=packets,
                    event_restart=restart,
                    stop_reason=(
                        "nonpositive_transport_affinity" if affinity_limited
                        else "adaptive q trial failed at minimum step"),
                    stop_detail=str(reasons),
                    branch_time_integrator=(
                        "direct adaptive prescribed-q fast-manifold continuation"),
                    explicit_max_fourth_order_courant=None,
                    restart_fast_precondition=precondition,
                    failed_trial_fixed_q_convergence=fixed_q)
            step = max(active_minimum_step, 0.5*(q_trial-q))
            continue

        affinity0 = float(previous["transport_affinity_Pa"])
        affinity1 = float(row["transport_affinity_Pa"])
        rate0 = transport.qdot_m_per_model_time(affinity0)
        rate1 = transport.qdot_m_per_model_time(affinity1)
        interval = 0.5*(q_trial-q)*transport.b_m*(1.0/rate0+1.0/rate1)
        elapsed_model += interval
        accepted_total += 1
        packet = dict(
            q_start_over_b=q, q_end_over_b=q_trial,
            q_start_m=q*transport.b_m, q_end_m=q_trial*transport.b_m,
            dq_m=(q_trial-q)*transport.b_m,
            transport_dt_model=interval,
            accumulated_event_time_model=elapsed_model,
            affinity_start_Pa=affinity0, affinity_end_Pa=affinity1,
            qdot_start_m_per_model_time=rate0,
            qdot_end_m_per_model_time=rate1,
            minimum_affinity_Pa=min(affinity0, affinity1),
            maximum_affinity_Pa=max(affinity0, affinity1),
            continuation_step_over_b=q_trial-q,
            fast_relax_iterations=int(fast["iterations"]),
            fast_relax_blocks=int(fast.get("blocks", 0)),
            fast_relax_calls=int(fixed_q["calls"]),
            q_start_fast_relax_calls=int(current_fixed_q["calls"]),
            q_start_fast_relax_blocks=int(
                current_fixed_q.get("total_fast_relax_blocks", 0)),
            q_start_affinity_before_fixed_q_convergence_Pa=float(
                current_fixed_q["initial_affinity_Pa"]),
            q_start_affinity_after_fixed_q_convergence_Pa=float(
                current_fixed_q["final_affinity_Pa"]),
            q_start_final_reciprocal_qdot_fraction=float(
                current_fixed_q["last_reciprocal_qdot_fraction"]),
            q_over_b_before_fixed_q_relaxation=q_trial,
            affinity_before_fixed_q_convergence_Pa=float(
                fixed_q["initial_affinity_Pa"]),
            affinity_after_fixed_q_convergence_Pa=float(
                fixed_q["final_affinity_Pa"]),
            final_fast_convergence_reciprocal_qdot_fraction=float(
                fixed_q["last_reciprocal_qdot_fraction"]),
            fast_convergence_reciprocal_qdot_target=float(
                fixed_q["reciprocal_fraction_target"]),
            trial_rejections_total=trial_rejections_total,
            rejection_reason_counts=dict(rejection_reason_counts),
            last_rejection_reasons=list(last_rejection_reasons),
            **{key: value for key, value in row.items()
               if (np.isscalar(value) and not isinstance(value, str)
                   and key not in {
                       "fast_relax_iterations", "fast_relax_blocks",
                       "fast_relax_calls"})},
            **{f"accepted_change_{key}": value
               for key, value in changes.items()})
        packets.append(packet)
        current = tuple(np.asarray(field).copy() for field in candidate)
        mapped_previous = tuple(np.asarray(field).copy()
                                for field in mapped_next)
        cumulative_source = float(mapping["source_cumulative_weighted"])
        previous = row
        q = q_trial
        # This accepted candidate is already the converged q_i endpoint for
        # the next interval; carry its proof forward instead of relaxing the
        # identical state a second time.
        current_fixed_q = fixed_q
        progress_restart = dict(
            base_fields=base, union_previous_fields=mapped_previous,
            cumulative_q_m=q*transport.b_m,
            cumulative_source_weighted=cumulative_source,
            event_time_model=elapsed_model,
            accepted_steps_total=accepted_total,
            mass_initial_weighted=mass_initial,
            cumulative_grain_flux_volume_m3={1: 0.0, 2: 0.0},
            quasistatic_trial_rejections_total=trial_rejections_total)
        progress_restart["restart_fast_precondition_calls_total"] = (
            precondition_calls_total)
        progress_restart["slow_clock_quadrature_refinements_total"] = (
            quadrature_refinements_total)
        progress_restart["active_minimum_step_over_b"] = (
            active_minimum_step)
        progress_restart["ordinary_fixed_q_calls_total"] = (
            ordinary_fixed_q_calls_total)
        if accepted_progress_callback is not None:
            accepted_progress_callback(packet, current, progress_restart)
        if accepted_state_callback is not None:
            accepted_state_callback(packet, current)
        clean += 1
        if clean >= 3:
            step = min(maximum_step_over_b, 2.0*step)
            clean = 0

    restart = dict(
        base_fields=base, union_previous_fields=mapped_previous,
        cumulative_q_m=q*transport.b_m,
        cumulative_source_weighted=cumulative_source,
        event_time_model=elapsed_model,
        accepted_steps_total=accepted_total,
        mass_initial_weighted=mass_initial,
        cumulative_grain_flux_volume_m3={1: 0.0, 2: 0.0},
        quasistatic_trial_rejections_total=trial_rejections_total,
        restart_fast_precondition_calls_total=precondition_calls_total,
        slow_clock_quadrature_refinements_total=(
            quadrature_refinements_total),
        active_minimum_step_over_b=active_minimum_step)
    restart["ordinary_fixed_q_calls_total"] = ordinary_fixed_q_calls_total
    return *current, True, dict(
        completed=True, event_progress_over_b=q,
        event_progress_m=q*transport.b_m,
        event_time_model=elapsed_model,
        n_subincrements=len(packets), n_subincrements_total=accepted_total,
        packets=packets, event_restart=restart,
        stop_reason=None,
        branch_time_integrator=(
            "direct adaptive prescribed-q fast-manifold continuation"),
        explicit_max_fourth_order_courant=None,
        slow_clock_quadrature_refinements_total=(
            quadrature_refinements_total),
        active_minimum_step_over_b=active_minimum_step,
        ordinary_fixed_q_calls_total=ordinary_fixed_q_calls_total,
        restart_fast_precondition=precondition,
        kinetic_time_basis=transport.kinetic_time_basis,
        physical_seconds_conversion=None,
        ordinary_M_s_step_applied=False)
