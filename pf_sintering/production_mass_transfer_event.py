"""Restartable production event using current-state conservative transfer."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .current_state_mass_transfer import (
    TransferMaskParameters,
    bounded_conservative_transfer,
    build_current_state_masks,
    transport_clock_increment,
)
from .rigid_rbm_deposition import _axisym_weighted_sum


EVENT_INTEGRATOR_NAME = "current-state conservative material-transfer continuation"


DEFAULT_STEP_LIMITS = dict(
    transport_affinity_MPa=0.50,
    sigma_local_MPa=0.10,
    sigma_integral_continuous_MPa=0.10,
    r_neck_nm=0.05,
    r_TJ_nm=0.05,
    A1_nm=0.01,
    z_TJ_nm=0.05,
)


def _changes(previous, current):
    output = {}
    for key in (
            "transport_affinity_MPa", "sigma_local_MPa",
            "sigma_integral_continuous_MPa", "r_neck_nm", "r_TJ_nm",
            "A1_nm"):
        output[key] = abs(float(current[key])-float(previous[key]))
    output["z_TJ_nm"] = abs(
        float(current["z_TJ_m"])-float(previous["z_TJ_m"]))*1.0e9
    return output


def _rate(row, transport):
    return (float(row["contact_area_m2"])
            * transport.qdot_m_per_model_time(
                float(row["transport_affinity_Pa"])))


def current_state_mass_transfer_event(
        state, setup, geom, state_evaluator, transport, quota_fraction_b,
        *, fast_relax_fn: Callable, state_metrics_fn: Callable,
        event_restart=None, accepted_state_callback=None,
        accepted_progress_callback=None, initial_step_over_b: float = 0.02,
        minimum_step_over_b: float = 0.0025,
        maximum_step_over_b: float = 0.02, step_limits=None,
        mask_parameters: TransferMaskParameters = TransferMaskParameters(),
        maximum_accepted_states: int = 400,
        transfer_fn=bounded_conservative_transfer, **_ignored_packet_options):
    """Advance accumulated transferred material to a requested event quota.

    Every trial begins from the current committed fields.  Its masks are
    rebuilt from the current field TJ, it transfers ``A_GB(current)*Delta q``
    with exact zero net source, then it relaxes the corrected fast PF state.
    Rejected trials leave state, q, time, and transfer totals untouched.
    """
    target = float(quota_fraction_b)
    if not 0.0 < target <= 1.0 + 1.0e-14:
        raise ValueError("event quota must lie in (0,1]")
    if not 0.0 < minimum_step_over_b <= initial_step_over_b <= maximum_step_over_b:
        raise ValueError("invalid material-transfer step bounds")
    limits = DEFAULT_STEP_LIMITS if step_limits is None else dict(step_limits)
    factor = 2.0*math.pi*float(setup["dr"])*float(setup["dz"])
    current = tuple(np.asarray(field).copy() for field in state)
    if event_restart is None:
        event_base = tuple(field.copy() for field in current)
        q = 0.0
        elapsed = 0.0
        accepted_total = 0
        rejected_total = 0
        transferred_m3 = 0.0
        mass_initial = _axisym_weighted_sum(current[0], setup["r_c"])
    else:
        event_base = tuple(np.asarray(field).copy()
                           for field in event_restart["base_fields"])
        q = float(event_restart["cumulative_q_m"])/float(transport.b_m)
        elapsed = float(event_restart["event_time_model"])
        accepted_total = int(event_restart["accepted_steps_total"])
        rejected_total = int(event_restart.get(
            "quasistatic_trial_rejections_total", 0))
        transferred_m3 = float(
            event_restart["cumulative_source_weighted"])*factor
        mass_initial = float(event_restart["mass_initial_weighted"])
    if target < q-1.0e-13:
        raise ValueError("event quota precedes committed restart q")

    previous = state_metrics_fn(current, q)
    previous["contact_area_m2"] = float(
        state_evaluator(*current)["contact_area_m2"])
    previous["sigma_integral_continuous_MPa"] = float(
        previous["sigma_integral_continuous_Pa"])*1.0e-6
    step = min(maximum_step_over_b, max(initial_step_over_b,
                                        minimum_step_over_b))
    packets = []
    clean = 0
    rejection_counts = {}
    last_rejections = []
    if event_restart is not None:
        step = float(event_restart.get("next_step_over_b", step))
        step = min(maximum_step_over_b, max(minimum_step_over_b, step))
        clean = int(event_restart.get("clean_acceptances", 0))
        rejection_counts = dict(event_restart.get("rejection_reason_counts", {}))
        last_rejections = list(event_restart.get("last_rejection_reasons", []))

    def restart_record():
        return dict(
            # base_fields is provenance/serialization only.  It is never used
            # to construct a subsequent morphology.
            base_fields=event_base,
            # Compatibility slot for the common checkpoint format.  The
            # current state is stored here; no union map is evaluated.
            union_previous_fields=tuple(field.copy() for field in current),
            cumulative_q_m=q*transport.b_m,
            cumulative_source_weighted=transferred_m3/factor,
            event_time_model=elapsed,
            accepted_steps_total=accepted_total,
            mass_initial_weighted=mass_initial,
            cumulative_grain_flux_volume_m3={i: 0.0 for i in range(1, len(current))},
            next_step_over_b=step,
            clean_acceptances=clean,
            rejection_reason_counts=dict(rejection_counts),
            last_rejection_reasons=list(last_rejections),
            quasistatic_trial_rejections_total=rejected_total,
            restart_fast_precondition_calls_total=0,
            slow_clock_quadrature_refinements_total=0,
            active_minimum_step_over_b=minimum_step_over_b,
            ordinary_fixed_q_calls_total=accepted_total,
            event_method=EVENT_INTEGRATOR_NAME,
            immutable_parent_geometry_used=False,
            transferred_volume_m3=transferred_m3)

    while q < target-1.0e-14:
        if accepted_total >= int(maximum_accepted_states):
            restart = restart_record()
            return *current, False, dict(
                completed=False, event_progress_over_b=q,
                event_progress_m=q*transport.b_m,
                event_time_model=elapsed, packets=packets,
                event_restart=restart,
                stop_reason="current-state event exceeded accepted-state cap",
                stop_detail=f"accepted={accepted_total}, cap={maximum_accepted_states}",
                branch_time_integrator=EVENT_INTEGRATOR_NAME,
                explicit_max_fourth_order_courant=None)
        q_trial = min(q+step, target)
        dq_over_b = q_trial-q
        dq_m = dq_over_b*transport.b_m
        try:
            record0 = state_evaluator(*current)
            affinity0 = float(previous["transport_affinity_Pa"])
            rate0 = _rate(previous, transport)
            if affinity0 <= 0.0 or rate0 <= 0.0:
                raise RuntimeError("nonpositive_transport_affinity")
            receiver, donor, mask_diagnostic = build_current_state_masks(
                current[0], setup["z"], setup["r_c"],
                z_TJ_m=float(record0["z_TJ_m"]),
                r_TJ_m=float(record0["r_TJ_m"]), W_m=float(setup["W"]),
                parameters=mask_parameters)
            transfer_volume = float(record0["contact_area_m2"])*dq_m
            candidate, transfer_diagnostic = transfer_fn(
                current, receiver, donor,
                transfer_volume_m3=transfer_volume,
                r_c=setup["r_c"], dr=setup["dr"], dz=setup["dz"])
            candidate, row, fast = fast_relax_fn(candidate, q_trial)
            if not bool(fast["converged"]):
                raise RuntimeError("fast_manifold")
            record1 = state_evaluator(*candidate)
            row["contact_area_m2"] = float(record1["contact_area_m2"])
            row["sigma_integral_continuous_MPa"] = float(
                row["sigma_integral_continuous_Pa"])*1.0e-6
            rate1 = _rate(row, transport)
            if float(row["transport_affinity_Pa"]) <= 0.0 or rate1 <= 0.0:
                raise RuntimeError("nonpositive_transport_affinity")
            changes = _changes(previous, row)
            reasons = [key for key, value in changes.items()
                       if value > float(limits[key])]
        except (RuntimeError, ValueError) as error:
            reasons = [str(error)]
        if reasons:
            rejected_total += 1
            last_rejections = list(reasons)
            for reason in reasons:
                rejection_counts[reason] = int(rejection_counts.get(reason, 0))+1
            clean = 0
            if q_trial-q <= minimum_step_over_b+1.0e-15:
                restart = restart_record()
                return *current, False, dict(
                    completed=False, event_progress_over_b=q,
                    event_progress_m=q*transport.b_m,
                    event_time_model=elapsed, packets=packets,
                    event_restart=restart,
                    stop_reason=("nonpositive_transport_affinity"
                                 if "nonpositive_transport_affinity" in reasons
                                 else "current-state transfer failed at minimum step"),
                    stop_detail=str(reasons),
                    branch_time_integrator=EVENT_INTEGRATOR_NAME,
                    explicit_max_fourth_order_courant=None)
            step = max(minimum_step_over_b, 0.5*(q_trial-q))
            continue

        interval = transport_clock_increment(transfer_volume, rate0, rate1)
        elapsed += interval
        transferred_m3 += transfer_volume
        accepted_total += 1
        packet = dict(
            q_start_over_b=q, q_end_over_b=q_trial,
            q_start_m=q*transport.b_m, q_end_m=q_trial*transport.b_m,
            dq_m=dq_m, continuation_step_over_b=dq_over_b,
            transferred_volume_increment_m3=transfer_volume,
            transferred_volume_cumulative_m3=transferred_m3,
            transport_dt_model=interval,
            accumulated_event_time_model=elapsed,
            affinity_start_Pa=affinity0,
            affinity_end_Pa=float(row["transport_affinity_Pa"]),
            transport_affinity_Pa=float(row["transport_affinity_Pa"]),
            volume_rate_start_m3_per_model_time=rate0,
            volume_rate_end_m3_per_model_time=rate1,
            fast_relax_iterations=int(fast.get("iterations", 0)),
            fast_relax_blocks=int(fast.get("blocks", 0)),
            fast_relax_calls=1,
            mask_rebuilt_from_current_state=True,
            immutable_parent_geometry_used=False,
            transfer_relative_mass_closure=float(
                transfer_diagnostic["relative_mass_closure"]),
            transfer_total_volume_relative_error=float(
                transfer_diagnostic["total_volume_relative_error"]),
            receiver_lambda=float(transfer_diagnostic["lambda_receiver"]),
            donor_lambda=float(transfer_diagnostic["lambda_donor"]),
            trial_rejections_total=rejected_total,
            rejection_reason_counts=dict(rejection_counts),
            last_rejection_reasons=list(last_rejections),
            **{key: value for key, value in row.items()
               if (np.isscalar(value) and not isinstance(value, str)
                   and key not in {
                       "transport_affinity_Pa", "fast_relax_iterations",
                       "fast_relax_blocks", "fast_relax_calls"})},
            **{f"accepted_change_{key}": value
               for key, value in changes.items()})
        current = tuple(np.asarray(field).copy() for field in candidate)
        previous = row
        q = q_trial
        packets.append(packet)
        clean += 1
        if clean >= 3:
            step = min(maximum_step_over_b, 2.0*step)
            clean = 0
        progress = restart_record()
        if accepted_progress_callback is not None:
            accepted_progress_callback(packet, current, progress)
        if accepted_state_callback is not None:
            accepted_state_callback(packet, current)

    restart = restart_record()
    return *current, True, dict(
        completed=True, event_progress_over_b=q,
        event_progress_m=q*transport.b_m,
        event_time_model=elapsed,
        n_subincrements=len(packets), n_subincrements_total=accepted_total,
        packets=packets, event_restart=restart, stop_reason=None,
        branch_time_integrator=EVENT_INTEGRATOR_NAME,
        explicit_max_fourth_order_courant=None,
        kinetic_time_basis=transport.kinetic_time_basis,
        physical_seconds_conversion=None,
        ordinary_M_s_step_applied=False,
        immutable_parent_geometry_used=False,
        masks_rebuilt_every_increment=True)


current_state_mass_transfer_event.event_integrator_name = EVENT_INTEGRATOR_NAME
