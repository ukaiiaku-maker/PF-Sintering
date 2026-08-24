"""Deterministic one-b PR event with model-time GB and branch transport.

This is the demonstration-only successor to the historical seconds-mapped
event paths.  It preserves the qualified cumulative body-union mechanics but
replaces finite TJ deposition supports with a boundary flux on each extracted
axisymmetric surface branch.  The instantaneous local chemical-potential
affinity is recomputed before every displacement increment.
"""
from __future__ import annotations

import math

import numpy as np

from pf_sintering.axisym_branch_boundary_flux import (
    apply_axisymmetric_branch_boundary_flux_step,
    extract_axisymmetric_surface_branches,
)
from pf_sintering.model_time_transport import ModelTimeGBTransport
from pf_sintering.rigid_rbm_deposition import (
    _axisym_weighted_sum,
    representation_corrected_union_transfer,
)
from pf_sintering.tj_transport_node import solve_model_time_tj_node


def _diagnostic_coordinates(record):
    """Drop full-grid fields from packet diagnostics."""
    return {key: value for key, value in record.items()
            if key not in ("mu_field_Pa",)}


def _node_state(record, f, r_c, z, transport, surface_mobility, *, active):
    mu_field = np.asarray(record["mu_field_Pa"], dtype=float)
    if mu_field.shape != np.asarray(f).shape:
        raise ValueError("mu_field_Pa must have the phase-field shape")
    branches = extract_axisymmetric_surface_branches(
        f, mu_field, r_c, z, z_tj=float(record["z_TJ_m"]),
        r_tj=float(record["r_TJ_m"]))
    node = solve_model_time_tj_node(
        branches, surface_mobility,
        mu_GB_Pa=float(record["mu_GB_source_Pa"]),
        contact_area_m2=float(record["contact_area_m2"]),
        transport=transport, gb_path_active=active)
    coordinates = _diagnostic_coordinates(record)
    coordinates.update(
        mu_TJ_Pa=node["mu_TJ_Pa"],
        mu_TJ_OFF_Pa=node["mu_TJ_OFF_Pa"],
        transport_affinity_Pa=node["transport_affinity_Pa"],
        Vdot_GB_m3_per_model_time=node["Vdot_GB_m3_per_model_time"],
        branch_rate_over_net_GB=node["branch_rate_over_net_GB"],
        node_zero_storage_closure_relative=node["zero_storage_closure_relative"])
    return branches, node, coordinates


def adaptive_explicit_sink_off_surface_step(
        f, particle, substrate, transport: ModelTimeGBTransport,
        *, dr: float, dz: float, r_c, z, W: float, state_evaluator,
        surface_flux_mobility_m6_per_J_model_time: float,
        B_m4_per_model_time: float,
        max_fourth_order_courant: float = 0.15,
        positive_branch_grain: int = 1,
        branch_mass_closure_relative_tolerance: float = 1e-8):
    """Advance one adaptive explicit surface-only step with ``L_GB=0``.

    The OFF node may transmit equal and opposite signed rates between the two
    branches, but it has zero net GB input.  The returned diagnostics expose
    both individual boundary rates, one-sided node gradients, spatial branch
    nonuniformity, total mass closure, and grain-wise exchange closure.
    """
    fields = tuple(
        np.asarray(field, dtype=float) for field in (f, particle, substrate))
    if any(field.shape != fields[0].shape for field in fields):
        raise ValueError("f and ownership fields must have identical shapes")
    if B_m4_per_model_time <= 0.0 or max_fourth_order_courant <= 0.0:
        raise ValueError("B and the explicit fourth-order Courant limit must be positive")
    before = state_evaluator(*fields)
    branches, node_before, coordinates_before = _node_state(
        before, fields[0], r_c, z, transport,
        surface_flux_mobility_m6_per_J_model_time, active=False)
    minimum_cell_length = min(
        float(np.min(branch.cell_lengths_m)) for branch in branches)
    dt_model = (
        0.9 * max_fourth_order_courant * minimum_cell_length ** 4
        / B_m4_per_model_time)
    attached = apply_axisymmetric_branch_boundary_flux_step(
        *fields, r_c, dr, dz, W, branches,
        surface_flux_mobility_m6_per_J_model_time,
        incoming_volume_rate_m3_per_model_time=0.0,
        dt_model=dt_model,
        positive_branch_grain=positive_branch_grain,
        branch_volume_rates_m3_per_model_time=(
            node_before["branch_volume_rates_m3_per_model_time"]),
        mass_closure_relative_tolerance=(
            branch_mass_closure_relative_tolerance))
    state = tuple(np.asarray(field, dtype=float) for field in attached[:3])
    attachment = attached[3]
    after = state_evaluator(*state)
    branches_after, node_after, coordinates_after = _node_state(
        after, state[0], r_c, z, transport,
        surface_flux_mobility_m6_per_J_model_time, active=False)

    def uniformity(branches_now, node_now):
        return {
            branch.side: dict(
                mu_span_Pa=float(np.ptp(branch.mu_Pa)),
                maximum_abs_mu_minus_node_Pa=float(np.max(np.abs(
                    branch.mu_Pa - node_now["mu_TJ_Pa"]))),
                one_sided_node_gradient_Pa_per_m=float(
                    node_now["branch"][branch.side][
                        "one_sided_node_gradient_Pa_per_m"]),
                boundary_rate_m3_per_model_time=float(
                    node_now["branch_volume_rates_m3_per_model_time"][
                        branch.side]))
            for branch in branches_now}

    return *state, dict(
        dt_model=dt_model,
        fourth_order_courant=(
            B_m4_per_model_time * dt_model / minimum_cell_length ** 4),
        GB_path_active=False,
        L_GB_m3_per_Pa_model_time=0.0,
        node_before=node_before,
        node_after=node_after,
        coordinates_before=coordinates_before,
        coordinates_after=coordinates_after,
        branch_uniformity_before=uniformity(branches, node_before),
        branch_uniformity_after=uniformity(branches_after, node_after),
        branch_boundary_flux=attachment,
        total_mass_closure_relative=attachment["closure_relative"],
        grain_flux_volume_change_m3=attachment[
            "grain_flux_volume_change_m3"],
        expected_grain_flux_volume_change_m3=attachment[
            "expected_grain_flux_volume_change_m3"],
        grain_flux_closure_relative=attachment[
            "grain_flux_closure_relative"],
        kinetic_time_basis=transport.kinetic_time_basis,
        physical_seconds_conversion=None,
        ordinary_M_s_step_applied=False)


def representation_corrected_model_time_boundary_flux_event(
        f, particle, substrate, transport: ModelTimeGBTransport,
        *, dr: float, dz: float, r_c, z, GB_z_hint: float, W: float,
        state_evaluator, surface_flux_mobility_m6_per_J_model_time: float,
        max_increment_fraction_b: float = 0.01,
        event_quota_m: float | None = None, max_subincrements: int = 512,
        positive_branch_grain: int = 1,
        implicit_mullins_B_m4_per_model_time: float | None = None,
        explicit_stability_B_m4_per_model_time: float | None = None,
        explicit_max_fourth_order_courant: float = 0.05,
        branch_mass_closure_relative_tolerance: float = 1e-8,
        packet_diagnostic_stride: int = 1,
        accepted_step_callback=None,
        accepted_state_callback=None,
        event_restart=None):
    """Integrate an active deterministic event on the model-time clock.

    ``state_evaluator(f,p,n)`` is called at every packet and must return the
    current ``mu_GB_source_Pa``, ``contact_area_m2``, ``mu_field_Pa``,
    ``z_TJ_m``, and ``r_TJ_m``.  The zero-storage TJ node determines
    ``mu_TJ``, total GB delivery, and the two generally unequal branch rates.
    No saved event-nucleation stress or prescribed half partition is accepted.

    Per increment the sequence is:

    ``Delta_mu -> Vdot_GB -> dq/body union -> branch boundary flux``.

    The branch update itself performs the surface redistribution over that
    same model-time interval.  An ordinary volumetric ``M_s`` step is
    therefore intentionally absent.

    ``accepted_state_callback(packet, state)`` is a diagnostics-only hook.
    It receives the same packet as ``accepted_step_callback`` plus the
    accepted ``(f, particle, substrate)`` state.  The state must be treated
    as read-only by callbacks.
    """
    if not 0.0 < max_increment_fraction_b <= 1.0:
        raise ValueError("max_increment_fraction_b must lie in (0,1]")
    if dr <= 0.0 or dz <= 0.0 or W <= 0.0:
        raise ValueError("dr, dz, and W must be positive")
    if surface_flux_mobility_m6_per_J_model_time < 0.0:
        raise ValueError("surface flux mobility must be non-negative")
    if (not isinstance(packet_diagnostic_stride, int)
            or packet_diagnostic_stride <= 0):
        raise ValueError("packet diagnostic stride must be a positive integer")
    if (implicit_mullins_B_m4_per_model_time is not None
            and explicit_stability_B_m4_per_model_time is not None):
        raise ValueError("implicit and adaptive-explicit branch modes are exclusive")
    if explicit_stability_B_m4_per_model_time is not None:
        if (explicit_stability_B_m4_per_model_time <= 0.0
                or not math.isfinite(explicit_stability_B_m4_per_model_time)):
            raise ValueError("explicit stability B must be positive and finite")
        if (explicit_max_fourth_order_courant <= 0.0
                or not math.isfinite(explicit_max_fourth_order_courant)):
            raise ValueError("explicit fourth-order Courant limit must be positive")
    quota = transport.b_m if event_quota_m is None else float(event_quota_m)
    if not 0.0 < quota <= transport.b_m:
        raise ValueError("event_quota_m must lie in (0,b]")

    input_state = tuple(
        np.asarray(field, dtype=float) for field in (f, particle, substrate))
    if any(field.shape != input_state[0].shape for field in input_state):
        raise ValueError("f and ownership fields must have identical shapes")
    factor = 2.0 * math.pi * float(dr) * float(dz)

    def grain_volumes_m3(fields):
        return {
            owner: factor * _axisym_weighted_sum(fields[owner], r_c)
            for owner in (1, 2)}
    if event_restart is None:
        base = tuple(field.copy() for field in input_state)
        state = tuple(field.copy() for field in input_state)
        union_previous = tuple(field.copy() for field in base)
        mass_initial_weighted = _axisym_weighted_sum(base[0], r_c)
        cumulative_q = 0.0
        cumulative_source_weighted = 0.0
        total_time_model = 0.0
        accepted_steps_before = 0
        cumulative_grain_flux_volume_m3 = {1: 0.0, 2: 0.0}
    else:
        required_restart = (
            "base_fields", "union_previous_fields", "cumulative_q_m",
            "cumulative_source_weighted", "event_time_model",
            "accepted_steps_total", "mass_initial_weighted",
            "cumulative_grain_flux_volume_m3")
        missing_restart = [
            key for key in required_restart if key not in event_restart]
        if missing_restart:
            raise ValueError(f"event restart omitted {missing_restart}")
        base = tuple(
            np.asarray(field, dtype=float).copy()
            for field in event_restart["base_fields"])
        union_previous = tuple(
            np.asarray(field, dtype=float).copy()
            for field in event_restart["union_previous_fields"])
        if any(field.shape != input_state[0].shape
               for field in (*base, *union_previous)):
            raise ValueError("event restart fields do not match current state")
        state = tuple(field.copy() for field in input_state)
        cumulative_q = float(event_restart["cumulative_q_m"])
        cumulative_source_weighted = float(
            event_restart["cumulative_source_weighted"])
        total_time_model = float(event_restart["event_time_model"])
        accepted_steps_before = int(event_restart["accepted_steps_total"])
        mass_initial_weighted = float(event_restart["mass_initial_weighted"])
        cumulative_grain_flux_volume_m3 = {
            owner: float(event_restart["cumulative_grain_flux_volume_m3"][owner])
            for owner in (1, 2)}
        if cumulative_q < 0.0 or cumulative_q >= quota:
            raise ValueError("event restart progress must lie in [0, quota)")
    packets = []
    accepted_steps = 0
    last_packet = None
    cached_state = None
    integrator_label = (
        "linearly implicit Mullins Rosenbrock step"
        if implicit_mullins_B_m4_per_model_time is not None else
        "adaptively subcycled explicit Euler"
        if explicit_stability_B_m4_per_model_time is not None else
        "explicit Euler")
    dose_tolerance = max(1e-14 * transport.b_m, 1e-12 * quota)

    def make_restart_state():
        return dict(
            schema="model_time_pr_event_restart_v1",
            base_fields=tuple(field.copy() for field in base),
            union_previous_fields=tuple(
                np.asarray(field, dtype=float).copy()
                for field in union_previous),
            cumulative_q_m=float(cumulative_q),
            cumulative_source_weighted=float(cumulative_source_weighted),
            event_time_model=float(total_time_model),
            accepted_steps_total=int(accepted_steps_before + accepted_steps),
            mass_initial_weighted=float(mass_initial_weighted),
            cumulative_grain_flux_volume_m3=dict(
                cumulative_grain_flux_volume_m3))

    def packet_samples_with_last():
        if last_packet is None or (packets and packets[-1] is last_packet):
            return packets
        return packets + [last_packet]

    for _ in range(max_subincrements):
        remaining = quota - cumulative_q
        if remaining <= dose_tolerance:
            break
        if cached_state is None:
            before = state_evaluator(*state)
            required = (
                "mu_GB_source_Pa", "contact_area_m2", "mu_field_Pa",
                "z_TJ_m", "r_TJ_m")
            missing = [name for name in required if name not in before]
            if missing:
                raise ValueError(f"state_evaluator omitted {missing}")
            branches, node_before, before_coordinates = _node_state(
                before, state[0], r_c, z, transport,
                surface_flux_mobility_m6_per_J_model_time, active=True)
        else:
            before, branches, node_before, before_coordinates = cached_state
        affinity = float(node_before["transport_affinity_Pa"])
        contact_area = float(before["contact_area_m2"])
        if not math.isfinite(affinity):
            raise ValueError("transport affinity must be finite")
        if affinity <= 0.0:
            return *state, False, dict(
                completed=False, paused=True,
                event_progress_m=cumulative_q,
                event_progress_over_b=cumulative_q / transport.b_m,
                remaining_to_quota_m=max(0.0, remaining),
                event_time_model=total_time_model,
                n_subincrements=accepted_steps,
                n_subincrements_total=accepted_steps_before + accepted_steps,
                packets=packet_samples_with_last(),
                event_restart=make_restart_state(),
                stop_state=before_coordinates,
                stop_reason="nonpositive instantaneous GB-to-TJ-node affinity",
                kinetic_time_basis=transport.kinetic_time_basis,
                physical_seconds_conversion=None,
                branch_time_integrator=integrator_label,
                ordinary_M_s_step_applied=False)
        volume_rate = float(node_before["Vdot_GB_m3_per_model_time"])
        if volume_rate <= 0.0:
            raise RuntimeError("positive affinity produced no GB delivery")

        dq = min(remaining, max_increment_fraction_b * transport.b_m)
        explicit_adaptive_reductions = 0
        while True:
            q_next = cumulative_q + dq
            transfer = representation_corrected_union_transfer(
                *base, q_next, dz, r_c, z, GB_z_hint)
            union_next = transfer[:3]
            source_next_weighted = transfer[3]["V_source_weighted"]
            source_increment_weighted = (
                source_next_weighted - cumulative_source_weighted)
            source_increment_m3 = source_increment_weighted * factor
            if source_increment_m3 <= 0.0:
                raise RuntimeError("cumulative body-union source is not increasing")
            packet_time_model = source_increment_m3 / volume_rate
            explicit_courant = None
            if explicit_stability_B_m4_per_model_time is not None:
                minimum_cell_length = min(
                    float(np.min(branch.cell_lengths_m)) for branch in branches)
                explicit_courant = (
                    explicit_stability_B_m4_per_model_time
                    * packet_time_model / minimum_cell_length ** 4)
                if explicit_courant > explicit_max_fourth_order_courant:
                    scale = 0.9 * (
                        explicit_max_fourth_order_courant / explicit_courant)
                    dq *= max(min(scale, 0.9), 0.1)
                    explicit_adaptive_reductions += 1
                    if dq <= 1e-14 * transport.b_m:
                        raise RuntimeError(
                            "adaptive explicit branch packet fell below 1e-14 b")
                    continue
            break

        state_before_step = state
        state_after_union = tuple(
            current + (new_union - old_union)
            for current, new_union, old_union
            in zip(state, union_next, union_previous))
        union_partition_residual = float(np.max(np.abs(
            state_after_union[1] + state_after_union[2] - state_after_union[0])))
        try:
            attached = apply_axisymmetric_branch_boundary_flux_step(
                *state_after_union, r_c, dr, dz, W, branches,
                surface_flux_mobility_m6_per_J_model_time, volume_rate,
                packet_time_model, positive_branch_grain=positive_branch_grain,
                implicit_mullins_B_m4_per_model_time=(
                    implicit_mullins_B_m4_per_model_time),
                branch_volume_rates_m3_per_model_time=(
                    node_before["branch_volume_rates_m3_per_model_time"]),
                mass_closure_relative_tolerance=(
                    branch_mass_closure_relative_tolerance))
        except RuntimeError as error:
            return *state, False, dict(
                completed=False, paused=False, numerical_failure=True,
                event_progress_m=cumulative_q,
                event_progress_over_b=cumulative_q / transport.b_m,
                attempted_q_end_m=q_next,
                attempted_q_end_over_b=q_next / transport.b_m,
                remaining_to_quota_m=max(0.0, quota - cumulative_q),
                event_time_model=total_time_model,
                attempted_packet_time_model=packet_time_model,
                n_subincrements=accepted_steps,
                n_subincrements_total=accepted_steps_before + accepted_steps,
                packets=packet_samples_with_last(),
                event_restart=make_restart_state(),
                stop_state=before_coordinates,
                stop_reason=f"branch time integration failed closed: {error}",
                kinetic_time_basis=transport.kinetic_time_basis,
                physical_seconds_conversion=None,
                branch_time_integrator=integrator_label,
                ordinary_M_s_step_applied=False)
        state = tuple(np.asarray(field, dtype=float) for field in attached[:3])
        attachment = attached[3]
        grain_before_m3 = grain_volumes_m3(state_before_step)
        grain_after_union_m3 = grain_volumes_m3(state_after_union)
        grain_after_flux_m3 = grain_volumes_m3(state)
        grain_union_change_m3 = {
            owner: grain_after_union_m3[owner] - grain_before_m3[owner]
            for owner in (1, 2)}
        grain_flux_change_m3 = {
            owner: grain_after_flux_m3[owner] - grain_after_union_m3[owner]
            for owner in (1, 2)}
        expected_grain_flux_change_m3 = {
            owner: float(attachment[
                "expected_grain_flux_volume_change_m3"][owner])
            for owner in (1, 2)}
        grain_flux_closure_m3 = {
            owner: grain_flux_change_m3[owner]
            - expected_grain_flux_change_m3[owner]
            for owner in (1, 2)}
        for owner in (1, 2):
            cumulative_grain_flux_volume_m3[owner] += (
                expected_grain_flux_change_m3[owner])
        mass_weighted = _axisym_weighted_sum(state[0], r_c)
        after = state_evaluator(*state)
        branches_after, node_after, after_coordinates = _node_state(
            after, state[0], r_c, z, transport,
            surface_flux_mobility_m6_per_J_model_time, active=True)
        packet = dict(
            q_start_m=cumulative_q, q_end_m=q_next,
            q_start_over_b=cumulative_q / transport.b_m,
            q_end_over_b=q_next / transport.b_m,
            dq_m=dq, fraction_b=dq / transport.b_m,
            transport_affinity_before_Pa=affinity,
            mu_GB_before_Pa=node_before["mu_GB_Pa"],
            mu_TJ_before_Pa=node_before["mu_TJ_Pa"],
            mu_TJ_OFF_before_Pa=node_before["mu_TJ_OFF_Pa"],
            branch_rate_over_net_GB=node_before["branch_rate_over_net_GB"],
            branch_volume_rates_m3_per_model_time=(
                node_before["branch_volume_rates_m3_per_model_time"]),
            node_zero_storage_closure_relative=(
                node_before["zero_storage_closure_relative"]),
            contact_area_m2=contact_area,
            tau_gb_model=transport.tau_gb_model(affinity),
            qdot_nominal_m_per_model_time=transport.qdot_m_per_model_time(
                affinity),
            Vdot_GB_m3_per_model_time=volume_rate,
            source_increment_weighted=source_increment_weighted,
            source_increment_m3=source_increment_m3,
            transport_dt_model=packet_time_model,
            explicit_fourth_order_courant=explicit_courant,
            explicit_adaptive_reductions=explicit_adaptive_reductions,
            coordinates_before=before_coordinates,
            coordinates_after_boundary_flux=after_coordinates,
            node_before=node_before,
            node_after=node_after,
            branch_boundary_flux=attachment,
            grain_volume_before_m3=grain_before_m3,
            grain_volume_after_union_m3=grain_after_union_m3,
            grain_volume_after_flux_m3=grain_after_flux_m3,
            grain_union_volume_change_m3=grain_union_change_m3,
            grain_flux_volume_change_m3=grain_flux_change_m3,
            expected_grain_flux_volume_change_m3=(
                expected_grain_flux_change_m3),
            grain_flux_closure_m3=grain_flux_closure_m3,
            cumulative_grain_flux_volume_m3=dict(
                cumulative_grain_flux_volume_m3),
            union_partition_residual=union_partition_residual,
            partition_residual_after_flux=float(np.max(np.abs(
                state[1] + state[2] - state[0]))),
            mass_relative_error_after_flux=(
                mass_weighted - mass_initial_weighted)
                / max(abs(mass_initial_weighted), 1e-300))
        if accepted_step_callback is not None:
            accepted_step_callback(packet)
        if accepted_state_callback is not None:
            accepted_state_callback(packet, state)
        accepted_steps += 1
        if (accepted_steps == 1
                or accepted_steps % packet_diagnostic_stride == 0):
            packets.append(packet)
        last_packet = packet
        cumulative_q = q_next
        cumulative_source_weighted = source_next_weighted
        union_previous = union_next
        total_time_model += packet_time_model
        cached_state = (after, branches_after, node_after, after_coordinates)
    else:
        raise RuntimeError("model-time boundary-flux event exceeded max_subincrements")

    if 0.0 < quota - cumulative_q <= dose_tolerance:
        cumulative_q = quota
    completed = cumulative_q >= quota - dose_tolerance
    if last_packet is not None and (not packets or packets[-1] is not last_packet):
        packets.append(last_packet)
    if cached_state is None:
        final = state_evaluator(*state)
        _, final_node, final_coordinates = _node_state(
            final, state[0], r_c, z, transport,
            surface_flux_mobility_m6_per_J_model_time, active=True)
    else:
        _, _, final_node, final_coordinates = cached_state
    return *state, completed, dict(
        completed=completed, paused=False,
        event_quota_m=quota, event_quota_fraction_b=quota / transport.b_m,
        event_progress_m=cumulative_q,
        event_progress_over_b=cumulative_q / transport.b_m,
        remaining_to_quota_m=(0.0 if completed else quota - cumulative_q),
        event_time_model=total_time_model,
        n_subincrements=accepted_steps, packets=packets,
        n_subincrements_total=accepted_steps_before + accepted_steps,
        packet_diagnostic_stride=packet_diagnostic_stride,
        cumulative_source_weighted=cumulative_source_weighted,
        transported_volume_m3=cumulative_source_weighted * factor,
        cumulative_grain_flux_volume_m3=dict(
            cumulative_grain_flux_volume_m3),
        final_grain_volumes_m3=grain_volumes_m3(state),
        event_restart=make_restart_state(),
        final_coordinates=final_coordinates,
        final_node=final_node,
        transport_affinity_updated_each_increment=True,
        cumulative_union_uses_single_pre_event_reference=True,
        surface_operator="axisymmetric 1-D FV TJ boundary flux",
        TJ_boundary_condition="zero-storage chemical-potential node",
        imposed_half_partition=False,
        boundary_flux_and_surface_redistribution_share_model_increment=True,
        ordinary_M_s_step_applied=False,
        branch_time_integrator=integrator_label,
        explicit_max_fourth_order_courant=(
            explicit_max_fourth_order_courant
            if explicit_stability_B_m4_per_model_time is not None else None),
        kinetic_time_basis=transport.kinetic_time_basis,
        physical_seconds_conversion=None,
        material_specific_diffusivity_used=False)
