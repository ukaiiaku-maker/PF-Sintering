import math

import numpy as np

from pf_sintering.interface_attachment import planar_tanh_profile
from pf_sintering.model_time_pr_event import (
    adaptive_explicit_sink_off_surface_step,
    representation_corrected_model_time_boundary_flux_event,
)
from pf_sintering.model_time_transport import ModelTimeGBTransport
from pf_sintering.rigid_rbm_deposition import (
    ownership_to_kinematic_bodies,
    representation_corrected_union_transfer,
)


def test_model_time_boundary_flux_event_reaches_b_and_conserves_mass():
    dr = dz = 1.0
    z = (np.arange(12) + 0.5) - 6.0
    r_c = np.arange(10) + 0.5
    f = np.broadcast_to(
        planar_tanh_profile(r_c[None, :] - 4.0, W=1.2), (12, 10)).copy()
    particle, neighbor = ownership_to_kinematic_bodies(f, z, 0.0)
    b = 0.2
    probe = representation_corrected_union_transfer(
        f, particle, neighbor, b, dz, r_c, z, 0.0)
    event_volume = probe[3]["V_source_weighted"] * 2.0 * math.pi * dr * dz
    contact_area = event_volume / b
    transport = ModelTimeGBTransport.from_reference_target(
        x_d_m=1.0, kB_J_per_K=1.0, temperature_K=1.0,
        atomic_volume_m3=1.0, b_m=b, reference_affinity_Pa=1.0,
        reference_tau_gb_model=0.5)
    calls = []

    def evaluator(f_now, _p_now, _n_now):
        calls.append(float(np.sum(r_c[None, :] * f_now)))
        return dict(
            mu_GB_source_Pa=1.0,
            contact_area_m2=contact_area,
            mu_field_Pa=np.zeros_like(f_now),
            z_TJ_m=0.0,
            r_TJ_m=4.0)

    event = representation_corrected_model_time_boundary_flux_event(
        f, particle, neighbor, transport, dr=dr, dz=dz, r_c=r_c, z=z,
        GB_z_hint=0.0, W=1.2, state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=1e6,
        max_increment_fraction_b=0.1)
    assert event[3]
    diag = event[4]
    assert diag["event_progress_over_b"] == 1.0
    assert diag["n_subincrements"] == 10
    assert diag["kinetic_time_basis"] == "model_time_demonstration"
    assert diag["physical_seconds_conversion"] is None
    assert diag["ordinary_M_s_step_applied"] is False
    assert diag["transport_affinity_updated_each_increment"] is True
    np.testing.assert_allclose(diag["event_time_model"], 0.5, rtol=2e-6)
    np.testing.assert_allclose(event[1] + event[2], event[0], atol=2e-13)
    np.testing.assert_allclose(
        np.sum(r_c[None, :] * event[0]), np.sum(r_c[None, :] * f),
        atol=2e-11)
    # The accepted post-flux state is cached as the next packet start.
    assert len(calls) == diag["n_subincrements"] + 1
    assert all(packet["branch_boundary_flux"]["endpoint_row_count"] is None
               for packet in diag["packets"])
    assert all(
        max(abs(value) for value in packet["grain_flux_closure_m3"].values())
        < 1e-10 for packet in diag["packets"])
    assert diag["imposed_half_partition"] is False

    accepted = []
    accepted_states = []
    adaptive = representation_corrected_model_time_boundary_flux_event(
        f, particle, neighbor, transport, dr=dr, dz=dz, r_c=r_c, z=z,
        GB_z_hint=0.0, W=1.2, state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=1e6,
        max_increment_fraction_b=0.2,
        explicit_stability_B_m4_per_model_time=1.0,
        explicit_max_fourth_order_courant=0.02,
        packet_diagnostic_stride=3,
        accepted_step_callback=accepted.append,
        accepted_state_callback=lambda packet, state: accepted_states.append((
            packet["q_end_over_b"],
            tuple(float(np.sum(field)) for field in state))),
        max_subincrements=100)
    adaptive_diag = adaptive[4]
    assert adaptive[3]
    assert adaptive_diag["n_subincrements"] == len(accepted)
    assert len(accepted_states) == len(accepted)
    assert accepted_states[-1][0] == 1.0
    np.testing.assert_allclose(
        accepted_states[-1][1],
        tuple(float(np.sum(field)) for field in adaptive[:3]))
    assert len(adaptive_diag["packets"]) < len(accepted)
    assert max(packet["explicit_fourth_order_courant"] for packet in accepted) <= 0.02
    assert adaptive_diag["branch_time_integrator"] == (
        "adaptively subcycled explicit Euler")

    first_segment = representation_corrected_model_time_boundary_flux_event(
        f, particle, neighbor, transport, dr=dr, dz=dz, r_c=r_c, z=z,
        GB_z_hint=0.0, W=1.2, state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=1e6,
        max_increment_fraction_b=0.1, event_quota_m=0.4 * b)
    assert first_segment[3]
    resumed = representation_corrected_model_time_boundary_flux_event(
        *first_segment[:3], transport, dr=dr, dz=dz, r_c=r_c, z=z,
        GB_z_hint=0.0, W=1.2, state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=1e6,
        max_increment_fraction_b=0.1, event_quota_m=b,
        event_restart=first_segment[4]["event_restart"])
    assert resumed[3]
    assert resumed[4]["n_subincrements"] == 6
    assert resumed[4]["n_subincrements_total"] == 10
    for uninterrupted, restarted in zip(event[:3], resumed[:3]):
        np.testing.assert_allclose(restarted, uninterrupted, atol=2e-14)
    np.testing.assert_allclose(
        resumed[4]["event_time_model"], diag["event_time_model"], rtol=2e-6)


def test_model_time_boundary_flux_event_stops_on_current_nonpositive_affinity():
    z = (np.arange(8) + 0.5) - 4.0
    r_c = np.arange(8) + 0.5
    f = np.broadcast_to(
        planar_tanh_profile(r_c[None, :] - 3.0, W=1.0), (8, 8)).copy()
    particle, neighbor = ownership_to_kinematic_bodies(f, z, 0.0)
    transport = ModelTimeGBTransport(
        x_d_m=1.0, kB_J_per_K=1.0, temperature_K=1.0,
        atomic_volume_m3=1.0, b_m=0.2,
        D_gb_m2_per_model_time=1.0)

    def evaluator(f_now, *_owners):
        return dict(
            mu_GB_source_Pa=-1.0, contact_area_m2=1.0,
            mu_field_Pa=np.zeros_like(f_now), z_TJ_m=0.0, r_TJ_m=3.0)

    event = representation_corrected_model_time_boundary_flux_event(
        f, particle, neighbor, transport, dr=1.0, dz=1.0,
        r_c=r_c, z=z, GB_z_hint=0.0, W=1.0,
        state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=1.0)
    assert not event[3]
    assert event[4]["paused"]
    assert event[4]["event_progress_m"] == 0.0
    assert event[4]["stop_reason"] == "nonpositive_transport_affinity"


def test_sink_off_surface_step_has_zero_gb_input_and_preserves_mass():
    z = (np.arange(12) + 0.5) - 6.0
    r_c = np.arange(10) + 0.5
    f = np.broadcast_to(
        planar_tanh_profile(r_c[None, :] - 4.0, W=1.2), (12, 10)).copy()
    particle, neighbor = ownership_to_kinematic_bodies(f, z, 0.0)
    transport = ModelTimeGBTransport(
        x_d_m=1.0, kB_J_per_K=1.0, temperature_K=1.0,
        atomic_volume_m3=1.0, b_m=0.2,
        D_gb_m2_per_model_time=1.0)

    def evaluator(f_now, *_owners):
        return dict(
            mu_GB_source_Pa=0.0, contact_area_m2=1.0,
            mu_field_Pa=np.zeros_like(f_now), z_TJ_m=0.0, r_TJ_m=4.0)

    step = adaptive_explicit_sink_off_surface_step(
        f, particle, neighbor, transport, dr=1.0, dz=1.0,
        r_c=r_c, z=z, W=1.2, state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=1.0,
        B_m4_per_model_time=1.0)
    np.testing.assert_allclose(step[0], f)
    assert step[3]["GB_path_active"] is False
    assert step[3]["node_before"]["Vdot_GB_m3_per_model_time"] == 0.0
    assert step[3]["node_before"]["branch_rate_over_net_GB"] == {
        "positive": None, "negative": None}
    assert abs(step[3]["total_mass_closure_relative"]) < 1e-15
