"""Milestone 16L Sections 3-4: deterministic unit tests for the one-
Burgers-vector event state machine (`nucleation_hazard_step` +
`active_sink_transport_step`, pf_sintering/axisym_sink_rbm.py).

These are pure state-machine tests on small synthetic fields -- no PF
solver, no geometry construction -- isolating exactly the three claims
Section 3/4 requires: (1) an event can never apply more than the
remaining fraction of `b` in a single step, even when the requested
displacement would overshoot; (2) once complete, the sink turns off
immediately and a subsequent transport call is a true no-op; (3) two
independent events never share/inherit displacement state.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, active_sink_transport_step, nucleation_hazard_step


def _synthetic_fields(Nz=40, Nr=10):
    """A simple e1 (particle, z>=Nz/2) / e2 (substrate, z<Nz/2) split with
    a smooth tanh transition, giving a well-defined, nonzero vz_field
    near the interface for the transport step to act on."""
    z = (np.arange(Nz) + 0.5) * 1e-9
    r_c = (np.arange(Nr) + 0.5) * 1e-9
    z_mid = z[Nz // 2]
    W = 3e-9
    e1 = 0.5 * (1.0 + np.tanh((z[:, None] - z_mid) / W)) * np.ones((1, Nr))
    e2 = 1.0 - e1
    f = e1 + e2  # exactly 1 everywhere by construction
    dz = float(z[1] - z[0])
    return f, e1, e2, r_c, z, dz, z_mid


def _hazard_params(V0_over_b3=12.5, A0_eV=0.859, GS_nm=200.0, sigma_target_check=None):
    b = 2.5e-10
    hp = HazardParams(kB=1.380649e-23, T=1000.0, Omega=1e-29, b=b,
                       D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * 1000.0)),
                       GS=GS_nm * 1e-9, r0=1e12, A0=A0_eV * 1.602176634e-19, V0=V0_over_b3 * b ** 3, tau_ex0=0.0)
    return hp


@pytest.mark.parametrize("eps_frac", [1e-6, 1e-3, 1e-2])
def test_event_completes_at_exactly_b_not_beyond(eps_frac):
    """Section 3: with delta_event = b - epsilon and a driving stress
    large enough that the naturally-requested displacement would exceed
    epsilon, the APPLIED displacement must be clipped to exactly
    epsilon (not the larger naturally-requested amount), delta_event
    must reach b, and completion must fire with sink.active->False."""
    f, e1, e2, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    epsilon = eps_frac * hp.b

    sink = AxisymSink(active=True, current_disp=hp.b - epsilon)
    dt = 1.0  # large model-time step so v_event*dt >> epsilon at high sigma_drive
    sigma_s = 100e6  # 100 MPa -- strong driving stress, should request far more than epsilon

    f2, e1_2, e2_2, completed, diag = active_sink_transport_step(
        f.copy(), e1.copy(), e2.copy(), sink, hp, sigma_s, dt, dz, r_c, z, z_mid)

    # `requested_d_delta` in the diag dict is already capped by `remaining`
    # (by design -- it's what actually gets used to scale the advection
    # duration). Confirm the cap is doing real work by checking the
    # UNCAPPED, rate-driven amount (v_event*dt) independently exceeds
    # epsilon -- i.e. this test setup genuinely exercises the clip, not a
    # case where the natural request was already <= epsilon.
    uncapped_request = diag["v_event"] * dt
    assert uncapped_request > epsilon, "test setup should naturally request more than the remaining budget"
    assert diag["requested_d_delta"] == pytest.approx(epsilon, rel=1e-9, abs=1e-25), \
        "requested_d_delta (already capped) should equal the remaining epsilon"
    assert diag["applied_d_delta"] == pytest.approx(epsilon, rel=1e-6, abs=1e-20), \
        f"applied displacement must be clipped to exactly the remaining epsilon={epsilon:.3e}, got {diag['applied_d_delta']:.3e}"
    assert completed is True
    assert diag["completed"] is True
    assert sink.active is False, "sink must turn off immediately upon reaching b"
    assert sink.current_disp == pytest.approx(0.0, abs=1e-20), "current_disp must reset for the next event"
    assert diag["delta_event"] == pytest.approx(0.0, abs=1e-20)
    assert sink.hazard == pytest.approx(0.0, abs=1e-20), "hazard accumulator must be rearmed (reset to 0)"


@pytest.mark.parametrize("eps_frac", [1e-6, 1e-3, 1e-2])
def test_no_further_transport_after_completion(eps_frac):
    """Section 3 (second half): after completion, calling the transport
    operator again (sink now inactive) must be a strict no-op -- zero
    applied displacement, unchanged fields, unchanged event count."""
    f, e1, e2, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    epsilon = eps_frac * hp.b
    sink = AxisymSink(active=True, current_disp=hp.b - epsilon, n_events=1)

    f, e1, e2, completed, _ = active_sink_transport_step(f, e1, e2, sink, hp, 100e6, 1.0, dz, r_c, z, z_mid)
    assert completed and not sink.active

    f_before, e1_before, e2_before = f.copy(), e1.copy(), e2.copy()
    n_events_before = sink.n_events

    f2, e1_2, e2_2, completed2, diag2 = active_sink_transport_step(f, e1, e2, sink, hp, 100e6, 1.0, dz, r_c, z, z_mid)

    assert completed2 is False
    assert diag2["active"] is False
    assert sink.n_events == n_events_before, "no new event should be counted by a transport call alone"
    np.testing.assert_array_equal(f2, f_before)
    np.testing.assert_array_equal(e1_2, e1_before)
    np.testing.assert_array_equal(e2_2, e2_before)


def test_event_pauses_not_stalls_when_stress_relaxes_to_zero():
    """Section 6 regression guard: sigma_drive<=0 before b is reached
    must pause (not force-complete, not silently drift) -- active stays
    True, completed is False, delta_event is unchanged, fields
    untouched."""
    f, e1, e2, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    sink = AxisymSink(active=True, current_disp=0.3 * hp.b)

    f2, e1_2, e2_2, completed, diag = active_sink_transport_step(
        f.copy(), e1.copy(), e2.copy(), sink, hp, -5e6, 1.0, dz, r_c, z, z_mid)

    assert completed is False
    assert diag["paused"] is True
    assert sink.active is True
    assert sink.current_disp == pytest.approx(0.3 * hp.b)
    np.testing.assert_array_equal(f2, f)
    np.testing.assert_array_equal(e1_2, e1)
    np.testing.assert_array_equal(e2_2, e2)


def test_delta_event_never_exceeds_b_across_many_steps():
    """Section 3/20: run many transport steps at a strong, sustained
    driving stress and confirm sink.current_disp never exceeds hp.b at
    any intermediate point, and the event completes cleanly."""
    f, e1, e2, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    sink = AxisymSink(active=True, current_disp=0.0)
    dt = 0.01
    n_completions = 0
    for _ in range(2000):
        f, e1, e2, completed, diag = active_sink_transport_step(f, e1, e2, sink, hp, 80e6, dt, dz, r_c, z, z_mid)
        assert sink.current_disp <= hp.b + 1e-18, "current_disp must never exceed b"
        assert diag["applied_d_delta"] >= -1e-18, "applied displacement must never be negative"
        if completed:
            n_completions += 1
            break
    assert n_completions == 1, "event should complete exactly once under sustained driving stress"


def test_two_independent_events_do_not_share_displacement_state():
    """Section 4: event 1 nucleates, completes at exactly b; while
    inactive, no transport occurs; event 2 is triggered explicitly via a
    fresh nucleation and must start from delta_event=0, not inherit any
    residual from event 1."""
    f, e1, e2, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    rng = np.random.default_rng(0)

    # --- event 1: drive it to completion directly ---
    sink = AxisymSink(active=True, current_disp=0.0)
    for _ in range(5000):
        f, e1, e2, completed, diag = active_sink_transport_step(f, e1, e2, sink, hp, 100e6, 0.05, dz, r_c, z, z_mid)
        if completed:
            break
    assert completed, "event 1 should reach completion within the step budget"
    assert sink.active is False
    assert sink.current_disp == pytest.approx(0.0)
    event1_n_events = sink.n_events
    f_after_event1, e1_after_event1, e2_after_event1 = f.copy(), e1.copy(), e2.copy()

    # --- "wait": no transport while inactive (transport is a no-op by
    # construction when sink.active is False) ---
    for _ in range(50):
        f, e1, e2, completed, diag = active_sink_transport_step(f, e1, e2, sink, hp, 100e6, 0.05, dz, r_c, z, z_mid)
        assert not completed
        assert diag["active"] is False
    np.testing.assert_array_equal(f, f_after_event1)
    np.testing.assert_array_equal(e1, e1_after_event1)
    np.testing.assert_array_equal(e2, e2_after_event1)

    # --- event 2: explicit fresh nucleation (force via a zero threshold
    # so it fires deterministically on the next hazard_step call) ---
    sink.threshold = 1e-30
    activated = nucleation_hazard_step(sink, 100e6, 0.05, hp, rng)
    assert activated is True
    assert sink.n_events == event1_n_events + 1
    assert sink.current_disp == pytest.approx(0.0), "event 2 must start with delta_event exactly zero"
    assert sink.active is True

    # event 2 must accumulate its OWN displacement, unaffected by event 1's history
    f, e1, e2, completed2, diag2 = active_sink_transport_step(f, e1, e2, sink, hp, 100e6, 0.05, dz, r_c, z, z_mid)
    assert diag2["delta_event"] > 0.0 or diag2["completed"], "event 2 should make its own independent progress"
    assert sink.current_disp <= hp.b + 1e-18
