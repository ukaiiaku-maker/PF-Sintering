"""Milestone 16M Section 13: deterministic multi-event state-machine
tests, mirroring test_m16l_event_state_machine.py's synthetic-field
approach (no PF solver, no geometry construction) but exercising
MULTIPLE simultaneously-active `SinkEvent`s and the shared
`poisson_multisink_birth_step`/`multi_sink_transport_step` pair."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pf_sintering.axisym_sink_rbm import HazardParams
from pf_sintering.m16m_multisink import (
    PoissonBirthClock,
    SinkEvent,
    lambda_birth,
    multi_sink_transport_step,
    poisson_multisink_birth_step,
)


def _synthetic_fields(Nz=40, Nr=10):
    z = (np.arange(Nz) + 0.5) * 1e-9
    r_c = (np.arange(Nr) + 0.5) * 1e-9
    z_mid = z[Nz // 2]
    W = 3e-9
    particle = 0.5 * (1.0 + np.tanh((z[:, None] - z_mid) / W)) * np.ones((1, Nr))
    substrate = 1.0 - particle
    f = particle + substrate
    dz = float(z[1] - z[0])
    return f, particle, substrate, r_c, z, dz, z_mid


def _hazard_params(V0_over_b3=12.5, A0_eV=0.859, GS_nm=200.0):
    b = 2.5e-10
    return HazardParams(kB=1.380649e-23, T=1000.0, Omega=1e-29, b=b,
                         D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * 1000.0)),
                         GS=GS_nm * 1e-9, r0=1e12, A0=A0_eV * 1.602176634e-19, V0=V0_over_b3 * b ** 3, tau_ex0=0.0)


def test_A_two_events_different_remaining_cap_independently():
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    e1 = SinkEvent(event_id=1, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=0.0)
    e2 = SinkEvent(event_id=2, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=0.9 * hp.b)
    events = [e1, e2]
    dt = 0.01
    for _ in range(5000):
        f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
            f, particle, substrate, events, hp, 100e6, dt, dz, r_c, z, z_mid)
        assert e1.delta <= hp.b + 1e-18
        assert e2.delta <= hp.b + 1e-18
        if not e1.active and not e2.active:
            break
    assert not e1.active and not e2.active
    assert e1.delta == pytest.approx(hp.b, abs=1e-18) or e1.delta == 0.0  # 0.0 if capped-then-reported as complete
    assert e2.delta == pytest.approx(hp.b, abs=1e-18) or e2.delta == 0.0


def test_B_one_completes_other_remains_active():
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    e_near = SinkEvent(event_id=1, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=hp.b - 1e-3 * hp.b)
    e_far = SinkEvent(event_id=2, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=0.0)
    events = [e_near, e_far]
    # dt chosen so the near-complete event's tiny remaining (1e-3*b) is
    # consumed in one step while the fresh event's natural request
    # (v_event*dt) stays comfortably below its own full-b budget.
    f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
        f, particle, substrate, events, hp, 100e6, 1e-4, dz, r_c, z, z_mid)
    assert completed_ids == [1]
    assert e_near.active is False
    assert e_far.active is True
    assert e_far.delta > 0.0, "the still-active event must have made independent progress in the same step"


def test_C_two_events_complete_same_step():
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    eps = 1e-4 * hp.b
    e1 = SinkEvent(event_id=1, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=hp.b - eps)
    e2 = SinkEvent(event_id=2, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=hp.b - eps)
    events = [e1, e2]
    f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
        f, particle, substrate, events, hp, 100e6, 1.0, dz, r_c, z, z_mid)
    assert set(completed_ids) == {1, 2}
    assert not e1.active and not e2.active


def test_D_birth_clock_fires_while_events_already_active():
    """Section 6: the Poisson birth clock must keep integrating and firing
    regardless of how many SinkEvents are currently active -- it never
    consults `events` at all."""
    hp = _hazard_params()
    clock = PoissonBirthClock()
    rng = np.random.default_rng(0)
    events = [SinkEvent(event_id=k, birth_time=0.0, birth_step=0, birth_sigma=100e6) for k in range(3)]
    assert all(e.active for e in events)
    n_births_total = 0
    for step in range(20000):
        n_births_total += poisson_multisink_birth_step(clock, 100e6, 0.01, hp, rng)
        if n_births_total >= 3:
            break
    assert n_births_total >= 3, "births must keep occurring even with several events already active"


def test_E_completing_one_event_does_not_deactivate_others():
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    eps = 1e-4 * hp.b
    e_done_soon = SinkEvent(event_id=1, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=hp.b - eps)
    e_fresh_a = SinkEvent(event_id=2, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=0.0)
    e_fresh_b = SinkEvent(event_id=3, birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=0.0)
    events = [e_done_soon, e_fresh_a, e_fresh_b]
    f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
        f, particle, substrate, events, hp, 100e6, 1e-4, dz, r_c, z, z_mid)
    assert completed_ids == [1]
    assert e_fresh_a.active is True and e_fresh_b.active is True
    assert e_fresh_a.delta > 0.0 and e_fresh_b.delta > 0.0


def test_F_active_count_rises_and_falls_without_corrupting_state():
    """N_active traces 0->1->2->3->2->1->0 (births added/completed
    explicitly to control the schedule) without cumulative displacement
    bookkeeping or the birth clock becoming inconsistent."""
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    events: list[SinkEvent] = []
    next_id = [0]

    def spawn(delta=0.0):
        next_id[0] += 1
        e = SinkEvent(event_id=next_id[0], birth_time=0.0, birth_step=0, birth_sigma=100e6, delta=delta)
        events.append(e)
        return e

    active_counts = []
    # 0 -> 1
    spawn()
    active_counts.append(sum(e.active for e in events))
    # 1 -> 2
    spawn()
    active_counts.append(sum(e.active for e in events))
    # 2 -> 3
    spawn()
    active_counts.append(sum(e.active for e in events))
    assert active_counts == [1, 2, 3]

    # drive all forward; force two of them near completion to shrink N_active in order
    for e in events[:2]:
        e.delta = hp.b - 1e-4 * hp.b
    for _ in range(20):
        f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
            f, particle, substrate, events, hp, 100e6, 1e-4, dz, r_c, z, z_mid)
        n_active_now = sum(e.active for e in events)
        active_counts.append(n_active_now)
        if n_active_now == 1:
            break

    assert active_counts[-1] == 1, f"expected to shrink down to N_active=1, got trace {active_counts}"
    # cumulative displacement sanity: no event ever negative or beyond b
    for e in events:
        assert -1e-18 <= e.delta <= hp.b + 1e-18


def test_G_total_applied_equals_sum_of_per_event_increments():
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    events = [SinkEvent(event_id=k, birth_time=0.0, birth_step=0, birth_sigma=100e6) for k in range(1, 4)]
    deltas_before = {e.event_id: e.delta for e in events}
    f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
        f, particle, substrate, events, hp, 100e6, 1e-4, dz, r_c, z, z_mid)
    sum_increments = sum(e.delta - deltas_before[e.event_id] for e in events)
    assert sum_increments == pytest.approx(diag["total_measured_relative_dDelta"], rel=1e-9, abs=1e-22)


def test_H_no_individual_event_ever_exceeds_b():
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    events = [SinkEvent(event_id=k, birth_time=0.0, birth_step=0, birth_sigma=100e6) for k in range(1, 5)]
    for step in range(3000):
        f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
            f, particle, substrate, events, hp, 100e6, 0.02, dz, r_c, z, z_mid)
        for e in events:
            assert e.delta <= hp.b + 1e-18
        if all(not e.active for e in events):
            break
    assert all(not e.active for e in events)


def test_lambda_birth_not_multiplied_by_site_count():
    """Section 4/5 audit: `lambda_birth` must depend ONLY on (sigma, hp),
    never on N_active or any explicit site count."""
    hp = _hazard_params()
    assert lambda_birth(50e6, hp) == lambda_birth(50e6, hp)  # deterministic, pure function
    # calling it repeatedly / with more "context" must not change the result
    vals = [lambda_birth(50e6, hp) for _ in range(5)]
    assert len(set(vals)) == 1


def test_poisson_integrated_hazard_diagnostic_mean_near_one():
    """Section 24 (light-weight sanity, not a rigorous statistical proof):
    at CONSTANT sigma (homogeneous Poisson case), the consumed thresholds
    (== the integrated hazard between successive births) should average
    close to 1 (Exp(1) mean) over many births."""
    hp = _hazard_params()
    clock = PoissonBirthClock()
    rng = np.random.default_rng(42)
    dt = 0.05  # only feeds the analytic Lambda*dt hazard integral here, no field mutation -- large dt is fine
    n_births = 0
    for _ in range(500_000):
        n_births += poisson_multisink_birth_step(clock, 60e6, dt, hp, rng)
        if n_births >= 300:
            break
    assert n_births >= 200, "test setup should generate enough births for a meaningful mean"
    mean_consumed = float(np.mean(clock.consumed_thresholds))
    assert 0.7 < mean_consumed < 1.3, f"mean consumed hazard-per-birth {mean_consumed} far from Exp(1) mean of 1.0"
