"""Milestone 16P Section 7: repeated one-b event mass-conservation
regression test.

Root cause of the previously-observed ~8.8e-6 relative mass drift per
event (found via a direct instrumented replay, see
scripts/m16n_transport_only_microtest.py's before/after): the excess-mass
redistribution only ever compensated the f>1 EXCESS side of
`np.clip(f, 0.0, 1.0)` -- small negative undershoots from the upwind
advection scheme were silently clipped UP to 0, adding uncompensated mass
every substep. Fixed in both `active_sink_transport_step`
(pf_sintering/axisym_sink_rbm.py) and `multi_sink_transport_step`
(pf_sintering/m16m_multisink.py) by tracking the f<0 deficit alongside
the f>1 excess and applying one combined correction.

This test drives MANY (10+) synthetic one-b events back-to-back on the
same fields and requires the mass residual stays within the M16P gate
(|Delta M/M| <= 1e-8 per event, preferred <= 1e-10) with no systematic
(monotonically accumulating) drift across repeated events.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, _axisym_weighted_sum, active_sink_transport_step
from pf_sintering.m16m_multisink import SinkEvent, multi_sink_transport_step

GATE_REQUIRED = 1e-8
GATE_PREFERRED = 1e-10


def _synthetic_fields(Nz=60, Nr=12):
    z = (np.arange(Nz) + 0.5) * 1e-9
    r_c = (np.arange(Nr) + 0.5) * 1e-9
    z_mid = z[Nz // 2]
    W = 3e-9
    e1 = 0.5 * (1.0 + np.tanh((z[:, None] - z_mid) / W)) * np.ones((1, Nr))
    e2 = 1.0 - e1
    f = e1 + e2
    dz = float(z[1] - z[0])
    return f, e1, e2, r_c, z, dz, z_mid


def _hazard_params():
    b = 2.5e-10
    return HazardParams(kB=1.380649e-23, T=1000.0, Omega=1e-29, b=b,
                         D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * 1000.0)),
                         GS=200e-9, r0=1e12, A0=0.859 * 1.602176634e-19, V0=12.5 * b ** 3, tau_ex0=0.0)


def test_ten_repeated_single_events_no_systematic_mass_drift():
    f, e1, e2, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    V_ref = _axisym_weighted_sum(f, r_c)

    residuals = []
    for event_idx in range(10):
        sink = AxisymSink(active=True, current_disp=0.0)
        n_steps = 0
        while sink.active and n_steps < 20000:
            f, e1, e2, completed, diag = active_sink_transport_step(f, e1, e2, sink, hp, 80e6, 0.02, dz, r_c, z, z_mid)
            n_steps += 1
        assert completed, f"event {event_idx} failed to complete within step budget"
        V_now = _axisym_weighted_sum(f, r_c)
        residual = abs(V_now - V_ref) / V_ref
        residuals.append(residual)
        assert residual <= GATE_REQUIRED, f"event {event_idx}: mass residual {residual:.3e} exceeds required gate {GATE_REQUIRED:.3e}"

    # no systematic (monotonically growing) accumulation across repeated events
    assert residuals[-1] <= GATE_REQUIRED
    assert max(residuals) <= GATE_PREFERRED, f"max residual {max(residuals):.3e} exceeds preferred gate {GATE_PREFERRED:.3e}"


def test_ten_repeated_multisink_events_no_systematic_mass_drift():
    f, particle, substrate, r_c, z, dz, z_mid = _synthetic_fields()
    hp = _hazard_params()
    V_ref = _axisym_weighted_sum(f, r_c)

    residuals = []
    for event_idx in range(10):
        ev = SinkEvent(event_id=event_idx, birth_time=0.0, birth_step=0, birth_sigma=80.0, delta=0.0)
        events = [ev]
        n_steps = 0
        while ev.active and n_steps < 20000:
            f, particle, substrate, completed_ids, diag = multi_sink_transport_step(
                f, particle, substrate, events, hp, 80e6, 0.02, dz, r_c, z, z_mid)
            n_steps += 1
        assert not ev.active, f"event {event_idx} failed to complete within step budget"
        V_now = _axisym_weighted_sum(f, r_c)
        residual = abs(V_now - V_ref) / V_ref
        residuals.append(residual)
        assert residual <= GATE_REQUIRED, f"event {event_idx}: mass residual {residual:.3e} exceeds required gate {GATE_REQUIRED:.3e}"

    assert max(residuals) <= GATE_PREFERRED, f"max residual {max(residuals):.3e} exceeds preferred gate {GATE_PREFERRED:.3e}"
