"""Milestone 16M: independent Poisson multi-sink baseline.

Generalizes the M16L single-event state machine
(`pf_sintering.axisym_sink_rbm.nucleation_hazard_step` +
`active_sink_transport_step`) to allow MULTIPLE simultaneously-active,
statistically-independent sink-nucleation events, WITHOUT introducing any
avalanche/hysteresis/site-interaction physics:

  - `PoissonBirthClock` + `poisson_multisink_birth_step`: an
    inhomogeneous-Poisson birth process using the SAME calibrated
    Arrhenius intensity Lambda(sigma,T) as M16L's `nucleation_hazard_step`
    (never multiplied by an arbitrary site count), but with the "no new
    births while one event is active" restriction REMOVED -- the
    integrated-hazard clock runs continuously regardless of how many
    events are currently active, using a while-loop over exponential
    thresholds so more than one birth in a single timestep is handled
    correctly (thinning-algorithm form: unconsumed hazard carries over to
    the next threshold, which is exactly what makes each inter-birth
    integrated hazard Exp(1)-distributed for a correctly-implemented
    nonhomogeneous Poisson process -- see `test_multisink_state_machine`'s
    Poisson-diagnostic test and Section 24 of the handoff).

  - `SinkEvent`: one independent event's own birth/progress/completion
    bookkeeping. Each event still obeys the M16L one-Burgers-vector
    contract EXACTLY (`delta_j` never exceeds `hp.b`) -- what's new is
    that MULTIPLE `SinkEvent`s can be simultaneously active.

  - `multi_sink_transport_step`: applies all currently-active events'
    requested displacements as ONE conservative particle-relative-
    substrate RBM remap per PF step (not one remap per event -- Section
    10 of the handoff explicitly warns against event-ordering-dependent
    repeated remaps). Each event's own `delta_j` (its `delta_sink` quota)
    accumulates its OWN requested share directly -- the deterministic
    analytic Coble-rate amount the field was actually advected by (M16N
    Section F correction: NOT a distribution of the measured COM
    response, which is a separate mesoscale diagnostic, per the
    transport-only microtest in M16N Section G showing ordinary capillary
    relaxation alone moves COM by a comparable amount with zero sink
    activity). A per-event cap at `hp.b` is enforced independently of how
    many events are simultaneously active (Section 12).

No event directly interacts with any other event (Section 26): the only
coupling is through the shared field state and the shared instantaneous
stress sigma(t).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from pf_sintering.axisym_sink_rbm import HazardParams, SECONDS_PER_MODEL_TIME, _axisym_weighted_sum, particle_com_z


@dataclass
class SinkEvent:
    event_id: int
    birth_time: float
    birth_step: int
    birth_sigma: float
    delta: float = 0.0  # current RBM progress (delta_j), 0 <= delta <= hp.b
    active: bool = True
    completion_time: float = float("nan")
    completion_step: int = -1
    completion_sigma: float = float("nan")

    def remaining(self, hp: HazardParams) -> float:
        return max(0.0, hp.b - self.delta)


@dataclass
class PoissonBirthClock:
    """Persistent integrated-hazard birth-process state, decoupled from
    any single event's own active/inactive status (Section 6: the clock
    runs continuously regardless of N_active)."""
    hazard: float = 0.0
    threshold: float = 0.0
    last_r_nuc: float = 0.0
    consumed_thresholds: List[float] = field(default_factory=list)  # Section 24 Poisson diagnostic


def lambda_birth(sigma_s: float, hp: HazardParams) -> float:
    """Total calibrated birth intensity Lambda(sigma,T) -- IDENTICAL
    formula to M16L's `nucleation_hazard_step`'s per-call r_nuc, and NOT
    multiplied by any site count (Section 4/5)."""
    sigma = max(0.0, sigma_s)
    if sigma <= 0:
        return 0.0
    return hp.r0 * (hp.b / hp.GS) ** 3 * math.exp(-max(0.0, hp.A0 - sigma * hp.V0) / (hp.kB * hp.T))


def poisson_multisink_birth_step(clock: PoissonBirthClock, sigma_s: float, dt: float, hp: HazardParams, rng,
                                  seconds_per_model_time: float = SECONDS_PER_MODEL_TIME) -> int:
    """Section 5/6: integrates dH_birth = Lambda(sigma,T)*dt into the
    persistent clock, then fires as many births as the accumulated hazard
    supports (a while-loop, not an if -- Section 6 explicitly requires
    handling >1 birth per timestep correctly). Returns the number of new
    births this call (0, 1, 2, ...). Does NOT look at or gate on any
    SinkEvent's active state -- the clock is single, global, and
    continuous."""
    if clock.threshold <= 0:
        clock.threshold = float(rng.exponential())
    dt_seconds = dt * seconds_per_model_time
    r_nuc = lambda_birth(sigma_s, hp)
    clock.last_r_nuc = r_nuc
    clock.hazard += r_nuc * dt_seconds
    n_births = 0
    while clock.hazard >= clock.threshold:
        clock.hazard -= clock.threshold
        clock.consumed_thresholds.append(clock.threshold)
        clock.threshold = float(rng.exponential())
        n_births += 1
    return n_births


def tau_event_estimate(sigma_mpa: float, hp: HazardParams) -> float:
    """Section 15/16: approximate time for ONE event to traverse the full
    b under the Coble law at a FIXED (not evolving) driving stress --
    tau_event ~= b / v_event = tau_Coble (v_event=b/tau_Coble by
    construction, so tau_event==tau_Coble exactly for constant sigma)."""
    sigma = sigma_mpa * 1e6
    if sigma <= 0:
        return math.inf
    xd = 0.5 * hp.GS / 2
    tau_Coble = (xd * xd * hp.kB * hp.T) / (sigma * hp.Omega * hp.D_gb) + hp.tau_ex0
    return tau_Coble


def overlap_number_table(sigma_mpa_list, hp: HazardParams, seconds_per_model_time: float = SECONDS_PER_MODEL_TIME):
    """Section 15/16: pure kinetic diagnostic table -- for each stress in
    `sigma_mpa_list`, evaluates Lambda(sigma), the mean Poisson waiting
    time 1/Lambda, tau_event(sigma), and the overlap number
    B(sigma)=Lambda*tau_event. NOT a PF run; NOT tuned to obtain any
    particular B. `seconds_per_model_time` lets the caller check whether
    conclusions about overlap depend on the (unvalidated, Section 30)
    PF/physical time mapping."""
    rows = []
    for sigma_mpa in sigma_mpa_list:
        sigma = sigma_mpa * 1e6
        Lam_model_time = lambda_birth(sigma, hp) * seconds_per_model_time  # events per MODEL time unit
        Lam_seconds = lambda_birth(sigma, hp)  # events per SI second
        tau_ev = tau_event_estimate(sigma_mpa, hp)  # seconds
        tau_ev_model_time = tau_ev / seconds_per_model_time if seconds_per_model_time > 0 else math.inf
        mean_wait_seconds = 1.0 / Lam_seconds if Lam_seconds > 0 else math.inf
        mean_wait_model_time = mean_wait_seconds / seconds_per_model_time if seconds_per_model_time > 0 else math.inf
        B = Lam_seconds * tau_ev if math.isfinite(Lam_seconds) and math.isfinite(tau_ev) else float("nan")
        rows.append(dict(sigma_MPa=sigma_mpa, Lambda_per_second=Lam_seconds, Lambda_per_model_time=Lam_model_time,
                          mean_wait_seconds=mean_wait_seconds, mean_wait_model_time=mean_wait_model_time,
                          tau_event_seconds=tau_ev, tau_event_model_time=tau_ev_model_time, B=B))
    return rows


def multi_sink_transport_step(f, particle, substrate, events: List[SinkEvent], hp: HazardParams, sigma_s: float,
                               dt: float, dz: float, r_c, z, GB_z_hint,
                               seconds_per_model_time: float = SECONDS_PER_MODEL_TIME):
    """Section 9-12: propagates ALL currently-active events by one PF
    step's worth of Coble-type diffusional advection, applied as ONE
    conservative particle-relative-substrate RBM remap (Section 10). Each
    event's own `delta_sink` quota accumulates its OWN requested share
    directly (M16N Section F correction), capping each event
    independently at `hp.b` (Section 12).

    Every active event experiences the SAME instantaneous sigma(t) and
    (since all events share identical kinetic parameters in this initial
    model) the SAME instantaneous velocity v_event -- but each retains its
    OWN independent birth time and remaining quota (Section 9).

    Returns (f, particle, substrate, newly_completed_event_ids, diag).
    `diag` includes total_requested_dDelta, total_measured_relative_dDelta,
    relative_error, n_active, mass_conservation_residual, e1e2f_residual --
    the Section 11 request/measurement audit fields."""
    active_events = [e for e in events if e.active]
    diag_empty = dict(paused=True, n_active=0, total_requested_dDelta=0.0, total_measured_relative_dDelta=0.0,
                       relative_error=0.0, sigma_drive=max(0.0, sigma_s),
                       mass_conservation_residual=0.0, e1e2f_residual=float(np.max(np.abs(particle + substrate - f))))
    if not active_events:
        return f, particle, substrate, [], diag_empty

    sigma_drive = max(0.0, sigma_s)
    dt_seconds = dt * seconds_per_model_time
    if sigma_drive <= 0.0:
        return f, particle, substrate, [], dict(diag_empty, n_active=len(active_events), sigma_drive=sigma_drive)

    xd = 0.5 * hp.GS / 2
    tau_Coble = (xd * xd * hp.kB * hp.T) / (sigma_drive * hp.Omega * hp.D_gb) + hp.tau_ex0
    v_event = hp.b / tau_Coble  # identical for every active event (Section 9: identical kinetic parameters)

    requested = {}
    total_requested = 0.0
    for e in active_events:
        remaining = e.remaining(hp)
        d_req = min(v_event * dt_seconds, remaining)
        requested[e.event_id] = d_req
        total_requested += d_req

    if total_requested <= 1e-30:
        return f, particle, substrate, [], dict(diag_empty, n_active=len(active_events), sigma_drive=sigma_drive)

    v_total = total_requested / dt_seconds
    den = particle + substrate + 1e-30
    vz_field = (-v_total) * (particle / den)
    vmax = float(np.max(np.abs(vz_field)))
    if vmax < 1e-40:
        return f, particle, substrate, [], dict(diag_empty, n_active=len(active_events), sigma_drive=sigma_drive)

    n_sub = max(1, math.ceil(dt_seconds * vmax / dz / 0.4))
    ds = dt_seconds / n_sub
    com0_particle = particle_com_z(particle, z, r_c, dz, dz)
    com0_substrate = particle_com_z(substrate, z, r_c, dz, dz)
    mass_resid_max = 0.0

    for _ in range(n_sub):
        v_face = 0.5 * (vz_field + np.roll(vz_field, -1, axis=0))
        f_face = np.maximum(v_face, 0) * f + np.minimum(v_face, 0) * np.roll(f, -1, axis=0)
        f = f - ds * (f_face - np.roll(f_face, 1, axis=0)) / dz
        for arr in (particle, substrate):
            fw = (np.roll(arr, -1, axis=0) - arr) / dz
            bw = (arr - np.roll(arr, 1, axis=0)) / dz
            arr -= ds * vz_field * np.where(vz_field >= 0, bw, fw)
            np.clip(arr, 0.0, 1.0, out=arr)
        # M16P Section 7 fix (mirrors axisym_sink_rbm.active_sink_transport_step):
        # track both the f>1 excess AND the f<0 deficit before clipping,
        # and apply one combined (excess-minus-deficit) correction -- the
        # uncompensated negative-clip was previously the entire source of
        # the ~8.8e-6 relative per-event mass residual.
        excess = np.maximum(0.0, f - 1.0)
        deficit = np.maximum(0.0, -f)
        f = np.clip(f, 0.0, 1.0)
        V_excess = _axisym_weighted_sum(excess, r_c)
        V_deficit = _axisym_weighted_sum(deficit, r_c)
        V_net_correction = V_excess - V_deficit
        if abs(V_net_correction) > 1e-30:
            surf_weight = 16.0 * f * f * (1.0 - f) ** 2
            j_gb = int(np.argmin(np.abs(z - GB_z_hint)))
            sigma_cells = max(3.0, 2.0)
            gauss = np.exp(-0.5 * ((np.arange(f.shape[0])[:, None] - j_gb) / sigma_cells) ** 2)
            dep = surf_weight * gauss * np.ones((1, f.shape[1]))
            V_dep = _axisym_weighted_sum(dep, r_c)
            if V_dep <= 1e-30:
                dep = surf_weight
                V_dep = _axisym_weighted_sum(dep, r_c)
            if V_dep > 1e-30:
                f = f + dep / V_dep * V_net_correction
                mass_resid_max = max(mass_resid_max,
                                      abs(_axisym_weighted_sum(dep / V_dep * V_net_correction, r_c)
                                          - V_net_correction) / max(V_excess, V_deficit, 1e-30))

    com1_particle = particle_com_z(particle, z, r_c, dz, dz)
    com1_substrate = particle_com_z(substrate, z, r_c, dz, dz)
    measured_particle_d = (com0_particle - com1_particle) if (math.isfinite(com0_particle) and
                                                                math.isfinite(com1_particle)) else 0.0
    measured_substrate_d = (com0_substrate - com1_substrate) if (math.isfinite(com0_substrate) and
                                                                   math.isfinite(com1_substrate)) else 0.0
    measured_relative_d = max(0.0, measured_particle_d - measured_substrate_d)
    relative_error = (abs(measured_relative_d - total_requested) / total_requested
                       if total_requested > 1e-30 else 0.0)

    # M16N Section F correction (mirrors the single-event fix in
    # axisym_sink_rbm.active_sink_transport_step): each event's own
    # `delta` (its `delta_sink` quota) accumulates `requested[e.event_id]`
    # DIRECTLY -- the deterministic, already-correctly-capped analytic
    # Coble-rate amount the field was actually advected by this step --
    # NOT a share of `measured_relative_d` (the mesoscale COM response,
    # which legitimately differs from the sink-transport request due to
    # concurrent natural capillary relaxation, per the transport-only
    # microtest in M16N Section G). `measured_relative_d` remains a
    # diagnostic output (`total_measured_relative_dDelta` /
    # `relative_error` below) and is never fed back into `e.delta`.
    newly_completed = []
    for e in active_events:
        e.delta = min(hp.b, e.delta + requested[e.event_id])
        if e.delta >= hp.b - 1e-15:
            e.active = False
            newly_completed.append(e.event_id)

    diag = dict(paused=False, n_active=len(active_events), sigma_drive=sigma_drive, v_event=v_event,
                total_requested_dDelta=total_requested, total_measured_relative_dDelta=measured_relative_d,
                relative_error=relative_error, mass_conservation_residual=mass_resid_max,
                e1e2f_residual=float(np.max(np.abs(particle + substrate - f))))
    return f, particle, substrate, newly_completed, diag
