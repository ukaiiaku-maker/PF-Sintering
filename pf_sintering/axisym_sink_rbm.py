"""Milestone 16H: axisymmetric sink/hazard/RBM analogue.

The project's established sink/RBM machinery (pf_sintering/model.py's
`Sink`, `hazard_step`, `rbm`) is a Cartesian (x,y), two-separate-body
implementation, tightly coupled to that grid's specific geometry helpers
(overlap_col, contact_width, curvature, x-direction advection). There is
no axisymmetric port of it in this project prior to this milestone; all
of M16A-16G's axisymmetric work has been sink-OFF only.

This module is a DELIBERATELY SIMPLER axisymmetric analogue of the SAME
essential physics -- the established scope-reduction pattern this
project uses throughout the axisym path (see e.g. axisym.py's own
module docstrings for the same "simpler but physically faithful, not a
port of every Cartesian refinement" stance):

  Hazard: same Arrhenius, stress-lowered-barrier, first-passage
  structure as `model.hazard_step` --
      r_nuc = r0 * exp(-max(0, A0 - sigma*V0) / (kB*T))
      hazard += r_nuc*dt; event when hazard >= Exp(1) threshold.
  Once active: tau_sink = xd^2*kB*T/(sigma*Omega*D_gb) + tau_ex0 (same
  functional form as the Cartesian model), controlling the RBM rate.

  RBM: rigid axial advection of the PARTICLE (e1) toward the substrate,
  velocity field v_z(z,r) = -(b/tau_sink)*e1/(e1+e2+eps) (mirrors
  `model.rbm`'s vx=(-b/tau_sink)*(e2/den) exactly, with the grain-1/
  grain-2 roles swapped to match this project's axisym convention:
  particle=e1 here, vs. particle=e2 in the Cartesian "substrate"
  geometry). Upwind-advects f and (e1,e2) along z using the SAME
  discrete upwind scheme as model.rbm (just transposed from the x-axis
  to the z-axis), redistributes any excess mass (f>1, from advecting a
  finite-width interface) via a Gaussian deposit localized at the
  current GB/neck position (mirroring the Cartesian's neck-localized
  redistribution), tracks cumulative axial displacement, and completes
  the event (deactivating, resetting hazard) once cumulative_disp
  reaches one atomic step `b`.

  "Separation"/densification: since this project's one-contact
  particle/asperity geometry is a single CONNECTED body with one GB
  (not two initially-separate bodies with a gap, unlike the Cartesian
  "substrate" geometry), "particle/substrate separation" is tracked as
  the axial position of the particle's own center of mass (mirroring
  axisym.py's axisym_m15_com_z) relative to its t=0 reference --
  RBM advection of the particle toward the substrate directly decreases
  this quantity, giving a densification-strain proxy consistent with
  the Cartesian model's own cumulative_disp/GS-normalized convention.

DIAGNOSTIC/EVENT MECHANISM ONLY where it touches PF fields (the RBM
advection genuinely modifies f/e1/e2, exactly as the Cartesian rbm()
does -- this is not diagnostic-only, it is the sink's actual physical
effect) -- but the specific numerical parameters (A0, V0, r0) are NOT
taken from the Cartesian model's own calibration (which targets
sigma_target~75 MPa, far above this milestone's currently-observed
~1-3 MPa stress scale): per this milestone's explicit instruction
("First establish the cycle topology with a physically smooth hazard"
before matching the eventual ~30-100 MPa physical target), they are
chosen to produce a hazard rate that becomes significant once sigma_s
reaches this construction's OWN currently-observed stress range, so
the demonstration completes in practical wall-clock time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AxisymSink:
    active: bool = False
    hazard: float = 0.0
    threshold: float = 0.0
    tau_sink: float = math.inf
    current_disp: float = 0.0
    cumulative_disp: float = 0.0
    cumulative_strain: float = 0.0
    n_events: int = 0
    r_nuc: float = 0.0


@dataclass
class HazardParams:
    kB: float = 1.380649e-23
    T: float = 1000.0
    Omega: float = 1e-29
    b: float = 2.5e-10
    D_gb: float = 1e-3 * math.exp(-1.5e5 / (8.314 * 1000.0))
    GS: float = 200e-9  # characteristic diffusion length scale (order R_cyl)
    r0: float = 1e12  # attempt-frequency-like prefactor (1/s), same order as model.py's 1e12*(b/GS)^3 baseline
    A0: float = 0.0  # barrier (J), calibrated via calibrate_barrier()
    V0: float = 100 * 1e-29  # activation volume (m^3), same convention as model.py (100*Omega)
    tau_ex0: float = 0.0


def calibrate_barrier(sigma_target: float, hp: HazardParams) -> HazardParams:
    """Sets A0 (and tau_ex0) so the hazard rate becomes O(1) event over a
    "reasonable" number of diagnostic windows once sigma_s reaches
    sigma_target -- same structural calibration model.build_params uses
    for its own p.A0/p.tau_ex0, just evaluated at THIS milestone's much
    smaller sigma_target (order MPa, not the Cartesian default's 75 MPa)."""
    xd = 0.5 * hp.GS / 2
    tau = (xd * xd * hp.kB * hp.T) / (sigma_target * hp.Omega * hp.D_gb)
    gamma0 = hp.r0 * (hp.b / hp.GS) ** 3
    A0 = hp.kB * hp.T * math.log(gamma0 * (tau + 0.25 * tau)) + sigma_target * hp.V0
    return HazardParams(kB=hp.kB, T=hp.T, Omega=hp.Omega, b=hp.b, D_gb=hp.D_gb, GS=hp.GS,
                         r0=hp.r0, A0=A0, V0=hp.V0, tau_ex0=0.25 * tau)


def hazard_step(sink: AxisymSink, sigma_s: float, dt: float, hp: HazardParams, rng) -> bool:
    """One hazard-accumulation step. Returns True if a NEW event just
    activated this call. `sigma_s` may be negative (Eq. 1b is signed);
    only its positive part drives nucleation (a compressive/negative
    stress does not accelerate vacancy emission toward the sink in this
    model, mirroring model.hazard_step's `sigma=max(0,st.sigma)`)."""
    if sink.threshold <= 0:
        sink.threshold = float(rng.exponential())
    sigma = max(0.0, sigma_s)
    if sigma > 0:
        xd = 0.5 * hp.GS / 2
        sink.r_nuc = hp.r0 * (hp.b / hp.GS) ** 3 * math.exp(-max(0.0, hp.A0 - sigma * hp.V0) / (hp.kB * hp.T))
        tau = (xd * xd * hp.kB * hp.T) / (sigma * hp.Omega * hp.D_gb)
        sink.tau_sink = tau + hp.tau_ex0
    else:
        sink.r_nuc = 0.0
        sink.tau_sink = math.inf
    activated = False
    if not sink.active:
        sink.hazard += sink.r_nuc * dt
        if sink.hazard >= sink.threshold:
            sink.active = True
            sink.current_disp = 0.0
            sink.hazard = 0.0
            sink.threshold = float(rng.exponential())
            sink.n_events += 1
            activated = True
    return activated


def update_tau_sink(sink: AxisymSink, sigma_s: float, hp: HazardParams) -> None:
    """Updates ONLY sink.tau_sink from the current (signed) stress, same
    physical formula as hazard_step's stress-dependent branch, WITHOUT
    touching the hazard/threshold/activation bookkeeping -- used by the
    LOW-barrier regime, where the sink is forced permanently active
    (bypassing the Arrhenius hazard entirely, see
    scripts/m16h_three_regime_sink_barrier.py) but the RBM rate should
    still respond physically to the instantaneous stress."""
    sigma = max(0.0, sigma_s)
    if sigma > 0:
        xd = 0.5 * hp.GS / 2
        tau = (xd * xd * hp.kB * hp.T) / (sigma * hp.Omega * hp.D_gb)
        sink.tau_sink = tau + hp.tau_ex0
    else:
        sink.tau_sink = math.inf


def particle_com_z(e1, z, r_c, dr, dz):
    """Mass-weighted particle (grain1) center-of-mass z, same convention
    as axisym.axisym_m15_com_z (mirrored to grain1 instead of grain2)."""
    w = r_c[None, :] * e1
    denom = float(np.sum(w))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * z[:, None])) / denom


def rbm_step(f, e1, e2, sink: AxisymSink, hp: HazardParams, dt: float, dz: float, r_c, z, GB_z_hint):
    """One RBM advection step (only has an effect if sink.active and
    tau_sink finite). Mirrors model.rbm's structure: upwind advection of
    f and (e1,e2) along z (NEGATIVE z = toward the substrate crest, this
    project's convention), velocity weighted by the local particle
    fraction e1/(e1+e2), excess-mass (f>1) redistribution via a
    Gaussian deposit localized at the current GB/neck position. Returns
    (f, e1, e2, event_completed)."""
    if not sink.active or not math.isfinite(sink.tau_sink):
        return f, e1, e2, False
    den = e1 + e2 + 1e-30
    vz_field = (-hp.b / sink.tau_sink) * (e1 / den)  # (Nz,Nr), <=0 (particle moves toward smaller z)
    vmax = float(np.max(np.abs(vz_field)))
    if vmax < 1e-40:
        return f, e1, e2, False
    n_sub = max(1, math.ceil(dt * vmax / dz / 0.4))
    ds = dt / n_sub
    com0 = particle_com_z(e1, z, r_c, dz, dz)  # dr not needed for a ratio; pass dz as placeholder-safe unit

    for _ in range(n_sub):
        v_face = 0.5 * (vz_field + np.roll(vz_field, -1, axis=0))
        f_face = np.maximum(v_face, 0) * f + np.minimum(v_face, 0) * np.roll(f, -1, axis=0)
        f = f - ds * (f_face - np.roll(f_face, 1, axis=0)) / dz
        for arr in (e1, e2):
            fw = (np.roll(arr, -1, axis=0) - arr) / dz
            bw = (arr - np.roll(arr, 1, axis=0)) / dz
            arr -= ds * vz_field * np.where(vz_field >= 0, bw, fw)
            np.clip(arr, 0.0, 1.0, out=arr)
        excess = np.maximum(0.0, f - 1.0)
        f = np.clip(f, 0.0, 1.0)
        ex_sum = float(np.sum(excess))
        if ex_sum > 1e-30:
            surf_weight = 16.0 * f * f * (1.0 - f) ** 2
            j_gb = int(np.argmin(np.abs(z - GB_z_hint)))
            sigma_cells = max(3.0, 2.0)  # a few grid cells, matching model.rbm's "2*interface_width/dx" order
            gauss = np.exp(-0.5 * ((np.arange(f.shape[0])[:, None] - j_gb) / sigma_cells) ** 2)
            dep = surf_weight * gauss * np.ones((1, f.shape[1]))
            dsum = float(np.sum(dep))
            if dsum <= 1e-30:
                dep = surf_weight
                dsum = float(np.sum(dep))
            if dsum > 1e-30:
                f = f + dep / dsum * ex_sum

    com1 = particle_com_z(e1, z, r_c, dz, dz)
    d = max(0.0, com0 - com1) if math.isfinite(com0) and math.isfinite(com1) else 0.0
    sink.current_disp += d
    sink.cumulative_disp += d
    completed = False
    if sink.current_disp >= hp.b:
        sink.active = False
        sink.current_disp = 0.0
        sink.hazard = 0.0
        completed = True
    return f, e1, e2, completed


# =============================================================================
# Milestone 16K continuation: corrected post-nucleation event physics.
#
# The original `hazard_step`+`rbm_step` pair conflated two distinct physical
# questions (does a NEW event nucleate? vs. how does an ALREADY-nucleated
# event propagate?) and had two confirmed bugs when exercised at the ~45 MPa
# stress scale (an order of magnitude above M16H/M16I's original ~1-3 MPa
# calibration target, where these issues had negligible practical effect):
#
#  (1) axisymmetric mass-conservation bug in the excess-mass redistribution:
#      `ex_sum`/`dsum` above are PLAIN, UNWEIGHTED array sums over the
#      (z,r) grid. For an axisymmetric field the correct volume element is
#      2*pi*r*dr*dz -- a cell's r-position must weight its contribution.
#      Since the excess mass and the deposit pattern generally have
#      DIFFERENT r-distributions, the unweighted normalization used above
#      does not exactly conserve physical volume in general. Fixed below
#      via `_axisym_weighted_sum`.
#
#  (2) `tau_sink=inf` whenever sigma<=0 was being (implicitly, via the
#      unmodified `rbm_step` above) used as the ONLY rule governing an
#      ALREADY-ACTIVE event's propagation, not just new-nucleation
#      suppression -- appropriate for (1) but not automatically valid for
#      (2). Separated below into `nucleation_hazard_step` (decides ONLY
#      whether a new event nucleates) and `active_sink_transport_step`
#      (governs an already-nucleated event's propagation under an
#      explicit physical contract: at most ONE Burgers-vector `b` of
#      total rigid-body displacement per event, propagating at a
#      finite, stress-dependent Coble-type rate that SLOWS as the local
#      stress relaxes and PAUSES -- never force-completes -- if the
#      driving stress reaches zero before `b` is reached).
#
# TIME-UNIT CAVEAT (Section 7/10 audit, unresolved): `dt` here is the same
# PF stepping variable used throughout M16H-M16K, treated as if it were
# literally SI seconds when combined with tau_sink/D_gb (which ARE real
# SI quantities). Investigation of `model.py`'s M_f/M_s calibration
# (`M_f_base=(20e-9)**4/(tau_target*k_f)`) found the PF surface-diffusion
# mobility is set by a CHOSEN numerical relaxation timescale `tau_target`,
# not a real atomistic surface diffusivity -- so "1 model time unit = 1
# real second" is NOT demonstrated, only assumed (as it has been,
# implicitly, since M16H). `SECONDS_PER_MODEL_TIME` below makes this
# assumption an explicit, named, overridable parameter (currently kept at
# 1.0, preserving existing behavior) rather than leaving it silently
# buried, per the "fail closed" instruction -- a full resolution would
# require calibrating M_s/M_eta against a real atomistic mobility, out of
# scope for this continuation.
# =============================================================================

SECONDS_PER_MODEL_TIME = 1.0  # UNVALIDATED, see caveat above.


def _axisym_weighted_sum(field, r_c):
    """sum(r_c[None,:]*field) -- proportional to the true axisymmetric
    volume represented by `field` (the constant 2*pi*dr*dz factor is
    omitted since callers only ever use RATIOS of these sums, in which
    that constant cancels)."""
    return float(np.sum(r_c[None, :] * field))


def nucleation_hazard_step(sink: AxisymSink, sigma_s: float, dt: float, hp: HazardParams, rng,
                            seconds_per_model_time: float = SECONDS_PER_MODEL_TIME) -> bool:
    """M16K Section 5: NUCLEATION ONLY. Decides whether a NEW sink event
    nucleates; does nothing if one is already active (an active event's
    propagation is governed entirely by `active_sink_transport_step`, not
    by this function). Same Arrhenius first-passage structure as the
    original `hazard_step` for the nucleation decision itself."""
    if sink.active:
        return False
    if sink.threshold <= 0:
        sink.threshold = float(rng.exponential())
    sigma = max(0.0, sigma_s)
    dt_seconds = dt * seconds_per_model_time
    sink.r_nuc = (hp.r0 * (hp.b / hp.GS) ** 3 * math.exp(-max(0.0, hp.A0 - sigma * hp.V0) / (hp.kB * hp.T))
                  if sigma > 0 else 0.0)
    sink.hazard += sink.r_nuc * dt_seconds
    if sink.hazard >= sink.threshold:
        sink.active = True
        sink.current_disp = 0.0  # delta_event: progress toward this event's own b-quota
        sink.hazard = 0.0
        sink.threshold = float(rng.exponential())
        sink.n_events += 1
        return True
    return False


def active_sink_transport_step(f, e1, e2, sink: AxisymSink, hp: HazardParams, sigma_s: float, dt: float, dz: float,
                                r_c, z, GB_z_hint, seconds_per_model_time: float = SECONDS_PER_MODEL_TIME):
    """M16K/M16L Sections 1-8: propagates an ALREADY-nucleated event by AT
    MOST one physical timestep's worth of Coble-type diffusional
    advection, capped so the event's cumulative displacement
    (`sink.current_disp`, playing the role of `delta_event`) NEVER
    exceeds one Burgers vector `hp.b` -- and, per the M16L correctness
    fix below, the FIELD MUTATION ITSELF is scaled to the capped amount,
    not merely the bookkeeping variable (see `frac` below: the M16K
    version computed `d_delta_requested=min(v_event*dt, remaining)` for
    reporting purposes but then always advected for the FULL `dt`,
    relying on a post-hoc `min(measured_d, remaining)` clip that left the
    FIELDS mutated by more than `remaining` whenever an event was close
    to completion -- this violates the one-b contract at the field level
    even though the reported `delta_event` never exceeded `b`. Fixed by
    scaling the advection SUBSTEP DURATION itself by
    `frac=d_delta_requested/(v_event*dt_seconds)` before ever touching
    f/e1/e2, so a request for `epsilon` produces a field mutation of
    `epsilon`, not `v_event*dt_seconds` silently clipped afterward.).

    Uses `sigma_drive=max(sigma_s,0)` (never abs(), never a floor) -- if
    the local stress has relaxed to <=0, the event PAUSES (v_event=0,
    sink stays active, no forced completion) rather than stalling forever
    OR being forced through; a later PF-coarsening-driven stress recovery
    can resume the same event. Volume-weighted (not raw-summed)
    axisymmetric mass conservation in the excess-mass redistribution.
    Tracks BOTH particle (e1) and substrate (e2) center-of-mass so the
    reported displacement is the RELATIVE particle-substrate motion
    (Section 6/7), not merely the particle's own absolute COM shift
    (which would silently absorb any spurious substrate drift from the
    shared advection velocity field near the neck).

    M16N Section F correction: the event's progress quota
    (`sink.current_disp`, `delta_event`) is now driven by `delta_sink`
    (== `d_delta_requested`, the analytic Coble-rate quantity the field
    was ACTUALLY advected by this step -- deterministic, monotonic,
    never gated on how the mesoscale morphology happens to respond) --
    NOT by `delta_COM` (`measured_relative_d_delta`, the measured
    particle-relative-substrate COM response), which is retained purely
    as a diagnostic output. Previously `sink.current_disp` accumulated
    `applied_d = min(measured_relative_d, remaining)`, conflating the
    microscopic sink-transport quota with a mesoscale response
    diagnostic that a full-lifetime audit showed smoothly declines from
    ~1.0x to ~0.47x the requested amount over an event's lifetime (a
    real, reproducible morphology-response effect, not noise or a bug --
    but the wrong quantity to gate a "one Burgers vector per event"
    contract on).

    Returns (f, e1, e2, completed, diag) where diag is a dict with
    requested_d_delta (== delta_sink_this_step, what the field was
    advected by and what the event's quota accumulates),
    measured_particle_COM_d_delta, measured_substrate_COM_d_delta,
    delta_COM_this_step (== measured_relative_d_delta, diagnostic only),
    applied_d_delta (== delta_sink_this_step, kept for backward
    compatibility with existing readers), delta_event, remaining_to_b,
    tau_Coble, v_event, sigma_drive, mass_conservation_residual,
    e1e2f_residual."""
    if not sink.active:
        return f, e1, e2, False, dict(paused=False, active=False)

    sigma_drive = max(0.0, sigma_s)
    dt_seconds = dt * seconds_per_model_time
    remaining = max(0.0, hp.b - sink.current_disp)

    if sigma_drive <= 0.0 or remaining <= 0.0:
        # event PAUSED (Section 6): no forced completion, no advection,
        # sink stays active so PF coarsening can rebuild positive stress
        # and let this same event resume later.
        return f, e1, e2, False, dict(paused=True, active=True, sigma_drive=sigma_drive,
                                       tau_Coble=math.inf, v_event=0.0, requested_d_delta=0.0,
                                       measured_particle_COM_d_delta=0.0, measured_substrate_COM_d_delta=0.0,
                                       measured_relative_d_delta=0.0, applied_d_delta=0.0,
                                       delta_event=sink.current_disp,
                                       remaining_to_b=remaining, mass_conservation_residual=0.0,
                                       e1e2f_residual=float(np.max(np.abs(e1 + e2 - f))))

    xd = 0.5 * hp.GS / 2
    tau_Coble = (xd * xd * hp.kB * hp.T) / (sigma_drive * hp.Omega * hp.D_gb) + hp.tau_ex0
    v_event = hp.b / tau_Coble
    d_delta_full = v_event * dt_seconds
    d_delta_requested = min(d_delta_full, remaining)
    # M16L correctness fix: scale the ADVECTION DURATION (not just the
    # bookkeeping) by the fraction of the full step that's actually
    # allowed, so the field mutation matches the request exactly rather
    # than being clipped after the fact.
    frac = (d_delta_requested / d_delta_full) if d_delta_full > 1e-300 else 0.0
    frac = min(1.0, max(0.0, frac))
    dt_scaled_seconds = dt_seconds * frac

    den = e1 + e2 + 1e-30
    vz_field = (-v_event) * (e1 / den)  # <=0, same convention as before but driven by v_event, not b/tau_sink directly
    vmax = float(np.max(np.abs(vz_field)))
    if vmax < 1e-40 or dt_scaled_seconds <= 0.0:
        return f, e1, e2, False, dict(paused=True, active=True, sigma_drive=sigma_drive, tau_Coble=tau_Coble,
                                       v_event=v_event, requested_d_delta=d_delta_requested,
                                       measured_particle_COM_d_delta=0.0, measured_substrate_COM_d_delta=0.0,
                                       measured_relative_d_delta=0.0, applied_d_delta=0.0,
                                       delta_event=sink.current_disp,
                                       remaining_to_b=remaining, mass_conservation_residual=0.0,
                                       e1e2f_residual=float(np.max(np.abs(e1 + e2 - f))))

    n_sub = max(1, math.ceil(dt_scaled_seconds * vmax / dz / 0.4))
    ds = dt_scaled_seconds / n_sub
    com0_particle = particle_com_z(e1, z, r_c, dz, dz)
    com0_substrate = particle_com_z(e2, z, r_c, dz, dz)
    mass_resid_max = 0.0

    for _ in range(n_sub):
        v_face = 0.5 * (vz_field + np.roll(vz_field, -1, axis=0))
        f_face = np.maximum(v_face, 0) * f + np.minimum(v_face, 0) * np.roll(f, -1, axis=0)
        f = f - ds * (f_face - np.roll(f_face, 1, axis=0)) / dz
        for arr in (e1, e2):
            fw = (np.roll(arr, -1, axis=0) - arr) / dz
            bw = (arr - np.roll(arr, 1, axis=0)) / dz
            arr -= ds * vz_field * np.where(vz_field >= 0, bw, fw)
            np.clip(arr, 0.0, 1.0, out=arr)
        excess = np.maximum(0.0, f - 1.0)
        f = np.clip(f, 0.0, 1.0)
        V_excess = _axisym_weighted_sum(excess, r_c)  # FIX: volume-weighted, not raw np.sum
        if V_excess > 1e-30:
            surf_weight = 16.0 * f * f * (1.0 - f) ** 2
            j_gb = int(np.argmin(np.abs(z - GB_z_hint)))
            sigma_cells = max(3.0, 2.0)
            gauss = np.exp(-0.5 * ((np.arange(f.shape[0])[:, None] - j_gb) / sigma_cells) ** 2)
            dep = surf_weight * gauss * np.ones((1, f.shape[1]))
            V_dep = _axisym_weighted_sum(dep, r_c)  # FIX: volume-weighted, not raw np.sum
            if V_dep <= 1e-30:
                dep = surf_weight
                V_dep = _axisym_weighted_sum(dep, r_c)
            if V_dep > 1e-30:
                f = f + dep / V_dep * V_excess
                mass_resid_max = max(mass_resid_max,
                                      abs(_axisym_weighted_sum(dep / V_dep * V_excess, r_c) - V_excess) / V_excess)

    com1_particle = particle_com_z(e1, z, r_c, dz, dz)
    com1_substrate = particle_com_z(e2, z, r_c, dz, dz)
    measured_particle_d = (com0_particle - com1_particle) if (math.isfinite(com0_particle) and
                                                                math.isfinite(com1_particle)) else 0.0
    measured_substrate_d = (com0_substrate - com1_substrate) if (math.isfinite(com0_substrate) and
                                                                   math.isfinite(com1_substrate)) else 0.0
    # M16N Section F correction: delta_COM (the mesoscale particle-
    # relative-substrate COM response) is a DIAGNOSTIC OUTPUT describing
    # how the morphology responded to this step's sink-mediated
    # advection -- it is NOT the event's progress variable. M16L/M16M's
    # dense-history audit found delta_COM/d_delta_requested smoothly
    # declining from ~1.0 to ~0.47 over a single event's lifetime; using
    # delta_COM to gate completion (the previous `applied_d`/
    # `sink.current_disp += applied_d` below) meant the field was
    # ALWAYS advected by exactly `d_delta_requested` (the frac-scaled
    # substep loop above guarantees this) while the completion counter
    # silently ran on a DIFFERENT, smaller number -- conflating the
    # microscopic sink-transport quota with a mesoscale response
    # diagnostic. Fixed: `delta_sink` (the event's actual quota
    # variable, `sink.current_disp`) now accumulates `d_delta_requested`
    # -- the quantity the field was ACTUALLY advected by this step
    # (already correctly capped so a single event never exceeds `b`) --
    # and `delta_COM` (measured_relative_d) is reported purely as an
    # output, never fed back into `sink.current_disp`.
    measured_relative_d = max(0.0, measured_particle_d - measured_substrate_d)
    delta_sink_this_step = d_delta_requested
    sink.current_disp += delta_sink_this_step
    sink.cumulative_disp += delta_sink_this_step

    completed = False
    if sink.current_disp >= hp.b - 1e-15:
        sink.active = False
        sink.current_disp = 0.0
        sink.hazard = 0.0
        completed = True

    diag = dict(paused=False, active=sink.active, sigma_drive=sigma_drive, tau_Coble=tau_Coble, v_event=v_event,
                requested_d_delta=d_delta_requested, delta_sink_this_step=delta_sink_this_step,
                measured_particle_COM_d_delta=measured_particle_d,
                measured_substrate_COM_d_delta=measured_substrate_d, delta_COM_this_step=measured_relative_d,
                measured_relative_d_delta=measured_relative_d,  # kept for backward-compat with existing readers
                applied_d_delta=delta_sink_this_step,
                delta_event=(0.0 if completed else sink.current_disp), remaining_to_b=max(0.0, hp.b - sink.current_disp),
                mass_conservation_residual=mass_resid_max, e1e2f_residual=float(np.max(np.abs(e1 + e2 - f))),
                completed=completed)
    return f, e1, e2, completed, diag
