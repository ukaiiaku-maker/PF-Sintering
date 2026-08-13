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
