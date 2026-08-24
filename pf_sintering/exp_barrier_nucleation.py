"""Legacy pure-EXP hazard and the authoritative transferable EXP-floor law.

Historical production outputs used the pure exponential barrier from the
nanopillar paper, replacing (for nucleation ONLY) the earlier linear
`A0 - sigma*V0` hazard used in the soft-barrier/overlap-regime run:

    DeltaG(sigma, T) = G0(T) * exp[-a_exp * (sigma/sigma_c)^n]

with sigma clamped to its positive (densifying) part -- no negative-
stress enhancement of the barrier. The system-level birth intensity
keeps the EXISTING prefactor convention (no change, no arbitrary site
count):

    Gamma(sigma) = Gamma_pref * exp(-DeltaG(sigma)/(kB*T)),
    Gamma_pref = nu0 * (b/GS)^3.

The transferable creep calibration has the distinct authoritative form

    DeltaG_floor = G0 [f + (1-f) exp(-(sigma/sigma*)^n)].

It is represented separately by :class:`CreepExpFloorParams`; required fit
values have no invented defaults.  The parameterized first-passage wiring uses
the q-conjugate creep-equivalent stress and explicit physical clock/site-count
inputs, so it cannot silently fall back to the legacy pure-EXP calibration.

Coble one-b completion physics (tau_Coble, v_event, D_gb(T)) is
completely UNCHANGED and untouched by this module -- this module only
ever answers "what is the nucleation birth rate", never "how fast does
an already-active event complete".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

KB = 1.380649e-23
EV = 1.602176634e-19


@dataclass
class ExpBarrierParams:
    G0_eV: float = 0.642329
    a_exp: float = 6.65607
    sigma_c_pa: float = 2.96345e9
    n: float = 1.0
    nu0: float = 1e12
    GS_m: float = 201.74e-9
    b_m: float = 2.5e-10
    T: float = 1200.0


@dataclass(frozen=True)
class CreepExpFloorParams:
    """Authoritative transferable creep EXP-floor barrier parameters.

    Values are deliberately required: this recovery repository does not
    contain the authoritative creep fit and must not manufacture defaults.
    The legacy :class:`ExpBarrierParams` remains available for provenance of
    older production outputs, but is not equivalent when ``floor_fraction``
    is nonzero.
    """

    G0_eV: float
    sigma_star_pa: float
    floor_fraction: float
    n: float

    def __post_init__(self):
        if self.G0_eV <= 0.0:
            raise ValueError("G0_eV must be positive")
        if self.sigma_star_pa <= 0.0:
            raise ValueError("sigma_star_pa must be positive")
        if not 0.0 <= self.floor_fraction <= 1.0:
            raise ValueError("floor_fraction must lie in [0,1]")
        if self.n <= 0.0:
            raise ValueError("n must be positive")


@dataclass(frozen=True)
class CompleteExpFloorParams:
    """Complete EXP-floor law with an independent exponent scale ``a``.

    ``G_floor + (G0-G_floor) exp[-a (sigma/sigma_hat)^n]`` is kept in
    this literal form so a demonstration clock cannot accidentally drop the
    floor, absorb ``a`` into another parameter, or revert to the legacy pure
    exponential law.
    """

    G0_eV: float
    G_floor_eV: float
    a: float
    sigma_hat_pa: float
    n: float

    def __post_init__(self):
        if self.G0_eV <= 0.0:
            raise ValueError("G0_eV must be positive")
        if not 0.0 <= self.G_floor_eV <= self.G0_eV:
            raise ValueError("G_floor_eV must lie in [0,G0_eV]")
        if self.a <= 0.0:
            raise ValueError("a must be positive")
        if self.sigma_hat_pa <= 0.0:
            raise ValueError("sigma_hat_pa must be positive")
        if self.n <= 0.0:
            raise ValueError("n must be positive")


def delta_G_complete_exp_floor_eV(
        sigma_pa: float, p: CompleteExpFloorParams) -> float:
    """Evaluate the complete user-specified EXP-floor barrier."""
    sigma_eff = max(0.0, float(sigma_pa))
    shape = math.exp(-p.a * (sigma_eff / p.sigma_hat_pa) ** p.n)
    return p.G_floor_eV + (p.G0_eV - p.G_floor_eV) * shape


def tj_site_count(r_TJ_m: float, b_m: float) -> float:
    """Return the fixed circular-TJ site formula ``2*pi*r_TJ/b``."""
    if r_TJ_m <= 0.0:
        raise ValueError("r_TJ_m must be positive")
    if b_m <= 0.0:
        raise ValueError("b_m must be positive")
    return 2.0 * math.pi * float(r_TJ_m) / float(b_m)


def gamma_complete_exp_floor_per_model_time(
        sigma_pa: float, r_TJ_m: float, *, barrier: CompleteExpFloorParams,
        clock_scale_per_model_time: float, b_m: float,
        temperature_K: float) -> float:
    """Demonstration intensity on the explicitly labelled model-time clock.

    ``clock_scale_per_model_time`` is the sole scalar calibrated by the
    stochastic preflight.  It is not represented as a physical attempt
    frequency and introduces no model-time-to-seconds claim.
    """
    if clock_scale_per_model_time <= 0.0:
        raise ValueError("clock scale must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature must be positive")
    barrier_eV = delta_G_complete_exp_floor_eV(sigma_pa, barrier)
    kT_eV = KB * temperature_K / EV
    return (
        tj_site_count(r_TJ_m, b_m) * clock_scale_per_model_time
        * math.exp(-barrier_eV / kT_eV))


@dataclass(frozen=True)
class CreepExpFloorHazardParams:
    """Required physical inputs for the creep-floor first-passage rate.

    The attempt frequency is explicitly per physical second.  Callers using a
    model-time evolution must supply an independently justified seconds
    conversion before invoking the clock step.
    """

    barrier: CreepExpFloorParams
    attempt_frequency_per_s: float
    b_m: float
    temperature_K: float

    def __post_init__(self):
        if self.attempt_frequency_per_s <= 0.0:
            raise ValueError("attempt frequency must be positive")
        if self.b_m <= 0.0:
            raise ValueError("b_m must be positive")
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")


def creep_equivalent_stress_pa(Fq_N: float, dVsource_dq_m2: float) -> float:
    """Return ``Fq/(dV_source/dq)``, the q-conjugate equivalent stress."""
    if dVsource_dq_m2 <= 0.0:
        raise ValueError("dVsource/dq must be positive")
    return float(Fq_N) / float(dVsource_dq_m2)


def delta_G_creep_floor_eV(
        sigma_pa: float, p: CreepExpFloorParams) -> float:
    """Evaluate ``G0[f+(1-f) exp(-(sigma/sigma*)^n)]``.

    Stress is clamped to its non-negative activation part.  The same function
    is used by the parameterized sintering first-passage rate below.
    """
    sigma_eff = max(0.0, float(sigma_pa))
    shape = math.exp(-((sigma_eff / p.sigma_star_pa) ** p.n))
    return p.G0_eV * (p.floor_fraction + (1.0 - p.floor_fraction) * shape)


def activation_volume_creep_floor_m3(
        sigma_pa: float, p: CreepExpFloorParams) -> float:
    """Return ``-d(Delta G*)/d sigma`` from that same EXP-floor law.

    The derivative is zero below the non-densifying clamp.  Exactly at zero,
    the densifying-side derivative is reported.
    """
    if sigma_pa < 0.0 or p.floor_fraction == 1.0:
        return 0.0
    sigma_eff = max(0.0, float(sigma_pa))
    x = sigma_eff / p.sigma_star_pa
    if x == 0.0:
        power = 1.0 if p.n == 1.0 else (0.0 if p.n > 1.0 else math.inf)
    else:
        power = x ** (p.n - 1.0)
    derivative_eV_per_pa = (
        p.G0_eV * (1.0 - p.floor_fraction) * p.n
        * power * math.exp(-(x ** p.n)) / p.sigma_star_pa)
    return derivative_eV_per_pa * EV


def creep_floor_site_count(r_TJ_m: float, p: CreepExpFloorHazardParams) -> float:
    """Continuous first site-count model ``N_sites=2*pi*r_TJ/b``."""
    return tj_site_count(r_TJ_m, p.b_m)


def gamma_birth_creep_floor_per_s(
        sigma_act_pa: float, r_TJ_m: float,
        p: CreepExpFloorHazardParams) -> float:
    """Total first-passage intensity for the authoritative floor barrier."""
    barrier_eV = delta_G_creep_floor_eV(sigma_act_pa, p.barrier)
    kT_eV = KB * p.temperature_K / EV
    return (
        creep_floor_site_count(r_TJ_m, p)
        * p.attempt_frequency_per_s
        * math.exp(-barrier_eV / kT_eV))


@dataclass
class CreepFloorFirstPassageClock:
    """Persistent one-source clock, frozen after nucleation until reset."""

    hazard: float = 0.0
    threshold: float = 0.0
    last_gamma_per_s: float = 0.0
    triggered: bool = False


def reset_creep_floor_clock(clock: CreepFloorFirstPassageClock, rng) -> None:
    """Draw a new exponential threshold for the next sink-OFF interval."""
    clock.hazard = 0.0
    clock.threshold = float(rng.exponential())
    clock.last_gamma_per_s = 0.0
    clock.triggered = False


def creep_floor_first_passage_step(
        clock: CreepFloorFirstPassageClock, sigma_act_pa: float,
        r_TJ_m: float, dt_seconds: float, p: CreepExpFloorHazardParams,
        rng) -> bool:
    """Accumulate sink-OFF hazard and report a single nucleation crossing.

    A triggered clock remains frozen while its sink is active.  The cyclic
    caller must invoke :func:`reset_creep_floor_clock` only after the one-b
    event terminates and the sink returns to OFF.
    """
    if dt_seconds < 0.0:
        raise ValueError("dt_seconds must be non-negative")
    if clock.triggered:
        clock.last_gamma_per_s = 0.0
        return False
    if clock.threshold <= 0.0:
        clock.threshold = float(rng.exponential())
    gamma = gamma_birth_creep_floor_per_s(sigma_act_pa, r_TJ_m, p)
    clock.last_gamma_per_s = gamma
    clock.hazard += gamma * dt_seconds
    if clock.hazard < clock.threshold:
        return False
    clock.triggered = True
    return True


def delta_G_eV(sigma_pa: float, p: ExpBarrierParams) -> float:
    """DeltaG(sigma) in eV. sigma clamped to >=0 (Section 3: 'for
    sigma<=0, do not create a negative-stress enhancement' -- i.e. the
    barrier at non-densifying stress is just G0, not larger)."""
    sigma_eff = max(0.0, sigma_pa)
    return p.G0_eV * math.exp(-p.a_exp * (sigma_eff / p.sigma_c_pa) ** p.n)


def activation_volume_m3(sigma_pa: float, p: ExpBarrierParams) -> float:
    """v*(sigma) = -dDeltaG/dsigma, analytic derivative of the SAME
    barrier used for nucleation (Section 4: 'the derivative of the EXP
    barrier IS the activation volume', not a separate constant). For
    n=1: dDeltaG/dsigma = -DeltaG(sigma)*a_exp/sigma_c (sigma>0), so
    v*(sigma) = DeltaG(sigma)*a_exp/sigma_c. For general n, differentiate
    the same closed form."""
    sigma_eff = max(0.0, sigma_pa)
    dG = delta_G_eV(sigma_pa, p)
    if p.n == 1.0:
        slope_eV_per_pa = dG * p.a_exp / p.sigma_c_pa
    else:
        # d/dsigma[-a_exp*(sigma/sigma_c)^n] = -a_exp*n*sigma^(n-1)/sigma_c^n
        dexponent_dsigma = -p.a_exp * p.n * (sigma_eff ** (p.n - 1.0)) / (p.sigma_c_pa ** p.n)
        slope_eV_per_pa = -dG * dexponent_dsigma
    slope_J_per_pa = slope_eV_per_pa * EV
    return slope_J_per_pa  # J/Pa == m^3


def gamma_pref(p: ExpBarrierParams) -> float:
    return p.nu0 * (p.b_m / p.GS_m) ** 3


def gamma_birth(sigma_pa: float, p: ExpBarrierParams) -> float:
    """Gamma(sigma) = Gamma_pref * exp(-DeltaG(sigma)/(kB*T)). sigma may
    be negative (signed stress); only its positive part drives DeltaG
    reduction (delta_G_eV already clamps), and Gamma itself is always
    well-defined and positive (never artificially zeroed for sigma<=0 --
    unlike the OLD linear-barrier model, the EXP barrier does not blow up
    or go complex for negative sigma, it simply saturates at Gamma(0))."""
    if sigma_pa <= 0.0:
        sigma_pa = 0.0
    dG = delta_G_eV(sigma_pa, p)
    kT_eV = KB * p.T / EV
    return gamma_pref(p) * math.exp(-dG / kT_eV)


@dataclass
class ExpBirthClock:
    """Same persistent integrated-hazard state as
    `pf_sintering.m16m_multisink.PoissonBirthClock`, driven by
    `gamma_birth` instead of the linear-barrier `lambda_birth`."""
    hazard: float = 0.0
    threshold: float = 0.0
    last_gamma: float = 0.0
    consumed_thresholds: List[float] = field(default_factory=list)


def poisson_birth_step_exp(clock: ExpBirthClock, sigma_pa: float, dt_seconds: float, p: ExpBarrierParams,
                            rng) -> int:
    """Identical thinning-algorithm structure to
    `m16m_multisink.poisson_multisink_birth_step` (while-loop so >1 birth
    per step is handled correctly), driven by the EXP-barrier Gamma(sigma)
    instead of the linear-barrier rate. `dt_seconds` is ALREADY in real
    seconds (caller applies seconds_per_model_time, matching the existing
    convention elsewhere)."""
    if clock.threshold <= 0:
        clock.threshold = float(rng.exponential())
    g = gamma_birth(sigma_pa, p)
    clock.last_gamma = g
    clock.hazard += g * dt_seconds
    n_births = 0
    while clock.hazard >= clock.threshold:
        clock.hazard -= clock.threshold
        clock.consumed_thresholds.append(clock.threshold)
        clock.threshold = float(rng.exponential())
        n_births += 1
    return n_births


def integrated_hazard(t_array, sigma_array_pa, p: ExpBarrierParams):
    """H(t) = integral_0^t Gamma(sigma(t')) dt', via dense linear
    resampling (same approach as the earlier A0 calibration's
    hazard_integral) -- used for the ensemble first-passage check
    (Section 11), which resamples random Exp(1) thresholds against this
    SAME precomputed curve rather than rerunning PF each time."""
    import numpy as np
    gam = np.array([gamma_birth(s, p) for s in sigma_array_pa])
    H = np.concatenate([[0.0], np.cumsum(0.5 * (gam[:-1] + gam[1:]) * np.diff(t_array))])
    return H
