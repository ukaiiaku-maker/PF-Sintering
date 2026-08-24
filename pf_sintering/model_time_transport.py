"""Material-agnostic transport closures for the PR demonstration campaign.

The phase-field surface operator already advances in model time.  This module
puts grain-boundary delivery on that same clock without assigning a physical
seconds conversion or borrowing a material diffusivity.  The effective
``D_gb_model`` is derived from a predeclared reference event time and the
instantaneous local chemical-potential affinity remains the only event-time
driving force.

The historical physical-units transport functions are deliberately left
unchanged elsewhere in the package.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


KINETIC_TIME_BASIS = "model_time_demonstration"


def derive_model_time_gb_diffusivity(
        x_d_m: float, kB_J_per_K: float, temperature_K: float,
        atomic_volume_m3: float, reference_affinity_Pa: float,
        reference_tau_gb_model: float, tau_ex_model: float = 0.0) -> float:
    """Derive ``D_gb`` from a declared model-time reference timescale.

    This is an algebraic inversion of

    ``tau_gb = x_d^2 kBT / (Delta_mu Omega D_gb) + tau_ex``.

    ``reference_affinity_Pa`` is the instantaneous local GB-to-surface
    chemical-potential density difference at the declared reference state.
    """
    values = (
        x_d_m, kB_J_per_K, temperature_K, atomic_volume_m3,
        reference_affinity_Pa, reference_tau_gb_model)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0
           for value in values):
        raise ValueError("reference geometry, thermodynamics, affinity, and time must be positive")
    if not math.isfinite(float(tau_ex_model)) or tau_ex_model < 0.0:
        raise ValueError("tau_ex_model must be finite and non-negative")
    transport_time = reference_tau_gb_model - tau_ex_model
    if transport_time <= 0.0:
        raise ValueError("reference_tau_gb_model must exceed tau_ex_model")
    return (
        x_d_m ** 2 * kB_J_per_K * temperature_K
        / (reference_affinity_Pa * atomic_volume_m3 * transport_time))


@dataclass(frozen=True)
class ModelTimeGBTransport:
    """Coble-like GB delivery law expressed entirely per model time."""

    x_d_m: float
    kB_J_per_K: float
    temperature_K: float
    atomic_volume_m3: float
    b_m: float
    D_gb_m2_per_model_time: float
    tau_ex_model: float = 0.0
    kinetic_time_basis: str = KINETIC_TIME_BASIS

    def __post_init__(self):
        positive = (
            self.x_d_m, self.kB_J_per_K, self.temperature_K,
            self.atomic_volume_m3, self.b_m, self.D_gb_m2_per_model_time)
        if any(not math.isfinite(float(value)) or float(value) <= 0.0
               for value in positive):
            raise ValueError("GB transport parameters must be positive and finite")
        if not math.isfinite(float(self.tau_ex_model)) or self.tau_ex_model < 0.0:
            raise ValueError("tau_ex_model must be finite and non-negative")
        if self.kinetic_time_basis != KINETIC_TIME_BASIS:
            raise ValueError(f"kinetic_time_basis must be {KINETIC_TIME_BASIS!r}")

    @classmethod
    def from_reference_target(
            cls, *, x_d_m: float, kB_J_per_K: float, temperature_K: float,
            atomic_volume_m3: float, b_m: float,
            reference_affinity_Pa: float, reference_tau_gb_model: float,
            tau_ex_model: float = 0.0):
        diffusivity = derive_model_time_gb_diffusivity(
            x_d_m, kB_J_per_K, temperature_K, atomic_volume_m3,
            reference_affinity_Pa, reference_tau_gb_model, tau_ex_model)
        return cls(
            x_d_m=x_d_m, kB_J_per_K=kB_J_per_K,
            temperature_K=temperature_K, atomic_volume_m3=atomic_volume_m3,
            b_m=b_m, D_gb_m2_per_model_time=diffusivity,
            tau_ex_model=tau_ex_model)

    @property
    def K_gb_Pa_model_time(self) -> float:
        """Lumped coefficient in ``tau_gb=K_gb/Delta_mu+tau_ex``."""
        return (
            self.x_d_m ** 2 * self.kB_J_per_K * self.temperature_K
            / (self.atomic_volume_m3 * self.D_gb_m2_per_model_time))

    def tau_gb_model(self, instantaneous_affinity_Pa: float) -> float:
        """Return the current event time; nonpositive drive stops transport."""
        affinity = float(instantaneous_affinity_Pa)
        if not math.isfinite(affinity):
            raise ValueError("instantaneous affinity must be finite")
        if affinity <= 0.0:
            return math.inf
        return self.K_gb_Pa_model_time / affinity + self.tau_ex_model

    def qdot_m_per_model_time(self, instantaneous_affinity_Pa: float) -> float:
        tau = self.tau_gb_model(instantaneous_affinity_Pa)
        return 0.0 if math.isinf(tau) else self.b_m / tau

    def volume_rate_m3_per_model_time(
            self, instantaneous_affinity_Pa: float,
            instantaneous_contact_area_m2: float) -> float:
        area = float(instantaneous_contact_area_m2)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("instantaneous contact area must be positive and finite")
        return area * self.qdot_m_per_model_time(instantaneous_affinity_Pa)

    def diagnostics(self, instantaneous_affinity_Pa: float) -> dict:
        tau = self.tau_gb_model(instantaneous_affinity_Pa)
        return dict(
            kinetic_time_basis=self.kinetic_time_basis,
            instantaneous_affinity_Pa=float(instantaneous_affinity_Pa),
            D_gb_m2_per_model_time=self.D_gb_m2_per_model_time,
            K_gb_Pa_model_time=self.K_gb_Pa_model_time,
            tau_ex_model=self.tau_ex_model,
            tau_gb_model=tau,
            qdot_m_per_model_time=self.qdot_m_per_model_time(
                instantaneous_affinity_Pa),
            physical_seconds_conversion=None,
            material_calibration=None)


def mullins_pf_coefficient_m4_per_model_time(
        M_s_m6_per_J_model_time: float, gamma_s_J_per_m2: float) -> float:
    """Qualified diffuse-contour coefficient ``B_PF=(9/4) M_s gamma``."""
    if M_s_m6_per_J_model_time <= 0.0 or gamma_s_J_per_m2 <= 0.0:
        raise ValueError("M_s and gamma_s must be positive")
    return 9.0 * M_s_m6_per_J_model_time * gamma_s_J_per_m2 / 4.0


def surface_mobility_for_target_time(
        redistribution_length_m: float, gamma_s_J_per_m2: float,
        target_tau_surface_model: float) -> float:
    """Choose ``M_s`` from ``tau_surface=ell_s^4/B_PF``."""
    if (redistribution_length_m <= 0.0 or gamma_s_J_per_m2 <= 0.0
            or target_tau_surface_model <= 0.0):
        raise ValueError("length, gamma_s, and target surface time must be positive")
    return (
        4.0 * redistribution_length_m ** 4
        / (9.0 * gamma_s_J_per_m2 * target_tau_surface_model))


def surface_relaxation_time_model(
        redistribution_length_m: float, M_s_m6_per_J_model_time: float,
        gamma_s_J_per_m2: float) -> float:
    if redistribution_length_m <= 0.0:
        raise ValueError("redistribution length must be positive")
    return redistribution_length_m ** 4 / mullins_pf_coefficient_m4_per_model_time(
        M_s_m6_per_J_model_time, gamma_s_J_per_m2)


def predeclared_timescale_matrix(
        tau_load_model: float = 1.0,
        gb_over_load=(0.01, 0.03, 0.10),
        surface_over_gb=(0.1, 0.3, 1.0)) -> list[dict]:
    """Return the frozen 3x3 dimensionless demonstration design."""
    if tau_load_model <= 0.0:
        raise ValueError("tau_load_model must be positive")
    rows = []
    for gb_ratio in gb_over_load:
        for surface_ratio in surface_over_gb:
            if gb_ratio <= 0.0 or surface_ratio <= 0.0:
                raise ValueError("timescale ratios must be positive")
            tau_gb = tau_load_model * float(gb_ratio)
            rows.append(dict(
                kinetic_time_basis=KINETIC_TIME_BASIS,
                tau_load_model=float(tau_load_model),
                tau_gb_over_tau_load=float(gb_ratio),
                tau_surface_over_tau_gb=float(surface_ratio),
                tau_gb_model=tau_gb,
                tau_surface_model=tau_gb * float(surface_ratio)))
    return rows
