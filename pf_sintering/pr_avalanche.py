"""Persistent-source avalanche clock layered outside the qualified PR 1b solver.

This module contains no phase-field evolution.  It owns only the facilitated
descendant barrier, single-source integrated-hazard bookkeeping, serialized
child activation, and correlation-window extinction.  The elementary
transport remains the existing qualified one-b event.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable

from .exp_barrier_nucleation import (
    EV,
    KB,
    CompleteExpFloorParams,
    delta_G_complete_exp_floor_eV,
    tj_site_count,
)


@dataclass(frozen=True)
class DescendantBarrier:
    """Fixed decrement from the instantaneous authoritative root barrier."""

    root: CompleteExpFloorParams
    delta_G_step_eV: float

    def __post_init__(self) -> None:
        if (not math.isfinite(self.delta_G_step_eV)
                or self.delta_G_step_eV < 0.0):
            raise ValueError("delta_G_step_eV must be non-negative and finite")

    @property
    def floor_fraction(self) -> float:
        return self.root.G_floor_eV / self.root.G0_eV

    def barrier_eV(
            self, sigma_local_Pa: float, source_amplitude: float = 1.0) -> float:
        amplitude = float(source_amplitude)
        if not math.isfinite(amplitude) or amplitude < 0.0:
            raise ValueError("source_amplitude must be finite and non-negative")
        root_eV = delta_G_complete_exp_floor_eV(
            sigma_local_Pa, self.root)
        return max(
            self.root.G_floor_eV,
            root_eV - amplitude * self.delta_G_step_eV)

    def minus_dbarrier_dsigma_eV_per_Pa(
            self, sigma_local_Pa: float,
            source_amplitude: float = 1.0) -> float:
        """Return ``-dGdesc/dsigma`` on the active EXP-floor branch."""
        sigma = max(float(sigma_local_Pa), 0.0)
        raw = delta_G_complete_exp_floor_eV(sigma, self.root)
        facilitated = raw - float(source_amplitude) * self.delta_G_step_eV
        if facilitated <= self.root.G_floor_eV:
            return 0.0
        if sigma == 0.0:
            return math.inf if self.root.n < 1.0 else (
                (self.root.G0_eV - self.root.G_floor_eV)
                * self.root.a / self.root.sigma_hat_pa
                if self.root.n == 1.0 else 0.0)
        reduced = sigma / self.root.sigma_hat_pa
        exponential = math.exp(-self.root.a * reduced ** self.root.n)
        return (
            (self.root.G0_eV - self.root.G_floor_eV) * exponential
            * self.root.a * self.root.n * reduced ** (self.root.n - 1.0)
            / self.root.sigma_hat_pa)


def descendant_rate_per_s(
        sigma_local_Pa: float, r_TJ_m: float, *,
        barrier: DescendantBarrier, temperature_K: float,
        attempt_frequency_per_s: float, b_m: float,
        source_amplitude: float = 1.0) -> float:
    """Total axisymmetric-ring descendant rate.

    The site population is exactly ``2*pi*r_TJ/b``.  There is no angular
    discretization and no DDD/disconnection elastic-stress contribution.
    """
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if attempt_frequency_per_s <= 0.0:
        raise ValueError("attempt_frequency_per_s must be positive")
    G_eV = barrier.barrier_eV(sigma_local_Pa, source_amplitude)
    kT_eV = KB * float(temperature_K) / EV
    return (
        tj_site_count(r_TJ_m, b_m) * float(attempt_frequency_per_s)
        * math.exp(-G_eV / kT_eV)
    )


@dataclass
class AvalancheState:
    avalanche_active: bool = False
    avalanche_id: int = 0
    root_cycle: int = 0
    child_number: int = 0
    S_completed: int = 0
    q_avalanche_over_b: float = 0.0
    descendant_threshold: float = math.nan
    descendant_hazard: float = 0.0
    # Retained in output schemas for compatibility; the persistent-source
    # controller never queues more than the one activation being launched.
    pending_children: int = 0
    delta_G_step_eV: float = math.nan
    source_amplitude: float = 1.0
    facilitation_decay_alpha: float = 1.0
    avalanche_start_time_s: float = math.nan
    last_child_completion_time_s: float = math.nan
    window_start_time_s: float = math.nan
    window_deadline_s: float = math.nan
    window_triggered: bool = False
    descendant_total_hazard: float = 0.0
    thresholds_drawn: list[float] = field(default_factory=list)


class AvalancheController:
    """One persistent facilitated source with serialized one-b activations."""

    def __init__(
            self, *, barrier: DescendantBarrier, temperature_K: float,
            attempt_frequency_per_s: float, b_m: float,
            correlation_time_s: float, rng,
            facilitation_decay_alpha: float = 1.0) -> None:
        if correlation_time_s <= 0.0:
            raise ValueError("correlation_time_s must be positive")
        self.barrier = barrier
        self.temperature_K = float(temperature_K)
        self.attempt_frequency_per_s = float(attempt_frequency_per_s)
        self.b_m = float(b_m)
        self.correlation_time_s = float(correlation_time_s)
        self.facilitation_decay_alpha = float(facilitation_decay_alpha)
        if (not math.isfinite(self.facilitation_decay_alpha)
                or not 0.0 < self.facilitation_decay_alpha <= 1.0):
            raise ValueError("facilitation_decay_alpha must be in (0,1]")
        self.rng = rng
        self.state = AvalancheState(
            delta_G_step_eV=barrier.delta_G_step_eV,
            facilitation_decay_alpha=self.facilitation_decay_alpha)
        self.crossings: list[dict[str, Any]] = []

    def _draw_threshold(self) -> float:
        value = float(self.rng.exponential())
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("RNG returned an invalid exponential threshold")
        value = max(value, float.fromhex("0x1.0p-1022"))
        self.state.thresholds_drawn.append(value)
        return value

    def start(self, *, avalanche_id: int, root_cycle: int,
              start_time_s: float) -> None:
        if self.state.avalanche_active:
            raise RuntimeError("an avalanche is already active")
        self.state = AvalancheState(
            avalanche_active=True,
            avalanche_id=int(avalanche_id),
            root_cycle=int(root_cycle),
            child_number=0,
            S_completed=0,
            q_avalanche_over_b=0.0,
            descendant_hazard=0.0,
            pending_children=0,
            delta_G_step_eV=self.barrier.delta_G_step_eV,
            source_amplitude=1.0,
            facilitation_decay_alpha=self.facilitation_decay_alpha,
            avalanche_start_time_s=float(start_time_s),
            last_child_completion_time_s=math.nan,
        )
        self.crossings = []

    def rate(self, sigma_local_Pa: float, r_TJ_m: float) -> float:
        return descendant_rate_per_s(
            sigma_local_Pa, r_TJ_m, barrier=self.barrier,
            temperature_K=self.temperature_K,
            attempt_frequency_per_s=self.attempt_frequency_per_s,
            b_m=self.b_m,
            source_amplitude=self.state.source_amplitude)

    def barrier_eV(self, sigma_local_Pa: float) -> float:
        return self.barrier.barrier_eV(
            sigma_local_Pa, self.state.source_amplitude)

    def stress_drop_feedback(
            self, sigma_local_Pa: float, r_TJ_m: float,
            drops_MPa=(1.0, 5.0, 10.0)) -> dict:
        sigma = float(sigma_local_Pa)
        barrier_now = self.barrier_eV(sigma)
        rate_now = self.rate(sigma, r_TJ_m)
        rows = {}
        for drop_MPa in drops_MPa:
            sigma_drop = max(sigma - float(drop_MPa) * 1.0e6, 0.0)
            barrier_drop = self.barrier_eV(sigma_drop)
            rate_drop = self.rate(sigma_drop, r_TJ_m)
            rows[f"{float(drop_MPa):g}_MPa"] = dict(
                barrier_change_eV=barrier_drop - barrier_now,
                rate_after_drop_per_s=rate_drop,
                rate_ratio=(rate_drop / rate_now if rate_now > 0.0 else math.nan))
        return dict(
            minus_dGdesc_dsigma_eV_per_Pa=(
                self.barrier.minus_dbarrier_dsigma_eV_per_Pa(
                    sigma, self.state.source_amplitude)),
            drops=rows)

    def begin_transit(self) -> int:
        if not self.state.avalanche_active:
            raise RuntimeError("cannot begin a transit outside an avalanche")
        # child_number=0 denotes the root transit; descendants are 1,2,...
        self.state.child_number = self.state.S_completed
        return self.state.child_number

    def integrate_transit_samples(
            self, samples: Iterable[dict[str, float]]) -> list[dict[str, float]]:
        """Record transit rates diagnostically while the source clock is frozen."""
        values = [dict(sample) for sample in samples]
        if len(values) < 2:
            raise ValueError("at least two transit samples are required")
        times = [float(item["t_s"]) for item in values]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("transit sample times must increase strictly")
        trace: list[dict[str, float]] = []
        for sample in values:
            rate = self.rate(sample["sigma_local_Pa"], sample["r_TJ_m"])
            trace.append(self._trace_row(sample, rate, 0))
        return trace

    def _trace_row(self, sample: dict[str, float], rate: float,
                   crossings: int) -> dict[str, float]:
        threshold = self.state.descendant_threshold
        return dict(
            **sample,
            descendant_rate_per_s=float(rate),
            G_desc_star_eV=self.barrier_eV(sample["sigma_local_Pa"]),
            delta_G_step_eV=self.barrier.delta_G_step_eV,
            source_amplitude=self.state.source_amplitude,
            delta_G_facilitation_eV=(
                self.state.source_amplitude
                * self.barrier.delta_G_step_eV),
            descendant_hazard=self.state.descendant_hazard,
            descendant_threshold=threshold,
            H_over_threshold=self.state.descendant_hazard / threshold,
            descendant_total_hazard=self.state.descendant_total_hazard,
            pending_children=self.state.pending_children,
            crossings_in_segment=int(crossings),
            avalanche_active=int(self.state.avalanche_active),
            S_completed=self.state.S_completed,
            child_number=self.state.child_number,
        )

    def complete_transit(self, completion_time_s: float) -> None:
        completed_was_descendant = self.state.S_completed >= 1
        self.state.S_completed += 1
        self.state.q_avalanche_over_b = float(self.state.S_completed)
        self.state.last_child_completion_time_s = float(completion_time_s)
        if completed_was_descendant:
            self.state.source_amplitude *= self.facilitation_decay_alpha
        self.restart_correlation_window(completion_time_s)

    def restart_correlation_window(self, start_time_s: float) -> None:
        """Reset the one-source clock after a completed one-b event."""
        self.state.descendant_hazard = 0.0
        self.state.descendant_threshold = self._draw_threshold()
        self.state.pending_children = 0
        self.state.window_start_time_s = float(start_time_s)
        self.state.window_deadline_s = (
            float(start_time_s) + self.correlation_time_s)
        self.state.window_triggered = False

    def accumulate_window_segment(
            self, *, rate_start_per_s: float, rate_end_per_s: float,
            start_time_s: float, end_time_s: float) -> dict[str, float | bool]:
        """Advance one evolving-state facilitated-window hazard segment.

        At most one activation can be committed.  Once crossed, the source
        clock freezes until the resulting serialized one-b event completes.
        """
        if self.state.window_triggered:
            raise RuntimeError("facilitated source already triggered")
        if not math.isfinite(self.state.descendant_threshold):
            raise RuntimeError("facilitated window has not been initialized")
        if end_time_s <= start_time_s:
            raise ValueError("window segment times must increase")
        if rate_start_per_s < 0.0 or rate_end_per_s < 0.0:
            raise ValueError("descendant rates must be non-negative")
        dt = float(end_time_s) - float(start_time_s)
        increment = 0.5 * (
            float(rate_start_per_s) + float(rate_end_per_s)) * dt
        before = self.state.descendant_hazard
        threshold = self.state.descendant_threshold
        crossed = before + increment >= threshold
        if crossed:
            needed = max(threshold - before, 0.0)
            fraction = 0.0 if increment == 0.0 else min(
                max(needed / increment, 0.0), 1.0)
            crossing_time = float(start_time_s) + fraction * dt
            self.commit_crossing(crossing_time_s=crossing_time)
            return dict(
                crossed=True, crossing_time_s=crossing_time,
                increment=needed)
        self.state.descendant_hazard += increment
        self.state.descendant_total_hazard += increment
        return dict(
            crossed=False, crossing_time_s=math.nan, increment=increment)

    def commit_crossing(self, *, crossing_time_s: float) -> None:
        """Commit the unique child crossing at an already-located time."""
        if self.state.window_triggered:
            raise RuntimeError("facilitated source already triggered")
        if not (self.state.window_start_time_s <= float(crossing_time_s)
                <= self.state.window_deadline_s):
            raise ValueError("crossing lies outside the facilitated-source window")
        threshold = self.state.descendant_threshold
        needed = max(threshold-self.state.descendant_hazard, 0.0)
        self.state.descendant_hazard = threshold
        self.state.descendant_total_hazard += needed
        self.state.window_triggered = True
        self.crossings.append(dict(
            avalanche_id=self.state.avalanche_id,
            committed_child_index=self.state.S_completed + 1,
            crossing_time_s=float(crossing_time_s),
            source="facilitated_source_window",
            crossed_threshold=threshold,
            pending_children_after=0))

    def expire_window(self, end_time_s: float) -> None:
        if self.state.window_triggered:
            raise RuntimeError("cannot expire a triggered facilitated window")
        tolerance = 64.0 * math.ulp(max(abs(self.state.window_deadline_s), 1.0))
        if float(end_time_s) + tolerance < self.state.window_deadline_s:
            raise RuntimeError("cannot expire before the correlation deadline")
        self.state.avalanche_active = False

    def correlation_window(
            self, *, sigma_local_Pa: float, r_TJ_m: float,
            start_time_s: float) -> dict[str, float | bool]:
        """Constant-state convenience wrapper for tests and reduced replays."""
        if not math.isfinite(self.state.descendant_threshold):
            self.restart_correlation_window(start_time_s)
        gamma = self.rate(sigma_local_Pa, r_TJ_m)
        residual = self.state.descendant_threshold - self.state.descendant_hazard
        dt_cross = residual / gamma if gamma > 0.0 else math.inf
        if dt_cross <= self.correlation_time_s:
            result = self.accumulate_window_segment(
                rate_start_per_s=gamma, rate_end_per_s=gamma,
                start_time_s=start_time_s,
                end_time_s=start_time_s + max(
                    dt_cross, float.fromhex("0x1.0p-52")))
            if not result["crossed"]:
                raise RuntimeError("correlation crossing accounting failed")
            return dict(
                continued=True, elapsed_s=dt_cross, rate_per_s=gamma,
                end_time_s=start_time_s + dt_cross)
        self.accumulate_window_segment(
            rate_start_per_s=gamma, rate_end_per_s=gamma,
            start_time_s=start_time_s,
            end_time_s=start_time_s + self.correlation_time_s)
        self.expire_window(start_time_s + self.correlation_time_s)
        return dict(
            continued=False, elapsed_s=self.correlation_time_s,
            rate_per_s=gamma,
            end_time_s=start_time_s + self.correlation_time_s)

    def manifest(self) -> dict[str, Any]:
        return dict(
            state=asdict(self.state),
            descendant_barrier=dict(
                root_G0_eV=self.barrier.root.G0_eV,
                root_Gfloor_eV=self.barrier.root.G_floor_eV,
                floor_fraction=self.barrier.floor_fraction,
                a=self.barrier.root.a,
                sigma_hat_Pa=self.barrier.root.sigma_hat_pa,
                n=self.barrier.root.n,
                delta_G_step_eV=self.barrier.delta_G_step_eV,
                envelope=("max(root_G_floor, root_G_star(sigma) - "
                          "delta_G_step); fixed and non-additive")),
            temperature_K=self.temperature_K,
            attempt_frequency_per_s=self.attempt_frequency_per_s,
            b_m=self.b_m,
            correlation_time_s=self.correlation_time_s,
            crossings=list(self.crossings))
