"""D2-style avalanche clock layered outside the qualified PR 1b solver.

This module contains no phase-field evolution.  It owns only the facilitated
descendant barrier, integrated-hazard bookkeeping, the serialized child queue,
and correlation-window extinction.  The elementary transport remains the
existing qualified one-b event.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable

from .exp_barrier_nucleation import EV, KB, CompleteExpFloorParams, tj_site_count


@dataclass(frozen=True)
class DescendantBarrier:
    """Absolute, non-additive step envelope for descendant nucleation."""

    root: CompleteExpFloorParams
    G0_step_eV: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.G0_step_eV) or self.G0_step_eV <= 0.0:
            raise ValueError("G0_step_eV must be positive and finite")

    @property
    def floor_fraction(self) -> float:
        return self.root.G_floor_eV / self.root.G0_eV

    @property
    def G0_effective_eV(self) -> float:
        return min(self.root.G0_eV, self.G0_step_eV)

    def barrier_eV(self, sigma_local_Pa: float) -> float:
        sigma = max(0.0, float(sigma_local_Pa))
        shape = math.exp(
            -self.root.a * (sigma / self.root.sigma_hat_pa) ** self.root.n)
        f = self.floor_fraction
        return self.G0_effective_eV * (f + (1.0 - f) * shape)


def descendant_rate_per_s(
        sigma_local_Pa: float, r_TJ_m: float, *,
        barrier: DescendantBarrier, temperature_K: float,
        attempt_frequency_per_s: float, b_m: float) -> float:
    """Total axisymmetric-ring descendant rate.

    The site population is exactly ``2*pi*r_TJ/b``.  There is no angular
    discretization and no DDD/disconnection elastic-stress contribution.
    """
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if attempt_frequency_per_s <= 0.0:
        raise ValueError("attempt_frequency_per_s must be positive")
    G_eV = barrier.barrier_eV(sigma_local_Pa)
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
    pending_children: int = 0
    G0_step_eV: float = math.nan
    avalanche_start_time_s: float = math.nan
    last_child_completion_time_s: float = math.nan
    descendant_total_hazard: float = 0.0
    thresholds_drawn: list[float] = field(default_factory=list)


class AvalancheController:
    """Integrated descendant clock and serialized one-b child queue."""

    def __init__(
            self, *, barrier: DescendantBarrier, temperature_K: float,
            attempt_frequency_per_s: float, b_m: float,
            correlation_time_s: float, rng) -> None:
        if correlation_time_s <= 0.0:
            raise ValueError("correlation_time_s must be positive")
        self.barrier = barrier
        self.temperature_K = float(temperature_K)
        self.attempt_frequency_per_s = float(attempt_frequency_per_s)
        self.b_m = float(b_m)
        self.correlation_time_s = float(correlation_time_s)
        self.rng = rng
        self.state = AvalancheState(G0_step_eV=barrier.G0_step_eV)
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
            G0_step_eV=self.barrier.G0_step_eV,
            avalanche_start_time_s=float(start_time_s),
            last_child_completion_time_s=math.nan,
        )
        self.state.descendant_threshold = self._draw_threshold()
        self.crossings = []

    def rate(self, sigma_local_Pa: float, r_TJ_m: float) -> float:
        return descendant_rate_per_s(
            sigma_local_Pa, r_TJ_m, barrier=self.barrier,
            temperature_K=self.temperature_K,
            attempt_frequency_per_s=self.attempt_frequency_per_s,
            b_m=self.b_m)

    def begin_transit(self) -> int:
        if not self.state.avalanche_active:
            raise RuntimeError("cannot begin a transit outside an avalanche")
        # child_number=0 denotes the root transit; descendants are 1,2,...
        self.state.child_number = self.state.S_completed
        return self.state.child_number

    def _consume_hazard(
            self, increment: float, *, t_start_s: float, t_end_s: float,
            source: str) -> int:
        if increment < 0.0 or not math.isfinite(increment):
            raise ValueError("hazard increment must be non-negative and finite")
        if increment == 0.0:
            return 0
        crossed = 0
        remaining = float(increment)
        consumed = 0.0
        while (self.state.descendant_hazard + remaining
               >= self.state.descendant_threshold):
            needed = self.state.descendant_threshold - self.state.descendant_hazard
            needed = max(needed, 0.0)
            consumed += needed
            remaining -= needed
            fraction = min(max(consumed / increment, 0.0), 1.0)
            crossing_time = float(t_start_s) + fraction * (
                float(t_end_s) - float(t_start_s))
            self.state.descendant_total_hazard += needed
            self.state.descendant_hazard = 0.0
            self.state.pending_children += 1
            crossed += 1
            self.crossings.append(dict(
                avalanche_id=self.state.avalanche_id,
                committed_child_index=(
                    self.state.S_completed + self.state.pending_children),
                crossing_time_s=crossing_time,
                source=source,
                crossed_threshold=self.state.descendant_threshold,
                pending_children_after=self.state.pending_children))
            self.state.descendant_threshold = self._draw_threshold()
        self.state.descendant_hazard += remaining
        self.state.descendant_total_hazard += remaining
        return crossed

    def integrate_transit_samples(
            self, samples: Iterable[dict[str, float]]) -> list[dict[str, float]]:
        """Trapezoid-integrate rate over accepted event checkpoint samples."""
        values = [dict(sample) for sample in samples]
        if len(values) < 2:
            raise ValueError("at least two transit samples are required")
        times = [float(item["t_s"]) for item in values]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("transit sample times must increase strictly")
        trace: list[dict[str, float]] = []
        rate_previous = self.rate(
            values[0]["sigma_local_Pa"], values[0]["r_TJ_m"])
        trace.append(self._trace_row(values[0], rate_previous, 0))
        for left, right in zip(values, values[1:]):
            rate_right = self.rate(
                right["sigma_local_Pa"], right["r_TJ_m"])
            dt = float(right["t_s"]) - float(left["t_s"])
            increment = 0.5 * (rate_previous + rate_right) * dt
            crossed = self._consume_hazard(
                increment, t_start_s=left["t_s"], t_end_s=right["t_s"],
                source="active_transit")
            trace.append(self._trace_row(right, rate_right, crossed))
            rate_previous = rate_right
        return trace

    def _trace_row(self, sample: dict[str, float], rate: float,
                   crossings: int) -> dict[str, float]:
        threshold = self.state.descendant_threshold
        return dict(
            **sample,
            descendant_rate_per_s=float(rate),
            G_desc_star_eV=self.barrier.barrier_eV(
                sample["sigma_local_Pa"]),
            G0_effective_eV=self.barrier.G0_effective_eV,
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
        self.state.S_completed += 1
        self.state.q_avalanche_over_b = float(self.state.S_completed)
        self.state.last_child_completion_time_s = float(completion_time_s)

    def launch_pending_child(self) -> bool:
        if self.state.pending_children <= 0:
            return False
        self.state.pending_children -= 1
        self.state.child_number = self.state.S_completed
        return True

    def correlation_window(
            self, *, sigma_local_Pa: float, r_TJ_m: float,
            start_time_s: float) -> dict[str, float | bool]:
        """Advance a constant post-event clock until crossing or extinction."""
        if self.state.pending_children:
            raise RuntimeError("correlation window requires an empty child queue")
        gamma = self.rate(sigma_local_Pa, r_TJ_m)
        residual = self.state.descendant_threshold - self.state.descendant_hazard
        dt_cross = residual / gamma if gamma > 0.0 else math.inf
        if dt_cross <= self.correlation_time_s:
            crossed = self._consume_hazard(
                residual, t_start_s=start_time_s,
                t_end_s=start_time_s + dt_cross,
                source="correlation_window")
            if crossed != 1:
                raise RuntimeError("correlation crossing accounting failed")
            return dict(
                continued=True, elapsed_s=dt_cross, rate_per_s=gamma,
                end_time_s=start_time_s + dt_cross)
        increment = gamma * self.correlation_time_s
        self._consume_hazard(
            increment, t_start_s=start_time_s,
            t_end_s=start_time_s + self.correlation_time_s,
            source="correlation_window")
        self.state.avalanche_active = False
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
                G0_step_eV=self.barrier.G0_step_eV,
                G0_effective_eV=self.barrier.G0_effective_eV,
                envelope="absolute min(root_G0, G0_step); non-additive"),
            temperature_K=self.temperature_K,
            attempt_frequency_per_s=self.attempt_frequency_per_s,
            b_m=self.b_m,
            correlation_time_s=self.correlation_time_s,
            crossings=list(self.crossings))
