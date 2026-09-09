"""Mass-conservative bounded projection for the conserved solid fraction."""
from __future__ import annotations

import math

import numpy as np


class ConservativeBoundedPhaseProjector:
    """Enforce 0<=f<=1 and e1+e2=f without changing solid volume.

    The raw conservative PF step can overshoot the phase-fraction interval,
    while the grain update already clips its target to that interval.  This
    projector clips the raw field, redistributes the clipped signed volume on
    the diffuse interface, then preserves the local grain ownership ratio.
    """

    def __init__(self, *, tolerance: float = 1.0e-12,
                 activation_tolerance: float = 1.0e-12):
        self.tolerance = float(tolerance)
        self.activation_tolerance = float(activation_tolerance)
        self.calls = 0
        self.active_calls = 0
        self.cumulative_abs_redistributed_m3 = 0.0
        self.maximum_abs_point_correction = 0.0
        self.maximum_raw_overshoot = 0.0
        self.maximum_relative_volume_error = 0.0
        self.maximum_final_partition_closure = 0.0
        self.partition_repair_calls = 0

    @staticmethod
    def _weights(setup) -> np.ndarray:
        return (2.0 * math.pi * np.asarray(setup["r_c"])[None, :]
                * float(setup["dr"]) * float(setup["dz"]))

    @staticmethod
    def _integral(field: np.ndarray, weights: np.ndarray) -> float:
        return float(np.sum(weights * field))

    def _redistribute(self, bounded: np.ndarray, delta_m3: float,
                      weights: np.ndarray) -> np.ndarray:
        result = bounded.copy()
        remaining = float(delta_m3)
        total_scale = max(abs(self._integral(result, weights)), 1e-300)
        tolerance_m3 = max(self.tolerance * total_scale, 1e-36)
        for _ in range(12):
            if abs(remaining) <= tolerance_m3:
                break
            if remaining > 0.0:
                capacity = np.maximum(1.0 - result, 0.0)
            else:
                capacity = np.maximum(result, 0.0)
            interface = 4.0 * result * (1.0 - result)
            basis = np.maximum(interface, 0.0) * capacity
            norm = self._integral(basis, weights)
            if norm <= 0.0:
                basis = capacity
                norm = self._integral(basis, weights)
            if norm <= 0.0:
                raise RuntimeError("bounded phase projection has no capacity")
            proposed = basis * (abs(remaining) / norm)
            applied = np.minimum(proposed, capacity)
            amount = self._integral(applied, weights)
            if amount <= 0.0:
                raise RuntimeError("bounded phase projection stalled")
            if remaining > 0.0:
                result += applied
                remaining -= amount
            else:
                result -= applied
                remaining += amount
        if abs(remaining) > 10.0 * tolerance_m3:
            raise RuntimeError(
                f"bounded phase projection residual {remaining:.6e} m3")
        return np.clip(result, 0.0, 1.0)

    def __call__(self, state, setup):
        f_raw, particle, substrate = (np.asarray(x) for x in state)
        raw_min = float(np.min(f_raw))
        raw_max = float(np.max(f_raw))
        raw_overshoot = max(-raw_min, raw_max-1.0, 0.0)
        self.calls += 1
        if (raw_min >= -self.activation_tolerance
                and raw_max <= 1.0 + self.activation_tolerance):
            fast_closure = float(np.max(np.abs(
                f_raw-particle-substrate)))
            partition_repaired = 0
            # A bounded f field does not by itself guarantee e1+e2=f.  Tiny
            # closure drift can accumulate in the fast path until the closed
            # surface-diffusion callback rejects an otherwise valid state.
            # Repair only the representation: keep f bit-for-bit unchanged and
            # preserve the pointwise ownership ratio.
            if fast_closure > 5.0e-15:
                raw_sum = particle + substrate
                ownership = np.divide(
                    particle, raw_sum, out=np.full_like(particle, 0.5),
                    where=np.abs(raw_sum) > 1.0e-30)
                ownership = np.clip(ownership, 0.0, 1.0)
                particle = f_raw * ownership
                substrate = f_raw * (1.0 - ownership)
                fast_closure = float(np.max(np.abs(
                    f_raw-particle-substrate)))
                if fast_closure > 5.0e-15:
                    raise RuntimeError(
                        "bounded phase fast-path partition repair failed: "
                        f"closure={fast_closure:.6e}")
                partition_repaired = 1
                self.partition_repair_calls += 1
            self.maximum_final_partition_closure = max(
                self.maximum_final_partition_closure, fast_closure)
            diagnostics = dict(
                raw_f_min=raw_min, raw_f_max=raw_max,
                raw_overshoot=raw_overshoot,
                redistributed_signed_m3=0.0,
                redistributed_abs_m3=0.0,
                maximum_abs_point_correction=0.0,
                relative_volume_error=0.0,
                final_partition_closure=fast_closure,
                partition_repaired=partition_repaired)
            return (f_raw, particle, substrate), diagnostics
        weights = self._weights(setup)
        bounded = np.clip(f_raw, 0.0, 1.0)
        target_volume = self._integral(f_raw, weights)
        clipped_volume = self._integral(bounded, weights)
        delta = target_volume - clipped_volume
        corrected = self._redistribute(bounded, delta, weights)

        raw_sum = particle + substrate
        ownership = np.divide(
            particle, raw_sum, out=np.full_like(particle, 0.5),
            where=raw_sum > 1e-30)
        ownership = np.clip(ownership, 0.0, 1.0)
        particle_new = corrected * ownership
        substrate_new = corrected * (1.0 - ownership)

        final_volume = self._integral(corrected, weights)
        relative_volume_error = abs(final_volume-target_volume) / max(
            abs(target_volume), 1e-300)
        if relative_volume_error > 5.0e-12:
            raise RuntimeError(
                "bounded phase projection changed conserved volume: "
                f"relative error={relative_volume_error:.6e}")
        bound_error = max(float(-np.min(corrected)),
                          float(np.max(corrected)-1.0), 0.0)
        closure = float(np.max(np.abs(
            corrected-particle_new-substrate_new)))
        if bound_error > 1e-14 or closure > 5e-15:
            raise RuntimeError(
                f"bounded phase projection constraint failure: "
                f"bound={bound_error}, closure={closure}")

        max_correction = float(np.max(np.abs(corrected-f_raw)))
        if raw_overshoot > 0.0:
            self.active_calls += 1
        self.cumulative_abs_redistributed_m3 += abs(delta)
        self.maximum_abs_point_correction = max(
            self.maximum_abs_point_correction, max_correction)
        self.maximum_raw_overshoot = max(
            self.maximum_raw_overshoot, raw_overshoot)
        self.maximum_relative_volume_error = max(
            self.maximum_relative_volume_error, relative_volume_error)
        self.maximum_final_partition_closure = max(
            self.maximum_final_partition_closure, closure)
        diagnostics = dict(
            raw_f_min=float(np.min(f_raw)), raw_f_max=float(np.max(f_raw)),
            raw_overshoot=raw_overshoot,
            redistributed_signed_m3=delta,
            redistributed_abs_m3=abs(delta),
            maximum_abs_point_correction=max_correction,
            relative_volume_error=relative_volume_error,
            final_partition_closure=closure)
        return (corrected, particle_new, substrate_new), diagnostics

    def manifest(self) -> dict:
        return dict(
            calls=self.calls, active_calls=self.active_calls,
            cumulative_abs_redistributed_m3=(
                self.cumulative_abs_redistributed_m3),
            maximum_abs_point_correction=self.maximum_abs_point_correction,
            maximum_raw_overshoot=self.maximum_raw_overshoot,
            maximum_relative_volume_error=self.maximum_relative_volume_error,
            maximum_final_partition_closure=(
                self.maximum_final_partition_closure),
            partition_repair_calls=self.partition_repair_calls,
            tolerance=self.tolerance,
            activation_tolerance=self.activation_tolerance,
            definition=(
                "clip f to [0,1], redistribute signed clipped volume on the "
                "diffuse interface, preserve local grain ownership"))
