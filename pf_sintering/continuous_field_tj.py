"""Continuous field-intersection tracker for the axisymmetric physical TJ.

The tracked point is the simultaneous outer-surface intersection
``f=0.5`` and ownership interface ``particle-substrate=0``.  Candidate
intersections are connected in time by nearest-neighbour continuation.  A
large discontinuous relocation is rejected rather than silently selecting a
different surface/neck branch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class TJTrackingError(RuntimeError):
    """Raised when the connected field-defined TJ cannot be continued."""


def _outer_half_level(f: np.ndarray, sampled: np.ndarray,
                      r_c: np.ndarray, level: float = 0.5):
    """Return outermost ``f=level`` radius and sampled value on each z row."""
    radius = np.full(f.shape[0], np.nan, dtype=float)
    value = np.full(f.shape[0], np.nan, dtype=float)
    for j, row in enumerate(np.asarray(f, dtype=float)):
        products = (row[:-1] - level) * (row[1:] - level)
        crossing = np.flatnonzero(products <= 0.0)
        if not len(crossing):
            continue
        # Ignore a flat segment that does not actually bracket the level.
        usable = [int(i) for i in crossing
                  if row[i + 1] != row[i]
                  and min(row[i], row[i + 1]) <= level
                  <= max(row[i], row[i + 1])]
        if not usable:
            continue
        i = usable[-1]
        fraction = (level - row[i]) / (row[i + 1] - row[i])
        radius[j] = r_c[i] + fraction * (r_c[i + 1] - r_c[i])
        value[j] = sampled[j, i] + fraction * (
            sampled[j, i + 1] - sampled[j, i])
    return radius, value


def field_intersection_candidates(f, particle, substrate, z, r_c):
    """Find all outer-surface crossings of the ownership interface."""
    z = np.asarray(z, dtype=float)
    r_c = np.asarray(r_c, dtype=float)
    radius, ownership = _outer_half_level(
        np.asarray(f), np.asarray(particle) - np.asarray(substrate), r_c)
    candidates = []
    for j in range(len(z) - 1):
        if not (np.isfinite(radius[j:j + 2]).all()
                and np.isfinite(ownership[j:j + 2]).all()):
            continue
        g0, g1 = float(ownership[j]), float(ownership[j + 1])
        if g0 == 0.0:
            alpha = 0.0
        elif g1 == 0.0 or g0 * g1 < 0.0:
            alpha = -g0 / (g1 - g0)
        else:
            continue
        z_tj = z[j] + alpha * (z[j + 1] - z[j])
        r_tj = radius[j] + alpha * (radius[j + 1] - radius[j])
        if np.isfinite(z_tj) and np.isfinite(r_tj) and r_tj > 0.0:
            candidates.append((float(z_tj), float(r_tj)))
    # Remove the duplicate produced when one row lands exactly on g=0.
    unique = []
    for candidate in candidates:
        if not unique or math.hypot(
                candidate[0] - unique[-1][0],
                candidate[1] - unique[-1][1]) > 1e-15:
            unique.append(candidate)
    return unique


@dataclass
class ContinuousFieldTJTracker:
    """Track the connected physical field intersection across evaluations."""

    z: np.ndarray
    r_c: np.ndarray
    initial_z_m: float
    max_jump_m: float
    previous_z_m: float | None = None
    previous_r_m: float | None = None
    evaluations: int = 0

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        self.r_c = np.asarray(self.r_c, dtype=float)
        if not math.isfinite(self.initial_z_m):
            raise ValueError("initial_z_m must be finite")
        if not math.isfinite(self.max_jump_m) or self.max_jump_m <= 0.0:
            raise ValueError("max_jump_m must be positive and finite")

    def locate(self, f, particle, substrate):
        candidates = field_intersection_candidates(
            f, particle, substrate, self.z, self.r_c)
        if not candidates:
            raise TJTrackingError(
                "no simultaneous f=0.5 and particle-substrate=0 intersection")
        if self.previous_z_m is None:
            reference = (float(self.initial_z_m), candidates[0][1])
            chosen = min(candidates, key=lambda p: abs(p[0] - reference[0]))
            jump = 0.0
        else:
            reference = (float(self.previous_z_m), float(self.previous_r_m))
            chosen = min(candidates, key=lambda p: math.hypot(
                p[0] - reference[0], p[1] - reference[1]))
            jump = math.hypot(
                chosen[0] - reference[0], chosen[1] - reference[1])
            if jump > self.max_jump_m:
                raise TJTrackingError(
                    "connected physical TJ relocation exceeds limit: "
                    f"jump={jump:.6e} m limit={self.max_jump_m:.6e} m; "
                    "refusing to switch branches")
        self.previous_z_m, self.previous_r_m = chosen
        self.evaluations += 1
        return dict(
            z_TJ_m=chosen[0], r_TJ_m=chosen[1],
            candidate_count=len(candidates), jump_m=float(jump),
            evaluations=self.evaluations,
            definition="intersection(f=0.5, particle-substrate=0)")

