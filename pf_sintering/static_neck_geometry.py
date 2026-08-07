"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Static substrate contact-geometry family at fixed V2, fixed separation L,
with an independently controllable neck/contact width x_neck.

This module *artificially constrains x_neck* (via a flat wall-side collar
merged into a circular far cap, see below) purely so the Milestone-3
static force map could sweep x_neck at fixed (V2, L). MILESTONE_3_FORCE_BALANCE_REPORT.md
Sections H-M found: (1) the resulting x0 (zero-force neck width) does NOT
behave as a dynamical attractor under real capillary relaxation -- most
likely because the collar/cap junction is a genuine curvature
discontinuity with no physical counterpart -- and (2) the local TJ force
probe is blind to V2 in this construction whenever the collar is
comfortably long, so x0(V2) could not be answered with it either. Do not
treat any x0 computed via this module as a physical zero-sintering-stress
state, and do not extend or "fix" this construction further without first
reviewing that report; the current constrained-relaxation work
(`constrained_relaxation.py`) supersedes it as the primary benchmark and
lets x_neck and psi emerge from real relaxation instead of being dialed in
here. This module is retained only as a convenient, already-precise (V2, L)
initial-condition generator for that relaxation, and as the diagnostic tool
that produced the Milestone-3 findings above.

`model.initialize_fields`'s single-parameter (`initial_overlap`) substrate
construction ties neck width, particle volume, and particle/substrate
separation together: increasing overlap simultaneously widens the neck,
shrinks V2 (more of the ellipse is masked behind the wall), and shrinks L.
Section 8 of the force-balance handoff requires a family with V2 and L held
fixed while x_neck varies, which that single-parameter family cannot produce.

Construction: the particle's half-thickness profile `y(x)` (its boundary is
`|Y| = y(X)`) is built directly rather than as a perturbed ellipse:

    y_far(x) = sqrt(max(0, Rfar**2 - (x - cx)**2))
    y(x) = max(x_neck, y_far(x))   for x <= cx  (wall-side: neck collar)
    y(x) = y_far(x)                for x  > cx  (far side: closes off to 0)

i.e. a flat collar of half-width `x_neck` (assigned directly, not searched
for) on the wall side wherever the far circular cap would be narrower than
that, merging continuously into the cap `y_far(x)` at `x = cx` (both
branches equal `Rfar` there) and tapering back to zero on the far side as a
plain circular cap. (Two earlier versions were tried and rejected: blending
`x_neck` into `y_far(x)` over a fixed transition length produced a spurious
dip *below* x_neck whenever `L` was large enough that the cap does not reach
the wall at all, since blending toward `y_far(wall) = 0` pulls the profile
down before the cap "catches up"; and applying `max(x_neck, y_far(x))`
*everywhere* -- rather than only on the wall side -- never let the shape
taper to zero on the far side at all, since `y_far -> 0` far from `cx` on
both sides.) `Rfar` (far-cap radius) and `cx` (far-cap center, `L ~= cx -
wall`) are found by a small nested bisection so the resulting V2 and L match
their targets; both sub-problems are individually monotonic and
well-conditioned (more `Rfar` monotonically raises V2; more `cx`
monotonically raises L).

Achieved `(V2, L, x_neck)` are always measured directly from the resulting
fields and reported alongside the targets rather than assumed exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .model import center, contact_width


@dataclass
class NeckState:
    f: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray
    Rfar: float
    cx: float
    x_neck_target: float
    x_neck_achieved: float
    V2_target: float
    V2_achieved: float
    L_target: float
    L_achieved: float


def _wall_x0(p) -> float:
    return (p.substrate_wall_frac - 0.5) * p.Nx * p.dx


def _fields_for(p, cx, Rfar, x_neck, wall):
    x = (np.arange(1, p.Nx + 1) - p.Nx / 2) * p.dx
    y = (np.arange(1, p.Ny + 1) - p.Ny / 2) * p.dx
    X, Y = np.meshgrid(x, y)
    W = p.interface_width

    e1_raw = 0.5 * (1 - np.tanh((X - wall) / W))

    y_far = np.sqrt(np.maximum(0.0, Rfar**2 - (X - cx) ** 2))
    y_prof = np.where(X <= cx, np.maximum(x_neck, y_far), y_far)
    e2_raw = 0.5 * (1 - np.tanh((np.abs(Y) - y_prof) / W))

    t1 = 0.5 * (1 - np.tanh((X - wall) / W))
    t2 = 0.5 * (1 + np.tanh((X - wall) / W))
    f = np.maximum(e1_raw, e2_raw)
    e1 = e1_raw * t1
    e2 = e2_raw * t2
    e3 = np.zeros_like(X)
    return f, e1, e2, e3


def build_neck_state(
    p, x_neck_target, v2_target, L_target,
    Rfar_bracket=None, cx_grid=None, n_cx_grid=80, xtol=1e-14,
) -> NeckState:
    """Solve for (cx, Rfar) hitting (L_target, v2_target) at the given
    x_neck_target.

    For a fixed `cx`, V2 is monotonic increasing in `Rfar` over
    `Rfar_bracket`, but not every `cx` admits a feasible `Rfar` at all: if
    `cx` is far enough from the wall that even the minimal cap
    (`Rfar = x_neck_target`, i.e. an almost-flat wall-side collar of length
    `cx - wall`) already exceeds `v2_target`, or if the largest allowed cap
    still undershoots it, no root exists for that `cx`. Rather than assume a
    fixed `cx` bracket is feasible at both ends (it was not, empirically, for
    every x_neck/L/V2 combination exercised by the force-map benchmark), scan
    a grid of `cx` candidates, solve the inner V2/Rfar root only where
    feasible, and bisect the outer L residual over the first feasible sign
    change found.
    """
    wall = _wall_x0(p)

    if Rfar_bracket is None:
        Rfar_bracket = (max(x_neck_target, p.dx), 6.0 * max(p.R2, x_neck_target))

    def v2_of(cx, Rfar):
        _, _, e2, _ = _fields_for(p, cx, Rfar, x_neck_target, wall)
        return float(e2.sum() * p.dx * p.dx)

    def L_of(cx, Rfar):
        _, _, e2, _ = _fields_for(p, cx, Rfar, x_neck_target, wall)
        return float(center(e2, p) - wall)

    def solve_Rfar(cx):
        lo, hi = Rfar_bracket
        flo, fhi = v2_of(cx, lo) - v2_target, v2_of(cx, hi) - v2_target
        if not np.isfinite(flo) or not np.isfinite(fhi) or flo > 0 or fhi < 0:
            return None  # infeasible at this cx: no Rfar in-bracket hits v2_target
        return brentq(lambda R: v2_of(cx, R) - v2_target, lo, hi, xtol=xtol)

    def L_residual(cx):
        Rfar = solve_Rfar(cx)
        if Rfar is None:
            return None
        return L_of(cx, Rfar) - L_target, Rfar

    if cx_grid is None:
        cx_grid = wall + np.linspace(0.05, 4.0, n_cx_grid) * L_target

    samples = []
    for cx in cx_grid:
        r = L_residual(cx)
        if r is not None:
            samples.append((cx, r[0], r[1]))
    if len(samples) == 0:
        raise RuntimeError(
            "no feasible (cx, Rfar) found on the search grid for "
            f"x_neck_target={x_neck_target}, v2_target={v2_target}, L_target={L_target}"
        )

    bracket = None
    for i in range(len(samples) - 1):
        c0, r0, _ = samples[i]
        c1, r1, _ = samples[i + 1]
        if r0 == 0:
            bracket = (c0, c0)
            break
        if r0 * r1 < 0:
            bracket = (c0, c1)
            break

    if bracket is None:
        # No sign change on the feasible grid: fall back to the closest
        # feasible sample rather than failing outright.
        cx_star, _, Rfar_star = min(samples, key=lambda s: abs(s[1]))
    elif bracket[0] == bracket[1]:
        cx_star = bracket[0]
        Rfar_star = solve_Rfar(cx_star)
    else:
        lo, hi = bracket

        def resid(cx):
            r = L_residual(cx)
            if r is None:
                raise RuntimeError(f"lost feasibility at cx={cx} during refinement")
            return r[0]

        cx_star = brentq(resid, lo, hi, xtol=xtol)
        Rfar_star = solve_Rfar(cx_star)

    f, e1, e2, e3 = _fields_for(p, cx_star, Rfar_star, x_neck_target, wall)
    x_neck_ach, _ = contact_width(e1, e2, p)
    v2_ach = float(e2.sum() * p.dx * p.dx)
    L_ach = float(center(e2, p) - wall)

    return NeckState(
        f=f, e1=e1, e2=e2, e3=e3, Rfar=Rfar_star, cx=cx_star,
        x_neck_target=x_neck_target, x_neck_achieved=x_neck_ach,
        V2_target=v2_target, V2_achieved=v2_ach,
        L_target=L_target, L_achieved=L_ach,
    )
