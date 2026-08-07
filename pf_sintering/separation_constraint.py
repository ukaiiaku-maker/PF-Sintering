"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not developed further, not wired into
production physics. See constrained_relaxation.py's module docstring and
MILESTONE_4_CONSTRAINED_RELAXATION_REPORT.md.

Explicit particle/substrate-separation (L) constraint corrector.

Used only by the constrained-relaxation benchmark
(`constrained_relaxation.py`). This is a *pure geometric correction*, not a
physical event: it carries no Sink/hazard/quota bookkeeping and is never
called from the production runner. It exists because nothing in the
production integrator exactly conserves `L = center(e2, p) - wall_x0` --
`rbm()` (real densification) is the only thing that ever moves the particle
along x, and it is switched off for this benchmark by construction (the
Sink is simply never activated). Ordinary CH/structural relaxation can still
drift the *mass-weighted centroid* of e2 by a small amount through
asymmetric shape evolution alone, without any true rigid-body translation
(this was already observed and reported in MILESTONE_1_2_REPORT.md, Section
D, at the ~0.3-0.4% level over an 827-step sink-off run). For a relaxation
run to genuine convergence at a *specified* L, that residual drift must be
corrected explicitly and its magnitude reported, not merely bounded by a
short time horizon.

The correction reuses exactly the same conservative upwind advection
stencil `model.rbm()` uses (confined to the e2-dominant region via the same
`e2 / (e1 + e2 + e3)` weighting), applied as a sequence of small corrective
substeps with the actual resulting centroid position re-measured after each
substep (the map from substep count to centroid displacement is not
perfectly linear because of the same excess-mass redistribution `rbm()`
performs after each substep), stopping once the residual is within the
requested tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import center
from .structural_projection import eta_masses, project_eta_mass_preserving


def _advect_substep(f, e1, e2, e3, p, vx, ds):
    """One substep of model.rbm()'s conservative upwind transport, generalized
    to an arbitrary velocity field vx and step size ds. Pure geometry; no
    Sink/event state touched."""
    vr = 0.5 * (vx + np.roll(vx, -1, 1))
    fr = np.maximum(vr, 0) * f + np.minimum(vr, 0) * np.roll(f, -1, 1)
    f = f - ds * (fr - np.roll(fr, 1, 1)) / p.dx
    out = []
    for e in (e1, e2, e3):
        fw = (np.roll(e, -1, 1) - e) / p.dx
        bw = (e - np.roll(e, 1, 1)) / p.dx
        e = e - ds * vx * np.where(vx >= 0, bw, fw)
        e = np.clip(e, 0, 1)
        out.append(e)
    e1, e2, e3 = out
    ex = np.maximum(0, f - 1)
    f = np.clip(f, 0, 1)
    exsum = float(ex.sum())
    if exsum > 0:
        surf = 16 * f * f * (1 - f) ** 2
        ss = float(surf.sum())
        if ss > 1e-30:
            f = f + surf / ss * exsum
    return f, e1, e2, e3


@dataclass
class SeparationCorrection:
    applied: bool
    n_substeps: int
    dL_before: float
    dL_after: float
    total_signed_displacement: float


def enforce_separation(
    f, e1, e2, e3, p, wall_x0_val, L_target, tol,
    max_substeps=60, max_step_frac_of_dx=0.02,
) -> tuple:
    """Correct center(e2, p) back to `wall_x0_val + L_target` if it has
    drifted beyond `tol`. Returns (f, e1, e2, e3, SeparationCorrection)."""
    L_before = center(e2, p) - wall_x0_val
    dL0 = L_before - L_target
    if abs(dL0) <= tol:
        return f, e1, e2, e3, SeparationCorrection(False, 0, dL0, dL0, 0.0)

    total_disp = 0.0
    n = 0
    for _ in range(max_substeps):
        L_now = center(e2, p) - wall_x0_val
        dL = L_now - L_target
        if abs(dL) <= tol:
            break
        den = e1 + e2 + (e3 if p.use_eta3 else 0) + 1e-30
        weight = e2 / den
        step = -math.copysign(min(abs(dL), max_step_frac_of_dx * p.dx), dL)
        vx = weight * math.copysign(1.0, step)
        f, e1, e2, e3 = _advect_substep(f, e1, e2, e3, p, vx, abs(step))
        total_disp += step
        n += 1

    L_after = center(e2, p) - wall_x0_val
    return f, e1, e2, e3, SeparationCorrection(True, n, dL0, L_after - L_target, total_disp)


@dataclass
class ConstraintReport:
    outer_iterations: int
    dL_final: float
    dV2_final: float


def enforce_V2_and_L(
    f, e1, e2, e3, p, wall_x0_val, L_target, v_targets, tol_L, tol_v2_rel=1e-10,
    max_outer=6,
) -> tuple:
    """Jointly enforce fixed grain masses and fixed separation `L_target`.

    `v_targets` (the *original*, session-start masses, in raw eta_masses()
    units) is used only to report the long-run dV2 drift. The reprojection
    that follows each `enforce_separation` call instead targets the masses
    measured *immediately before that call* (self-referential, matching how
    the main relaxation loop's own CH/AC-step projections work): if
    `enforce_separation` is a no-op (separation already within tolerance --
    the common case on most steps once converged), the following
    reprojection then has target == current and is *also* a true no-op.

    Re-pinning to the fixed original `v_targets` unconditionally on every
    call was tried first and rejected: repeated over thousands of steps it
    measurably raised G_interface (a slow but real energy increase from
    otherwise-unnecessary redistribution/smoothing on steps where nothing
    had actually drifted), which was caught by comparing G_interface(t) at
    long time against a mid-run value during the constrained-relaxation
    convergence check.
    """
    v2_target_phys = v_targets[1] * p.dx * p.dx
    for it in range(1, max_outer + 1):
        pre_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f, e1, e2, e3, corr = enforce_separation(f, e1, e2, e3, p, wall_x0_val, L_target, tol_L)
        if corr.applied:
            e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=pre_targets)
        dL = center(e2, p) - wall_x0_val - L_target
        v2_now_phys = float(e2.sum() * p.dx * p.dx)
        dv2 = (v2_now_phys - v2_target_phys) / v2_target_phys
        if abs(dL) <= tol_L and abs(dv2) <= tol_v2_rel:
            break
    return f, e1, e2, e3, ConstraintReport(it, dL, dv2)
