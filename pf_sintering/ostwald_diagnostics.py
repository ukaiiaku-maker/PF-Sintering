"""DIAGNOSTIC ONLY -- not wired into production physics.

Exact-formula mirror of the active production Ostwald kernel
(`parity_kernels.ostwald_substrate`, confirmed monkey-patched onto
`model.ostwald_substrate` at package-import time -- see
MILESTONE_6_OPERATOR_RESOLVED_COARSENING_AUDIT.md Section B for the verified
call chain), instrumented to expose the spatial removal/addition fields
*before* they are normalized/applied, plus two deliberately nonphysical
mechanism-isolation variants requested for the operator audit:

- O1 "removal-only": the same removal field as the production kernel, but
  the removed material is tracked in a scalar external-reservoir
  bookkeeping variable instead of being deposited on e1/f. Isolates whether
  particle shrinkage alone (with no local destination-side deposition)
  already produces the observed neck response.
- O2 "addition-only": the same addition field as the production kernel,
  applied without removing the corresponding particle material. Isolates
  how much of the response comes from the destination-side deposition
  pattern by itself.

`ostwald_substrate_diagnostic` is a byte-for-byte transcription of the
active kernel's formula (verified in tests/test_ostwald_diagnostics.py to
reproduce identical (f, e1, e2, e3) output to the production kernel for the
same input); nothing about the physics is changed. O1/O2 reuse its exact
`remove`/`add` fields rather than recomputing anything, so they are directly
comparable to the O0 (normal) case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d

from .model import overlap_col


@dataclass
class OstwaldDiagnostics:
    removal_candidate: np.ndarray  # surf*e2*incl, pre-convolution ("before normalized")
    addition_candidate: np.ndarray  # surf*e1*incl, pre-convolution
    src: np.ndarray  # convolved source density
    snk: np.ndarray  # convolved sink density
    remove: np.ndarray  # final applied removal field (subtracted from e2/f)
    add: np.ndarray  # final applied addition field (added to e1/f)
    tot_src: float
    tot_snk: float
    transfer_target: float
    actual_removed: float
    actual_added: float
    max_depositable: float
    capped: bool
    col: float


def ostwald_substrate_diagnostic(f, e1, e2, e3, p):
    """Exact transcription of parity_kernels.ostwald_substrate with the
    intermediate spatial fields captured. Returns (f_new, e1_new, e2_new,
    e3, diagnostics)."""
    V = float(e2.sum())
    fb = np.clip(f, 0.0, 1.0)
    surf = 16.0 * fb * fb * (1.0 - fb) ** 2
    ker = np.ones((3, 3), dtype=float) / 9.0
    cr = p.Ny // 2

    _, col = overlap_col(e1[cr, :] * e2[cr, :])
    if not math.isfinite(col):
        col = p.substrate_wall_frac * p.Nx

    if getattr(p, "reservoir_neck_unprotected", False):
        incl = np.ones_like(f)
    else:
        CC, RR = np.meshgrid(np.arange(1, p.Nx + 1), np.arange(1, p.Ny + 1))
        ex_c = max(5, round(2.0 * p.interface_width / p.dx))
        ex_r = max(5, round(3.0 * p.interface_width / p.dx))
        incl = 1.0 - np.exp(
            -0.5 * (((CC - col) / ex_c) ** 2 + ((RR - (cr + 1)) / ex_r) ** 2)
        )

    removal_candidate = surf * e2 * incl
    addition_candidate = surf * e1 * incl
    src = convolve2d(removal_candidate, ker, mode="same", boundary="fill")
    snk = convolve2d(addition_candidate, ker, mode="same", boundary="fill")
    tot_src = float(src.sum())
    tot_snk = float(snk.sum())

    zeros = np.zeros_like(f)
    if tot_src < 1e-15 or tot_snk < 1e-15:
        diag = OstwaldDiagnostics(
            removal_candidate, addition_candidate, src, snk, zeros, zeros,
            tot_src, tot_snk, 0.0, 0.0, 0.0, 0.0, False, col,
        )
        return f, e1, e2, e3, diag

    transfer = min(V * p.dt / p.tau_ripening, 0.002 * V)
    remove = np.minimum(src / tot_src * transfer, 0.9 * e2)
    remove = np.maximum(remove, 0.0)
    actual = float(remove.sum())

    add_try = snk / (tot_snk + 1e-30) * actual
    cap_trial = np.maximum(0.0, np.minimum(1.0 - f, 1.0 - e1))
    max_depositable = float(np.minimum(add_try, cap_trial).sum())

    capped = max_depositable < actual and actual > 1e-30
    if capped:
        sf = max_depositable / actual
        remove = remove * sf
        add_try = add_try * sf

    e2_new = e2 - remove
    f_new = f - remove

    cap = np.maximum(0.0, np.minimum(1.0 - f_new, 1.0 - e1))
    add1 = np.maximum(0.0, np.minimum(add_try, cap))
    e1_new = e1 + add1
    f_new = f_new + add1

    diag = OstwaldDiagnostics(
        removal_candidate=removal_candidate, addition_candidate=addition_candidate,
        src=src, snk=snk, remove=remove, add=add1,
        tot_src=tot_src, tot_snk=tot_snk, transfer_target=transfer,
        actual_removed=float(remove.sum()), actual_added=float(add1.sum()),
        max_depositable=max_depositable, capped=capped, col=col,
    )
    return f_new, e1_new, e2_new, e3, diag


def ostwald_removal_only(f, e1, e2, e3, p):
    """DIAGNOSTIC ONLY -- NOT PRODUCTION PHYSICS.

    Same removal field as ostwald_substrate_diagnostic (O0); the removed
    material is NOT deposited on e1/f. Returns
    (f_new, e1_unchanged, e2_new, e3, removed_to_external_reservoir).
    """
    _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    remove = diag.remove
    e2_new = e2 - remove
    f_new = f - remove
    return f_new, e1, e2_new, e3, diag.actual_removed


def ostwald_addition_only(f, e1, e2, e3, p):
    """DIAGNOSTIC ONLY -- NOT PRODUCTION PHYSICS.

    Same addition field as ostwald_substrate_diagnostic (O0), applied
    without removing the corresponding particle material (so this
    deliberately does not conserve total solid mass). Returns
    (f_new, e1_new, e2_unchanged, e3, artificial_mass_added).
    """
    _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    add1 = diag.add
    e1_new = e1 + add1
    f_new = f + add1
    return f_new, e1_new, e2, e3, diag.actual_added


def near_neck_fractions(field, tj_xy_top, tj_xy_bottom, p, widths_in_W=(1, 2, 3, 5)):
    """Fraction of `field`'s (non-negative) mass within N interface widths of
    either TJ, for N in widths_in_W. Uses the tj_force uncentered coordinate
    convention ((index+1)*dx), matching tj_force.locate_neck_tjs."""
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    d_top = np.hypot(X - tj_xy_top[0], Y - tj_xy_top[1])
    d_bot = np.hypot(X - tj_xy_bottom[0], Y - tj_xy_bottom[1])
    d = np.minimum(d_top, d_bot)
    total = float(np.sum(field))
    out = {}
    for n in widths_in_W:
        mask = d <= n * p.interface_width
        out[f"frac_within_{n}W"] = float(np.sum(field[mask])) / total if total > 0 else math.nan
    return out


def centroid(field, p):
    """Mass-weighted centroid of a non-negative spatial field, in the same
    uncentered coordinate convention as tj_force/locate_neck_tjs."""
    total = float(np.sum(field))
    if total <= 0:
        return (math.nan, math.nan)
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    cx = float(np.sum(field * X)) / total
    cy = float(np.sum(field * Y)) / total
    return (cx, cy)
