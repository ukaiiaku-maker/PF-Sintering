"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 9: Ostwald receiver/reservoir closure diagnostics.

Milestone 8's O0/O1/O2 decomposition (ostwald_diagnostics.py) showed removal
alone shrinks the physical contact while addition alone widens it more than
normal (O0) operation. This module goes one level deeper on the addition
side specifically:

- `region_masks_by_tj_distance`: partitions space into bands by distance to
  the nearest continuous sub-grid TJ (Milestone 6C), for spatially
  decomposing WHERE in the production `add(x,y)` field the widening comes
  from (Section 6).
- `ostwald_partial_addition_step` / `ostwald_external_reservoir_step`:
  apply only a spatially- and/or fractionally-restricted portion of the
  EXACT, UNMODIFIED production addition field (`ostwald_diagnostics.
  ostwald_substrate_diagnostic`'s own `diag.add`) -- never renormalized,
  since renormalizing would artificially re-concentrate deposition and
  destroy the spatial attribution the handoff explicitly warns against.
  The omitted/undeposited mass is reported as `reservoir_delta` rather than
  silently discarded, so a diagnostic external reservoir can be tracked and
  global mass conservation verified (Section 9).

Three named closures (Section 10), all built from the same primitive:

- Model A (current production): `phi_local=1.0, region_mask=None` --
  identical to O0.
- Model B (external reservoir, no local redeposition): `phi_local=0.0` --
  identical to O1's field effect, but with reservoir mass tracked instead
  of silently discarded.
- Model C (physically-scaled local receiver): `0 < phi_local < 1`, derived
  from actual receiver-geometry considerations, not tuned for a desired
  sign (see MILESTONE_9_OSTWALD_RESERVOIR_CLOSURE_AUDIT.md Section 10 for
  the derivation attempted, or its absence).
"""

from __future__ import annotations

import math

import numpy as np

from .ostwald_diagnostics import ostwald_substrate_diagnostic


def region_masks_by_tj_distance(p, tj_top_xy, tj_bottom_xy, band_edges_in_W=(0, 1, 3, 5)):
    """Boolean masks partitioning the domain by distance (in units of
    p.interface_width) to whichever TJ is nearer. band_edges_in_W=(0,1,3,5)
    gives bands '0W_1W', '1W_3W', '3W_5W', and '>5W' (Section 6 A-D)."""
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    d_top = np.hypot(X - tj_top_xy[0], Y - tj_top_xy[1])
    d_bot = np.hypot(X - tj_bottom_xy[0], Y - tj_bottom_xy[1])
    d = np.minimum(d_top, d_bot)

    masks = {}
    edges = [e * p.interface_width for e in band_edges_in_W]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f"{band_edges_in_W[i]:g}W_{band_edges_in_W[i + 1]:g}W"
        masks[label] = (d >= lo) & (d < hi)
    masks[f">{band_edges_in_W[-1]:g}W"] = d >= edges[-1]
    return masks


def ostwald_external_reservoir_step(f, e1, e2, e3, p, phi_local=1.0, region_mask=None):
    """General Ostwald closure primitive. Removal is always the exact
    production removal field (unchanged). Of the exact production addition
    field `add(x,y)`, only `phi_local` times the portion inside
    `region_mask` (None = everywhere) is deposited locally; everything else
    (both the masked-out region and the (1-phi_local) fraction of the kept
    region) is reported as `reservoir_delta` (mass leaving the locally
    represented field, meters^2 in physical presets) rather than discarded.

    phi_local=1.0, region_mask=None reproduces O0 (Model A) with
    reservoir_delta identically 0. phi_local=0.0 reproduces O1's field
    effect (Model B) while additionally tracking the diverted mass.
    Returns (f_new, e1_new, e2_new, e3_new, reservoir_delta)."""
    _, _, _, _, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
    add_full = diag.add
    add_region = add_full if region_mask is None else add_full * region_mask
    add_kept = phi_local * add_region
    reservoir_delta = float(np.sum(add_full - add_kept)) * p.dx * p.dx

    e2_new = e2 - diag.remove
    f_after_removal = f - diag.remove
    f_new = f_after_removal + add_kept
    e1_new = e1 + add_kept
    return f_new, e1_new, e2_new, e3, reservoir_delta


def ostwald_partial_addition_step(f, e1, e2, e3, p, region_mask):
    """Convenience wrapper: phi_local=1.0 within region_mask, 0 outside --
    the exact production addition field restricted to one spatial region,
    NOT renormalized (Section 6/7's one-step spatial-partition test)."""
    return ostwald_external_reservoir_step(f, e1, e2, e3, p, phi_local=1.0, region_mask=region_mask)
