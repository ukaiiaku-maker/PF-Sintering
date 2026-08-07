"""Physical contact-geometry diagnostics -- DIAGNOSTIC ONLY, not wired into
production physics.

Milestone 6 established that the legacy neck metrics (`x_neck` via
`model.contact_width`, `A_GB` via `diagnostics.contact_area`) are pure
functions of `(e1, e2)` alone -- so raw CH (which only evolves `f`)
*necessarily* gives `Delta x_neck = 0`, `Delta A_GB = 0` by construction.
That is a property of the diagnostic, not evidence that the physical
solid-solid contact is unchanged by CH. This module adds metrics that
depend on the *physical* solid field `f` as well as grain identity
`(e1, e2)`, so raw CH's effect on the actual contact geometry can be
measured directly rather than inferred from an eta-only proxy.

All functions here reuse the already-validated `tj_force.locate_neck_tjs`
/ `tj_force.compute_neck_tj_forces` for TJ location and GB branch
directions; none of that module is modified.

New metrics (exact definitions; see MILESTONE_6B_PHYSICAL_CONTACT_GEOMETRY_AUDIT.md
for the full report):

- `t_GB`, `n_GB`: mean GB tangent/normal, built from the two TJs' own
  `v_gb` (GB branch directions, each already validated to point away from
  its TJ into the bulk GB). Since the two point roughly toward each other
  across the GB, `t_GB = normalize(v_gb_bottom - v_gb_top)` combines them
  into a single "bottom-TJ-to-top-TJ" direction (sign convention only;
  `L_contact_TJ`/`d_n_TJ` use `abs(...)` so the direction's sign never
  matters, only its orientation relative to `t_GB` vs `n_GB`).
- `L_contact_TJ = abs(dot(r_top - r_bottom, t_GB))`: the physical distance
  between the two TJs, projected onto the local mean GB tangent -- i.e. the
  TJ-to-TJ contact length along the GB, not assumed to be exactly vertical.
- `d_n_TJ = abs(dot(r_top - r_bottom, n_GB))`: the TJ separation
  perpendicular to the mean GB tangent (should be small for a
  near-straight, near-vertical GB; a genuine diagnostic quantity, not
  assumed zero).
- `L_GB_geom`: arc length of the connected `e1=e2` contour segment between
  the two TJs, extracted from the level set of `e1-e2` masked to the
  physical solid region (`f > 0.5`) -- depends on both grain identity
  (`e1=e2`) and physical solid occupancy (`f`), unlike the old
  `integral(e1*e2)`-based `A_GB`.
- `L_f_farfield`: the x-position (same centered coordinate convention as
  `model.center()`/`diagnostics.wall_x0()`) of the farthest point of the
  `f=0.5` contour from the wall, minus `wall_x0`. This tracks the tip of
  the particle cap, which by construction of this geometry (wall near
  small x, cap tip near large x) is always many interface widths from the
  neck, so local TJ reshaping cannot masquerade as a change in this
  metric -- only genuine overall particle displacement or far-cap shape
  change can move it.
- `V2_f_eta = integral(f * e2/(e1+e2+e3+eps))`: solid-occupancy-weighted
  grain-2 volume -- differs from the legacy `V2_eta = integral(e2)`
  whenever `e1+e2+e3 != f` locally (i.e. whenever the eta fields don't
  exactly fill the available solid capacity).
"""

from __future__ import annotations

import math

import numpy as np
from skimage.measure import find_contours

from .model import center
from .tj_force import compute_neck_tj_forces, locate_neck_tjs


def mean_gb_tangent(rep):
    """t_GB, n_GB from the two TJs' own v_gb (tj_force.TJForceReport)."""
    if rep is None or not (rep.top and rep.top.resolved and rep.bottom and rep.bottom.resolved):
        return None, None
    raw = np.asarray(rep.bottom.v_gb) - np.asarray(rep.top.v_gb)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-30:
        return None, None
    t_gb = raw / norm
    n_gb = np.array([-t_gb[1], t_gb[0]])
    return t_gb, n_gb


def contact_length_tj(f, e1, e2, e3, s, p):
    """L_contact_TJ, d_n_TJ, and both TJ coordinates. Returns a dict with
    NaNs / None where the geometry isn't resolved, rather than raising."""
    tjs = locate_neck_tjs(f, e1, e2, p)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    out = dict(
        tj_top_x=math.nan, tj_top_y=math.nan, tj_bottom_x=math.nan, tj_bottom_y=math.nan,
        L_contact_TJ=math.nan, d_n_TJ=math.nan,
    )
    if tjs is None:
        return out
    r_top = np.asarray(tjs["tj_top"], dtype=float)
    r_bot = np.asarray(tjs["tj_bottom"], dtype=float)
    out["tj_top_x"], out["tj_top_y"] = float(r_top[0]), float(r_top[1])
    out["tj_bottom_x"], out["tj_bottom_y"] = float(r_bot[0]), float(r_bot[1])

    t_gb, n_gb = mean_gb_tangent(rep)
    if t_gb is None:
        return out
    d = r_top - r_bot
    out["L_contact_TJ"] = float(abs(np.dot(d, t_gb)))
    out["d_n_TJ"] = float(abs(np.dot(d, n_gb)))
    return out


def _gb_contour_polylines(f, e1, e2, p):
    """e1=e2 level-set contours, masked to the physical solid region
    (f > 0.5) so the result depends on both grain identity and physical
    occupancy. Returns a list of (N,2) arrays in the uncentered coordinate
    convention used throughout tj_force.py."""
    d = np.where(f > 0.5, e1 - e2, np.nan)
    polylines = []
    for rc in find_contours(d, 0.0):
        polylines.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
    return polylines


def gb_geom_length(f, e1, e2, p, tj_top=None, tj_bottom=None, snap_tol_widths=1.5):
    """Arc length of the connected e1=e2 contour segment (within physical
    solid) between the two TJs. Returns NaN if no contour connects both."""
    if tj_top is None or tj_bottom is None:
        tjs = locate_neck_tjs(f, e1, e2, p)
        if tjs is None:
            return math.nan
        tj_top, tj_bottom = tjs["tj_top"], tjs["tj_bottom"]
    tj_top = np.asarray(tj_top, dtype=float)
    tj_bottom = np.asarray(tj_bottom, dtype=float)
    tol = snap_tol_widths * p.interface_width

    best_len = None
    for poly in _gb_contour_polylines(f, e1, e2, p):
        if len(poly) < 2:
            continue
        d_top = np.hypot(poly[:, 0] - tj_top[0], poly[:, 1] - tj_top[1])
        d_bot = np.hypot(poly[:, 0] - tj_bottom[0], poly[:, 1] - tj_bottom[1])
        if d_top.min() > tol or d_bot.min() > tol:
            continue
        i_top = int(np.argmin(d_top))
        i_bot = int(np.argmin(d_bot))
        lo, hi = min(i_top, i_bot), max(i_top, i_bot)
        seg = poly[lo:hi + 1]
        if len(seg) < 2:
            continue
        length = float(np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))))
        if best_len is None or length < best_len:
            best_len = length
    return best_len if best_len is not None else math.nan


def far_field_position(f, wall_x0_val, p):
    """x-position (centered convention, matching model.center()) of the
    farthest point of the f=0.5 contour from the wall, minus wall_x0."""
    contours = find_contours(f, 0.5)
    if not contours:
        return math.nan
    pts = np.vstack([
        np.c_[(rc[:, 1] + 1 - p.Nx / 2) * p.dx, (rc[:, 0] + 1 - p.Ny / 2) * p.dx]
        for rc in contours
    ])
    x_tip = float(np.max(pts[:, 0]))
    return x_tip - wall_x0_val


def eta_centroid_separation(e2, wall_x0_val, p):
    """L_eta_centroid: unchanged from the existing separation diagnostic
    (diagnostics.sample's `separation_m`), reused here under an explicit
    name for direct comparison against L_f_farfield."""
    return float(center(e2, p) - wall_x0_val)


def V2_f_weighted(f, e1, e2, e3, p):
    """Solid-occupancy-weighted grain-2 volume: integral(f * e2/(e1+e2+e3))."""
    den = e1 + e2 + (e3 if p.use_eta3 else 0.0) + 1e-30
    q2 = e2 / den
    return float(np.sum(f * q2)) * p.dx * p.dx
