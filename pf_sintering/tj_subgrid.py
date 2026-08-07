"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Continuous sub-grid TJ locator (Milestone 6C).

MILESTONE_6B_PHYSICAL_CONTACT_GEOMETRY_AUDIT.md found that
`tj_force.locate_neck_tjs`'s TJ coordinate is locked to the array's
discrete row index (`(solid[-1]+1)*dx`) *and* inherits the discrete column
selected by its narrowest-extent search -- i.e. quantized in both x and y,
in steps of `~dx`, roughly 250x coarser than the physical signal at the
`|dV2|/V20 ~ 3e-4` scale used throughout this investigation.
`locate_neck_tjs` is unchanged and still used here only as a local search
seed (per its own docstring, it remains useful as a backwards-compatible
diagnostic and an explicit demonstration of grid quantization).

Each physical TJ is defined as the continuous 2D intersection of two level
sets:

    f(x, y) = 0.5          (solid/vapor boundary)
    g(x, y) = e1(x,y) - e2(x,y) = 0   (solid-solid grain ownership boundary)

Algorithm (matches the handoff's recommended implementation exactly):

1. Extract the `f=0.5` and `e1-e2=0` contours as sub-pixel polylines via
   `skimage.measure.find_contours` (marching squares already linearly
   interpolates each grid-line crossing, so these polylines are
   continuous, not grid-snapped).
2. Restrict both to segments within a local window (`search_radius`) of
   the legacy discrete TJ seed.
3. Find every pairwise line-segment intersection between the two local
   polyline sets.
4. Merge near-duplicate candidates (adjacent segments sharing a vertex
   commonly produce the same intersection twice); if more than one
   distinct candidate remains, the result is reported unresolved
   ("ambiguous") rather than silently picking one.
5. Refine the single surviving candidate with a few Newton iterations on
   `F1 = f_interp(x,y) - 0.5`, `F2 = e1_interp(x,y) - e2_interp(x,y)`,
   using bilinear interpolation (`tj_force._sample_bilinear`) and its
   finite-difference gradient (`tj_force._local_gradient`) -- both reused
   unmodified.
6. Reject (mark unresolved) if Newton fails to converge or the solution
   leaves the local neighborhood.

Quality/conditioning metrics are always reported: the field residuals at
the solution, distance from the legacy discrete TJ, the local angle
between the two level-set contours (near 0/180 degrees means a
near-tangent, ill-conditioned intersection), the local 2x2 level-set
Jacobian determinant, the candidate count, and an explicit resolved flag
with failure reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.measure import find_contours

from .tj_force import _local_gradient, _sample_bilinear, locate_neck_tjs


# ---------------------------------------------------------------------------
# Contour / segment primitives
# ---------------------------------------------------------------------------

def _contour_polylines(field, level, p):
    """Sub-pixel polylines of `field == level`, in the same uncentered
    physical coordinate convention as tj_force.py ((col+1)*dx, (row+1)*dx)."""
    polylines = []
    for rc in find_contours(field, level):
        polylines.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
    return polylines


def _segments_near(poly, center, radius):
    segs = []
    for i in range(len(poly) - 1):
        p1, p2 = poly[i], poly[i + 1]
        mid = 0.5 * (p1 + p2)
        if math.hypot(mid[0] - center[0], mid[1] - center[1]) <= radius:
            segs.append((p1, p2))
    return segs


def _closest_point_on_segment(p1, p2, q):
    d = p2 - p1
    denom = float(np.dot(d, d))
    if denom < 1e-30:
        return p1.copy()
    t = float(np.clip(np.dot(q - p1, d) / denom, 0.0, 1.0))
    return p1 + t * d


def _closest_approach(segs_a, segs_b):
    """Smallest distance (and the corresponding midpoint) between any
    segment in segs_a and any segment in segs_b, evaluated via closest-point-
    on-segment against each endpoint of the other (exact for the common case
    where the true closest approach is realized at an endpoint or a normal
    foot on one of the two segments)."""
    best_d = math.inf
    best_mid = None
    for a1, a2 in segs_a:
        for b1, b2 in segs_b:
            for q in (b1, b2):
                c = _closest_point_on_segment(a1, a2, q)
                d = float(np.hypot(c[0] - q[0], c[1] - q[1]))
                if d < best_d:
                    best_d, best_mid = d, 0.5 * (c + q)
            for q in (a1, a2):
                c = _closest_point_on_segment(b1, b2, q)
                d = float(np.hypot(c[0] - q[0], c[1] - q[1]))
                if d < best_d:
                    best_d, best_mid = d, 0.5 * (c + q)
    return best_d, best_mid


def _segment_intersection(p1, p2, p3, p4, eps=1e-9):
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-18:
        return None  # parallel (or degenerate) segments
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return p1 + t * d1
    return None


def _newton_refine(f, gb, p, xy0, max_iter=20, tol_frac_dx=1e-7):
    """Newton iteration on bilinearly-interpolated (f-0.5, e1-e2). Uses a
    finer finite-difference stencil (0.02*dx rather than tj_force's default
    0.5*dx) for the gradient: a coarser stencil averages the bilinear
    interpolant's (piecewise-constant, cell-wise) gradient across multiple
    cells and was found empirically to produce a damped-oscillatory (linear,
    ratio ~0.7/step) rather than quadratic Newton convergence -- correct in
    direction but needlessly slow. The finer stencil restores standard
    quadratic Newton convergence (machine precision within ~8 iterations)."""
    xy = np.array(xy0, dtype=float)
    tol = tol_frac_dx * p.dx
    h = 0.02 * p.dx
    detJ = math.nan
    for _ in range(max_iter):
        F1 = float(_sample_bilinear(f, [xy[0]], [xy[1]], p)[0]) - 0.5
        F2 = float(_sample_bilinear(gb, [xy[0]], [xy[1]], p)[0])
        g1 = _local_gradient(f, xy, p, h=h)
        g2 = _local_gradient(gb, xy, p, h=h)
        J = np.array([g1, g2])
        detJ = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
        if abs(detJ) < 1e-30:
            return xy, False, (F1, F2), detJ
        delta = np.linalg.solve(J, -np.array([F1, F2]))
        xy = xy + delta
        if float(np.linalg.norm(delta)) < tol:
            F1f = float(_sample_bilinear(f, [xy[0]], [xy[1]], p)[0]) - 0.5
            F2f = float(_sample_bilinear(gb, [xy[0]], [xy[1]], p)[0])
            return xy, True, (F1f, F2f), detJ
    F1f = float(_sample_bilinear(f, [xy[0]], [xy[1]], p)[0]) - 0.5
    F2f = float(_sample_bilinear(gb, [xy[0]], [xy[1]], p)[0])
    return xy, False, (F1f, F2f), detJ


# ---------------------------------------------------------------------------
# Single-TJ locator
# ---------------------------------------------------------------------------

@dataclass
class SubgridTJ:
    resolved: bool = False
    reason: str = ""
    x_sub: float = math.nan
    y_sub: float = math.nan
    f_residual: float = math.nan
    gb_residual: float = math.nan
    dist_from_legacy: float = math.nan
    contour_angle_deg: float = math.nan
    jacobian_det: float = math.nan
    n_candidates: int = 0
    newton_converged: bool = False
    used_closest_approach_fallback: bool = False


def locate_tj_subgrid_single(f, e1, e2, p, seed_xy, search_radius, merge_tol_frac_dx=0.1):
    gb = e1 - e2
    f_polys = _contour_polylines(f, 0.5, p)
    gb_polys = _contour_polylines(gb, 0.0, p)

    f_segs = []
    for poly in f_polys:
        f_segs.extend(_segments_near(poly, seed_xy, search_radius))
    gb_segs = []
    for poly in gb_polys:
        gb_segs.extend(_segments_near(poly, seed_xy, search_radius))

    candidates = []
    for a1, a2 in f_segs:
        for b1, b2 in gb_segs:
            xy = _segment_intersection(a1, a2, b1, b2)
            if xy is not None:
                candidates.append(xy)

    fallback_used = False
    if not candidates:
        # No two segments literally cross -- can happen when the true
        # intersection lies very close to a marching-squares vertex, where
        # the two independently-traced polylines can pass near each other
        # without crossing. Fall back to the closest approach between the
        # two local polyline sets as the Newton seed, but only if the gap is
        # itself small (a real near-miss, not two unrelated contours).
        closest_d, closest_mid = _closest_approach(f_segs, gb_segs)
        if closest_mid is None or closest_d > 0.5 * p.dx:
            return SubgridTJ(resolved=False, reason="no candidate f/GB contour intersections in neighborhood")
        candidates = [closest_mid]
        fallback_used = True

    merge_tol = merge_tol_frac_dx * p.dx
    merged = []
    for c in candidates:
        if not any(math.hypot(c[0] - m[0], c[1] - m[1]) < merge_tol for m in merged):
            merged.append(c)

    if len(merged) > 1:
        return SubgridTJ(resolved=False, n_candidates=len(merged),
                          reason=f"{len(merged)} distinct candidate intersections (ambiguous)")

    xy_refined, converged, (F1, F2), detJ = _newton_refine(f, gb, p, merged[0])
    dist_seed = float(math.hypot(xy_refined[0] - seed_xy[0], xy_refined[1] - seed_xy[1]))

    if not converged:
        return SubgridTJ(resolved=False, n_candidates=1, reason="Newton did not converge",
                          x_sub=float(xy_refined[0]), y_sub=float(xy_refined[1]),
                          f_residual=abs(F1), gb_residual=abs(F2), jacobian_det=detJ,
                          used_closest_approach_fallback=fallback_used)
    if dist_seed > search_radius:
        return SubgridTJ(resolved=False, n_candidates=1, reason="Newton solution left local neighborhood",
                          x_sub=float(xy_refined[0]), y_sub=float(xy_refined[1]),
                          f_residual=abs(F1), gb_residual=abs(F2), jacobian_det=detJ,
                          used_closest_approach_fallback=fallback_used)

    g1 = _local_gradient(f, xy_refined, p)
    g2 = _local_gradient(gb, xy_refined, p)
    n1n, n2n = float(np.linalg.norm(g1)), float(np.linalg.norm(g2))
    if n1n < 1e-30 or n2n < 1e-30:
        angle_deg = math.nan
    else:
        cosang = float(np.clip(np.dot(g1 / n1n, g2 / n2n), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(abs(cosang)))  # 0/180 = tangent (ill-conditioned), 90 = well-conditioned

    return SubgridTJ(
        resolved=True, reason="", n_candidates=1, newton_converged=True,
        x_sub=float(xy_refined[0]), y_sub=float(xy_refined[1]),
        f_residual=abs(F1), gb_residual=abs(F2),
        contour_angle_deg=angle_deg, jacobian_det=detJ,
        used_closest_approach_fallback=fallback_used,
    )


def locate_neck_tjs_subgrid(f, e1, e2, p, search_radius_widths=3.0):
    """Wraps the unmodified locate_neck_tjs() for a local seed, then locates
    both TJs as continuous 2D level-set intersections. Returns None if the
    legacy locator itself fails (same failure mode as before).

    The search radius is capped at a fraction of the legacy locator's own
    `neck_height` (top-to-bottom TJ separation), matching the same
    adaptive-radius convention `tj_force._radii_for_tj` already uses for its
    circle-crossing search -- found necessary empirically: for a narrower
    neck (e.g. the 5nm-overlap baseline geometry), a fixed
    `search_radius_widths * interface_width` can exceed half the TJ
    separation, so each TJ's local neighborhood picks up a spurious second
    candidate belonging to the *other* TJ and both correctly (but
    unhelpfully) report "ambiguous" rather than resolving either.
    """
    legacy = locate_neck_tjs(f, e1, e2, p)
    if legacy is None:
        return None
    radius = min(search_radius_widths * p.interface_width, 0.4 * legacy["neck_height"])
    radius = max(radius, 0.5 * p.interface_width)
    top = locate_tj_subgrid_single(f, e1, e2, p, legacy["tj_top"], radius)
    bottom = locate_tj_subgrid_single(f, e1, e2, p, legacy["tj_bottom"], radius)
    if top.resolved:
        top.dist_from_legacy = float(math.hypot(top.x_sub - legacy["tj_top"][0], top.y_sub - legacy["tj_top"][1]))
    if bottom.resolved:
        bottom.dist_from_legacy = float(
            math.hypot(bottom.x_sub - legacy["tj_bottom"][0], bottom.y_sub - legacy["tj_bottom"][1])
        )
    return dict(legacy=legacy, top=top, bottom=bottom)


# ---------------------------------------------------------------------------
# Continuous local tangents and contact-length metrics
# ---------------------------------------------------------------------------

def _gb_tangent_at(gb_field, xy, away_reference_xy, p):
    g = _local_gradient(gb_field, xy, p)
    gn = float(np.linalg.norm(g))
    if gn < 1e-30:
        return None
    t = np.array([-g[1], g[0]]) / gn
    away = np.array(away_reference_xy) - np.array(xy)
    if np.dot(t, away) < 0:
        t = -t
    return t


def gb_geom_length_sub(e1, e2, p, xy_top, xy_bottom, snap_tol_widths=3.0):
    """Arc length of the e1=e2 contour between the two CONTINUOUS TJ
    coordinates. The terminal vertices of the traced discrete polyline
    segment (already within one marching-squares cell of the true
    intersection) are replaced by the continuous coordinates directly,
    rather than left at the nearest contour vertex -- i.e. the terminal
    segments are split/extended at the true sub-grid endpoint."""
    gb = e1 - e2
    tol = snap_tol_widths * p.interface_width
    best = None
    for poly in _contour_polylines(gb, 0.0, p):
        if len(poly) < 2:
            continue
        d_top = np.hypot(poly[:, 0] - xy_top[0], poly[:, 1] - xy_top[1])
        d_bot = np.hypot(poly[:, 0] - xy_bottom[0], poly[:, 1] - xy_bottom[1])
        if d_top.min() > tol or d_bot.min() > tol:
            continue
        i_top, i_bot = int(np.argmin(d_top)), int(np.argmin(d_bot))
        lo, hi = min(i_top, i_bot), max(i_top, i_bot)
        seg = poly[lo:hi + 1].copy()
        if len(seg) < 2:
            continue
        if i_top <= i_bot:
            seg[0], seg[-1] = xy_top, xy_bottom
        else:
            seg[0], seg[-1] = xy_bottom, xy_top
        length = float(np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1]))))
        if best is None or length < best:
            best = length
    return best if best is not None else math.nan


@dataclass
class SubgridContact:
    resolved: bool
    top: SubgridTJ
    bottom: SubgridTJ
    t_GB_sub: object = None
    n_GB_sub: object = None
    L_contact_TJ_sub: float = math.nan
    d_n_TJ_sub: float = math.nan
    L_GB_geom_sub: float = math.nan


def compute_subgrid_contact(f, e1, e2, p, search_radius_widths=3.0):
    sub = locate_neck_tjs_subgrid(f, e1, e2, p, search_radius_widths=search_radius_widths)
    if sub is None or not (sub["top"].resolved and sub["bottom"].resolved):
        top = sub["top"] if sub else SubgridTJ(resolved=False, reason="legacy locator failed")
        bottom = sub["bottom"] if sub else SubgridTJ(resolved=False, reason="legacy locator failed")
        return SubgridContact(resolved=False, top=top, bottom=bottom)

    gb = e1 - e2
    r_top = np.array([sub["top"].x_sub, sub["top"].y_sub])
    r_bot = np.array([sub["bottom"].x_sub, sub["bottom"].y_sub])

    t_top = _gb_tangent_at(gb, r_top, r_bot, p)
    t_bot = _gb_tangent_at(gb, r_bot, r_top, p)
    if t_top is None or t_bot is None:
        return SubgridContact(resolved=False, top=sub["top"], bottom=sub["bottom"])

    raw = t_bot - t_top
    norm = float(np.linalg.norm(raw))
    if norm < 1e-30:
        return SubgridContact(resolved=False, top=sub["top"], bottom=sub["bottom"])
    t_gb = raw / norm
    n_gb = np.array([-t_gb[1], t_gb[0]])

    d = r_top - r_bot
    L_contact = float(abs(np.dot(d, t_gb)))
    d_n = float(abs(np.dot(d, n_gb)))
    L_geom = gb_geom_length_sub(e1, e2, p, r_top, r_bot)

    return SubgridContact(
        resolved=True, top=sub["top"], bottom=sub["bottom"],
        t_GB_sub=tuple(float(x) for x in t_gb), n_GB_sub=tuple(float(x) for x in n_gb),
        L_contact_TJ_sub=L_contact, d_n_TJ_sub=d_n, L_GB_geom_sub=L_geom,
    )
