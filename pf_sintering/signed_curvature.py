"""Signed local free-surface curvature -- audit companion to `tj_force.py`.

`model.curvature()` (used inside `model.compute_stress` for the legacy
`sigma_curv = gamma_s * kappa` term) fits a circle to local contour points via
a Kasa least-squares fit and returns `1/R`. That value is **unsigned**: the
Kasa fit's `R2 = c + cu**2 + cv**2` term has no information about which side
of the interface the fitted center lies on, so `1/sqrt(R2)` is a magnitude
regardless of whether the surface is locally convex (bulging into vapor, e.g.
a particle cap) or concave (a neck/groove fillet).

This module adds a *separate* signed curvature, used only by the new TJ-force
diagnostics -- it never overwrites or feeds into `model.curvature()` or the
legacy `sigma`. Sign convention: positive when the fitted circle's center
lies inside the solid (`f > 0.5`) -- the classic "convex particle cap"
case, driving evaporation/shrinkage in the Gibbs-Thomson sense; negative when
the center lies in the vapor -- the "concave neck/groove fillet" case, which
promotes local deposition.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.measure import find_contours


def _contour_points(f, p):
    pts = []
    for rc in find_contours(f, 0.5):
        pts.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
    return np.vstack(pts) if pts else np.empty((0, 2))


def _sample_f(f, xy, p):
    ci = xy[0] / p.dx - 1.0
    ri = xy[1] / p.dx - 1.0
    ci = int(np.clip(round(ci), 0, p.Nx - 1))
    ri = int(np.clip(round(ri), 0, p.Ny - 1))
    return float(f[ri, ci])


def signed_curvature_at(f, center_xy, p, radius=None, min_points=6):
    """Signed curvature of the f=0.5 contour within `radius` of `center_xy`."""
    if radius is None:
        radius = max(3 * p.interface_width, 6 * p.dx)
    pts = _contour_points(f, p)
    if len(pts) == 0:
        return math.nan
    d = np.hypot(pts[:, 0] - center_xy[0], pts[:, 1] - center_xy[1])
    P = pts[d < radius]
    if len(P) < min_points:
        return math.nan
    x, y = P[:, 0], P[:, 1]
    x0, y0 = x.mean(), y.mean()
    sc = max(np.max(np.abs(x - x0)), np.max(np.abs(y - y0)))
    if sc <= 0:
        return math.nan
    u, v = (x - x0) / sc, (y - y0) / sc
    A = np.c_[2 * u, 2 * v, np.ones_like(u)]
    q = np.linalg.lstsq(A, u * u + v * v, rcond=None)[0]
    cu, cv, c = q
    R2 = c + cu**2 + cv**2
    if R2 <= 0:
        return math.nan
    R = math.sqrt(R2) * sc
    center = (x0 + cu * sc, y0 + cv * sc)
    inside_solid = _sample_f(f, center, p) > 0.5
    sign = 1.0 if inside_solid else -1.0
    return sign / R


def signed_curvature_top_bottom(f, tj_top, tj_bottom, p, radius=None):
    """Convenience wrapper reporting signed curvature near each TJ, matching
    the per-TJ organization of the tj_force diagnostics."""
    return (
        signed_curvature_at(f, tj_top, p, radius=radius),
        signed_curvature_at(f, tj_bottom, p, radius=radius),
    )
