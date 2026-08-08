"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 12B Sections 11-12: two independent curvature-extraction methods,
reported as averages over FIXED PHYSICAL arclength windows (not fixed cell
counts, not single TJ pixels), for comparing donor/receiver capillary state
across grid resolutions.

Method A (diffuse-interface / production thermodynamics): the local
isotropic chemical potential `mu = mu_isotropic(...)` (bulk + gradient,
`ch_exact_energy.py`) sampled along the f=0.5 contour and averaged over the
window, converted to a curvature via `kappa_A = mu / (1.5*gamma_s)`.

The 1.5 factor (NOT production's `sigma_curv = gamma_s*kappa` mechanical
convention, which is an independent, separately-normalized quantity) comes
directly from this model's k_f=3*gamma_s*W, W_f=12*gamma_s/W normalization:
substituting the planar equilibrium profile f(r)=0.5(1-tanh((r-R)/W)) into
mu = W_f*f(1-f)(1-2f) - k_f*(f''(r)+f'(r)/r) (2D radial Laplacian), the
bracketed planar part [W_f*f(1-f)(1-2f) - k_f*f''] vanishes identically (the
defining ODE of the planar profile), leaving mu(R) = -k_f*f'(R)/R =
-k_f*(-1/(2W))/R = k_f/(2WR) = 1.5*gamma_s/R -- i.e. mu = 1.5*gamma_s*kappa,
not gamma_s*kappa. Verified numerically to <0.03% on synthetic circular
interfaces (R=300/500/800nm, W=20nm) against the exact analytic kappa=1/R,
with independent agreement from Method B's geometric fit at the same
window (see tests/test_curvature_extraction.py). Valid to leading order in
W/interface radius; degrades inside a strongly perturbed diffuse core (e.g.
very close to a TJ), which is exactly why the far/near-TJ window split is
reported separately rather than a single blended number.

Method B (geometric contour curvature): direct Kasa circle fit
(`signed_curvature.signed_curvature_at`'s core, reused here) to the actual
f=0.5 contour points falling within an explicit arclength window, walked
from a TJ along a given branch direction (reusing
`ch_crossover_diagnostics.trace_branch_profile`'s contour-walk). Makes no
reference to mu at all -- a purely geometric cross-check on Method A.

Both methods share the same window definition (arclength interval [s_lo,
s_hi] measured in physical length units from the TJ along a branch), so
their outputs are directly comparable at any grid spacing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.measure import find_contours

from .ch_exact_energy import mu_isotropic
from .tj_force import _sample_bilinear


def _all_contour_points(f, p):
    pts = []
    for rc in find_contours(f, 0.5):
        pts.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
    return np.vstack(pts) if pts else np.empty((0, 2))


def _walk_branch(f, p, tj_xy, branch_dir, max_arclength):
    """Greedy nearest-neighbor arclength walk along the f=0.5 contour from
    tj_xy in branch_dir -- identical algorithm to
    ch_crossover_diagnostics.trace_branch_profile's contour walk, factored
    out here so Methods A and B use exactly the same point set/ordering.
    Returns (path Nx2 array, s_cum 1D array) or (None, None) if unresolved."""
    all_pts = _all_contour_points(f, p)
    if len(all_pts) == 0:
        return None, None
    tj = np.asarray(tj_xy, dtype=float)
    d_ = np.asarray(branch_dir, dtype=float)
    rel = all_pts - tj
    dist = np.linalg.norm(rel, axis=1)
    proj = rel @ d_
    cand = all_pts[(dist < max_arclength * 1.5) & (proj > -2 * p.interface_width)]
    if len(cand) < 5:
        return None, None

    remaining = cand.copy()
    d0 = np.linalg.norm(remaining - tj, axis=1)
    start_idx = int(np.argmin(d0))
    path = [remaining[start_idx]]
    remaining = np.delete(remaining, start_idx, axis=0)
    s_cum = [0.0]
    cur = path[0]
    total_s = 0.0
    while len(remaining) > 0 and total_s < max_arclength:
        dd = np.linalg.norm(remaining - cur, axis=1)
        j = int(np.argmin(dd))
        step_d = float(dd[j])
        if step_d > 4 * p.dx:
            break
        total_s += step_d
        cur = remaining[j]
        path.append(cur)
        s_cum.append(total_s)
        remaining = np.delete(remaining, j, axis=0)
    if len(path) < 3:
        return None, None
    return np.array(path), np.array(s_cum)


def _kasa_fit_curvature(points, f, p, min_points=6):
    """Kasa least-squares circle fit + inside-solid sign convention --
    identical construction to signed_curvature.signed_curvature_at, but
    applied to an explicit window's point set rather than a Euclidean
    radius around a center."""
    if len(points) < min_points:
        return math.nan
    x, y = points[:, 0], points[:, 1]
    x0, y0 = x.mean(), y.mean()
    sc = max(np.max(np.abs(x - x0)), np.max(np.abs(y - y0)))
    if sc <= 0:
        return math.nan
    u, v = (x - x0) / sc, (y - y0) / sc
    A = np.c_[2 * u, 2 * v, np.ones_like(u)]
    q = np.linalg.lstsq(A, u * u + v * v, rcond=None)[0]
    cu, cv, c = q
    R2 = c + cu ** 2 + cv ** 2
    if R2 <= 0:
        return math.nan
    R = math.sqrt(R2) * sc
    center = (x0 + cu * sc, y0 + cv * sc)
    ci = int(np.clip(round(center[0] / p.dx - 1.0), 0, p.Nx - 1))
    ri = int(np.clip(round(center[1] / p.dx - 1.0), 0, p.Ny - 1))
    inside_solid = f[ri, ci] > 0.5
    sign = 1.0 if inside_solid else -1.0
    return sign / R


@dataclass
class WindowCurvature:
    resolved: bool = False
    n_points: int = 0
    s_lo: float = math.nan
    s_hi: float = math.nan
    kappa_A: float = math.nan       # Method A: mean(mu)/gamma_s over the window
    kappa_B: float = math.nan       # Method B: single Kasa fit over the window
    mu_mean: float = math.nan


def window_curvature(f, e1, e2, e3, s, p, tj_xy, branch_dir, s_lo, s_hi, mu_field=None):
    """Curvature via both methods, averaged/fit over the physical arclength
    window [s_lo, s_hi] measured from tj_xy along branch_dir."""
    path, s_cum = _walk_branch(f, p, tj_xy, branch_dir, max_arclength=s_hi + 2 * p.interface_width)
    out = WindowCurvature(s_lo=s_lo, s_hi=s_hi)
    if path is None:
        return out
    mask = (s_cum >= s_lo) & (s_cum <= s_hi)
    win_pts = path[mask]
    out.n_points = int(mask.sum())
    if out.n_points < 6:
        return out

    if mu_field is None:
        mu_field = mu_isotropic(f, e1, e2, e3, s, p)
    mu_win = _sample_bilinear(mu_field, win_pts[:, 0], win_pts[:, 1], p)
    out.mu_mean = float(np.mean(mu_win))
    out.kappa_A = out.mu_mean / (1.5 * p.gamma_s)
    out.kappa_B = _kasa_fit_curvature(win_pts, f, p)
    out.resolved = math.isfinite(out.kappa_A) and math.isfinite(out.kappa_B)
    return out


def branch_window_report(f, e1, e2, e3, s, p, tj_xy, branch_dir, window_width_factors=(1.0, 2.0, 3.0)):
    """Report kappa_A/kappa_B in consecutive windows [0,1W], [1W,2W], [2W,3W]
    (Section 11's suggested ~1W/2W/3W windows) along one branch from a TJ.
    Returns a dict keyed by window index (0='near_TJ', ..., last='far')."""
    W = p.interface_width
    mu_field = mu_isotropic(f, e1, e2, e3, s, p)
    edges = [0.0] + [w * W for w in window_width_factors]
    windows = {}
    for i in range(len(edges) - 1):
        wc = window_curvature(f, e1, e2, e3, s, p, tj_xy, branch_dir, edges[i], edges[i + 1], mu_field=mu_field)
        windows[i] = wc
    return windows


def delta_kappa_donor_receiver(windows_donor, windows_receiver, window_idx=-1):
    """Delta kappa = <kappa_donor_region> - <kappa_receiver_region>
    (Section 12) using kappa_A (production thermodynamics) at the requested
    window index (default: the last/far window of each branch report)."""
    wd = windows_donor[window_idx if window_idx >= 0 else max(windows_donor)]
    wr = windows_receiver[window_idx if window_idx >= 0 else max(windows_receiver)]
    if not (wd.resolved and wr.resolved):
        return math.nan, math.nan
    dkappa = wd.kappa_A - wr.kappa_A
    dmu = wd.mu_mean - wr.mu_mean
    return dkappa, dmu
