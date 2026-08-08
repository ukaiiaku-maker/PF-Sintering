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
from .tj_force import _circle_crossings, _sample_bilinear, _tangent_at


def _all_contour_points(f, p):
    pts = []
    for rc in find_contours(f, 0.5):
        pts.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
    return np.vstack(pts) if pts else np.empty((0, 2))


def _walk_branch(f, p, tj_xy, branch_dir, max_arclength, step=None):
    """Local circle-marching walk along the f=0.5 contour from tj_xy in
    branch_dir, using tj_force._circle_crossings/_tangent_at (the same
    local level-crossing machinery TJ branch detection itself uses) at
    each step instead of a global nearest-neighbor search.

    A first version of this used ch_crossover_diagnostics.
    trace_branch_profile's global nearest-unvisited-point walk, factored
    out verbatim. That algorithm has a real bug uncovered by this module's
    own use (never surfaced in Milestone 11, which only ever calls it with
    one branch_dir at a time, never needing two branches to actually
    diverge): the initial `proj > -2*W` candidate pre-filter cannot
    distinguish two branches whose directions are less than 90 degrees
    apart (a common TJ configuration -- confirmed on a real TJ: points on
    EITHER branch project positively onto the OTHER branch's direction
    whenever the interior angle between them is acute), so the walk from
    v_s1 and from v_s2 converged onto the IDENTICAL path, confirmed
    identical to float precision.

    This version instead only ever looks at a small circle of radius
    `step` (~1.5*dx) around the CURRENT point, and advances to whichever
    crossing on that circle is most aligned with the running tangent.
    Because the search radius is much smaller than the real separation
    between distinct branches (except within ~step of the TJ itself, where
    genuine ambiguity is unavoidable), it cannot jump to the wrong branch.
    Returns (path Nx2 array, s_cum 1D array) or (None, None) if unresolved."""
    step = step or 1.5 * p.dx
    tj = np.asarray(tj_xy, dtype=float)
    tangent = np.asarray(branch_dir, dtype=float)
    tnorm = np.linalg.norm(tangent)
    if tnorm < 1e-30:
        return None, None
    tangent = tangent / tnorm

    cur = tj.copy()
    path = [cur.copy()]
    s_cum = [0.0]
    total_s = 0.0
    n_max = int(max_arclength / step) + 5
    for _ in range(n_max):
        if total_s >= max_arclength:
            break
        angles = _circle_crossings(f, 0.5, cur, step, p)
        if not angles:
            break
        best_xy, best_tangent, best_score = None, None, -2.0
        for th in angles:
            xy = np.array([cur[0] + step * math.cos(th), cur[1] + step * math.sin(th)])
            pos_dir = xy - cur
            pos_dir = pos_dir / (np.linalg.norm(pos_dir) + 1e-30)
            score = float(np.dot(pos_dir, tangent))
            if score > best_score:
                t_local = _tangent_at(f, xy, cur, p)
                best_score = score
                best_xy, best_tangent = xy, t_local
        if best_xy is None or best_score < 0.3:
            break
        total_s += step
        cur = best_xy
        tn = np.linalg.norm(best_tangent)
        tangent = best_tangent / tn if tn > 1e-30 else tangent
        path.append(cur.copy())
        s_cum.append(total_s)
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


def branch_window_report(f, e1, e2, e3, s, p, tj_xy, branch_dir, window_width_factors=(2.0, 5.0)):
    """Report kappa_A/kappa_B in consecutive windows [0,2W], [2W,5W]
    (Section 11's near-TJ / far categories) along one branch from a TJ.
    Windows are wider than the literal "~1W/2W/3W" suggestion because a
    window must contain enough raw f=0.5 contour points for a well-posed
    Kasa fit (>=6): at the coarsest primary grid (dx=5nm, W=20nm) a 1W
    window contains only ~5 points (unresolved); 2W/3W-wide windows give
    >=8 at dx=5nm and proportionally more at finer dx. Returns a dict keyed
    by window index (0='near_TJ', ..., last='far')."""
    W = p.interface_width
    mu_field = mu_isotropic(f, e1, e2, e3, s, p)
    edges = [0.0] + [w * W for w in window_width_factors]
    windows = {}
    for i in range(len(edges) - 1):
        wc = window_curvature(f, e1, e2, e3, s, p, tj_xy, branch_dir, edges[i], edges[i + 1], mu_field=mu_field)
        windows[i] = wc
    return windows


def branch_mu_J_profile(f, mu_field, Jx, Jy, p, tj_xy, branch_dir, max_arclength, n_samples=40):
    """Sections 13-14: mu(s), J_tangent(s), J_normal(s), kappa(s) along a
    branch from tj_xy, using this module's fixed _walk_branch (see its
    docstring) rather than ch_crossover_diagnostics.trace_branch_profile
    (left unmodified -- an existing, in-service Milestone 11 module; its
    walk uses the same underlying algorithm and is presumed subject to the
    same branch-ambiguity failure mode near a TJ, since it is only ever
    exercised with a single branch_dir at a time in production use, but is
    out of scope to change here). Interpretable orientation: s increases
    away from the TJ along branch_dir; J_tangent>0 means flux directed
    AWAY from the TJ along the branch, J_tangent<0 means flux directed
    TOWARD the TJ."""
    path, s_cum = _walk_branch(f, p, tj_xy, branch_dir, max_arclength)
    if path is None or len(path) < 5:
        return None
    s_target = np.linspace(0.0, s_cum[-1], n_samples)
    x_r = np.interp(s_target, s_cum, path[:, 0])
    y_r = np.interp(s_target, s_cum, path[:, 1])

    tx = np.gradient(x_r, s_target)
    ty = np.gradient(y_r, s_target)
    tnorm = np.hypot(tx, ty) + 1e-30
    tx, ty = tx / tnorm, ty / tnorm
    nx_, ny_ = -ty, tx

    mu_s = _sample_bilinear(mu_field, x_r, y_r, p)
    Jx_s = _sample_bilinear(Jx, x_r, y_r, p)
    Jy_s = _sample_bilinear(Jy, x_r, y_r, p)
    J_tangent = Jx_s * tx + Jy_s * ty
    J_normal = Jx_s * nx_ + Jy_s * ny_
    local_half_window = max(3, len(path) // (2 * max(1, n_samples // 4)))
    kappa_s = []
    for st in s_target:
        i0 = int(np.searchsorted(s_cum, st))
        lo, hi = max(0, i0 - local_half_window), min(len(path), i0 + local_half_window + 1)
        kappa_s.append(_kasa_fit_curvature(path[lo:hi], f, p, min_points=3))
    kappa_s = np.array(kappa_s)
    dJ_tangent_ds = np.gradient(J_tangent, s_target)

    return dict(s=s_target.tolist(), x=x_r.tolist(), y=y_r.tolist(), mu=mu_s.tolist(),
                J_tangent=J_tangent.tolist(), J_normal=J_normal.tolist(), kappa=kappa_s.tolist(),
                dJ_tangent_ds=dJ_tangent_ds.tolist())


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
