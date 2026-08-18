"""Milestone 16Q Sections 2-5: branch-resolved neck curvature.

M16Q's central metrology correction: for finite-chi (finite neighbor-
curvature) geometry, the particle-side free-surface branch and the
neighbor-side free-surface branch meet at the triple junction (TJ) with a
physical Young-Herring dihedral-angle CORNER (a genuine slope
discontinuity), not a smooth feature, unless chi=1 exactly. Every prior
driver (`m16k_neck_tracking.NeckTracker` + `hussein_neck_stress.
neck_curvature_windows`) fit ONE circle across a window straddling that
corner -- fitting a physical kink as though it were one smooth arc. That
is retired here for finite-chi use (the old machinery is untouched and
still valid for genuinely smooth single-branch situations).

This module instead:
  1. Locates the TJ as the intersection of the f=0.5 free-surface contour
     with the particle/substrate grain-identity boundary (reusing
     `m16k_neck_tracking.find_tj_from_contour`, sub-grid interpolated).
  2. Classifies every contour point as particle-side (e2 dominant) or
     neighbor/substrate-side (e1 dominant) via the SAME grain-identity
     test, using `pf_sintering.grain_roles`' e1=substrate, e2=particle
     convention.
  3. Builds each branch's LOCAL point set (a physical window on ONE side
     of the TJ only -- points are NEVER mixed across the TJ into a single
     fit), reparameterizes by cumulative chord/arc length s, and fits
     z(s), R(s) with an independent LOCAL LEAST-SQUARES quadratic
     polynomial (an arc-length generalization of `m16m_stress_consensus.
     polynomial_curvature_r_neck`'s z-based quadratic fit). An exact
     interpolating cubic spline was tried first and rejected: forcing the
     fit through the TJ point itself created a pathologically short first
     arc-length segment next to normal-length ones, destabilizing the
     boundary curvature estimate by >30x at chi=1.5 -- a smooth
     least-squares fit over the branch's own real sample points does not
     have that failure mode.
  4. Evaluates the meridional curvature at the TJ-nearest real sample of
     each branch as
     kappa = |z'(s) R''(s) - R'(s) z''(s)| / (z'(s)^2+R'(s)^2)^1.5
     (the literal formula requested, general enough for near-vertical
     branches), giving a POSITIVE geometric radius r_branch = 1/|kappa|
     per branch.

Feeds `hussein_neck_stress.hussein_eq1b_sigma` once per branch (same
X_neck=2*r_tj for both, since X is the shared TJ contact width), never
producing a single "operative" sigma silently -- callers get
sigma_particle, sigma_neighbor, AND sigma_avg explicitly.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma
from pf_sintering.m16k_neck_tracking import find_tj_from_contour

PARTICLE = "particle"
NEIGHBOR = "neighbor"

# M16Q Section 8 finding: a per-PF-step diagnostic replay at chi=1.5,
# W=4nm/dx=0.55nm showed R_of_z carries a genuine period-2 (single-
# timestep, checkerboard/Nyquist) numerical wobble with a TINY raw
# amplitude (~0.003nm in z_tj) that a curvature (2nd-derivative) estimator
# amplifies into several-nm swings in r_branch (i.e. several MPa in sigma)
# every sample -- reproduced directly against the raw field, independent
# of any window/branch-splitting choice.
#
# A per-branch Gaussian pre-smoothing of R_pts (same precedent as
# `m16m_stress_consensus.diffuse_interface_r_neck`'s
# `gaussian_filter(..., sigma=2.0)` on the phase field itself) was tried
# and REJECTED: the branch's own point set near the TJ is short (5-10
# points) and curvature is evaluated AT its edge (s=0, the TJ-nearest
# point) by design -- smoothing with any real boundary condition biases
# exactly that edge, degrading the t=0 ground-truth accuracy from <=5% to
# >40% when tried. Left OFF by default (0) for that reason; the
# machinery is kept (see `branch_local_curvature`'s `smooth_sigma_cells`
# argument) in case a future TEMPORAL (not spatial) filter -- e.g. a
# lag-1 moving average across consecutive PF steps, matching the
# oscillation's actual period-2-in-TIME character -- is built on top of
# it. Instead: W=6nm/dx=0.85nm (this module's primary validated
# resolution, Section 6) does NOT show this sustained oscillation in
# practice (a full t=0..0.10 sink-off replay showed only 3 large jumps,
# all within the first few samples of the initial interface-profile
# relaxation transient, none afterward) -- W=4nm/dx=0.55nm is documented
# as exhibiting this fine-grid checkerboard mode and is NOT used for
# continuous per-step stress monitoring without further work.
DEFAULT_SMOOTH_SIGMA_CELLS = 0.0


def classify_contour_points(R_of_z, z, e1, e2, r_c):
    """Grain-identity label per finite contour point: e1(z,R(z))-e2(z,R(z)),
    via nearest r-index (grid resolution is fine enough, same convention as
    `find_tj_from_contour`). >0 => neighbor/substrate(e1) dominant,
    <0 => particle(e2) dominant."""
    diff = np.full_like(np.asarray(R_of_z, dtype=float), np.nan)
    idx = np.where(np.isfinite(R_of_z))[0]
    for j in idx:
        i = int(np.argmin(np.abs(r_c - R_of_z[j])))
        diff[j] = e1[j, i] - e2[j, i]
    return diff


def _spline_curvature_at_end(s, z_pts, R_pts, s_eval, degree=2):
    """Fits z(s), R(s) with a LOCAL LEAST-SQUARES polynomial (default
    quadratic -- an arc-length-parameterized generalization of
    `m16m_stress_consensus.polynomial_curvature_r_neck`'s z-based quadratic
    fit, using the general parametric curvature formula
    kappa=|z'R''-R'z''|/(z'^2+R'^2)^1.5 valid for near-vertical branches
    too) and evaluates at s_eval. A least-squares fit over the whole
    window is used rather than an EXACT interpolating spline: an
    interpolating cubic spline forced through every sample, evaluated
    right at the boundary nearest a genuine Young-Herring corner, was
    verified to be numerically unstable there (an artificially short first
    segment inflated the measured curvature by >30x at chi=1.5) -- a
    smooth low-order least-squares fit does not have that failure mode.
    Returns (kappa, r_branch)."""
    if len(s) < degree + 2:
        return float("nan"), float("nan")
    cz = np.polyfit(s, z_pts, deg=degree)
    cR = np.polyfit(s, R_pts, deg=degree)
    dcz1 = np.polyder(cz, 1)
    dcz2 = np.polyder(cz, 2)
    dcR1 = np.polyder(cR, 1)
    dcR2 = np.polyder(cR, 2)
    zp = float(np.polyval(dcz1, s_eval))
    zpp = float(np.polyval(dcz2, s_eval)) if dcz2.size else 0.0
    Rp = float(np.polyval(dcR1, s_eval))
    Rpp = float(np.polyval(dcR2, s_eval)) if dcR2.size else 0.0
    denom = (zp * zp + Rp * Rp) ** 1.5
    if denom <= 1e-300:
        return float("nan"), float("nan")
    kappa = abs(zp * Rpp - Rp * zpp) / denom
    r_branch = 1.0 / kappa if kappa > 1e-300 else float("inf")
    return kappa, r_branch


def branch_local_curvature(R_of_z, z, diff, z_tj, r_tj, side, half_window, min_points=5,
                            smooth_sigma_cells=DEFAULT_SMOOTH_SIGMA_CELLS):
    """Local spline-based curvature of ONE branch only, using points within
    `half_window` of the TJ on the physical side selected by `side`
    (PARTICLE: diff<0, i.e. e2 dominant; NEIGHBOR: diff>0, i.e. e1
    dominant). Points from the OTHER branch are never included.

    The TJ point itself is NOT forced into the spline's data: at a genuine
    Young-Herring corner the two branches' arc-length spacing is
    physically discontinuous there, and a knot placed at zero (or near-
    zero) chord distance from its neighbor destabilizes a cubic spline's
    boundary derivative estimate (verified directly: including it inflated
    the measured curvature by >30x at chi=1.5). Instead the branch's OWN
    innermost sampled point (closest real grid sample to the TJ, on this
    branch only) anchors s=0, and curvature is evaluated there -- the
    natural, numerically stable proxy for "curvature at the neck" on this
    branch.

    `smooth_sigma_cells>0` lightly Gaussian-smooths this branch's OWN
    R_pts (only, AFTER the side_mask split -- never the other branch's
    points, never anything spanning the TJ corner) before fitting: a
    per-PF-step replay showed R_of_z carries a period-2 (single-timestep)
    numerical wobble of tiny raw amplitude that a curvature (2nd-
    derivative) estimator amplifies by orders of magnitude; this
    suppresses it while leaving the much longer-wavelength true curvature
    signal essentially unchanged. Returns a dict."""
    if side == PARTICLE:
        side_mask = diff < 0
    elif side == NEIGHBOR:
        side_mask = diff > 0
    else:
        raise ValueError(f"side must be {PARTICLE!r} or {NEIGHBOR!r}, got {side!r}")

    mask = np.isfinite(R_of_z) & np.isfinite(diff) & side_mask & (np.abs(z - z_tj) <= half_window)
    idx = np.where(mask)[0]
    n_points = len(idx)
    if n_points < min_points:
        return dict(side=side, n_points=n_points, kappa=float("nan"), r_branch=float("nan"), s_tj=float("nan"))

    z_pts, R_pts = z[idx], R_of_z[idx]
    order = np.argsort(z_pts)
    z_pts, R_pts = z_pts[order], R_pts[order]
    # branch is ordered so index 0 is nearest the TJ on this side (particle:
    # ascending z with TJ below; neighbor: ascending z with TJ above) --
    # reorient so the TJ-nearest sample is always first.
    if side == PARTICLE:
        pass  # z ascending, TJ is at the smallest z already
    else:
        z_pts, R_pts = z_pts[::-1], R_pts[::-1]  # TJ is at the largest z

    if smooth_sigma_cells and smooth_sigma_cells > 0 and len(R_pts) >= 3:
        eff_sigma = min(smooth_sigma_cells, max(0.1, (len(R_pts) - 1) / 6.0))
        R_pts = gaussian_filter1d(R_pts, sigma=eff_sigma, mode="nearest")

    ds = np.hypot(np.diff(z_pts), np.diff(R_pts))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s, z_pts, R_pts = s[keep], z_pts[keep], R_pts[keep]
    if len(s) < 4:
        return dict(side=side, n_points=n_points, kappa=float("nan"), r_branch=float("nan"), s_tj=float("nan"))

    s_tj = float(s[0])
    try:
        kappa, r_branch = _spline_curvature_at_end(s, z_pts, R_pts, s_tj)
    except (ValueError, np.linalg.LinAlgError):
        kappa, r_branch = float("nan"), float("nan")
    return dict(side=side, n_points=n_points, kappa=float(kappa), r_branch=float(r_branch), s_tj=s_tj)


def branch_resolved_curvature_windows(R_of_z, z, e1, e2, r_c, z_tj, r_tj, W,
                                       window_widths_in_W=(1.5, 2.5, 3.0), min_points=5,
                                       smooth_sigma_cells=DEFAULT_SMOOTH_SIGMA_CELLS):
    """Per-window (`window_widths_in_W`), per-branch local curvature. Returns
    dict(particle=[...], neighbor=[...]) each a list of
    `branch_local_curvature` dicts, one per window width -- mirrors
    `hussein_neck_stress.neck_curvature_windows`'s list-of-dicts shape but
    NEVER mixes points across the TJ."""
    diff = classify_contour_points(R_of_z, z, e1, e2, r_c)
    out = {PARTICLE: [], NEIGHBOR: []}
    for w_mult in window_widths_in_W:
        half_window = w_mult * W
        for side in (PARTICLE, NEIGHBOR):
            res = branch_local_curvature(R_of_z, z, diff, z_tj, r_tj, side, half_window, min_points=min_points,
                                          smooth_sigma_cells=smooth_sigma_cells)
            res["window_W"] = w_mult
            out[side].append(res)
    return out


def measure_branch_resolved_sigma(f, e1, e2, r_c, z, R_of_z, z_gb_guess, W, gamma_s, gamma_gb,
                                   window_widths_in_W=(0.75, 1.0, 1.5, 2.5), window_W_for_sigma=1.0,
                                   search_frac=0.15, min_points=5,
                                   smooth_sigma_cells=DEFAULT_SMOOTH_SIGMA_CELLS):
    """Top-level M16Q measurement: locates the TJ, computes BOTH branches'
    local curvature/radius, and reports sigma_particle, sigma_neighbor,
    sigma_avg via `hussein_eq1b_sigma` with a SHARED X_neck=2*r_tj (the TJ
    is a single shared geometric point -- X is not branch-specific).
    `window_W_for_sigma` selects which entry of `window_widths_in_W` is
    used for the headline sigma values (all windows are still returned for
    QC/convergence checks). Never silently assumes symmetry: both branch
    values are always returned, even when they disagree strongly.

    `smooth_sigma_cells` (default `DEFAULT_SMOOTH_SIGMA_CELLS`) is forwarded
    to each branch's OWN curvature fit (see `branch_local_curvature`) to
    suppress the period-2 checkerboard artifact documented above (pass 0
    to disable, e.g. for an exact t=0 ground-truth check where the input
    is already analytically smooth). TJ location itself uses the RAW
    (unsmoothed) contour -- deliberately: smoothing R_of_z GLOBALLY before
    branch-splitting was tried first and rejected, because the raw
    contour is continuous straight across the TJ corner (a single R(z)
    array spans both branches with no gap there), so a global filter
    blends points from BOTH sides of the corner into each other right at
    the corner -- precisely the branch-mixing this module exists to
    prevent, and it degraded the t=0 ground-truth accuracy by >10x when
    tried. Smoothing is applied per-branch, AFTER the grain-identity split,
    on that branch's own points only."""
    tj = find_tj_from_contour(f, e1, e2, r_c, z, R_of_z, z_gb_guess, search_frac=search_frac)
    z_tj, r_tj = tj["z_tj"], tj["r_tj"]
    if not (np.isfinite(z_tj) and np.isfinite(r_tj)):
        return dict(z_tj=z_tj, r_tj=r_tj, tj_info=tj, windows=None,
                    r_particle=float("nan"), r_neighbor=float("nan"),
                    sigma_particle=float("nan"), sigma_neighbor=float("nan"), sigma_avg=float("nan"),
                    X_neck=float("nan"), C_GB=float("nan"))

    windows = branch_resolved_curvature_windows(R_of_z, z, e1, e2, r_c, z_tj, r_tj, W,
                                                 window_widths_in_W=window_widths_in_W, min_points=min_points,
                                                 smooth_sigma_cells=smooth_sigma_cells)
    X_neck = 2.0 * r_tj

    def _pick(side):
        for entry in windows[side]:
            if entry["window_W"] == window_W_for_sigma:
                return entry
        return windows[side][0] if windows[side] else None

    entry_p = _pick(PARTICLE)
    entry_n = _pick(NEIGHBOR)
    r_particle = entry_p["r_branch"] if entry_p is not None else float("nan")
    r_neighbor = entry_n["r_branch"] if entry_n is not None else float("nan")

    def _sigma(r_branch):
        if not (np.isfinite(r_branch) and r_branch != 0):
            return float("nan"), float("nan")
        sigma, _, _, C_GB = hussein_eq1b_sigma(r_branch, X_neck, gamma_s, gamma_gb)
        return sigma, C_GB

    sigma_particle, C_GB = _sigma(r_particle)
    sigma_neighbor, C_GB2 = _sigma(r_neighbor)
    C_GB_final = C_GB if np.isfinite(C_GB) else C_GB2
    finite_sigmas = [s for s in (sigma_particle, sigma_neighbor) if np.isfinite(s)]
    sigma_avg = float(np.mean(finite_sigmas)) if finite_sigmas else float("nan")

    return dict(z_tj=z_tj, r_tj=r_tj, tj_info=tj, windows=windows,
                r_particle=r_particle, r_neighbor=r_neighbor,
                sigma_particle=sigma_particle, sigma_neighbor=sigma_neighbor, sigma_avg=sigma_avg,
                X_neck=X_neck, C_GB=C_GB_final)
