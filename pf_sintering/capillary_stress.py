"""Milestone 14C: capillary/sintering-stress diagnostic construction.

DIAGNOSTIC ONLY -- does not feed back into any evolution equation. Builds,
from the geometric f=0.5 particle free-surface contour (never from mu), the
capillary force transmitted across the particle/substrate contact and an
"apparent 2-D sintering stress" derived from it, plus local capillary-
pressure and curvature diagnostics. `F_TJ` (tj_force.py) remains a SEPARATE,
purely local Young-Herring-residual diagnostic -- it is not, and does not
substitute for, the quantities defined here (Milestone 14C Section 2).

Derivation (Sections 4-6): for an isotropic interface parametrized by
physical arclength s with unit tangent t(s), the Frenet relation
dt/ds = kappa(s)*n(s) holds by construction once n(s) is defined as t(s)
rotated +90 degrees (a FIXED, consistent rotation applied at every point --
this is what makes integral(kappa*n ds) telescope to Delta_t up to pure
discretization error, not a separately-assumed sign convention). This
"curvature form" is the whole-arc contour integral of gamma_s*kappa(s)*n(s);
the "endpoint form" gamma_s*(t_end - t_start), using the TWO independently-
measured TJ free-surface branch directions (from tj_force.compute_tj_force
+ classify_branches -- a completely different local computation, not
derived from the traced arc at all), is a genuinely independent check on
the same physical identity. Validated empirically (Milestone 14C report
Section 4): the two agree to ~a few percent with a smoothing scale of
1.5*interface_width applied to the resampled arc before differentiating --
both smaller (raw walk noise) and larger (endpoint-tangent bias from the
smoothing kernel's edge handling) smoothing scales give worse agreement,
so 1.5*W is used as the default.

For future anisotropic generalization (Section 20, not enabled here):
`gamma_s * t` is the isotropic Cahn-Hoffman vector xi; replacing it with
the anisotropic xi(theta) would not change any function signature here,
only how `_walk_branch`'s underlying field or the isotropic gamma_s
scalars are provided upstream.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .curvature_extraction import _walk_branch, window_curvature

DEFAULT_MAX_ARCLENGTH = 560e-9
DEFAULT_N_SAMPLES = 400
DEFAULT_SMOOTHING_WIDTHS = 1.5


def trace_particle_arc(f, p, tj_top_xy, tj_bottom_xy, top_particle_dir, bot_particle_dir,
                        max_arclength=DEFAULT_MAX_ARCLENGTH, n_samples=DEFAULT_N_SAMPLES,
                        smoothing_widths=DEFAULT_SMOOTHING_WIDTHS):
    """Milestone 14C Section 4: walk the f=0.5 contour from the top TJ in
    the particle-branch direction (never the substrate or GB branch),
    truncate at closest approach to the bottom TJ, resample onto a uniform
    arclength grid, and return smoothed tangent/normal/signed-curvature
    arrays along physical arclength s. `closest_approach_dist` (should be
    ~1 grid cell) is the truncation-quality diagnostic; `max_arclength`
    must exceed the true particle-side arc length (~full particle
    perimeter minus the short buried/contact arc) or the walk will not
    reach the bottom TJ region at all."""
    path, s_cum = _walk_branch(f, p, tj_top_xy, top_particle_dir, max_arclength)
    if path is None or len(path) < 8:
        return dict(resolved=False, reason="walk failed or too short")
    dist = np.hypot(path[:, 0] - tj_bottom_xy[0], path[:, 1] - tj_bottom_xy[1])
    imin = int(np.argmin(dist))
    closest_dist = float(dist[imin])
    if imin < 5:
        return dict(resolved=False, reason="closest approach too near start of walk")
    if imin == len(path) - 1:
        return dict(resolved=False, reason="walk truncated by max_arclength before reaching bottom TJ",
                     closest_approach_dist=closest_dist)
    path, s_cum = path[:imin + 1], s_cum[:imin + 1]
    arc_length = float(s_cum[-1])

    s_target = np.linspace(0.0, arc_length, n_samples)
    ds = s_target[1] - s_target[0]
    x_r = np.interp(s_target, s_cum, path[:, 0])
    y_r = np.interp(s_target, s_cum, path[:, 1])
    sigma_idx = smoothing_widths * p.interface_width / ds
    xs = gaussian_filter1d(x_r, sigma_idx, mode="nearest")
    ys = gaussian_filter1d(y_r, sigma_idx, mode="nearest")

    tx = np.gradient(xs, s_target)
    ty = np.gradient(ys, s_target)
    tn = np.hypot(tx, ty) + 1e-30
    tx, ty = tx / tn, ty / tn
    nx, ny = -ty, tx  # t rotated +90deg, held fixed for every point

    dtx = np.gradient(tx, s_target)
    dty = np.gradient(ty, s_target)
    kappa = dtx * nx + dty * ny  # dt/ds . n -- dt/ds is parallel to n by |t|=1, so this is |dt/ds| signed by n

    return dict(resolved=True, s=s_target, x=xs, y=ys, tx=tx, ty=ty, nx=nx, ny=ny,
                kappa=kappa, arc_length=arc_length, closest_approach_dist=closest_dist,
                n_raw_points=len(path))


def capillary_force_curvature_form(arc, gamma_s):
    """Section 5: F_cap_vector = integral gamma_s*kappa(s)*n(s) ds over the
    traced particle arc."""
    Fx = float(gamma_s * np.trapezoid(arc["kappa"] * arc["nx"], arc["s"]))
    Fy = float(gamma_s * np.trapezoid(arc["kappa"] * arc["ny"], arc["s"]))
    return Fx, Fy


def capillary_force_endpoint_form(top_particle_dir, bot_particle_dir, gamma_s):
    """Section 6: the SAME resultant computed only from the two
    independently-measured TJ free-surface branch directions (never the
    traced arc). `top_particle_dir` already points away from the top TJ
    (the arc's start-of-travel direction); `bot_particle_dir` points away
    from the bottom TJ, so the arc's direction of travel AT the bottom TJ
    (arriving, not departing) is its negative."""
    t_start = np.asarray(top_particle_dir, dtype=float)
    t_end = -np.asarray(bot_particle_dir, dtype=float)
    return float(gamma_s * (t_end[0] - t_start[0])), float(gamma_s * (t_end[1] - t_start[1]))


def force_validation_relative_error(F_curvature, F_endpoints):
    Fx_c, Fy_c = F_curvature
    Fx_e, Fy_e = F_endpoints
    num = math.hypot(Fx_c - Fx_e, Fy_c - Fy_e)
    den = max(math.hypot(Fx_e, Fy_e), 1e-30)
    return num / den


def apparent_sintering_stress(F_cap_vector, n_GB, L_contact):
    """Section 7-8: sigma_sint_app = -F_cap_n/L_contact, F_cap_n =
    F_cap_vector . n_GB. `n_GB` must point from substrate toward particle
    (see `oriented_contact_normal`) so that a capillary force pulling the
    particle cap back toward the substrate (the physically expected
    densifying direction, empirically confirmed negative-x here) yields a
    POSITIVE apparent stress. Returns (sigma_sint_app, F_cap_n)."""
    F_cap_n = float(F_cap_vector[0] * n_GB[0] + F_cap_vector[1] * n_GB[1])
    if not (L_contact and math.isfinite(L_contact) and L_contact > 0):
        return math.nan, F_cap_n
    return -F_cap_n / L_contact, F_cap_n


def oriented_contact_normal(n_GB_sub):
    """Section 3: `tj_subgrid.compute_subgrid_contact`'s own `n_GB_sub` is
    a bare +90deg rotation of the GB tangent with no guaranteed physical
    orientation. Empirically (Milestone 14C Section 3) it points from
    PARTICLE toward SUBSTRATE (-x, toward lower x where the substrate
    bulk sits) in this geometry; this function returns its negation, i.e.
    the direction from substrate toward particle (+x here), which is the
    orientation `apparent_sintering_stress` assumes."""
    return (-n_GB_sub[0], -n_GB_sub[1])


def window_mean_kappa(arc, s_lo, s_hi):
    """Mean of the traced arc's own smoothed kappa(s) within [s_lo, s_hi]
    measured from the arc's start (top TJ, s=0)."""
    if not arc.get("resolved", False):
        return math.nan
    s = arc["s"]
    mask = (s >= s_lo) & (s <= s_hi)
    if not np.any(mask):
        return math.nan
    return float(np.mean(arc["kappa"][mask]))


def window_mean_kappa_from_end(arc, s_lo, s_hi):
    """Same as `window_mean_kappa` but measured from the arc's END (bottom
    TJ, s=arc_length) -- for the bottom-TJ near/far windows."""
    if not arc.get("resolved", False):
        return math.nan
    L = arc["arc_length"]
    return window_mean_kappa(arc, L - s_hi, L - s_lo)


def substrate_curvature_windows(f, e1, e2, e3, s, p, tj_xy, substrate_dir,
                                 windows=((1.5, 3.0), (3.0, 5.0))):
    """Section 11: windowed geometric (Kasa-fit) curvature along the
    SUBSTRATE branch from a TJ, reusing the existing, previously-qualified
    `curvature_extraction.window_curvature` machinery (never re-deriving a
    new curvature estimator for the substrate side)."""
    out = {}
    for lo_f, hi_f in windows:
        wc = window_curvature(f, e1, e2, e3, s, p, tj_xy, substrate_dir,
                               lo_f * p.interface_width, hi_f * p.interface_width)
        out[f"{lo_f}W-{hi_f}W"] = dict(resolved=wc.resolved, kappa_geom=wc.kappa_geom,
                                        n_points=wc.n_points)
    return out
