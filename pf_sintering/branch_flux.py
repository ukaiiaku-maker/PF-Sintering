"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 13B: branch-resolved surface-flux closure.

Milestone 13's two flux diagnostics disagreed: `curvature_extraction.
branch_mu_J_profile`'s single-point `J_tangent` samples at/near a TJ showed
a clear particle-toward-TJ / substrate-away-from-TJ pattern with
comparable magnitudes (~1e-9), while `flux_closure.
neck_boundary_face_flux_balance`'s Cartesian-position particle/substrate
split of the SAME neck control volume's boundary showed the
substrate-facing contribution at ~1e-5 to 1e-6 of the particle-facing one.

The resolution (confirmed below): these were never measuring the same
thing. `neck_boundary_face_flux_balance` classifies each boundary FACE of
a disk-shaped control volume by ABSOLUTE POSITION (X-deviation from a
far-field substrate baseline) -- but the flux-carrying part of that
boundary is concentrated ONLY where the actual free surface crosses it
(q(f) vanishes away from any interface), typically at just two points (one
per branch). If BOTH of those crossing points happen to fall on the same
side of the absolute-position threshold (plausible for a small control
volume near a still-nearly-planar contact, where the substrate branch's
own crossing point is still geometrically closer to "the particle side" of
a threshold calibrated against the FAR-FIELD substrate baseline), the
position classifier silently misattributes the substrate branch's real
flux to the "particle" bucket -- explaining the apparent absence of
substrate-directed flux without requiring any inconsistency in the
underlying conservative flux field itself.

This module instead identifies the two branches by IDENTITY (reusing
`curvature_extraction._walk_branch`'s already-validated local
circle-marching walker), and measures the flux carried by each branch
directly: `branch_cut_flux` integrates `J . t` across a cut of finite
physical width normal to the branch's local tangent, at fixed arclength
`s_cut` from the TJ, capturing the full interface-localized flux (q(f) is
concentrated within ~W of the f=0.5 contour, so the cut must span several
W in the interface-normal direction -- convergence with the integration
half-width is checked explicitly, not assumed).
"""

from __future__ import annotations

import math

import numpy as np

from .bc_ops import flux_divergence
from .curvature_extraction import _walk_branch
from .surface_transport import face_average
from .tj_force import _sample_bilinear


def _local_tangent_at_arclength(path, s_cum, s_cut, window=2):
    """Point and unit tangent on `path` at arclength `s_cut`, via linear
    interpolation for the point and a short symmetric finite difference
    (±`window` path samples) for the tangent -- consistent with how
    `curvature_extraction.branch_mu_J_profile` estimates local tangents,
    but evaluated at one specific arclength rather than a dense grid."""
    if s_cum[-1] < s_cut:
        return None
    x_cut = float(np.interp(s_cut, s_cum, path[:, 0]))
    y_cut = float(np.interp(s_cut, s_cum, path[:, 1]))
    idx = int(np.searchsorted(s_cum, s_cut))
    lo, hi = max(0, idx - window), min(len(path), idx + window + 1)
    if hi - lo < 2:
        return None
    tvec = path[hi - 1] - path[lo]
    tnorm = np.linalg.norm(tvec)
    if tnorm < 1e-30:
        return None
    t_hat = tvec / tnorm
    n_hat = np.array([-t_hat[1], t_hat[0]])
    return dict(x_cut=x_cut, y_cut=y_cut, t_hat=t_hat, n_hat=n_hat)


def branch_cut_flux(f, Jx, Jy, p, tj_xy, branch_dir, s_cut, normal_halfwidths,
                     samples_per_dx=4):
    """Section 3-4: integrate J.t_hat across a cut normal to the branch's
    local tangent, at arclength s_cut from tj_xy along branch_dir, for
    each half-width in `normal_halfwidths` (physical length, e.g. multiples
    of p.interface_width) -- reports Q at every half-width so convergence
    with increasing integration width can be checked directly rather than
    assumed. Sign convention: positive Q means flux directed AWAY from the
    TJ along the branch (t_hat's own orientation, matching
    curvature_extraction.branch_mu_J_profile's J_tangent convention) --
    callers apply the physical inflow/outflow sign (Section 5) themselves.
    Returns None if the branch does not reach s_cut."""
    max_arclength = s_cut + max(normal_halfwidths) + 2 * p.dx
    path, s_cum = _walk_branch(f, p, tj_xy, branch_dir, max_arclength)
    if path is None:
        return None
    cut = _local_tangent_at_arclength(path, s_cum, s_cut)
    if cut is None:
        return None
    x_cut, y_cut, t_hat, n_hat = cut["x_cut"], cut["y_cut"], cut["t_hat"], cut["n_hat"]

    Q_by_halfwidth = {}
    for hw in normal_halfwidths:
        n_pts = max(21, int(round(2 * hw / p.dx * samples_per_dx)) | 1)
        xi = np.linspace(-hw, hw, n_pts)
        sx = x_cut + xi * n_hat[0]
        sy = y_cut + xi * n_hat[1]
        Jx_s = _sample_bilinear(Jx, sx, sy, p)
        Jy_s = _sample_bilinear(Jy, sx, sy, p)
        J_t = Jx_s * t_hat[0] + Jy_s * t_hat[1]
        Q_by_halfwidth[hw] = float(np.trapezoid(J_t, xi))

    return dict(x_cut=x_cut, y_cut=y_cut, t_hat=t_hat.tolist(), n_hat=n_hat.tolist(),
                s_cut=s_cut, Q_by_halfwidth=Q_by_halfwidth)


def disk_control_volume_mask(p, center_xy, radius):
    """Simple Euclidean disk mask (uncentered (index+1)*dx coordinate
    convention, matching tj_force/tj_subgrid/flux_closure), centered at
    `center_xy` (e.g. a TJ location) with the given physical `radius` --
    used to build a control volume directly tied to the same s_cut
    parameter governing the branch cuts, for an apples-to-apples
    three-way closure test (Section 6)."""
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    d = np.hypot(X - center_xy[0], Y - center_xy[1])
    return d <= radius


def cartesian_total_boundary_flux(Jx, Jy, mask, p, bc_x, bc_y):
    """The exact volume-integrated -div(J) over `mask` (== the net
    boundary-face inflow by bc_ops.flux_divergence's discrete divergence
    theorem) -- Section 6's "B" quantity, using the TOTAL only (no
    positional particle/substrate split, per Section 2's explicit
    instruction not to use that split as the physical branch decomposition)."""
    Jx_face = face_average(Jx, axis=1, bc=bc_x)
    Jy_face = face_average(Jy, axis=0, bc=bc_y)
    neg_div = -flux_divergence(Jx_face, Jy_face, p.dx, bc_x=bc_x, bc_y=bc_y)
    return float(np.sum(neg_div[mask])) * p.dx * p.dx


def branch_flux_balance_at_tj(f, Jx, Jy, p, tj_xy, particle_dir, substrate_dir,
                               s_cut, normal_halfwidths):
    """Section 5: Q_p (flux from the particle branch INTO the neck/TJ,
    positive inward) and Q_s (flux from the neck/TJ OUT along the
    substrate branch, positive outward), at every normal_halfwidths entry,
    plus the disk control-volume total (Section 6's "B") at radius=s_cut
    for the SAME cut location, so all three closure quantities share one
    physical scale parameter."""
    qp_raw = branch_cut_flux(f, Jx, Jy, p, tj_xy, particle_dir, s_cut, normal_halfwidths)
    qs_raw = branch_cut_flux(f, Jx, Jy, p, tj_xy, substrate_dir, s_cut, normal_halfwidths)
    if qp_raw is None or qs_raw is None:
        return None

    Q_p = {hw: -qp_raw["Q_by_halfwidth"][hw] for hw in normal_halfwidths}   # toward TJ = inflow, positive
    Q_s = {hw: qs_raw["Q_by_halfwidth"][hw] for hw in normal_halfwidths}    # away from TJ = outflow, positive

    return dict(
        tj_xy=tuple(float(v) for v in tj_xy), s_cut=s_cut,
        particle_cut=qp_raw, substrate_cut=qs_raw,
        Q_p=Q_p, Q_s=Q_s,
        net_by_halfwidth={hw: Q_p[hw] - Q_s[hw] for hw in normal_halfwidths},
    )
