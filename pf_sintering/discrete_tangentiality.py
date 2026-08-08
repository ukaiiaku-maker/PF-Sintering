"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 13C Sections 4-5, 13: cell-centered and face-centered
tangentiality diagnostics for the unified transport law, using the EXACT
`n_f` the model's own tangential projector `P_t = I - n_f (x) n_f` is
built from -- not a fitted branch tangent.

Milestone 13B found `|J_normal / J_tangent| ~ 0.87` right at a TJ and
provisionally read it as evidence of a genuinely 2-D TJ-core transport
region. That measurement used `curvature_extraction.branch_mu_J_profile`'s
`J_normal`, defined relative to a LOCALLY FITTED CONTOUR TANGENT (a
geometric fit to the walked f=0.5 branch) -- a completely different object
from the model's own `n_f = grad(f)/sqrt(|grad f|^2+eps_n^2)`, the vector
`P_t` actually annihilates by construction. Algebraically, for a unit
vector `n`, `P_t . v` is exactly orthogonal to `n` for any `v`, so
`J . n_f = -M_s q(f) [(I - n_f n_f^T) grad(mu)] . n_f = 0` identically
(exact for `|n_f|=1`; `interface_normal`'s `eps_n` regularization makes
`|n_f|` strictly less than 1 by `O(eps_n^2/|grad f|^2)`, giving a
correspondingly tiny, controlled deviation). This module tests that
directly, distinguishing it explicitly from the OLD, branch-fit-relative
quantity, which is renamed `J_normal_relative_to_branch_fit` wherever
still used (Section 13).

Two distinct tests, because they can behave differently:

- `cell_tangentiality`: `J_cell . n_f` where both are evaluated at the
  SAME cell center, using `surface_transport.interface_normal` exactly --
  this must be at (eps_n-controlled) roundoff by algebraic construction,
  and confirming that numerically is a check on the implementation, not a
  new physics question.
- `face_tangentiality`: the ACTUAL face-centered flux the conservative
  update uses (`face_average` of cell-centered Jx/Jy, one component per
  face family) is NOT algebraically guaranteed tangential to any single
  face normal -- averaging two flux vectors that are each individually
  tangential to DIFFERENT local normals (if the interface curves between
  the two cells) generally does not produce a vector orthogonal to either.
  This is evaluated using a MAC-consistent face gradient/normal (the
  x-derivative at an x-face uses the true one-sided difference across
  that face; the y-derivative averages each neighboring cell's own
  centered y-derivative; the flux's own other component is likewise
  averaged onto the face) and is the genuine numerical qualification
  question Section 5 poses.
"""

from __future__ import annotations

import numpy as np

from .bc_ops import _shift, face_average
from .surface_transport import interface_normal


def cell_tangentiality(f, Jx, Jy, p, bc_x, bc_y, eps_n=None):
    """J . n_f at cell centers, using the exact interface_normal P_t is
    built from. Returns dict with the raw dot product field, |J|, and the
    normalized ratio (nan where |J| is negligible)."""
    if eps_n is None:
        eps_n = 1e-6 / p.interface_width
    nx, ny, _ = interface_normal(f, p.dx, bc_x, bc_y, eps_n)
    J_dot_n = Jx * nx + Jy * ny
    J_mag = np.hypot(Jx, Jy)
    ratio = np.divide(np.abs(J_dot_n), J_mag, out=np.full_like(J_mag, np.nan), where=J_mag > 0)
    return dict(J_dot_n=J_dot_n, J_mag=J_mag, ratio=ratio)


def _face_normal_and_flux_x(f, Jx, Jy, dx, bc_x, bc_y):
    """MAC-consistent normal and flux vector at each cell's '+x' face
    (between cell (r,c) and (r,c+1))."""
    f_right = _shift(f, -1, axis=1, bc=bc_x)
    gx_face = (f_right - f) / dx
    f_up = _shift(f, 1, axis=0, bc=bc_y)
    f_down = _shift(f, -1, axis=0, bc=bc_y)
    gy_cell = (f_down - f_up) / (2 * dx)
    gy_face = 0.5 * (gy_cell + _shift(gy_cell, -1, axis=1, bc=bc_x))
    gmag = np.hypot(gx_face, gy_face)
    nx = np.divide(gx_face, gmag, out=np.zeros_like(gmag), where=gmag > 0)
    ny = np.divide(gy_face, gmag, out=np.zeros_like(gmag), where=gmag > 0)

    Jx_face = face_average(Jx, axis=1, bc=bc_x)  # the ACTUAL flux the conservative update uses
    Jy_face = 0.5 * (Jy + _shift(Jy, -1, axis=1, bc=bc_x))  # Jy interpolated onto the SAME x-face
    return nx, ny, Jx_face, Jy_face


def _face_normal_and_flux_y(f, Jx, Jy, dx, bc_x, bc_y):
    """MAC-consistent normal and flux vector at each cell's '+y' face."""
    f_up = _shift(f, -1, axis=0, bc=bc_y)
    gy_face = (f_up - f) / dx
    f_right = _shift(f, 1, axis=1, bc=bc_x)
    f_left = _shift(f, -1, axis=1, bc=bc_x)
    gx_cell = (f_left - f_right) / (2 * dx)
    gx_face = 0.5 * (gx_cell + _shift(gx_cell, -1, axis=0, bc=bc_y))
    gmag = np.hypot(gx_face, gy_face)
    nx = np.divide(gx_face, gmag, out=np.zeros_like(gmag), where=gmag > 0)
    ny = np.divide(gy_face, gmag, out=np.zeros_like(gmag), where=gmag > 0)

    Jy_face = face_average(Jy, axis=0, bc=bc_y)  # the ACTUAL flux the conservative update uses
    Jx_face = 0.5 * (Jx + _shift(Jx, -1, axis=0, bc=bc_y))  # Jx interpolated onto the SAME y-face
    return nx, ny, Jx_face, Jy_face


def face_tangentiality(f, Jx, Jy, p, bc_x, bc_y):
    """J_face . n_face at both x-faces and y-faces, using the EXACT
    Jx_face/Jy_face `surface_flux_divergence_conservative` builds (not a
    re-derived diagnostic flux) and a MAC-consistent face normal. Returns
    combined (stacked x-face and y-face) dot-product, |J_face|, and
    normalized ratio fields, each shaped like f (one value per face-family
    per cell)."""
    nx_x, ny_x, Jx_x, Jy_x = _face_normal_and_flux_x(f, Jx, Jy, p.dx, bc_x, bc_y)
    nx_y, ny_y, Jx_y, Jy_y = _face_normal_and_flux_y(f, Jx, Jy, p.dx, bc_x, bc_y)

    dot_x = Jx_x * nx_x + Jy_x * ny_x
    mag_x = np.hypot(Jx_x, Jy_x)
    dot_y = Jx_y * nx_y + Jy_y * ny_y
    mag_y = np.hypot(Jx_y, Jy_y)

    ratio_x = np.divide(np.abs(dot_x), mag_x, out=np.full_like(mag_x, np.nan), where=mag_x > 0)
    ratio_y = np.divide(np.abs(dot_y), mag_y, out=np.full_like(mag_y, np.nan), where=mag_y > 0)

    return dict(dot_x=dot_x, mag_x=mag_x, ratio_x=ratio_x,
                dot_y=dot_y, mag_y=mag_y, ratio_y=ratio_y)


def _region_masks(p, tj_xy_list, radii_in_W):
    """Boolean masks for 'near a TJ within radius' at each requested
    radius (physical, in units of W), for the region-conditioned
    tangentiality report (Section 4-5: 'ordinary free surface', '3W of TJ',
    '1W of TJ')."""
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    out = {}
    for rf in radii_in_W:
        r = rf * p.interface_width
        near = np.zeros(X.shape, dtype=bool)
        for tj_xy in tj_xy_list:
            near |= np.hypot(X - tj_xy[0], Y - tj_xy[1]) <= r
        out[rf] = near
    return out


def tangentiality_summary(f, Jx, Jy, p, bc_x, bc_y, tj_xy_list, mag_floor_frac=1e-6):
    """Section 4-5's full report: max/RMS normalized |J.n| globally and
    within 1W/3W of any TJ (ordinary free surface = outside 3W), for BOTH
    the cell-centered and face-centered tests. `mag_floor_frac` sets the
    |J| threshold (relative to the field's own max |J|) below which a
    point is excluded from the ratio statistics (avoids division-by-
    (near)-zero dominating the max/RMS with meaningless bulk noise)."""
    cell = cell_tangentiality(f, Jx, Jy, p, bc_x, bc_y)
    face = face_tangentiality(f, Jx, Jy, p, bc_x, bc_y)
    masks = _region_masks(p, tj_xy_list, [1.0, 3.0])
    near_1W, near_3W = masks[1.0], masks[3.0]
    far = ~near_3W

    def stats(ratio, mag, region_mask=None):
        floor = mag_floor_frac * np.nanmax(mag)
        valid = np.isfinite(ratio) & (mag > floor)
        if region_mask is not None:
            valid = valid & region_mask
        if not np.any(valid):
            return dict(max=float("nan"), rms=float("nan"), n=0)
        vals = ratio[valid]
        return dict(max=float(np.max(vals)), rms=float(np.sqrt(np.mean(vals ** 2))), n=int(valid.sum()))

    return dict(
        cell=dict(
            global_=stats(cell["ratio"], cell["J_mag"]),
            within_1W=stats(cell["ratio"], cell["J_mag"], near_1W),
            within_3W=stats(cell["ratio"], cell["J_mag"], near_3W),
            far_from_tj=stats(cell["ratio"], cell["J_mag"], far),
        ),
        face_x=dict(
            global_=stats(face["ratio_x"], face["mag_x"]),
            within_1W=stats(face["ratio_x"], face["mag_x"], near_1W),
            within_3W=stats(face["ratio_x"], face["mag_x"], near_3W),
            far_from_tj=stats(face["ratio_x"], face["mag_x"], far),
        ),
        face_y=dict(
            global_=stats(face["ratio_y"], face["mag_y"]),
            within_1W=stats(face["ratio_y"], face["mag_y"], near_1W),
            within_3W=stats(face["ratio_y"], face["mag_y"], near_3W),
            far_from_tj=stats(face["ratio_y"], face["mag_y"], far),
        ),
    )
