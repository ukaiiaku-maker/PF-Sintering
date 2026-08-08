"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 11: instantaneous CH-only contact tendency, neck-region CH mass
balance, and free-surface chemical-potential/flux branch profiles, used to
test whether ordinary (Ostwald-off) surface diffusion on the sinusoidal
contact geometry naturally reverses from widening to thinning as the
particle+substrate morphology relaxes.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from skimage.measure import find_contours

from .ch_flux_diagnostics import evolve_f_diagnostic
from .model import Sink, evolve_f
from .signed_curvature import signed_curvature_at
from .tj_force import _local_gradient, _sample_bilinear
from .tj_subgrid import compute_subgrid_contact


@dataclass
class GChProbe:
    resolved: bool = False
    L_contact_before: float = math.nan
    g_ch: dict = field(default_factory=dict)  # {dt_frac: g_CH value, 1/s}


def g_ch_probe(f, e1, e2, e3, s, p, dt_fracs=(1.0, 0.5, 0.25)):
    """Instantaneous CH-only contact tendency: freeze eta, no Ostwald, no
    mass-preserving projection -- apply ONE raw evolve_f step at each of
    several dt fractions of p.dt (a fresh Params copy via
    dataclasses.replace, production p is never mutated), measure
    (L_contact_TJ_sub_after - L_contact_TJ_sub_before)/dt_i. Multiple
    fractions let the sign be checked for timestep-artifact sensitivity
    without changing the production p.dt anywhere."""
    sub_before = compute_subgrid_contact(f, e1, e2, p)
    out = GChProbe(resolved=sub_before.resolved,
                    L_contact_before=sub_before.L_contact_TJ_sub if sub_before.resolved else math.nan)
    if not sub_before.resolved:
        return out
    for frac in dt_fracs:
        p_i = p if frac == 1.0 else dataclasses.replace(p, dt=p.dt * frac)
        f_after = evolve_f(f, e1, e2, e3, s, Sink(), p_i)
        sub_after = compute_subgrid_contact(f_after, e1, e2, p)
        if sub_after.resolved:
            out.g_ch[frac] = (sub_after.L_contact_TJ_sub - sub_before.L_contact_TJ_sub) / p_i.dt
        else:
            out.g_ch[frac] = math.nan
    return out


def neck_region_mask(p, tj_top_xy, tj_bottom_xy, radius_widths=2.0):
    """Boolean mask (uncentered (index+1)*dx convention, matching
    tj_force/tj_subgrid) of a control region around the neck, sized from
    the instantaneous TJ separation and interface width -- not tuned."""
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    mid = ((tj_top_xy[0] + tj_bottom_xy[0]) / 2.0, (tj_top_xy[1] + tj_bottom_xy[1]) / 2.0)
    sep = math.hypot(tj_top_xy[0] - tj_bottom_xy[0], tj_top_xy[1] - tj_bottom_xy[1])
    radius = max(radius_widths * p.interface_width, 0.6 * sep)
    d = np.hypot(X - mid[0], Y - mid[1])
    return d <= radius


def neck_ch_mass_balance(f_before, f_after, mask, p):
    """Net change in local total-f mass (m^2 in physical presets) within
    the neck control region due to a CH-only update."""
    return float(((f_after - f_before) * mask).sum()) * p.dx * p.dx


def trace_branch_profile(f, mu_field, Jx, Jy, p, tj_xy, branch_dir, max_arclength, n_samples=60):
    """Walk the f=0.5 contour from tj_xy in branch_dir (a unit vector, e.g.
    tj_force's v_s1/v_s2), sampling curvature/mu/J_tangent/J_normal as
    functions of arclength s. Diagnostic-only greedy nearest-neighbor
    contour walk (the free surface away from a TJ is expected to be a
    simple, non-self-intersecting curve, so this is adequate here without
    needing a general polyline-graph solver). Returns None if too few
    contour points are found near tj_xy in that direction."""
    pts_list = []
    for rc in find_contours(f, 0.5):
        pts_list.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
    if not pts_list:
        return None
    all_pts = np.vstack(pts_list)
    tj = np.asarray(tj_xy, dtype=float)
    d_ = np.asarray(branch_dir, dtype=float)
    rel = all_pts - tj
    dist = np.linalg.norm(rel, axis=1)
    proj = rel @ d_
    cand = all_pts[(dist < max_arclength * 1.5) & (proj > -2 * p.interface_width)]
    if len(cand) < 5:
        return None

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
    if len(path) < 5:
        return None
    path = np.array(path)
    s_cum = np.array(s_cum)

    s_target = np.linspace(0.0, min(total_s, max_arclength), n_samples)
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
    kappa_s = np.array([signed_curvature_at(f, (xi, yi), p) for xi, yi in zip(x_r, y_r)])
    theta_s = np.arctan2(ty, tx)
    dJ_tangent_ds = np.gradient(J_tangent, s_target)

    return dict(s=s_target.tolist(), x=x_r.tolist(), y=y_r.tolist(), mu=mu_s.tolist(),
                J_tangent=J_tangent.tolist(), J_normal=J_normal.tolist(), kappa=kappa_s.tolist(),
                theta=theta_s.tolist(), dJ_tangent_ds=dJ_tangent_ds.tolist())
