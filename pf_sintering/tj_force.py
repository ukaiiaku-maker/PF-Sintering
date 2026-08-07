"""Direct interfacial capillary-vector (triple-junction force-balance) diagnostic.

This module implements the force-based construction requested to replace the
old practice of *imposing* the equilibrium dihedral angle (`psi_eq`) inside
the local stress calculation. It does not modify `model.compute_stress`,
`model.measure_dihedral`, or `model.curvature`; the legacy `sigma` pipeline is
left completely intact as a parallel diagnostic (see MILESTONE_1_2_REPORT.md
and PHYSICS_BACKGROUND.md Section 6 for why the legacy local-stress
construction must be preserved, not replaced).

Physical picture
-----------------
At a triple junction (TJ) where two free-surface branches and one solid-solid
GB branch meet, each interface exerts a capillary "pull" on the TJ directed
along its own outward tangent (the unit tangent pointing from the TJ into the
bulk of that interface), with magnitude/direction set by the interface's
Cahn-Hoffman capillary vector:

    xi = gamma(theta) * t + (dgamma/dtheta) * n

for the (possibly anisotropic) free surfaces, and the isotropic

    xi_gb = gamma_gb_eff * t_gb

for the solid-solid GB. At mechanical equilibrium these three vectors sum to
zero (Young-Herring balance):

    F_TJ = xi_s1 + xi_s2 + xi_gb  ->  0

Away from equilibrium, F_TJ != 0 is the local capillary driving force. `psi`
(the angle between the two free-surface tangents) is reported here purely as
an independent, geometry-derived diagnostic -- it is never substituted into
the force calculation.

Branch-direction extraction
----------------------------
`measure_dihedral` in model.py extracts branches by windowing raw
`skimage.measure.find_contours` points around an estimated TJ and splitting
them by the largest angular gap. In practice (see MILESTONE_1_2_REPORT.md,
finding J.1) this failed to resolve a measured angle on every real sintering
geometry tested, always falling back to `psi_eq`.

This module uses a different, more robust technique instead: sample the
relevant scalar field (`f` for the free surfaces, `e1 - e2` for the GB, the
latter masked to the solid interior) on a small circle centered at the TJ,
locate its zero/level crossings by sign change along the circle, and refine
each crossing to a true local tangent via a centered finite-difference
gradient at the crossing point (tangent = 90-degree rotation of the local
gradient, oriented away from the TJ). This needs no clustering heuristics and
degrades gracefully: if a circle radius fails to resolve to a clean crossing
count, that branch/TJ is reported as unresolved (`None`) without raising, but
resolution failure does not depend on the same brittle windowed-contour path
`measure_dihedral` uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

from .model import _aniso_gamma, _vapor_normal, effective_gamma, overlap_col


# ---------------------------------------------------------------------------
# Field sampling / level-crossing primitives
# ---------------------------------------------------------------------------

def _sample_bilinear(field: np.ndarray, xs, ys, p) -> np.ndarray:
    """Bilinear-sample `field` at physical (x, y) using the model's pixel-center
    convention x = (col_index + 1) * dx, y = (row_index + 1) * dx."""
    ci = np.asarray(xs) / p.dx - 1.0
    ri = np.asarray(ys) / p.dx - 1.0
    ci = np.clip(ci, 0.0, p.Nx - 1.0)
    ri = np.clip(ri, 0.0, p.Ny - 1.0)
    return map_coordinates(field, [ri, ci], order=1, mode="nearest")


def _local_gradient(field: np.ndarray, xy, p, h=None):
    if h is None:
        h = 0.5 * p.dx
    x, y = xy
    gx = (
        _sample_bilinear(field, [x + h], [y], p)[0]
        - _sample_bilinear(field, [x - h], [y], p)[0]
    ) / (2 * h)
    gy = (
        _sample_bilinear(field, [x], [y + h], p)[0]
        - _sample_bilinear(field, [x], [y - h], p)[0]
    ) / (2 * h)
    return np.array([gx, gy])


def _tangent_at(field: np.ndarray, crossing_xy, tj_xy, p) -> np.ndarray:
    """Local unit tangent to the `field` level set at `crossing_xy`, oriented
    away from `tj_xy` (i.e. pointing into the bulk of the branch)."""
    g = _local_gradient(field, crossing_xy, p)
    gn = float(np.linalg.norm(g))
    away = np.array(crossing_xy) - np.array(tj_xy)
    away_n = float(np.linalg.norm(away))
    if gn < 1e-30:
        if away_n < 1e-30:
            return np.array([1.0, 0.0])
        return away / away_n
    t = np.array([-g[1], g[0]]) / gn
    if np.dot(t, away) < 0:
        t = -t
    return t


def _circle_crossings(field, level, center_xy, radius, p, n_theta=1440, mask_field=None):
    """Angles (radians, in [0, 2*pi)) where `field` crosses `level` on a circle
    of `radius` about `center_xy`. If `mask_field` is given, both neighboring
    samples of a candidate crossing must have `mask_field > 0.5` (used to
    restrict the GB crossing search to the solid interior)."""
    thetas = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    xs = center_xy[0] + radius * np.cos(thetas)
    ys = center_xy[1] + radius * np.sin(thetas)
    vals = _sample_bilinear(field, xs, ys, p) - level
    valid = np.ones(n_theta, dtype=bool)
    if mask_field is not None:
        valid = _sample_bilinear(mask_field, xs, ys, p) > 0.5

    crossings = []
    for i in range(n_theta):
        j = (i + 1) % n_theta
        if not (valid[i] and valid[j]):
            continue
        vi, vj = vals[i], vals[j]
        if vi == 0.0:
            crossings.append(thetas[i])
            continue
        if vj == 0.0:
            # Handled as the `vi == 0.0` case when the loop reaches index j;
            # counting it again here would double the crossing.
            continue
        if np.sign(vi) != np.sign(vj):
            t = vi / (vi - vj)
            th_i = thetas[i]
            th_j = thetas[j] if j > i else thetas[j] + 2 * math.pi
            th = (th_i + t * (th_j - th_i)) % (2 * math.pi)
            crossings.append(th)
    return crossings


def _branch_directions(field, level, center_xy, radius, p, mask_field=None, expect=None):
    """Find level-crossing directions (unit tangents, refined via local
    gradient) on a circle around `center_xy`. Returns a list of unit vectors,
    or None if the resolved crossing count does not match `expect` (when given)."""
    angles = _circle_crossings(field, level, center_xy, radius, p, mask_field=mask_field)
    if expect is not None and len(angles) != expect:
        return None
    out = []
    for th in angles:
        xy = (center_xy[0] + radius * math.cos(th), center_xy[1] + radius * math.sin(th))
        out.append(_tangent_at(field, xy, center_xy, p))
    return out


# ---------------------------------------------------------------------------
# TJ location (reuses the same neck-column / solid-extent convention as
# model.compute_stress / model.measure_dihedral, for consistency)
# ---------------------------------------------------------------------------

def locate_neck_tjs(f, e1, e2, p, search_radius=12):
    """Locate the neck column and its top/bottom TJ grid points.

    `overlap_col` returns the e1*e2-weighted centroid column, which for a
    small initial overlap can sit measurably away (found empirically: 1-2
    grid cells for the Milestone-1/2 candidate geometry) from the column
    where the combined solid actually has a *finite* (non-domain-spanning)
    y-extent -- i.e. away from the visible neck. `model.measure_dihedral`
    uses the centroid column directly and silently fails
    (`solid[0]==0 or solid[-1]==Ny-1`) whenever that offset lands it back in
    the semi-infinite substrate bulk; this was the dominant cause of the
    Milestone-1/2 "psi never measured" finding (MILESTONE_1_2_REPORT.md,
    J.1). Rather than trusting the centroid column exactly, search a small
    window around it for the column with the *narrowest* finite, resolved
    (non-domain-spanning) solid extent -- i.e. the true neck.
    """
    cr = p.Ny // 2
    ov, col = overlap_col(e1[cr] * e2[cr])
    if ov < 1e-3 or not math.isfinite(col):
        return None
    nc0 = int(np.clip(round(col) - 1, 0, p.Nx - 1))
    best = None
    for dc in range(-search_radius, search_radius + 1):
        nc = nc0 + dc
        if not (0 <= nc < p.Nx):
            continue
        solid = np.flatnonzero(f[:, nc] > 0.5)
        if len(solid) < 2 or solid[0] == 0 or solid[-1] == p.Ny - 1:
            continue
        extent = solid[-1] - solid[0]
        if best is None or extent < best[0]:
            best = (extent, nc, solid)
    if best is None:
        return None
    _, nc, solid = best
    x_col = (nc + 1) * p.dx
    neck_height = (solid[-1] - solid[0]) * p.dx
    return dict(
        col=col, nc=nc, x_col=x_col, neck_height=neck_height,
        tj_top=np.array([x_col, (solid[-1] + 1) * p.dx]),
        tj_bottom=np.array([x_col, (solid[0] + 1) * p.dx]),
    )


# ---------------------------------------------------------------------------
# Capillary vectors
# ---------------------------------------------------------------------------

def cahn_hoffman_vector(v, f, tj, p):
    """Interfacial capillary vector for a free-surface branch with outward
    tangent `v`, at TJ location `tj`. Isotropic gamma_s if anisotropy is off."""
    n1 = np.array([-v[1], v[0]])
    n2 = -n1
    n = _vapor_normal(f, tj, n1, n2, p)
    if p.use_aniso_surface:
        th = math.atan2(n[1], n[0])
        th0 = p.theta_grain[0] if v[0] < 0 else p.theta_grain[1]
        gam, gp = _aniso_gamma(th, th0, p)
    else:
        gam, gp = p.gamma_s, 0.0
    return gam * v + gp * n, n


def gb_vector(v_gb, s, p):
    """Isotropic GB capillary vector (Section 4: GB can remain isotropic)."""
    return effective_gamma(s, p) * v_gb


@dataclass
class TJForce:
    tj_xy: np.ndarray
    resolved: bool = False
    reason: str = ""
    v_s1: np.ndarray = field(default_factory=lambda: np.full(2, math.nan))
    v_s2: np.ndarray = field(default_factory=lambda: np.full(2, math.nan))
    v_gb: np.ndarray = field(default_factory=lambda: np.full(2, math.nan))
    xi_s1: np.ndarray = field(default_factory=lambda: np.full(2, math.nan))
    xi_s2: np.ndarray = field(default_factory=lambda: np.full(2, math.nan))
    xi_gb: np.ndarray = field(default_factory=lambda: np.full(2, math.nan))
    F_TJ: np.ndarray = field(default_factory=lambda: np.full(2, math.nan))
    F_TJ_x: float = math.nan
    F_TJ_mag: float = math.nan
    F_gb_normal: float = math.nan
    F_gb_parallel: float = math.nan
    psi_deg: float = math.nan


def _radii_for_tj(neck_height, interface_width):
    r_surf = min(2.5 * interface_width, 0.4 * max(neck_height, 1e-30))
    r_surf = max(r_surf, interface_width)
    r_gb = min(0.7 * r_surf, 0.35 * max(neck_height, 1e-30))
    r_gb = max(r_gb, 0.5 * interface_width)
    return r_surf, r_gb


def compute_tj_force(field_f, field_e1, field_e2, tj_xy, s, p) -> TJForce:
    out = TJForce(tj_xy=np.asarray(tj_xy, dtype=float))
    tjs = locate_neck_tjs(field_f, field_e1, field_e2, p)
    neck_height = tjs["neck_height"] if tjs else 8 * p.interface_width
    r_surf, r_gb = _radii_for_tj(neck_height, p.interface_width)

    surf_dirs = _branch_directions(field_f, 0.5, tj_xy, r_surf, p, expect=2)
    if surf_dirs is None:
        out.reason = "free-surface branches not resolved (!= 2 crossings)"
        return out

    d = field_e1 - field_e2
    gb_dirs = _branch_directions(d, 0.0, tj_xy, r_gb, p, mask_field=field_f, expect=1)
    if gb_dirs is None:
        out.reason = "GB branch not resolved (!= 1 crossing)"
        return out

    v_s1, v_s2 = surf_dirs
    v_gb = gb_dirs[0]
    xi_s1, _ = cahn_hoffman_vector(v_s1, field_f, tj_xy, p)
    xi_s2, _ = cahn_hoffman_vector(v_s2, field_f, tj_xy, p)
    xi_gb = gb_vector(v_gb, s, p)
    F_TJ = xi_s1 + xi_s2 + xi_gb

    n_gb = np.array([-v_gb[1], v_gb[0]])
    if np.dot(n_gb, np.array([1.0, 0.0])) < 0:
        n_gb = -n_gb

    psi = math.degrees(math.acos(float(np.clip(np.dot(v_s1, v_s2), -1.0, 1.0))))

    out.resolved = True
    out.v_s1, out.v_s2, out.v_gb = v_s1, v_s2, v_gb
    out.xi_s1, out.xi_s2, out.xi_gb = xi_s1, xi_s2, xi_gb
    out.F_TJ = F_TJ
    out.F_TJ_x = float(F_TJ[0])
    out.F_TJ_mag = float(np.linalg.norm(F_TJ))
    out.F_gb_normal = float(np.dot(F_TJ, n_gb))
    out.F_gb_parallel = float(np.dot(F_TJ, v_gb))
    out.psi_deg = psi
    return out


@dataclass
class TJForceReport:
    resolved: bool
    top: TJForce | None = None
    bottom: TJForce | None = None
    F_drive: float = math.nan  # sum of F_TJ_x over resolved TJs
    n_resolved: int = 0


def compute_neck_tj_forces(f, e1, e2, e3, s, p) -> TJForceReport:
    tjs = locate_neck_tjs(f, e1, e2, p)
    if tjs is None:
        return TJForceReport(resolved=False)
    top = compute_tj_force(f, e1, e2, tjs["tj_top"], s, p)
    bottom = compute_tj_force(f, e1, e2, tjs["tj_bottom"], s, p)
    resolved = [t for t in (top, bottom) if t.resolved]
    F_drive = float(sum(t.F_TJ_x for t in resolved)) if resolved else math.nan
    return TJForceReport(
        resolved=len(resolved) > 0, top=top, bottom=bottom,
        F_drive=F_drive, n_resolved=len(resolved),
    )
