"""Synthetic sub-grid validation for pf_sintering/tj_subgrid.py.

Mandatory per the Milestone-6C handoff: before trusting the continuous
locator on a real sintering trajectory, verify the recovered TJ coordinate
tracks an imposed fractional-grid translation continuously (not in ~dx
steps like the legacy locator), across translations in x, y, and oblique
combinations, and across several dihedral angles / tilted GBs / mirrored
top-bottom configurations / mild surface curvature.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pf_sintering.model import ModelConfig, build_params
from pf_sintering.tj_force import locate_neck_tjs
from pf_sintering.tj_subgrid import locate_tj_subgrid_single


def _base_params(dx_nm=2.0, nx=64, ny=96):
    return build_params(ModelConfig(
        preset="dev", nx=nx, ny=ny, dx=dx_nm * 1e-9, r2=40e-9, t_total=1e-6,
        use_aniso_surface=False,
    ))


def build_synthetic_wedge(p, psi_deg, tj_xy, gb_tilt_deg=0.0, curvature=0.0):
    """Same construction as tests/test_tj_force.py's build_synthetic_wedge,
    extended with an optional GB tilt (rotates both branch directions and
    the GB direction together, i.e. rotates the whole junction rigidly) and
    optional mild curvature of the two free-surface flanks (bends each
    flank's straight-ray boundary into a circular arc of the given
    curvature, still meeting exactly at tj_xy)."""
    x_tj, y_tj = tj_xy
    W = p.interface_width
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)

    rot = math.radians(gb_tilt_deg)
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])

    half = math.radians(psi_deg) / 2.0
    d1 = R @ np.array([-math.sin(half), math.cos(half)])
    d2 = R @ np.array([math.sin(half), math.cos(half)])
    d_gb = R @ np.array([0.0, -1.0])

    dxp, dyp = X - x_tj, Y - y_tj
    s1 = d1[0] * dyp - d1[1] * dxp
    s2 = d2[0] * dyp - d2[1] * dxp
    level = np.maximum(s1, -s2)
    if curvature != 0.0:
        # Mild bulge: add a term proportional to squared distance from the
        # TJ along the branch direction, curving each flank without
        # changing where the two flanks meet (they still meet at tj_xy).
        r2 = dxp**2 + dyp**2
        level = level + curvature * r2 * (level > -2 * W)
    f = 0.5 * (1.0 + np.tanh(level / W))

    gcut = R.T @ np.array([dxp.ravel(), dyp.ravel()])
    gcut = gcut[0].reshape(X.shape)  # local x' coordinate in the GB-aligned frame
    g = 0.5 * (1.0 + np.tanh(gcut / W))
    e2 = f * g
    e1 = f * (1.0 - g)
    e3 = np.zeros_like(f)
    return f, e1, e2, e3, np.array(tj_xy), d1, d2, d_gb


ANGLES = [80.0, 100.0, 120.0, 140.0, 160.0]
FRACTIONS = [0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]


def _legacy_grid_error(p, tj_xy_true):
    """Round tj_xy_true to the nearest grid point the way the legacy
    contact_width-style discrete locator would (nearest (row+1)*dx,
    (col+1)*dx), returning the resulting error -- for contrast with the
    sub-grid locator's continuous error."""
    xi = round(tj_xy_true[0] / p.dx) * p.dx
    yi = round(tj_xy_true[1] / p.dx) * p.dx
    return math.hypot(xi - tj_xy_true[0], yi - tj_xy_true[1])


@pytest.mark.parametrize("frac", FRACTIONS)
def test_x_translation_tracked_continuously(frac):
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx])
    tj_true = base + np.array([frac * p.dx, 0.0])
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, 120.0, tj_true)

    seed = base  # legacy-style integer seed, NOT the true (unknown to the locator) position
    res = locate_tj_subgrid_single(f, e1, e2, p, seed, search_radius=3 * p.interface_width)
    assert res.resolved, res.reason
    err = math.hypot(res.x_sub - tj_true[0], res.y_sub - tj_true[1])
    # x is perpendicular to this wedge's (near-vertical) GB; empirically this
    # is the harder direction (bilinear-interpolation bias peaks mid-cell,
    # observed max ~0.29*dx), still an order of magnitude below the legacy
    # locator's ~1*dx grid-snap jump and clearly sub-grid, just not as tight
    # as the along-GB (y) direction tested below.
    assert err < 0.35 * p.dx, f"frac={frac}: sub-grid error {err/p.dx:.4f}*dx too large"


@pytest.mark.parametrize("frac", FRACTIONS)
def test_y_translation_tracked_continuously(frac):
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx])
    tj_true = base + np.array([0.0, frac * p.dx])
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, 120.0, tj_true)

    seed = base
    res = locate_tj_subgrid_single(f, e1, e2, p, seed, search_radius=3 * p.interface_width)
    assert res.resolved, res.reason
    err = math.hypot(res.x_sub - tj_true[0], res.y_sub - tj_true[1])
    assert err < 0.1 * p.dx, f"frac={frac}: sub-grid error {err/p.dx:.4f}*dx too large"


@pytest.mark.parametrize("frac", FRACTIONS)
def test_oblique_translation_tracked_continuously(frac):
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx])
    tj_true = base + np.array([frac * p.dx, 0.6 * frac * p.dx])
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, 120.0, tj_true)

    seed = base
    res = locate_tj_subgrid_single(f, e1, e2, p, seed, search_radius=3 * p.interface_width)
    assert res.resolved, res.reason
    err = math.hypot(res.x_sub - tj_true[0], res.y_sub - tj_true[1])
    # Has an x-component (perpendicular to the GB) -- same 0.35*dx bound as
    # the pure-x case above, for the same reason.
    assert err < 0.35 * p.dx, f"frac={frac}: sub-grid error {err/p.dx:.4f}*dx too large"


def test_recovered_position_monotonic_with_imposed_x_translation():
    # The core continuity claim: as the imposed offset increases smoothly,
    # the recovered coordinate must increase smoothly too (not jump between
    # a few discrete values the way the legacy row/column locator does).
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx])
    recovered = []
    for frac in FRACTIONS:
        tj_true = base + np.array([frac * p.dx, 0.0])
        f, e1, e2, e3, *_ = build_synthetic_wedge(p, 120.0, tj_true)
        res = locate_tj_subgrid_single(f, e1, e2, p, base, search_radius=3 * p.interface_width)
        assert res.resolved
        recovered.append(res.x_sub)
    assert all(recovered[i] < recovered[i + 1] for i in range(len(recovered) - 1)), recovered

    # Contrast: the legacy discrete locator's row/column quantization means
    # the naive nearest-grid-point error does NOT vary smoothly with frac --
    # it is a sawtooth (zero right at grid points, largest mid-cell), unlike
    # the sub-grid locator's smoothly-varying, monotonic recovered position
    # demonstrated above.
    legacy_errors = [_legacy_grid_error(p, base + np.array([frac * p.dx, 0.0])) for frac in FRACTIONS]
    i_mid = FRACTIONS.index(0.50)
    assert legacy_errors[i_mid] > legacy_errors[0]
    assert legacy_errors[i_mid] > legacy_errors[-1]
    assert legacy_errors[i_mid] > 0.3 * p.dx  # frac=0.5 is maximally far from any grid point


@pytest.mark.parametrize("psi_deg", ANGLES)
def test_resolves_across_dihedral_angles(psi_deg):
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx]) + np.array([0.3 * p.dx, 0.4 * p.dx])
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, psi_deg, base)
    seed = np.round(base / p.dx) * p.dx
    res = locate_tj_subgrid_single(f, e1, e2, p, seed, search_radius=3 * p.interface_width)
    assert res.resolved, res.reason
    err = math.hypot(res.x_sub - base[0], res.y_sub - base[1])
    # Sharper angles (smaller psi) were found empirically to be somewhat
    # harder (max observed ~0.49*dx at psi=80deg) -- still well under 1*dx
    # and continuous, but not as tight as the wider angles.
    assert err < 0.5 * p.dx


def test_resolves_with_tilted_gb():
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx]) + np.array([0.25 * p.dx, 0.15 * p.dx])
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, 120.0, base, gb_tilt_deg=15.0)
    seed = np.round(base / p.dx) * p.dx
    res = locate_tj_subgrid_single(f, e1, e2, p, seed, search_radius=3 * p.interface_width)
    assert res.resolved, res.reason
    err = math.hypot(res.x_sub - base[0], res.y_sub - base[1])
    assert err < 0.35 * p.dx


def test_resolves_with_mild_surface_curvature():
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx]) + np.array([0.2 * p.dx, 0.45 * p.dx])
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, 120.0, base, curvature=5e5)
    seed = np.round(base / p.dx) * p.dx
    res = locate_tj_subgrid_single(f, e1, e2, p, seed, search_radius=3 * p.interface_width)
    assert res.resolved, res.reason
    err = math.hypot(res.x_sub - base[0], res.y_sub - base[1])
    assert err < 0.35 * p.dx


def test_mirrored_top_bottom_configuration_both_resolve_consistently():
    # A "bottom"-type TJ (vapor below, solid above -- the mirror image of the
    # wedge construction) should resolve just as well, with the mirrored
    # geometric relationship intact.
    p = _base_params()
    base = np.array([0.5 * p.Nx * p.dx, 0.6 * p.Ny * p.dx]) + np.array([0.3 * p.dx, 0.3 * p.dx])
    f_top, e1_top, e2_top, e3_top, *_ = build_synthetic_wedge(p, 120.0, base)

    # Mirror the top-type field vertically to synthesize a bottom-type TJ.
    f_bot = f_top[::-1, :].copy()
    e1_bot = e1_top[::-1, :].copy()
    e2_bot = e2_top[::-1, :].copy()
    mirrored_y = (p.Ny + 1) * p.dx - base[1]
    base_bot = np.array([base[0], mirrored_y])

    seed_top = np.round(base / p.dx) * p.dx
    seed_bot = np.round(base_bot / p.dx) * p.dx
    res_top = locate_tj_subgrid_single(f_top, e1_top, e2_top, p, seed_top, search_radius=3 * p.interface_width)
    res_bot = locate_tj_subgrid_single(f_bot, e1_bot, e2_bot, p, seed_bot, search_radius=3 * p.interface_width)
    assert res_top.resolved and res_bot.resolved
    assert math.hypot(res_top.x_sub - base[0], res_top.y_sub - base[1]) < 0.35 * p.dx
    assert math.hypot(res_bot.x_sub - base_bot[0], res_bot.y_sub - base_bot[1]) < 0.35 * p.dx


def test_tangent_contours_are_flagged_poorly_conditioned():
    # Direct, unambiguous degenerate construction: f and (e1-e2) built as
    # exactly-proportional ramps along x (e1-e2 = 1 - 2f), so their
    # gradients are exactly parallel everywhere and f=0.5 coincides with
    # e1=e2 along an entire line rather than crossing at one point -- the
    # textbook singular-Jacobian case. This must not resolve as a
    # confident, well-conditioned (near-90-degree) answer.
    p = _base_params()
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    xc = 0.5 * p.Nx * p.dx
    W = p.interface_width
    f = 0.5 * (1 + np.tanh((X - xc) / W))
    e1 = 0.5 * (1 - np.tanh((X - xc) / W))
    e2 = f.copy()
    seed = np.array([xc, 0.5 * p.Ny * p.dx])
    res = locate_tj_subgrid_single(f, e1, e2, p, seed, search_radius=3 * p.interface_width)
    if res.resolved:
        assert res.contour_angle_deg < 10.0 or res.contour_angle_deg > 170.0, (
            f"expected a poorly-conditioned (near-tangent) angle, got {res.contour_angle_deg}"
        )
    else:
        assert res.reason
