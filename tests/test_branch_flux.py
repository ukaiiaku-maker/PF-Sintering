import math

import numpy as np

from pf_sintering.branch_flux import (
    branch_cut_flux,
    branch_flux_balance_at_tj,
    cartesian_total_boundary_flux,
    disk_control_volume_mask,
)
from pf_sintering.model import ModelConfig, build_params


def _flat_grid(dx=1e-9, W=20e-9, Nx=300, Ny=200):
    p = build_params(ModelConfig(
        preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W, use_aniso_surface=False,
    ))
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    return p, x, y


def test_branch_cut_flux_matches_analytic_uniform_flux():
    # A straight interface with a known UNIFORM flux field: the cut
    # integral must exactly equal 2*halfwidth*flux_magnitude, for any
    # halfwidth (validates the trapezoid/bilinear-sampling machinery
    # independent of any curved-TJ complexity).
    p, x, y = _flat_grid()
    X, Y = np.meshgrid(x, y)
    xc = x[len(x) // 2]
    W = p.interface_width
    f = 0.5 * (1 - np.tanh((X - xc) / W))
    Jy = np.ones_like(f)
    Jx = np.zeros_like(f)
    tj_xy = (xc, y[len(y) // 2])
    branch_dir = np.array([0.0, 1.0])

    for hw in (0.5 * W, 1.0 * W, 2.0 * W):
        res = branch_cut_flux(f, Jx, Jy, p, tj_xy, branch_dir, 2 * W, [hw])
        assert math.isclose(res["Q_by_halfwidth"][hw], 2 * hw, rel_tol=1e-9)


def test_disk_control_volume_mask_closure_matches_cartesian_divergence():
    # A=B exact check on a synthetic curved field: sum(-div(J)) over any
    # mask must equal the boundary flux by construction (bc_ops.
    # flux_divergence's own telescoping identity) -- sanity-checks
    # cartesian_total_boundary_flux + disk_control_volume_mask together.
    p, x, y = _flat_grid(dx=2e-9, W=20e-9, Nx=150, Ny=100)
    X, Y = np.meshgrid(x, y)
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    Jx = gaussian_filter(rng.normal(size=X.shape), sigma=3)
    Jy = gaussian_filter(rng.normal(size=X.shape), sigma=3)
    center = (x[len(x) // 2], y[len(y) // 2])
    mask = disk_control_volume_mask(p, center, 10 * p.dx)

    from pf_sintering.bc_ops import flux_divergence
    from pf_sintering.surface_transport import face_average
    Jx_face = face_average(Jx, axis=1, bc="reflecting")
    Jy_face = face_average(Jy, axis=0, bc="periodic")
    neg_div = -flux_divergence(Jx_face, Jy_face, p.dx, bc_x="reflecting", bc_y="periodic")
    expected = float(np.sum(neg_div[mask])) * p.dx * p.dx

    got = cartesian_total_boundary_flux(Jx, Jy, mask, p, "reflecting", "periodic")
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_branch_cut_flux_none_when_branch_does_not_reach_cut():
    p, x, y = _flat_grid()
    f = np.zeros((len(y), len(x)))  # no interface anywhere
    Jx = np.zeros_like(f)
    Jy = np.zeros_like(f)
    res = branch_cut_flux(f, Jx, Jy, p, (x.mean(), y.mean()), np.array([1.0, 0.0]),
                           2 * p.interface_width, [p.interface_width])
    assert res is None


def test_branch_flux_balance_sign_convention_on_synthetic_v_shape():
    # Two straight branches emanating from a common TJ at a right angle,
    # each carrying a known uniform tangential flux either toward or away
    # from the TJ; Q_p/Q_s must report the correct sign per the module's
    # documented convention (Q_p positive = into the TJ, Q_s positive =
    # out of the TJ), verified against the raw, unambiguous construction.
    p, x, y = _flat_grid(dx=1e-9, W=20e-9, Nx=200, Ny=200)
    X, Y = np.meshgrid(x, y)
    xc, yc = x[100], y[100]
    W = p.interface_width
    # branch A: vertical interface at x=xc for y>yc (tangent +y, away from TJ)
    # branch B: horizontal interface at y=yc for x>xc (tangent +x, away from TJ)
    fA = 0.5 * (1 - np.tanh((X - xc) / W))
    fB = 0.5 * (1 - np.tanh((Y - yc) / W))
    f = np.maximum(fA, fB)
    # uniform flux fields matching each branch's own tangent direction
    Jx = np.where(Y > yc, 0.0, 1.0)   # along branch B (horizontal) only where relevant
    Jy = np.where(Y > yc, -2.0, 0.0)  # along branch A: flux toward the TJ (-y direction, since t_hat=+y)

    tj_xy = (xc, yc)
    v_p = np.array([0.0, 1.0])   # "particle" branch, tangent +y
    v_s = np.array([1.0, 0.0])   # "substrate" branch, tangent +x
    res = branch_flux_balance_at_tj(f, Jx, Jy, p, tj_xy, v_p, v_s, 2 * W, [W])
    assert res is not None
    # particle branch: J.t_hat = Jy = -2 (toward TJ) -> Q_p = -(-2)*2W = +4W (inflow, positive)
    assert res["Q_p"][W] > 0
    # substrate branch: J.t_hat = Jx = 1 (away from TJ) -> Q_s = +1*2W (outflow, positive)
    assert res["Q_s"][W] > 0
