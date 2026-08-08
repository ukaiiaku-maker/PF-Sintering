import math

import numpy as np

from pf_sintering.discrete_tangentiality import cell_tangentiality, face_tangentiality, tangentiality_summary
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import (
    m_s_ref,
    surface_flux,
    surface_mobility_tensor,
)

BC_X, BC_Y = "reflecting", "periodic"


def _circle_state(dx=2e-9, W=20e-9, R=100e-9, Nx=200, Ny=200):
    p = build_params(ModelConfig(
        preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W, use_aniso_surface=False, surface_mobility_scale=0.3,
    ))
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    cx, cy = x.mean(), y.mean()
    r = np.hypot(X - cx, Y - cy)
    theta = np.arctan2(Y - cy, X - cx)
    f = 0.5 * (1 - np.tanh((r - R) / W))
    mu = 0.3 * np.sin(3 * theta)
    return p, f, mu


def test_cell_tangentiality_is_near_roundoff_by_construction():
    # J.n_f at cell centers uses the SAME n_f the tangential projector P_t
    # is built from, so J.n_f = 0 exactly for a unit normal (P_t . v is
    # always orthogonal to n); the only nonzero residual comes from
    # eps_n's regularization keeping |n|<1 -- must stay tiny everywhere,
    # not grow near high-curvature regions.
    p, f, mu = _circle_state()
    M_s = m_s_ref(p.M_f, p.interface_width)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y,
                                             eps_n=1e-6 / p.interface_width)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)
    out = cell_tangentiality(f, Jx, Jy, p, BC_X, BC_Y)
    valid = np.isfinite(out["ratio"]) & (out["J_mag"] > 1e-6 * np.nanmax(out["J_mag"]))
    assert np.max(out["ratio"][valid]) < 1e-4


def test_face_tangentiality_converges_with_dx_on_smooth_interface():
    # A gently-curved sinusoidal interface (the geometry class actually
    # relevant to this milestone's substrate) shows cleanly DECREASING
    # face-normal error as dx shrinks (a standard, converging
    # discretization error), not a persistent or growing one. (A tightly
    # curved circle is a harder, direction-dependent case -- see
    # MILESTONE_13C's Section 6 for that more nuanced finding; this test
    # targets the well-behaved case.)
    rms_by_dx = {}
    for dx in (5e-9, 2.5e-9, 1.25e-9):
        wavelength = 200e-9
        amplitude = 20e-9
        W = 20e-9
        Ny = max(80, round(wavelength / dx))
        Nx = max(80, round(3 * wavelength / dx))
        p = build_params(ModelConfig(
            preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
            contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
            interface_width_override=W, use_aniso_surface=False, surface_mobility_scale=0.3,
        ))
        x = (np.arange(1, Nx + 1)) * dx
        y = (np.arange(1, Ny + 1)) * dx
        X, Y = np.meshgrid(x, y)
        xc = x.mean()
        h = xc + amplitude * np.cos(2 * math.pi * Y / wavelength)
        f = 0.5 * (1 - np.tanh((X - h) / W))
        mu = 0.3 * np.cos(2 * math.pi * Y / wavelength)
        M_s = m_s_ref(p.M_f, W)
        Mxx, Mxy, Myy = surface_mobility_tensor(f, dx, W, M_s, BC_X, BC_Y, eps_n=1e-6 / W)
        Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, dx, BC_X, BC_Y)
        out = face_tangentiality(f, Jx, Jy, p, BC_X, BC_Y)
        valid_y = np.isfinite(out["ratio_y"]) & (out["mag_y"] > 1e-3 * np.nanmax(out["mag_y"]))
        rms_by_dx[dx] = float(np.sqrt(np.mean(out["ratio_y"][valid_y] ** 2)))

    dxs = sorted(rms_by_dx)  # ascending: dxs[0]=finest dx, dxs[-1]=coarsest dx
    assert rms_by_dx[dxs[0]] < rms_by_dx[dxs[-1]]  # finest dx has smaller error than coarsest


def test_tangentiality_summary_regions_resolve():
    p, f, mu = _circle_state()
    M_s = m_s_ref(p.M_f, p.interface_width)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y,
                                             eps_n=1e-6 / p.interface_width)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    tj_probe = [(x.mean() + 100e-9, y.mean())]
    summ = tangentiality_summary(f, Jx, Jy, p, BC_X, BC_Y, tj_probe)
    for key in ("cell", "face_x", "face_y"):
        assert summ[key]["global_"]["n"] > 0
        assert math.isfinite(summ[key]["global_"]["rms"])
