import math

import numpy as np

from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.curvature_extraction import window_curvature
from pf_sintering.model import ModelConfig, Sink, build_params


def _flat_grid(dx=1e-9, W=20e-9, Nx=400, Ny=200):
    p = build_params(ModelConfig(
        preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W, use_aniso_surface=False,
    ))
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    return p, x, y


def test_flat_interface_kappa_A_is_zero():
    # A planar interface is the exact 1D equilibrium profile: mu vanishes
    # identically (both bulk and gradient terms cancel), so kappa_A must be
    # exactly zero regardless of window position.
    p, x, y = _flat_grid()
    X, Y = np.meshgrid(x, y)
    xc = x[len(x) // 2]
    f = 0.5 * (1 - np.tanh((X - xc) / p.interface_width))
    e1 = e2 = e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    tj_xy = (xc, y.mean())
    wc = window_curvature(f, e1, e2, e3, s, p, tj_xy, np.array([0.0, 1.0]), 0.0, 3 * p.interface_width, mu_field=mu)
    assert wc.resolved
    assert abs(wc.kappa_A) < 1e-6 / p.interface_width  # zero to numerical roundoff


def test_circle_curvature_methods_A_and_B_match_analytic_kappa():
    # Method A (mu/(1.5*gamma_s), see module docstring for the 1.5 factor's
    # derivation) and Method B (geometric Kasa fit) must both recover the
    # known 1/R curvature of a synthetic circular interface, independently.
    p, x, y = _flat_grid(dx=1e-9, W=20e-9, Nx=400, Ny=200)
    X, Y = np.meshgrid(x, y)
    e1 = e2 = e3 = np.zeros_like(X)
    s = Sink(threshold=math.inf)
    xc = x[len(x) // 2]

    for R in (300e-9, 500e-9, 800e-9):
        cy0 = -R + y.mean()
        r = np.hypot(X - xc, Y - cy0)
        f = 0.5 * (1 - np.tanh((r - R) / p.interface_width))
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        tj_xy = (xc, y.mean())
        wc = window_curvature(f, e1, e2, e3, s, p, tj_xy, np.array([1.0, 0.0]),
                               0.0, 2 * p.interface_width, mu_field=mu)
        assert wc.resolved
        kappa_exact = 1.0 / R
        assert math.isclose(wc.kappa_A, kappa_exact, rel_tol=0.01)
        assert math.isclose(wc.kappa_B, kappa_exact, rel_tol=0.01)


def test_window_curvature_unresolved_when_too_few_contour_points():
    p, x, y = _flat_grid()
    f = np.zeros((len(y), len(x)))  # no interface anywhere
    e1 = e2 = e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    wc = window_curvature(f, e1, e2, e3, s, p, (x.mean(), y.mean()), np.array([0.0, 1.0]),
                           0.0, 3 * p.interface_width)
    assert not wc.resolved
