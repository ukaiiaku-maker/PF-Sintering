import math

import numpy as np

from pf_sintering.constrained_eta import (
    constrained_variational_eta_update,
    f_weighted_ownership_volumes,
    structural_thermodynamic_force,
)
from pf_sintering.model import ModelConfig, build_params


def _state(seed=0):
    p = build_params(ModelConfig(
        preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=20e-9, t_total=1e-6, eta_mobility_scale=1.0,
    ))
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    f = np.clip(0.5 + 0.3 * gaussian_filter(rng.normal(size=(p.Ny, p.Nx)), sigma=3), 0.0, 1.0)
    w1 = np.clip(0.5 + 0.3 * gaussian_filter(rng.normal(size=(p.Ny, p.Nx)), sigma=3), 0.0, 1.0)
    e1 = f * w1
    e2 = f * (1 - w1)
    e3 = np.zeros_like(f)
    return p, f, e1, e2, e3


def test_zero_sum_constraint_preserved_by_variational_update_before_projection():
    # sum_i d(eta_i)/dt = 0 is an exact algebraic identity -- check the
    # variational deltas alone (before reproject) sum to (near) zero.
    p, f, e1, e2, e3 = _state()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, p)
    d1, d2 = diag["d1"], diag["d2"]
    assert np.max(np.abs(d1 + d2)) < 1e-6 * max(np.max(np.abs(d1)), np.max(np.abs(d2)))


def test_total_f_unaffected_by_eta_update():
    p, f, e1, e2, e3 = _state()
    f_before = f.copy()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, p)
    assert np.array_equal(f, f_before)  # eta update never touches f


def test_sum_eta_stays_close_to_f_after_projection():
    p, f, e1, e2, e3 = _state()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, p)
    residual = np.abs((e1n + e2n) - f)
    assert np.max(residual) < 0.05  # local projection keeps this small


def test_projection_correction_small_relative_to_variational_change():
    p, f, e1, e2, e3 = _state()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, p)
    # the projection should be a minor correction, not comparable to the
    # variational update itself (Section 15's explicit requirement)
    assert diag["projection_fraction"] < 0.5


def test_ownership_transfer_allowed_between_grains():
    # Construct a case where e1's own thermodynamic force clearly differs
    # from e2's (e1 rough/high-curvature, e2 smooth) -- ownership should
    # shift without changing total f.
    p = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                  contact_orientation="short_plane", initial_overlap=20e-9,
                                  t_total=1e-6, eta_mobility_scale=1.0))
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    f = np.ones((p.Ny, p.Nx)) * 0.9
    e1 = 0.45 + 0.05 * np.sin(2 * math.pi * Y / (y.max() / 4))  # rough (high curvature)
    e2 = f - e1  # smooth complement
    e3 = np.zeros_like(f)
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, p)
    assert np.max(np.abs(e1n - e1)) > 1e-9  # ownership actually moves
    assert diag["variational_change"] > 0
    assert math.isclose(float((e1n + e2n).sum()), float(f.sum()), rel_tol=1e-6)


def test_f_weighted_ownership_volumes_sum_to_v2_f_total():
    p, f, e1, e2, e3 = _state()
    V1, V2, V3 = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    V_f_total = float(f.sum()) * p.dx * p.dx
    assert math.isclose(V1 + V2 + V3, V_f_total, rel_tol=1e-6)


def test_structural_force_matches_evolve_eta_sign():
    from pf_sintering.model import lap9
    p, f, e1, e2, e3 = _state()
    g1 = structural_thermodynamic_force(e1, p.dx, p.k_eta, bc_x="periodic", bc_y="reflecting")
    expected = -p.k_eta * lap9(e1, p.dx)
    assert np.allclose(g1, expected)
