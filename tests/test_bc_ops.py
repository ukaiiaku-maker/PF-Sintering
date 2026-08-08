import math

import numpy as np

from pf_sintering.bc_ops import (
    div_bc,
    face_flux_x,
    face_flux_y,
    flux_divergence,
    grad_bc,
    lap9_bc,
)
from pf_sintering.model import div as legacy_div
from pf_sintering.model import grad as legacy_grad
from pf_sintering.model import lap9 as legacy_lap9


def _rand(shape=(40, 60), seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape)


def test_lap9_bc_matches_legacy_at_default_bc():
    a = _rand()
    dx = 5e-9
    assert np.allclose(lap9_bc(a, dx, bc_x="periodic", bc_y="reflecting"), legacy_lap9(a, dx))


def test_grad_bc_periodic_axis_matches_legacy_gx():
    a = _rand()
    dx = 5e-9
    gx_new, gy_new = grad_bc(a, dx, bc_x="periodic", bc_y="reflecting")
    gx_legacy, gy_legacy = legacy_grad(a, dx)
    assert np.allclose(gx_new, gx_legacy)  # periodic axis: identical central difference


def test_constant_field_has_zero_gradient_any_bc():
    a = np.full((30, 40), 3.7)
    dx = 5e-9
    for bc_x in ("periodic", "reflecting"):
        for bc_y in ("periodic", "reflecting"):
            gx, gy = grad_bc(a, dx, bc_x=bc_x, bc_y=bc_y)
            assert np.allclose(gx, 0.0)
            assert np.allclose(gy, 0.0)


def test_periodic_sinusoid_derivative_is_correct():
    Ny, Nx = 8, 200
    dx = 1e-9
    wavelength = Nx * dx
    x = (np.arange(1, Nx + 1)) * dx
    a = np.tile(np.cos(2 * math.pi * x / wavelength), (Ny, 1))
    gx, gy = grad_bc(a, dx, bc_x="periodic", bc_y="reflecting")
    expected = np.tile(-(2 * math.pi / wavelength) * np.sin(2 * math.pi * x / wavelength), (Ny, 1))
    assert np.max(np.abs(gx - expected)) < 1e-3 * np.max(np.abs(expected))


def test_grad_div_adjoint_holds_for_periodic_axis():
    # The periodic axis IS exactly self-adjoint (a roll-based central
    # difference always is); documents that the known non-adjointness
    # (module docstring) is specific to the reflecting-axis ghost cells.
    a = _rand(seed=1)
    v = _rand(seed=2)
    dx = 5e-9
    gx, _ = grad_bc(a, dx, bc_x="periodic", bc_y="periodic")
    d = div_bc(v, np.zeros_like(v), dx, bc_x="periodic", bc_y="periodic")
    lhs = np.sum(gx * v)
    rhs = -np.sum(a * d)
    assert math.isclose(lhs, rhs, rel_tol=1e-10)


def test_grad_div_not_exactly_adjoint_at_reflecting_boundary_documented():
    # Documents the known limitation (module docstring): the simple
    # ghost-cell grad_bc/div_bc pair is NOT an exact discrete adjoint under
    # a reflecting BC, unlike flux_divergence's telescoping-sum guarantee.
    # This is why grad_bc/div_bc are used only for non-conservative
    # diagnostics, never for the mass-conservative update itself.
    a = _rand(seed=1)
    vx = _rand(seed=2)
    vy = _rand(seed=3)
    dx = 5e-9
    gx, gy = grad_bc(a, dx, bc_x="reflecting", bc_y="reflecting")
    lhs = np.sum(gx * vx + gy * vy)
    rhs = -np.sum(a * div_bc(vx, vy, dx, bc_x="reflecting", bc_y="reflecting"))
    assert not math.isclose(lhs, rhs, rel_tol=1e-3)


def test_flux_divergence_sums_to_zero_periodic_periodic():
    f = 0.5 + 0.1 * _rand(seed=4)
    M = f * f * (1 - f) ** 2
    mu = _rand(seed=5)
    dx = 5e-9
    Jx = face_flux_x(M, mu, dx)
    Jy = face_flux_y(M, mu, dx)
    d = flux_divergence(Jx, Jy, dx, bc_x="periodic", bc_y="periodic")
    assert abs(float(np.sum(d))) < 1e-6 * np.max(np.abs(d))


def test_flux_divergence_sums_to_zero_reflecting_periodic():
    f = 0.5 + 0.1 * _rand(seed=6)
    M = f * f * (1 - f) ** 2
    mu = _rand(seed=7)
    dx = 5e-9
    Jx = face_flux_x(M, mu, dx)
    Jy = face_flux_y(M, mu, dx)
    d = flux_divergence(Jx, Jy, dx, bc_x="reflecting", bc_y="periodic")
    assert abs(float(np.sum(d))) < 1e-6 * np.max(np.abs(d))


def test_flux_divergence_matches_legacy_evolve_f_pattern():
    # model.evolve_f's own Jx/Jy/divergence construction, reproduced by
    # hand, must match flux_divergence(..., bc_x="periodic", bc_y="reflecting").
    f = 0.5 + 0.1 * _rand(seed=8)
    M = np.minimum((16 * f * f * (1 - f) ** 2) ** 2, 1.0)
    mu = _rand(seed=9)
    dx = 5e-9

    Mx = 0.5 * (M + np.roll(M, -1, 1))
    Jx = -Mx * (np.roll(mu, -1, 1) - mu) / dx
    Jy = np.zeros_like(f)
    My = 0.5 * (M[:-1] + M[1:])
    Jy[:-1] = -My * (mu[1:] - mu[:-1]) / dx
    Jyd = np.zeros_like(f)
    Jyd[1:] = Jy[:-1]
    legacy_total_div = ((Jx - np.roll(Jx, 1, 1)) + (Jy - Jyd)) / dx

    Jx_new = face_flux_x(M, mu, dx)
    Jy_new = face_flux_y(M, mu, dx)
    new_total_div = flux_divergence(Jx_new, Jy_new, dx, bc_x="periodic", bc_y="reflecting")
    assert np.allclose(legacy_total_div, new_total_div)


def test_reflecting_axis_zero_boundary_flux_conserves_mass_exactly():
    # sum(d) is a combinatorial telescoping-sum identity that must hold to
    # floating-point roundoff of the RAW (unamplified) flux values -- check
    # sum(efflux) directly (natural O(1) scale) rather than sum(d)=sum(efflux)/dx,
    # since dividing an already-roundoff-level sum by a tiny SI dx (~5e-9)
    # amplifies float64 noise by ~1/dx and produces a misleadingly large
    # *absolute* number despite the identity holding exactly in exact
    # arithmetic (verified: the raw efflux sums are ~1e-7, at the scale of
    # summing ~2400 O(1) terms to float64 precision, not a real violation).
    f = 0.5 + 0.1 * _rand(seed=10)
    M = f * f * (1 - f) ** 2
    mu = _rand(seed=11)
    dx = 5e-9
    Jx = face_flux_x(M, mu, dx)
    Jy = face_flux_y(M, mu, dx)
    from pf_sintering.bc_ops import _axis_efflux
    sum_x = float(np.sum(_axis_efflux(Jx, axis=1, bc="periodic")))
    sum_y = float(np.sum(_axis_efflux(Jy, axis=0, bc="reflecting")))
    scale = max(np.max(np.abs(Jx)), np.max(np.abs(Jy)))
    assert abs(sum_x) < 1e-8 * scale * f.size
    assert abs(sum_y) < 1e-8 * scale * f.size

    # end-to-end: a realistic (small, dx-appropriate) dt must conserve mass
    # to a tight *relative* tolerance -- the physically meaningful check.
    d = flux_divergence(Jx, Jy, dx, bc_x="periodic", bc_y="reflecting")
    total_before = float(f.sum()) * dx * dx
    dt = 1e-3 * dx  # dx-appropriate small step, not an arbitrary O(1) dt
    f_new = f - dt * d
    total_after = float(f_new.sum()) * dx * dx
    assert math.isclose(total_before, total_after, rel_tol=1e-6)
