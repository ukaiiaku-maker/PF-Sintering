import math

import numpy as np

from pf_sintering.surface_transport import (
    interface_localization_q,
    interface_normal,
    m_s_ref,
    q_normalization_numeric,
    surface_divergence_update,
    surface_mobility_tensor,
    tangential_projector,
)


def test_q_normalization_is_one_independent_of_dx():
    W = 20e-9
    for dx in (5e-9, 2.5e-9):
        val = q_normalization_numeric(W, dx)
        assert math.isclose(val, 1.0, rel_tol=1e-6)


def test_q_normalization_analytic_prefactor():
    W = 20e-9
    f0 = np.array([0.0, 0.5, 1.0])
    q = interface_localization_q(f0, W)
    assert q[0] == 0.0 and q[2] == 0.0  # zero in bulk (f=0 or f=1)
    assert q[1] == (12.0 / W) * 0.25 * 0.25


def test_m_s_ref_scales_linearly_with_m_f_and_w():
    assert math.isclose(m_s_ref(2.0, 1.0) / m_s_ref(1.0, 1.0), 2.0)
    assert math.isclose(m_s_ref(1.0, 2.0) / m_s_ref(1.0, 1.0), 2.0)


def test_tangential_projector_symmetric_and_psd():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * math.pi, size=50)
    nx, ny = np.cos(theta), np.sin(theta)
    Pxx, Pxy, Pyy = tangential_projector(nx, ny)
    # eigenvalues of [[Pxx,Pxy],[Pxy,Pyy]] at each point must be in {0,1}
    for i in range(len(nx)):
        M = np.array([[Pxx[i], Pxy[i]], [Pxy[i], Pyy[i]]])
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs >= -1e-12)
        assert np.allclose(sorted(eigs), [0.0, 1.0], atol=1e-10)


def test_tangential_projector_annihilates_normal():
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * math.pi, size=20)
    nx, ny = np.cos(theta), np.sin(theta)
    Pxx, Pxy, Pyy = tangential_projector(nx, ny)
    Pn_x = Pxx * nx + Pxy * ny
    Pn_y = Pxy * nx + Pyy * ny
    assert np.allclose(Pn_x, 0.0, atol=1e-12)
    assert np.allclose(Pn_y, 0.0, atol=1e-12)


def test_mobility_tensor_vanishes_in_bulk():
    W = 20e-9
    dx = 5e-9
    f = np.zeros((30, 30))
    f[:, 15:] = 1.0  # sharp step -> far from it, both f=0 and f=1 bulk
    Mxx, Mxy, Myy = surface_mobility_tensor(f, dx, W, M_s=1.0, bc_x="reflecting", bc_y="periodic", eps_n=1e-6 / W)
    assert np.allclose(Mxx[:, :5], 0.0, atol=1e-20)
    assert np.allclose(Myy[:, :5], 0.0, atol=1e-20)


def test_eps_n_sensitivity_negligible_over_reasonable_range():
    W = 20e-9
    dx = 5e-9
    x = (np.arange(1, 41)) * dx
    f = np.tile(0.5 * (1 + np.tanh((x - x.mean()) / W)), (20, 1))
    mu = np.tile(np.sin(x / (5 * W)), (20, 1)) * 1e7
    results = []
    for eps_n in (1e-8 / W, 1e-6 / W, 1e-4 / W):
        f_new, diag = surface_divergence_update(f, mu, dx, dt=1e-8, W=W, M_s=1e-20,
                                                  bc_x="reflecting", bc_y="periodic", eps_n=eps_n)
        results.append(f_new)
    # max relative difference across a 10000x range of eps_n choices,
    # measured against the field's own peak scale (avoids pointwise
    # hypersensitivity deep in the bulk where f_new itself is ~0)
    scale = np.max(np.abs(results[1]))
    assert np.max(np.abs(results[0] - results[1])) / scale < 1e-4
    assert np.max(np.abs(results[1] - results[2])) / scale < 1e-4


def test_planar_interface_uniform_mu_gives_zero_flux():
    W = 20e-9
    dx = 5e-9
    x = (np.arange(1, 61)) * dx
    f = np.tile(0.5 * (1 + np.tanh((x - x.mean()) / W)), (20, 1))
    mu = np.full_like(f, 1234.5)  # spatially uniform -> grad(mu)=0 everywhere
    f_new, diag = surface_divergence_update(f, mu, dx, dt=1e-6, W=W, M_s=1e-18,
                                             bc_x="reflecting", bc_y="periodic")
    assert np.allclose(diag["Jx"], 0.0, atol=1e-30)
    assert np.allclose(diag["Jy"], 0.0, atol=1e-30)
    assert np.allclose(f_new, f)


def test_mass_conservation_one_step():
    W = 20e-9
    dx = 5e-9
    x = (np.arange(1, 61)) * dx
    y = (np.arange(1, 41)) * dx
    X, Y = np.meshgrid(x, y)
    f = 0.5 * (1 + np.tanh((X - x.mean() + 0.1 * W * np.sin(2 * math.pi * Y / (y.max()))) / W))
    mu = np.sin(2 * math.pi * X / x.max()) * 1e7 + np.cos(2 * math.pi * Y / y.max()) * 1e7
    total_before = float(f.sum())
    f_new, diag = surface_divergence_update(f, mu, dx, dt=1e-9, W=W, M_s=1e-19,
                                             bc_x="reflecting", bc_y="periodic")
    total_after = float(f_new.sum())
    assert math.isclose(total_before, total_after, rel_tol=1e-9)


def test_mass_conservation_many_steps():
    # Gentle synthetic mu (fixed in space, not recomputed from f each step,
    # and small enough that f stays within/near [0,1] throughout) -- this
    # test checks the CONSERVATIVE UPDATE's mass bookkeeping over repeated
    # steps, not morphological stability of an arbitrary synthetic mu field
    # unrelated to any real double-well restoring force (a real mu_f,
    # tested separately via the Mullins/particle benchmarks, always keeps
    # f close to [0,1] because of its own bulk term).
    W = 20e-9
    dx = 5e-9
    x = (np.arange(1, 61)) * dx
    y = (np.arange(1, 41)) * dx
    X, Y = np.meshgrid(x, y)
    f = 0.5 * (1 + np.tanh((X - x.mean() + 0.1 * W * np.sin(2 * math.pi * Y / (y.max()))) / W))
    mu = np.sin(2 * math.pi * X / x.max()) * 1e6
    total0 = float(f.sum())
    for _ in range(20):
        f, _ = surface_divergence_update(f, mu, dx, dt=1e-10, W=W, M_s=1e-20,
                                          bc_x="reflecting", bc_y="periodic")
    assert np.all(np.isfinite(f))
    assert math.isclose(total0, float(f.sum()), rel_tol=1e-7)


def test_dissipation_density_nonnegative():
    W = 20e-9
    dx = 5e-9
    rng = np.random.default_rng(2)
    f = rng.uniform(0, 1, size=(20, 30))
    mu = rng.normal(size=(20, 30)) * 1e7
    Mxx, Mxy, Myy = surface_mobility_tensor(f, dx, W, M_s=1.0, bc_x="reflecting", bc_y="periodic", eps_n=1e-6 / W)
    from pf_sintering.surface_transport import dissipation_density
    D = dissipation_density(Mxx, Mxy, Myy, mu, dx, bc_x="reflecting", bc_y="periodic")
    assert np.all(D >= -1e-6 * np.max(np.abs(D)))
