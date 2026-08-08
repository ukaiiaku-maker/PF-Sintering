import math

import numpy as np
from scipy.ndimage import gaussian_filter

from pf_sintering.model import ModelConfig, Sink, build_params, lap9
from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic


def _p():
    return build_params(ModelConfig(
        preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=20e-9, t_total=1e-6, coarsening_rate_scale=0.0, surface_mobility_scale=0.3,
        eta_mobility_scale=1.0, use_aniso_surface=False,
    ))


def _smooth_field(rng, shape, sigma=3.0):
    return gaussian_filter(rng.normal(size=shape), sigma=sigma)


def test_lap9_is_symmetric_under_plain_dot_product():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 60))
    b = rng.normal(size=(40, 60))
    dx = 5e-9
    lhs = np.sum(a * lap9(b, dx))
    rhs = np.sum(b * lap9(a, dx))
    assert math.isclose(lhs, rhs, rel_tol=1e-12)


def test_exact_energy_directional_derivative_matches_mu_isotropic():
    p = _p()
    rng = np.random.default_rng(1)
    f = np.clip(0.5 + 0.1 * _smooth_field(rng, (p.Ny, p.Nx)), 0.05, 0.95)
    e1 = 0.3 * np.ones_like(f)
    e2 = 0.3 * np.ones_like(f)
    e3 = np.zeros_like(f)
    q = _smooth_field(rng, (p.Ny, p.Nx))
    q -= q.mean()
    s = Sink(threshold=math.inf)

    mu = mu_isotropic(f, e1, e2, e3, s, p)
    rhs = p.dx * p.dx * np.sum(mu * q)

    errs = []
    for eps in (1e-2, 1e-3, 1e-4):
        Fp = exact_free_energy_isotropic(f + eps * q, e1, e2, e3, s, p)
        Fm = exact_free_energy_isotropic(f - eps * q, e1, e2, e3, s, p)
        lhs = (Fp - Fm) / (2 * eps)
        errs.append(abs(lhs - rhs) / abs(rhs))
    # O(eps^2) central-difference convergence: each halving of the log-step
    # (10x here) should reduce the relative error by roughly two orders of
    # magnitude, down to a floating-point-roundoff floor.
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-6
