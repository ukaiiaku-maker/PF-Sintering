from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pr_coarsening_driven_fourier_loading import (  # noqa: E402
    integral, reservoir_remove_continuous_surface,
    reservoir_transfer_particle_to_substrate_closed,
)


def synthetic_state():
    r_c = np.asarray([0.5, 1.5, 2.5])
    f = np.asarray([
        [0.20, 0.45, 0.75],
        [0.35, 0.55, 0.80],
        [0.25, 0.50, 0.70],
        [0.30, 0.60, 0.85],
    ])
    ownership = np.asarray([
        [0.05, 0.20, 0.40],
        [0.15, 0.35, 0.55],
        [0.45, 0.65, 0.80],
        [0.60, 0.85, 0.95],
    ])
    particle = f * ownership
    substrate = f * (1.0 - ownership)
    setup = dict(r_c=r_c, dr=1.0, dz=1.0, dt=1.0e-4,
                 tau_coarsen_model=20.0)
    return (f, particle, substrate), setup, ownership


def test_continuous_surface_reservoir_exact_particle_quota_and_ledgers():
    state, setup, ownership = synthetic_state()
    f0, particle0, substrate0 = state
    vp0 = integral(particle0, setup)
    expected_particle_loss = vp0 * (
        1.0 - math.exp(-setup["dt"] / setup["tau_coarsen_model"]))

    f1, particle1, substrate1, external = (
        reservoir_remove_continuous_surface(state, setup))

    assert math.isclose(
        integral(particle0 - particle1, setup), expected_particle_loss,
        rel_tol=1e-11)
    assert math.isclose(
        integral(f0 - f1, setup), external, rel_tol=1e-10)
    assert math.isclose(
        integral(f1, setup) + external, integral(f0, setup), rel_tol=1e-10)
    np.testing.assert_allclose(particle1 + substrate1, f1, rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        particle1 / f1, ownership, rtol=2e-14, atol=2e-14)


def test_continuous_surface_recession_has_no_ownership_velocity_jump():
    state, setup, _ = synthetic_state()
    f0, _, _ = state
    f1, _, _, _ = reservoir_remove_continuous_surface(state, setup)
    removal = f0 - f1
    fb = np.clip(f0, 0.0, 1.0)
    surface = 16.0 * fb**2 * (1.0 - fb)**2
    active = surface > 1e-12
    ratio = removal[active] / surface[active]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10, atol=0.0)


def test_closed_surface_transfer_preserves_total_and_particle_quota():
    state, setup, _ = synthetic_state()
    f0, particle0, substrate0 = state
    expected = integral(particle0, setup) * (
        1.0 - math.exp(-setup["dt"] / setup["tau_coarsen_model"]))
    f1, particle1, substrate1, diagnostic = (
        reservoir_transfer_particle_to_substrate_closed(state, setup))
    assert math.isclose(integral(particle0-particle1, setup), expected,
                        rel_tol=1e-11)
    assert math.isclose(integral(substrate1-substrate0, setup), expected,
                        rel_tol=1e-11)
    assert math.isclose(integral(f1, setup), integral(f0, setup), rel_tol=1e-13)
    np.testing.assert_allclose(particle1+substrate1, f1, rtol=0, atol=3e-16)
    assert abs(diagnostic["total_volume_relative_error"]) < 1e-13
    assert diagnostic["external_reservoir_m3"] == 0.0
