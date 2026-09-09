from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pr_avalanche_renewal_five as renewal  # noqa: E402
from pr_coarsening_driven_fourier_loading import (  # noqa: E402
    closed_surface_diffusion_only, integral,
)


def state_and_setup():
    f = np.asarray([[.2, .5], [.4, .7]])
    e1 = .4*f
    e2 = f-e1
    setup = dict(r_c=np.asarray([.5, 1.5]), dr=1.0, dz=1.0)
    return (f, e1, e2), setup


def test_closed_callback_is_exact_identity():
    state, setup = state_and_setup()
    result = closed_surface_diffusion_only(state, setup)
    for before, after in zip(state, result[:3]):
        assert after is before
    assert result[3] == 0.0


def test_closed_production_invariant_accepts_roundoff_and_rejects_loss():
    state, setup = state_and_setup()
    previous = (renewal.SYSTEM_MASS_BOUNDARY,
                renewal.SOLID_VOLUME_INITIAL_M3,
                renewal.SOLID_VOLUME_RELATIVE_TOLERANCE)
    try:
        renewal.SYSTEM_MASS_BOUNDARY = "closed"
        renewal.SOLID_VOLUME_INITIAL_M3 = integral(state[0], setup)
        renewal.SOLID_VOLUME_RELATIVE_TOLERANCE = 1e-8
        renewal.assert_closed_solid_volume(state, setup, context="test")
        lost = (state[0]*(1-2e-8), state[1], state[2])
        with pytest.raises(RuntimeError, match="closed solid-volume invariant failed"):
            renewal.assert_closed_solid_volume(lost, setup, context="test")
    finally:
        (renewal.SYSTEM_MASS_BOUNDARY,
         renewal.SOLID_VOLUME_INITIAL_M3,
         renewal.SOLID_VOLUME_RELATIVE_TOLERANCE) = previous
