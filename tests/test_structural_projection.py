import numpy as np
import pytest

from pf_sintering.model import ModelConfig, build_params
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving


def test_projection_preserves_grain_masses_and_local_capacity():
    p = build_params(ModelConfig(preset="dev", nx=8, ny=8, t_total=1e-6))
    f = np.ones((8, 8), dtype=float)

    # Two grains overlap too strongly in the middle, while empty solid capacity
    # exists on their outer sides.  Total requested ownership equals total solid
    # capacity, so a conservative redistribution is feasible.
    e1 = np.zeros_like(f)
    e2 = np.zeros_like(f)
    e3 = np.zeros_like(f)
    e1[:, :5] = 0.8
    e2[:, 3:] = 0.8

    target = eta_masses(e1, e2, e3, False)
    a1, a2, a3 = project_eta_mass_preserving(
        f, e1, e2, e3, p, target_masses=target
    )
    final = eta_masses(a1, a2, a3, False)

    assert np.allclose(final[:2], target[:2], rtol=2e-11, atol=2e-10)
    assert np.max(a1 + a2 + a3 - f) <= 1e-11
    assert np.min(a1) >= -1e-12
    assert np.min(a2) >= -1e-12
    assert np.all(a3 == 0)


def test_projection_fails_when_requested_mass_exceeds_solid_capacity():
    p = build_params(ModelConfig(preset="dev", nx=8, ny=8, t_total=1e-6))
    f = np.full((8, 8), 0.5)
    e1 = np.full((8, 8), 0.4)
    e2 = np.full((8, 8), 0.4)
    e3 = np.zeros_like(f)

    with pytest.raises(RuntimeError, match="infeasible"):
        project_eta_mass_preserving(
            f,
            e1,
            e2,
            e3,
            p,
            target_masses=eta_masses(e1, e2, e3, False),
        )
