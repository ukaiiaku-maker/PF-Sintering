from __future__ import annotations

import math

import numpy as np

from pf_sintering.conservative_bounded_phase import (
    ConservativeBoundedPhaseProjector,
)


def test_projection_is_bounded_conservative_and_ownership_preserving():
    f = np.asarray([
        [-0.02, 0.25, 0.75, 1.03],
        [0.05, 0.40, 0.60, 0.95],
    ])
    ownership = np.asarray([
        [0.1, 0.2, 0.8, 0.9],
        [0.3, 0.4, 0.6, 0.7],
    ])
    clipped = np.clip(f, 0.0, 1.0)
    state = (f, clipped*ownership, clipped*(1.0-ownership))
    setup = dict(r_c=np.asarray([0.5, 1.5, 2.5, 3.5]), dr=0.8, dz=1.2)
    weights = (2.0*math.pi*setup["r_c"][None, :]
               * setup["dr"]*setup["dz"])
    volume_before = float(np.sum(weights*f))

    projector = ConservativeBoundedPhaseProjector()
    projected, diagnostics = projector(state, setup)
    f1, p1, s1 = projected

    assert float(np.min(f1)) >= 0.0
    assert float(np.max(f1)) <= 1.0
    assert math.isclose(float(np.sum(weights*f1)), volume_before, rel_tol=2e-13)
    np.testing.assert_allclose(p1+s1, f1, rtol=0.0, atol=2e-16)
    occupied = f1 > 1e-15
    np.testing.assert_allclose(
        p1[occupied]/f1[occupied], ownership[occupied],
        rtol=2e-14, atol=2e-14)
    assert math.isclose(diagnostics["raw_overshoot"], 0.03)
    assert projector.manifest()["active_calls"] == 1


def test_bounded_fast_path_repairs_partition_without_changing_f_or_ownership():
    f = np.asarray([
        [0.05, 0.25, 0.75, 0.95],
        [0.10, 0.40, 0.60, 0.90],
    ])
    ownership = np.asarray([
        [0.1, 0.2, 0.8, 0.9],
        [0.3, 0.4, 0.6, 0.7],
    ])
    drift = 6.2e-13
    particle = f*ownership + drift
    substrate = f*(1.0-ownership)
    raw_sum = particle+substrate
    raw_ownership = particle/raw_sum
    setup = dict(r_c=np.asarray([0.5, 1.5, 2.5, 3.5]), dr=0.8, dz=1.2)

    projector = ConservativeBoundedPhaseProjector()
    projected, diagnostics = projector((f, particle, substrate), setup)
    f1, p1, s1 = projected

    assert np.array_equal(f1, f)
    np.testing.assert_allclose(p1+s1, f1, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(p1/f1, raw_ownership, rtol=2e-14, atol=2e-14)
    assert diagnostics["partition_repaired"] == 1
    assert projector.manifest()["partition_repair_calls"] == 1
