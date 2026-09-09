import math

import numpy as np

from pf_sintering.model_time_transport import ModelTimeGBTransport
from pf_sintering.quasistatic_event_continuation import (
    default_q_schedule,
    fast_manifold_converged,
    integrate_slow_clock,
)


def transport():
    return ModelTimeGBTransport(
        x_d_m=1.0, kB_J_per_K=1.0, temperature_K=1.0,
        atomic_volume_m3=1.0, b_m=2.0,
        D_gb_m2_per_model_time=1.0)


def test_schedule_contains_requested_early_points_and_endpoint():
    q = default_q_schedule()
    assert np.allclose(q[:4], [0.0, 0.005, 0.01, 0.02])
    assert q[-1] == 1.0
    assert np.all(np.diff(q) > 0.0)
    assert np.count_nonzero(q == 1.0) == 1


def test_slow_clock_is_exact_for_constant_affinity():
    q = np.array([0.0, 0.25, 1.0])
    cumulative, intervals = integrate_slow_clock(q, [2.0, 2.0, 2.0], transport())
    # K=1 and affinity=2 give tau=0.5, hence qdot=b/tau=4.
    assert np.allclose(intervals, [0.0, 0.125, 0.375])
    assert np.allclose(cumulative, [0.0, 0.125, 0.5])


def test_nonpositive_affinity_blocks_clock():
    cumulative, intervals = integrate_slow_clock(
        [0.0, 0.5, 1.0], [2.0, 0.0, 3.0], transport())
    assert math.isinf(intervals[1])
    assert np.all(np.isinf(cumulative[1:]))


def test_fast_gate_requires_all_limits():
    increment = {"a": 0.1, "b": 0.2}
    assert fast_manifold_converged(increment, {"a": 0.1, "b": 0.3})
    assert not fast_manifold_converged(increment, {"a": 0.09, "b": 0.3})
