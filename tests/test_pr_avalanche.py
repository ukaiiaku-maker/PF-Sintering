import math

import numpy as np

from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
from pf_sintering.pr_avalanche import (
    AvalancheController,
    DescendantBarrier,
    descendant_rate_per_s,
)


ROOT = CompleteExpFloorParams(
    G0_eV=5.0, G_floor_eV=2.0, a=1.2,
    sigma_hat_pa=1.7e9, n=0.5)


class SequenceRNG:
    def __init__(self, values):
        self.values = iter(values)

    def exponential(self):
        return next(self.values)


def test_descendant_barrier_is_absolute_and_nonadditive():
    barrier = DescendantBarrier(ROOT, G0_step_eV=3.0)
    assert barrier.G0_effective_eV == 3.0
    assert math.isclose(barrier.barrier_eV(0.0), 3.0)
    first = barrier.barrier_eV(200e6)
    assert first == barrier.barrier_eV(200e6)
    assert math.isclose(barrier.floor_fraction, 0.4)


def test_step_above_root_leaves_root_barrier_unchanged():
    barrier = DescendantBarrier(ROOT, G0_step_eV=8.0)
    assert barrier.G0_effective_eV == ROOT.G0_eV
    assert math.isclose(barrier.barrier_eV(0.0), ROOT.G0_eV)


def test_ring_rate_uses_full_site_count():
    barrier = DescendantBarrier(ROOT, G0_step_eV=3.0)
    rate = descendant_rate_per_s(
        0.0, 50e-9, barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9)
    kT_eV = 1.380649e-23 * 1800.0 / 1.602176634e-19
    expected = 2.0 * math.pi * 50e-9 / 0.25e-9 * 1e12 * math.exp(-3.0/kT_eV)
    assert math.isclose(rate, expected)


def test_multiple_thresholds_commit_during_one_transit():
    barrier = DescendantBarrier(ROOT, G0_step_eV=3.0)
    ctl = AvalancheController(
        barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9,
        correlation_time_s=2e-6, rng=SequenceRNG([0.2, 0.3, 10.0]))
    ctl.start(avalanche_id=1, root_cycle=1, start_time_s=0.0)
    ctl.rate = lambda sigma, radius: 1.0e6
    ctl.begin_transit()
    trace = ctl.integrate_transit_samples([
        dict(t_s=0.0, sigma_local_Pa=80e6, r_TJ_m=50e-9),
        dict(t_s=1.0e-6, sigma_local_Pa=70e6, r_TJ_m=49e-9),
    ])
    assert ctl.state.pending_children == 2
    assert len(ctl.crossings) == 2
    assert trace[-1]["crossings_in_segment"] == 2
    assert math.isclose(ctl.state.descendant_hazard, 0.5)


def test_correlation_window_crosses_once_or_extinguishes():
    barrier = DescendantBarrier(ROOT, G0_step_eV=3.0)
    crossing = AvalancheController(
        barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9,
        correlation_time_s=2e-6, rng=SequenceRNG([0.5, 2.0]))
    crossing.start(avalanche_id=1, root_cycle=1, start_time_s=0.0)
    crossing.rate = lambda sigma, radius: 1.0e6
    result = crossing.correlation_window(
        sigma_local_Pa=70e6, r_TJ_m=50e-9, start_time_s=0.0)
    assert result["continued"]
    assert math.isclose(result["elapsed_s"], 0.5e-6)
    assert crossing.state.pending_children == 1
    assert crossing.state.avalanche_active

    extinct = AvalancheController(
        barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9,
        correlation_time_s=2e-6, rng=SequenceRNG([10.0]))
    extinct.start(avalanche_id=2, root_cycle=2, start_time_s=0.0)
    extinct.rate = lambda sigma, radius: 1.0e6
    result = extinct.correlation_window(
        sigma_local_Pa=70e6, r_TJ_m=50e-9, start_time_s=0.0)
    assert not result["continued"]
    assert not extinct.state.avalanche_active
    assert math.isclose(extinct.state.descendant_hazard, 2.0)
