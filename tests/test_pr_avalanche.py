import math

from pf_sintering.exp_barrier_nucleation import (
    CompleteExpFloorParams,
    delta_G_complete_exp_floor_eV,
)
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


def test_descendant_barrier_is_fixed_root_decrement_and_nonadditive():
    barrier = DescendantBarrier(ROOT, delta_G_step_eV=0.3)
    for sigma in (0.0, 200e6, 2e9, 1e15):
        root = delta_G_complete_exp_floor_eV(sigma, ROOT)
        expected = max(ROOT.G_floor_eV, root - 0.3)
        assert math.isclose(barrier.barrier_eV(sigma), expected)
        assert barrier.barrier_eV(sigma) == barrier.barrier_eV(sigma)


def test_descendant_barrier_never_drops_below_root_floor():
    barrier = DescendantBarrier(ROOT, delta_G_step_eV=0.3)
    assert barrier.barrier_eV(1e30) == ROOT.G_floor_eV


def test_ring_rate_uses_full_site_count():
    barrier = DescendantBarrier(ROOT, delta_G_step_eV=0.3)
    rate = descendant_rate_per_s(
        0.0, 50e-9, barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9)
    kT_eV = 1.380649e-23 * 1800.0 / 1.602176634e-19
    expected = (
        2.0 * math.pi * 50e-9 / 0.25e-9 * 1e12
        * math.exp(-4.7 / kT_eV))
    assert math.isclose(rate, expected)


def test_transit_clock_is_frozen_and_never_queues_children():
    barrier = DescendantBarrier(ROOT, delta_G_step_eV=0.3)
    ctl = AvalancheController(
        barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9,
        correlation_time_s=9e-3, rng=SequenceRNG([0.2]))
    ctl.start(avalanche_id=1, root_cycle=1, start_time_s=0.0)
    ctl.begin_transit()
    trace = ctl.integrate_transit_samples([
        dict(t_s=0.0, sigma_local_Pa=80e6, r_TJ_m=50e-9),
        dict(t_s=1.0e-6, sigma_local_Pa=70e6, r_TJ_m=49e-9),
    ])
    assert ctl.state.pending_children == 0
    assert ctl.state.descendant_hazard == 0.0
    assert not ctl.crossings
    assert trace[-1]["crossings_in_segment"] == 0


def test_completed_event_restarts_one_source_window():
    barrier = DescendantBarrier(ROOT, delta_G_step_eV=0.3)
    ctl = AvalancheController(
        barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9,
        correlation_time_s=9e-3, rng=SequenceRNG([0.5, 2.0]))
    ctl.start(avalanche_id=1, root_cycle=1, start_time_s=0.0)
    ctl.complete_transit(1.0)
    assert ctl.state.descendant_threshold == 0.5
    assert ctl.state.window_start_time_s == 1.0
    assert ctl.state.window_deadline_s == 1.009

    crossing = ctl.accumulate_window_segment(
        rate_start_per_s=100.0, rate_end_per_s=100.0,
        start_time_s=1.0, end_time_s=1.009)
    assert crossing["crossed"]
    assert ctl.state.window_triggered
    assert ctl.state.pending_children == 0
    assert len(ctl.crossings) == 1

    ctl.begin_transit()
    ctl.complete_transit(2.0)
    assert ctl.state.descendant_threshold == 2.0
    assert not ctl.state.window_triggered
    assert ctl.state.descendant_hazard == 0.0
    assert ctl.state.window_deadline_s == 2.009


def test_window_extinction_after_exact_nine_ms():
    barrier = DescendantBarrier(ROOT, delta_G_step_eV=0.3)
    ctl = AvalancheController(
        barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9,
        correlation_time_s=9e-3, rng=SequenceRNG([10.0]))
    ctl.start(avalanche_id=2, root_cycle=2, start_time_s=0.0)
    ctl.complete_transit(1.0)
    result = ctl.accumulate_window_segment(
        rate_start_per_s=1.0, rate_end_per_s=1.0,
        start_time_s=1.0, end_time_s=1.009)
    assert not result["crossed"]
    ctl.expire_window(1.009)
    assert not ctl.state.avalanche_active
    assert math.isclose(ctl.state.descendant_hazard, 0.009)


def test_only_one_activation_can_cross_per_window():
    barrier = DescendantBarrier(ROOT, delta_G_step_eV=0.3)
    ctl = AvalancheController(
        barrier=barrier, temperature_K=1800.0,
        attempt_frequency_per_s=1e12, b_m=0.25e-9,
        correlation_time_s=9e-3, rng=SequenceRNG([0.1]))
    ctl.start(avalanche_id=1, root_cycle=1, start_time_s=0.0)
    ctl.complete_transit(0.0)
    result = ctl.accumulate_window_segment(
        rate_start_per_s=1e6, rate_end_per_s=1e6,
        start_time_s=0.0, end_time_s=1e-3)
    assert result["crossed"]
    try:
        ctl.accumulate_window_segment(
            rate_start_per_s=1e6, rate_end_per_s=1e6,
            start_time_s=1e-3, end_time_s=2e-3)
    except RuntimeError as error:
        assert "already triggered" in str(error)
    else:
        raise AssertionError("a triggered window accepted a second activation")
