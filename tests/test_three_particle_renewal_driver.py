import numpy as np
import pytest

from scripts.three_particle_forced_event import MANIFEST
from scripts.three_particle_renewal_run import (
    accepted_reload_step_hint, event_checkpoint_due, event_physical_time_s,
    event_progress_over_b, event_accepted_state_cap, incomplete_event_phase,
    pending_event_milestones, save_immutable_event_checkpoint)
from pf_sintering.three_particle_event import load_event_checkpoint


def test_reload_step_hint_is_not_collapsed_by_short_final_remainder():
    assert accepted_reload_step_hint(.01, .001, .01) == .01
    assert accepted_reload_step_hint(.001, .001, .25) == .0016
    assert accepted_reload_step_hint(.003, .003, .688) == .003


def test_event_state_cap_covers_a_full_event_at_the_manifest_floor():
    assert event_accepted_state_cap(.0025) == 400
    assert event_accepted_state_cap(.00125) == 800


def test_partial_event_pause_preserves_quota_and_adds_frozen_clock_time():
    restart = {
        "cumulative_q_m": .8775*MANIFEST["b_event_m"],
        "event_time_model": 3.25,
    }
    assert event_progress_over_b(restart) == pytest.approx(.8775)
    expected = 11. + .0004 + 3.25*MANIFEST["seconds_per_model_time"]
    assert event_physical_time_s(11., .0004, restart) == pytest.approx(expected)


def test_only_affinity_exhaustion_enters_transport_pause():
    assert incomplete_event_phase(
        False, {"stop_reason": "nonpositive_transport_affinity"}) == (
            "EVENT_TRANSPORT_PAUSED")
    assert incomplete_event_phase(True, {"stop_reason": None}) is None
    with pytest.raises(RuntimeError, match="three-grain topology guard"):
        incomplete_event_phase(
            False, {"stop_reason": "three-grain topology guard"})


def test_late_event_checkpoints_preserve_each_expensive_minimum_step():
    assert not event_checkpoint_due(.9775, .97)
    assert event_checkpoint_due(.98, .97)
    assert event_checkpoint_due(.9825, .98)
    assert not event_checkpoint_due(.98125, .98)
    assert event_checkpoint_due(1., .9975)


def test_quarter_event_milestones_use_first_accepted_state_and_never_repeat():
    assert pending_event_milestones(.249, set()) == []
    assert pending_event_milestones(.2525, set()) == [.25]
    assert pending_event_milestones(.755, {"q_0.25b.npz"}) == [.50, .75]
    assert pending_event_milestones(1., {
        "q_0.25b.npz", "q_0.50b.npz", "q_0.75b.npz"}) == []


def test_event_milestone_preserves_first_accepted_state(tmp_path):
    first = tuple(np.full((2, 3), value, dtype=float) for value in range(4))
    later = tuple(value + 10 for value in first)
    restart = {
        "base_fields": first,
        "union_previous_fields": first,
        "cumulative_q_m": .2525*MANIFEST["b_event_m"],
        "event_time_model": 1.0,
    }
    path = tmp_path/"q_0.25b.npz"
    assert save_immutable_event_checkpoint(
        path, first, restart, contact="LEFT")
    assert not save_immutable_event_checkpoint(
        path, later, restart, contact="LEFT")
    restored, metadata, contact, label = load_event_checkpoint(path)
    for actual, expected in zip(restored, first):
        np.testing.assert_array_equal(actual, expected)
    assert metadata["cumulative_q_m"]/MANIFEST["b_event_m"] == pytest.approx(.2525)
    assert contact == "LEFT"
    assert label == "GENUINE_STOCHASTIC_EVENT_IMMUTABLE_MILESTONE"
