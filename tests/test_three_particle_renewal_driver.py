import pytest

from scripts.three_particle_forced_event import MANIFEST
from scripts.three_particle_renewal_run import (
    event_physical_time_s, event_progress_over_b, incomplete_event_phase)


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
