import pytest

from scripts.three_particle_campaign_report import (
    completed_event_contacts, phase_volume_changes)


def test_center_volume_changes_are_separated_by_physical_phase():
    history = [
        {"phase": "POST_TRANSIENT_NEW_TRAJECTORY", "center_volume_m3": 10.0},
        {"phase": "RELOAD", "center_volume_m3": 9.0},
        {"phase": "ROOT_CROSSING", "center_volume_m3": 8.5},
        {"phase": "ACTIVE_ONE_B", "center_volume_m3": 8.7},
        {"phase": "EVENT_TRANSPORT_PAUSED", "center_volume_m3": 8.6},
        {"phase": "ONE_B_COMPLETE", "center_volume_m3": 8.9},
        {"phase": "SOURCE_WINDOW_OPEN", "center_volume_m3": 8.8},
        {"phase": "FACILITATED_WINDOW", "center_volume_m3": 8.4},
        {"phase": "AVALANCHE_EXTINCT_REPINNED", "center_volume_m3": 8.3},
    ]

    changes = phase_volume_changes(history)

    assert changes["passive_reload"] == pytest.approx(-1.5)
    assert changes["active_event_transit"] == pytest.approx(0.4)
    assert changes["facilitated_source_window"] == pytest.approx(-0.6)
    assert sum(changes.values()) == pytest.approx(-1.7)


def test_completed_transfers_are_counted_once_at_source_window_entry():
    history = [
        {"phase": "ACTIVE_ONE_B", "event_number": 1, "contact": "LEFT"},
        {"phase": "SOURCE_WINDOW_OPEN", "event_number": 1, "contact": "LEFT"},
        {"phase": "SOURCE_WINDOW_OPEN", "event_number": 1, "contact": "LEFT"},
        {"phase": "ACTIVE_ONE_B", "event_number": 2, "contact": "RIGHT"},
        {"phase": "SOURCE_WINDOW_OPEN", "event_number": 2, "contact": "RIGHT"},
    ]

    assert completed_event_contacts(history) == ["LEFT", "RIGHT"]
