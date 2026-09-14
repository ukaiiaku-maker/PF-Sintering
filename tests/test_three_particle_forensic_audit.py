import pytest

from scripts.three_particle_forensic_audit import interval_totals, pause_rows


def test_interval_totals_separate_transfer_clock_from_fixed_q_recovery():
    rows = [
        {"time_s": 0., "cumulative_q_over_b": 0., "phase": "ACTIVE_ONE_B"},
        {"time_s": 10., "cumulative_q_over_b": .1,
         "phase": "EVENT_TRANSPORT_PAUSED"},
        {"time_s": 10.01, "cumulative_q_over_b": .1,
         "phase": "EVENT_TRANSPORT_RECOVERY"},
        {"time_s": 25.01, "cumulative_q_over_b": .2,
         "phase": "ACTIVE_ONE_B"},
        {"time_s": 25.019, "cumulative_q_over_b": .2,
         "phase": "FACILITATED_WINDOW"},
    ]

    totals = interval_totals(rows)

    assert totals["accepted_transfer_clock_s"] == pytest.approx(25.)
    assert totals["fixed_q_recovery_s"] == pytest.approx(.01)
    assert totals["source_window_s"] == pytest.approx(.009)


def test_pause_audit_groups_repeated_recovery_at_one_fixed_quota():
    base = dict(
        avalanche_id=2, event_number=9, LEFT_transport_affinity_Pa=1.,
        LEFT_sigma_local_Pa=2., RIGHT_sigma_local_Pa=3., V_center_m3=4.,
        center_span_m=5., LEFT_TJ_radius_m=6., RIGHT_TJ_radius_m=7.,
        LEFT_neck_radius_m=6., RIGHT_neck_radius_m=7.,
        GB_LEFT_m=-1., GB_RIGHT_m=1., centroid_strain=0., quota_strain=.5,
        energy_J=8.)
    rows = [
        {**base, "time_s": 0., "q_over_b": .5, "phase": "ACTIVE_ONE_B"},
        {**base, "time_s": 1., "q_over_b": .5,
         "phase": "EVENT_TRANSPORT_PAUSED"},
        {**base, "time_s": 1.1, "q_over_b": .5,
         "phase": "EVENT_TRANSPORT_RECOVERY", "LEFT_transport_affinity_Pa": -2.},
        {**base, "time_s": 1.1, "q_over_b": .5,
         "phase": "EVENT_TRANSPORT_PAUSED", "LEFT_transport_affinity_Pa": -2.},
        {**base, "time_s": 1.2, "q_over_b": .5,
         "phase": "EVENT_TRANSPORT_RECOVERY", "LEFT_transport_affinity_Pa": .5},
        {**base, "time_s": 2., "q_over_b": .51, "phase": "ACTIVE_ONE_B"},
    ]

    episodes = pause_rows(rows)

    assert len(episodes) == 1
    assert episodes[0]["recovery_steps"] == 2
    assert episodes[0]["duration_s"] == pytest.approx(.2)
    assert episodes[0]["affinity_min_Pa"] == -2.
