import numpy as np

import pf_sintering.production_q_event as production_q_event
from pf_sintering.production_q_event import (
    _precondition_restarted_fast_manifold,
    prescribed_q_event,
)


class _LinearTransport:
    b_m = 1.0
    kinetic_time_basis = "test model time"

    def qdot_m_per_model_time(self, affinity_pa):
        return max(float(affinity_pa), 0.0)


def _row(affinity):
    return {
        "transport_affinity_Pa": float(affinity),
        "transport_affinity_MPa": float(affinity) * 1.0e-6,
        "sigma_local_MPa": 1.0,
        "sigma_integral_MPa": 1.0,
        "r_neck_nm": 1.0,
        "r_TJ_nm": 1.0,
        "A1_nm": 1.0,
        "z_TJ_m": 0.0,
        "kappa1_negative_per_m": 1.0,
        "kappa1_positive_per_m": 1.0,
    }


def test_restart_precondition_converges_at_fixed_q_without_external_state():
    seen_q = []

    def metrics(state, q_over_b):
        seen_q.append(float(q_over_b))
        return _row(state[0][0])

    def relax(state, q_over_b):
        seen_q.append(float(q_over_b))
        advanced = (np.asarray([state[0][0] + 1.0]),)
        return advanced, _row(advanced[0][0]), {
            "converged": True, "iterations": 1, "blocks": 2}

    original = (np.asarray([1.0]),)
    settled, row, info = _precondition_restarted_fast_manifold(
        original, 0.25, fast_relax_fn=relax, state_metrics_fn=metrics,
        transport=_LinearTransport(), reciprocal_fraction_target=0.1,
        required_consecutive=2, maximum_calls=32)

    assert info["converged"]
    assert info["calls"] == 19
    assert info["total_fast_relax_blocks"] == 38
    assert info["last_reciprocal_qdot_fraction"] <= 0.1
    assert row["transport_affinity_Pa"] == 20.0
    assert settled[0][0] == 20.0
    assert original[0][0] == 1.0
    assert set(seen_q) == {0.25}


def test_restart_precondition_leaves_nonpositive_affinity_for_pause_logic():
    def metrics(state, q_over_b):
        return _row(state[0][0])

    def should_not_relax(state, q_over_b):
        raise AssertionError("nonpositive state must be handled by pause logic")

    original = (np.asarray([0.0]),)
    settled, row, info = _precondition_restarted_fast_manifold(
        original, 0.4, fast_relax_fn=should_not_relax,
        state_metrics_fn=metrics, transport=_LinearTransport(),
        reciprocal_fraction_target=0.01)

    assert not info["converged"]
    assert info["calls"] == 0
    assert info["skipped_reason"] == "nonpositive transport affinity"
    assert row["transport_affinity_Pa"] == 0.0
    assert settled[0][0] == 0.0


def test_every_ordinary_q_endpoint_is_clock_converged_before_acceptance(
        monkeypatch):
    seen_relax_q = []

    monkeypatch.setattr(
        production_q_event, "_axisym_weighted_sum",
        lambda field, radial_coordinates: 1.0)
    monkeypatch.setattr(
        production_q_event, "representation_corrected_union_transfer",
        lambda *args, **kwargs: (
            np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0]),
            {"V_source_weighted": 0.0}))

    def direct_trial(*args, q_end_over_b, **kwargs):
        return (
            (np.asarray([q_end_over_b]), np.asarray([0.0]),
             np.asarray([0.0])),
            {"source_cumulative_weighted": 0.0})

    monkeypatch.setattr(
        production_q_event, "direct_prescribed_q_trial", direct_trial)

    def metrics(state, q_over_b):
        return _row(state[0][0])

    def relax(state, q_over_b):
        seen_relax_q.append(float(q_over_b))
        advanced = (
            np.asarray([state[0][0] + 1.0]),
            np.asarray(state[1]).copy(), np.asarray(state[2]).copy())
        return advanced, _row(advanced[0][0]), {
            "converged": True, "iterations": 10, "blocks": 1}

    limits = dict(production_q_event.DEFAULT_STEP_LIMITS)
    limits["reciprocal_qdot_fraction"] = 0.5
    fields = (np.asarray([10.0]), np.asarray([0.0]), np.asarray([0.0]))
    *_, completed, result = prescribed_q_event(
        fields, {"dz": 1.0, "r_c": np.asarray([1.0]),
                 "z": np.asarray([0.0])}, {"z1": 0.0}, None,
        _LinearTransport(), 0.1, fast_relax_fn=relax,
        state_metrics_fn=metrics,
        node_surface_mobility_m6_per_J_model_time=1.0,
        initial_step_over_b=0.1, minimum_step_over_b=0.1,
        maximum_step_over_b=0.1,
        restart_precondition_fraction_of_qdot_limit=0.5,
        restart_precondition_required_consecutive=3, step_limits=limits)

    assert completed
    assert seen_relax_q == [0.0, 0.0, 0.0, 0.1, 0.1, 0.1]
    packet = result["packets"][0]
    assert packet["fast_relax_calls"] == 3
    assert packet["fast_relax_blocks"] == 3
    assert packet["q_over_b_before_fixed_q_relaxation"] == 0.1
    assert packet["q_start_fast_relax_calls"] == 3
    assert packet["affinity_before_fixed_q_convergence_Pa"] == 13.1
    assert packet["affinity_after_fixed_q_convergence_Pa"] == 16.1
    assert (packet["final_fast_convergence_reciprocal_qdot_fraction"]
            <= packet["fast_convergence_reciprocal_qdot_target"])


def test_accepted_endpoint_is_reused_without_duplicate_fixed_q_relaxation(
        monkeypatch):
    seen_relax_q = []

    monkeypatch.setattr(
        production_q_event, "_axisym_weighted_sum",
        lambda field, radial_coordinates: 1.0)
    monkeypatch.setattr(
        production_q_event, "representation_corrected_union_transfer",
        lambda *args, **kwargs: (
            np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0]),
            {"V_source_weighted": 0.0}))

    def direct_trial(*args, q_end_over_b, **kwargs):
        return (
            (np.asarray([q_end_over_b]), np.asarray([0.0]),
             np.asarray([0.0])),
            {"source_cumulative_weighted": 0.0})

    monkeypatch.setattr(
        production_q_event, "direct_prescribed_q_trial", direct_trial)

    def metrics(state, q_over_b):
        return _row(state[0][0])

    def relax(state, q_over_b):
        seen_relax_q.append(float(q_over_b))
        # Keep the mock endpoint stationary so exactly the configured three
        # calls are sufficient for each cumulative convergence window.
        copied = tuple(np.asarray(field).copy() for field in state)
        return copied, _row(copied[0][0]), {
            "converged": True, "iterations": 1, "blocks": 1}

    limits = dict(production_q_event.DEFAULT_STEP_LIMITS)
    limits["reciprocal_qdot_fraction"] = 0.5
    fields = (np.asarray([10.0]), np.asarray([0.0]), np.asarray([0.0]))
    *_, completed, result = prescribed_q_event(
        fields, {"dz": 1.0, "r_c": np.asarray([1.0]),
                 "z": np.asarray([0.0])}, {"z1": 0.0}, None,
        _LinearTransport(), 0.2, fast_relax_fn=relax,
        state_metrics_fn=metrics,
        node_surface_mobility_m6_per_J_model_time=1.0,
        initial_step_over_b=0.1, minimum_step_over_b=0.1,
        maximum_step_over_b=0.1,
        restart_precondition_fraction_of_qdot_limit=0.5,
        restart_precondition_required_consecutive=3, step_limits=limits)

    assert completed
    assert [packet["q_end_over_b"] for packet in result["packets"]] == [
        0.1, 0.2]
    assert seen_relax_q == [0.0] * 3 + [0.1] * 3 + [0.2] * 3
