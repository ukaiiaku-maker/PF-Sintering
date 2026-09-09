#!/usr/bin/env python3
"""Next corrected production campaign with slower GB-controlled avalanches.

The passive PR physics, root barrier, surface mobility, event displacement,
source lifetime, and facilitation decay remain unchanged. The intentional
next-run changes are recorded explicitly below.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_renewal_five as renewal  # noqa: E402
import pr_corrected_normalized_ownership_production as corrected  # noqa: E402
import pr_full_corrected_production_campaign as campaign  # noqa: E402
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.corrected_axisym_dynamics import (  # noqa: E402
    CorrectedNumbaScratch,
    axisym_corrected_normalized_ownership_step_fast,
)
from pf_sintering.axisym import axisym_volume  # noqa: E402
from pf_sintering.production_q_event import prescribed_q_event  # noqa: E402
from pr_constant_time_movie_rendering import (  # noqa: E402
    render_constant_time_production_movies,
)
from pr_full_deterministic_cycle import make_transport  # noqa: E402
from pr_native_postevent_one_avalanche_replay import (  # noqa: E402
    copy_energy_diagnostic,
)
from pr_quasistatic_q_continuation import (  # noqa: E402
    relax_fast_manifold,
    scalar_state,
)


OUT = ROOT / "runs/pr_full_corrected_production_slowgb_deltaG1p500_qcont_v3"
CALIBRATION = campaign.CALIBRATION
EVENT_QUALIFICATION = ROOT / (
    "runs/pr_direct_adaptive_q_event_regression_v5/decision.json")
AVALANCHES_REQUESTED = 5
MAX_DESCENDANTS_SAFETY = 25
REUSED_SEED = None
CONTINUATION = None
EXTRA_RUN_METADATA = {}

# The accepted 13-event lineage had an 8.792371 microsecond median 1b time.
# Ten events in 1.5 s implies a 0.15 s median 1b target. With tau_ex=0, the
# Coble-like delivery time is inversely proportional to D_GB.
D_GB_REFERENCE = 5.228329357301188e-11
REFERENCE_MEDIAN_EVENT_S = 8.792371092776342e-06
TARGET_EVENTS_PER_AVALANCHE = 10
TARGET_AVALANCHE_RELAXATION_S = 0.5
TARGET_MEDIAN_EVENT_S = (
    TARGET_AVALANCHE_RELAXATION_S / TARGET_EVENTS_PER_AVALANCHE)
D_GB_TARGET = (
    D_GB_REFERENCE * REFERENCE_MEDIAN_EVENT_S / TARGET_MEDIAN_EVENT_S)

DESCENDANT_DELTA_G0_EV = 1.500000
DESCENDANT_FACILITATION_DECAY_ALPHA = 0.60
ROOT_THRESHOLD_MULTIPLIER = 1.0
DESCENDANT_ESCALATION_AFTER_AVALANCHES = None
DESCENDANT_ESCALATION_IF_ALL_S_BELOW = None
DESCENDANT_ESCALATION_NEW_DELTA_G_EV = None
SOLID_VOLUME_RELATIVE_TOLERANCE = 1.0e-7
EVENT_MAX_INCREMENT_FRACTION_B = 0.001
EVENT_MINIMUM_INCREMENT_FRACTION_B = 0.00125
EVENT_RESTART_FAST_PRECONDITION = False
EVENT_FAST_CLOCK_CONVERGENCE_FRACTION = 0.25
EVENT_FAST_CLOCK_MAX_CALLS = 2048
# Qualified local branch-redistribution length used by the existing
# physical-timescale assessment (not the diffuse-interface thickness).
EVENT_SURFACE_ACTIVE_LENGTH_M = 15.0e-9


def slow_gb_transport(geom):
    return replace(
        make_transport(geom), D_gb_m2_per_model_time=D_GB_TARGET)


def quasistatic_event_call(
        state, setup, geom, evaluator, transport, quota_fraction_b,
        *, event_restart=None, state_callback=None, progress_callback=None,
        **kwargs):
    """Production adapter for the qualified prescribed-q propagator."""
    event_base = (state if event_restart is None
                  else event_restart["base_fields"])
    volume0 = axisym_volume(
        event_base[0], setup["r_c"], setup["dr"], setup["dz"])

    def metrics(fields, q_over_b):
        return scalar_state(
            fields, q_over_b, setup, geom, evaluator, volume0)

    def fast_relax(fields, q_over_b):
        relaxed, row, _history, _projector = relax_fast_manifold(
            fields, q_over_b, setup, geom, volume0)
        return relaxed, row, dict(
            converged=bool(row.get("fast_manifold_converged", 0)),
            iterations=int(row.get("fast_relax_blocks", 0))*10,
            blocks=int(row.get("fast_relax_blocks", 0)),
            model_time=float(row.get("fast_relax_model_time", 0.0)))

    return prescribed_q_event(
        state, setup, geom, evaluator, transport, quota_fraction_b,
        fast_relax_fn=fast_relax, state_metrics_fn=metrics,
        node_surface_mobility_m6_per_J_model_time=(
            corrected.EVENT_SURFACE_MOBILITY),
        event_restart=event_restart,
        accepted_state_callback=state_callback,
        accepted_progress_callback=progress_callback,
        initial_step_over_b=(
            EVENT_MINIMUM_INCREMENT_FRACTION_B
            if event_restart is not None
            and int(event_restart.get(
                "quasistatic_trial_rejections_total", 0)) > 0
            else 0.02),
        minimum_step_over_b=EVENT_MINIMUM_INCREMENT_FRACTION_B,
        maximum_step_over_b=0.02,
        precondition_restart_fast_manifold=(
            EVENT_RESTART_FAST_PRECONDITION),
        restart_precondition_fraction_of_qdot_limit=(
            EVENT_FAST_CLOCK_CONVERGENCE_FRACTION),
        restart_precondition_maximum_calls=EVENT_FAST_CLOCK_MAX_CALLS,
        **kwargs)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    if not EVENT_QUALIFICATION.exists():
        raise RuntimeError(
            "slow-GB adaptive event has not passed its deterministic timing gate; "
            "production launch is intentionally blocked")
    event_gate = json.loads(EVENT_QUALIFICATION.read_text())
    if (event_gate.get("status") not in {
            "PASS_PRODUCTION_EVENT_PROPAGATOR",
            "PASS_CURRENT_STATE_TRANSFER_TWO_EVENT_GATE"}
            or not event_gate.get("production_launch_authorized")):
        raise RuntimeError("configured deterministic event gate did not pass")
    calibration = json.loads(CALIBRATION.read_text())
    if calibration.get("status") != "PASS":
        raise RuntimeError("root penalty calibration did not pass")
    root_penalty = float(calibration["selected_root_penalty_eV"])
    if not math.isfinite(root_penalty) or root_penalty <= 0.0:
        raise RuntimeError("root penalty is invalid")

    renewal.OUT = OUT
    renewal.AVALANCHES_REQUESTED = AVALANCHES_REQUESTED
    renewal.CASE_BUILDER = campaign.padded_case_builder
    renewal.EVALUATOR_BUILDER = corrected.evaluator_builder
    renewal.RESERVOIR_STEP = renewal.closed_surface_diffusion_only
    renewal.PHASE_PROJECTOR = ConservativeBoundedPhaseProjector()
    renewal.PASSIVE_SCRATCH_BUILDER = CorrectedNumbaScratch
    renewal.PASSIVE_STEP = axisym_corrected_normalized_ownership_step_fast
    renewal.SYSTEM_MASS_BOUNDARY = "closed"
    renewal.TRANSPORT_BUILDER = slow_gb_transport
    renewal.EVENT_SURFACE_MOBILITY_MODEL = corrected.EVENT_SURFACE_MOBILITY
    renewal.EVENT_B_PF_MODEL = corrected.EVENT_B_PF
    renewal.EVENT_INTEGRATOR_REQUESTED = (
        renewal.DEFAULT_PRODUCTION_EVENT_INTEGRATOR)
    renewal.EVENT_BRANCH_TIME_INTEGRATOR = "adaptive_explicit"
    renewal.EVENT_SURFACE_ACTIVE_LENGTH_M = EVENT_SURFACE_ACTIVE_LENGTH_M
    renewal.EVENT_SURFACE_ACTIVE_LENGTH_SOURCE = (
        "qualified 15 nm local branch-redistribution length")
    renewal.QUASISTATIC_EVENT_CALL = quasistatic_event_call
    renewal.EVENT_MAX_INCREMENT_FRACTION_B = EVENT_MAX_INCREMENT_FRACTION_B
    renewal.SOLID_VOLUME_RELATIVE_TOLERANCE = (
        SOLID_VOLUME_RELATIVE_TOLERANCE)
    renewal.ROOT_BARRIER_PENALTY_EV = root_penalty
    renewal.ROOT_THRESHOLD_MULTIPLIER = ROOT_THRESHOLD_MULTIPLIER
    renewal.DELTA_G_STEP_EV = DESCENDANT_DELTA_G0_EV
    renewal.DESCENDANT_ESCALATION_AFTER_AVALANCHES = (
        DESCENDANT_ESCALATION_AFTER_AVALANCHES)
    renewal.DESCENDANT_ESCALATION_IF_ALL_S_BELOW = (
        DESCENDANT_ESCALATION_IF_ALL_S_BELOW)
    renewal.DESCENDANT_ESCALATION_NEW_DELTA_G_EV = (
        DESCENDANT_ESCALATION_NEW_DELTA_G_EV)
    renewal.FACILITATION_DECAY_ALPHA = (
        DESCENDANT_FACILITATION_DECAY_ALPHA)
    renewal.TAU_CORR_S = 9.0e-3
    renewal.MAX_DESCENDANTS_PER_AVALANCHE = MAX_DESCENDANTS_SAFETY
    renewal.STOP_IF_FIRST_AVALANCHE_HAS_NO_DESCENDANTS = False
    renewal.REUSED_SEED = REUSED_SEED
    renewal.CONTINUATION = CONTINUATION
    renewal.ENFORCE_FIRST_ROOT_STRESS_CHECK = False
    renewal.PERSIST_NUCLEATED_SOURCE_UNTIL_TRANSPORT = True
    renewal.ENFORCE_VOLUME_CUTOFF = False
    renewal.ENFORCE_RN_CUTOFF = False
    renewal.REQUIRE_UNIQUE_FIELD_TJ = True
    renewal.MINIMUM_BOUNDARY_CLEARANCE_W = 6.0
    renewal.MIN_EQUILIBRIUM_CAP_CLEARANCE_W = 10.0
    renewal.GEOMETRY_METRICS = corrected.geometry_metrics
    renewal.TRANSPORT_DIAGNOSTIC = corrected.transport_diagnostic
    renewal.ENFORCE_EVENT_ENERGY_DESCENT = False
    renewal.SPARSE_THERMO_PROBE = corrected.sparse_reaction_coordinate_probe
    renewal.SPARSE_THERMO_PROBE_DT_MODEL = 5.0
    renewal.EVENT_COPY_ENERGY_DIAGNOSTIC = copy_energy_diagnostic
    renewal.FINAL_MOVIE_RENDERER = render_constant_time_production_movies
    renewal.RUN_METADATA = dict(
        production_mode="slow_GB_large_avalanche_quasistatic_q_v2",
        deterministic_event_qualification=str(EVENT_QUALIFICATION.resolve()),
        parent_final_result=str((
            ROOT / "runs/pr_full_corrected_production_campaign_8avalanche_v7_"
            "recover_v6_latest/FINAL_RESULT_PARAMETER_RECORD.json").resolve()),
        intentional_parameter_changes_only=[
            "D_GB_m2_per_model_time",
            "event_branch_time_integrator",
            "solid_volume_relative_tolerance",
            "descendant_delta_G0_eV",
            "root_threshold_multiplier",
            "predeclared_descendant_escalation_rule"],
        D_GB_reference_m2_per_model_time=D_GB_REFERENCE,
        D_GB_target_m2_per_model_time=D_GB_TARGET,
        D_GB_scale_factor=D_GB_TARGET / D_GB_REFERENCE,
        target_events_per_avalanche=TARGET_EVENTS_PER_AVALANCHE,
        target_avalanche_relaxation_s=TARGET_AVALANCHE_RELAXATION_S,
        target_median_one_b_s=TARGET_MEDIAN_EVENT_S,
        event_time_integrator=(
            "auto selection from tau_GB/tau_surface; packet explicit fallback; "
            "direct adaptive prescribed-q continuation when R_tau>=10"),
        packet_resolved_slow_GB_event_called=False,
        descendant_clock_during_parent_event=(
            "frozen by physical model assumption: descendants become eligible "
            "only after the parent reaches and commits q=b"),
        competing_descendant_event_inside_parent_allowed=False,
        descendant_escalation_after_avalanches=(
            DESCENDANT_ESCALATION_AFTER_AVALANCHES),
        descendant_escalation_if_all_sizes_below=(
            DESCENDANT_ESCALATION_IF_ALL_S_BELOW),
        descendant_escalation_new_delta_G0_eV=(
            DESCENDANT_ESCALATION_NEW_DELTA_G_EV),
        **EXTRA_RUN_METADATA,
        root_clock_during_active_event="frozen after root commitment",
        rejected_event_trials_advance_time=False,
        rejected_event_trials_mutate_state=False,
        descendant_delta_G0_eV=DESCENDANT_DELTA_G0_EV,
        descendant_h1=1.0,
        descendant_h_decay_alpha=DESCENDANT_FACILITATION_DECAY_ALPHA,
        descendant_h_reset_after_child=False,
        avalanche_size_target_is_statistical_not_forced=True,
        prior_solid_volume_relative_tolerance=1.0e-8,
        root_barrier_changed=False,
        surface_mobility_changed=False,
        passive_PR_physics_changed=False,
        event_displacement_changed=False,
        source_lifetime_changed=False,
        OS_seed_drawn_once_before_dynamics=True,
        requested_complete_avalanches=AVALANCHES_REQUESTED)
    renewal.main()


if __name__ == "__main__":
    main()
