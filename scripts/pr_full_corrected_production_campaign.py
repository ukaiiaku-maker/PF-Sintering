#!/usr/bin/env python3
"""Launch the frozen corrected PR avalanche-renewal production campaign.

The only pre-dynamics geometry operation is exact outer-radial vacuum padding.
Existing fields and coordinates are copied bit-for-bit; the analytical
equilibrium envelopes are diagnostics and are never imposed on the fields.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_renewal_five as renewal  # noqa: E402
import pr_corrected_normalized_ownership_production as corrected  # noqa: E402
from pr_constant_time_movie_rendering import (  # noqa: E402
    render_constant_time_production_movies,
)
from pr_native_postevent_one_avalanche_replay import (  # noqa: E402
    copy_energy_diagnostic,
)
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.corrected_axisym_dynamics import (  # noqa: E402
    CorrectedNumbaScratch,
    axisym_corrected_normalized_ownership_step_fast,
)
from pf_sintering.radial_vacuum_padding import (  # noqa: E402
    pad_case_radially_with_vacuum,
)


CALIBRATION = ROOT / (
    "runs/pr_root_analytical_hazard_calibration_56p5/"
    "root_penalty_calibration.json"
)
OUT = ROOT / "runs/pr_full_corrected_production_campaign_8avalanche_v3"
TARGET_RADIAL_FACE_M = 325.0e-9
AVALANCHES_REQUESTED = 8
MAX_DESCENDANTS_SAFETY = 25
PRESERVED_PRE_DYNAMICS_OS_SEED = 174663140833073965671601611669420566972


def padded_case_builder():
    geom, setup = corrected.build_case()
    padded_geom, padded_setup, diagnostics = pad_case_radially_with_vacuum(
        geom, setup, target_outer_face_m=TARGET_RADIAL_FACE_M)
    renewal.RADIAL_PADDING_DIAGNOSTICS = diagnostics
    return padded_geom, padded_setup


def main() -> None:
    if not CALIBRATION.exists():
        raise FileNotFoundError(
            f"authoritative root calibration has not completed: {CALIBRATION}")
    calibration = json.loads(CALIBRATION.read_text())
    if calibration.get("status") != "PASS":
        raise RuntimeError("root penalty calibration did not pass")
    penalty = float(calibration["selected_root_penalty_eV"])
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise RuntimeError("calibrated root penalty is invalid")
    print("FROZEN ROOT CALIBRATION " + json.dumps(dict(
        target_statistic=calibration["target_statistic"],
        target_sigma_MPa=calibration["target_sigma_MPa"],
        H_fit_at_target=calibration["H_fit_at_target"],
        selected_root_penalty_eV=penalty,
        no_stress_trigger=True,
        penalty_frozen_for_entire_trajectory=True,
    ), sort_keys=True), flush=True)

    renewal.OUT = OUT
    renewal.AVALANCHES_REQUESTED = AVALANCHES_REQUESTED
    renewal.CASE_BUILDER = padded_case_builder
    renewal.EVALUATOR_BUILDER = corrected.evaluator_builder
    renewal.RESERVOIR_STEP = renewal.closed_surface_diffusion_only
    renewal.PHASE_PROJECTOR = ConservativeBoundedPhaseProjector()
    renewal.PASSIVE_SCRATCH_BUILDER = CorrectedNumbaScratch
    renewal.PASSIVE_STEP = axisym_corrected_normalized_ownership_step_fast
    renewal.SYSTEM_MASS_BOUNDARY = "closed"
    renewal.EVENT_SURFACE_MOBILITY_MODEL = corrected.EVENT_SURFACE_MOBILITY
    renewal.EVENT_B_PF_MODEL = corrected.EVENT_B_PF
    renewal.ROOT_BARRIER_PENALTY_EV = penalty
    renewal.DELTA_G_STEP_EV = 0.300000
    renewal.FACILITATION_DECAY_ALPHA = 0.6
    renewal.TAU_CORR_S = 9.0e-3
    renewal.MAX_DESCENDANTS_PER_AVALANCHE = MAX_DESCENDANTS_SAFETY
    renewal.STOP_IF_FIRST_AVALANCHE_HAS_NO_DESCENDANTS = False
    renewal.REUSED_SEED = PRESERVED_PRE_DYNAMICS_OS_SEED
    renewal.CONTINUATION = None
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
        production_mode="full_corrected_8avalanche_campaign",
        calibrated_root_penalty_source=str(CALIBRATION.resolve()),
        root_penalty_target_is_statistical_not_trigger=True,
        OS_seed_drawn_once_before_dynamics=True,
        OS_seed_source=(
            "preserved from pre-dynamics manifest-assembly stop in sibling "
            "production directory"),
        source_alive_semantics_frozen=True,
        event_transport_active_semantics_frozen=True,
        immediate_child_at_exact_hazard_crossing=True,
        descendant_delta_G0_eV=0.300000,
        descendant_h1=1.0,
        descendant_h_decay_alpha=0.6,
        descendant_h_reset_after_child=False,
        tau_source_is_one_b_sweep_time=False,
        descendant_safety_cap=MAX_DESCENDANTS_SAFETY,
        descendant_safety_cap_role="blocker only; not physical extinction",
        requested_complete_avalanches=AVALANCHES_REQUESTED,
        long_movie_constant_time_step_s=1.0e-3,
        exact_event_states_stored_separately=True,
        radial_target_requested_m=TARGET_RADIAL_FACE_M,
        radial_padding_vacuum_only=True,
        equilibrium_cap_geometry_diagnostic_only=True,
        equilibrium_angle_imposed_dynamically=False,
        live_state_profile_equilibration=False,
        copy_only_profile_equilibration=True,
        energy_ledgers="G_PF_native, G_PF_equilibrated_copy, G_gamma",
    )
    renewal.main()


if __name__ == "__main__":
    main()
