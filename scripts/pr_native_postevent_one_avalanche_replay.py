#!/usr/bin/env python3
"""Replay one realized avalanche with native live fields and copy-only ledgers.

The fixed-geometry profile equilibration is deliberately restricted to deep
copies passed through ``EVENT_COPY_ENERGY_DIAGNOSTIC``.  The renewal driver
continues exclusively from the native output of each qualified one-b event.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_renewal_five as renewal  # noqa: E402
import pr_corrected_normalized_ownership_production as corrected  # noqa: E402
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.corrected_axisym_dynamics import (  # noqa: E402
    CorrectedNumbaScratch,
    axisym_corrected_normalized_ownership_step_fast,
)
from pf_sintering.corrected_pr_thermodynamics import (  # noqa: E402
    corrected_total_energy,
)
from pf_sintering.fixed_geometry_profile_equilibration import (  # noqa: E402
    equilibrate_fixed_geometry_profiles,
    equilibrate_ownership_variational_fixed_geometry,
)
from pf_sintering.pr_experimental_geometry import (  # noqa: E402
    experimental_geometry_record,
)


PARENT = ROOT / (
    "runs/pr_corrected_normalized_ownership_closed_production_"
    "rootpenalty1p000_continuation"
)
CHECKPOINT = PARENT / "checkpoints/avalanche1_before_root.npz"
OUT = ROOT / (
    "runs/pr_corrected_normalized_ownership_closed_production_"
    "rootpenalty1p000_native_copyledger_avalanche1_replay"
)
SEED = 143123746821450290773542065133003127952
H_THRESHOLD = 0.3410154114713397
ROOT_PENALTY_EV = 1.000000
DELTA_G_STEP_EV = 0.300000
FACILITATION_DECAY_ALPHA = 0.6
MAX_DESCENDANTS_SAFETY = 15


def _digest(state) -> str:
    digest = hashlib.sha256()
    for field in state:
        digest.update(np.ascontiguousarray(field).view(np.uint8))
    return digest.hexdigest()


def _equilibrated_copy_energy(state, *, z_tj_m: float, setup: dict):
    surface_copy, surface_diagnostics = equilibrate_fixed_geometry_profiles(
        state, setup, z_tj_m=z_tj_m,
        equilibrate_f=True, equilibrate_phi=False)
    equilibrated_copy, ownership_diagnostics = (
        equilibrate_ownership_variational_fixed_geometry(surface_copy, setup))
    return (
        corrected_total_energy(equilibrated_copy, setup),
        surface_diagnostics,
        ownership_diagnostics,
    )


def copy_energy_diagnostic(pre_state, post_state, *, setup, geom,
                           pre_row, post_row):
    """Return scalar energies from detached copies; never return a state."""
    del geom
    pre_digest = _digest(pre_state)
    post_digest = _digest(post_state)
    print("COPY-ONLY ENERGY LEDGER: equilibrating detached pre-event copy",
          flush=True)
    pre_energy, pre_surface, pre_ownership = _equilibrated_copy_energy(
        pre_state, z_tj_m=float(pre_row["z_TJ_m"]), setup=setup)
    print("COPY-ONLY ENERGY LEDGER: equilibrating detached post-event copy",
          flush=True)
    post_energy, post_surface, post_ownership = _equilibrated_copy_energy(
        post_state, z_tj_m=float(post_row["z_TJ_m"]), setup=setup)
    pre_geometry, _ = experimental_geometry_record(
        pre_state, setup, z_tj_m=float(pre_row["z_TJ_m"]),
        r_tj_m=float(pre_row["r_n_m"]),
        local_window_m=float(setup["local_metrology_window_m"]))
    post_geometry, _ = experimental_geometry_record(
        post_state, setup, z_tj_m=float(post_row["z_TJ_m"]),
        r_tj_m=float(post_row["r_n_m"]),
        local_window_m=float(setup["local_metrology_window_m"]))
    inputs_unchanged = (
        _digest(pre_state) == pre_digest and _digest(post_state) == post_digest)
    if not inputs_unchanged:
        raise RuntimeError("copy-only equilibration mutated its input copy")
    ledger = dict(
        G_PF_equilibrated_copy_before_J=float(pre_energy),
        G_PF_equilibrated_copy_after_J=float(post_energy),
        delta_G_PF_equilibrated_copy_J=float(post_energy - pre_energy),
        G_gamma_geometry_before_J=float(pre_geometry["G_gamma_J"]),
        G_gamma_geometry_after_J=float(post_geometry["G_gamma_J"]),
        delta_G_gamma_geometry_J=float(
            post_geometry["G_gamma_J"]-pre_geometry["G_gamma_J"]),
        copy_equilibration_inputs_unchanged=1,
        copy_equilibration_advanced_model_time=0,
        copy_pre_ownership_iterations=int(
            pre_ownership.get("accepted_iterations", -1)),
        copy_post_ownership_iterations=int(
            post_ownership.get("accepted_iterations", -1)),
        copy_pre_volume_relative_error=float(
            pre_surface["volume_relative_error"]),
        copy_post_volume_relative_error=float(
            post_surface["volume_relative_error"]),
    )
    print("COPY-ONLY ENERGY REPORT " + json.dumps(ledger, sort_keys=True),
          flush=True)
    return ledger


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite replay directory {OUT}")
    with np.load(CHECKPOINT) as saved:
        t_model = float(saved["t_model"])
        hazard = float(saved["H"])
        threshold = float(saved["Hstar"])
    if threshold != H_THRESHOLD or hazard < threshold:
        raise RuntimeError("checkpoint is not the preserved realized root state")
    parent_manifest = json.loads((PARENT / "launch_manifest.json").read_text())

    renewal.OUT = OUT
    renewal.AVALANCHES_REQUESTED = 1
    renewal.CASE_BUILDER = corrected.build_case
    renewal.EVALUATOR_BUILDER = corrected.evaluator_builder
    renewal.RESERVOIR_STEP = renewal.closed_surface_diffusion_only
    renewal.PHASE_PROJECTOR = ConservativeBoundedPhaseProjector()
    renewal.PASSIVE_SCRATCH_BUILDER = CorrectedNumbaScratch
    renewal.PASSIVE_STEP = axisym_corrected_normalized_ownership_step_fast
    renewal.SYSTEM_MASS_BOUNDARY = "closed"
    renewal.EVENT_SURFACE_MOBILITY_MODEL = corrected.EVENT_SURFACE_MOBILITY
    renewal.EVENT_B_PF_MODEL = corrected.EVENT_B_PF
    renewal.ROOT_BARRIER_PENALTY_EV = ROOT_PENALTY_EV
    renewal.DELTA_G_STEP_EV = DELTA_G_STEP_EV
    renewal.FACILITATION_DECAY_ALPHA = FACILITATION_DECAY_ALPHA
    renewal.MAX_DESCENDANTS_PER_AVALANCHE = MAX_DESCENDANTS_SAFETY
    renewal.STOP_IF_FIRST_AVALANCHE_HAS_NO_DESCENDANTS = False
    renewal.REUSED_SEED = SEED
    renewal.ENFORCE_FIRST_ROOT_STRESS_CHECK = False
    renewal.PERSIST_NUCLEATED_SOURCE_UNTIL_TRANSPORT = True
    renewal.ENFORCE_VOLUME_CUTOFF = False
    renewal.ENFORCE_RN_CUTOFF = False
    renewal.REQUIRE_UNIQUE_FIELD_TJ = True
    renewal.MINIMUM_BOUNDARY_CLEARANCE_W = corrected.MINIMUM_BOUNDARY_CLEARANCE_W
    renewal.GEOMETRY_METRICS = corrected.geometry_metrics
    renewal.TRANSPORT_DIAGNOSTIC = corrected.transport_diagnostic
    renewal.ENFORCE_EVENT_ENERGY_DESCENT = False
    renewal.SPARSE_THERMO_PROBE = corrected.sparse_reaction_coordinate_probe
    renewal.SPARSE_THERMO_PROBE_DT_MODEL = 5.0
    renewal.EVENT_COPY_ENERGY_DIAGNOSTIC = copy_energy_diagnostic
    renewal.CONTINUATION = dict(
        parent_checkpoint=str(CHECKPOINT.resolve()),
        t_model=t_model,
        root_hazard=hazard,
        root_threshold=threshold,
        root_already_nucleated=True,
        total_completed=0,
        vp_cycle0_m3=2.42608592212333e-20,
        solid_volume_initial_m3=float(parent_manifest["solid_volume_initial_m3"]),
    )
    renewal.RUN_METADATA = dict(
        production_mode="native_postevent_one_avalanche_replay",
        parent_run=str(PARENT.resolve()),
        parent_checkpoint=str(CHECKPOINT.resolve()),
        preserved_root_time_model=t_model,
        preserved_root_hazard=hazard,
        preserved_root_threshold=threshold,
        preserved_OS_seed=SEED,
        root_barrier_unchanged=True,
        RNG_streams_reconstructed_from_preserved_seed=True,
        first_descendant_threshold_preserved=True,
        qualified_repaired_one_b_event_operator=True,
        live_state_profile_equilibration=False,
        post_event_dynamics_state="native qualified event output",
        copy_only_profile_equilibration=True,
        copy_only_energy_definition=(
            "identical fixed-contour surface reconstruction plus variational "
            "ownership equilibration applied independently to detached pre/post "
            "copies"),
        energy_ledgers=(
            "G_PF_native, G_PF_equilibrated_copy, G_gamma"),
        descendant_delta_G0_eV=DELTA_G_STEP_EV,
        descendant_h1=1.0,
        descendant_h_decay_alpha=FACILITATION_DECAY_ALPHA,
        descendant_h_reset_after_child=False,
        descendant_correlation_window_s=renewal.TAU_CORR_S,
        safety_cap_role="blocker only; natural extinction required",
        descendant_safety_cap=MAX_DESCENDANTS_SAFETY,
        no_barrier_retuning=True,
    )
    renewal.main()


if __name__ == "__main__":
    main()
