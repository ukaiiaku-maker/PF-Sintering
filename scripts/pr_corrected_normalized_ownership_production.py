#!/usr/bin/env python3
"""Five-avalanche closed PR production with corrected interfacial energy.

The equilibrium Young--Herring angle is metadata only.  Every dynamic stress,
chemical potential, transport evaluation, and stochastic rate uses the
instantaneous fields and the measured current TJ geometry.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_final_balanced as balanced  # noqa: E402
import pr_avalanche_renewal_five as renewal  # noqa: E402
import pr_coarsening_stochastic_two_event as stochastic_base  # noqa: E402
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.continuous_field_tj import ContinuousFieldTJTracker  # noqa: E402
from pf_sintering.corrected_axisym_dynamics import (  # noqa: E402
    CorrectedNumbaScratch,
    axisym_corrected_normalized_ownership_step_fast,
)
from pf_sintering.corrected_pr_thermodynamics import (  # noqa: E402
    corrected_mu_f,
    corrected_total_energy,
    make_corrected_state_evaluator,
)
from pf_sintering.experimental_pr_metrology import (  # noqa: E402
    _side_first_stresses,
    capillary_neck_particle_drive,
)
from pf_sintering.model_time_pr_event import _node_state  # noqa: E402
from pf_sintering.model_time_transport import (  # noqa: E402
    mullins_pf_coefficient_m4_per_model_time,
)
from pf_sintering.pr_experimental_geometry import (  # noqa: E402
    experimental_geometry_record,
)
from pr_closed_volume_no_event_t34 import outer_radius  # noqa: E402
from pr_closed_volume_passive_continuation import profile_metrics  # noqa: E402
from pr_passive_courant_ceiling_gate import (  # noqa: E402
    EPSILON1, R0_OVER_RCYL, build_eps040_case,
)
from pr_tj_node_coupling_gate import (  # noqa: E402
    B_PF_MODEL, SURFACE_MOBILITY_MODEL,
)
from pr_full_deterministic_cycle import source_volume_derivative_m2  # noqa: E402


OUT = ROOT / "runs/pr_corrected_normalized_ownership_closed_production_v2"
ROOT_FORMATION_PENALTY_EV = 1.090000
MINIMUM_BOUNDARY_CLEARANCE_W = 6.0
EVENT_SURFACE_MOBILITY = SURFACE_MOBILITY_MODEL * balanced.SURFACE_RATE_SCALE
EVENT_B_PF = mullins_pf_coefficient_m4_per_model_time(
    EVENT_SURFACE_MOBILITY, 1.0)


def energy_evaluator(state, setup):
    return corrected_total_energy(state, setup)


def chemical_potential_evaluator(state, setup):
    return corrected_mu_f(state, setup)


def build_case():
    geom, setup, target_volume, relative = build_eps040_case()
    if abs(relative) >= 1.0e-10:
        raise RuntimeError(f"epsilon=.40 volume mismatch {relative:+.9e}")
    expected_gamma_gb = 2.0 * setup["gamma_s"] * math.cos(math.radians(80.0))
    if not math.isclose(setup["gamma_gb"], expected_gamma_gb, rel_tol=1.0e-13):
        raise RuntimeError("production interfacial energies no longer imply 160 degrees")
    setup["local_metrology_window_m"] = 3.0 * setup["W"]
    setup["chemical_potential_evaluator"] = chemical_potential_evaluator
    setup["interfacial_energy_evaluator"] = energy_evaluator
    return geom, setup


def evaluator_builder(setup, geom):
    tracker = ContinuousFieldTJTracker(
        setup["z"], setup["r_c"], geom["z1"],
        6.0 * max(setup["dr"], setup["dz"]))
    return make_corrected_state_evaluator(setup, geom, tracker)


def geometry_metrics(state, setup, geom):
    radius = outer_radius(np.asarray(state[0]), setup["r_c"])
    finite = np.flatnonzero(np.isfinite(radius))
    if finite.size == 0:
        raise RuntimeError("f=0.5 exterior contour disappeared")
    return dict(
        **profile_metrics(radius, setup, geom),
        radial_margin_m=float(setup["r_f"][-1] - np.nanmax(radius)),
        left_z_margin_m=float(setup["z"][finite[0]] - 0.5 * setup["dz"]),
        right_z_margin_m=float(
            setup["z"][-1] + 0.5 * setup["dz"] - setup["z"][finite[-1]]))


def transport_diagnostic(state, setup, geom, evaluator):
    record = evaluator(*state)
    _, node, coordinates = _node_state(
        record, state[0], setup["r_c"], setup["z"],
        renewal.TRANSPORT_BUILDER(geom), EVENT_SURFACE_MOBILITY, active=True)
    tracking = record["tj_tracking"]
    return dict(
        mu_bulk_Pa=float(record["mu_bulk_Pa"]),
        mu_GB_Pa=float(record["mu_GB_source_Pa"]),
        mu_TJ_local_Pa=float(record["mu_TJ_local_Pa"]),
        mu_TJ_local_negative_Pa=float(record["mu_TJ_local_negative_Pa"]),
        mu_TJ_local_positive_Pa=float(record["mu_TJ_local_positive_Pa"]),
        mu_node_Pa=float(node["mu_TJ_Pa"]),
        mu_TJ_Pa=float(node["mu_TJ_Pa"]),
        mu_TJ_OFF_Pa=float(node["mu_TJ_OFF_Pa"]),
        delta_mu_GB_minus_TJ_local_Pa=float(
            record["delta_mu_GB_minus_TJ_local_Pa"]),
        transport_affinity_Pa=float(node["transport_affinity_Pa"]),
        Vdot_GB_m3_per_model_time=float(
            node["Vdot_GB_m3_per_model_time"]),
        node_zero_storage_closure_relative=float(
            node["zero_storage_closure_relative"]),
        tj_candidate_count=int(tracking["candidate_count"]),
        tj_jump_m=float(tracking["jump_m"]),
        r_TJ_transport_m=float(coordinates["r_TJ_m"]),
        z_TJ_transport_m=float(coordinates["z_TJ_m"]),
        equilibrium_angle_correction_applied=0,
        herring_balance_imposed=0)


def sparse_reaction_coordinate_probe(state, setup, geom):
    """Copy-only infinitesimal, volume-conserving corrected-energy probe."""
    quota_over_b = 1.0e-5
    evaluator = evaluator_builder(setup, geom)
    transport = renewal.TRANSPORT_BUILDER(geom)
    energy_before = corrected_total_energy(state, setup)
    result = renewal.event_call(
        tuple(np.asarray(field).copy() for field in state),
        setup, geom, evaluator, transport, quota_over_b,
        explicit_max_fourth_order_courant=renewal.PRODUCTION_C4,
        surface_flux_mobility_m6_per_J_model_time=EVENT_SURFACE_MOBILITY,
        explicit_stability_B_m4_per_model_time=EVENT_B_PF)
    if not result[3]:
        return dict(
            reaction_probe_available=0,
            Fq_reaction_N=math.nan,
            delta_mu_q_Pa=math.nan,
            reaction_probe_reason=str(result[4].get("stop_reason")))
    energy_after = corrected_total_energy(result[:3], setup)
    dq = quota_over_b * renewal.B_EVENT_M
    force = -(energy_after - energy_before) / dq
    volume_derivative = source_volume_derivative_m2(state, setup, geom)
    return dict(
        reaction_probe_available=1,
        reaction_probe_q_over_b=quota_over_b,
        reaction_probe_copy_only=1,
        reaction_probe_state_mutated=0,
        reaction_probe_volume_conserving=1,
        reaction_probe_delta_G_J=energy_after - energy_before,
        Fq_reaction_N=force,
        dVsource_dq_m2=volume_derivative,
        delta_mu_q_Pa=force / volume_derivative,
        reaction_probe_event_C4=(
            math.nan
            if result[4].get("explicit_max_fourth_order_courant") is None
            else float(result[4]["explicit_max_fourth_order_courant"])),
        reaction_probe_event_integrator=str(
            result[4].get("branch_time_integrator", "unknown")))


def fixed_angle_call_path_audit() -> dict:
    """Hard gate: activation/transport callables may not contain fixed psi."""
    dynamic_callables = {
        "activation_stress": _side_first_stresses,
        "capillary_mu_diagnostic": capillary_neck_particle_drive,
        "geometry_measurement": experimental_geometry_record,
        "corrected_state_evaluator": make_corrected_state_evaluator,
        "production_measure": stochastic_base.measure,
        "passive_step": axisym_corrected_normalized_ownership_step_fast,
    }
    forbidden = ("psi_reference_deg", "160.0")
    rows = {}
    for name, function in dynamic_callables.items():
        source = inspect.getsource(function)
        hits = [token for token in forbidden if token in source]
        rows[name] = dict(module=function.__module__, forbidden_hits=hits)
        if hits:
            raise RuntimeError(
                f"fixed-angle token in dynamic production callable {name}: {hits}")
    return dict(
        passed=True,
        rule=("equilibrium angle is diagnostic metadata only; instantaneous "
              "measured branch angles feed local stress"),
        dynamic_callables=rows,
        diagnostic_only_fixed_angle_path=(
            "Eq4 Fourier stability metadata in legacy observe; excluded from "
            "stress, hazards, transport, event mechanics, and chemical potentials"))


def main():
    expected_event_B = B_PF_MODEL * balanced.SURFACE_RATE_SCALE
    if not math.isclose(EVENT_B_PF, expected_event_B, rel_tol=1.0e-14):
        raise AssertionError("qualified event mobility/B mapping changed")
    angle_audit = fixed_angle_call_path_audit()

    renewal.OUT = OUT
    renewal.AVALANCHES_REQUESTED = 5
    renewal.CASE_BUILDER = build_case
    renewal.EVALUATOR_BUILDER = evaluator_builder
    renewal.RESERVOIR_STEP = renewal.closed_surface_diffusion_only
    renewal.PHASE_PROJECTOR = ConservativeBoundedPhaseProjector()
    renewal.PASSIVE_SCRATCH_BUILDER = CorrectedNumbaScratch
    renewal.PASSIVE_STEP = axisym_corrected_normalized_ownership_step_fast
    renewal.SYSTEM_MASS_BOUNDARY = "closed"
    renewal.EVENT_SURFACE_MOBILITY_MODEL = EVENT_SURFACE_MOBILITY
    renewal.EVENT_B_PF_MODEL = EVENT_B_PF
    renewal.ROOT_BARRIER_PENALTY_EV = ROOT_FORMATION_PENALTY_EV
    renewal.REUSED_SEED = None
    renewal.ENFORCE_FIRST_ROOT_STRESS_CHECK = False
    renewal.PERSIST_NUCLEATED_SOURCE_UNTIL_TRANSPORT = True
    renewal.ENFORCE_VOLUME_CUTOFF = False
    renewal.ENFORCE_RN_CUTOFF = False
    renewal.REQUIRE_UNIQUE_FIELD_TJ = True
    renewal.MINIMUM_BOUNDARY_CLEARANCE_W = MINIMUM_BOUNDARY_CLEARANCE_W
    renewal.GEOMETRY_METRICS = geometry_metrics
    renewal.TRANSPORT_DIAGNOSTIC = transport_diagnostic
    renewal.ENFORCE_EVENT_ENERGY_DESCENT = True
    renewal.SPARSE_THERMO_PROBE = sparse_reaction_coordinate_probe
    renewal.SPARSE_THERMO_PROBE_DT_MODEL = 5.0
    renewal.RUN_METADATA = dict(
        production_mode=(
            "full_closed_volume_corrected_normalized_ownership_avalanche_renewal"),
        interfacial_formulation="normalized_ownership_corrected",
        rejected_background_subtraction_used=False,
        epsilon1=EPSILON1, epsilon2=0.0,
        R0_over_Rcyl=R0_OVER_RCYL,
        lambda_over_R_cyl=8.8857658763,
        root_effective_zero_stress_barrier_eV=6.4546516047,
        root_stress_target_or_forced_trigger=None,
        root_penalty_eV=ROOT_FORMATION_PENALTY_EV,
        local_stress_window_over_W=3.0,
        activation_angle_source="instantaneous measured free-surface branches",
        equilibrium_angle_dynamic_boundary_condition=False,
        herring_balance_imposed=False,
        herring_residual_recorded=True,
        chemical_potential_equilibrium_angle_correction=False,
        fixed_angle_call_path_audit=angle_audit,
        ownership_dynamics=(
            "phi_dot=-(M_eta/2)*f*g_phi; eta1=f*phi; eta2=f*(1-phi)"),
        gb_mu_sampling=(
            "GB center plane radial-area mean ending 3W before TJ"),
        tj_local_mu_sampling="side-resolved 1W-to-3W surface mean",
        node_role="zero-storage kinetic flux partition only",
        unique_continuous_field_TJ_required=True,
        old_particle_volume_validity_cutoff_disabled=True,
        old_35nm_neck_cutoff_disabled=True)
    renewal.main()


if __name__ == "__main__":
    main()
