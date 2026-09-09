"""Asymmetric zero-storage TJ-node and deterministic t=1 event gate.

This is intentionally a single-state, central-timescale experiment.  It does
not run the historical 3x3 matrix, evaluate a nucleation hazard, convert model
time to seconds, or use any material-specific diffusivity.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(__file__))

from m16a_gb_benchmark import measure_R_of_z
from m16g_pr_derived_particle_asperity import find_gb_trough
from pf_sintering.axisym import axisym_mu_f_gb
from pf_sintering.axisym_branch_boundary_flux import (
    AxisymmetricSurfaceBranch,
    extract_axisymmetric_surface_branches,
)
from pf_sintering.model_time_pr_event import (
    representation_corrected_model_time_boundary_flux_event,
)
from pf_sintering.model_time_transport import ModelTimeGBTransport
from pf_sintering.pr_stress_metrology import pf_contour_estimators
from pf_sintering.rigid_rbm_deposition import (
    _axisym_weighted_sum,
    normalized_axisym_support,
)
from pf_sintering.tj_transport_node import (
    solve_model_time_tj_node,
    solve_prescribed_rate_tj_node,
)
from pr_concurrent_tj_transport_gate import SOURCE_STATES, load_state
from pr_event_metrology_resolution_audit import B_M
from pr_tj_activation_gate import make_setup


OUT_DIR = "/private/tmp/pr_tj_node_coupling"
OUT_JSON = os.path.join(OUT_DIR, "tj_node_coupling_gate.json")
OUT_TRACE = os.path.join(OUT_DIR, "adaptive_explicit_trace.npz")
# Frozen central model-time pair.  D_GB is a model-time demonstration
# coefficient, not a physical/material diffusivity.
D_GB_MODEL = 5.228329357301188e-11
SURFACE_MOBILITY_MODEL = 2.5e-30
B_PF_MODEL = 5.625e-30
K_B = 1.380649e-23
TEMPERATURE_K = 1200.0
ATOMIC_VOLUME_M3 = 1e-29
TAU_EX_MODEL = 0.0

LOCAL_QUOTA_FRACTION_B = 1e-4
LONG_REFERENCE_QUOTA_FRACTION_B = float(
    os.environ.get("PR_TJ_NODE_REFERENCE_QUOTA_B", "0.05"))
EXPLICIT_COURANT_LIMITS = (0.15, 0.05, 0.025)
LONG_EXPLICIT_COURANT = 0.15
IMPLICIT_LOCAL_PACKET_FRACTIONS_B = (1e-6, 5e-7, 2.5e-7)
IMPLICIT_SCOUT_PACKET_FRACTIONS_B = (0.0025, 0.001, 0.0005, 0.00025, 0.0001)


def make_evaluator(setup, geom, fixed_z_tj=None, tj_tracker=None):
    """Return only the instantaneous quantities used by the TJ node."""
    Z = setup["z"][:, None]
    RC = setup["r_c"][None, :]

    def evaluate(f, particle, neighbor):
        radius = measure_R_of_z(f, setup["r_c"])
        tracker_record = None
        if tj_tracker is not None:
            tracker_record = tj_tracker.locate(f, particle, neighbor)
            z_tj = float(tracker_record["z_TJ_m"])
            r_tj = float(tracker_record["r_TJ_m"])
        elif fixed_z_tj is None:
            z_tj, _ = find_gb_trough(
                radius, setup["z"], geom["z1"], lam=setup["lam"])
            r_tj = None
        else:
            z_tj = float(fixed_z_tj)
            r_tj = None
        contour = pf_contour_estimators(
            radius, setup["z"], z_tj, gamma_s=setup["gamma_s"],
            psi_reference_deg=160.0)
        mu = axisym_mu_f_gb(
            f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
            setup["dz"], setup["r_c"], setup["r_f"], bc_z="noflux")
        if r_tj is None:
            r_tj = float(contour["r_neck"])
        contact_area = math.pi * r_tj * r_tj
        gb_raw = (
            np.maximum(particle * neighbor, 0.0)
            * (np.abs(Z - z_tj) <= 2.0 * setup["W"])
            * (RC <= r_tj + setup["W"]))
        gb_support = normalized_axisym_support(gb_raw, setup["r_c"])
        return dict(
            mu_GB_source_Pa=float(
                _axisym_weighted_sum(mu * gb_support, setup["r_c"])),
            contact_area_m2=float(contact_area),
            mu_field_Pa=mu,
            z_TJ_m=float(z_tj),
            r_TJ_m=float(r_tj),
            tj_tracking=(None if tracker_record is None else tracker_record),
            X_Sigma_diffuse_MPa=float(
                contour["local_reference"]["sigma_3D_local_MPa"]))

    return evaluate


def manufactured_branch(side, radius, mu_first):
    return AxisymmetricSurfaceBranch(
        side=side,
        row_indices=np.arange(4),
        s_centers_m=np.arange(4) + 0.5,
        s_faces_m=np.arange(5.0),
        r_centers_m=np.full(4, radius),
        r_faces_m=np.full(5, radius),
        mu_Pa=np.array([
            mu_first, mu_first - 0.1, mu_first - 0.2, mu_first - 0.3]))


def manufactured_gate():
    branches = (
        manufactured_branch("positive", 2.0, 3.0),
        manufactured_branch("negative", 1.0, 1.0))
    mobility = 1.0 / (4.0 * math.pi)
    prescribed = solve_prescribed_rate_tj_node(branches, mobility, 5.0)
    transport = ModelTimeGBTransport(
        x_d_m=1.0, kB_J_per_K=1.0, temperature_K=1.0,
        atomic_volume_m3=1.0, b_m=1.0,
        D_gb_m2_per_model_time=1.0)
    coupled = solve_model_time_tj_node(
        branches, mobility, mu_GB_Pa=10.0, contact_area_m2=4.0,
        transport=transport)
    off = solve_model_time_tj_node(
        branches, mobility, mu_GB_Pa=10.0, contact_area_m2=4.0,
        transport=transport, gb_path_active=False)
    passed = bool(
        math.isclose(prescribed["mu_TJ_Pa"], 4.0, abs_tol=1e-14)
        and math.isclose(coupled["mu_TJ_Pa"], 47.0 / 7.0, abs_tol=1e-14)
        and math.isclose(off["mu_TJ_Pa"], 7.0 / 3.0, abs_tol=1e-14)
        and max(abs(row["zero_storage_closure_relative"])
                for row in (prescribed, coupled, off)) < 1e-14)
    return dict(
        passed=passed,
        analytical=dict(
            prescribed_mu_TJ_Pa=4.0,
            coupled_mu_TJ_Pa=47.0 / 7.0,
            sink_OFF_mu_TJ_Pa=7.0 / 3.0),
        computed=dict(
            prescribed_mu_TJ_Pa=prescribed["mu_TJ_Pa"],
            prescribed_branch_rates=prescribed[
                "branch_volume_rates_m3_per_model_time"],
            coupled_mu_TJ_Pa=coupled["mu_TJ_Pa"],
            coupled_branch_rate_over_net_GB=coupled[
                "branch_rate_over_net_GB"],
            sink_OFF_mu_TJ_Pa=off["mu_TJ_Pa"],
            sink_OFF_branch_rates=off[
                "branch_volume_rates_m3_per_model_time"]))


def node_at_state(state, evaluator, setup, transport, *, active):
    record = evaluator(*state)
    branches = extract_axisymmetric_surface_branches(
        state[0], record["mu_field_Pa"], setup["r_c"], setup["z"],
        record["z_TJ_m"], record["r_TJ_m"])
    node = solve_model_time_tj_node(
        branches, SURFACE_MOBILITY_MODEL,
        record["mu_GB_source_Pa"], record["contact_area_m2"],
        transport, gb_path_active=active)
    return record, node


def compact_node(node):
    return dict(
        mu_GB_Pa=node["mu_GB_Pa"],
        mu_TJ_Pa=node["mu_TJ_Pa"],
        mu_TJ_OFF_Pa=node["mu_TJ_OFF_Pa"],
        transport_affinity_Pa=node["transport_affinity_Pa"],
        Vdot_GB_m3_per_model_time=node["Vdot_GB_m3_per_model_time"],
        branch_rate_over_net_GB=node["branch_rate_over_net_GB"],
        branch_volume_rates_m3_per_model_time=(
            node["branch_volume_rates_m3_per_model_time"]),
        branch_first_cell_mu_Pa={
            side: node["branch"][side]["first_cell_mu_Pa"]
            for side in ("positive", "negative")},
        branch_conductance_m3_per_Pa_model_time={
            side: node["branch"][side][
                "conductance_m3_per_Pa_model_time"]
            for side in ("positive", "negative")},
        zero_storage_closure_relative=node["zero_storage_closure_relative"])


def event_summary(event):
    diag = event[4]
    final = diag.get("final_node") or diag.get("stop_state")
    sampled = diag.get("packets", [])
    return dict(
        completed=bool(event[3]),
        progress_over_b=float(diag["event_progress_over_b"]),
        n_subincrements=int(diag["n_subincrements"]),
        event_time_model=float(diag["event_time_model"]),
        stop_reason=diag.get("stop_reason"),
        numerical_failure=bool(diag.get("numerical_failure", False)),
        branch_time_integrator=diag.get("branch_time_integrator"),
        final_mu_GB_Pa=final.get("mu_GB_Pa", final.get("mu_GB_source_Pa")),
        final_mu_TJ_Pa=final.get("mu_TJ_Pa"),
        final_mu_TJ_OFF_Pa=final.get("mu_TJ_OFF_Pa"),
        final_transport_affinity_Pa=final.get("transport_affinity_Pa"),
        final_branch_rate_over_net_GB=final.get("branch_rate_over_net_GB"),
        maximum_sampled_field_closure_relative=max(
            (abs(row["branch_boundary_flux"]["closure_relative"])
             for row in sampled), default=0.0),
        maximum_sampled_total_mass_relative_error=max(
            (abs(row["mass_relative_error_after_flux"])
             for row in sampled), default=0.0))


def run_event(state, setup, geom, evaluator, transport, *, packet_fraction_b,
              quota_fraction_b, implicit=False, explicit_courant=None,
              callback=None, diagnostic_stride=1):
    return representation_corrected_model_time_boundary_flux_event(
        *state, transport,
        dr=setup["dr"], dz=setup["dz"], r_c=setup["r_c"], z=setup["z"],
        GB_z_hint=geom["z1"], W=setup["W"], state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=SURFACE_MOBILITY_MODEL,
        max_increment_fraction_b=packet_fraction_b,
        event_quota_m=quota_fraction_b * B_M,
        max_subincrements=1_000_000,
        implicit_mullins_B_m4_per_model_time=(B_PF_MODEL if implicit else None),
        explicit_stability_B_m4_per_model_time=(
            B_PF_MODEL if explicit_courant is not None else None),
        explicit_max_fourth_order_courant=(
            0.05 if explicit_courant is None else explicit_courant),
        branch_mass_closure_relative_tolerance=1e-5,
        packet_diagnostic_stride=diagnostic_stride,
        accepted_step_callback=callback)


def relative_error(value, reference):
    return abs(value - reference) / max(abs(reference), 1e-300)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.monotonic()
    geom, setup = make_setup(10.0, 1.25)
    evaluator = make_evaluator(setup, geom)
    state_t1 = load_state(np.load(SOURCE_STATES), "coarse", "t1")
    transport = ModelTimeGBTransport(
        x_d_m=geom["R_z1"] / 2.0,
        kB_J_per_K=K_B,
        temperature_K=TEMPERATURE_K,
        atomic_volume_m3=ATOMIC_VOLUME_M3,
        b_m=B_M,
        D_gb_m2_per_model_time=D_GB_MODEL,
        tau_ex_model=TAU_EX_MODEL)
    manufactured = manufactured_gate()

    _, t1_off = node_at_state(
        state_t1, evaluator, setup, transport, active=False)
    _, t1_active = node_at_state(
        state_t1, evaluator, setup, transport, active=True)

    local_explicit = {}
    for courant in EXPLICIT_COURANT_LIMITS:
        label = f"C_{courant:g}"
        print("local adaptive explicit", label, flush=True)
        event = run_event(
            state_t1, setup, geom, evaluator, transport,
            packet_fraction_b=0.0025,
            quota_fraction_b=LOCAL_QUOTA_FRACTION_B,
            explicit_courant=courant)
        local_explicit[label] = event_summary(event)

    explicit_reference = local_explicit["C_0.025"]
    implicit_local = {}
    for packet in IMPLICIT_LOCAL_PACKET_FRACTIONS_B:
        label = f"packet_{packet:g}b"
        print("local implicit", label, flush=True)
        event = run_event(
            state_t1, setup, geom, evaluator, transport,
            packet_fraction_b=packet,
            quota_fraction_b=LOCAL_QUOTA_FRACTION_B,
            implicit=True)
        summary = event_summary(event)
        summary["affinity_relative_error_vs_explicit_C0p025"] = relative_error(
            summary["final_transport_affinity_Pa"],
            explicit_reference["final_transport_affinity_Pa"])
        implicit_local[label] = summary

    implicit_scout = {}
    for packet in IMPLICIT_SCOUT_PACKET_FRACTIONS_B:
        label = f"packet_{packet:g}b"
        print("implicit scout", label, flush=True)
        event = run_event(
            state_t1, setup, geom, evaluator, transport,
            packet_fraction_b=packet, quota_fraction_b=0.05,
            implicit=True)
        implicit_scout[label] = event_summary(event)

    trace = {key: [] for key in (
        "q_over_b", "event_time_model", "mu_GB_Pa", "mu_TJ_Pa",
        "mu_TJ_OFF_Pa", "transport_affinity_Pa",
        "positive_branch_rate_over_net_GB",
        "negative_branch_rate_over_net_GB",
        "positive_branch_rate_m3_per_model_time",
        "negative_branch_rate_m3_per_model_time",
        "positive_first_cell_mu_Pa", "negative_first_cell_mu_Pa",
        "node_closure_relative", "field_closure_relative",
        "total_mass_relative_error", "explicit_fourth_order_courant")}
    cumulative_time = 0.0
    max_field_closure = 0.0
    max_total_mass_error = 0.0

    def record(packet):
        nonlocal cumulative_time, max_field_closure, max_total_mass_error
        node = packet["node_before"]
        cumulative_time += packet["transport_dt_model"]
        closure = abs(packet["branch_boundary_flux"]["closure_relative"])
        mass_error = abs(packet["mass_relative_error_after_flux"])
        max_field_closure = max(max_field_closure, closure)
        max_total_mass_error = max(max_total_mass_error, mass_error)
        values = dict(
            q_over_b=packet["q_end_over_b"],
            event_time_model=cumulative_time,
            mu_GB_Pa=node["mu_GB_Pa"],
            mu_TJ_Pa=node["mu_TJ_Pa"],
            mu_TJ_OFF_Pa=node["mu_TJ_OFF_Pa"],
            transport_affinity_Pa=node["transport_affinity_Pa"],
            positive_branch_rate_over_net_GB=(
                node["branch_rate_over_net_GB"]["positive"]),
            negative_branch_rate_over_net_GB=(
                node["branch_rate_over_net_GB"]["negative"]),
            positive_branch_rate_m3_per_model_time=(
                node["branch_volume_rates_m3_per_model_time"]["positive"]),
            negative_branch_rate_m3_per_model_time=(
                node["branch_volume_rates_m3_per_model_time"]["negative"]),
            positive_first_cell_mu_Pa=(
                node["branch"]["positive"]["first_cell_mu_Pa"]),
            negative_first_cell_mu_Pa=(
                node["branch"]["negative"]["first_cell_mu_Pa"]),
            node_closure_relative=node["zero_storage_closure_relative"],
            field_closure_relative=packet["branch_boundary_flux"][
                "closure_relative"],
            total_mass_relative_error=packet["mass_relative_error_after_flux"],
            explicit_fourth_order_courant=packet[
                "explicit_fourth_order_courant"])
        for key, value in values.items():
            trace[key].append(value)
        if len(trace["q_over_b"]) % 1000 == 0:
            print(
                "adaptive explicit progress",
                len(trace["q_over_b"]), values["q_over_b"],
                values["transport_affinity_Pa"], flush=True)

    print("long adaptive explicit reference", LONG_REFERENCE_QUOTA_FRACTION_B,
          flush=True)
    long_event = run_event(
        state_t1, setup, geom, evaluator, transport,
        packet_fraction_b=0.0025,
        quota_fraction_b=LONG_REFERENCE_QUOTA_FRACTION_B,
        explicit_courant=LONG_EXPLICIT_COURANT,
        callback=record,
        diagnostic_stride=100)
    long_summary = event_summary(long_event)
    long_summary.update(
        maximum_field_closure_relative=max_field_closure,
        maximum_total_mass_relative_error=max_total_mass_error,
        all_accepted_steps_recorded_in_npz=True)
    np.savez_compressed(OUT_TRACE, **{
        key: np.asarray(values, dtype=float) for key, values in trace.items()})

    local_explicit_error = relative_error(
        local_explicit["C_0.05"]["final_transport_affinity_Pa"],
        explicit_reference["final_transport_affinity_Pa"])
    long_courant_local_error = relative_error(
        local_explicit[f"C_{LONG_EXPLICIT_COURANT:g}"][
            "final_transport_affinity_Pa"],
        explicit_reference["final_transport_affinity_Pa"])
    implicit_errors = [
        implicit_local[f"packet_{packet:g}b"][
            "affinity_relative_error_vs_explicit_C0p025"]
        for packet in IMPLICIT_LOCAL_PACKET_FRACTIONS_B]
    explicit_reference_pass = bool(
        all(row["completed"] for row in local_explicit.values())
        and local_explicit_error < 1e-4
        and long_courant_local_error < 1e-4)
    implicit_local_convergence_pass = bool(
        all(row["completed"] for row in implicit_local.values())
        and all(right < left for left, right in zip(
            implicit_errors[:-1], implicit_errors[1:]))
        and implicit_errors[-1] < 1e-4)
    large_scout_packet_dependent = len({
        row["progress_over_b"] for row in implicit_scout.values()}) > 1
    first_0p05b_pass = bool(
        manufactured["passed"] and explicit_reference_pass
        and implicit_local_convergence_pass and long_summary["completed"]
        and long_summary["progress_over_b"] >= 0.05)
    full_one_b_pass = bool(
        first_0p05b_pass
        and LONG_REFERENCE_QUOTA_FRACTION_B >= 1.0
        and long_summary["progress_over_b"] >= 1.0)

    result = dict(
        frozen_contract=dict(
            geometry="saved coarse t=1 two-mode PR state only",
            timescale_case="central model-time pair only",
            q_event_m=B_M,
            q_event_over_b=1.0,
            D_GB_m2_per_model_time=D_GB_MODEL,
            surface_mobility_m6_per_J_model_time=SURFACE_MOBILITY_MODEL,
            B_PF_m4_per_model_time=B_PF_MODEL,
            tau_ex_model=TAU_EX_MODEL,
            kinetic_time_basis="model_time_demonstration",
            physical_seconds_conversion=None,
            material_specific_diffusivity_used=False,
            ScSZ_data_used=False,
            hazard_or_barrier_evaluated=False,
            full_3x3_matrix_rerun=False,
            forced_half_branch_partition=False,
            Young_Herring_force_balance_imposed=False),
        manufactured_asymmetric_node=manufactured,
        saved_t1=dict(
            sink_OFF_node=compact_node(t1_off),
            active_node=compact_node(t1_active),
            interpretation=(
                "TJ chemical potentials and branch rates are transport-only; "
                "none is a nucleation activation stress")),
        local_explicit_reference=dict(
            quota_fraction_b=LOCAL_QUOTA_FRACTION_B,
            cases=local_explicit,
            C0p05_vs_C0p025_affinity_relative_difference=(
                local_explicit_error),
            long_C0p15_vs_C0p025_affinity_relative_difference=(
                long_courant_local_error),
            passed=explicit_reference_pass),
        local_linearly_implicit_convergence=dict(
            quota_fraction_b=LOCAL_QUOTA_FRACTION_B,
            cases=implicit_local,
            affinity_relative_errors=implicit_errors,
            passed=implicit_local_convergence_pass),
        large_packet_implicit_scout=dict(
            quota_fraction_b=0.05,
            cases=implicit_scout,
            stop_is_packet_dependent=large_scout_packet_dependent,
            interpretation=(
                "not a physical arrest; the stop shifts under packet refinement")),
        adaptive_explicit_reference=dict(
            requested_quota_fraction_b=LONG_REFERENCE_QUOTA_FRACTION_B,
            fourth_order_courant_limit=LONG_EXPLICIT_COURANT,
            trace_npz=OUT_TRACE,
            summary=long_summary),
        gates=dict(
            manufactured_node_pass=manufactured["passed"],
            explicit_small_step_reference_pass=explicit_reference_pass,
            linearly_implicit_converges_to_reference=(
                implicit_local_convergence_pass),
            reaches_first_0p05b=bool(
                long_summary["completed"]
                and long_summary["progress_over_b"] >= 0.05),
            saved_t1_first_0p05b_pass=first_0p05b_pass,
            full_one_b_event_pass=full_one_b_pass,
            matrix_rerun_authorized=first_0p05b_pass,
            hazard_authorized=full_one_b_pass),
        disposition=(
            "PASS first 0.05 b; proceed to the next deterministic stage, not hazard"
            if first_0p05b_pass else
            "STOP before 3x3 matrix and hazard; deterministic event remains unqualified"),
        wall_seconds=time.monotonic() - started)
    with open(OUT_JSON, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["gates"], indent=2), flush=True)
    print(OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
