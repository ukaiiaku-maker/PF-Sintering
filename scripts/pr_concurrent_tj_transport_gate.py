"""Concurrent GB delivery/TJ arrival/surface-PF qualification gate.

For every packet the physical ordering is

``Delta mu_tr -> Vdot_GB -> dq -> conservative TJ arrival -> M_s PF(Delta t)``.

The event quota remains exactly b and no hazard or barrier is evaluated.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z
from m16g_pr_derived_particle_asperity import find_gb_trough
from pf_sintering.axisym import axisym_mu_f_gb, axisym_volume
from pf_sintering.axisym_numba_kernel import (
    NumbaScratch,
    axisym_gb_face_projected_step_fast,
)
from pf_sintering.axisym_sink_rbm import HazardParams
from pf_sintering.pr_stress_metrology import pf_contour_estimators
from pf_sintering.pr_tj_activation import (
    add_equilibrium_referenced_tj_stresses,
    tj_functionals_from_mu,
)
from pf_sintering.rigid_rbm_deposition import (
    _axisym_weighted_sum,
    apply_fixed_tj_source,
    make_fixed_tj_source_support,
    normalized_axisym_support,
    representation_corrected_concurrent_transport_event,
    representation_corrected_union_transfer,
)
from pr_event_metrology_resolution_audit import B_M, SECONDS_PER_MODEL_TIME
from pr_tj_activation_gate import make_setup


SOURCE_JSON = "/private/tmp/pr_tj_activation_gate/tj_activation_gate.json"
SOURCE_STATES = "/private/tmp/pr_tj_activation_gate/saved_states.npz"
OUT_DIR = "/private/tmp/pr_concurrent_tj_transport_gate"
DOSES_B = (0.001, 0.0025, 0.005, 0.01, 0.02)
STATE_KEYS = ("t0.25", "t0.625", "t1")
FULL_PACKET_FRACTION_B = 0.001
DECOMPOSITION_PACKET_FRACTION_B = 0.0005
PRIMARY_CASES = (
    ("coarse", 10.0, 1.25),
    ("fine", 10.0, 1.0),
)
W_CASES = (
    ("coarse", 10.0, 1.25),
    ("W8p75_dx1p09375", 8.75, 1.09375),
    ("W7p5_dx0p9375", 7.5, 0.9375),
)


def load_state(saved, label, key):
    return (
        saved[f"{label}_f_{key}"],
        saved[f"{label}_particle_{key}"],
        saved[f"{label}_neighbor_{key}"],
    )


def make_evaluator(setup, geom, equilibrium, fixed_support):
    Z = setup["z"][:, None]
    RC = setup["r_c"][None, :]

    def evaluate(f, particle, neighbor):
        R = measure_R_of_z(f, setup["r_c"])
        z_tj, _ = find_gb_trough(
            R, setup["z"], geom["z1"], lam=setup["lam"])
        contour = pf_contour_estimators(
            R, setup["z"], z_tj, gamma_s=setup["gamma_s"],
            psi_reference_deg=160.0)
        mu = axisym_mu_f_gb(
            f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
            setup["dz"], setup["r_c"], setup["r_f"], bc_z="noflux")
        tj_raw = tj_functionals_from_mu(
            mu, f, particle, neighbor, setup["dr"], setup["dz"],
            setup["r_c"])
        tj = add_equilibrium_referenced_tj_stresses(tj_raw, equilibrium, B_M)
        gb_raw = (
            np.maximum(particle * neighbor, 0.0)
            * (np.abs(Z - z_tj) <= 2.0 * setup["W"])
            * (RC <= contour["r_neck"] + setup["W"]))
        gb_support = normalized_axisym_support(gb_raw, setup["r_c"])
        mu_source = _axisym_weighted_sum(mu * gb_support, setup["r_c"])
        mu_sink = _axisym_weighted_sum(mu * fixed_support, setup["r_c"])
        return dict(
            transport_affinity_Pa=float(mu_source - mu_sink),
            contact_area_m2=contour["contact_area"],
            mu_GB_source_Pa=float(mu_source),
            mu_TJ_sink_Pa=float(mu_sink),
            mu_PF_TJ_Pa=tj["mu_PF_TJ_Pa"],
            sigma_act_TJ_mu_Pa=tj["sigma_act_TJ_mu_Pa"],
            lambda_TJ_excess_N_per_m=tj["lambda_TJ_excess_N_per_m"],
            sigma_act_TJ_lambda_Pa=tj["sigma_act_TJ_lambda_Pa"],
            X_Sigma_diffuse_MPa=(
                contour["local_reference"]["sigma_3D_local_MPa"]
                - equilibrium["Sigma_contact_3D_MPa"]),
            z_TJ_m=z_tj, r_TJ_m=contour["r_neck"],
            total_volume_m3=axisym_volume(
                f, setup["r_c"], setup["dr"], setup["dz"]))
    return evaluate


def make_surface_step(setup):
    scratch = None

    def surface_step(f, particle, neighbor, dt_model):
        nonlocal scratch
        if scratch is None:
            scratch = NumbaScratch(*f.shape)
        state = axisym_gb_face_projected_step_fast(
            f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
            setup["dz"], setup["r_c"], setup["r_f"], float(dt_model),
            setup["M_s"], setup["M_eta"], setup["W"], scratch)
        return tuple(field.copy() for field in state)
    return surface_step


def make_event_context(pre, setup, geom, equilibrium):
    R = measure_R_of_z(pre[0], setup["r_c"])
    z_tj, _ = find_gb_trough(R, setup["z"], geom["z1"], lam=setup["lam"])
    contour = pf_contour_estimators(
        R, setup["z"], z_tj, gamma_s=setup["gamma_s"],
        psi_reference_deg=160.0)
    support, particle_fraction = make_fixed_tj_source_support(
        *pre, setup["r_c"], setup["z"], z_tj, contour["r_neck"], setup["W"])
    evaluator = make_evaluator(setup, geom, equilibrium, support)
    Dgb = 1e-3 * math.exp(-1.5e5 / (8.314 * 1200.0))
    hp = HazardParams(T=1200.0, D_gb=Dgb, GS=2.0 * geom["R_z1"], b=B_M)
    return z_tj, support, particle_fraction, evaluator, hp


def deltas(row, baseline):
    keys = (
        "mu_PF_TJ_Pa", "sigma_act_TJ_mu_Pa",
        "lambda_TJ_excess_N_per_m", "sigma_act_TJ_lambda_Pa",
        "mu_GB_source_Pa", "mu_TJ_sink_Pa", "transport_affinity_Pa")
    return {key: row[key] - baseline[key] for key in keys}


def three_path_decomposition(pre, setup, geom, equilibrium):
    z_tj, support, particle_fraction, evaluator, hp = make_event_context(
        pre, setup, geom, equilibrium)
    pre_coord = evaluator(*pre)
    factor = 2.0 * math.pi * setup["dr"] * setup["dz"]
    rows = []
    for dose in DOSES_B:
        q = dose * B_M
        transfer = representation_corrected_union_transfer(
            *pre, q, setup["dz"], setup["r_c"], setup["z"], z_tj)
        union = transfer[:3]
        reservoir = apply_fixed_tj_source(
            *union, setup["r_c"], support, particle_fraction,
            transfer[3]["V_source_weighted"])[:3]
        concurrent = representation_corrected_concurrent_transport_event(
            *pre, hp, setup["dz"], setup["r_c"], setup["z"], z_tj,
            support, particle_fraction, evaluator, make_surface_step(setup),
            max_increment_fraction_b=DECOMPOSITION_PACKET_FRACTION_B,
            event_quota_m=q, seconds_per_model_time=SECONDS_PER_MODEL_TIME)
        if not concurrent[3]:
            raise RuntimeError(f"small-dose concurrent event stalled at dose={dose}b")
        union_coord = evaluator(*union)
        reservoir_coord = evaluator(*reservoir)
        concurrent_coord = evaluator(*concurrent[:3])
        rows.append(dict(
            dose_b=dose, q_m=q,
            removed_source_volume_m3=transfer[3]["V_source_weighted"] * factor,
            union_only=dict(coordinates=union_coord, delta=deltas(union_coord, pre_coord),
                            physical=False, role="nonconservative algebraic decomposition"),
            cumulative_reservoir=dict(
                coordinates=reservoir_coord,
                delta=deltas(reservoir_coord, pre_coord),
                role="present cumulative storage implementation"),
            concurrent_surface_PF=dict(
                coordinates=concurrent_coord,
                delta=deltas(concurrent_coord, pre_coord),
                physical_transport_time_s=concurrent[4]["physical_time_consumed_s"],
                surface_PF_model_time=concurrent[4]["surface_PF_model_time"],
                n_packets=concurrent[4]["n_subincrements"],
                mass_relative_error=concurrent[4]["packets"][-1][
                    "mass_relative_error_after_surface_PF"],
                role="packetwise arrival then M_s PF over identical elapsed time")))
    sigma_reservoir = np.array([
        row["cumulative_reservoir"]["delta"]["sigma_act_TJ_mu_Pa"] for row in rows])
    sigma_concurrent = np.array([
        row["concurrent_surface_PF"]["delta"]["sigma_act_TJ_mu_Pa"] for row in rows])
    dose = np.array(DOSES_B)
    return dict(
        pre=pre_coord, rows=rows,
        small_dose_slopes=dict(
            cumulative_reservoir_sigma_mu_Pa_per_b=float(
                np.polyfit(dose, sigma_reservoir, 1)[0]),
            concurrent_sigma_mu_Pa_per_b=float(
                np.polyfit(dose, sigma_concurrent, 1)[0])),
        reservoir_confound_removed_at_0p01b=bool(
            abs(rows[3]["concurrent_surface_PF"]["delta"]["sigma_act_TJ_mu_Pa"])
            < 0.5 * abs(rows[3]["cumulative_reservoir"]["delta"]["sigma_act_TJ_mu_Pa"])))


def full_event(pre, setup, geom, equilibrium,
               packet_fraction_b=FULL_PACKET_FRACTION_B):
    z_tj, support, particle_fraction, evaluator, hp = make_event_context(
        pre, setup, geom, equilibrium)
    mass0 = axisym_volume(pre[0], setup["r_c"], setup["dr"], setup["dz"])
    event = representation_corrected_concurrent_transport_event(
        *pre, hp, setup["dz"], setup["r_c"], setup["z"], z_tj,
        support, particle_fraction, evaluator, make_surface_step(setup),
        max_increment_fraction_b=packet_fraction_b,
        event_quota_m=B_M, seconds_per_model_time=SECONDS_PER_MODEL_TIME)
    diag = event[4]
    initial = evaluator(*pre)
    final = evaluator(*event[:3])
    packets = diag["packets"]
    surface_effect_sigma = np.array([
        row["coordinates_after_surface_PF"]["sigma_act_TJ_mu_Pa"]
        - row["coordinates_after_arrival"]["sigma_act_TJ_mu_Pa"]
        for row in packets])
    return dict(
        completed_one_b=bool(event[3]),
        packet_fraction_b=packet_fraction_b,
        event_progress_over_b=diag["event_progress_over_b"],
        first_nonpositive_q_over_b=(
            None if event[3] else diag["event_progress_over_b"]),
        n_packets=len(packets),
        initial=initial, final=final,
        net_delta=deltas(final, initial),
        physical_transport_time_s=diag["physical_time_consumed_s"],
        surface_PF_model_time=diag["surface_PF_model_time"],
        maximum_absolute_packet_surface_sigma_mu_change_Pa=(
            float(np.max(np.abs(surface_effect_sigma))) if len(packets) else 0.0),
        cumulative_packet_surface_sigma_mu_change_Pa=float(np.sum(surface_effect_sigma)),
        mass_relative_error=(
            axisym_volume(event[0], setup["r_c"], setup["dr"], setup["dz"])
            - mass0) / mass0,
        packets=packets,
        stop_state=diag.get("stop_state"),
        diagnosis=dict(
            affinity_zero_removed_or_moved_near_b=bool(
                event[3] or diag["event_progress_over_b"] >= 0.8),
            activation_mu_relaxed=bool(
                final["sigma_act_TJ_mu_Pa"] < initial["sigma_act_TJ_mu_Pa"]),
            activation_lambda_relaxed=bool(
                final["sigma_act_TJ_lambda_Pa"]
                < initial["sigma_act_TJ_lambda_Pa"])))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SOURCE_JSON) as handle:
        source = json.load(handle)
    saved = np.load(SOURCE_STATES)
    decomposition = {}
    full = {}
    contexts = {}
    for label, W_nm, dx_nm in set(PRIMARY_CASES + W_CASES):
        geom, setup = make_setup(W_nm, dx_nm)
        block = (source["primary_grid"].get(label)
                 or source["W_ladder"].get(label))
        contexts[label] = (geom, setup, block["equilibrium_reference"]["baseline"])
    for label, _, _ in PRIMARY_CASES:
        geom, setup, equilibrium = contexts[label]
        decomposition[label] = three_path_decomposition(
            load_state(saved, label, "t1"), setup, geom, equilibrium)
        full[label] = {
            key: full_event(
                load_state(saved, label, key), setup, geom, equilibrium)
            for key in STATE_KEYS}
    W_event = {}
    for label, _, _ in W_CASES:
        geom, setup, equilibrium = contexts[label]
        pre = load_state(saved, label, "t1")
        z_tj, support, fraction, evaluator, hp = make_event_context(
            pre, setup, geom, equilibrium)
        initial = evaluator(*pre)
        event = representation_corrected_concurrent_transport_event(
            *pre, hp, setup["dz"], setup["r_c"], setup["z"], z_tj,
            support, fraction, evaluator, make_surface_step(setup),
            max_increment_fraction_b=DECOMPOSITION_PACKET_FRACTION_B,
            event_quota_m=0.01 * B_M,
            seconds_per_model_time=SECONDS_PER_MODEL_TIME)
        final = evaluator(*event[:3])
        W_event[label] = dict(
            W_nm=setup["W"] * 1e9, dx_nm=setup["dr"] * 1e9,
            completed=bool(event[3]), delta=deltas(final, initial),
            physical_transport_time_s=event[4]["physical_time_consumed_s"],
            surface_PF_model_time=event[4]["surface_PF_model_time"])

    coarse_geom, coarse_setup, coarse_equilibrium = contexts["coarse"]
    coarse_t1 = load_state(saved, "coarse", "t1")
    packet_convergence = {}
    for fraction in (0.002, 0.001, 0.0005):
        row = full_event(
            coarse_t1, coarse_setup, coarse_geom, coarse_equilibrium,
            packet_fraction_b=fraction)
        packet_convergence[f"{fraction:g}b"] = dict(
            packet_fraction_b=fraction,
            event_progress_over_b=row["event_progress_over_b"],
            physical_transport_time_s=row["physical_transport_time_s"],
            surface_PF_model_time=row["surface_PF_model_time"],
            net_delta=row["net_delta"],
            mass_relative_error=row["mass_relative_error"])

    crossings = {
        label: {key: row["first_nonpositive_q_over_b"] for key, row in states.items()}
        for label, states in full.items()}
    result = dict(
        frozen_contract=dict(
            q_definition="delta_RBM in metres", q_event_m=B_M,
            q_event_over_b=1.0, D_gb_changed=False, M_s_changed=False,
            cumulative_body_union_preserved=True,
            fixed_arrival_support_used_as_entry_not_deferred_storage=True,
            hazard_implemented=False, barrier_evaluated=False),
        operator_order=(
            "Delta mu_tr -> Vdot_GB -> dq -> conservative TJ arrival -> "
            "ordinary M_s PF over the same physical dt"),
        three_path_t1=decomposition,
        full_b_attempts=full,
        W_ladder_0p01b=W_event,
        packet_convergence_coarse_t1=packet_convergence,
        assessment=dict(
            reservoir_confound_removed_at_0p01b=bool(all(
                block["reservoir_confound_removed_at_0p01b"]
                for block in decomposition.values())),
            affinity_crossings_q_over_b=crossings,
            all_crossings_removed_or_near_b=bool(all(
                row["diagnosis"]["affinity_zero_removed_or_moved_near_b"]
                for states in full.values() for row in states.values())),
            all_activation_mu_event_responses_relax=bool(all(
                row["diagnosis"]["activation_mu_relaxed"]
                for states in full.values() for row in states.values())),
            hazard_ready=False),
        hazard_implemented=False,
        barrier_refit=False)
    output = os.path.join(OUT_DIR, "concurrent_tj_transport_gate.json")
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    print(json.dumps(result["assessment"], indent=2), flush=True)
    for label, block in decomposition.items():
        print(label, "decomposition 0.01b", json.dumps(block["rows"][3], default=float), flush=True)
    print("OUT", output, flush=True)


if __name__ == "__main__":
    main()
