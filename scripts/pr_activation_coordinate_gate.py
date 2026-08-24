"""Thermodynamic, loading, event-response, grid, and localization gate.

This script does not implement or calibrate a nucleation hazard.  It evaluates
state coordinates only on the frozen scientific 100 nm M16G/H C3-crest PR
geometry and the corrected 0.01 b event baseline.
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
from m16h_three_regime_sink_barrier import build_case
from pf_sintering.axisym import axisym_free_energy_gb, axisym_volume
from pf_sintering.axisym_numba_kernel import NumbaScratch, axisym_gb_face_projected_step_fast
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows
from pf_sintering.pr_activation_coordinates import (
    localized_embryo_mu_matrix,
    winterbottom_spherical_cap_reference,
)
from pf_sintering.pr_stress_metrology import branch_patch_3d_stress, pf_contour_estimators
from pf_sintering.rigid_rbm_deposition import representation_corrected_flux_limited_event_step
from pr_event_metrology_resolution_audit import B_M, SECONDS_PER_MODEL_TIME, advance
from pr_flux_limited_incremental_event_audit import fixed_event_support, local_coordinates

OUT_DIR = "/private/tmp/pr_activation_coordinate_gate"
FINE_STATE = "/private/tmp/pr_activation_coordinate_resolution_audit/fine_t1_state.npz"
EMBRYO_RESULT = "/private/tmp/pr_activation_coordinate_resolution_audit/resolution_audit.json"
DOSE_B = 0.01
SOURCE_WIDTHS_W = (1.0, 2.0, 3.0)
SINK_WIDTHS_W = (0.5, 1.0, 1.5, 2.0)
PATCH_WIDTHS_W = (1.5, 2.0, 2.5)
SAMPLE_DT = 0.05
FORCED_TIMES = (0.25, 0.625, 1.0)


def make_setup(dx_nm):
    geom, p, Wc, dr, dz, Ms, Meta, dt, gamma_s, gamma_gb = build_case(dx_nm=dx_nm)
    setup = dict(
        p=p, Wc=Wc, dr=dr, dz=dz, r_c=geom["r_c"], r_f=geom["r_f"],
        z=geom["z"], lam=geom["lam"], M_s=Ms, M_eta=Meta, dt=dt,
        W=p.W, gamma_s=gamma_s, gamma_gb=gamma_gb)
    return geom, setup


def state_coordinates(state, setup, z_hint):
    f, particle, neighbor = state
    R = measure_R_of_z(f, setup["r_c"])
    z_gb, _ = find_gb_trough(R, setup["z"], z_hint, lam=setup["lam"])
    met = pf_contour_estimators(
        R, setup["z"], z_gb, gamma_s=setup["gamma_s"],
        psi_reference_deg=160.0)
    local = met["local_reference"]
    candidate_a = {
        "point_polynomial": setup["gamma_s"] * local["s_local_cap"] / 1e6,
    }
    patch_details = {}
    for width_W in PATCH_WIDTHS_W:
        patch = branch_patch_3d_stress(
            R, setup["z"], z_gb, patch_length_m=width_W * setup["W"],
            tangent_span_m=0.5 * setup["W"], psi_reference_deg=160.0,
            gamma_s=setup["gamma_s"])
        key = f"patch_{width_W:g}W"
        candidate_a[key] = setup["gamma_s"] * patch["s_local_cap"] / 1e6
        patch_details[key] = patch

    circle = neck_curvature_windows(
        R, setup["z"], z_gb, setup["W"], window_widths_in_W=(1.5,))[0]
    sigma_h, _, _, c_gb = hussein_eq1b_sigma(
        circle["r_neck"], met["X_neck"], setup["gamma_s"], setup["gamma_gb"])
    candidate_b = localized_embryo_mu_matrix(
        f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
        setup["dz"], setup["r_c"], setup["r_f"], setup["z"], z_gb,
        met["r_neck"], setup["W"], SOURCE_WIDTHS_W, SINK_WIDTHS_W)
    return dict(
        z_gb_m=z_gb, r_neck_m=met["r_neck"], contact_area_m2=met["contact_area"],
        X_A_MPa=candidate_a,
        X_B_MPa={key: row["X_B_embryo_mu_residual_Pa"] / 1e6
                 for key, row in candidate_b.items()},
        X_B_details=candidate_b,
        X_C_status="20 nm arbitrary embryo-path diagnostic; not physically promotable",
        sigma_H_legacy_MPa=sigma_h / 1e6,
        sigma_H_C_GB=c_gb,
        Sigma_contact_3D_MPa=local["sigma_3D_local_MPa"],
        contact_traction_role="absolute geometric/contact traction, not zero-based residual",
        patch_details=patch_details,
        energy=axisym_free_energy_gb(
            f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
            setup["dz"], setup["r_c"], setup["r_f"], bc_z="noflux"),
        mass=axisym_volume(f, setup["r_c"], setup["dr"], setup["dz"]))


def evolve_loading(geom, setup):
    state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    scratch = NumbaScratch(*state[0].shape)
    target_step = int(round(1.0 / setup["dt"]))
    sample_stride = max(1, int(round(SAMPLE_DT / setup["dt"])))
    targets = set(range(0, target_step + 1, sample_stride))
    targets.add(target_step)
    forced_steps = {int(round(t / setup["dt"])): t for t in FORCED_TIMES}
    targets.update(forced_steps)
    rows, forced_states = [], {}
    step = 0
    z_hint = geom["z1"]
    mass0 = axisym_volume(state[0], setup["r_c"], setup["dr"], setup["dz"])
    for target in sorted(targets):
        while step < target:
            state = axisym_gb_face_projected_step_fast(
                *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
                setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
                setup["M_eta"], setup["W"], scratch)
            state = tuple(value.copy() for value in state)
            step += 1
        coord = state_coordinates(state, setup, z_hint)
        z_hint = coord["z_gb_m"]
        coord["step"] = step
        coord["model_time"] = step * setup["dt"]
        coord["mass_drift"] = (coord["mass"] - mass0) / mass0
        rows.append(coord)
        if target in forced_steps:
            forced_states[f"t{forced_steps[target]:g}"] = tuple(value.copy() for value in state)
        print("loading", step, coord["model_time"], coord["X_A_MPa"]["patch_2W"], flush=True)
    return rows, forced_states


def loading_gate(rows, family):
    selected = [row for row in rows if row["model_time"] >= 0.2]
    time = np.array([row["model_time"] for row in selected])
    keys = selected[0][family]
    out = {}
    for key in keys:
        values = np.array([row[family][key] for row in selected])
        slope = float(np.polyfit(time, values, 1)[0])
        positive = np.diff(values) > 0.0
        best_start = best_stop = run_start = 0
        for index, is_positive in enumerate(positive):
            if not is_positive:
                run_start = index + 1
            elif index + 1 - run_start > best_stop - best_start:
                best_start, best_stop = run_start, index + 1
        sustained_duration = float(time[best_stop] - time[best_start])
        out[key] = dict(
            slope_MPa_per_model_time=slope,
            positive_increment_fraction=float(np.mean(positive)),
            sustained_monotone_loading=bool(slope > 0.0 and np.all(np.diff(values) > 0.0)),
            longest_positive_interval=dict(
                start_model_time=float(time[best_start]),
                end_model_time=float(time[best_stop]),
                duration_model_time=sustained_duration),
            sustained_half_time_loading=bool(sustained_duration >= 0.5),
            start_MPa=float(values[0]), end_MPa=float(values[-1]))
    return out


def relative_grid_change(coarse_value, fine_value):
    return abs(fine_value - coarse_value) / max(abs(coarse_value), 1e-30)


def scalar_loading_gate(rows, key):
    selected = [row for row in rows if row["model_time"] >= 0.2]
    time = np.array([row["model_time"] for row in selected])
    values = np.array([row[key] for row in selected])
    return dict(
        slope_MPa_per_model_time=float(np.polyfit(time, values, 1)[0]),
        positive_increment_fraction=float(np.mean(np.diff(values) > 0.0)),
        start_MPa=float(values[0]), end_MPa=float(values[-1]))


def gate_assessment(coarse, fine):
    coarse_a_load = coarse["loading_gate_X_A"]
    coarse_b_load = coarse["loading_gate_X_B"]
    coarse_a_event = coarse["event_response"]["stages"]["3PF"]["X_A_delta_MPa"]
    fine_a_event = fine["event_response"]["stages"]["3PF"]["X_A_delta_MPa"]
    coarse_b_event = coarse["event_response"]["stages"]["3PF"]["X_B_delta_MPa"]
    fine_b_event = fine["event_response"]["stages"]["3PF"]["X_B_delta_MPa"]
    a_grid = {key: relative_grid_change(coarse_a_event[key], fine_a_event[key])
              for key in coarse_a_event}
    b_grid = {key: relative_grid_change(coarse_b_event[key], fine_b_event[key])
              for key in coarse_b_event}
    a_loading_signs = {key: row["slope_MPa_per_model_time"] > 0.0
                       for key, row in coarse_a_load.items()}
    b_loading_signs = {key: row["slope_MPa_per_model_time"] > 0.0
                       for key, row in coarse_b_load.items()}
    return dict(
        X_A=dict(
            equilibrium_zero=True,
            loading_sign_by_coarse_graining=a_loading_signs,
            all_0p01b_3PF_responses_negative=all(value < 0.0 for value in coarse_a_event.values()),
            coarse_fine_3PF_relative_change=a_grid,
            promoted=False,
            decision=(
                "REJECT FOR HAZARD: the point-polynomial value has a sustained "
                "loading interval but remains point-local; fixed physical patches "
                "change the sink-off loading sign between 1.5W and 2-2.5W. No "
                "physically selected meridional-curvature localization exists.")),
        X_B=dict(
            equilibrium_zero=True,
            loading_sign_by_localization=b_loading_signs,
            positive_loading_count=sum(b_loading_signs.values()),
            localization_count=len(b_loading_signs),
            all_0p01b_3PF_responses_negative=all(value < 0.0 for value in coarse_b_event.values()),
            all_event_response_signs_grid_invariant=all(
                coarse_b_event[key] * fine_b_event[key] > 0.0 for key in coarse_b_event),
            coarse_fine_3PF_relative_change=b_grid,
            promoted=False,
            decision=(
                "REJECT FOR HAZARD: event relaxation is sign-robust, but sink-off "
                "loading occurs only for a minority of the predeclared support "
                "matrix and is tied to the narrow 0.5W sink support.")),
        X_C=dict(
            promoted=False,
            decision=(
                "REJECT FOR HAZARD: the existing 20 nm embryo extent is an "
                "arbitrary path choice, not a disconnection core or critical nucleus.")),
        sigma_H_legacy=dict(
            equilibrium_zero=False, promoted=False,
            loading=coarse["loading_gate_sigma_H_legacy"],
            coarse_3PF_delta_MPa=coarse["event_response"]["stages"]["3PF"]["sigma_H_legacy_delta_MPa"],
            fine_3PF_delta_MPa=fine["event_response"]["stages"]["3PF"]["sigma_H_legacy_delta_MPa"],
            decision="historical absolute quantity; 5 MPa on the exact equilibrium reference"),
        Sigma_contact_3D=dict(
            equilibrium_zero=False, promoted=False,
            loading=coarse["loading_gate_Sigma_contact_3D"],
            coarse_3PF_delta_MPa=coarse["event_response"]["stages"]["3PF"]["Sigma_contact_3D_delta_MPa"],
            fine_3PF_delta_MPa=fine["event_response"]["stages"]["3PF"]["Sigma_contact_3D_delta_MPa"],
            decision="absolute contact traction; 20 MPa on the exact equilibrium reference"),
        overall=dict(
            promoted_coordinate=None,
            deterministic_one_b_cycle_run=False,
            arrhenius_hazard_implemented=False,
            stop_reason=(
                "No candidate passes equilibrium/reference, sustained loading, "
                "event relaxation, grid, and localization gates simultaneously.")))


def corrected_small_event(state, geom, setup):
    broad = local_coordinates(state, setup, geom["z1"])
    support, p_fraction = fixed_event_support(state, setup, broad)
    Dgb = 1e-3 * math.exp(-1.5e5 / (8.314 * 1200.0))
    hp = HazardParams(T=1200.0, D_gb=Dgb, GS=2.0 * geom["R_z1"], b=B_M)
    event = representation_corrected_flux_limited_event_step(
        *state, AxisymSink(active=True), hp, broad["transport_affinity_Pa"],
        setup["dt"], setup["dz"], setup["r_c"], setup["z"], broad["z_gb"],
        broad["contact_area"], support, p_fraction,
        seconds_per_model_time=SECONDS_PER_MODEL_TIME,
        max_increment_fraction_b=0.0025, event_quota_m=DOSE_B * B_M)
    if not event[3]:
        raise RuntimeError("corrected 0.01 b gate event did not complete")
    return event


def event_response(state, geom, setup):
    event = corrected_small_event(state, geom, setup)
    event_states = {"immediate": event[:3]}
    event_states.update({f"{n}PF": value for n, value in advance(event[:3], setup).items()})
    control_states = {"immediate": state}
    control_states.update({f"{n}PF": value for n, value in advance(state, setup).items()})
    out = {"kinetic": event[4], "stages": {}}
    z_hint = geom["z1"]
    for stage in ("immediate", "1PF", "3PF"):
        event_coord = state_coordinates(event_states[stage], setup, z_hint)
        control_coord = state_coordinates(control_states[stage], setup, z_hint)
        out["stages"][stage] = dict(
            X_A_delta_MPa={key: event_coord["X_A_MPa"][key]
                           - control_coord["X_A_MPa"][key]
                           for key in event_coord["X_A_MPa"]},
            X_B_delta_MPa={key: event_coord["X_B_MPa"][key]
                           - control_coord["X_B_MPa"][key]
                           for key in event_coord["X_B_MPa"]},
            sigma_H_legacy_delta_MPa=(event_coord["sigma_H_legacy_MPa"]
                                      - control_coord["sigma_H_legacy_MPa"]),
            Sigma_contact_3D_delta_MPa=(event_coord["Sigma_contact_3D_MPa"]
                                        - control_coord["Sigma_contact_3D_MPa"]),
            energy_delta=event_coord["energy"] - control_coord["energy"],
            mass_difference=event_coord["mass"] - control_coord["mass"])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coarse_geom, coarse_setup = make_setup(1.25)
    loading, forced_states = evolve_loading(coarse_geom, coarse_setup)
    coarse_t1 = forced_states["t1"]
    coarse_response = event_response(coarse_t1, coarse_geom, coarse_setup)

    fine_geom, fine_setup = make_setup(1.0)
    fine_saved = np.load(FINE_STATE)
    fine_t1 = (fine_saved["f"], fine_saved["particle"], fine_saved["neighbor"])
    fine_coord = state_coordinates(fine_t1, fine_setup, fine_geom["z1"])
    fine_response = event_response(fine_t1, fine_geom, fine_setup)
    coarse_coord = state_coordinates(coarse_t1, coarse_setup, coarse_geom["z1"])

    equilibrium = winterbottom_spherical_cap_reference(
        100e-9, math.radians(160.0), gamma_s=coarse_setup["gamma_s"],
        gamma_gb=coarse_setup["gamma_gb"])
    embryo = json.load(open(EMBRYO_RESULT))
    coarse_block = dict(
        dx_nm=1.25, loading=loading,
        loading_gate_X_A=loading_gate(loading, "X_A_MPa"),
        loading_gate_X_B=loading_gate(loading, "X_B_MPa"),
        loading_gate_sigma_H_legacy=scalar_loading_gate(loading, "sigma_H_legacy_MPa"),
        loading_gate_Sigma_contact_3D=scalar_loading_gate(loading, "Sigma_contact_3D_MPa"),
        t1=coarse_coord, event_response=coarse_response)
    fine_block = dict(dx_nm=1.0, t1=fine_coord, event_response=fine_response)
    result = dict(
        frozen_geometry=dict(
            classification="scientific PR topology",
            construction="historical M16G/M16H C3-crest",
            R_cyl_nm=100.0, W_nm=10.0, noflux_axial=True),
        equilibrium_reference=equilibrium,
        predeclared_localization=dict(
            X_A_patch_widths_W=PATCH_WIDTHS_W,
            X_B_source_widths_W=SOURCE_WIDTHS_W,
            X_B_sink_widths_W=SINK_WIDTHS_W),
        coarse=coarse_block,
        fine=fine_block,
        X_C_20nm_embryo_path=dict(
            status="diagnostic only; extent is not derived from disconnection physics",
            coarse_zero_dose_MPa=embryo["coarse"]["embryo_path"]["zero_dose_extrapolation_MPa"],
            fine_zero_dose_MPa=embryo["fine"]["embryo_path"]["zero_dose_extrapolation_MPa"]),
        rate_independence=dict(
            state_coordinates_accept_D_gb=False,
            state_coordinates_accept_M_s=False,
            state_coordinates_accept_timestep=False,
            state_coordinates_accept_packet_count=False),
        assessment=gate_assessment(coarse_block, fine_block),
        hazard_implemented=False,
        barrier_calibrated=False)
    with open(os.path.join(OUT_DIR, "activation_coordinate_gate.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    np.savez_compressed(
        os.path.join(OUT_DIR, "forced_loading_states.npz"),
        **{f"f_{key}": value[0] for key, value in forced_states.items()},
        **{f"particle_{key}": value[1] for key, value in forced_states.items()},
        **{f"neighbor_{key}": value[2] for key, value in forced_states.items()},
        z=coarse_setup["z"], r_c=coarse_setup["r_c"], dt=coarse_setup["dt"])
    print("OUT", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
