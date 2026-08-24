"""Equilibrium-referenced local-TJ activation audit for scientific PR.

This script does not evaluate or alter a nucleation barrier.  It preserves the
scientific C3-crest geometry and phase-field dynamics, evaluates two
dimensionally distinct TJ measures, and applies the existing corrected 0.01 b
matched-event gate.  A bounded W ladder keeps eight cells through the diffuse
width so numerical width and grid resolution are not conflated.
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
from m16g_pr_derived_particle_asperity import find_gb_trough, grain_volumes
from m16h_three_regime_sink_barrier import build_case
from pf_sintering.axisym import axisym_free_energy_gb, axisym_volume
from pf_sintering.axisym_numba_kernel import (
    NumbaScratch,
    axisym_gb_face_projected_step_fast,
    eta_update_kernel,
)
from pf_sintering.pr_contact_excess import (
    build_volume_matched_winterbottom,
    project_original_pr_modes,
)
from pf_sintering.pr_stress_metrology import pf_contour_estimators
from pf_sintering.pr_tj_activation import (
    add_equilibrium_referenced_tj_stresses,
    tj_thermodynamic_state,
)
from pf_sintering.rigid_rbm_deposition import local_chemical_potential_transport_drive
from pr_activation_coordinate_gate import corrected_small_event
from pr_event_metrology_resolution_audit import B_M, advance


OUT_DIR = "/private/tmp/pr_tj_activation_gate"
SAMPLE_DT = 0.05
SAVED_TIMES = (0.25, 0.625, 1.0)
REFERENCE_RELAX_MODEL_TIME = 1.0
PRIMARY_GRID_CASES = (
    ("coarse", 10.0, 1.25),
    ("fine", 10.0, 1.0),
)
W_CASES = (
    ("coarse", 10.0, 1.25),
    ("W8p75_dx1p09375", 8.75, 1.09375),
    ("W7p5_dx0p9375", 7.5, 0.9375),
)
COORDINATES = ("sigma_act_TJ_mu_Pa", "sigma_act_TJ_lambda_Pa")


def make_setup(W_nm, dx_nm):
    geom, p, Wc, dr, dz, Ms, Meta, dt, gamma_s, gamma_gb = build_case(
        W_nm=W_nm, dx_nm=dx_nm)
    return geom, dict(
        p=p, Wc=Wc, dr=dr, dz=dz, r_c=geom["r_c"], r_f=geom["r_f"],
        z=geom["z"], lam=geom["lam"], M_s=Ms, M_eta=Meta, dt=dt,
        W=p.W, gamma_s=gamma_s, gamma_gb=gamma_gb,
        W_nm=float(W_nm), dx_nm=float(dx_nm))


def build_reference(geom, setup):
    target_volume = grain_volumes(
        geom["e1"], geom["e2"], setup["r_c"], setup["dr"], setup["dz"])[0]
    built = build_volume_matched_winterbottom(
        target_volume, math.radians(160.0), setup["W"], setup["z"],
        setup["r_c"], setup["dr"], setup["dz"], geom["z1"])
    state = (built.pop("f"), built.pop("e1"), built.pop("e2"))
    f_fixed = state[0].copy()
    scratch = NumbaScratch(*f_fixed.shape)
    n_steps = int(round(REFERENCE_RELAX_MODEL_TIME / setup["dt"]))
    for _ in range(n_steps):
        eta_update_kernel(
            state[1], state[2], f_fixed, setup["Wc"], setup["p"].k_eta,
            setup["dr"], setup["dz"], setup["r_c"], setup["r_f"],
            setup["dt"], setup["M_eta"], 1e-4,
            scratch.e1_new, scratch.e2_new)
        state = (f_fixed, scratch.e1_new.copy(), scratch.e2_new.copy())
    tj = tj_thermodynamic_state(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], bc_z="noflux")
    R = measure_R_of_z(state[0], setup["r_c"])
    contour = pf_contour_estimators(
        R, setup["z"], geom["z1"], gamma_s=setup["gamma_s"],
        psi_reference_deg=160.0)
    self_excess = add_equilibrium_referenced_tj_stresses(tj, tj, B_M)
    return dict(
        construction=built,
        relaxation_model_time=REFERENCE_RELAX_MODEL_TIME,
        relaxation="eta ownership only at fixed analytical f",
        baseline=dict(
            **tj,
            Sigma_contact_3D_MPa=contour["local_reference"]["sigma_3D_local_MPa"],
            z_TJ_m=float(geom["z1"]),
            r_TJ_m=contour["r_neck"],
            self_reference_sigma_act_TJ_mu_Pa=self_excess["sigma_act_TJ_mu_Pa"],
            self_reference_lambda_TJ_excess_N_per_m=(
                self_excess["lambda_TJ_excess_N_per_m"]),
            self_reference_sigma_act_TJ_lambda_Pa=(
                self_excess["sigma_act_TJ_lambda_Pa"]),
        )), state


def observe(state, setup, geom, equilibrium, z_hint):
    f, particle, neighbor = state
    R = measure_R_of_z(f, setup["r_c"])
    z_tj, _ = find_gb_trough(R, setup["z"], z_hint, lam=setup["lam"])
    contour = pf_contour_estimators(
        R, setup["z"], z_tj, gamma_s=setup["gamma_s"],
        psi_reference_deg=160.0)
    tj_raw = tj_thermodynamic_state(
        f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
        setup["dz"], setup["r_c"], setup["r_f"], bc_z="noflux")
    tj = add_equilibrium_referenced_tj_stresses(tj_raw, equilibrium, B_M)
    transport = local_chemical_potential_transport_drive(
        f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
        setup["dz"], setup["r_c"], setup["r_f"], setup["z"], z_tj,
        contour["r_neck"], setup["W"], tj_source_support=None)
    return dict(
        **tj,
        z_TJ_m=z_tj,
        r_TJ_m=contour["r_neck"],
        X_Sigma_diffuse_MPa=(
            contour["local_reference"]["sigma_3D_local_MPa"]
            - equilibrium["Sigma_contact_3D_MPa"]),
        mu_GB_source_Pa=transport["mu_source_Pa"],
        mu_TJ_transport_sink_Pa=transport["mu_sink_Pa"],
        delta_mu_transport_Pa=transport["transport_affinity_Pa"],
        transport_support_role=(
            "diagnostic frozen transport definition; not activation support"),
        energy_J=axisym_free_energy_gb(
            f, particle, neighbor, setup["p"], setup["Wc"], setup["dr"],
            setup["dz"], setup["r_c"], setup["r_f"], bc_z="noflux"),
        total_volume_m3=axisym_volume(
            f, setup["r_c"], setup["dr"], setup["dz"]),
        A_PR=project_original_pr_modes(
            R, setup["z"], setup["lam"], geom["R_cyl"])["A_PR"],
    )


def advance_one(state, setup, scratch):
    next_state = axisym_gb_face_projected_step_fast(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
        setup["M_eta"], setup["W"], scratch)
    return tuple(field.copy() for field in next_state)


def evolve_loading(geom, setup, equilibrium):
    state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    scratch = NumbaScratch(*state[0].shape)
    final_step = int(round(1.0 / setup["dt"]))
    stride = max(1, int(round(SAMPLE_DT / setup["dt"])))
    targets = set(range(0, final_step + 1, stride)) | {final_step}
    saved_steps = {int(round(time / setup["dt"])): time for time in SAVED_TIMES}
    targets.update(saved_steps)
    rows, states = [], {}
    step = 0
    z_hint = geom["z1"]
    mass0 = axisym_volume(state[0], setup["r_c"], setup["dr"], setup["dz"])
    for target in sorted(targets):
        while step < target:
            state = advance_one(state, setup, scratch)
            step += 1
        row = observe(state, setup, geom, equilibrium, z_hint)
        z_hint = row["z_TJ_m"]
        row["step"] = step
        row["model_time"] = step * setup["dt"]
        row["mass_relative_error"] = (row["total_volume_m3"] - mass0) / mass0
        rows.append(row)
        if target in saved_steps:
            states[f"t{saved_steps[target]:g}"] = tuple(x.copy() for x in state)
        print(
            "loading", setup["W_nm"], setup["dx_nm"], row["model_time"],
            row["sigma_act_TJ_mu_Pa"] / 1e6,
            row["sigma_act_TJ_lambda_Pa"] / 1e6, flush=True)
    return rows, states


def loading_gate(rows, coordinate):
    selected = [row for row in rows if row["model_time"] >= 0.2]
    t = np.array([row["model_time"] for row in selected])
    x = np.array([row[coordinate] for row in selected])
    increments = np.diff(x) > 0.0
    best = run = 0
    for positive in increments:
        run = run + 1 if positive else 0
        best = max(best, run)
    return dict(
        slope_Pa_per_model_time=float(np.polyfit(t, x, 1)[0]),
        positive_increment_fraction=float(np.mean(increments)),
        longest_positive_interval_model_time=float(best * SAMPLE_DT),
        sustained_positive_loading=bool(
            np.polyfit(t, x, 1)[0] > 0.0 and best * SAMPLE_DT >= 0.5),
        start_Pa=float(x[0]), end_Pa=float(x[-1]))


def corrected_event_response(state, geom, setup, equilibrium):
    event = corrected_small_event(state, geom, setup)
    event_states = {"immediate": event[:3]}
    event_states.update({f"{n}PF": value for n, value in advance(event[:3], setup).items()})
    controls = {"immediate": state}
    controls.update({f"{n}PF": value for n, value in advance(state, setup).items()})
    stages = {}
    for stage in ("immediate", "1PF", "3PF"):
        ev = observe(event_states[stage], setup, geom, equilibrium, geom["z1"])
        co = observe(controls[stage], setup, geom, equilibrium, geom["z1"])
        stages[stage] = dict(
            event=ev, matched_control=co,
            delta={key: ev[key] - co[key] for key in COORDINATES})
    return dict(kinetic=event[4], stages=stages)


def run_case(label, W_nm, dx_nm):
    geom, setup = make_setup(W_nm, dx_nm)
    reference, _ = build_reference(geom, setup)
    equilibrium = reference["baseline"]
    loading, states = evolve_loading(geom, setup, equilibrium)
    events = {
        key: corrected_event_response(state, geom, setup, equilibrium)
        for key, state in states.items()}
    saved = {}
    for coordinate in COORDINATES:
        values = [
            min(loading, key=lambda row: abs(row["model_time"] - time))[coordinate]
            for time in SAVED_TIMES]
        saved[coordinate] = dict(
            times_model=SAVED_TIMES,
            values_Pa=[float(value) for value in values],
            strictly_increasing=bool(values[0] < values[1] < values[2]))
    block = dict(
        label=label, W_nm=W_nm, dx_nm=dx_nm,
        cells_per_W=W_nm / dx_nm, dt_model=setup["dt"],
        equilibrium_reference=reference,
        loading=loading,
        loading_gate={key: loading_gate(loading, key) for key in COORDINATES},
        saved_sequence=saved,
        events_0p01b=events)
    state_arrays = {
        f"{label}_f_{key}": value[0] for key, value in states.items()}
    state_arrays.update({
        f"{label}_particle_{key}": value[1] for key, value in states.items()})
    state_arrays.update({
        f"{label}_neighbor_{key}": value[2] for key, value in states.items()})
    return block, state_arrays


def coordinate_assessment(coarse, fine, coordinate):
    c_events = {
        key: row["stages"]["3PF"]["delta"][coordinate]
        for key, row in coarse["events_0p01b"].items()}
    f_events = {
        key: row["stages"]["3PF"]["delta"][coordinate]
        for key, row in fine["events_0p01b"].items()}
    return dict(
        coarse_loading=coarse["loading_gate"][coordinate],
        fine_loading=fine["loading_gate"][coordinate],
        coarse_saved_sequence=coarse["saved_sequence"][coordinate],
        fine_saved_sequence=fine["saved_sequence"][coordinate],
        coarse_3PF_event_delta_Pa=c_events,
        fine_3PF_event_delta_Pa=f_events,
        all_event_deltas_negative=bool(
            all(value < 0.0 for value in c_events.values())
            and all(value < 0.0 for value in f_events.values())),
        event_sign_grid_invariant=bool(
            all(c_events[key] * f_events[key] > 0.0 for key in c_events)),
        grid_gate_passed=bool(
            coarse["loading_gate"][coordinate]["sustained_positive_loading"]
            and fine["loading_gate"][coordinate]["sustained_positive_loading"]
            and coarse["saved_sequence"][coordinate]["strictly_increasing"]
            and fine["saved_sequence"][coordinate]["strictly_increasing"]
            and all(value < 0.0 for value in c_events.values())
            and all(value < 0.0 for value in f_events.values())
            and all(c_events[key] * f_events[key] > 0.0 for key in c_events)))


def W_assessment(w_cases):
    rows = []
    for label, _, _ in W_CASES:
        block = w_cases[label]
        last = block["loading"][-1]
        rows.append(dict(
            label=label, W_nm=block["W_nm"], dx_nm=block["dx_nm"],
            cells_per_W=block["cells_per_W"],
            lambda_TJ_excess_N_per_m=last["lambda_TJ_excess_N_per_m"],
            sigma_act_TJ_lambda_Pa=last["sigma_act_TJ_lambda_Pa"],
            sigma_act_TJ_mu_Pa=last["sigma_act_TJ_mu_Pa"]))
    W_m = np.array([row["W_nm"] * 1e-9 for row in rows])
    lam = np.array([row["lambda_TJ_excess_N_per_m"] for row in rows])
    fit = np.polyfit(W_m, lam, 1)
    intercept = float(fit[1])
    changes = [
        float(abs(lam[i + 1] - lam[i]) / max(abs(lam[i]), 1e-300))
        for i in range(len(lam) - 1)]
    convergence_demonstrated = bool(
        np.all(lam != 0.0)
        and np.all(np.sign(lam) == np.sign(lam[0]))
        and changes[-1] < changes[0])
    return dict(
        fixed_resolution_cells_per_W=8.0,
        rows=rows,
        linear_W_to_zero_extrapolation_N_per_m=intercept,
        extrapolated_fraction_of_smallest_W_value=(
            intercept / lam[-1] if lam[-1] != 0.0 else float("nan")),
        successive_relative_changes=changes,
        convergence_demonstrated=convergence_demonstrated,
        passed=convergence_demonstrated,
        note=(
            "bounded W audit only; readiness requires a finite nonzero trend "
            "rather than lambda proportional to numerical W"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    blocks, arrays = {}, {}
    unique_cases = []
    for case in PRIMARY_GRID_CASES + W_CASES:
        if case not in unique_cases:
            unique_cases.append(case)
    for label, W_nm, dx_nm in unique_cases:
        block, state_arrays = run_case(label, W_nm, dx_nm)
        blocks[label] = block
        arrays.update(state_arrays)
    primary = {label: blocks[label] for label, _, _ in PRIMARY_GRID_CASES}
    w_blocks = {label: blocks[label] for label, _, _ in W_CASES}
    assessment = {
        coordinate: coordinate_assessment(
            primary["coarse"], primary["fine"], coordinate)
        for coordinate in COORDINATES}
    assessment["W_convergence"] = W_assessment(w_blocks)
    result = dict(
        frozen_contract=dict(
            geometry="scientific 100 nm C3-crest PR",
            q_definition="physical rigid-body approach delta_RBM in metres",
            q_event_m=B_M,
            q_event_over_b=1.0,
            EXP_floor_barrier_modified=False,
            transport_affinity_separate_from_activation_stress=True,
            hazard_implemented=False),
        nomenclature=dict(
            mu_PF="J/m^3 = N/m^2 = Pa",
            lambda_TJ="J/m^2 = N/m",
            sigma_act_TJ="lambda_TJ/b = Pa",
            diffuse_width_W_is_physical_activation_width=False),
        support=dict(
            definition="(eta1*eta2)*16*f^2*(1-f)^2 intrinsic intersection",
            fitted_width=False, Gaussian=False, radial_cutoff=False),
        primary_grid=primary,
        W_ladder=w_blocks,
        assessment=assessment,
        hazard_implemented=False,
        barrier_refit=False)
    output = os.path.join(OUT_DIR, "tj_activation_gate.json")
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    np.savez_compressed(os.path.join(OUT_DIR, "saved_states.npz"), **arrays)
    print(json.dumps(assessment, indent=2, default=float), flush=True)
    print("OUT", output, flush=True)


if __name__ == "__main__":
    main()
