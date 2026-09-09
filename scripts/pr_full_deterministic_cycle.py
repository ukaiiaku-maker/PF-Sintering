"""Finish the frozen deterministic Plateau--Rayleigh one-b cycle.

The active event is exactly the qualified asymmetric zero-storage-node,
width-free branch-flux, adaptive-explicit path.  This driver adds only:

* restart files (including the event integrator's complete restart state),
* sparse measurements of the actual phase-field free energy, and
* an ordinary sink-OFF phase-field reload after the fixed one-b quota.

No physical-seconds conversion or material-specific diffusivity is used.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(__file__))

from pf_sintering.axisym import axisym_free_energy_gb
from pf_sintering.axisym_numba_kernel import (
    NumbaScratch,
    axisym_gb_face_projected_step_fast,
)
from pf_sintering.exp_barrier_nucleation import creep_equivalent_stress_pa
from pf_sintering.model_time_pr_event import (
    representation_corrected_model_time_boundary_flux_event,
)
from pf_sintering.model_time_transport import ModelTimeGBTransport
from pf_sintering.rigid_rbm_deposition import (
    _axisym_weighted_sum,
    representation_corrected_union_transfer,
)
from pr_concurrent_tj_transport_gate import SOURCE_STATES, load_state
from pr_event_metrology_resolution_audit import B_M
from pr_tj_activation_gate import make_setup
from pr_tj_node_coupling_gate import (
    ATOMIC_VOLUME_M3,
    B_PF_MODEL,
    D_GB_MODEL,
    K_B,
    LONG_EXPLICIT_COURANT,
    SURFACE_MOBILITY_MODEL,
    TAU_EX_MODEL,
    TEMPERATURE_K,
    make_evaluator,
    node_at_state,
)


OUT_DIR = os.environ.get(
    "PR_FULL_CYCLE_OUT", "/private/tmp/pr_full_deterministic_cycle")
TRACE_CSV = os.path.join(OUT_DIR, "cycle_trace.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "cycle_summary.json")
AUTOSAVE = os.path.join(OUT_DIR, "event_autosave.npz")
RELOAD_SAVE = os.path.join(OUT_DIR, "reload_autosave.npz")
CHECKPOINT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
EVENT_SAMPLE_STRIDE = int(os.environ.get("PR_EVENT_SAMPLE_STRIDE", "500"))
EVENT_AUTOSAVE_STRIDE = int(os.environ.get("PR_EVENT_AUTOSAVE_STRIDE", "5000"))
PROBE_QUOTA_FRACTION_B = 1e-5
RELOAD_SAMPLE_MODEL_TIME = 0.05
RELOAD_AUTOSAVE_MODEL_TIME = 0.10
RELOAD_MIN_MODEL_TIME = 0.50
RELOAD_MAX_MODEL_TIME = float(os.environ.get("PR_RELOAD_MAX_MODEL_TIME", "2.0"))

TRACE_FIELDS = (
    "phase", "model_time", "q_over_b", "sink_state", "accepted_step",
    "free_energy_J", "Fq_N", "dVsource_dq_m2", "sigma_act_Pa",
    "transport_affinity_Pa", "mu_GB_Pa", "mu_TJ_Pa", "contact_area_m2",
    "r_TJ_m", "z_TJ_m",
)


def atomic_npz(path, **arrays):
    temporary = path + ".writing"
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def energy(state, setup):
    return float(axisym_free_energy_gb(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], bc_z="noflux"))


def activation_stress_pa(Fq_N, dVsource_dq_m2):
    """Return the stress thermodynamically conjugate to source volume."""
    return creep_equivalent_stress_pa(Fq_N, dVsource_dq_m2)


def source_volume_derivative_m2(state, setup, geom, *, q_m=0.0,
                                half_width_m=1e-5 * B_M):
    """Differentiate the qualified cumulative body-union source kinematics."""
    q_lo = max(0.0, float(q_m) - float(half_width_m))
    q_hi = min(B_M, float(q_m) + float(half_width_m))
    if q_hi <= q_lo:
        raise ValueError("source-volume derivative interval is empty")
    factor = 2.0 * math.pi * setup["dr"] * setup["dz"]

    def source_volume(q_value):
        transfer = representation_corrected_union_transfer(
            *state, q_value, setup["dz"], setup["r_c"], setup["z"],
            geom["z1"])
        return float(transfer[3]["V_source_weighted"]) * factor

    return (source_volume(q_hi) - source_volume(q_lo)) / (q_hi - q_lo)


def make_transport(geom):
    return ModelTimeGBTransport(
        x_d_m=geom["R_z1"] / 2.0,
        kB_J_per_K=K_B,
        temperature_K=TEMPERATURE_K,
        atomic_volume_m3=ATOMIC_VOLUME_M3,
        b_m=B_M,
        D_gb_m2_per_model_time=D_GB_MODEL,
        tau_ex_model=TAU_EX_MODEL)


def event_call(state, setup, geom, evaluator, transport, quota_fraction_b,
               *, event_restart=None, state_callback=None,
               max_increment_fraction_b=0.0025,
               explicit_max_fourth_order_courant=LONG_EXPLICIT_COURANT,
               surface_flux_mobility_m6_per_J_model_time=SURFACE_MOBILITY_MODEL,
               explicit_stability_B_m4_per_model_time=B_PF_MODEL,
               implicit_mullins_B_m4_per_model_time=None,
               branch_mu_filter_length_m=None,
               implicit_max_normal_displacement_m=None,
               implicit_max_tj_displacement_m=None):
    return representation_corrected_model_time_boundary_flux_event(
        *state, transport,
        dr=setup["dr"], dz=setup["dz"], r_c=setup["r_c"], z=setup["z"],
        GB_z_hint=geom["z1"], W=setup["W"], state_evaluator=evaluator,
        surface_flux_mobility_m6_per_J_model_time=(
            surface_flux_mobility_m6_per_J_model_time),
        max_increment_fraction_b=max_increment_fraction_b,
        event_quota_m=quota_fraction_b * B_M,
        max_subincrements=1_000_000,
        explicit_stability_B_m4_per_model_time=(
            explicit_stability_B_m4_per_model_time),
        implicit_mullins_B_m4_per_model_time=(
            implicit_mullins_B_m4_per_model_time),
        branch_mu_filter_length_m=branch_mu_filter_length_m,
        implicit_max_normal_displacement_m=(
            implicit_max_normal_displacement_m),
        implicit_max_tj_displacement_m=implicit_max_tj_displacement_m,
        explicit_max_fourth_order_courant=(
            explicit_max_fourth_order_courant),
        branch_mass_closure_relative_tolerance=1e-5,
        packet_diagnostic_stride=10_000,
        accepted_state_callback=state_callback,
        event_restart=event_restart)


def force_probe(state, setup, geom, evaluator, transport):
    """Forward derivative along a tiny copy of the qualified event path."""
    G0 = energy(state, setup)
    event = event_call(
        state, setup, geom, evaluator, transport, PROBE_QUOTA_FRACTION_B)
    if not event[3]:
        return dict(
            available=False, Fq_N=float("nan"), sigma_act_Pa=float("nan"),
            reason=event[4].get("stop_reason"))
    G1 = energy(event[:3], setup)
    record = evaluator(*state)
    Fq = -(G1 - G0) / (PROBE_QUOTA_FRACTION_B * B_M)
    dVsource_dq = source_volume_derivative_m2(state, setup, geom)
    return dict(
        available=True, Fq_N=Fq,
        dVsource_dq_m2=dVsource_dq,
        sigma_act_Pa=activation_stress_pa(Fq, dVsource_dq),
        contact_area_m2=record["contact_area_m2"],
        probe_steps=event[4]["n_subincrements"],
        probe_delta_G_J=G1 - G0)


def read_trace():
    if not os.path.exists(TRACE_CSV):
        return []
    with open(TRACE_CSV, newline="") as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for row in rows:
        converted = {"phase": row["phase"]}
        for key in TRACE_FIELDS[1:]:
            converted[key] = float(row[key])
        converted["sigma_act_Pa"] = activation_stress_pa(
            converted["Fq_N"], converted["dVsource_dq_m2"])
        out.append(converted)
    return out


def write_trace(rows):
    temporary = TRACE_CSV + ".writing"
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, TRACE_CSV)


def append_measurement(rows, *, phase, model_time, q_over_b, sink_state,
                       accepted_step, G, Fq, dVsource_dq, record, node):
    rows.append(dict(
        phase=phase, model_time=float(model_time), q_over_b=float(q_over_b),
        sink_state=float(sink_state), accepted_step=float(accepted_step),
        free_energy_J=float(G), Fq_N=float(Fq),
        dVsource_dq_m2=float(dVsource_dq),
        sigma_act_Pa=activation_stress_pa(Fq, dVsource_dq),
        transport_affinity_Pa=float(node["transport_affinity_Pa"]),
        mu_GB_Pa=float(node["mu_GB_Pa"]), mu_TJ_Pa=float(node["mu_TJ_Pa"]),
        contact_area_m2=float(record["contact_area_m2"]),
        r_TJ_m=float(record["r_TJ_m"]), z_TJ_m=float(record["z_TJ_m"])))


def restart_arrays(state, restart):
    arrays = dict(
        current_f=state[0], current_particle=state[1],
        current_neighbor=state[2],
        base_f=restart["base_fields"][0],
        base_particle=restart["base_fields"][1],
        base_neighbor=restart["base_fields"][2],
        union_f=restart["union_previous_fields"][0],
        union_particle=restart["union_previous_fields"][1],
        union_neighbor=restart["union_previous_fields"][2],
        cumulative_q_m=np.array(restart["cumulative_q_m"]),
        cumulative_source_weighted=np.array(
            restart["cumulative_source_weighted"]),
        event_time_model=np.array(restart["event_time_model"]),
        accepted_steps_total=np.array(restart["accepted_steps_total"]),
        mass_initial_weighted=np.array(restart["mass_initial_weighted"]),
        cumulative_grain1_m3=np.array(
            restart["cumulative_grain_flux_volume_m3"][1]),
        cumulative_grain2_m3=np.array(
            restart["cumulative_grain_flux_volume_m3"][2]),
        quasistatic_trial_rejections_total=np.array(
            restart.get("quasistatic_trial_rejections_total", 0)),
        restart_fast_precondition_calls_total=np.array(
            restart.get("restart_fast_precondition_calls_total", 0)),
        slow_clock_quadrature_refinements_total=np.array(
            restart.get("slow_clock_quadrature_refinements_total", 0)),
        active_minimum_step_over_b=np.array(
            restart.get("active_minimum_step_over_b", float("nan"))),
        ordinary_fixed_q_calls_total=np.array(
            restart.get("ordinary_fixed_q_calls_total", 0)),
        explicit_max_fourth_order_courant=np.array(
            restart.get("explicit_max_fourth_order_courant", float("nan"))))
    if "event_branch_time_integrator" in restart:
        arrays["event_branch_time_integrator"] = np.array(
            restart["event_branch_time_integrator"])
    decision = restart.get("event_integrator_decision")
    if decision is not None:
        for key, value in decision.items():
            arrays[f"event_selection_{key}"] = np.array(value)
    if "event_origin_row" in restart:
        arrays["event_origin_row_json"] = np.array(json.dumps(
            restart["event_origin_row"]))
    return arrays


def load_event_save(path):
    with np.load(path) as saved:
        state = tuple(saved[key].copy() for key in (
            "current_f", "current_particle", "current_neighbor"))
        restart = dict(
            base_fields=tuple(saved[key].copy() for key in (
                "base_f", "base_particle", "base_neighbor")),
            union_previous_fields=tuple(saved[key].copy() for key in (
                "union_f", "union_particle", "union_neighbor")),
            cumulative_q_m=float(saved["cumulative_q_m"]),
            cumulative_source_weighted=float(saved[
                "cumulative_source_weighted"]),
            event_time_model=float(saved["event_time_model"]),
            accepted_steps_total=int(saved["accepted_steps_total"]),
            mass_initial_weighted=float(saved["mass_initial_weighted"]),
            cumulative_grain_flux_volume_m3={
                1: float(saved["cumulative_grain1_m3"]),
                2: float(saved["cumulative_grain2_m3"])},
            quasistatic_trial_rejections_total=int(
                saved["quasistatic_trial_rejections_total"])
            if "quasistatic_trial_rejections_total" in saved.files else 0)
        restart["restart_fast_precondition_calls_total"] = (
            int(saved["restart_fast_precondition_calls_total"])
            if "restart_fast_precondition_calls_total" in saved.files else 0)
        restart["slow_clock_quadrature_refinements_total"] = (
            int(saved["slow_clock_quadrature_refinements_total"])
            if "slow_clock_quadrature_refinements_total" in saved.files else 0)
        if "active_minimum_step_over_b" in saved.files:
            active_minimum = float(saved["active_minimum_step_over_b"])
            if math.isfinite(active_minimum):
                restart["active_minimum_step_over_b"] = active_minimum
        restart["ordinary_fixed_q_calls_total"] = (
            int(saved["ordinary_fixed_q_calls_total"])
            if "ordinary_fixed_q_calls_total" in saved.files else 0)
        if "event_branch_time_integrator" in saved.files:
            restart["event_branch_time_integrator"] = str(
                saved["event_branch_time_integrator"].item())
        if "explicit_max_fourth_order_courant" in saved.files:
            restart["explicit_max_fourth_order_courant"] = float(
                saved["explicit_max_fourth_order_courant"])
        selection_names = [
            name for name in saved.files if name.startswith("event_selection_")]
        if selection_names:
            decision = {
                name.removeprefix("event_selection_"): saved[name].item()
                for name in selection_names}
            required = {
                "event_integrator_requested", "event_integrator_selected",
                "tau_GB_estimate", "tau_surface_estimate", "timescale_ratio",
                "selection_threshold_low", "selection_threshold_high",
                "intermediate_timescale_warning"}
            if not required.issubset(decision):
                raise RuntimeError("event checkpoint has incomplete integrator decision")
            restart["event_integrator_decision"] = decision
        if "event_origin_row_json" in saved.files:
            restart["event_origin_row"] = json.loads(
                saved["event_origin_row_json"].item())
    return state, restart


def save_event(path, state, restart):
    atomic_npz(path, **restart_arrays(state, restart))


def reconstruct_restart(base_state, current_state, packet, setup, GB_z_hint,
                        event_time_model, accepted_steps_total):
    union = representation_corrected_union_transfer(
        *base_state, packet["q_end_m"], setup["dz"], setup["r_c"],
        setup["z"], GB_z_hint)
    return dict(
        base_fields=tuple(field.copy() for field in base_state),
        union_previous_fields=tuple(field.copy() for field in union[:3]),
        cumulative_q_m=float(packet["q_end_m"]),
        cumulative_source_weighted=float(union[3]["V_source_weighted"]),
        event_time_model=float(event_time_model),
        accepted_steps_total=int(accepted_steps_total),
        mass_initial_weighted=float(_axisym_weighted_sum(
            base_state[0], setup["r_c"])),
        cumulative_grain_flux_volume_m3=dict(
            packet["cumulative_grain_flux_volume_m3"]))


def initial_trace(rows, saved, geom, setup, evaluator, transport):
    if rows:
        return
    equilibrium = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    record, node = node_at_state(
        equilibrium, evaluator, setup, transport, active=False)
    append_measurement(
        rows, phase="equilibrium", model_time=0.0, q_over_b=0.0,
        sink_state=0.0, accepted_step=0, G=energy(equilibrium, setup),
        Fq=0.0, dVsource_dq=source_volume_derivative_m2(
            equilibrium, setup, geom), record=record, node=node)
    for label, model_time in (("t0.25", 0.25), ("t0.625", 0.625), ("t1", 1.0)):
        state = load_state(saved, "coarse", label)
        record, node = node_at_state(
            state, evaluator, setup, transport, active=False)
        probe = force_probe(state, setup, geom, evaluator, transport)
        if not probe["available"]:
            raise RuntimeError(
                f"activation-force probe failed at {label}: {probe['reason']}")
        append_measurement(
            rows, phase="load", model_time=model_time, q_over_b=0.0,
            sink_state=0.0, accepted_step=0, G=energy(state, setup),
            Fq=probe["Fq_N"],
            dVsource_dq=probe["dVsource_dq_m2"],
            record=record, node=node)
    write_trace(rows)


def finish_active_event(rows, base_state, geom, setup, evaluator, transport):
    if os.path.exists(os.path.join(OUT_DIR, "checkpoint_q1p00b.npz")):
        return load_event_save(os.path.join(OUT_DIR, "checkpoint_q1p00b.npz"))
    if os.path.exists(AUTOSAVE):
        state, restart = load_event_save(AUTOSAVE)
        cutoff = restart["accepted_steps_total"]
        rows[:] = [row for row in rows if not (
            row["phase"] == "event" and row["accepted_step"] > cutoff)]
        write_trace(rows)
        print("resuming event", restart["cumulative_q_m"] / B_M,
              cutoff, flush=True)
    else:
        state = tuple(field.copy() for field in base_state)
        restart = None

    q0 = 0.0 if restart is None else restart["cumulative_q_m"] / B_M
    event_rows = [row for row in rows if row["phase"] in ("load", "event")]
    previous = event_rows[-1]
    previous_q = previous["q_over_b"] * B_M
    previous_G = previous["free_energy_J"]
    event_time = 0.0 if restart is None else restart["event_time_model"]
    accepted_total = 0 if restart is None else restart["accepted_steps_total"]
    event_dVsource_dq = source_volume_derivative_m2(
        base_state, setup, geom, q_m=0.5 * B_M,
        half_width_m=0.5 * B_M)

    for target in CHECKPOINT_FRACTIONS:
        if q0 >= target - 1e-13:
            continue
        segment_start_steps = accepted_total
        print("event target", target, "from", q0, "step", accepted_total,
              flush=True)

        def accepted(packet, accepted_state):
            nonlocal event_time, accepted_total, previous_q, previous_G
            event_time += float(packet["transport_dt_model"])
            accepted_total += 1
            should_sample = accepted_total % EVENT_SAMPLE_STRIDE == 0
            should_save = accepted_total % EVENT_AUTOSAVE_STRIDE == 0
            if should_sample:
                G = energy(accepted_state, setup)
                dq = packet["q_end_m"] - previous_q
                Fq = -(G - previous_G) / dq
                record = packet["coordinates_after_boundary_flux"]
                node = packet["node_after"]
                append_measurement(
                    rows, phase="event", model_time=1.0 + event_time,
                    q_over_b=packet["q_end_over_b"], sink_state=1.0,
                    accepted_step=accepted_total, G=G, Fq=Fq,
                    dVsource_dq=event_dVsource_dq,
                    record=record, node=node)
                previous_q = packet["q_end_m"]
                previous_G = G
            if should_save:
                autosave_restart = reconstruct_restart(
                    base_state, accepted_state, packet, setup, geom["z1"],
                    event_time, accepted_total)
                save_event(AUTOSAVE, accepted_state, autosave_restart)
                write_trace(rows)
                print("event autosave", packet["q_end_over_b"],
                      accepted_total, event_time, flush=True)

        event = event_call(
            state, setup, geom, evaluator, transport, target,
            event_restart=restart, state_callback=accepted)
        if not event[3]:
            save_event(AUTOSAVE, event[:3], event[4]["event_restart"])
            write_trace(rows)
            raise RuntimeError(
                "active event failed at q/b="
                f"{event[4]['event_progress_over_b']}: "
                f"{event[4].get('stop_reason')}")
        state = tuple(field.copy() for field in event[:3])
        restart = event[4]["event_restart"]
        event_time = restart["event_time_model"]
        accepted_total = restart["accepted_steps_total"]
        if accepted_total == segment_start_steps:
            raise RuntimeError("event target completed without an accepted step")

        if not (rows[-1]["phase"] == "event"
                and abs(rows[-1]["q_over_b"] - target) < 1e-13):
            G = energy(state, setup)
            dq = target * B_M - previous_q
            Fq = -(G - previous_G) / dq
            record, node = node_at_state(
                state, evaluator, setup, transport, active=True)
            append_measurement(
                rows, phase="event", model_time=1.0 + event_time,
                q_over_b=target, sink_state=1.0,
                accepted_step=accepted_total, G=G, Fq=Fq,
                dVsource_dq=event_dVsource_dq,
                record=record, node=node)
            previous_q, previous_G = target * B_M, G

        checkpoint = os.path.join(
            OUT_DIR, f"checkpoint_q{target:.2f}b".replace(".", "p") + ".npz")
        save_event(checkpoint, state, restart)
        save_event(AUTOSAVE, state, restart)
        write_trace(rows)
        q0 = target
        print("event checkpoint", target, accepted_total, event_time,
              checkpoint, flush=True)
    return state, restart


def save_reload(state, elapsed, step):
    atomic_npz(
        RELOAD_SAVE, current_f=state[0], current_particle=state[1],
        current_neighbor=state[2], reload_elapsed_model=np.array(elapsed),
        reload_step=np.array(step))


def finish_reload(rows, one_b_state, geom, setup, evaluator, transport,
                  event_time):
    if os.path.exists(RELOAD_SAVE):
        with np.load(RELOAD_SAVE) as saved:
            state = tuple(saved[key].copy() for key in (
                "current_f", "current_particle", "current_neighbor"))
            elapsed = float(saved["reload_elapsed_model"])
            step = int(saved["reload_step"])
        rows[:] = [row for row in rows if not (
            row["phase"] == "reload"
            and row["model_time"] > 1.0 + event_time + elapsed + 1e-14)]
        print("resuming reload", elapsed, step, flush=True)
    else:
        state = tuple(field.copy() for field in one_b_state)
        elapsed = 0.0
        step = 0
        record, node = node_at_state(
            state, evaluator, setup, transport, active=False)
        probe = force_probe(state, setup, geom, evaluator, transport)
        append_measurement(
            rows, phase="reload", model_time=1.0 + event_time,
            q_over_b=1.0, sink_state=0.0, accepted_step=step,
            G=energy(state, setup), Fq=probe["Fq_N"],
            dVsource_dq=probe["dVsource_dq_m2"], record=record, node=node)
        write_trace(rows)

    scratch = NumbaScratch(*state[0].shape)
    sample_steps = max(1, int(round(RELOAD_SAMPLE_MODEL_TIME / setup["dt"])))
    autosave_steps = max(1, int(round(
        RELOAD_AUTOSAVE_MODEL_TIME / setup["dt"])))
    maximum_steps = int(math.ceil(RELOAD_MAX_MODEL_TIME / setup["dt"]))
    sigma_history = [
        row["sigma_act_Pa"] for row in rows if row["phase"] == "reload"]

    def clearly_rebuilt():
        if elapsed < RELOAD_MIN_MODEL_TIME or len(sigma_history) < 4:
            return False
        initial = sigma_history[0]
        rise = sigma_history[-1] - min(sigma_history)
        scale = max(abs(initial), 1e6)
        return bool(
            all(np.diff(sigma_history[-4:]) > 0.0)
            and rise >= 0.05 * scale)

    rebuilt = clearly_rebuilt()
    while step < maximum_steps and not rebuilt:
        new_state = axisym_gb_face_projected_step_fast(
            *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
            setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
            setup["M_eta"], setup["W"], scratch)
        state = tuple(field.copy() for field in new_state)
        step += 1
        elapsed = step * setup["dt"]
        if step % autosave_steps == 0:
            save_reload(state, elapsed, step)
            write_trace(rows)
            print("reload autosave", elapsed, step, flush=True)
        if step % sample_steps != 0 and step < maximum_steps:
            continue
        record, node = node_at_state(
            state, evaluator, setup, transport, active=False)
        probe = force_probe(state, setup, geom, evaluator, transport)
        if not probe["available"]:
            save_reload(state, elapsed, step)
            write_trace(rows)
            raise RuntimeError(
                f"reload force probe failed at {elapsed}: {probe['reason']}")
        sigma_history.append(probe["sigma_act_Pa"])
        append_measurement(
            rows, phase="reload", model_time=1.0 + event_time + elapsed,
            q_over_b=1.0, sink_state=0.0, accepted_step=step,
            G=energy(state, setup), Fq=probe["Fq_N"],
            dVsource_dq=probe["dVsource_dq_m2"], record=record, node=node)
        write_trace(rows)
        print("reload sample", elapsed, probe["sigma_act_Pa"] / 1e6,
              "MPa", flush=True)
        rebuilt = clearly_rebuilt()
    save_reload(state, elapsed, step)
    return state, elapsed, rebuilt


def plot_cycle(rows):
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(OUT_DIR, "matplotlib-cache"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([row["model_time"] for row in rows])
    q = np.array([row["q_over_b"] for row in rows])
    sigma = np.array([row["sigma_act_Pa"] for row in rows]) / 1e6
    affinity = np.array([row["transport_affinity_Pa"] for row in rows]) / 1e6
    sink = np.array([row["sink_state"] for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.4), sharex=True)
    axes[0].plot(t, q, color="#0072B2")
    axes[0].set_ylabel("q / b")
    axes[1].plot(t, sigma, color="#D55E00")
    axes[1].set_ylabel(r"$\sigma_{act}$ (MPa)")
    axes[2].plot(t, affinity, color="#009E73")
    axes[2].set_ylabel(r"$\Delta\mu_{transport}$ (MPa)")
    axes[3].step(t, sink, where="post", color="black")
    axes[3].set_ylabel("sink")
    axes[3].set_xlabel("model time")
    axes[3].set_yticks((0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "complete_deterministic_cycle.png")
    fig.savefig(png, dpi=180)
    plt.close(fig)
    return png


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.monotonic()
    geom, setup = make_setup(10.0, 1.25)
    evaluator = make_evaluator(setup, geom)
    transport = make_transport(geom)
    with np.load(SOURCE_STATES) as saved:
        base_state = load_state(saved, "coarse", "t1")
        rows = read_trace()
        initial_trace(rows, saved, geom, setup, evaluator, transport)

    one_b_checkpoint_was_present = os.path.exists(os.path.join(
        OUT_DIR, "checkpoint_q1p00b.npz"))
    one_b_state, restart = finish_active_event(
        rows, base_state, geom, setup, evaluator, transport)
    event_time = restart["event_time_model"]
    _, reload_elapsed, rebuilt = finish_reload(
        rows, one_b_state, geom, setup, evaluator, transport, event_time)
    write_trace(rows)
    figure = plot_cycle(rows)

    event_sigma = np.array([
        row["sigma_act_Pa"] for row in rows if row["phase"] == "event"])
    reload_sigma = np.array([
        row["sigma_act_Pa"] for row in rows if row["phase"] == "reload"])
    volume_jacobian_over_area = np.array([
        row["dVsource_dq_m2"] / row["contact_area_m2"] for row in rows])
    result = dict(
        frozen_contract=dict(
            q_event_m=B_M, checkpoint_fractions_b=CHECKPOINT_FRACTIONS,
            D_GB_m2_per_model_time=D_GB_MODEL,
            surface_mobility_m6_per_J_model_time=SURFACE_MOBILITY_MODEL,
            B_PF_m4_per_model_time=B_PF_MODEL,
            explicit_fourth_order_courant=LONG_EXPLICIT_COURANT,
            physical_seconds_conversion=None,
            material_specific_diffusivity_used=False,
            TJ_resultant_activation_used=False,
            fixed_f_mu_reference_used=False),
        activation_stress_definition=dict(
            Fq="-dG*/dq, finite secants/probes along the actual accepted PF event path",
            G="axisym_free_energy_gb with no-flux z boundary (J)",
            q="rigid densification displacement (m)",
            Fq_units="J/m = N",
            conjugate_volume_jacobian="dV_source/dq from cumulative body-union kinematics",
            dVsource_dq_units="m^2",
            sigma_act="Fq/(dV_source/dq)",
            sigma_act_units="N/m^2 = Pa",
            equilibrium_reference="constructed fully equilibrated geometry is exactly zero by definition",
            dVsource_dq_over_contact_area_min=float(
                np.min(volume_jacobian_over_area)),
            dVsource_dq_over_contact_area_max=float(
                np.max(volume_jacobian_over_area)),
            line_localized_conversion_used=False),
        event=dict(
            completed=bool(restart["cumulative_q_m"] >= B_M * (1 - 1e-12)),
            q_over_b=restart["cumulative_q_m"] / B_M,
            accepted_steps=restart["accepted_steps_total"],
            event_time_model=event_time,
            sigma_start_Pa=float(event_sigma[0]),
            sigma_end_Pa=float(event_sigma[-1]),
            stress_decreased=bool(event_sigma[-1] < event_sigma[0])),
        reload=dict(
            elapsed_model_time=reload_elapsed,
            clearly_rebuilding=rebuilt,
            sigma_start_Pa=float(reload_sigma[0]),
            sigma_end_Pa=float(reload_sigma[-1])),
        complete_deterministic_cycle=bool(
            restart["cumulative_q_m"] >= B_M * (1 - 1e-12)
            and rebuilt),
        active_event_checkpoint_reused=one_b_checkpoint_was_present,
        stochastic_wiring=dict(
            corrected_creep_equivalent_stress_wired=True,
            creep_EXP_floor_first_passage_clock_wired=True,
            repeated_cycles_run=False,
            blocker=(
                "authoritative G0/sigma_star/floor/n, attempt frequency, "
                "and model-time-to-seconds conversion are absent")),
        trace_csv=TRACE_CSV, figure_png=figure,
        wall_seconds_this_invocation=time.monotonic() - started)
    temporary = SUMMARY_JSON + ".writing"
    with open(temporary, "w") as stream:
        json.dump(result, stream, indent=2)
    os.replace(temporary, SUMMARY_JSON)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
