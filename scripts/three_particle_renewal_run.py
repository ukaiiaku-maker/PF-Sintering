"""Restartable full-field stochastic renewal for the selected 0.65 geometry.

The production seed is fixed before the first draw. Source selection and all
mechanical/refinement work are deterministic and cannot inspect these draws.
LEFT and RIGHT clocks are independent; symmetry is never enforced after the
one-time full-domain handoff.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parent)]

import numpy as np
from numba import get_num_threads

from three_particle_buffered_event_probe import BufferedContactEvent,LargerDtBufferedContactEvent
from three_particle_forced_event import ContactEvent, MANIFEST
from three_particle_implicit_run import advance
from three_particle_source_window import (
    DESCENDANT_CROSSING_TOLERANCE_S, advance_source_window)
from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
from pf_sintering.pr_avalanche import (
    AvalancheController, AvalancheState, DescendantBarrier)
from pf_sintering.three_particle_cmc import compatible_chain, map_to_pf
from pf_sintering.three_particle_sharp_initial import load_mapped_sharp_state
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_diagnostics import diagnostics, curvature_watch, radius_profile
from pf_sintering.three_particle_event import (
    load_event_checkpoint, save_event_checkpoint, update_ownership)
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from pf_sintering.three_particle_geometry import grain_volumes, topology_status
from pf_sintering.three_particle_renewal import (
    RootClocks, cumulative_event_quota, locate_first_root)


D = Path("docs/three_particle/production_065")
STATUS_PATH = Path("CAMPAIGN_STATUS.md")
PRODUCTION_SEED = 20260910
DEFAULT_SOURCE = Path(
    "runs/three_particle_production_065/production_initial_2pct.npz")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path, value):
    temporary = path.with_suffix(".writing" + path.suffix)
    temporary.write_text(value)
    os.replace(temporary, path)


def event_progress_over_b(restart):
    return float(restart["cumulative_q_m"])/MANIFEST["b_event_m"]


def event_physical_time_s(event_start_t, event_pause_time_s, restart):
    return (float(event_start_t) + float(event_pause_time_s) +
            float(restart["event_time_model"])*MANIFEST["seconds_per_model_time"])


def incomplete_event_phase(completed, info):
    if completed:
        return None
    reason = info.get("stop_reason")
    if reason == "nonpositive_transport_affinity":
        return "EVENT_TRANSPORT_PAUSED"
    raise RuntimeError(reason or "event stopped without a reason")


def event_checkpoint_due(q_now, last_saved_q):
    cadence = .0025 if q_now >= .98-1e-12 else .01
    return bool(q_now >= 1-1e-12 or
                q_now-last_saved_q >= cadence-1e-12)


EVENT_MILESTONE_TARGETS = (0.25, 0.50, 0.75)


def pending_event_milestones(q_now, existing_names):
    """Return nominal quota milestones first reached by this accepted state."""
    return [target for target in EVENT_MILESTONE_TARGETS
            if f"q_{target:.2f}b.npz" not in existing_names and
            q_now >= target-1e-12]


def save_immutable_event_checkpoint(path, state, restart, *, contact):
    """Write an accepted event state once; an existing milestone is immutable."""
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    save_event_checkpoint(
        path, state, restart, contact=contact,
        label="GENUINE_STOCHASTIC_EVENT_IMMUTABLE_MILESTONE")
    return True


def accepted_reload_step_hint(proposed_h, accepted_h, error):
    """Retain the incoming hint when only the interval remainder clipped it."""
    grown = accepted_h*min(2., max(1., .8/max(error, 1e-12)**.5))
    return max(proposed_h, grown) if accepted_h < proposed_h else grown


def event_accepted_state_cap(minimum_step_over_b):
    """Allow a full one-b event when every accepted step uses its floor."""
    return max(400, int(math.ceil(1.0/float(minimum_step_over_b))))


def require_campaign(source):
    old = json.loads((D/"qualification.json").read_text())
    authorization = json.loads((D/"campaign_authorization.json").read_text())
    overlap = json.loads((D/"buffered_event_probe.json").read_text())
    window = json.loads((D/"source_window_overlap.json").read_text())
    pause_overlap = json.loads(
        (D/"event4_transport_pause_overlap.json").read_text())
    odd_buffer_overlap = json.loads(
        (D/"odd_buffer_reuse_2x_overlap.json").read_text())
    bound_metrics_overlap = json.loads(
        (D/"bound_fast_metrics_overlap.json").read_text())
    reload_no_collapse_overlap = json.loads(
        (D/"post_avalanche_reload_no_collapse_overlap.json").read_text())
    initial = json.loads((D/"production_initial_2pct.json").read_text())
    if not all(old.get(key) is True for key in
               ("reload_qualified", "one_b_qualified",
                "root_quadrature_qualified")):
        raise RuntimeError("campaign requires qualified reload, one-b, and root quadrature")
    if not authorization.get("phase_b_enabled_for_campaign"):
        raise RuntimeError("end-to-end campaign is not authorized")
    if authorization.get("morphology_decision") != "NON_BLOCKING_MONITORED_STRUCTURE":
        raise RuntimeError("campaign morphology decision is missing")
    if not overlap.get("native_overlap_pass") or not window.get("passed"):
        raise RuntimeError("accelerated event/source-window overlap is not qualified")
    if not pause_overlap.get("passed"):
        raise RuntimeError("nucleated transport-pause continuation is not qualified")
    if not odd_buffer_overlap.get("passed"):
        raise RuntimeError("2x odd-step reusable buffers are not qualified")
    if not bound_metrics_overlap.get("passed"):
        raise RuntimeError("already-bound fast metrics are not qualified")
    if not reload_no_collapse_overlap.get("passed"):
        raise RuntimeError("reload step-hint no-collapse policy is not qualified")
    if sha256(source) != initial["output_sha256"]:
        raise RuntimeError("production source differs from the pre-draw selection")
    return dict(
        authorization=authorization, numerical=old,
        event_overlap=overlap, source_window_overlap=window,
        event_transport_pause_overlap=pause_overlap,
        odd_buffer_reuse_2x_overlap=odd_buffer_overlap,
        bound_fast_metrics_overlap=bound_metrics_overlap,
        reload_no_collapse_overlap=reload_no_collapse_overlap,
        production_initial=initial)


def require_earlier_stage_campaign(source):
    authorization=json.loads(Path(
        'docs/three_particle/initial_state_design/campaign_authorization.json').read_text())
    qualification=json.loads(Path(
        'docs/three_particle/initial_state_design/qualification.json').read_text())
    if qualification.get('status')!='SHORT_NUMERICAL_QUALIFICATION_PASS':
        raise RuntimeError('earlier-stage source lacks short numerical qualification')
    if str(source)!=authorization['source'] or sha256(source)!=authorization['source_sha256']:
        raise RuntimeError('earlier-stage production source identity mismatch')
    if authorization.get('physical_parameters_changed') or authorization.get('stress_history_prescribed'):
        raise RuntimeError('earlier-stage authorization violates physical-model isolation')
    old=json.loads((D/'qualification.json').read_text())
    existing=json.loads((D/'campaign_authorization.json').read_text())
    return dict(authorization={**existing,
        'phase_b_enabled_for_campaign':True,
        'source_selection_precedes_random_draw':True,
        'source':str(source),'source_sha256':sha256(source),
        'production_seed':authorization['production_seed'],
        'earlier_stage_initial_state':True},numerical=old,
        earlier_stage_authorization=authorization,short_qualification=qualification)


def require_c2_campaign(source):
    authorization=json.loads(Path(
        'docs/three_particle/mapped_pf_initial_screen/c2_campaign_authorization.json').read_text())
    source_report=json.loads(Path(
        'docs/three_particle/mapped_pf_initial_screen/c2_source_report.json').read_text())
    if source_report.get('status')!='C2_SOURCE_FROZEN_BEFORE_STOCHASTIC_DRAW':
        raise RuntimeError('C2 source was not frozen before stochastic sampling')
    if str(source)!=authorization['source'] or sha256(source)!=authorization['source_sha256']:
        raise RuntimeError('C2 production source identity mismatch')
    if source_report['source_sha256']!=authorization['source_sha256']:
        raise RuntimeError('C2 source report identity mismatch')
    if authorization.get('physical_parameters_changed') or authorization.get('stress_history_prescribed'):
        raise RuntimeError('C2 authorization violates physical-model isolation')
    old=json.loads((D/'qualification.json').read_text())
    existing=json.loads((D/'campaign_authorization.json').read_text())
    return dict(authorization={**existing,
        'phase_b_enabled_for_campaign':True,
        'source_selection_precedes_random_draw':True,
        'source':str(source),'source_sha256':sha256(source),
        'production_seed':authorization['production_seed'],
        'c2_initial_state':True,
        'event_minimum_increment_over_b':authorization['event_minimum_increment_over_b']},
        numerical=old,c2_authorization=authorization,c2_source_report=source_report)


def build_controller(clocks):
    p = MANIFEST["root_barrier_slice"]
    barrier = DescendantBarrier(CompleteExpFloorParams(
        p["G0_eV"], p["Gfloor_eV"], p["a"], p["sigmahat_Pa"], p["n"]), 1.5)
    return AvalancheController(
        barrier=barrier, temperature_K=MANIFEST["temperature_K"],
        attempt_frequency_per_s=(
            MANIFEST["clock_scale"]/MANIFEST["seconds_per_model_time"]),
        b_m=MANIFEST["b_event_m"], correlation_time_s=.009,
        rng=clocks.rng, facilitation_decay_alpha=.70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-name", default="stochastic_seed20260910")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minimum-avalanches", type=int, default=2)
    parser.add_argument("--maximum-production-seconds", type=float, default=3600.)
    parser.add_argument("--maximum-wall-seconds", type=float)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--earlier-stage", action="store_true")
    parser.add_argument("--c2", action="store_true")
    args = parser.parse_args()
    if args.earlier_stage and args.c2:
        raise ValueError("choose only one mapped-source campaign mode")
    if args.minimum_avalanches < 2:
        raise ValueError("production requires at least two complete avalanches")
    gate = (require_c2_campaign(args.source) if args.c2 else
            require_earlier_stage_campaign(args.source) if args.earlier_stage
            else require_campaign(args.source))
    production_seed=(int(gate['c2_authorization']['production_seed']) if args.c2 else
                     int(gate['earlier_stage_authorization']['production_seed'])
                     if args.earlier_stage else PRODUCTION_SEED)
    if args.validate_only:
        print("CAMPAIGN_READY_NO_THRESHOLDS_DRAWN", sha256(args.source))
        return
    required_threads = int(gate["authorization"].get("numba_threads", 1))
    if get_num_threads() != required_threads:
        raise RuntimeError(
            f"campaign requires NUMBA_NUM_THREADS={required_threads}; got {get_num_threads()}")

    old = gate["numerical"]
    event_class = (LargerDtBufferedContactEvent
                   if gate["authorization"].get("native_dt_factor") == 2
                   else BufferedContactEvent if old.get("event_engine") ==
                   "buffered_native" else ContactEvent)
    out = (Path("runs/three_particle_c2_campaign") if args.c2 else
           Path("runs/three_particle_earlier_stage_campaign") if args.earlier_stage
           else Path("runs/three_particle_production_065"))/args.out_name
    if args.earlier_stage or args.c2:
        g, source_fields, source_metadata = load_mapped_sharp_state(args.source)
    else:
        c, offsets = compatible_chain(.65, 119.999*1e-9)
        g = map_to_pf(c, offsets, 4e-9, .5e-9)
        source_fields=source_metadata=None
    rule = json.loads(Path(
        "docs/three_particle/cmc/angle_calibration.json").read_text())["rule"]

    if args.resume:
        launch = json.loads((out/"launch.json").read_text())
        with np.load(out/"trajectory.npz") as data:
            state = tuple(value.copy() for value in data["fields"])
            g["ownership"] = data["ownership"].copy()
            g["gb"] = data["gb"].copy()
            t = float(data["time_s"])
            metadata = json.loads(str(data["metadata"]))
        records = json.loads((out/"history.json").read_text())
        clocks = RootClocks.restore(metadata["root"])
        avalanche = build_controller(clocks)
        avalanche.state = AvalancheState(**metadata["avalanche"])
        avalanche.crossings = metadata.get("avalanche_crossings", [])
        event_number = int(metadata["event_number"])
        completed_avalanches = int(metadata["completed_avalanches"])
        phase = metadata["phase"]
        event_start_t = metadata.get("event_start_t")
        event_pause_time_s = float(metadata.get("event_pause_time_s", 0.))
        status = "RUNNING"
        start_t = float(launch["start_time_s"])
        mass0 = float(launch["material_volume_m3"])
        reference_span = float(launch["densification_reference_length_m"])
        if int(launch["seed"]) != production_seed:
            raise RuntimeError("production seed changed on restart")
        promoted_increment = gate["authorization"].get(
            "event_max_increment_over_b", old["event_max_increment_over_b"])
        if launch.get("accepted_event_increment_over_b") != promoted_increment:
            launch.setdefault("numerical_amendments", []).append(dict(
                event_max_increment_over_b=promoted_increment,
                evidence=gate["authorization"].get("event_increment_overlap"),
                physics_and_acceptance_tolerances_unchanged=True,
                stochastic_state_modified=False))
            launch["accepted_event_increment_over_b"] = promoted_increment
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        if launch.get("numba_threads", 1) != required_threads:
            launch.setdefault("numerical_amendments", []).append(dict(
                numba_threads=required_threads,
                evidence=gate["authorization"].get("native_parallel_overlap"),
                field_linf_difference=0.0, local_stress_difference_Pa=0.0,
                physics_and_acceptance_tolerances_unchanged=True,
                stochastic_state_modified=False))
            launch["numba_threads"] = required_threads
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        if "event_transport_pause_semantics" not in launch:
            pause_semantics = dict(
                source_persists=True, root_clocks_frozen=True,
                descendant_clocks_frozen=True,
                recovery_solver="qualified_sink_off_source_alive_current_field",
                recovery_block_model_time=(
                    MANIFEST["passive_hazard_quadrature_dt_model"]),
                retry_condition="exact_event_integrator_accepts_next_increment",
                no_fitted_affinity_threshold=True,
                evidence="event4_transport_pause_overlap.json")
            launch.setdefault("numerical_amendments", []).append(dict(
                event_transport_pause_semantics=pause_semantics,
                physics_changed=False, stochastic_state_modified=False))
            launch["event_transport_pause_semantics"] = pause_semantics
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        if not launch.get("odd_step_event_buffer_reuse"):
            launch.setdefault("numerical_amendments", []).append(dict(
                odd_step_event_buffer_reuse=True,
                evidence="odd_buffer_reuse_2x_overlap.json",
                fields_Linf=0.0, local_stress_difference_Pa=0.0,
                event_time_model_difference=0.0,
                physics_and_acceptance_tolerances_unchanged=True,
                stochastic_state_modified=False))
            launch["odd_step_event_buffer_reuse"] = True
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        if not launch.get("already_bound_fast_metrics"):
            launch.setdefault("numerical_amendments", []).append(dict(
                already_bound_fast_metrics=True,
                evidence="bound_fast_metrics_overlap.json",
                fields_Linf=0.0, metrics_max_absolute_difference=0.0,
                physics_and_acceptance_tolerances_unchanged=True,
                stochastic_state_modified=False))
            launch["already_bound_fast_metrics"] = True
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        if not launch.get("reload_step_hint_no_collapse"):
            launch.setdefault("numerical_amendments", []).append(dict(
                reload_step_hint_no_collapse=True,
                evidence="post_avalanche_reload_no_collapse_overlap.json",
                field_Linf=gate["reload_no_collapse_overlap"][
                    "field_Linf_to_reference"],
                hazard_increment_relative_difference_max=gate[
                    "reload_no_collapse_overlap"][
                        "hazard_increment_relative_difference_max"],
                field_guard_and_error_tolerance_unchanged=True,
                physics_and_stochastic_state_modified=False))
            launch["reload_step_hint_no_collapse"] = True
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        manifest_minimum = MANIFEST["event_minimum_increment_fraction_b"]
        if launch.get("event_minimum_increment_over_b") != manifest_minimum:
            launch.setdefault("numerical_amendments", []).append(dict(
                event_minimum_increment_over_b=manifest_minimum,
                evidence="bicrystal_launch_manifest.json",
                trigger="event adapter retained its older default floor",
                acceptance_limits_unchanged=True,
                physics_and_stochastic_state_modified=False))
            launch["event_minimum_increment_over_b"] = manifest_minimum
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        if not launch.get("continuous_far_contact_tangent"):
            launch.setdefault("numerical_amendments", []).append(dict(
                continuous_far_contact_tangent=True,
                evidence="event9_far_endpoint_tangent_continuity.json",
                trigger="far-contact last chord crossed a z-grid row",
                acceptance_limits_unchanged=True,
                local_stress_and_root_law_definitions_unchanged=True,
                physics_and_stochastic_state_modified=False))
            launch["continuous_far_contact_tangent"] = True
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
        accepted_state_cap = event_accepted_state_cap(manifest_minimum)
        if launch.get("event_accepted_state_cap") != accepted_state_cap:
            launch.setdefault("numerical_amendments", []).append(dict(
                event_accepted_state_cap=accepted_state_cap,
                derivation="ceil(1b / event minimum increment)",
                trigger="event 9 reached the historical 400-state cap",
                acceptance_limits_unchanged=True,
                physics_and_stochastic_state_modified=False))
            launch["event_accepted_state_cap"] = accepted_state_cap
            atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")
    else:
        if out.exists():
            raise RuntimeError("refusing to overwrite production output; use --resume")
        out.mkdir(parents=True)
        with np.load(args.source) as data:
            state = tuple(value.copy() for value in data["fields"])
            g["ownership"] = data["ownership"].copy()
            g["gb"] = data["gb"].copy()
            t = float(data["t_model"])*MANIFEST["seconds_per_model_time"]
            if bool(data["symmetry_enforcement_enabled"]):
                raise ValueError("production requires a symmetry-released source")
        start_t = t
        pre_draw_provenance=dict(
            status="SOURCE_AND_SEED_FROZEN_BEFORE_THRESHOLD_DRAW",
            source=str(args.source),source_sha256=sha256(args.source),
            seed=production_seed,branch=subprocess.check_output(
                ["git","branch","--show-current"],text=True).strip(),
            branch_head=subprocess.check_output(
                ["git","rev-parse","HEAD"],text=True).strip(),
            working_tree_porcelain=subprocess.check_output(
                ["git","status","--porcelain=v1"],text=True),
            physical_manifest=MANIFEST,campaign_gate=gate,
            threshold_drawn=False)
        atomic_text(out/"pre_draw_manifest.json",
                    json.dumps(pre_draw_provenance,indent=2,default=float)+"\n")
        clocks = RootClocks(np.random.default_rng(production_seed))
        avalanche = build_controller(clocks)
        records = []
        event_number = 0
        completed_avalanches = 0
        phase = "POST_TRANSIENT_NEW_TRAJECTORY"
        event_start_t = None
        event_pause_time_s = 0.
        status = "RUNNING"
        mass0 = float(grain_volumes(state[0], g).sum())
        reference_probe = ContactEvent(g)
        reference_span = reference_probe.metrics(state, 0.)["chain_span_m"]
        launch = dict(
            label="GENUINE_STOCHASTIC_THREE_PARTICLE_RENEWAL",
            seed=production_seed, source=str(args.source),
            source_sha256=sha256(args.source), start_time_s=start_t,
            pre_draw_manifest=str(out/"pre_draw_manifest.json"),
            pre_draw_branch_head=pre_draw_provenance["branch_head"],
            source_selected_before_random_draw=True,
            first_root_thresholds=clocks.threshold.copy(),
            first_root_hazards=clocks.hazard.copy(), stochastic_result=True,
            minimum_completed_avalanches=args.minimum_avalanches,
            maximum_production_seconds=args.maximum_production_seconds,
            maximum_wall_seconds=args.maximum_wall_seconds,
            symmetry_enforcement_enabled=False, reflection_guard_enabled=False,
            no_symmetry_projection=True,
            contact_selected_only_by_localized_root_crossing=True,
            passive_solver="full_native_flux_jacobian",
            passive_preconditioner_reuse=True,
            numba_threads=required_threads,
            accepted_event_increment_over_b=gate["authorization"].get(
                "event_max_increment_over_b", old["event_max_increment_over_b"]),
            event_minimum_increment_over_b=gate["authorization"].get(
                "event_minimum_increment_over_b",MANIFEST["event_minimum_increment_fraction_b"]),
            event_accepted_state_cap=event_accepted_state_cap(
                MANIFEST["event_minimum_increment_fraction_b"]),
            event_checkpoint_cadence_over_b=.01,
            late_event_checkpoint_cadence_over_b=.0025,
            late_event_checkpoint_start_over_b=.98,
            odd_step_event_buffer_reuse=True,
            already_bound_fast_metrics=True,
            reload_step_hint_no_collapse=True,
            event_transport_pause_semantics=dict(
                source_persists=True, root_clocks_frozen=True,
                descendant_clocks_frozen=True,
                recovery_solver="qualified_sink_off_source_alive_current_field",
                recovery_block_model_time=(
                    MANIFEST["passive_hazard_quadrature_dt_model"]),
                retry_condition="exact_event_integrator_accepts_next_increment",
                no_fitted_affinity_threshold=True,
                evidence="event4_transport_pause_overlap.json"),
            material_volume_m3=mass0,
            densification_reference_length_m=reference_span,
            strain_definitions=dict(
                production_densification_strain=(
                    "sum of accepted event quota times b / initial outer-grain centroid separation"),
                geometric_chain_strain=(
                    "1 - current outer-grain centroid separation / initial separation")),
            initial_geometry_kind=('c2_minimally_cleaned_post_mapping' if args.c2 else
                'earlier_stage_sharp_mapped' if args.earlier_stage else 'selected_065_cmc'),
            initial_geometry_metadata=source_metadata,
            all_physical_parameters_unchanged=True,
            no_prescribed_stress_or_geometry_trajectory=True,
            campaign_gate=gate)
        atomic_text(out/"launch.json", json.dumps(launch, indent=2)+"\n")

    reference = ContactEvent(g)
    reference.initial_span = reference_span
    reference.bind(state)
    op = reference.op
    passive = FullJacobianSurfaceDiffusion(op, reuse_preconditioner=True)
    wall = time.perf_counter()
    reload_h_hint = .1
    snapshot_last_time = -float("inf")

    def field_advance(field, seconds):
        nonlocal reload_h_hint
        current = field.copy()
        elapsed = 0.
        h = min(seconds, reload_h_hint)
        reload_progress = bool(int(os.environ.get(
            "THREE_PARTICLE_RELOAD_PROGRESS", "0")))
        while elapsed < seconds-1e-14:
            proposed_h = h
            h = min(h, seconds-elapsed)
            try:
                trial, error = advance(
                    current, h/MANIFEST["seconds_per_model_time"], passive, rule)
                if error > 1:
                    raise RuntimeError(f"embedded field error {error:.9g}")
            except (FloatingPointError, RuntimeError) as exc:
                if reload_progress:
                    print("RELOAD_SUBSTEP_REJECT", "elapsed/target", elapsed,
                          seconds, "h", h, "reason", str(exc), flush=True)
                h *= .2
                if h < 1e-10:
                    raise RuntimeError("reload numerical timestep floor")
                continue
            current = trial
            elapsed += h
            if reload_progress:
                print("RELOAD_SUBSTEP_ACCEPT", "elapsed/target", elapsed,
                      seconds, "h", h, "error", error, flush=True)
            h = accepted_reload_step_hint(proposed_h, h, error)
        reload_h_hint = h
        return current

    def rates(field):
        return {key: value["root_rate_per_s"] for key, value in
                evaluate_contacts(field, op, MANIFEST).items()}

    def record(new_phase, q=0.):
        nonlocal phase
        phase = new_phase
        reference.bind(state)
        volumes = grain_volumes(state[0], g)
        if abs(float(volumes.sum())/mass0-1) > 1e-11:
            raise RuntimeError("trajectory mass guard")
        if state[0].min() < -1e-8 or state[0].max() > 1+1e-8:
            raise RuntimeError("unchanged field guard")
        if np.max(np.abs(sum(state[1:])-state[0])) > 5e-15:
            raise RuntimeError("ownership closure guard")
        contacts = evaluate_contacts(state[0], op, MANIFEST)
        scalar, _ = diagnostics(
            state[0], op, gb_positions=[contacts[key]["z_TJ_m"]
                                        for key in ("LEFT", "RIGHT")])
        chain = reference.metrics(state, q)["chain_strain"]
        areas = {key: np.pi*value["r_n_m"]**2
                 for key, value in contacts.items()}
        area_sum = sum(areas.values())
        radius=radius_profile(state[0],g);valid=np.flatnonzero(np.isfinite(radius))
        surface_area=float(2*np.pi*np.sum(.5*(radius[valid][:-1]+radius[valid][1:])
            *np.hypot(np.diff(g['z'][valid]),np.diff(radius[valid]))))
        interfacial_energy=float(scalar['energy_J'])
        energy_balance_stress=None
        if records:
            prior=records[-1];delta_strain=chain-prior['geometric_chain_strain']
            if abs(delta_strain)>1e-12:
                energy_balance_stress=-(interfacial_energy-prior['total_interfacial_energy_J'])/(mass0*delta_strain)
        row = dict(
            time_s=t, phase=phase, event_number=event_number,
            completed_avalanches=completed_avalanches,
            avalanche_id=clocks.avalanche_id, contact=clocks.active,
            q_over_b=q, chain_strain=chain, geometric_chain_strain=chain,
            contacts=contacts,
            mirror_error_diagnostic=float(np.max(np.abs(state[0]-state[0][::-1]))),
            cumulative_event_quota_over_b=cumulative_event_quota(
                event_number, phase, q),
            production_densification_strain=(
                cumulative_event_quota(event_number, phase, q)
                * MANIFEST["b_event_m"]/reference.initial_span),
            strain_reference_length_m=reference.initial_span,
            descendant_hazard=avalanche.state.descendant_hazard,
            descendant_threshold=avalanche.state.descendant_threshold,
            descendant_H_over_Hstar=(
                avalanche.state.descendant_hazard/
                avalanche.state.descendant_threshold
                if np.isfinite(avalanche.state.descendant_threshold) else None),
            source_amplitude=avalanche.state.source_amplitude,
            source_window_deadline_s=avalanche.state.window_deadline_s,
            center_volume_m3=float(volumes[1]),
            grain_volumes_m3=volumes.tolist(),
            center_particle_mean_local_Pa=.5*(
                contacts["LEFT"]["sigma_local_positive_Pa"]+
                contacts["RIGHT"]["sigma_local_negative_Pa"]),
            cluster_area_weighted_local_Pa=sum(
                areas[key]*contacts[key]["sigma_local_Pa"] for key in areas)/area_sum,
            cluster_area_weighted_integral_Pa=sum(
                areas[key]*contacts[key]["sigma_integral_continuous_Pa"]
                for key in areas)/area_sum,
            surface_area_m2=surface_area,GB_area_m2=area_sum,
            total_interfacial_energy_J=interfacial_energy,
            energy_balance_sintering_stress_Pa=energy_balance_stress,
            energy_balance_definition="-Delta E/(Vsolid Delta geometric_strain), backward record interval; diagnostic only",
            hazards=clocks.hazard.copy(), thresholds=clocks.threshold.copy(),
            H_over_Hstar={key: clocks.hazard[key]/clocks.threshold[key]
                          for key in clocks.hazard}, diagnostics=scalar)
        baseline_contacts=(records[0]["contacts"] if records else contacts)
        row["stress_decomposition"]={}
        row["root_rate_decomposition"]={}
        kBT_eV=8.617333262145e-5*MANIFEST["temperature_K"]
        for key in ("LEFT","RIGHT"):
            c=contacts[key];base=baseline_contacts[key]
            curvature=-.5*(c["kappa1_negative_per_m"]+c["kappa1_positive_per_m"])
            tj=1.5*(math.sin(c["theta_negative_rad"]/2)+math.sin(c["theta_positive_rad"]/2))/c["r_n_m"]
            row["stress_decomposition"][key]=dict(
                curvature_term_Pa=curvature,TJ_term_Pa=tj,total_Pa=curvature+tj)
            sites=math.log(c["r_n_m"]/base["r_n_m"])
            barrier=(base["G_root_eV"]-c["G_root_eV"])/kBT_eV
            row["root_rate_decomposition"][key]=dict(
                delta_ln_Gamma_sites=sites,delta_ln_Gamma_barrier=barrier,
                delta_ln_Gamma=sites+barrier)
        if phase in ("POST_TRANSIENT_NEW_TRAJECTORY", "ONE_B_COMPLETE"):
            row["curvature_watch"] = curvature_watch(
                state[0], op, two_contact_center=True,
                gb_positions=[contacts[key]["z_TJ_m"]
                              for key in ("LEFT", "RIGHT")])
        records.append(row)

    def snapshot(force=False):
        nonlocal snapshot_last_time
        cadence=float(gate.get("c2_authorization",{}).get(
            "snapshot_cadence_physical_s",float("inf")))
        if not args.c2 or (not force and t-snapshot_last_time<cadence-1e-12):
            return
        folder=out/"snapshots";folder.mkdir(exist_ok=True)
        path=folder/f"t_{t:014.9f}_{phase}.npz"
        temporary=path.with_name(path.stem+".writing.npz")
        np.savez_compressed(temporary,f=state[0],ownership=g["ownership"],gb=g["gb"],
            time_s=t,phase=phase,event_number=event_number,
            completed_avalanches=completed_avalanches)
        os.replace(temporary,path);snapshot_last_time=t

    def campaign_status():
        row = records[-1]
        left = row["contacts"]["LEFT"]
        right = row["contacts"]["RIGHT"]
        center_loss = 1.0 - (
            float(row["center_volume_m3"])/float(records[0]["center_volume_m3"]))
        return f"""# Three-particle production campaign status

Updated automatically: {time.strftime('%Y-%m-%d %H:%M:%S')}

- Current phase: `{phase}`
- Status: `{status}`
- Physical time: `{t:.12g} s`
- Time since production source: `{t-start_t:.12g} s`
- Center volume: `{row['center_volume_m3']:.12g} m^3`
- Center-volume loss from production source: `{center_loss:.12g}`
- LEFT/RIGHT local stress: `{left['sigma_local_Pa']/1e6:.9g} / {right['sigma_local_Pa']/1e6:.9g} MPa`
- LEFT/RIGHT hazard ratios: `{row['H_over_Hstar']['LEFT']:.12g} / {row['H_over_Hstar']['RIGHT']:.12g}`
- Active contact: `{clocks.active}`
- Active event: `{event_number if clocks.active else 'none'}`
- Completed avalanches: `{completed_avalanches}`
- Cumulative production strain: `{row['production_densification_strain']:.12g}`
- Geometric strain: `{row['geometric_chain_strain']:.12g}`
- Latest checkpoint: `{out/'trajectory.npz'}`
- Mirror error diagnostic: `{row['mirror_error_diagnostic']:.12g}`
- Numerical health: bounds, mass, ownership closure, topology, and nonlinear guards active
- Symmetry enforcement: disabled; reflection diagnostic only
- No clipping; no fitted correction
"""

    def save():
        metadata = dict(
            status=status, phase=phase, root=clocks.snapshot(),
            avalanche=asdict(avalanche.state),
            avalanche_crossings=avalanche.crossings,
            event_number=event_number,
            completed_avalanches=completed_avalanches,
            event_start_t=event_start_t)
        metadata["event_pause_time_s"] = event_pause_time_s
        temporary = out/"trajectory.writing.npz"
        np.savez_compressed(
            temporary, fields=np.array(state), ownership=g["ownership"],
            gb=g["gb"], time_s=t, metadata=json.dumps(metadata))
        os.replace(temporary, out/"trajectory.npz")
        atomic_text(out/"history.json",
                    json.dumps(records, indent=2, default=float)+"\n")
        atomic_text((out/'CAMPAIGN_STATUS.md') if (args.earlier_stage or args.c2) else STATUS_PATH,
                    campaign_status())

    def save_immutable_lineage_checkpoint(relative_path):
        """Preserve a full stochastic/controller state without overwriting it."""
        path = out/"lineage_milestones"/relative_path
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            status=status, phase=phase, root=clocks.snapshot(),
            avalanche=asdict(avalanche.state),
            avalanche_crossings=avalanche.crossings,
            event_number=event_number,
            completed_avalanches=completed_avalanches,
            event_start_t=event_start_t,
            event_pause_time_s=event_pause_time_s)
        temporary = path.with_suffix(".writing.npz")
        np.savez_compressed(
            temporary, fields=np.array(state), ownership=g["ownership"],
            gb=g["gb"], time_s=t, metadata=json.dumps(metadata))
        os.replace(temporary, path)
        return True

    if not args.resume:
        record(phase)
        snapshot(force=True)
        save()

    try:
        while (completed_avalanches < args.minimum_avalanches and
               t < start_t+args.maximum_production_seconds and
               (args.maximum_wall_seconds is None or
                time.perf_counter()-wall < args.maximum_wall_seconds)):
            if clocks.active is None:
                update_ownership(op, g["ownership"])
                step = min(old["root_macro_step_s"],
                           start_t+args.maximum_production_seconds-t)
                fn, elapsed, increment, contact = locate_first_root(
                    state[0], step, field_advance, rates, clocks,
                    MANIFEST["passive_crossing_tolerance_model"]
                    * MANIFEST["seconds_per_model_time"])
                t += elapsed
                state = (fn, *(g["ownership"]*fn[None]))
                clocks.commit(increment, contact)
                record("ROOT_CROSSING" if contact else "RELOAD")
                snapshot(force=bool(contact))
                if contact:
                    save_immutable_lineage_checkpoint(
                        Path("root_crossings")/
                        f"avalanche_{clocks.avalanche_id:03d}.npz")
                save()
                if topology_status(fn, g)["stop"]:
                    status = "PHYSICAL_TOPOLOGY_TERMINAL"
                    break
                if contact is None:
                    continue

            contact = clocks.active
            pair = (0, 1) if contact == "LEFT" else (1, 2)
            if not avalanche.state.avalanche_active:
                avalanche.start(
                    avalanche_id=clocks.avalanche_id,
                    root_cycle=clocks.avalanche_id, start_time_s=t)

            if phase in ("ROOT_CROSSING", "CHILD_CROSSING"):
                event_number += 1
                avalanche.begin_transit()
                event = event_class(
                    g, pair, max_fast_blocks=old.get("event_max_fast_blocks", 512))
                event_start_t = t
                event_pause_time_s = 0.
                zero = event.run(state, maximum_accepted_states=0)
                checkpoint = out/f"event_{event_number}.npz"
                save_event_checkpoint(
                    checkpoint, state, zero[5]["event_restart"],
                    contact=contact, label="GENUINE_STOCHASTIC_EVENT")
                save_immutable_event_checkpoint(
                    out/"milestones"/f"event_{event_number:02d}"/"start.npz",
                    state, zero[5]["event_restart"], contact=contact)
                record("ACTIVE_ONE_B", 0.)
                save()
                pending_restart = zero[5]["event_restart"]
            elif phase in ("ACTIVE_ONE_B", "EVENT_TRANSPORT_PAUSED",
                           "EVENT_TRANSPORT_RECOVERY"):
                checkpoint = out/f"event_{event_number}.npz"
                state, pending_restart, saved_contact, label = load_event_checkpoint(
                    checkpoint)
                if saved_contact != contact or label != "GENUINE_STOCHASTIC_EVENT":
                    raise RuntimeError("active-event restart identity mismatch")
                t = event_physical_time_s(
                    event_start_t, event_pause_time_s, pending_restart)
                event = event_class(
                    g, pair, max_fast_blocks=old.get("event_max_fast_blocks", 512))
                if phase == "EVENT_TRANSPORT_PAUSED":
                    recovery_seconds = (
                        MANIFEST["passive_hazard_quadrature_dt_model"]
                        * MANIFEST["seconds_per_model_time"])
                    state = advance_source_window(
                        state, recovery_seconds, g, pair, rule,
                        reuse_small_step_preconditioner=True)
                    event_pause_time_s += recovery_seconds
                    t = event_physical_time_s(
                        event_start_t, event_pause_time_s, pending_restart)
                    save_event_checkpoint(
                        checkpoint, state, pending_restart, contact=contact,
                        label="GENUINE_STOCHASTIC_EVENT")
                    record("EVENT_TRANSPORT_RECOVERY",
                           event_progress_over_b(pending_restart))
                    save()
                    continue
            else:
                pending_restart = None

            if phase in ("ACTIVE_ONE_B", "EVENT_TRANSPORT_RECOVERY"):
                checkpoint = out/f"event_{event_number}.npz"

                def progress(packet, fields, restart):
                    nonlocal state, t
                    q_now = float(packet["q_end_over_b"])
                    milestone_dir = out/"milestones"/f"event_{event_number:02d}"
                    existing = ({path.name for path in milestone_dir.glob("*.npz")}
                                if milestone_dir.exists() else set())
                    for target in pending_event_milestones(q_now, existing):
                        save_immutable_event_checkpoint(
                            milestone_dir/f"q_{target:.2f}b.npz", fields, restart,
                            contact=contact)
                        existing.add(f"q_{target:.2f}b.npz")
                    last_saved_q = (float(records[-1]["q_over_b"])
                                    if (records and records[-1]["phase"] ==
                                        "ACTIVE_ONE_B" and
                                        records[-1]["event_number"] == event_number)
                                    else -float("inf"))
                    # The native event still accepts every qualified 0.0025b
                    # increment. Persist only each 0.01b (and the final state),
                    # so an interruption replays at most four deterministic
                    # accepted increments from the last exact checkpoint.
                    if not event_checkpoint_due(q_now, last_saved_q):
                        return
                    save_event_checkpoint(
                        checkpoint, fields, restart, contact=contact,
                        label="GENUINE_STOCHASTIC_EVENT")
                    state = fields
                    t = event_physical_time_s(
                        event_start_t, event_pause_time_s, restart)
                    record("ACTIVE_ONE_B", q_now)
                    save()

                dq = gate["authorization"].get(
                    "event_max_increment_over_b", old["event_max_increment_over_b"])
                result = event.run(
                    state, restart=pending_restart, callback=progress,
                    maximum_step_over_b=dq, initial_step_over_b=dq,
                    minimum_step_over_b=launch["event_minimum_increment_over_b"],
                    maximum_accepted_states=launch["event_accepted_state_cap"])
                state = result[:4]
                info = result[5]
                t = event_physical_time_s(
                    event_start_t, event_pause_time_s, info["event_restart"])
                save_event_checkpoint(
                    out/f"event_{event_number}_final.npz", state,
                    info["event_restart"], contact=contact,
                    label="GENUINE_STOCHASTIC_EVENT")
                save_immutable_event_checkpoint(
                    out/"milestones"/f"event_{event_number:02d}"/"q_1.00b.npz",
                    state, info["event_restart"], contact=contact)
                interrupted_phase = incomplete_event_phase(result[4], info)
                if interrupted_phase is not None:
                    save_event_checkpoint(
                        checkpoint, state, info["event_restart"],
                        contact=contact, label="GENUINE_STOCHASTIC_EVENT")
                    record(interrupted_phase,
                           event_progress_over_b(info["event_restart"]))
                    save()
                    continue
                reference.bind(state)
                record("ONE_B_COMPLETE", info["event_progress_over_b"])
                snapshot(force=True)
                if reference.metrics(state, info["event_progress_over_b"])["topology_stop"]:
                    raise RuntimeError("post-event topology terminal")
                avalanche.complete_transit(t)
                event_start_t = None
                event_pause_time_s = 0.
                record("SOURCE_WINDOW_OPEN")
                save()

            if phase in ("SOURCE_WINDOW_OPEN", "FACILITATED_WINDOW"):
                window_end = avalanche.state.window_deadline_s
                while t < window_end-1e-14:
                    step = min(
                        MANIFEST["passive_hazard_quadrature_dt_model"]
                        * MANIFEST["seconds_per_model_time"], window_end-t)

                    def child_rates(fields):
                        reference.bind(fields)
                        cc = evaluate_contacts(fields[0], op, MANIFEST)[contact]
                        value = avalanche.rate(cc["sigma_local_Pa"], cc["r_n_m"])
                        return {"LEFT": value, "RIGHT": 0.}

                    probe = RootClocks.__new__(RootClocks)
                    probe.active = None
                    probe.hazard = {
                        "LEFT": avalanche.state.descendant_hazard, "RIGHT": 0.}
                    probe.threshold = {
                        "LEFT": avalanche.state.descendant_threshold,
                        "RIGHT": float("inf")}

                    def window_advance(fields, seconds):
                        return advance_source_window(
                            fields, seconds, g, pair, rule,
                            reuse_small_step_preconditioner=True)

                    evolved, elapsed, increment, child = locate_first_root(
                        state, step, window_advance, child_rates, probe,
                        DESCENDANT_CROSSING_TOLERANCE_S)
                    t += elapsed
                    state = tuple(evolved)
                    reference.bind(state)
                    if child:
                        avalanche.commit_crossing(crossing_time_s=t)
                    else:
                        avalanche.state.descendant_hazard += increment["LEFT"]
                        avalanche.state.descendant_total_hazard += increment["LEFT"]
                    record("CHILD_CROSSING" if child else "FACILITATED_WINDOW")
                    snapshot(force=bool(child))
                    if child:
                        save_immutable_lineage_checkpoint(
                            Path("child_crossings")/
                            f"event_{event_number:03d}.npz")
                    save()
                    if child:
                        break
                if phase == "CHILD_CROSSING":
                    continue
                avalanche.expire_window(window_end)
                contacts = evaluate_contacts(state[0], op, MANIFEST)
                g["gb"] = np.array([contacts[key]["z_TJ_m"]
                                     for key in ("LEFT", "RIGHT")])
                clocks.extinct()
                completed_avalanches += 1
                record("AVALANCHE_EXTINCT_REPINNED")
                snapshot(force=True)
                save_immutable_lineage_checkpoint(
                    Path("extinction_and_reload_sources")/
                    f"avalanche_{completed_avalanches:03d}.npz")
                save()

        if status == "RUNNING":
            status = ("TARGET_RENEWAL_CYCLES_COMPLETE" if
                      completed_avalanches >= args.minimum_avalanches else
                      "MAXIMUM_WALL_TIME_REACHED" if
                      args.maximum_wall_seconds is not None and
                      time.perf_counter()-wall >= args.maximum_wall_seconds else
                      "MAXIMUM_PRODUCTION_TIME_REACHED")
    except (RuntimeError, ValueError, FloatingPointError) as error:
        status = "STOPPED: " + str(error)
    save()
    print(status, t, "avalanches", completed_avalanches,
          "wall", time.perf_counter()-wall, flush=True)


if __name__ == "__main__":
    main()
