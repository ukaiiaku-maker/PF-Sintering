#!/usr/bin/env python3
"""One continuous five-avalanche PR renewal trajectory.

The qualified production loading and one-b event mechanics are unchanged.  A
D2 descendant controller is active only between a root crossing and avalanche
extinction.  High-cadence geometry is a diagnostics-only state callback.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads

os.environ["PR_RENEWAL_CASE"] = "fourier"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pr_authoritative_stochastic_ten_event as authoritative  # noqa: E402
import pr_coarsening_stochastic_two_event as base  # noqa: E402
from pf_sintering.axisym_numba_kernel import (  # noqa: E402
    NumbaScratch, axisym_gb_face_projected_step_fast,
)
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.event_integrator_selection import (  # noqa: E402
    estimate_and_select_event_integrator,
)
from pf_sintering.model_time_pr_event import _node_state  # noqa: E402
from pf_sintering.pr_avalanche import AvalancheController, DescendantBarrier  # noqa: E402
from pf_sintering.pr_movie_geometry import FRAME_TYPES, MovieGeometryArchive  # noqa: E402
from pr_coarsening_driven_fourier_loading import (  # noqa: E402
    closed_surface_diffusion_only, integral, reservoir_remove_particle,
)
from pr_experimental_long_sinkoff import build_case  # noqa: E402
from pr_full_deterministic_cycle import (  # noqa: E402
    event_call, load_event_save, make_transport, restart_arrays,
)
from pr_tj_node_coupling_gate import make_evaluator  # noqa: E402


OUT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_five_deltaG0p300_taucorr9ms")
TEMPERATURE_K = 1830.15
SECONDS_PER_MODEL_TIME = 0.01557994316955921
CLOCK_SCALE = 15579943169.55921
B_EVENT_M = 0.25e-9
PRODUCTION_C4 = 0.05
DELTA_G_STEP_EV = 0.300000
TAU_CORR_S = 9.0e-3
FACILITATION_DECAY_ALPHA = 0.6
V_CUTOFF = 0.88
RN_CUTOFF_M = 35e-9
ROOT_ANALYSIS_DT = 0.25
ROOT_MOVIE_DT = 0.025
COMPETING_CROSSING_DT = 0.005
FACILITATED_MOVIE_DT = 0.025
EVENT_MOVIE_DQ = 0.02
N_BRANCH = 256
AVALANCHES_REQUESTED = 5
EVENT_CHECKPOINTS = (0.10, 0.25, 0.50, 0.75, 1.00)
CASE_BUILDER = build_case
EVALUATOR_BUILDER = make_evaluator
RESERVOIR_STEP = closed_surface_diffusion_only
EVENT_SURFACE_MOBILITY_MODEL = None
EVENT_B_PF_MODEL = None
EVENT_BRANCH_TIME_INTEGRATOR = "adaptive_explicit"
DEFAULT_PRODUCTION_EVENT_INTEGRATOR = "auto"
# This module is also a historical standalone replay, whose old behavior must
# remain explicit. New production drivers request the default above.
EVENT_INTEGRATOR_REQUESTED = "packet"
EVENT_INTEGRATOR_SELECTION_THRESHOLD_LOW = 0.1
EVENT_INTEGRATOR_SELECTION_THRESHOLD_HIGH = 10.0
EVENT_INTEGRATOR_INTERMEDIATE_FALLBACK = "packet"
EVENT_SURFACE_ACTIVE_LENGTH_M = None
EVENT_SURFACE_ACTIVE_LENGTH_SOURCE = (
    "driver-provided physical active surface-transport length")
PACKET_EVENT_CALL = event_call
QUASISTATIC_EVENT_CALL = None
EVENT_MAX_INCREMENT_FRACTION_B = 0.0025
TRANSPORT_BUILDER = make_transport
RUN_METADATA = {}
ENFORCE_VOLUME_CUTOFF = True
ENFORCE_RN_CUTOFF = True
ROOT_BARRIER_PENALTY_EV = 0.0
ROOT_THRESHOLD_MULTIPLIER = 1.0
DESCENDANT_ESCALATION_AFTER_AVALANCHES = None
DESCENDANT_ESCALATION_IF_ALL_S_BELOW = None
DESCENDANT_ESCALATION_NEW_DELTA_G_EV = None
REUSED_SEED = None
CONTINUATION = None
ENFORCE_FIRST_ROOT_STRESS_CHECK = True
PERSIST_NUCLEATED_SOURCE_UNTIL_TRANSPORT = False
REQUIRE_UNIQUE_FIELD_TJ = False
MINIMUM_BOUNDARY_CLEARANCE_W = None
GEOMETRY_METRICS = None
TRANSPORT_DIAGNOSTIC = None
PASSIVE_SCRATCH_BUILDER = NumbaScratch
PASSIVE_STEP = axisym_gb_face_projected_step_fast
ENFORCE_EVENT_ENERGY_DESCENT = False
MAX_DESCENDANTS_PER_AVALANCHE = None
STOP_IF_FIRST_AVALANCHE_HAS_NO_DESCENDANTS = False
SPARSE_THERMO_PROBE = None
SPARSE_THERMO_PROBE_DT_MODEL = 5.0
# Diagnostics-only callback: it receives deep copies of native event states
# and may return scalar ledger entries, but can never replace the live state.
EVENT_COPY_ENERGY_DIAGNOSTIC = None
MIN_EQUILIBRIUM_CAP_CLEARANCE_W = None
RADIAL_PADDING_DIAGNOSTICS = {}
_SPARSE_THERMO_PROBE_LAST_T = -math.inf
_SPARSE_THERMO_PROBE_LAST_RESULT = None
K_B_EV_PER_K = 8.617333262145e-5
PHASE_PROJECTOR = ConservativeBoundedPhaseProjector()
SYSTEM_MASS_BOUNDARY = "closed"
SOLID_VOLUME_INITIAL_M3 = None
DENSIFICATION_LENGTH_INITIAL_M = None
SOLID_VOLUME_RELATIVE_TOLERANCE = 1.0e-8
FINAL_MOVIE_RENDERER = None
# Optional production-only passive macrostep path.  It is dormant for legacy
# drivers and is enabled explicitly by a continuation wrapper after the
# passive C4 ceiling has been qualified.
PASSIVE_MACROSTEP_ENABLED = False
PASSIVE_MAX_NUMERICAL_DT_MODEL = None
PASSIVE_HAZARD_QUADRATURE_DT_MODEL = 0.025
PASSIVE_CROSSING_TOLERANCE_MODEL = 1.0e-4
PASSIVE_MIN_NUMERICAL_DT_MODEL = None


def event_time_integrator_kwargs(selected="packet") -> dict:
    """Return the explicitly selected active-event surface time integrator."""
    if selected == "quasistatic":
        return {}
    if selected != "packet":
        raise RuntimeError(f"unknown selected event integrator {selected!r}")
    if EVENT_BRANCH_TIME_INTEGRATOR == "adaptive_explicit":
        return dict(
            explicit_stability_B_m4_per_model_time=EVENT_B_PF_MODEL,
            implicit_mullins_B_m4_per_model_time=None)
    if EVENT_BRANCH_TIME_INTEGRATOR == "adaptive_q_linearly_implicit":
        if EVENT_B_PF_MODEL is None:
            raise RuntimeError("implicit event integration requires EVENT_B_PF_MODEL")
        return dict(
            explicit_stability_B_m4_per_model_time=None,
            implicit_mullins_B_m4_per_model_time=EVENT_B_PF_MODEL)
    raise RuntimeError(
        f"unknown EVENT_BRANCH_TIME_INTEGRATOR={EVENT_BRANCH_TIME_INTEGRATOR!r}")


def event_integrator_decision(current_row, setup, transport, event_restart=None):
    """Select once at event initiation and restore exactly on an event restart."""
    if event_restart is not None and "event_integrator_decision" in event_restart:
        return dict(event_restart["event_integrator_decision"])
    length = (EVENT_SURFACE_ACTIVE_LENGTH_M
              if EVENT_SURFACE_ACTIVE_LENGTH_M is not None
              else setup.get("local_metrology_window_m", 3.0*setup["W"]))
    decision = estimate_and_select_event_integrator(
        EVENT_INTEGRATOR_REQUESTED, transport=transport,
        affinity_Pa=float(current_row["transport_affinity_Pa"]),
        surface_length_m=float(length),
        surface_B_m4_per_model_time=float(EVENT_B_PF_MODEL),
        seconds_per_model_time=SECONDS_PER_MODEL_TIME,
        threshold_low=EVENT_INTEGRATOR_SELECTION_THRESHOLD_LOW,
        threshold_high=EVENT_INTEGRATOR_SELECTION_THRESHOLD_HIGH,
        intermediate_fallback=EVENT_INTEGRATOR_INTERMEDIATE_FALLBACK)
    if decision["intermediate_timescale_warning"]:
        print(
            "INTERMEDIATE TIMESCALE REGIME:\n"
            f"tau_GB/tau_surface = {decision['timescale_ratio']:.9g}\n"
            "neither packet nor quasi-static separation is asymptotically "
            "well justified.\n"
            f"Using configured fallback = {decision['event_integrator_selected']}",
            flush=True)
    return decision


class SolidVolumeInvariantError(RuntimeError):
    """Closed-system mass loss with the exact offending state attached."""

    def __init__(self, message, *, state, volume_m3, relative_error, context):
        super().__init__(message)
        self.state = tuple(np.asarray(field) for field in state)
        self.volume_m3 = float(volume_m3)
        self.relative_error = float(relative_error)
        self.context = str(context)


class ProductionGeometryInvariantError(RuntimeError):
    """A hard closed-geometry/TJ invariant failed at an analyzed state."""


def assert_closed_solid_volume(state, setup, *, context: str) -> float:
    """Hard production invariant for a declared closed mass boundary."""
    volume = integral(np.asarray(state[0]), setup)
    if SYSTEM_MASS_BOUNDARY != "closed":
        return volume
    if SOLID_VOLUME_INITIAL_M3 is None:
        raise RuntimeError("closed system has no initial solid-volume reference")
    relative = volume / SOLID_VOLUME_INITIAL_M3 - 1.0
    if abs(relative) >= SOLID_VOLUME_RELATIVE_TOLERANCE:
        message = (
            "closed solid-volume invariant failed "
            f"during {context}: relative={relative:+.9e} "
            f"tolerance={SOLID_VOLUME_RELATIVE_TOLERANCE:.3e}")
        raise SolidVolumeInvariantError(
            message, state=state, volume_m3=volume,
            relative_error=relative, context=context)
    return volume


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: dict) -> None:
    """Publish a small live-status document without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_continuation_history_prefix(paths) -> list[dict]:
    """Concatenate immutable restart segments into one scalar history.

    A continuation writes its checkpoint state once at the start of the new
    segment.  If that row is semantically identical to the last row of the
    preceding segment, retain only the earlier row.  Distinct state changes at
    the same physical time (for example active transport -> source window at
    exactly q=b) are intentionally retained.
    """
    rows: list[dict] = []
    numeric_identity_fields = (
        "cycle", "avalanche_id", "event_number", "q_event_over_b",
        "Q_avalanche_over_b", "Q_cumulative_over_b", "S_completed",
        "source_alive", "event_transport_active")
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            segment = [dict(row) for row in csv.DictReader(handle)]
        for row in segment:
            if rows:
                previous_time = float(rows[-1]["t_model"])
                current_time = float(row["t_model"])
                tolerance = 64.0 * math.ulp(max(
                    abs(previous_time), abs(current_time), 1.0))
                if current_time < previous_time - tolerance:
                    raise RuntimeError(
                        "continuation history moves backward in model time: "
                        f"{path}: {current_time} < {previous_time}")
                same_numeric_state = all(math.isclose(
                    float(row.get(key, 0.0)),
                    float(rows[-1].get(key, 0.0)),
                    rel_tol=0.0, abs_tol=0.0)
                    for key in numeric_identity_fields)
                if (abs(current_time - previous_time) <= tolerance
                        and row.get("frame_type", "") == rows[-1].get(
                            "frame_type", "")
                        and same_numeric_state):
                    continue
            rows.append(row)
    for sample_id, row in enumerate(rows):
        row["sample_id"] = sample_id
    return rows


def enrich_avalanche_record(record: dict, *, root_row: dict, final_row: dict,
                            scalar_rows: list[dict], subevents: list[dict],
                            delta_G0_eV: float) -> None:
    """Add production duration, affinity, volume, and strain accounting."""
    avalanche_id = int(record["avalanche_id"])
    start = float(record["start_time_s"])
    end = float(record["end_time_s"])
    events = [row for row in subevents
              if int(row["avalanche_id"]) == avalanche_id]
    interval_rows = [row for row in scalar_rows
                     if start - 1e-15 <= float(row["t_model"])
                     * SECONDS_PER_MODEL_TIME <= end + 1e-15]
    propagation = sum(float(row["duration_s"]) for row in events)
    duration = end - start
    waiting = max(duration - propagation, 0.0)
    reference_length = float(root_row.get(
        "center_separation_m", DENSIFICATION_LENGTH_INITIAL_M))
    event_strain = len(events) * B_EVENT_M / reference_length
    volume_errors = [abs(float(row.get(
        "V_solid_relative_error", math.nan))) for row in interval_rows]
    volume_errors = [value for value in volume_errors if math.isfinite(value)]
    affinities = [float(row.get("transport_affinity_Pa", math.nan))
                  for row in interval_rows]
    affinities = [value for value in affinities if math.isfinite(value)]
    record.update(
        root_initiation_time_s=start,
        root_hazard_threshold=float(record["root_threshold"]),
        root_hazard_at_firing=float(root_row["H"]),
        descendant_delta_G0_eV=float(delta_G0_eV),
        number_of_b_events=len(events),
        propagation_duration_s=propagation,
        interevent_waiting_duration_s=waiting,
        total_avalanche_duration_s=duration,
        initial_stress_Pa=float(root_row["sigma_local_Pa"]),
        final_stress_Pa=float(final_row["sigma_local_Pa"]),
        stress_drop_Pa=(float(root_row["sigma_local_Pa"])
                        - float(final_row["sigma_local_Pa"])),
        integral_stress_drop_Pa=(float(root_row["sigma_integral_Pa"])
                                 - float(final_row["sigma_integral_Pa"])),
        initial_neck_radius_m=float(root_row["r_n_m"]),
        final_neck_radius_m=float(final_row["r_n_m"]),
        densification_reference_length_m=reference_length,
        avalanche_densification_strain=event_strain,
        cumulative_densification_strain=float(final_row.get(
            "densification_strain", math.nan)),
        minimum_affinity_Pa=(min(affinities) if affinities else math.nan),
        final_affinity_Pa=float(final_row.get(
            "transport_affinity_Pa", math.nan)),
        maximum_abs_deltaV_over_V=(max(volume_errors)
                                   if volume_errors else math.nan),
        termination_reason="natural_source_extinction")


def save_state(path: Path, state, **metadata) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, f=state[0], particle=state[1], substrate=state[2],
            **{key: np.asarray(value) for key, value in metadata.items()})
    os.replace(temporary, path)


class ScalarOutput:
    """Normal-cadence scalar/raw-contour output; independent of movie HDF5."""

    def __init__(self, out: Path, *, history_prefix_rows=None):
        self.out = out
        self.history_prefix_rows = [
            dict(row) for row in (history_prefix_rows or [])]
        self.rows: list[dict] = []
        self.contours: list[dict] = []

    def add(self, row: dict, branches: dict, **extra) -> dict:
        merged = dict(
            row, **extra,
            sample_id=len(self.history_prefix_rows) + len(self.rows))
        merged["t_s"] = merged["t_model"] * SECONDS_PER_MODEL_TIME
        self.rows.append(merged)
        self.contours.append(dict(
            sample_id=merged["sample_id"],
            z_negative=np.asarray(branches["negative"]["z_m"]),
            r_negative=np.asarray(branches["negative"]["r_m"]),
            z_positive=np.asarray(branches["positive"]["z_m"]),
            r_positive=np.asarray(branches["positive"]["r_m"])))
        return merged

    def flush(self) -> None:
        write_csv(
            self.out / "avalanche_renewal_history.csv",
            self.history_prefix_rows + self.rows)
        arrays = {}
        for item in self.contours:
            sample_id = item["sample_id"]
            for key, value in item.items():
                if key != "sample_id":
                    arrays[f"sample_{sample_id:05d}_{key}"] = value
        temporary = self.out / "avalanche_renewal_sparse_contours.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, self.out / "avalanche_renewal_sparse_contours.npz")


def geometry_invalid(row: dict) -> list[str]:
    reasons = []
    if ENFORCE_VOLUME_CUTOFF and row["Vp_over_Vp_cycle"] < V_CUTOFF:
        reasons.append("Vp_over_Vp_cycle_below_0.88")
    if ENFORCE_RN_CUTOFF and row["r_n_m"] < RN_CUTOFF_M:
        reasons.append("r_n_below_35_nm")
    return reasons


def movie_payload(row: dict, *, avalanche_id: int, event_number: int,
                  avalanche_active: int, sink_state: int,
                  q_event_over_b: float, Q_avalanche_over_b: float,
                  Q_cumulative_over_b: float, S_completed: int,
                  pending_children: int, controller=None,
                  source_alive: int | None = None,
                  event_transport_active: int | None = None) -> dict:
    state = None if controller is None else controller.state
    if source_alive is None:
        source_alive = int(avalanche_active)
    if event_transport_active is None:
        event_transport_active = 0
    return dict(
        t_model=float(row["t_model"]),
        t_s=float(row["t_model"] * SECONDS_PER_MODEL_TIME),
        cycle=int(row["cycle"]), avalanche_id=int(avalanche_id),
        event_number=int(event_number), sink_state=int(sink_state),
        avalanche_active=int(avalanche_active),
        source_alive=int(source_alive),
        event_transport_active=int(event_transport_active),
        q_event_over_b=float(q_event_over_b),
        Q_avalanche_over_b=float(Q_avalanche_over_b),
        Q_cumulative_over_b=float(Q_cumulative_over_b),
        sigma_local_Pa=float(row["sigma_local_Pa"]),
        sigma_integral_Pa=float(row["sigma_integral_Pa"]),
        r_neck_m=float(row["r_n_m"]),
        Vp_over_Vp_cycle=float(row["Vp_over_Vp_cycle"]),
        z_TJ_m=float(row["z_TJ_m"]), r_TJ_m=float(row["r_n_m"]),
        z_GB_m=float(row["z_TJ_m"]), r_GB_m=float(row["r_n_m"]),
        root_hazard=float(row.get("H", np.nan)),
        root_threshold=float(row.get("H_threshold", np.nan)),
        descendant_hazard=(float(state.descendant_hazard)
                           if state is not None else np.nan),
        descendant_threshold=(float(state.descendant_threshold)
                              if state is not None else np.nan),
        S_completed=int(S_completed),
        pending_children=(int(state.pending_children)
                          if state is not None else int(pending_children)))


def append_movie(movie: MovieGeometryArchive, branches: dict, row: dict,
                 *, frame_type: str, avalanche_id: int, event_number: int,
                 avalanche_active: int, sink_state: int,
                 q_event_over_b: float, Q_avalanche_over_b: float,
                 Q_cumulative_over_b: float, S_completed: int,
                 pending_children: int = 0, controller=None,
                 source_alive: int | None = None,
                 event_transport_active: int | None = None,
                 flush: bool = False) -> int:
    if source_alive is None:
        source_alive = int(avalanche_active or frame_type == "root_nucleation")
    if event_transport_active is None:
        event_transport_active = int(frame_type in (
            "active_1b_transit", "nucleated_transport_wait"))
    payload = movie_payload(
        row, avalanche_id=avalanche_id, event_number=event_number,
        avalanche_active=avalanche_active, sink_state=sink_state,
        q_event_over_b=q_event_over_b,
        Q_avalanche_over_b=Q_avalanche_over_b,
        Q_cumulative_over_b=Q_cumulative_over_b,
        S_completed=S_completed, pending_children=pending_children,
        controller=controller, source_alive=source_alive,
        event_transport_active=event_transport_active)
    return movie.append(branches, payload, frame_type=frame_type, flush=flush)


def output_extra(*, avalanche_id: int, event_number: int,
                 avalanche_active: int, q_event: float,
                 Q_avalanche: float, Q_cumulative: float,
                 S_completed: int, controller=None, frame_type: str,
                 source_alive: int | None = None,
                 event_transport_active: int | None = None) -> dict:
    state = None if controller is None else controller.state
    if source_alive is None:
        source_alive = int(avalanche_active or frame_type == "root_nucleation")
    if event_transport_active is None:
        event_transport_active = int(frame_type in (
            "active_1b_transit", "nucleated_transport_wait"))
    return dict(
        avalanche_id=avalanche_id, event_number=event_number,
        avalanche_active=avalanche_active, q_event_over_b=q_event,
        source_alive=int(source_alive),
        event_transport_active=int(event_transport_active),
        Q_avalanche_over_b=Q_avalanche,
        Q_cumulative_over_b=Q_cumulative, S_completed=S_completed,
        pending_children=(state.pending_children if state is not None else 0),
        descendant_hazard=(state.descendant_hazard if state is not None else np.nan),
        descendant_threshold=(state.descendant_threshold if state is not None else np.nan),
        frame_type=frame_type)


def measure(state, *, t_model: float, cycle: int, sink: int, q: float,
            qcum: float, hazard: float, threshold: float, setup, geom,
            evaluator, vp_cycle0: float):
    row, branches = base.measure(
        state, t_model=t_model, cycle=cycle, sink=sink, q=q, qcum=qcum,
        hazard=hazard, threshold=threshold, clock_scale=CLOCK_SCALE,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp_cycle0)
    fitted_barrier = float(row["G_star_eV"])
    penalty = float(ROOT_BARRIER_PENALTY_EV)
    if penalty < 0.0 or not math.isfinite(penalty):
        raise ValueError("ROOT_BARRIER_PENALTY_EV must be finite and nonnegative")
    row["G_fit_star_eV"] = fitted_barrier
    row["G_root_formation_penalty_eV"] = penalty
    row["G_star_eV"] = fitted_barrier + penalty
    row["Gamma_per_model_time"] *= math.exp(
        -penalty / (K_B_EV_PER_K * TEMPERATURE_K))
    f, particle, substrate = (np.asarray(field) for field in state)
    radial_weights = np.asarray(setup["r_c"])[None, :]
    z_coordinates = np.asarray(setup["z"])[:, None]
    particle_weight = particle * radial_weights
    substrate_weight = substrate * radial_weights
    z_particle = float(
        np.sum(particle_weight * z_coordinates) / np.sum(particle_weight))
    z_substrate = float(
        np.sum(substrate_weight * z_coordinates) / np.sum(substrate_weight))
    center_separation = abs(z_particle - z_substrate)
    vsolid = assert_closed_solid_volume(state, setup, context="analysis")
    vparticle = integral(particle, setup)
    vsubstrate = integral(substrate, setup)
    closure = float(np.max(np.abs(f - particle - substrate)))
    row.update(
        f_min=float(np.min(f)), f_max=float(np.max(f)),
        maximum_partition_closure=closure,
        V_solid_m3=vsolid, V_particle_m3=vparticle,
        V_substrate_m3=vsubstrate,
        V_solid_relative_error=(
            vsolid / SOLID_VOLUME_INITIAL_M3 - 1.0
            if SOLID_VOLUME_INITIAL_M3 is not None else math.nan),
        partition_volume_closure_relative=(
            (vsolid - vparticle - vsubstrate) / max(abs(vsolid), 1e-300)),
        z_centroid_particle_m=z_particle,
        z_centroid_substrate_m=z_substrate,
        center_separation_m=center_separation,
        densification_strain=(
            qcum * B_EVENT_M / DENSIFICATION_LENGTH_INITIAL_M
            if DENSIFICATION_LENGTH_INITIAL_M is not None else math.nan))
    if closure >= 1.0e-12:
        raise ProductionGeometryInvariantError(
            f"phase-closure invariant failed: max={closure:.9e}")
    if GEOMETRY_METRICS is not None:
        row.update(GEOMETRY_METRICS(state, setup, geom))
    if TRANSPORT_DIAGNOSTIC is not None:
        row.update(TRANSPORT_DIAGNOSTIC(state, setup, geom, evaluator))
    if (REQUIRE_UNIQUE_FIELD_TJ
            and int(row.get("tj_candidate_count", -1)) != 1):
        raise ProductionGeometryInvariantError(
            "continuous field TJ is no longer unique: "
            f"candidate_count={row.get('tj_candidate_count')}")
    if MINIMUM_BOUNDARY_CLEARANCE_W is not None:
        clearance = min(
            float(row["radial_margin_m"]),
            float(row["left_z_margin_m"]),
            float(row["right_z_margin_m"]))
        if clearance <= float(MINIMUM_BOUNDARY_CLEARANCE_W) * setup["W"]:
            raise ProductionGeometryInvariantError(
                "interface approached padded boundary: "
                f"clearance/W={clearance/setup['W']:.6g}")
    if all(key in row for key in (
            "left_z_margin_m", "right_z_margin_m", "radial_margin_m")):
        row.update(
            left_z_margin_W=float(row["left_z_margin_m"] / setup["W"]),
            right_z_margin_W=float(row["right_z_margin_m"] / setup["W"]),
            radial_margin_W=float(row["radial_margin_m"] / setup["W"]))
    v1 = float(row["V_particle_m3"])
    v2 = float(row["V_substrate_m3"])
    vsolid = float(row["V_solid_m3"])
    x_equilibrium = float(np.clip(
        setup["gamma_gb"] / (2.0*setup["gamma_s"]), -1.0, 1.0))
    sphere = [(3.0*volume/(4.0*math.pi))**(1.0/3.0)
              for volume in (v1, v2)]
    cap = [(3.0*volume / (
        math.pi*(1.0+x_equilibrium)**2*(2.0-x_equilibrium)))**(1.0/3.0)
           for volume in (v1, v2)]
    contact = [radius*math.sqrt(1.0-x_equilibrium*x_equilibrium)
               for radius in cap]
    height = [radius*(1.0+x_equilibrium) for radius in cap]
    all_sphere = (3.0*vsolid/(4.0*math.pi))**(1.0/3.0)
    radial_box = float(setup["r_f"][-1])
    row.update(
        domain_envelope_x_equilibrium=x_equilibrium,
        domain_envelope_psi_equilibrium_deg=(
            math.degrees(2.0*math.acos(x_equilibrium))),
        R_sph_1_m=sphere[0], R_sph_2_m=sphere[1],
        R_cap_1_m=cap[0], R_cap_2_m=cap[1],
        r_contact_1_m=contact[0], r_contact_2_m=contact[1],
        h_cap_1_m=height[0], h_cap_2_m=height[1],
        pair_equilibrium_axial_span_m=height[0]+height[1],
        R_all_m=all_sphere,
        radial_box_m=radial_box,
        equilibrium_cap_radial_clearance_W=(
            (radial_box-max(contact))/setup["W"]),
        all_mass_sphere_radial_clearance_W=(
            (radial_box-all_sphere)/setup["W"]),
        domain_envelope_diagnostic_only=1,
        equilibrium_angle_imposed=0)
    global _SPARSE_THERMO_PROBE_LAST_T, _SPARSE_THERMO_PROBE_LAST_RESULT
    if SPARSE_THERMO_PROBE is not None:
        checkpoint_q = sink and any(
            abs(q - target) <= 1.0e-10 for target in EVENT_CHECKPOINTS)
        due = (t_model == 0.0 or checkpoint_q
               or t_model - _SPARSE_THERMO_PROBE_LAST_T
               >= SPARSE_THERMO_PROBE_DT_MODEL - 1.0e-12)
        if due:
            _SPARSE_THERMO_PROBE_LAST_RESULT = SPARSE_THERMO_PROBE(
                state, setup, geom)
            _SPARSE_THERMO_PROBE_LAST_T = t_model
        if _SPARSE_THERMO_PROBE_LAST_RESULT is not None:
            row.update(_SPARSE_THERMO_PROBE_LAST_RESULT)
    return row, branches


def advance_pf_steps(state, nstep: int, *, setup, scratches, which: int):
    for _ in range(nstep):
        destination = 0 if which != 0 else 1
        state = PASSIVE_STEP(
            *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
            setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
            setup["M_eta"], setup["W"], scratches[destination])
        which = destination
        if PHASE_PROJECTOR is not None:
            state, _ = PHASE_PROJECTOR(state, setup)
        f, particle, substrate, _ = RESERVOIR_STEP(state, setup)
        state = f, particle, substrate
        assert_closed_solid_volume(
            state, setup, context="closed passive PF update")
    return state, which


def advance_pf_duration(state, duration_model: float, *, setup, dt_ceiling,
                        scratches, which: int):
    """Advance an exact passive duration with a qualified explicit ceiling.

    The last numerical step is shortened so model time is preserved exactly.
    Every accepted numerical state still passes the existing bounded
    projection and closed-volume invariant.
    """
    remaining = float(duration_model)
    tolerance = 64.0 * math.ulp(max(abs(remaining), 1.0))
    while remaining > tolerance:
        dt = min(float(dt_ceiling), remaining)
        destination = 0 if which != 0 else 1
        state = PASSIVE_STEP(
            *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
            setup["r_c"], setup["r_f"], dt, setup["M_s"],
            setup["M_eta"], setup["W"], scratches[destination])
        which = destination
        if PHASE_PROJECTOR is not None:
            state, _ = PHASE_PROJECTOR(state, setup)
        f, particle, substrate, _ = RESERVOIR_STEP(state, setup)
        state = f, particle, substrate
        assert_closed_solid_volume(
            state, setup, context="adaptive passive PF update")
        remaining -= dt
    return state, which


def _root_crossing_bisection(
        state_low, *, duration_model, t_low, hazard_low, rate_low,
        cycle, total_completed, threshold, setup, geom, vp_cycle0,
        dt_ceiling):
    """Locate first passage by rollback/bisection from an accepted low state."""
    low_state = tuple(np.asarray(field).copy() for field in state_low)
    low_t = float(t_low)
    low_hazard = float(hazard_low)
    low_rate = float(rate_low)
    width = float(duration_model)
    high = None
    while width > PASSIVE_CROSSING_TOLERANCE_MODEL:
        trial_duration = 0.5 * width
        scratches = [PASSIVE_SCRATCH_BUILDER(*low_state[0].shape)
                     for _ in range(2)]
        trial_state, _ = advance_pf_duration(
            tuple(field.copy() for field in low_state), trial_duration,
            setup=setup, dt_ceiling=dt_ceiling, scratches=scratches, which=0)
        trial_t = low_t + trial_duration
        trial, branches = _measure_with_fresh_tracker(
            trial_state, t_model=trial_t, avalanche_id=cycle,
            total_completed=total_completed, root_hazard=low_hazard,
            root_threshold=threshold, vp_cycle0=vp_cycle0,
            setup=setup, geom=geom)
        trial_rate = float(trial["Gamma_per_model_time"])
        trial_hazard = low_hazard + 0.5 * (
            low_rate + trial_rate) * trial_duration
        trial["H"] = trial_hazard
        trial["H_over_threshold"] = trial_hazard / threshold
        if trial_hazard >= threshold:
            high = (trial_state, trial_t, trial_hazard, trial_rate,
                    trial, branches)
            width = trial_duration
        else:
            low_state = tuple(np.asarray(field).copy() for field in trial_state)
            low_t = trial_t
            low_hazard = trial_hazard
            low_rate = trial_rate
            width -= trial_duration
    if high is None:
        scratches = [PASSIVE_SCRATCH_BUILDER(*low_state[0].shape)
                     for _ in range(2)]
        high_state, _ = advance_pf_duration(
            tuple(field.copy() for field in low_state), width,
            setup=setup, dt_ceiling=dt_ceiling, scratches=scratches, which=0)
        high_t = low_t + width
        high_row, high_branches = _measure_with_fresh_tracker(
            high_state, t_model=high_t, avalanche_id=cycle,
            total_completed=total_completed, root_hazard=low_hazard,
            root_threshold=threshold, vp_cycle0=vp_cycle0,
            setup=setup, geom=geom)
        high_rate = float(high_row["Gamma_per_model_time"])
        high_hazard = low_hazard + 0.5 * (
            low_rate + high_rate) * width
        high_row["H"] = high_hazard
        high_row["H_over_threshold"] = high_hazard / threshold
        high = (high_state, high_t, high_hazard, high_rate,
                high_row, high_branches)
    return high


def wait_for_root_or_censor_macrostep(
        state, *, cycle: int, t_model: float, threshold: float,
        total_completed: int, setup, geom, evaluator, scalar, movie,
        first_reload: bool, initial_hazard: float = 0.0,
        initial_vp_cycle0: float | None = None):
    """Qualified passive C4 macrosteps with state-rate hazard localization."""
    vp_cycle0 = (integral(state[1], setup) if initial_vp_cycle0 is None
                 else float(initial_vp_cycle0))
    current, _ = measure(
        state, t_model=t_model, cycle=cycle, sink=0, q=0.0,
        qcum=total_completed, hazard=initial_hazard, threshold=threshold,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp_cycle0)
    hazard = float(initial_hazard)
    previous_rate = float(current["Gamma_per_model_time"])
    elapsed = 0.0
    dt_ceiling = float(PASSIVE_MAX_NUMERICAL_DT_MODEL)
    dt_floor = float(PASSIVE_MIN_NUMERICAL_DT_MODEL
                     if PASSIVE_MIN_NUMERICAL_DT_MODEL is not None
                     else setup["dt"] / 16.0)
    while True:
        macro_start_state = tuple(np.asarray(field).copy() for field in state)
        macro_start_elapsed = elapsed
        macro_start_hazard = hazard
        macro_start_rate = previous_rate
        segment_elapsed = 0.0
        invalid = None
        crossing = False
        trial = None
        branches = None
        while segment_elapsed < ROOT_ANALYSIS_DT - 1.0e-15:
            segment = min(
                PASSIVE_HAZARD_QUADRATURE_DT_MODEL,
                ROOT_ANALYSIS_DT - segment_elapsed)
            segment_state = tuple(np.asarray(field).copy() for field in state)
            segment_t = t_model + elapsed
            segment_hazard = hazard
            segment_rate = previous_rate
            try:
                scratches = [PASSIVE_SCRATCH_BUILDER(*state[0].shape)
                             for _ in range(2)]
                state, _ = advance_pf_duration(
                    state, segment, setup=setup, dt_ceiling=dt_ceiling,
                    scratches=scratches, which=0)
            except (FloatingPointError, ProductionGeometryInvariantError,
                    SolidVolumeInvariantError):
                state = macro_start_state
                elapsed = macro_start_elapsed
                hazard = macro_start_hazard
                previous_rate = macro_start_rate
                dt_ceiling *= 0.5
                if dt_ceiling < dt_floor:
                    raise
                print(
                    f"PASSIVE MACROSTEP ROLLBACK a{cycle} "
                    f"t={t_model+elapsed:.6f} dt_ceiling={dt_ceiling:.12g}",
                    flush=True)
                break
            elapsed += segment
            segment_elapsed += segment
            trial, branches = measure(
                state, t_model=t_model + elapsed, cycle=cycle, sink=0, q=0.0,
                qcum=total_completed, hazard=hazard, threshold=threshold,
                setup=setup, geom=geom, evaluator=evaluator,
                vp_cycle0=vp_cycle0)
            trial_rate = float(trial["Gamma_per_model_time"])
            trial_hazard = hazard + 0.5 * (
                previous_rate + trial_rate) * segment
            trial["H"] = trial_hazard
            trial["H_over_threshold"] = trial_hazard / threshold
            invalid = geometry_invalid(trial)
            crossing = trial_hazard >= threshold
            if crossing and hazard < threshold:
                (state, crossing_t, hazard, previous_rate,
                 trial, branches) = _root_crossing_bisection(
                    segment_state, duration_model=segment,
                    t_low=segment_t, hazard_low=segment_hazard,
                    rate_low=segment_rate, cycle=cycle,
                    total_completed=total_completed, threshold=threshold,
                    setup=setup, geom=geom, vp_cycle0=vp_cycle0,
                    dt_ceiling=dt_ceiling)
                elapsed = crossing_t - t_model
                # Commit the accepted crossing through the production tracker.
                trial, branches = measure(
                    state, t_model=crossing_t, cycle=cycle, sink=0, q=0.0,
                    qcum=total_completed, hazard=hazard, threshold=threshold,
                    setup=setup, geom=geom, evaluator=evaluator,
                    vp_cycle0=vp_cycle0)
                trial["H"] = hazard
                trial["H_over_threshold"] = hazard / threshold
                invalid = geometry_invalid(trial)
                break
            hazard = trial_hazard
            previous_rate = trial_rate
            if invalid:
                break
        else:
            pass
        # A numerical rejection restarts the same atomic macrostep.
        if elapsed == macro_start_elapsed and state is macro_start_state:
            continue
        if trial is None:
            continue
        append_wait_buffer(
            movie, [(trial, branches)], avalanche_id=cycle,
            total_completed=total_completed, first_reload=first_reload,
            root_nucleation=crossing and not invalid)
        first_reload = False
        scalar.add(
            trial, branches, **output_extra(
                avalanche_id=cycle if crossing and not invalid else 0,
                event_number=0, avalanche_active=0, q_event=0.0,
                Q_avalanche=0.0, Q_cumulative=total_completed,
                S_completed=0,
                frame_type=("root_nucleation"
                            if crossing and not invalid else "loading")))
        scalar.flush()
        save_state(
            OUT / "checkpoints" / f"avalanche{cycle}_waiting_latest.npz",
            state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
            Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"],
            r_n_m=trial["r_n_m"])
        print(
            f"RENEWAL MACRO WAIT a{cycle} t={elapsed:.3f} "
            f"t_s={(t_model+elapsed)*SECONDS_PER_MODEL_TIME:.6f} "
            f"sigL={trial['sigma_local_Pa']/1e6:.3f}MPa "
            f"sigI={trial['sigma_integral_Pa']/1e6:.3f}MPa "
            f"dmu={trial.get('delta_mu_GB_minus_TJ_local_Pa', math.nan)/1e6:.3f}MPa "
            f"rn={trial['r_n_m']*1e9:.2f}nm "
            f"dV/V={trial.get('V_solid_relative_error', math.nan):+.3e} "
            f"H/H*={hazard/threshold:.5f} "
            f"dtmax={dt_ceiling:.9g}", flush=True)
        if invalid:
            save_state(
                OUT / "checkpoints" / f"avalanche{cycle}_validity_censored.npz",
                state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
                Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"],
                r_n_m=trial["r_n_m"])
            return ("censored", state, t_model + elapsed, trial, branches,
                    vp_cycle0, invalid)
        if crossing:
            save_state(
                OUT / "checkpoints" / f"avalanche{cycle}_before_root.npz",
                state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
                Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"],
                r_n_m=trial["r_n_m"])
            return ("nucleated", state, t_model + elapsed, trial, branches,
                    vp_cycle0, None)
        current = trial


def append_wait_buffer(movie, buffer, *, avalanche_id, total_completed,
                       first_reload, root_nucleation=False, flush=True):
    for index, (row, branches) in enumerate(buffer):
        if root_nucleation and index == len(buffer) - 1:
            frame_type = "root_nucleation"
        elif first_reload and index == 0:
            frame_type = "post_avalanche_reload"
        else:
            frame_type = "loading"
        append_movie(
            movie, branches, row, frame_type=frame_type,
            avalanche_id=avalanche_id if root_nucleation else 0,
            event_number=0, avalanche_active=0, sink_state=0,
            q_event_over_b=0.0, Q_avalanche_over_b=0.0,
            Q_cumulative_over_b=total_completed, S_completed=0,
            flush=flush and index == len(buffer) - 1)


def wait_for_root_or_censor(
        state, *, cycle: int, t_model: float, threshold: float,
        total_completed: int, setup, geom, evaluator, scalar, movie,
        first_reload: bool, initial_hazard: float = 0.0,
        initial_vp_cycle0: float | None = None):
    if PASSIVE_MACROSTEP_ENABLED:
        return wait_for_root_or_censor_macrostep(
            state, cycle=cycle, t_model=t_model, threshold=threshold,
            total_completed=total_completed, setup=setup, geom=geom,
            evaluator=evaluator, scalar=scalar, movie=movie,
            first_reload=first_reload, initial_hazard=initial_hazard,
            initial_vp_cycle0=initial_vp_cycle0)
    vp_cycle0 = (integral(state[1], setup) if initial_vp_cycle0 is None
                 else float(initial_vp_cycle0))
    initial, _ = measure(
        state, t_model=t_model, cycle=cycle, sink=0, q=0.0,
        qcum=total_completed, hazard=initial_hazard, threshold=threshold,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp_cycle0)
    hazard = float(initial_hazard)
    previous_gamma = initial["Gamma_per_model_time"]
    current = initial
    scratches = [PASSIVE_SCRATCH_BUILDER(*state[0].shape) for _ in range(2)]
    which = 0
    analysis_steps = int(round(ROOT_ANALYSIS_DT / setup["dt"]))
    movie_steps = max(1, int(round(ROOT_MOVIE_DT / setup["dt"])))
    elapsed = 0.0
    while True:
        state_before = tuple(field.copy() for field in state)
        elapsed_before = elapsed
        hazard_before = hazard
        previous_before = previous_gamma
        current_before = current
        block_buffer = []
        remaining = analysis_steps
        while remaining:
            step_count = min(movie_steps, remaining)
            state, which = advance_pf_steps(
                state, step_count, setup=setup, scratches=scratches, which=which)
            elapsed += step_count * setup["dt"]
            movie_row, movie_branches = measure(
                state, t_model=t_model + elapsed, cycle=cycle, sink=0, q=0.0,
                qcum=total_completed, hazard=hazard, threshold=threshold,
                setup=setup, geom=geom, evaluator=evaluator,
                vp_cycle0=vp_cycle0)
            block_buffer.append((movie_row, movie_branches))
            remaining -= step_count
        trial, branches = block_buffer[-1]
        block_dt = elapsed - elapsed_before
        hazard += 0.5 * (previous_gamma + trial["Gamma_per_model_time"]) * block_dt
        trial["H"] = hazard
        trial["H_over_threshold"] = hazard / threshold
        block_buffer[-1][0]["H"] = hazard
        block_buffer[-1][0]["H_over_threshold"] = hazard / threshold
        invalid = geometry_invalid(trial)
        crossing = hazard >= threshold

        # A stochastic crossing is always located on the same fixed PF
        # timestep grid at the finer production crossing cadence.  This is
        # hazard quadrature/refinement only; it is not a stress trigger and
        # does not change the passive integration timestep.
        refine = crossing and hazard_before < threshold

        if refine:
            state = state_before
            elapsed = elapsed_before
            hazard = hazard_before
            previous_gamma = previous_before
            scratches = [PASSIVE_SCRATCH_BUILDER(*state[0].shape) for _ in range(2)]
            which = 0
            ref_steps = max(1, int(round(COMPETING_CROSSING_DT / setup["dt"])))
            movie_stride = max(1, int(round(ROOT_MOVIE_DT / COMPETING_CROSSING_DT)))
            ref_index = 0
            ref_buffer = []
            while True:
                state, which = advance_pf_steps(
                    state, ref_steps, setup=setup, scratches=scratches, which=which)
                ref_dt = ref_steps * setup["dt"]
                elapsed += ref_dt
                refined, branches = measure(
                    state, t_model=t_model + elapsed, cycle=cycle, sink=0,
                    q=0.0, qcum=total_completed, hazard=hazard,
                    threshold=threshold, setup=setup, geom=geom,
                    evaluator=evaluator, vp_cycle0=vp_cycle0)
                hazard += 0.5 * (
                    previous_gamma + refined["Gamma_per_model_time"]) * ref_dt
                refined["H"] = hazard
                refined["H_over_threshold"] = hazard / threshold
                ref_index += 1
                if ref_index % movie_stride == 0:
                    ref_buffer.append((dict(refined), branches))
                invalid = geometry_invalid(refined)
                crossing = hazard >= threshold
                if crossing or invalid:
                    if not ref_buffer or ref_buffer[-1][0]["t_model"] < refined["t_model"]:
                        ref_buffer.append((dict(refined), branches))
                    append_wait_buffer(
                        movie, ref_buffer, avalanche_id=cycle,
                        total_completed=total_completed, first_reload=first_reload,
                        root_nucleation=crossing and not invalid)
                    scalar.add(
                        refined, branches, **output_extra(
                            avalanche_id=cycle if crossing and not invalid else 0,
                            event_number=0, avalanche_active=0, q_event=0.0,
                            Q_avalanche=0.0, Q_cumulative=total_completed,
                            S_completed=0,
                            frame_type=("root_nucleation"
                                        if crossing and not invalid else "loading")))
                    scalar.flush()
                    if crossing and not invalid:
                        save_state(
                            OUT / "checkpoints" / f"avalanche{cycle}_before_root.npz",
                            state, t_model=t_model + elapsed, H=hazard,
                            Hstar=threshold, Vp_over_Vp_cycle=refined["Vp_over_Vp_cycle"],
                            r_n_m=refined["r_n_m"])
                        return "nucleated", state, t_model + elapsed, refined, branches, vp_cycle0, None
                    save_state(
                        OUT / "checkpoints" / f"avalanche{cycle}_validity_censored.npz",
                        state, t_model=t_model + elapsed, H=hazard,
                        Hstar=threshold, Vp_over_Vp_cycle=refined["Vp_over_Vp_cycle"],
                        r_n_m=refined["r_n_m"])
                    return "censored", state, t_model + elapsed, refined, branches, vp_cycle0, invalid
                previous_gamma = refined["Gamma_per_model_time"]

        append_wait_buffer(
            movie, block_buffer, avalanche_id=cycle,
            total_completed=total_completed, first_reload=first_reload,
            root_nucleation=crossing and not invalid)
        first_reload = False
        scalar.add(
            trial, branches, **output_extra(
                avalanche_id=cycle if crossing and not invalid else 0,
                event_number=0, avalanche_active=0, q_event=0.0,
                Q_avalanche=0.0, Q_cumulative=total_completed,
                S_completed=0,
                frame_type=("root_nucleation"
                            if crossing and not invalid else "loading")))
        scalar.flush()
        save_state(
            OUT / "checkpoints" / f"avalanche{cycle}_waiting_latest.npz",
            state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
            Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"], r_n_m=trial["r_n_m"])
        print(
            f"RENEWAL WAIT a{cycle} t={elapsed:.3f} "
            f"t_s={(t_model+elapsed)*SECONDS_PER_MODEL_TIME:.6f} "
            f"source_alive=0 event_transport_active=0 "
            f"avalanche_id=0 event_number=0 h=0 facilitation_eV=0 "
            f"V={trial['Vp_over_Vp_cycle']:.5f} rn={trial['r_n_m']*1e9:.2f} "
            f"sigL={trial['sigma_local_Pa']/1e6:.3f}MPa "
            f"sigI={trial['sigma_integral_Pa']/1e6:.3f}MPa "
            f"psi={trial.get('psi_measured_deg', math.nan):.3f}deg "
            f"muBulk={trial.get('mu_bulk_Pa', math.nan)/1e6:.3f}MPa "
            f"muGB={trial.get('mu_GB_Pa', math.nan)/1e6:.3f}MPa "
            f"muTJloc={trial.get('mu_TJ_local_Pa', math.nan)/1e6:.3f}MPa "
            f"muNode={trial.get('mu_node_Pa', math.nan)/1e6:.3f}MPa "
            f"dmu={trial.get('delta_mu_GB_minus_TJ_local_Pa', math.nan)/1e6:.3f}MPa "
            f"A1={trial.get('A1_cos_m', math.nan)*1e9:.4f}nm "
            f"neck/bulge={trial.get('neck_to_bulge_ratio', math.nan):.6f} "
            f"leftW={trial.get('left_z_margin_W', math.nan):.3f} "
            f"rightW={trial.get('right_z_margin_W', math.nan):.3f} "
            f"radialW={trial.get('radial_margin_W', math.nan):.3f} "
            f"dV/V={trial.get('V_solid_relative_error', math.nan):+.3e} "
            f"H/H*={hazard/threshold:.4f}",
            flush=True)
        if invalid:
            save_state(
                OUT / "checkpoints" / f"avalanche{cycle}_validity_censored.npz",
                state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
                Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"], r_n_m=trial["r_n_m"])
            return "censored", state, t_model + elapsed, trial, branches, vp_cycle0, invalid
        if crossing:
            save_state(
                OUT / "checkpoints" / f"avalanche{cycle}_before_root.npz",
                state, t_model=t_model + elapsed, H=hazard, Hstar=threshold,
                Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"], r_n_m=trial["r_n_m"])
            return "nucleated", state, t_model + elapsed, trial, branches, vp_cycle0, None
        previous_gamma = trial["Gamma_per_model_time"]
        current = trial


def _passive_trial_step(state, dt_model: float, *, setup):
    """Advance one candidate passive interval without changing model physics."""
    scratch = PASSIVE_SCRATCH_BUILDER(*state[0].shape)
    trial = PASSIVE_STEP(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], float(dt_model), setup["M_s"],
        setup["M_eta"], setup["W"], scratch)
    if PHASE_PROJECTOR is not None:
        trial, _ = PHASE_PROJECTOR(trial, setup)
    f, particle, substrate, _ = RESERVOIR_STEP(trial, setup)
    trial = tuple(np.asarray(field).copy()
                  for field in (f, particle, substrate))
    assert_closed_solid_volume(
        trial, setup, context="facilitated-source passive trial")
    return trial


def _measure_with_fresh_tracker(
        state, *, t_model, avalanche_id, total_completed, root_hazard,
        root_threshold, vp_cycle0, setup, geom):
    trial_evaluator = EVALUATOR_BUILDER(setup, geom)
    return measure(
        state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
        qcum=total_completed, hazard=root_hazard, threshold=root_threshold,
        setup=setup, geom=geom, evaluator=trial_evaluator,
        vp_cycle0=vp_cycle0)


def wait_for_descendant_or_extinction(
        state, *, avalanche_id: int, event_number: int, t_model: float,
        total_completed: int, root_hazard: float, root_threshold: float,
        vp_cycle0: float, setup, geom, evaluator, scalar, movie, controller,
        descendant_trace: list[dict]):
    """Evolve native PF fields during one facilitated-source lifetime."""
    if not math.isfinite(controller.state.descendant_threshold):
        raise RuntimeError("descendant window was not reset after the event")
    if controller.state.window_triggered:
        raise RuntimeError("cannot wait on an already-triggered source")

    current, branches = measure(
        state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
        qcum=total_completed, hazard=root_hazard, threshold=root_threshold,
        setup=setup, geom=geom, evaluator=evaluator, vp_cycle0=vp_cycle0)
    rate_previous = controller.rate(
        current["sigma_local_Pa"], current["r_n_m"])
    next_movie_model = t_model + FACILITATED_MOVIE_DT
    sample_index = 0
    # A facilitated-source interval is passive PF evolution just like a
    # pristine-root reload.  When the qualified passive macrostep path is
    # enabled, retain its stable internal C4 ceiling but expose only the
    # hidden hazard-quadrature endpoints.  This keeps the hazard tied to the
    # actual state-dependent rate without writing/metrologizing every
    # Courant-limited internal step.
    passive_dt_ceiling = (
        float(PASSIVE_MAX_NUMERICAL_DT_MODEL)
        if PASSIVE_MACROSTEP_ENABLED else None)
    passive_dt_floor = (
        float(PASSIVE_MIN_NUMERICAL_DT_MODEL)
        if PASSIVE_MIN_NUMERICAL_DT_MODEL is not None else setup["dt"] / 16.0)

    def passive_trial(base_state, duration_model):
        nonlocal passive_dt_ceiling
        if not PASSIVE_MACROSTEP_ENABLED:
            return _passive_trial_step(
                base_state, duration_model, setup=setup)
        while True:
            scratches = [PASSIVE_SCRATCH_BUILDER(*base_state[0].shape)
                         for _ in range(2)]
            try:
                trial_state, _ = advance_pf_duration(
                    tuple(np.asarray(field).copy() for field in base_state),
                    duration_model, setup=setup,
                    dt_ceiling=passive_dt_ceiling,
                    scratches=scratches, which=0)
                return trial_state
            except (FloatingPointError, ProductionGeometryInvariantError,
                    SolidVolumeInvariantError):
                passive_dt_ceiling *= 0.5
                if passive_dt_ceiling < passive_dt_floor:
                    raise
                print(
                    f"FACILITATED MACROSTEP ROLLBACK a{avalanche_id} "
                    f"after_e{event_number} t={t_model:.6f} "
                    f"dt_ceiling={passive_dt_ceiling:.12g}",
                    flush=True)

    while True:
        time_s = t_model * SECONDS_PER_MODEL_TIME
        remaining_s = controller.state.window_deadline_s - time_s
        tolerance_s = 64.0 * math.ulp(max(abs(time_s), 1.0))
        if remaining_s <= tolerance_s:
            controller.expire_window(controller.state.window_deadline_s)
            t_model = controller.state.window_deadline_s / SECONDS_PER_MODEL_TIME
            current, branches = measure(
                state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
                qcum=total_completed, hazard=root_hazard,
                threshold=root_threshold, setup=setup, geom=geom,
                evaluator=evaluator, vp_cycle0=vp_cycle0)
            scalar.add(
                current, branches, **output_extra(
                    avalanche_id=avalanche_id, event_number=event_number,
                    avalanche_active=0, q_event=0.0,
                    Q_avalanche=controller.state.S_completed,
                    Q_cumulative=total_completed,
                    S_completed=controller.state.S_completed,
                    controller=controller, frame_type="avalanche_extinction"))
            append_movie(
                movie, branches, current, frame_type="avalanche_extinction",
                avalanche_id=avalanche_id, event_number=event_number,
                avalanche_active=0, sink_state=0, q_event_over_b=0.0,
                Q_avalanche_over_b=controller.state.S_completed,
                Q_cumulative_over_b=total_completed,
                S_completed=controller.state.S_completed,
                controller=controller, flush=True)
            scalar.flush()
            return "extinct", state, t_model, current, branches, None

        segment_start_s = time_s
        dt_model = min(
            (PASSIVE_HAZARD_QUADRATURE_DT_MODEL
             if PASSIVE_MACROSTEP_ENABLED else setup["dt"]),
            remaining_s / SECONDS_PER_MODEL_TIME)
        state_before = tuple(np.asarray(field).copy() for field in state)
        t_before_model = t_model
        provisional_state = passive_trial(state_before, dt_model)
        provisional_t_model = t_before_model + dt_model
        provisional, _ = _measure_with_fresh_tracker(
            provisional_state, t_model=provisional_t_model,
            avalanche_id=avalanche_id, total_completed=total_completed,
            root_hazard=root_hazard, root_threshold=root_threshold,
            vp_cycle0=vp_cycle0, setup=setup, geom=geom)
        provisional_rate = controller.rate(
            provisional["sigma_local_Pa"], provisional["r_n_m"])
        available = 0.5*(rate_previous+provisional_rate) * (
            dt_model*SECONDS_PER_MODEL_TIME)
        needed = (controller.state.descendant_threshold
                  - controller.state.descendant_hazard)
        crossed = available >= needed
        if crossed:
            lo, hi = 0.0, dt_model
            state_hi = provisional_state
            row_hi = provisional
            rate_hi = provisional_rate
            for _ in range(48):
                mid = 0.5*(lo+hi)
                state_mid = passive_trial(state_before, mid)
                row_mid, _ = _measure_with_fresh_tracker(
                    state_mid, t_model=t_before_model+mid,
                    avalanche_id=avalanche_id,
                    total_completed=total_completed,
                    root_hazard=root_hazard,
                    root_threshold=root_threshold,
                    vp_cycle0=vp_cycle0, setup=setup, geom=geom)
                rate_mid = controller.rate(
                    row_mid["sigma_local_Pa"], row_mid["r_n_m"])
                increment_mid = 0.5*(rate_previous+rate_mid)*(
                    mid*SECONDS_PER_MODEL_TIME)
                if increment_mid >= needed:
                    hi = mid
                    state_hi, row_hi, rate_hi = state_mid, row_mid, rate_mid
                else:
                    lo = mid
                if (hi-lo)*SECONDS_PER_MODEL_TIME <= 1.0e-12:
                    break
            state = state_hi
            t_model = t_before_model + hi
            current, branches = measure(
                state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
                qcum=total_completed, hazard=root_hazard,
                threshold=root_threshold, setup=setup, geom=geom,
                evaluator=evaluator, vp_cycle0=vp_cycle0)
            rate_trial = controller.rate(
                current["sigma_local_Pa"], current["r_n_m"])
            controller.commit_crossing(
                crossing_time_s=t_model*SECONDS_PER_MODEL_TIME)
            trial = current
        else:
            state = provisional_state
            t_model = provisional_t_model
            current, branches = measure(
                state, t_model=t_model, cycle=avalanche_id, sink=0, q=0.0,
                qcum=total_completed, hazard=root_hazard,
                threshold=root_threshold, setup=setup, geom=geom,
                evaluator=evaluator, vp_cycle0=vp_cycle0)
            rate_trial = controller.rate(
                current["sigma_local_Pa"], current["r_n_m"])
            controller.accumulate_window_segment(
                rate_start_per_s=rate_previous, rate_end_per_s=rate_trial,
                start_time_s=segment_start_s,
                end_time_s=t_model*SECONDS_PER_MODEL_TIME)
            trial = current
        trace = controller._trace_row(
            descendant_sample(trial), rate_trial, int(crossed))
        descendant_trace.append(dict(
            trace, avalanche_id=avalanche_id, event_number=event_number,
            phase="facilitated_source_wait"))
        write_csv(OUT / "descendant_hazard_trace.csv", descendant_trace)
        invalid = geometry_invalid(trial)
        frame_type = "correlation_window"
        movie_due = crossed or bool(invalid) or (
            t_model >= next_movie_model-1e-12)
        if movie_due:
            append_movie(
                movie, branches, trial, frame_type=frame_type,
                avalanche_id=avalanche_id, event_number=event_number,
                avalanche_active=1, sink_state=0, q_event_over_b=0.0,
                Q_avalanche_over_b=controller.state.S_completed,
                Q_cumulative_over_b=total_completed,
                S_completed=controller.state.S_completed,
                controller=controller, source_alive=1,
                event_transport_active=0,
                flush=crossed or bool(invalid) or sample_index % 10 == 0)
            while next_movie_model <= t_model+1e-12:
                next_movie_model += FACILITATED_MOVIE_DT
        sample_index += 1
        if crossed or invalid or (movie_due and sample_index % 10 == 0):
            scalar.add(
                trial, branches, **output_extra(
                    avalanche_id=avalanche_id,
                    event_number=event_number,
                    avalanche_active=1, q_event=0.0,
                    Q_avalanche=controller.state.S_completed,
                    Q_cumulative=total_completed,
                    S_completed=controller.state.S_completed,
                    controller=controller, frame_type=frame_type,
                    source_alive=1, event_transport_active=0))
            scalar.flush()
            print(
                f"DESCENDANT WINDOW a{avalanche_id} after_e{event_number} "
                f"t={t_model:.6f} sigL={trial['sigma_local_Pa']/1e6:.3f}MPa "
                f"sigI={trial['sigma_integral_Pa']/1e6:.3f}MPa "
                f"Hdesc/H*={controller.state.descendant_hazard/max(controller.state.descendant_threshold, 1e-300):.4f} "
                f"source_alive=1 event_transport_active=0",
                flush=True)
        save_state(
            OUT / "checkpoints"
            / f"avalanche{avalanche_id}_facilitated_wait_latest.npz",
            state, t_model=t_model,
            descendant_hazard=controller.state.descendant_hazard,
            descendant_threshold=controller.state.descendant_threshold,
            window_deadline_s=controller.state.window_deadline_s,
            Vp_over_Vp_cycle=trial["Vp_over_Vp_cycle"],
            r_n_m=trial["r_n_m"])
        if invalid:
            return "invalid", state, t_model, trial, branches, invalid
        if crossed:
            save_state(
                OUT / "checkpoints"
                / f"avalanche{avalanche_id}_event{event_number + 1}_before_descendant.npz",
                state, t_model=t_model,
                descendant_hazard=controller.state.descendant_hazard,
                descendant_threshold=controller.state.descendant_threshold)
            return "continued", state, t_model, trial, branches, None
        rate_previous = rate_trial


def descendant_sample(row: dict) -> dict:
    return dict(
        t_s=float(row["t_model"] * SECONDS_PER_MODEL_TIME),
        sigma_local_Pa=float(row["sigma_local_Pa"]),
        sigma_integral_Pa=float(row["sigma_integral_Pa"]),
        r_TJ_m=float(row["r_n_m"]))


def wait_for_nucleated_transport(
        state, *, avalanche_id: int, event_number: int, total_completed: int,
        t_model: float, current_row: dict, root_threshold: float,
        vp_cycle0: float, setup, geom, evaluator, scalar, movie, controller,
        q_event_over_b: float, force_one_block: bool = False):
    """Keep a nucleated source while closed PF evolution restores transport.

    The stochastic clock remains frozen while the already-nucleated source is
    alive and the one-b transport is active/paused at its native state.  This
    interval is not a pristine sink-OFF/root-clock state and does not change
    the transport closure.
    """
    if not PERSIST_NUCLEATED_SOURCE_UNTIL_TRANSPORT:
        return state, t_model, current_row, None
    if "transport_affinity_Pa" not in current_row:
        raise RuntimeError("transport diagnostics are required for source persistence")
    scratches = [PASSIVE_SCRATCH_BUILDER(*state[0].shape) for _ in range(2)]
    which = 0
    sample_steps = max(1, int(round(ROOT_ANALYSIS_DT / setup["dt"])))
    branches = None
    first_block = bool(force_one_block)
    while first_block or float(current_row["transport_affinity_Pa"]) <= 0.0:
        first_block = False
        state, which = advance_pf_steps(
            state, sample_steps, setup=setup, scratches=scratches, which=which)
        t_model += sample_steps * setup["dt"]
        current_row, branches = measure(
            state, t_model=t_model, cycle=avalanche_id, sink=0,
            q=q_event_over_b, qcum=total_completed + q_event_over_b,
            hazard=current_row["H"], threshold=root_threshold,
            setup=setup, geom=geom, evaluator=evaluator,
            vp_cycle0=vp_cycle0)
        scalar.add(
            current_row, branches, **output_extra(
                avalanche_id=avalanche_id, event_number=event_number,
                avalanche_active=1, q_event=q_event_over_b,
                Q_avalanche=controller.state.S_completed + q_event_over_b,
                Q_cumulative=total_completed + q_event_over_b,
                S_completed=controller.state.S_completed,
                controller=controller, frame_type="nucleated_transport_wait"))
        append_movie(
            movie, branches, current_row, frame_type="correlation_window",
            avalanche_id=avalanche_id, event_number=event_number,
            avalanche_active=1, sink_state=0,
            q_event_over_b=q_event_over_b,
            Q_avalanche_over_b=controller.state.S_completed + q_event_over_b,
            Q_cumulative_over_b=total_completed + q_event_over_b,
            S_completed=controller.state.S_completed,
            controller=controller, flush=True)
        scalar.flush()
        save_state(
            OUT / "checkpoints"
            / f"avalanche{avalanche_id}_event{event_number}_source_wait_latest.npz",
            state, t_model=t_model, q_event_over_b=q_event_over_b,
            transport_affinity_Pa=current_row["transport_affinity_Pa"],
            sigma_local_Pa=current_row["sigma_local_Pa"],
            sigma_integral_Pa=current_row["sigma_integral_Pa"])
        print(
            f"NUCLEATED SOURCE WAIT a{avalanche_id} e{event_number} "
            f"t={t_model:.3f} q={q_event_over_b:.4f} "
            f"sigL={current_row['sigma_local_Pa']/1e6:.3f}MPa "
            f"sigI={current_row['sigma_integral_Pa']/1e6:.3f}MPa "
            f"muGB={current_row['mu_GB_Pa']/1e6:.3f}MPa "
            f"muTJ={current_row['mu_TJ_Pa']/1e6:.3f}MPa "
            f"affinity={current_row['transport_affinity_Pa']/1e6:.3f}MPa",
            flush=True)
    return state, t_model, current_row, branches


def run_one_b_event(
        state, *, avalanche_id: int, event_number: int, total_completed: int,
        t_model: float, current_row: dict, root_threshold: float,
        vp_cycle0: float, setup, geom, evaluator, transport, scalar, movie,
        controller, event_restart=None):
    start_model = t_model
    restart = event_restart
    event_origin_row = dict(
        restart.get("event_origin_row", current_row)
        if restart is not None else current_row)
    if restart is not None:
        restart["event_origin_row"] = dict(event_origin_row)
    integrator_decision = event_integrator_decision(
        current_row, setup, transport, event_restart=restart)
    selected_integrator = integrator_decision["event_integrator_selected"]
    selected_event_call = (
        PACKET_EVENT_CALL if selected_integrator == "packet"
        else QUASISTATIC_EVENT_CALL)
    if selected_event_call is None:
        raise RuntimeError(
            f"{selected_integrator} event integrator was selected but its "
            "production call has not been configured")
    selected_event_integrator_name = getattr(
        selected_event_call, "event_integrator_name",
        ("adaptively subcycled explicit Euler" if selected_integrator == "packet"
         and EVENT_BRANCH_TIME_INTEGRATOR == "adaptive_explicit" else
         "linearly implicit Mullins Rosenbrock step" if selected_integrator == "packet"
         else "direct adaptive prescribed-q fast-manifold continuation"))
    print("EVENT INTEGRATOR " + json.dumps(
        integrator_decision, sort_keys=True), flush=True)
    active_time_before = (
        0.0 if restart is None else float(restart["event_time_model"]))
    initial_q = (
        0.0 if restart is None
        else float(restart["cumulative_q_m"]) / B_EVENT_M)
    callback_elapsed = 0.0
    next_movie_q = EVENT_MOVIE_DQ * (
        math.floor(initial_q / EVENT_MOVIE_DQ + 1.0e-12) + 1)
    checkpoint_samples = [descendant_sample(current_row)]
    continuation_rows = []
    final_branches = None
    event_wall_started = time.monotonic()

    def accepted(packet, accepted_state):
        nonlocal callback_elapsed, next_movie_q
        callback_elapsed += float(packet["transport_dt_model"])
        q = float(packet["q_end_over_b"])
        if q + 1e-12 < next_movie_q or q >= 1.0 - 1e-12:
            return
        row, branches = measure(
            accepted_state, t_model=start_model + callback_elapsed,
            cycle=avalanche_id, sink=1, q=q,
            qcum=total_completed + q, hazard=current_row["H"],
            threshold=root_threshold, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp_cycle0)
        row.update(integrator_decision)
        append_movie(
            movie, branches, row, frame_type="active_1b_transit",
            avalanche_id=avalanche_id, event_number=event_number,
            avalanche_active=1, sink_state=1, q_event_over_b=q,
            Q_avalanche_over_b=controller.state.S_completed + q,
            Q_cumulative_over_b=total_completed + q,
            S_completed=controller.state.S_completed, controller=controller)
        while next_movie_q <= q + 1e-12:
            next_movie_q += EVENT_MOVIE_DQ

    def progress(packet, accepted_state, progress_restart):
        """Expose each committed q state and periodically checkpoint it."""
        q = float(packet["q_end_over_b"])
        accepted_total = int(progress_restart["accepted_steps_total"])
        rejected_total = int(progress_restart.get(
            "quasistatic_trial_rejections_total", 0))
        event_time_model = float(progress_restart["event_time_model"])
        payload = dict(
            wall_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            avalanche_id=int(avalanche_id),
            event_number=int(event_number),
            source_alive=1,
            event_transport_active=1,
            q_over_b=q,
            physical_event_time_accumulated_s=(
                event_time_model * SECONDS_PER_MODEL_TIME),
            simulation_time_s=(
                start_model + event_time_model - active_time_before)
                * SECONDS_PER_MODEL_TIME,
            affinity_Pa=float(packet.get("transport_affinity_Pa",
                                         packet.get("affinity_end_Pa", math.nan))),
            sigma_local_Pa=float(packet.get("sigma_local_Pa", math.nan)),
            sigma_integral_Pa=float(packet.get("sigma_integral_Pa", math.nan)),
            neck_radius_m=float(packet.get("r_n_m", math.nan)),
            volume_relative_error=float(
                packet.get("V_solid_relative_error",
                           integral(accepted_state[0], setup)
                           / SOLID_VOLUME_INITIAL_M3 - 1.0)),
            accepted_continuation_states=accepted_total,
            rejected_trials=rejected_total,
            rejection_reason_counts=dict(packet.get(
                "rejection_reason_counts", {})),
            last_rejection_reasons=list(packet.get(
                "last_rejection_reasons", [])),
            continuation_step_over_b=float(packet.get(
                "continuation_step_over_b", math.nan)),
            accepted_change_sigma_local_MPa=float(packet.get(
                "accepted_change_sigma_local_MPa", math.nan)),
            accepted_change_sigma_integral_MPa=float(packet.get(
                "accepted_change_sigma_integral_MPa", math.nan)),
            accepted_change_kappa1_negative_per_m=float(packet.get(
                "accepted_change_kappa1_negative_per_m", math.nan)),
            accepted_change_kappa1_positive_per_m=float(packet.get(
                "accepted_change_kappa1_positive_per_m", math.nan)),
            accepted_change_reciprocal_qdot_fraction=float(packet.get(
                "accepted_change_reciprocal_qdot_fraction", math.nan)),
            slow_clock_quadrature_refinements_total=int(
                progress_restart.get(
                    "slow_clock_quadrature_refinements_total", 0)),
            active_minimum_step_over_b=float(progress_restart.get(
                "active_minimum_step_over_b", math.nan)),
            ordinary_fixed_q_calls_total=int(progress_restart.get(
                "ordinary_fixed_q_calls_total", 0)),
            restart_fast_precondition_calls_total=int(
                progress_restart.get(
                    "restart_fast_precondition_calls_total", 0)),
            fast_relax_iterations=int(packet.get("fast_relax_iterations", 0)),
            fast_relax_blocks=int(packet.get("fast_relax_blocks", 0)),
            fast_relax_calls=int(packet.get("fast_relax_calls", 0)),
            q_start_fast_relax_calls=int(packet.get(
                "q_start_fast_relax_calls", 0)),
            q_start_fast_relax_blocks=int(packet.get(
                "q_start_fast_relax_blocks", 0)),
            q_start_affinity_before_fixed_q_convergence_Pa=float(packet.get(
                "q_start_affinity_before_fixed_q_convergence_Pa", math.nan)),
            q_start_affinity_after_fixed_q_convergence_Pa=float(packet.get(
                "q_start_affinity_after_fixed_q_convergence_Pa", math.nan)),
            q_start_final_reciprocal_qdot_fraction=float(packet.get(
                "q_start_final_reciprocal_qdot_fraction", math.nan)),
            q_over_b_before_fixed_q_relaxation=float(packet.get(
                "q_over_b_before_fixed_q_relaxation", math.nan)),
            affinity_before_fixed_q_convergence_Pa=float(packet.get(
                "affinity_before_fixed_q_convergence_Pa", math.nan)),
            affinity_after_fixed_q_convergence_Pa=float(packet.get(
                "affinity_after_fixed_q_convergence_Pa", math.nan)),
            final_fast_convergence_reciprocal_qdot_fraction=float(packet.get(
                "final_fast_convergence_reciprocal_qdot_fraction", math.nan)),
            fast_convergence_reciprocal_qdot_target=float(packet.get(
                "fast_convergence_reciprocal_qdot_target", math.nan)),
            physical_event_time_increment_s=float(packet.get(
                "transport_dt_model", math.nan)) * SECONDS_PER_MODEL_TIME,
            wall_elapsed_current_event_s=time.monotonic() - event_wall_started,
            event_integrator_selected=selected_integrator)
        atomic_json(OUT / "current_event_progress.json", payload)
        if accepted_total == 1 or accepted_total % 5 == 0 or q >= 1.0-1e-12:
            checkpoint_restart = dict(progress_restart)
            checkpoint_restart["event_integrator_decision"] = dict(
                integrator_decision)
            checkpoint_restart["explicit_max_fourth_order_courant"] = math.nan
            checkpoint_restart["event_branch_time_integrator"] = (
                selected_event_integrator_name)
            path = OUT / "checkpoints" / "current_event_latest.npz"
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle, **restart_arrays(accepted_state, checkpoint_restart),
                    avalanche_id=np.asarray(avalanche_id),
                    event_number=np.asarray(event_number),
                    total_completed=np.asarray(total_completed),
                    event_start_model=np.asarray(start_model),
                    root_threshold=np.asarray(root_threshold),
                    vp_cycle0_m3=np.asarray(vp_cycle0),
                    solid_volume_initial_m3=np.asarray(
                        SOLID_VOLUME_INITIAL_M3),
                    root_row_json=np.asarray(json.dumps(current_row)),
                    event_origin_row_json=np.asarray(json.dumps(
                        event_origin_row)),
                    controller_manifest_json=np.asarray(json.dumps(
                        controller.manifest())))
            os.replace(temporary, path)

    for target in EVENT_CHECKPOINTS:
        if target <= initial_q + 1.0e-12:
            continue
        progress_kwargs = (
            {"progress_callback": progress}
            if selected_integrator == "quasistatic" else {})
        result = selected_event_call(
            state, setup, geom, evaluator, transport, target,
            event_restart=restart, state_callback=accepted,
            max_increment_fraction_b=EVENT_MAX_INCREMENT_FRACTION_B,
            explicit_max_fourth_order_courant=PRODUCTION_C4,
            **({} if EVENT_SURFACE_MOBILITY_MODEL is None else dict(
                surface_flux_mobility_m6_per_J_model_time=(
                    EVENT_SURFACE_MOBILITY_MODEL))),
            **event_time_integrator_kwargs(selected_integrator),
            **progress_kwargs)
        actual_c4 = result[4].get("explicit_max_fourth_order_courant")
        actual_integrator = result[4].get("branch_time_integrator")
        if (selected_integrator == "packet"
                and EVENT_BRANCH_TIME_INTEGRATOR == "adaptive_explicit"):
            assert actual_c4 == PRODUCTION_C4
            if actual_integrator != "adaptively subcycled explicit Euler":
                raise AssertionError("active event did not use adaptive explicit integration")
        elif selected_integrator == "packet":
            if actual_c4 is not None:
                raise AssertionError("implicit active event unexpectedly reported a C4")
            if actual_integrator != "linearly implicit Mullins Rosenbrock step":
                raise AssertionError("active event did not use implicit Mullins integration")
        else:
            if actual_c4 is not None:
                raise AssertionError("quasi-static q event unexpectedly reported a C4")
            if actual_integrator != selected_event_integrator_name:
                raise AssertionError(
                    "active event did not use configured slow-timescale "
                    f"propagator: actual={actual_integrator!r}, "
                    f"expected={selected_event_integrator_name!r}")
        for packet in result[4].get("packets", []):
            continuation_rows.append(dict(
                packet, **integrator_decision,
                avalanche_id=avalanche_id, event_number=event_number,
                event_physical_time_start_s=(
                    start_model + float(packet["accumulated_event_time_model"])
                    - float(packet["transport_dt_model"])
                    - active_time_before) * SECONDS_PER_MODEL_TIME,
                event_physical_time_end_s=(
                    start_model + float(packet["accumulated_event_time_model"])
                    - active_time_before) * SECONDS_PER_MODEL_TIME))
        continuation_path = (OUT / "event_continuations" /
                             f"avalanche{avalanche_id}_event{event_number}.csv")
        continuation_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(continuation_path, continuation_rows)
        if not result[3]:
            state = tuple(field.copy() for field in result[:3])
            restart = result[4]["event_restart"]
            restart["event_origin_row"] = dict(event_origin_row)
            restart["event_integrator_decision"] = dict(integrator_decision)
            restart["explicit_max_fourth_order_courant"] = (
                math.nan if actual_c4 is None else actual_c4)
            restart["event_branch_time_integrator"] = actual_integrator
            q_pause = float(result[4]["event_progress_over_b"])
            pause_model = start_model + (
                float(restart["event_time_model"]) - active_time_before)
            row, final_branches = measure(
                state, t_model=pause_model, cycle=avalanche_id, sink=0,
                q=q_pause, qcum=total_completed + q_pause,
                hazard=current_row["H"], threshold=root_threshold,
                setup=setup, geom=geom, evaluator=evaluator,
                vp_cycle0=vp_cycle0)
            row.update(integrator_decision)
            scalar.add(
                row, final_branches, **output_extra(
                    avalanche_id=avalanche_id, event_number=event_number,
                    avalanche_active=1, q_event=q_pause,
                    Q_avalanche=controller.state.S_completed + q_pause,
                    Q_cumulative=total_completed + q_pause,
                    S_completed=controller.state.S_completed,
                    controller=controller, frame_type="nucleated_transport_wait"))
            scalar.flush()
            save_state(
                OUT / "checkpoints"
                / f"avalanche{avalanche_id}_event{event_number}_transport_paused.npz",
                state, q_over_b=q_pause, t_model=pause_model,
                stop_reason=result[4].get("stop_reason", ""))
            if (PERSIST_NUCLEATED_SOURCE_UNTIL_TRANSPORT
                    and result[4].get("stop_reason") == "nonpositive_transport_affinity"):
                print(
                    f"EVENT TRANSPORT PAUSE a{avalanche_id} e{event_number} "
                    f"q={q_pause:.5f} sigL={row['sigma_local_Pa']/1e6:.3f}MPa "
                    f"sigI={row['sigma_integral_Pa']/1e6:.3f}MPa "
                    f"affinity={row.get('transport_affinity_Pa', np.nan)/1e6:.3f}MPa",
                    flush=True)
                return ("paused", state, pause_model, restart, row,
                        final_branches, checkpoint_samples)
            raise RuntimeError(
                f"avalanche {avalanche_id} event {event_number} failed at "
                f"q/b={q_pause}: {result[4].get('stop_reason')}; "
                f"detail={result[4].get('stop_detail')}")
        state = tuple(field.copy() for field in result[:3])
        restart = result[4]["event_restart"]
        restart["event_origin_row"] = dict(event_origin_row)
        restart["event_integrator_decision"] = dict(integrator_decision)
        restart["explicit_max_fourth_order_courant"] = (
            math.nan if actual_c4 is None else actual_c4)
        restart["event_branch_time_integrator"] = actual_integrator
        row, final_branches = measure(
            state, t_model=start_model + restart["event_time_model"]
            - active_time_before,
            cycle=avalanche_id, sink=1, q=target,
            qcum=total_completed + target, hazard=current_row["H"],
            threshold=root_threshold, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp_cycle0)
        row.update(integrator_decision)
        scalar.add(
            row, final_branches, **output_extra(
                avalanche_id=avalanche_id, event_number=event_number,
                avalanche_active=1, q_event=target,
                Q_avalanche=controller.state.S_completed + target,
                Q_cumulative=total_completed + target,
                S_completed=controller.state.S_completed,
                controller=controller, frame_type="active_1b_transit"))
        checkpoint_samples.append(descendant_sample(row))
        path = (OUT / "checkpoints"
                / f"avalanche{avalanche_id}_event{event_number}_q{target:.2f}b.npz")
        with path.open("wb") as handle:
            np.savez_compressed(handle, **restart_arrays(state, restart))
        scalar.flush()
        print(
            f"RENEWAL EVENT a{avalanche_id} e{event_number} q={target:.2f} "
            f"sigL={row['sigma_local_Pa']/1e6:.3f}MPa "
            f"sigI={row['sigma_integral_Pa']/1e6:.3f}MPa "
            f"affinity={row.get('transport_affinity_Pa', np.nan)/1e6:.3f}MPa",
            flush=True)
    return ("complete", state,
            start_model + restart["event_time_model"] - active_time_before,
            restart, row, final_branches, checkpoint_samples)


def controller_trace_rows(trace: list[dict], *, avalanche_id: int,
                          event_number: int,
                          phase: str = "active_1b_frozen") -> list[dict]:
    return [dict(item, avalanche_id=avalanche_id,
                 event_number=event_number, phase=phase)
            for item in trace]


def save_controller_checkpoint(controller, *, avalanche_id: int,
                               event_number: int) -> None:
    path = OUT / "controller_latest.json"
    path.write_text(json.dumps(dict(
        avalanche_id=avalanche_id, event_number=event_number,
        manifest=controller.manifest()), indent=2) + "\n")


def final_figure(movie_path: Path, avalanches: list[dict]) -> None:
    with h5py.File(movie_path, "r") as file:
        t_ms = np.asarray(file["time/t_s"]) * 1e3
        local = np.asarray(file["state/sigma_local_Pa"]) / 1e6
        integral_stress = np.asarray(file["state/sigma_integral_Pa"]) / 1e6
        cumulative = np.asarray(file["state/Q_cumulative_over_b"])
        sink = np.asarray(file["state/sink_state"])
        active = np.asarray(file["state/avalanche_active"])
        neck = np.asarray(file["state/r_neck_m"]) * 1e9
        tj_position = np.asarray(file["state/z_TJ_m"]) * 1e9
    fig, axes = plt.subplots(
        6, 1, figsize=(12, 14), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.7, 1.1, 1.0, .65, .65, 1.0]})
    axes[0].plot(t_ms, local, color="#174a73", label="local")
    axes[0].plot(t_ms, integral_stress, color="#9a6b28", linestyle="--",
                 label="integral")
    axes[0].set_ylabel("Stress (MPa)")
    axes[0].legend(frameon=False)
    axes[1].plot(t_ms, cumulative, color="#2f6f58")
    axes[1].set_ylabel(r"$Q_{cumulative}/b$")
    axes[2].set_ylabel("Avalanche size S")
    for avalanche in avalanches:
        start = avalanche["start_time_s"] * 1e3
        end = avalanche["end_time_s"] * 1e3
        midpoint = 0.5 * (start + end)
        size = avalanche["S"]
        axes[2].scatter([midpoint], [size], color="#67507e", zorder=3)
        axes[2].text(midpoint, size, f"  S={size}", va="center", fontsize=9)
        for ax in axes:
            ax.axvspan(start, end, color="#c65f37", alpha=.13, linewidth=0)
    axes[3].step(t_ms, sink, where="post", color="#30343b")
    axes[3].set_ylabel("Sink")
    axes[3].set_yticks([0, 1], ["OFF", "ON"])
    axes[4].step(t_ms, active, where="post", color="#8b4f43")
    axes[4].set_ylabel("Source state")
    axes[4].set_yticks([0, 1], ["pristine", "facilitated"])
    axes[5].plot(t_ms, neck, color="#4d6d72")
    axes[5].set_ylabel(r"$r_{TJ}$ (nm)")
    tj_position_axis = axes[5].twinx()
    tj_position_axis.plot(t_ms, tj_position, color="#7d4e57", alpha=.75)
    tj_position_axis.set_ylabel(r"$z_{TJ}$ (nm)", color="#7d4e57")
    axes[5].set_xlabel("Physical time (ms)")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Continuous stochastic PR avalanche renewal")
    fig.savefig(OUT / "avalanche_renewal_five_aligned.png", dpi=220)
    fig.savefig(OUT / "avalanche_renewal_five_aligned.pdf")
    plt.close(fig)


def morphology_movie(movie_path: Path) -> dict:
    with h5py.File(movie_path, "r") as file:
        frame_types = np.asarray(file["state/frame_type"])
        nframe = len(frame_types)
        active_codes = {
            FRAME_TYPES["root_nucleation"], FRAME_TYPES["active_1b_transit"],
            FRAME_TYPES["correlation_window"], FRAME_TYPES["avalanche_extinction"],
            FRAME_TYPES["child_completion"]}
        active_indices = np.asarray(
            [i for i, code in enumerate(frame_types) if code in active_codes],
            dtype=int)
        loading_indices = np.asarray(
            [i for i, code in enumerate(frame_types) if code not in active_codes],
            dtype=int)
        keep_active_stride = max(1, math.ceil(len(active_indices) / 320))
        keep_loading_stride = max(1, math.ceil(len(loading_indices) / 100))
        indices = sorted(set(
            active_indices[::keep_active_stride].tolist()
            + loading_indices[::keep_loading_stride].tolist()
            + [0, nframe - 1]))
        all_z = np.concatenate([
            np.asarray(file["geometry/z_negative_m"]),
            np.asarray(file["geometry/z_positive_m"])]) * 1e9
        all_r = np.concatenate([
            np.asarray(file["geometry/r_negative_m"]),
            np.asarray(file["geometry/r_positive_m"])]) * 1e9
        xlim = (float(np.min(all_z) - 5), float(np.max(all_z) + 5))
        rmax = float(np.max(all_r) + 5)
        target = OUT / "avalanche_renewal_morphology.gif"
        with imageio.get_writer(target, mode="I", duration=0.08, loop=0) as writer:
            for index in indices:
                fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=90)
                for side, color in (("negative", "#b45309"),
                                    ("positive", "#1d4ed8")):
                    z = np.asarray(file[f"geometry/z_{side}_m"][index]) * 1e9
                    r = np.asarray(file[f"geometry/r_{side}_m"][index]) * 1e9
                    ax.plot(z, r, color=color, linewidth=1.8)
                    ax.plot(z, -r, color=color, linewidth=1.8)
                ztj = float(file["state/z_TJ_m"][index]) * 1e9
                rtj = float(file["state/r_TJ_m"][index]) * 1e9
                ax.scatter([ztj, ztj], [rtj, -rtj], c="crimson", s=18)
                t_ms = float(file["time/t_s"][index]) * 1e3
                aid = int(file["state/avalanche_id"][index])
                event = int(file["state/event_number"][index])
                sigma = float(file["state/sigma_local_Pa"][index]) / 1e6
                sink = int(file["state/sink_state"][index])
                ax.set_xlim(*xlim)
                ax.set_ylim(-rmax, rmax)
                ax.set_aspect("equal")
                ax.set_title(
                    f"t={t_ms:.3f} ms   avalanche={aid} event={event}   "
                    f"sigma={sigma:.1f} MPa   sink={'ON' if sink else 'OFF'}")
                ax.set_xlabel("z (nm)")
                ax.set_ylabel("r (nm)")
                ax.grid(alpha=.15)
                fig.canvas.draw()
                image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(image)
                plt.close(fig)
    return dict(source_frames=nframe, rendered_frames=len(indices), path=str(target))


def main() -> None:
    global SOLID_VOLUME_INITIAL_M3, DENSIFICATION_LENGTH_INITIAL_M
    global _SPARSE_THERMO_PROBE_LAST_T, _SPARSE_THERMO_PROBE_LAST_RESULT
    _SPARSE_THERMO_PROBE_LAST_T = -math.inf
    _SPARSE_THERMO_PROBE_LAST_RESULT = None
    if SYSTEM_MASS_BOUNDARY != "closed":
        raise RuntimeError("production requires system_mass_boundary=closed")
    if RESERVOIR_STEP is not closed_surface_diffusion_only:
        raise RuntimeError(
            "production external-reservoir mass exchange is disabled")
    if PHASE_PROJECTOR is None:
        raise RuntimeError("production requires the conservative bounded projector")
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    set_num_threads(8)
    assert PRODUCTION_C4 == 0.05
    if not 0.0 < EVENT_MAX_INCREMENT_FRACTION_B <= 1.0:
        raise AssertionError("event packet fraction must lie in (0,1]")
    if not math.isfinite(DELTA_G_STEP_EV) or DELTA_G_STEP_EV <= 0.0:
        raise AssertionError(
            "production descendant lowering must be explicitly fixed, positive, "
            "and finite")
    if (not math.isfinite(ROOT_THRESHOLD_MULTIPLIER)
            or ROOT_THRESHOLD_MULTIPLIER <= 0.0):
        raise AssertionError("root threshold multiplier must be positive and finite")
    escalation_values = (
        DESCENDANT_ESCALATION_AFTER_AVALANCHES,
        DESCENDANT_ESCALATION_IF_ALL_S_BELOW,
        DESCENDANT_ESCALATION_NEW_DELTA_G_EV)
    if any(value is not None for value in escalation_values):
        if any(value is None for value in escalation_values):
            raise AssertionError("descendant escalation rule is incomplete")
        if int(DESCENDANT_ESCALATION_AFTER_AVALANCHES) < 1:
            raise AssertionError("descendant escalation gate must follow an avalanche")
        if int(DESCENDANT_ESCALATION_IF_ALL_S_BELOW) < 1:
            raise AssertionError("descendant escalation size bound must be positive")
        if (not math.isfinite(DESCENDANT_ESCALATION_NEW_DELTA_G_EV)
                or DESCENDANT_ESCALATION_NEW_DELTA_G_EV <= DELTA_G_STEP_EV):
            raise AssertionError("descendant escalation must increase delta G")
    assert TAU_CORR_S == 9.0e-3
    exported, root_slice = authoritative.configure_barrier()
    assert base.BARRIER.G0_eV == root_slice["G0_eV"]
    base.OUT = OUT
    base.MIN_SOLVABLE_VOLUME_RATIO = -math.inf

    reused_seed = REUSED_SEED is not None
    seed = (int(REUSED_SEED) if reused_seed
            else int.from_bytes(os.urandom(16), "big"))
    seed_record = dict(
        seed=seed,
        source=("reused preserved OS seed" if reused_seed
                else "128 bits from os.urandom"),
        recorded_unix_time=time.time(), thresholds_generated=False,
        root_threshold_multiplier=ROOT_THRESHOLD_MULTIPLIER,
        root_thresholds_unscaled=[], root_thresholds=[],
        stream_model="SeedSequence spawned root/descendant streams")
    (OUT / "rng_seed_record.json").write_text(json.dumps(seed_record, indent=2) + "\n")
    streams = np.random.SeedSequence(seed).spawn(2)
    root_rng = np.random.default_rng(streams[0])
    descendant_rng = np.random.default_rng(streams[1])
    preserved_root_thresholds = (
        [] if CONTINUATION is None else list(
            CONTINUATION.get("preserved_root_thresholds", [])))
    if not preserved_root_thresholds:
        preserved_root_thresholds = [
            None if CONTINUATION is None
            else float(CONTINUATION["root_threshold"])]
    reconstructed_root_thresholds_unscaled = []
    reconstructed_root_thresholds = []
    for expected in preserved_root_thresholds:
        raw = float(root_rng.exponential())
        actual = ROOT_THRESHOLD_MULTIPLIER * raw
        reconstructed_root_thresholds_unscaled.append(raw)
        reconstructed_root_thresholds.append(actual)
        if expected is not None and not math.isclose(
                actual, float(expected), rel_tol=0.0,
                abs_tol=8.0 * math.ulp(float(expected))):
            raise RuntimeError(
                "reconstructed root RNG stream does not reproduce the "
                f"preserved threshold: {actual} != {expected}")
    root_threshold = reconstructed_root_thresholds[-1]
    preserved_descendant_thresholds = (
        [] if CONTINUATION is None else list(
            CONTINUATION.get("preserved_descendant_thresholds", [])))
    reconstructed_descendant_thresholds = []
    for expected in preserved_descendant_thresholds:
        actual = float(descendant_rng.exponential())
        reconstructed_descendant_thresholds.append(actual)
        if not math.isclose(
                actual, float(expected), rel_tol=0.0,
                abs_tol=8.0 * math.ulp(float(expected))):
            raise RuntimeError(
                "reconstructed descendant RNG stream does not reproduce the "
                f"preserved threshold: {actual} != {expected}")
    if CONTINUATION is not None:
        seed_record["RNG_state_preserved"] = True
        seed_record["root_bit_generator_state_after_preserved_threshold"] = (
            root_rng.bit_generator.state)
        seed_record["descendant_bit_generator_state_after_preserved_thresholds"] = (
            descendant_rng.bit_generator.state)
        seed_record["preserved_descendant_thresholds"] = (
            reconstructed_descendant_thresholds)
    seed_record["thresholds_generated"] = True
    seed_record["root_thresholds_unscaled"].extend(
        reconstructed_root_thresholds_unscaled)
    seed_record["root_thresholds"].extend(reconstructed_root_thresholds)
    (OUT / "rng_seed_record.json").write_text(json.dumps(seed_record, indent=2) + "\n")

    geom, setup = CASE_BUILDER()
    evaluator = EVALUATOR_BUILDER(setup, geom)
    transport = TRANSPORT_BUILDER(geom)
    assert transport.b_m == B_EVENT_M
    continuation_root_already_nucleated = bool(
        CONTINUATION is not None
        and CONTINUATION.get("root_already_nucleated", False))
    continuation_active_avalanche = bool(
        CONTINUATION is not None
        and CONTINUATION.get("active_avalanche", False))
    continuation_active_event = bool(
        continuation_active_avalanche
        and CONTINUATION.get("active_event", False))
    active_event_restart = None
    start_avalanche_id = int(
        1 if CONTINUATION is None
        else CONTINUATION.get("start_avalanche_id", 1))
    completed_avalanches_before = int(
        0 if CONTINUATION is None
        else CONTINUATION.get("completed_avalanches_before", 0))
    if CONTINUATION is None:
        state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
        initial_t_model = 0.0
        initial_root_hazard = 0.0
        initial_total_completed = 0
        initial_vp_cycle0 = integral(state[1], setup)
        SOLID_VOLUME_INITIAL_M3 = integral(state[0], setup)
    else:
        checkpoint_path = Path(CONTINUATION["parent_checkpoint"])
        if continuation_active_event:
            state, active_event_restart = load_event_save(checkpoint_path)
            if "active_event_pre_row" in CONTINUATION:
                active_event_restart["event_origin_row"] = dict(
                    CONTINUATION["active_event_pre_row"])
        else:
            with np.load(checkpoint_path) as saved:
                state = tuple(np.asarray(saved[key]).copy()
                              for key in ("f", "particle", "substrate"))
        continuation_projection = None
        if CONTINUATION.get("repair_partition_on_load", False):
            f_before = state[0].copy()
            state, continuation_projection = PHASE_PROJECTOR(state, setup)
            if not np.array_equal(state[0], f_before):
                raise RuntimeError(
                    "continuation partition repair changed the conserved f field")
        initial_t_model = float(CONTINUATION["t_model"])
        initial_root_hazard = float(CONTINUATION["root_hazard"])
        initial_total_completed = int(CONTINUATION.get("total_completed", 0))
        initial_vp_cycle0 = float(CONTINUATION.get(
            "vp_cycle0_m3", integral(state[1], setup)))
        SOLID_VOLUME_INITIAL_M3 = float(CONTINUATION["solid_volume_initial_m3"])
        if (initial_root_hazard >= root_threshold
                and not continuation_root_already_nucleated):
            raise RuntimeError(
                "continuation root hazard is already at or above threshold")
        assert_closed_solid_volume(
            state, setup, context="continuation checkpoint load")
        preserved_probe = CONTINUATION.get("sparse_thermo_probe_result")
        if preserved_probe is not None:
            _SPARSE_THERMO_PROBE_LAST_RESULT = dict(preserved_probe)
            _SPARSE_THERMO_PROBE_LAST_T = initial_t_model
    radial_weights = np.asarray(setup["r_c"])[None, :]
    z_coordinates = np.asarray(setup["z"])[:, None]
    initial_centroids = []
    for ownership in state[1:]:
        weights = np.asarray(ownership) * radial_weights
        initial_centroids.append(float(
            np.sum(weights * z_coordinates) / np.sum(weights)))
    DENSIFICATION_LENGTH_INITIAL_M = abs(
        initial_centroids[0] - initial_centroids[1])
    if not math.isfinite(DENSIFICATION_LENGTH_INITIAL_M) or (
            DENSIFICATION_LENGTH_INITIAL_M <= 0.0):
        raise RuntimeError("initial densification reference length is invalid")
    root_params = base.BARRIER
    controller = AvalancheController(
        barrier=DescendantBarrier(root_params, DELTA_G_STEP_EV),
        temperature_K=TEMPERATURE_K,
        attempt_frequency_per_s=float(exported["constants"]["nu0_sinv"]),
        b_m=B_EVENT_M,
        correlation_time_s=TAU_CORR_S,
        rng=descendant_rng,
        facilitation_decay_alpha=FACILITATION_DECAY_ALPHA)
    if continuation_active_avalanche:
        saved_controller = dict(CONTINUATION["controller_manifest"])
        saved_state = dict(saved_controller["state"])
        saved_state["delta_G_step_eV"] = DELTA_G_STEP_EV
        saved_state["facilitation_decay_alpha"] = FACILITATION_DECAY_ALPHA
        saved_state["source_amplitude"] = float(
            CONTINUATION.get("source_amplitude", saved_state["source_amplitude"]))
        controller.state = type(controller.state)(**saved_state)
        controller.crossings = list(saved_controller.get("crossings", []))
        if not controller.state.avalanche_active:
            raise RuntimeError("active-avalanche continuation is not active")
        if controller.state.window_triggered and not continuation_active_event:
            raise RuntimeError(
                "active-avalanche continuation must precede descendant crossing")
        if continuation_active_event:
            if not controller.state.window_triggered:
                raise RuntimeError(
                    "active-event continuation requires a committed child crossing")
            if active_event_restart is None:
                raise RuntimeError("active-event restart state was not loaded")
        if not math.isclose(
                controller.state.descendant_threshold,
                reconstructed_descendant_thresholds[-1], rel_tol=0.0,
                abs_tol=8.0 * math.ulp(controller.state.descendant_threshold)):
            raise RuntimeError(
                "restored controller threshold does not match reconstructed RNG")
    facilitation_schedule = [dict(
        effective_from_avalanche_id=start_avalanche_id,
        descendant_delta_G0_eV=controller.barrier.delta_G_step_eV,
        reason="initial production configuration")]
    (OUT / "facilitation_schedule.json").write_text(
        json.dumps(facilitation_schedule, indent=2) + "\n")
    history_sources = (
        [] if CONTINUATION is None else list(
            CONTINUATION.get("history_sources", [])))
    history_prefix_rows = load_continuation_history_prefix(history_sources)
    manifest = dict(
        accepted_architecture="D2 serialized finite-1b avalanche renewal",
        production_parent_commit="d2aafbf7df99728acb67af941f2c078126455e1f",
        implementation_commit=os.popen("git rev-parse HEAD").read().strip(),
        temperature_K=TEMPERATURE_K,
        descendant_delta_G_step_eV=DELTA_G_STEP_EV,
        descendant_delta_G_step_initial_eV=DELTA_G_STEP_EV,
        descendant_barrier_definition=(
            "max(G_floor, G_fit_star(sigma_local)-delta_G_step)"),
        tau_corr_s=controller.correlation_time_s,
        tau_source_s=controller.correlation_time_s,
        tau_source_definition=(
            "facilitated-source lifetime/correlation interval after each "
            "completed 1b; not the 1b sweep time"),
        facilitation_decay_alpha=controller.facilitation_decay_alpha,
        facilitation_state_definition=(
            "Gdesc=max(Gfloor,Gfit-h*deltaG0); h1=1; "
            "h<-alpha*h after each completed descendant"),
        root_barrier_export=str(authoritative.EXPORT),
        root_barrier_export_sha256=hashlib.sha256(
            authoritative.EXPORT.read_bytes()).hexdigest(),
        root_barrier_slice=root_slice, seconds_per_model_time=SECONDS_PER_MODEL_TIME,
        clock_scale=CLOCK_SCALE, b_event_m=B_EVENT_M, C4=PRODUCTION_C4,
        C4_applies_to_active_event=(
            "only when event_integrator_selected=packet and "
            "event_branch_time_integrator=adaptive_explicit"),
        event_integrator_global_default=DEFAULT_PRODUCTION_EVENT_INTEGRATOR,
        event_integrator_requested=EVENT_INTEGRATOR_REQUESTED,
        event_integrator_auto_selection=dict(
            timescale_ratio_definition="tau_GB/tau_surface",
            tau_GB_definition="b/qdot(A_event_start)",
            tau_surface_definition="L_surface^4/B_surface",
            surface_active_length_m=float(
                EVENT_SURFACE_ACTIVE_LENGTH_M
                if EVENT_SURFACE_ACTIVE_LENGTH_M is not None
                else setup.get("local_metrology_window_m", 3.0*setup["W"])),
            surface_active_length_source=EVENT_SURFACE_ACTIVE_LENGTH_SOURCE,
            selection_threshold_low=EVENT_INTEGRATOR_SELECTION_THRESHOLD_LOW,
            selection_threshold_high=EVENT_INTEGRATOR_SELECTION_THRESHOLD_HIGH,
            intermediate_fallback=EVENT_INTEGRATOR_INTERMEDIATE_FALLBACK,
            selection_frozen_for_complete_one_b=True,
            reevaluated_before_each_event=True),
        event_branch_time_integrator=EVENT_BRANCH_TIME_INTEGRATOR,
        event_max_increment_fraction_b=EVENT_MAX_INCREMENT_FRACTION_B,
        event_minimum_nominal_packets_per_one_b=math.ceil(
            1.0 / EVENT_MAX_INCREMENT_FRACTION_B),
        event_surface_B_m4_per_model_time=EVENT_B_PF_MODEL,
        D_GB_m2_per_model_time=transport.D_gb_m2_per_model_time,
        V_cutoff=V_CUTOFF, r_neck_cutoff_m=RN_CUTOFF_M,
        enforce_volume_cutoff=ENFORCE_VOLUME_CUTOFF,
        enforce_r_neck_cutoff=ENFORCE_RN_CUTOFF,
        enforce_first_root_stress_check=ENFORCE_FIRST_ROOT_STRESS_CHECK,
        persist_nucleated_source_until_transport=(
            PERSIST_NUCLEATED_SOURCE_UNTIL_TRANSPORT),
        require_unique_field_TJ=REQUIRE_UNIQUE_FIELD_TJ,
        minimum_boundary_clearance_W=MINIMUM_BOUNDARY_CLEARANCE_W,
        root_formation_penalty_eV=ROOT_BARRIER_PENALTY_EV,
        root_barrier_definition=(
            "G_fit_star(sigma_local)+root_formation_penalty_eV"),
        descendant_excludes_root_formation_penalty=True,
        root_threshold_multiplier=ROOT_THRESHOLD_MULTIPLIER,
        first_root_threshold_unscaled=(
            reconstructed_root_thresholds_unscaled[-1]),
        first_root_threshold_effective=root_threshold,
        root_threshold_definition=(
            "root_threshold = root_threshold_multiplier * Exp(1) draw"),
        descendant_escalation_rule=dict(
            evaluate_after_avalanches=DESCENDANT_ESCALATION_AFTER_AVALANCHES,
            escalate_if_all_sizes_below=(
                DESCENDANT_ESCALATION_IF_ALL_S_BELOW),
            new_delta_G_desc_0_eV=(
                DESCENDANT_ESCALATION_NEW_DELTA_G_EV),
            applies_only_to_subsequent_avalanches=True),
        continuation_active_avalanche=continuation_active_avalanche,
        continuation_start_avalanche_id=start_avalanche_id,
        completed_avalanches_before=completed_avalanches_before,
        continuation_history_sources=[str(Path(path).resolve())
                                      for path in history_sources],
        continuation_history_prefix_rows=len(history_prefix_rows),
        continuation_contour_sources=(
            [] if CONTINUATION is None else list(
                CONTINUATION.get("contour_sources", []))),
        continuation_movie_sources=(
            [] if CONTINUATION is None else list(
                CONTINUATION.get("movie_sources", []))),
        continuation_partition_projection=(
            None if CONTINUATION is None else continuation_projection),
        rng_seed_reused=reused_seed,
        root_analysis_dt_model=ROOT_ANALYSIS_DT,
        loading_movie_dt_model=ROOT_MOVIE_DT,
        passive_macrostep_enabled=PASSIVE_MACROSTEP_ENABLED,
        passive_max_numerical_dt_model=PASSIVE_MAX_NUMERICAL_DT_MODEL,
        passive_hazard_quadrature_dt_model=(
            PASSIVE_HAZARD_QUADRATURE_DT_MODEL),
        passive_crossing_tolerance_model=(
            PASSIVE_CROSSING_TOLERANCE_MODEL),
        passive_macrostep_hazard_semantics=(
            "state-dependent endpoint rates with trapezoidal subinterval "
            "quadrature; rollback and bisection on H=Hstar"),
        event_movie_dq_over_b=EVENT_MOVIE_DQ, N_branch=N_BRANCH,
        facilitated_movie_dt_model=FACILITATED_MOVIE_DT,
        descendant_clock_frozen_during_active_1b=True,
        source_alive_during_active_1b=True,
        event_transport_active_during_1b=True,
        source_alive_during_facilitated_window=True,
        event_transport_active_during_facilitated_window=False,
        child_launch_from_exact_native_crossing_state=True,
        pristine_root_clock_disabled_while_source_alive=True,
        descendant_pending_queue_enabled=False,
        facilitated_source_window_evolves_closed_passive_PF=True,
        facilitated_source_window_is_pristine_sink_off=False,
        bounded_phase_projector=(
            None if PHASE_PROJECTOR is None else PHASE_PROJECTOR.manifest()),
        system_mass_boundary=SYSTEM_MASS_BOUNDARY,
        solid_volume_initial_m3=SOLID_VOLUME_INITIAL_M3,
        densification_strain_reference_length_m=(
            DENSIFICATION_LENGTH_INITIAL_M),
        solid_volume_relative_tolerance=SOLID_VOLUME_RELATIVE_TOLERANCE,
        external_reservoir_enabled=(
            RESERVOIR_STEP is not closed_surface_diffusion_only),
        external_reservoir_mass_exchange="disabled",
        total_solid_volume_conserved=True,
        equilibrium_cap_clearance_requirement_W=(
            MIN_EQUILIBRIUM_CAP_CLEARANCE_W),
        radial_vacuum_padding=RADIAL_PADDING_DIAGNOSTICS,
        domain_future_clearance_definition=(
            "conserved-volume isolated-sphere, equilibrium-cap, and "
            "all-mass-sphere envelopes; diagnostic only"),
        no_descendant_memory_between_avalanches=True,
        max_descendants_per_avalanche=MAX_DESCENDANTS_PER_AVALANCHE,
        stop_if_first_avalanche_has_no_descendants=(
            STOP_IF_FIRST_AVALANCHE_HAS_NO_DESCENDANTS),
        no_elastic_facilitation=True, no_parameter_retuning=True,
        grid_shape=list(state[0].shape), dr_m=setup["dr"], dz_m=setup["dz"],
        W_m=setup["W"], R_cyl_m=geom["R_cyl"], lambda_m=geom["lam"],
        **RUN_METADATA)
    (OUT / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    scalar = ScalarOutput(
        OUT, history_prefix_rows=history_prefix_rows)
    movie_path = OUT / "avalanche_renewal_movie_geometry.h5"
    t_model = initial_t_model
    total_completed = initial_total_completed
    # Carry immutable completed-avalanche records across an atomic restart.
    # This is required both for a continuous aggregate table and for any
    # predeclared campaign gate whose decision depends on earlier avalanche
    # sizes.  These rows are bookkeeping only; no saved physical state is
    # reconstructed from them.
    prior_avalanches = (
        [] if CONTINUATION is None else [
            dict(row) for row in CONTINUATION.get("prior_avalanches", [])])
    if any(int(row["avalanche_id"]) >= start_avalanche_id
           for row in prior_avalanches):
        raise RuntimeError(
            "prior avalanche records must precede continuation start id")
    prior_avalanche_count = len(prior_avalanches)
    avalanches = prior_avalanches
    # A restart may occur after one or more committed 1b events in the active
    # avalanche.  Carry those immutable records forward so the completed
    # avalanche retains its exact event count, propagation time, and strain.
    subevents = (
        [] if CONTINUATION is None else [
            dict(row) for row in CONTINUATION.get("prior_subevents", [])])
    descendant_trace = []
    censor = None
    blocker = None
    outcome = "RUNNING"
    started = time.monotonic()

    with MovieGeometryArchive(
            movie_path, nbranch=N_BRANCH,
            seconds_per_model_time=SECONDS_PER_MODEL_TIME,
            metadata=manifest) as movie:
        vp0 = initial_vp_cycle0
        initial, branches = measure(
            state, t_model=t_model, cycle=start_avalanche_id, sink=0, q=0.0,
            qcum=total_completed, hazard=initial_root_hazard,
            threshold=root_threshold, setup=setup, geom=geom,
            evaluator=evaluator, vp_cycle0=vp0)
        if (MIN_EQUILIBRIUM_CAP_CLEARANCE_W is not None
                and initial["equilibrium_cap_radial_clearance_W"]
                < float(MIN_EQUILIBRIUM_CAP_CLEARANCE_W)):
            raise RuntimeError(
                "padded radial domain fails equilibrium-cap clearance gate: "
                f"{initial['equilibrium_cap_radial_clearance_W']:.6f}W < "
                f"{MIN_EQUILIBRIUM_CAP_CLEARANCE_W:.6f}W")
        print(
            "INITIAL METROLOGY "
            f"sigma_local={initial['sigma_local_Pa']/1e6:.9f} MPa "
            f"sigma_integral={initial['sigma_integral_Pa']/1e6:.9f} MPa "
            f"zTJ={initial['z_TJ_m']*1e9:.9f} nm "
            f"rTJ={initial['r_n_m']*1e9:.9f} nm",
            flush=True)
        print("DOMAIN ENVELOPE " + json.dumps(dict(
            psi_equilibrium_deg=initial["domain_envelope_psi_equilibrium_deg"],
            R_sph_1_nm=initial["R_sph_1_m"]*1e9,
            R_sph_2_nm=initial["R_sph_2_m"]*1e9,
            R_cap_1_nm=initial["R_cap_1_m"]*1e9,
            R_cap_2_nm=initial["R_cap_2_m"]*1e9,
            r_contact_1_nm=initial["r_contact_1_m"]*1e9,
            r_contact_2_nm=initial["r_contact_2_m"]*1e9,
            h_cap_1_nm=initial["h_cap_1_m"]*1e9,
            h_cap_2_nm=initial["h_cap_2_m"]*1e9,
            pair_equilibrium_axial_span_nm=(
                initial["pair_equilibrium_axial_span_m"]*1e9),
            R_all_nm=initial["R_all_m"]*1e9,
            radial_box_nm=initial["radial_box_m"]*1e9,
            equilibrium_cap_radial_clearance_W=(
                initial["equilibrium_cap_radial_clearance_W"]),
            all_mass_sphere_radial_clearance_W=(
                initial["all_mass_sphere_radial_clearance_W"]),
            equilibrium_geometry_imposed=False,
        ), sort_keys=True), flush=True)
        initial_controller = controller if continuation_active_avalanche else None
        initial_event_number = (
            int(CONTINUATION.get("active_event_number"))
            if continuation_active_event
            else (controller.state.S_completed
                  if continuation_active_avalanche else 0))
        initial_S = (
            controller.state.S_completed if continuation_active_avalanche else 0)
        initial_frame_type = (
            "active_1b_transit" if continuation_active_event
            else ("correlation_window"
                  if continuation_active_avalanche else "loading"))
        initial_q_event = (
            float(active_event_restart["cumulative_q_m"]) / B_EVENT_M
            if continuation_active_event else 0.0)
        scalar.add(
            initial, branches, **output_extra(
                avalanche_id=(start_avalanche_id
                              if continuation_active_avalanche else 0),
                event_number=initial_event_number,
                avalanche_active=int(continuation_active_avalanche),
                q_event=initial_q_event,
                Q_avalanche=initial_S + initial_q_event,
                Q_cumulative=total_completed + initial_q_event,
                S_completed=initial_S, frame_type=initial_frame_type,
                controller=initial_controller,
                source_alive=int(continuation_active_avalanche),
                event_transport_active=int(continuation_active_event)))
        append_movie(
            movie, branches, initial, frame_type=initial_frame_type,
            avalanche_id=(start_avalanche_id
                          if continuation_active_avalanche else 0),
            event_number=initial_event_number,
            avalanche_active=int(continuation_active_avalanche), sink_state=0,
            q_event_over_b=initial_q_event,
            Q_avalanche_over_b=initial_S + initial_q_event,
            Q_cumulative_over_b=total_completed + initial_q_event,
            S_completed=initial_S,
            controller=initial_controller,
            source_alive=int(continuation_active_avalanche),
            event_transport_active=int(continuation_active_event), flush=True)
        scalar.flush()
        save_state(
            OUT / "checkpoints" / "initial_state.npz", state,
            t_model=t_model, H=initial_root_hazard, Hstar=root_threshold)

        try:
            for avalanche_id in range(
                    start_avalanche_id, AVALANCHES_REQUESTED + 1):
                active_resume_this_avalanche = (
                    continuation_active_avalanche
                    and avalanche_id == start_avalanche_id)
                resume_this_avalanche = (
                    CONTINUATION is not None
                    and avalanche_id == start_avalanche_id)
                wait_start_model = float(
                    CONTINUATION.get("wait_start_model", t_model)
                    if resume_this_avalanche else t_model)
                if active_resume_this_avalanche:
                    status = "nucleated"
                    root_row = dict(CONTINUATION["root_row"])
                    vp_cycle0 = initial_vp_cycle0
                    reasons = None
                    print(
                        "RESUMING ACTIVE AVALANCHE "
                        f"a={avalanche_id} S={controller.state.S_completed} "
                        f"t_model={t_model:.12f} "
                        f"Hdesc/H*={controller.state.descendant_hazard/controller.state.descendant_threshold:.12f} "
                        f"deltaG0={controller.barrier.delta_G_step_eV:.6f}eV",
                        flush=True)
                elif avalanche_id == 1 and continuation_root_already_nucleated:
                    status = "nucleated"
                    root_row = initial
                    vp_cycle0 = initial_vp_cycle0
                    reasons = None
                    print(
                        "RESUMING REALIZED ROOT "
                        f"t_model={t_model:.12f} "
                        f"H/H*={initial_root_hazard/root_threshold:.12f} "
                        f"sigma_local={root_row['sigma_local_Pa']/1e6:.9f}MPa",
                        flush=True)
                else:
                    (status, state, t_model, root_row, branches,
                     vp_cycle0, reasons) = wait_for_root_or_censor(
                        state, cycle=avalanche_id, t_model=t_model,
                        threshold=root_threshold, total_completed=total_completed,
                        setup=setup, geom=geom, evaluator=evaluator,
                        scalar=scalar, movie=movie,
                        first_reload=avalanche_id > 1,
                        initial_hazard=(
                            initial_root_hazard
                            if avalanche_id == start_avalanche_id else 0.0),
                        initial_vp_cycle0=(
                            initial_vp_cycle0
                            if avalanche_id == start_avalanche_id else None))
                if status == "censored":
                    censor = dict(
                        avalanche_id=avalanche_id, reasons=reasons,
                        t_model=t_model, t_s=t_model * SECONDS_PER_MODEL_TIME,
                        wait_time_model=t_model - wait_start_model,
                        H=root_row["H"], Hstar=root_threshold,
                        H_over_Hstar=root_row["H_over_threshold"],
                        sigma_local_Pa=root_row["sigma_local_Pa"],
                        Vp_over_Vp_cycle=root_row["Vp_over_Vp_cycle"],
                        r_n_m=root_row["r_n_m"])
                    outcome = "STOPPED_AT_EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                    break
                if (ENFORCE_FIRST_ROOT_STRESS_CHECK and avalanche_id == 1
                        and abs(root_row["sigma_local_Pa"] - 65.0e6) > 1.0e6):
                    blocker = dict(
                        reason="first_root_crossing_outside_fixed_65MPa_check",
                        expected_stress_Pa=65.0e6,
                        tolerance_Pa=1.0e6,
                        actual_stress_Pa=root_row["sigma_local_Pa"],
                        H=root_row["H"], Hstar=root_threshold)
                    outcome = "STOPPED_FOR_ROOT_HAZARD_BOOKKEEPING_AUDIT"
                    break

                print("ROOT NUCLEATION " + json.dumps(dict(
                    t_s=t_model*SECONDS_PER_MODEL_TIME,
                    cycle=avalanche_id,
                    source_alive=1,
                    event_transport_active=0,
                    avalanche_id=avalanche_id,
                    event_number=1,
                    h=1.0,
                    facilitation_eV=controller.barrier.delta_G_step_eV,
                    sigma_local_MPa=root_row["sigma_local_Pa"]*1e-6,
                    sigma_integral_MPa=root_row["sigma_integral_Pa"]*1e-6,
                    mu_GB_minus_mu_TJ_local_MPa=root_row.get(
                        "delta_mu_GB_minus_TJ_local_Pa", math.nan)*1e-6,
                    r_neck_nm=root_row["r_n_m"]*1e9,
                    A1_nm=root_row.get("A1_cos_m", math.nan)*1e9,
                    neck_to_bulge=root_row.get(
                        "neck_to_bulge_ratio", math.nan),
                    psi_measured_deg=root_row.get(
                        "psi_measured_deg", math.nan),
                    H_root_over_Hstar=root_row["H"] / root_threshold,
                    H_desc_over_Hstar=0.0,
                    left_z_margin_W=root_row.get(
                        "left_z_margin_W", math.nan),
                    right_z_margin_W=root_row.get(
                        "right_z_margin_W", math.nan),
                    radial_margin_W=root_row.get(
                        "radial_margin_W", math.nan),
                    equilibrium_cap_radial_clearance_W=root_row.get(
                        "equilibrium_cap_radial_clearance_W", math.nan),
                    all_mass_sphere_radial_clearance_W=root_row.get(
                        "all_mass_sphere_radial_clearance_W", math.nan),
                ), sort_keys=True), flush=True)

                if not active_resume_this_avalanche:
                    controller.start(
                        avalanche_id=avalanche_id, root_cycle=avalanche_id,
                        start_time_s=t_model * SECONDS_PER_MODEL_TIME)
                avalanche_start_model = float(
                    CONTINUATION.get("avalanche_start_model", t_model)
                    if active_resume_this_avalanche else t_model)
                root_local = root_row["sigma_local_Pa"]
                root_integral = root_row["sigma_integral_Pa"]
                event_number = (
                    controller.state.S_completed
                    if active_resume_this_avalanche else 0)
                current_row = initial if active_resume_this_avalanche else root_row
                current_branches = branches
                avalanche_blocker = None

                if active_resume_this_avalanche and not continuation_active_event:
                    (window_status, state, t_model, current_row,
                     current_branches, window_reasons) = (
                        wait_for_descendant_or_extinction(
                            state, avalanche_id=avalanche_id,
                            event_number=event_number, t_model=t_model,
                            total_completed=total_completed,
                            root_hazard=root_row["H"],
                            root_threshold=root_threshold,
                            vp_cycle0=vp_cycle0, setup=setup, geom=geom,
                            evaluator=evaluator, scalar=scalar, movie=movie,
                            controller=controller,
                            descendant_trace=descendant_trace))
                    if window_status == "invalid":
                        avalanche_blocker = dict(
                            reason="geometry_validity_boundary_during_avalanche",
                            avalanche_id=avalanche_id,
                            event_number=event_number,
                            reasons=window_reasons,
                            Vp_over_Vp_cycle=current_row["Vp_over_Vp_cycle"],
                            r_n_m=current_row["r_n_m"])
                    elif window_status == "extinct":
                        avalanches.append(dict(
                            avalanche_id=avalanche_id,
                            root_threshold=root_threshold,
                            start_time_model=avalanche_start_model,
                            end_time_model=t_model,
                            start_time_s=avalanche_start_model
                            * SECONDS_PER_MODEL_TIME,
                            end_time_s=t_model * SECONDS_PER_MODEL_TIME,
                            duration_s=(t_model-avalanche_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            wait_time_model=avalanche_start_model-wait_start_model,
                            wait_time_s=(avalanche_start_model-wait_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            t_reload_s=(avalanche_start_model-wait_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            S=controller.state.S_completed,
                            Q_avalanche_nm=controller.state.S_completed
                            * B_EVENT_M * 1e9,
                            sigma_root_local_Pa=root_local,
                            sigma_min_local_Pa=min(
                                row["sigma_local_Pa"] for row in scalar.rows),
                            sigma_min_integral_Pa=min(
                                row["sigma_integral_Pa"] for row in scalar.rows),
                            sigma_final_local_Pa=current_row["sigma_local_Pa"],
                            sigma_final_integral_Pa=current_row[
                                "sigma_integral_Pa"],
                            delta_sigma_local_Pa=(
                                root_local-current_row["sigma_local_Pa"]),
                            delta_sigma_integral_Pa=(
                                root_integral-current_row["sigma_integral_Pa"]),
                            r_n_root_m=root_row["r_n_m"],
                            r_n_final_m=current_row["r_n_m"],
                            source_alive=0, event_transport_active=0,
                            left_z_margin_W=current_row.get(
                                "left_z_margin_W", math.nan),
                            right_z_margin_W=current_row.get(
                                "right_z_margin_W", math.nan),
                            radial_margin_W=current_row.get(
                                "radial_margin_W", math.nan),
                            equilibrium_cap_radial_clearance_W=current_row.get(
                                "equilibrium_cap_radial_clearance_W", math.nan),
                            all_mass_sphere_radial_clearance_W=current_row.get(
                                "all_mass_sphere_radial_clearance_W", math.nan),
                            descendant_thresholds_drawn=len(
                                controller.state.thresholds_drawn)))
                        enrich_avalanche_record(
                            avalanches[-1], root_row=root_row,
                            final_row=current_row, scalar_rows=scalar.rows,
                            subevents=subevents,
                            delta_G0_eV=controller.barrier.delta_G_step_eV)
                        write_csv(OUT / "avalanche_summary.csv", avalanches)
                    # If the threshold crossed, the existing child-event loop
                    # below starts immediately from this exact native state.

                while (controller.state.avalanche_active
                       and avalanche_blocker is None):
                    event_number += 1
                    resume_active_event_now = bool(
                        continuation_active_event
                        and active_resume_this_avalanche
                        and event_number == int(CONTINUATION[
                            "active_event_number"]))
                    event_start_model = (
                        float(CONTINUATION["active_event_start_model"])
                        if resume_active_event_now else t_model)
                    pre_event_row = dict(
                        CONTINUATION["active_event_pre_row"]
                        if resume_active_event_now else current_row)
                    pre_local = pre_event_row["sigma_local_Pa"]
                    pre_integral = pre_event_row["sigma_integral_Pa"]
                    pre_energy = pre_event_row.get(
                        "G_phasefield_J", math.nan)
                    pre_gamma_energy = pre_event_row.get(
                        "G_gamma_J", math.nan)
                    pre_z_tj = pre_event_row["z_TJ_m"]
                    pre_r_tj = pre_event_row["r_n_m"]
                    pre_volume = pre_event_row.get("V_solid_m3", math.nan)
                    pre_delta_mu = pre_event_row.get(
                        "delta_mu_GB_minus_TJ_local_Pa", math.nan)
                    native_pre_event_state = tuple(
                        np.asarray(field).copy() for field in (
                            active_event_restart["base_fields"]
                            if resume_active_event_now else state))
                    try:
                        if event_number == 1:
                            (state, t_model, current_row, waited_branches) = (
                                wait_for_nucleated_transport(
                                    state, avalanche_id=avalanche_id,
                                    event_number=event_number,
                                    total_completed=total_completed,
                                    t_model=t_model, current_row=current_row,
                                    root_threshold=root_threshold,
                                    vp_cycle0=vp_cycle0, setup=setup, geom=geom,
                                    evaluator=evaluator, scalar=scalar,
                                    movie=movie, controller=controller,
                                    q_event_over_b=0.0))
                            if waited_branches is not None:
                                current_branches = waited_branches
                        elif not resume_active_event_now:
                            # A child starts from the exact native threshold-
                            # crossing state. There is no pristine interval,
                            # new root threshold, or extra passive evolution.
                            print("CHILD CROSSING " + json.dumps(dict(
                                t_s=t_model*SECONDS_PER_MODEL_TIME,
                                cycle=avalanche_id,
                                source_alive=1,
                                event_transport_active=1,
                                avalanche_id=avalanche_id,
                                event_number=event_number,
                                h=controller.state.source_amplitude,
                                facilitation_eV=(
                                    controller.state.source_amplitude
                                    * controller.barrier.delta_G_step_eV),
                                sigma_local_MPa=(
                                    current_row["sigma_local_Pa"]*1e-6),
                                sigma_integral_MPa=(
                                    current_row["sigma_integral_Pa"]*1e-6),
                                mu_GB_minus_mu_TJ_local_MPa=current_row.get(
                                    "delta_mu_GB_minus_TJ_local_Pa",
                                    math.nan)*1e-6,
                                r_neck_nm=current_row["r_n_m"]*1e9,
                                A1_nm=current_row.get(
                                    "A1_cos_m", math.nan)*1e9,
                                neck_to_bulge=current_row.get(
                                    "neck_to_bulge_ratio", math.nan),
                                psi_measured_deg=current_row.get(
                                    "psi_measured_deg", math.nan),
                                H_root_over_Hstar=(
                                    current_row["H"] / root_threshold),
                                H_desc_over_Hstar=(
                                    controller.state.descendant_hazard
                                    / controller.state.descendant_threshold),
                                left_z_margin_W=current_row.get(
                                    "left_z_margin_W", math.nan),
                                right_z_margin_W=current_row.get(
                                    "right_z_margin_W", math.nan),
                                radial_margin_W=current_row.get(
                                    "radial_margin_W", math.nan),
                            ), sort_keys=True), flush=True)
                        if not resume_active_event_now:
                            controller.begin_transit()
                        restart = (
                            active_event_restart
                            if resume_active_event_now else None)
                        samples = []
                        while True:
                            (event_status, state, t_model, restart,
                             current_row, current_branches,
                             attempt_samples) = run_one_b_event(
                                state, avalanche_id=avalanche_id,
                                event_number=event_number,
                                total_completed=total_completed,
                                t_model=t_model, current_row=current_row,
                                root_threshold=root_threshold,
                                vp_cycle0=vp_cycle0, setup=setup, geom=geom,
                                evaluator=evaluator, transport=transport,
                                scalar=scalar, movie=movie,
                                controller=controller,
                                event_restart=restart)
                            samples.extend(attempt_samples)
                            if event_status == "complete":
                                break
                            q_pause = float(restart["cumulative_q_m"]) / B_EVENT_M
                            (state, t_model, current_row,
                             waited_branches) = wait_for_nucleated_transport(
                                state, avalanche_id=avalanche_id,
                                event_number=event_number,
                                total_completed=total_completed,
                                t_model=t_model, current_row=current_row,
                                root_threshold=root_threshold,
                                vp_cycle0=vp_cycle0, setup=setup, geom=geom,
                                evaluator=evaluator, scalar=scalar, movie=movie,
                                controller=controller,
                                q_event_over_b=q_pause,
                                force_one_block=True)
                            if waited_branches is not None:
                                current_branches = waited_branches
                        if resume_active_event_now:
                            active_event_restart = None
                    except Exception as error:
                        avalanche_blocker = dict(
                            reason="qualified_1b_event_failed",
                            avalanche_id=avalanche_id,
                            event_number=event_number,
                            message=f"{type(error).__name__}: {error}")
                        break
                    # ``run_one_b_event`` freezes this once at event start and
                    # persists it in the returned restart record.  Recover it
                    # here for the event report instead of referring to the
                    # helper's local variable (which caused the post-event
                    # NameError in the first large-avalanche continuation).
                    integrator_decision = dict(
                        restart["event_integrator_decision"])
                    trace = controller.integrate_transit_samples(samples)
                    descendant_trace.extend(controller_trace_rows(
                        trace, avalanche_id=avalanche_id,
                        event_number=event_number, phase="active_1b_frozen"))
                    controller.complete_transit(t_model * SECONDS_PER_MODEL_TIME)
                    total_completed += 1
                    # The production validity ratio is referenced to the
                    # start of each new post-event waiting interval.  Reset
                    # only that diagnostic reference at exactly q=b; the PF
                    # state and every event/clock parameter are unchanged.
                    vp_cycle0 = integral(state[1], setup)
                    current_row, current_branches = measure(
                        state, t_model=t_model, cycle=avalanche_id, sink=1,
                        q=1.0, qcum=total_completed, hazard=root_row["H"],
                        threshold=root_threshold, setup=setup, geom=geom,
                        evaluator=evaluator, vp_cycle0=vp_cycle0)
                    completion_local = current_row["sigma_local_Pa"]
                    completion_integral = current_row["sigma_integral_Pa"]
                    delta_energy = current_row.get(
                        "G_phasefield_J", math.nan) - pre_energy
                    delta_volume_relative = (
                        current_row.get("V_solid_m3", math.nan) / pre_volume - 1.0
                        if math.isfinite(pre_volume) and pre_volume != 0.0
                        else math.nan)
                    copy_energy_ledger = {}
                    if EVENT_COPY_ENERGY_DIAGNOSTIC is not None:
                        copy_energy_ledger = dict(
                            EVENT_COPY_ENERGY_DIAGNOSTIC(
                                tuple(field.copy()
                                      for field in native_pre_event_state),
                                tuple(np.asarray(field).copy()
                                      for field in state),
                                setup=setup, geom=geom,
                                pre_row=pre_event_row,
                                post_row=dict(current_row)))
                        for key, value in copy_energy_ledger.items():
                            if not np.isscalar(value):
                                raise TypeError(
                                    "EVENT_COPY_ENERGY_DIAGNOSTIC may return "
                                    f"only scalar values; {key!r} is not scalar")
                    transition = dict(
                        delta_G_J=delta_energy,
                        delta_sigma_local_Pa=completion_local - pre_local,
                        delta_sigma_integral_Pa=(
                            completion_integral - pre_integral),
                        delta_z_TJ_m=current_row["z_TJ_m"] - pre_z_tj,
                        delta_r_TJ_m=current_row["r_n_m"] - pre_r_tj,
                        delta_volume_relative=delta_volume_relative,
                        delta_mu_GB_minus_TJ_local_Pa=(
                            current_row.get(
                                "delta_mu_GB_minus_TJ_local_Pa", math.nan)
                            - pre_delta_mu))
                    descendant_Gstar_eV = controller.barrier_eV(
                        completion_local)
                    descendant_rate_per_s = controller.rate(
                        completion_local, current_row["r_n_m"])
                    barrier_feedback = controller.stress_drop_feedback(
                        completion_local, current_row["r_n_m"])
                    remaining_corr_window_ms = max(
                        controller.state.window_deadline_s
                        - t_model * SECONDS_PER_MODEL_TIME, 0.0) * 1.0e3
                    print(
                        f"EVENT TRANSITION a{avalanche_id} e{event_number} "
                        f"dG={delta_energy:+.9e}J "
                        f"dsigL={transition['delta_sigma_local_Pa']/1e6:+.6f}MPa "
                        f"dsigI={transition['delta_sigma_integral_Pa']/1e6:+.6f}MPa "
                        f"dzTJ={transition['delta_z_TJ_m']*1e9:+.6f}nm "
                        f"drTJ={transition['delta_r_TJ_m']*1e9:+.6f}nm "
                        f"dV/V={delta_volume_relative:+.3e} "
                        f"dDeltaMu={transition['delta_mu_GB_minus_TJ_local_Pa']/1e6:+.6f}MPa",
                        flush=True)
                    print("ONE_B REPORT " + json.dumps(dict(
                        **integrator_decision,
                        t_s=t_model*SECONDS_PER_MODEL_TIME,
                        cycle=avalanche_id,
                        source_alive=1,
                        event_transport_active=0,
                        avalanche_id=avalanche_id,
                        event_number=event_number,
                        root_or_descendant=(
                            "root" if event_number == 1 else "descendant"),
                        q_total_over_b=controller.state.S_completed,
                        sigma_local_before_MPa=pre_local / 1.0e6,
                        sigma_local_after_MPa=completion_local / 1.0e6,
                        sigma_integral_before_MPa=pre_integral / 1.0e6,
                        sigma_integral_after_MPa=completion_integral / 1.0e6,
                        mu_GB_minus_mu_TJ_before_MPa=pre_delta_mu / 1.0e6,
                        mu_GB_minus_mu_TJ_after_MPa=current_row.get(
                            "delta_mu_GB_minus_TJ_local_Pa", math.nan) / 1.0e6,
                        G_PF_native_before_J=pre_energy,
                        G_PF_native_after_J=current_row.get(
                            "G_phasefield_J", math.nan),
                        G_gamma_before_J=copy_energy_ledger.get(
                            "G_gamma_geometry_before_J", pre_gamma_energy),
                        G_gamma_after_J=copy_energy_ledger.get(
                            "G_gamma_geometry_after_J",
                            current_row.get("G_gamma_J", math.nan)),
                        r_neck_nm=current_row["r_n_m"] * 1.0e9,
                        A1_nm=current_row.get("A1_cos_m", math.nan) * 1.0e9,
                        neck_to_bulge=current_row.get(
                            "neck_to_bulge_ratio", math.nan),
                        psi_measured_deg=current_row.get(
                            "psi_measured_deg", math.nan),
                        H_root_over_Hstar=current_row["H"] / root_threshold,
                        H_desc_over_Hstar=(
                            controller.state.descendant_hazard
                            / controller.state.descendant_threshold),
                        left_z_margin_W=current_row.get(
                            "left_z_margin_W", math.nan),
                        right_z_margin_W=current_row.get(
                            "right_z_margin_W", math.nan),
                        radial_margin_W=current_row.get(
                            "radial_margin_W", math.nan),
                        equilibrium_cap_radial_clearance_W=current_row.get(
                            "equilibrium_cap_radial_clearance_W", math.nan),
                        all_mass_sphere_radial_clearance_W=current_row.get(
                            "all_mass_sphere_radial_clearance_W", math.nan),
                        descendant_Gstar_eV=descendant_Gstar_eV,
                        h=controller.state.source_amplitude,
                        deltaG_facilitation_eV=(
                            controller.state.source_amplitude
                            * controller.barrier.delta_G_step_eV),
                        descendant_rate_per_s=descendant_rate_per_s,
                        descendant_H=controller.state.descendant_hazard,
                        descendant_Hstar=controller.state.descendant_threshold,
                        remaining_corr_window_ms=remaining_corr_window_ms,
                        minus_dGdesc_dsigma_eV_per_Pa=barrier_feedback[
                            "minus_dGdesc_dsigma_eV_per_Pa"],
                        stress_drop_feedback=barrier_feedback["drops"],
                        **copy_energy_ledger,
                    ), sort_keys=True), flush=True)
                    subevents.append(dict(
                        **integrator_decision,
                        avalanche_id=avalanche_id,
                        event_number=event_number,
                        event_type="root" if event_number == 1 else "descendant",
                        start_time_model=event_start_model, end_time_model=t_model,
                        start_time_s=event_start_model * SECONDS_PER_MODEL_TIME,
                        end_time_s=t_model * SECONDS_PER_MODEL_TIME,
                        duration_s=(t_model - event_start_model)
                        * SECONDS_PER_MODEL_TIME,
                        sigma_start_local_Pa=pre_local,
                        sigma_end_local_Pa=completion_local,
                        delta_sigma_local_Pa=pre_local - completion_local,
                        delta_sigma_integral_Pa=pre_integral - completion_integral,
                        delta_G_J=delta_energy,
                        delta_z_TJ_m=transition["delta_z_TJ_m"],
                        delta_r_TJ_m=transition["delta_r_TJ_m"],
                        delta_volume_relative=delta_volume_relative,
                        delta_mu_GB_minus_TJ_local_Pa=(
                            transition["delta_mu_GB_minus_TJ_local_Pa"]),
                        mu_GB_minus_mu_TJ_before_Pa=pre_delta_mu,
                        mu_GB_minus_mu_TJ_after_Pa=current_row.get(
                            "delta_mu_GB_minus_TJ_local_Pa", math.nan),
                        G_PF_native_before_J=pre_energy,
                        G_PF_native_after_J=current_row.get(
                            "G_phasefield_J", math.nan),
                        G_gamma_before_J=copy_energy_ledger.get(
                            "G_gamma_geometry_before_J", pre_gamma_energy),
                        G_gamma_after_J=copy_energy_ledger.get(
                            "G_gamma_geometry_after_J",
                            current_row.get("G_gamma_J", math.nan)),
                        r_neck_after_m=current_row["r_n_m"],
                        A1_after_m=current_row.get("A1_cos_m", math.nan),
                        descendant_Gstar_eV=descendant_Gstar_eV,
                        h=controller.state.source_amplitude,
                        facilitation_eV=(
                            controller.state.source_amplitude
                            * controller.barrier.delta_G_step_eV),
                        descendant_rate_per_s=descendant_rate_per_s,
                        descendant_H=controller.state.descendant_hazard,
                        descendant_Hstar=controller.state.descendant_threshold,
                        remaining_corr_window_ms=remaining_corr_window_ms,
                        descendants_committed_during_transit=0,
                        pending_children_after=controller.state.pending_children,
                        C4=restart["explicit_max_fourth_order_courant"],
                        event_branch_time_integrator=restart[
                            "event_branch_time_integrator"],
                        event_packets_total=restart["accepted_steps_total"],
                        event_max_increment_fraction_b=(
                            EVENT_MAX_INCREMENT_FRACTION_B),
                        D_GB_m2_per_model_time=transport.D_gb_m2_per_model_time,
                        **copy_energy_ledger))
                    write_csv(OUT / "one_b_subevents.csv", subevents)
                    write_csv(OUT / "descendant_hazard_trace.csv", descendant_trace)
                    append_movie(
                        movie, current_branches, current_row,
                        frame_type="child_completion",
                        avalanche_id=avalanche_id,
                        event_number=event_number, avalanche_active=1,
                        sink_state=1, q_event_over_b=1.0,
                        Q_avalanche_over_b=controller.state.S_completed,
                        Q_cumulative_over_b=total_completed,
                        S_completed=controller.state.S_completed,
                        controller=controller, flush=True)
                    save_state(
                        OUT / "checkpoints"
                        / f"avalanche{avalanche_id}_event{event_number}_complete.npz",
                        state, t_model=t_model,
                        total_completed=total_completed,
                        controller_manifest_json=json.dumps(controller.manifest()))
                    save_controller_checkpoint(
                        controller, avalanche_id=avalanche_id,
                        event_number=event_number)
                    descendants_completed = max(
                        controller.state.S_completed - 1, 0)
                    if (MAX_DESCENDANTS_PER_AVALANCHE is not None
                            and descendants_completed
                            >= int(MAX_DESCENDANTS_PER_AVALANCHE)):
                        avalanche_blocker = dict(
                            reason="descendant_safety_cap_reached",
                            avalanche_id=avalanche_id,
                            event_number=event_number,
                            S_completed=controller.state.S_completed,
                            descendants_completed=descendants_completed,
                            max_descendants_per_avalanche=int(
                                MAX_DESCENDANTS_PER_AVALANCHE),
                            checkpoint=str(
                                OUT / "checkpoints"
                                / f"avalanche{avalanche_id}_event{event_number}_complete.npz"))
                        break
                    if (ENFORCE_EVENT_ENERGY_DESCENT and event_number == 1
                            and (not math.isfinite(delta_energy)
                                 or delta_energy >= 0.0)):
                        avalanche_blocker = dict(
                            reason="first_corrected_event_failed_energy_descent",
                            avalanche_id=avalanche_id,
                            event_number=event_number,
                            delta_G_J=delta_energy,
                            delta_sigma_local_Pa=(
                                completion_local - pre_local),
                            delta_sigma_integral_Pa=(
                                completion_integral - pre_integral),
                            checkpoint=str(
                                OUT / "checkpoints"
                                / f"avalanche{avalanche_id}_event{event_number}_complete.npz"))
                        break
                    invalid = geometry_invalid(current_row)
                    if invalid:
                        avalanche_blocker = dict(
                            reason="geometry_validity_boundary_during_avalanche",
                            avalanche_id=avalanche_id,
                            event_number=event_number, reasons=invalid,
                            Vp_over_Vp_cycle=current_row["Vp_over_Vp_cycle"],
                            r_n_m=current_row["r_n_m"])
                        break
                    (window_status, state, t_model, current_row,
                     current_branches, window_reasons) = (
                        wait_for_descendant_or_extinction(
                            state, avalanche_id=avalanche_id,
                            event_number=event_number, t_model=t_model,
                            total_completed=total_completed,
                            root_hazard=root_row["H"],
                            root_threshold=root_threshold,
                            vp_cycle0=vp_cycle0, setup=setup, geom=geom,
                            evaluator=evaluator, scalar=scalar, movie=movie,
                            controller=controller,
                            descendant_trace=descendant_trace))
                    if window_status == "invalid":
                        avalanche_blocker = dict(
                            reason="geometry_validity_boundary_during_avalanche",
                            avalanche_id=avalanche_id,
                            event_number=event_number,
                            reasons=window_reasons,
                            Vp_over_Vp_cycle=current_row["Vp_over_Vp_cycle"],
                            r_n_m=current_row["r_n_m"])
                        break
                    continued = window_status == "continued"
                    subevents[-1].update(
                        sigma_local_before_next_descendant_or_extinction_Pa=(
                            current_row["sigma_local_Pa"]),
                        sigma_integral_before_next_descendant_or_extinction_Pa=(
                            current_row["sigma_integral_Pa"]),
                        mu_GB_minus_mu_TJ_before_next_descendant_or_extinction_Pa=(
                            current_row.get(
                                "delta_mu_GB_minus_TJ_local_Pa", math.nan)),
                        descendant_H_at_next_descendant_or_extinction=(
                            controller.state.descendant_hazard),
                        descendant_Hstar_at_next_descendant_or_extinction=(
                            controller.state.descendant_threshold),
                        Hdesc_over_Hstar_at_next_descendant_or_extinction=(
                            controller.state.descendant_hazard
                            / controller.state.descendant_threshold),
                        window_outcome=("next_descendant" if continued
                                        else "natural_extinction"))
                    write_csv(OUT / "one_b_subevents.csv", subevents)
                    print("ONE_B WINDOW REPORT " + json.dumps(dict(
                        t_s=t_model*SECONDS_PER_MODEL_TIME,
                        cycle=avalanche_id,
                        source_alive=int(continued),
                        event_transport_active=int(continued),
                        avalanche_id=avalanche_id,
                        event_number=event_number,
                        h=controller.state.source_amplitude,
                        facilitation_eV=(
                            controller.state.source_amplitude
                            * controller.barrier.delta_G_step_eV),
                        sigma_local_before_next_descendant_or_extinction_MPa=(
                            current_row["sigma_local_Pa"] / 1.0e6),
                        sigma_integral_before_next_descendant_or_extinction_MPa=(
                            current_row["sigma_integral_Pa"] / 1.0e6),
                        mu_GB_minus_mu_TJ_MPa=current_row.get(
                            "delta_mu_GB_minus_TJ_local_Pa", math.nan) / 1.0e6,
                        Hdesc_over_Hstar=(
                            controller.state.descendant_hazard
                            / controller.state.descendant_threshold),
                        H_root_over_Hstar=current_row["H"] / root_threshold,
                        r_neck_nm=current_row["r_n_m"]*1e9,
                        A1_nm=current_row.get("A1_cos_m", math.nan)*1e9,
                        neck_to_bulge=current_row.get(
                            "neck_to_bulge_ratio", math.nan),
                        psi_measured_deg=current_row.get(
                            "psi_measured_deg", math.nan),
                        left_z_margin_W=current_row.get(
                            "left_z_margin_W", math.nan),
                        right_z_margin_W=current_row.get(
                            "right_z_margin_W", math.nan),
                        radial_margin_W=current_row.get(
                            "radial_margin_W", math.nan),
                        outcome=("next_descendant" if continued
                                 else "natural_extinction"),
                    ), sort_keys=True), flush=True)
                    if not continued:
                        avalanches.append(dict(
                            avalanche_id=avalanche_id,
                            root_threshold=root_threshold,
                            start_time_model=avalanche_start_model,
                            end_time_model=t_model,
                            start_time_s=avalanche_start_model
                            * SECONDS_PER_MODEL_TIME,
                            end_time_s=t_model * SECONDS_PER_MODEL_TIME,
                            duration_s=(t_model - avalanche_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            wait_time_model=avalanche_start_model
                            - wait_start_model,
                            wait_time_s=(avalanche_start_model - wait_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            t_reload_s=(avalanche_start_model - wait_start_model)
                            * SECONDS_PER_MODEL_TIME,
                            S=controller.state.S_completed,
                            Q_avalanche_nm=controller.state.S_completed
                            * B_EVENT_M * 1e9,
                            sigma_root_local_Pa=root_local,
                            sigma_min_local_Pa=min(
                                row["sigma_local_Pa"] for row in scalar.rows
                                if (row["t_model"] >= avalanche_start_model
                                    and row["t_model"] <= t_model)),
                            sigma_min_integral_Pa=min(
                                row["sigma_integral_Pa"] for row in scalar.rows
                                if (row["t_model"] >= avalanche_start_model
                                    and row["t_model"] <= t_model)),
                            sigma_final_local_Pa=current_row["sigma_local_Pa"],
                            sigma_final_integral_Pa=(
                                current_row["sigma_integral_Pa"]),
                            delta_sigma_local_Pa=root_local
                            - current_row["sigma_local_Pa"],
                            delta_sigma_integral_Pa=root_integral
                            - current_row["sigma_integral_Pa"],
                            r_n_root_m=root_row["r_n_m"],
                            r_n_final_m=current_row["r_n_m"],
                            source_alive=0,
                            event_transport_active=0,
                            left_z_margin_W=current_row.get(
                                "left_z_margin_W", math.nan),
                            right_z_margin_W=current_row.get(
                                "right_z_margin_W", math.nan),
                            radial_margin_W=current_row.get(
                                "radial_margin_W", math.nan),
                            equilibrium_cap_radial_clearance_W=current_row.get(
                                "equilibrium_cap_radial_clearance_W", math.nan),
                            all_mass_sphere_radial_clearance_W=current_row.get(
                                "all_mass_sphere_radial_clearance_W", math.nan),
                            descendant_thresholds_drawn=len(
                                controller.state.thresholds_drawn)))
                        enrich_avalanche_record(
                            avalanches[-1], root_row=root_row,
                            final_row=current_row, scalar_rows=scalar.rows,
                            subevents=subevents,
                            delta_G0_eV=controller.barrier.delta_G_step_eV)
                        write_csv(OUT / "avalanche_summary.csv", avalanches)
                        save_state(
                            OUT / "checkpoints"
                            / f"avalanche{avalanche_id}_extinct.npz",
                            state, t_model=t_model,
                            total_completed=total_completed,
                            controller_manifest_json=json.dumps(
                                controller.manifest()))
                        print("AVALANCHE COMPLETE", json.dumps(avalanches[-1]),
                              flush=True)
                        break

                if avalanche_blocker is not None:
                    blocker = avalanche_blocker
                    outcome = ("STOPPED_AT_EXISTING_GEOMETRY_VALIDITY_BOUNDARY"
                               if blocker["reason"].startswith("geometry_validity")
                               else "STOPPED_ON_QUALIFIED_EVENT_BLOCKER")
                    break
                if (not avalanches
                        or int(avalanches[-1]["avalanche_id"]) != avalanche_id):
                    blocker = dict(
                        reason="avalanche_did_not_reach_extinction",
                        avalanche_id=avalanche_id)
                    outcome = "STOPPED_ON_AVALANCHE_CONTROLLER_BLOCKER"
                    break
                if (STOP_IF_FIRST_AVALANCHE_HAS_NO_DESCENDANTS
                        and avalanche_id == 1
                        and int(avalanches[-1]["S"]) == 1):
                    blocker = dict(
                        reason="first_avalanche_has_no_descendants",
                        avalanche_id=1,
                        S=1,
                        next_action="replay_same_realized_root_with_0.350_eV")
                    outcome = "STOPPED_FOR_FIXED_FACILITATION_ESCALATION"
                    break
                if (DESCENDANT_ESCALATION_AFTER_AVALANCHES is not None
                        and avalanche_id == int(
                            DESCENDANT_ESCALATION_AFTER_AVALANCHES)):
                    gate_count = int(DESCENDANT_ESCALATION_AFTER_AVALANCHES)
                    gate_rows = avalanches[:gate_count]
                    gate_sizes = [int(row["S"]) for row in gate_rows]
                    should_escalate = (
                        len(gate_sizes) == gate_count
                        and all(size < int(
                            DESCENDANT_ESCALATION_IF_ALL_S_BELOW)
                                for size in gate_sizes))
                    old_delta = controller.barrier.delta_G_step_eV
                    if should_escalate:
                        new_delta = float(
                            DESCENDANT_ESCALATION_NEW_DELTA_G_EV)
                        controller.barrier = DescendantBarrier(
                            root_params, new_delta)
                        controller.state.delta_G_step_eV = new_delta
                        action = "escalated_for_subsequent_avalanches"
                    else:
                        new_delta = old_delta
                        action = "retained_initial_facilitation"
                    gate_record = dict(
                        evaluated_after_avalanche_id=avalanche_id,
                        observed_sizes=gate_sizes,
                        criterion_all_sizes_below=int(
                            DESCENDANT_ESCALATION_IF_ALL_S_BELOW),
                        old_delta_G0_eV=old_delta,
                        new_delta_G0_eV=new_delta,
                        action=action,
                        effective_from_avalanche_id=avalanche_id + 1)
                    facilitation_schedule.append(gate_record)
                    (OUT / "facilitation_schedule.json").write_text(
                        json.dumps(facilitation_schedule, indent=2) + "\n")
                    print("FACILITATION GATE " + json.dumps(
                        gate_record, sort_keys=True), flush=True)
                if avalanche_id < AVALANCHES_REQUESTED:
                    root_threshold_unscaled = float(root_rng.exponential())
                    root_threshold = (
                        ROOT_THRESHOLD_MULTIPLIER * root_threshold_unscaled)
                    seed_record["root_thresholds_unscaled"].append(
                        root_threshold_unscaled)
                    seed_record["root_thresholds"].append(root_threshold)
                    (OUT / "rng_seed_record.json").write_text(
                        json.dumps(seed_record, indent=2) + "\n")
            else:
                outcome = f"{AVALANCHES_REQUESTED}_COMPLETE_AVALANCHES"
        except Exception as error:
            if isinstance(error, SolidVolumeInvariantError):
                save_state(
                    OUT / "checkpoints" / "solver_error_total_volume.npz",
                    error.state, t_model=t_model,
                    V_solid_m3=error.volume_m3,
                    V_solid_relative_error=error.relative_error,
                    invariant_context=error.context)
                blocker = dict(
                    reason="closed_total_solid_volume_solver_error",
                    message=str(error), context=error.context,
                    V_solid_m3=error.volume_m3,
                    V_solid_relative_error=error.relative_error,
                    checkpoint=str(
                        OUT / "checkpoints" / "solver_error_total_volume.npz"))
                outcome = "STOPPED_ON_TOTAL_SOLID_VOLUME_SOLVER_ERROR"
            else:
                blocker = dict(reason="unexpected_exception",
                               message=f"{type(error).__name__}: {error}")
                outcome = "STOPPED_ON_UNEXPECTED_BLOCKER"
        finally:
            scalar.flush()
            movie.flush()

    final_figure(movie_path, avalanches)
    movie_result = (
        morphology_movie(movie_path) if FINAL_MOVIE_RENDERER is None
        else FINAL_MOVIE_RENDERER(movie_path))
    result = dict(
        outcome=outcome,
        avalanches_completed=(completed_avalanches_before
                              + len(avalanches)-prior_avalanche_count),
        avalanches_completed_in_this_directory=(
            len(avalanches)-prior_avalanche_count),
        completed_avalanches_before=completed_avalanches_before,
        prior_avalanche_records_carried=prior_avalanche_count,
        avalanches_requested=AVALANCHES_REQUESTED,
        total_one_b_events=total_completed, censor=censor, blocker=blocker,
        seed=seed, root_thresholds=seed_record["root_thresholds"],
        descendant_delta_G_step_initial_eV=DELTA_G_STEP_EV,
        descendant_delta_G_step_final_eV=(
            controller.barrier.delta_G_step_eV),
        facilitation_schedule=facilitation_schedule,
        root_threshold_multiplier=ROOT_THRESHOLD_MULTIPLIER,
        tau_corr_s=controller.correlation_time_s,
        packet_C4_configured=PRODUCTION_C4,
        packet_C4_asserted_per_selected_event=(
            EVENT_BRANCH_TIME_INTEGRATOR == "adaptive_explicit"),
        event_branch_time_integrator=EVENT_BRANCH_TIME_INTEGRATOR,
        D_GB_m2_per_model_time=transport.D_gb_m2_per_model_time,
        bounded_phase_projector=(
            None if PHASE_PROJECTOR is None else PHASE_PROJECTOR.manifest()),
        movie=movie_result, wall_seconds=time.monotonic() - started,
        avalanches=avalanches)
    (OUT / "avalanche_renewal_result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
