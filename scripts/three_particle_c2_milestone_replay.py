"""Reconstruct durable C2 event quarter checkpoints from saved event bases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parent)]

import numpy as np

from three_particle_buffered_event_probe import LargerDtBufferedContactEvent
from three_particle_forced_event import MANIFEST
from three_particle_source_window import advance_source_window
from pf_sintering.three_particle_event import save_event_checkpoint
from pf_sintering.three_particle_sharp_initial import load_mapped_sharp_state


TARGETS = (0.25, 0.50, 0.75)


def atomic_json(path, payload):
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_row(history, event_number, q):
    candidates = [row for row in history
                  if int(row.get("event_number", 0)) == event_number and
                  row["phase"] == "ACTIVE_ONE_B" and
                  abs(float(row["q_over_b"]) - q) < 1e-10]
    if len(candidates) != 1:
        raise RuntimeError(
            f"event {event_number} q={q}: expected one history row, got {len(candidates)}")
    return candidates[0]


def overlap(event, fields, history_row, nominal_q, actual_q):
    metrics = event.metrics(tuple(fields), actual_q)
    return {
        "nominal_q_over_b": nominal_q,
        "actual_accepted_q_over_b": actual_q,
        "nominal_offset_over_b": actual_q - nominal_q,
        "selected_stress_difference_Pa": (
            metrics["sigma_local_Pa"] -
            history_row["contacts"][event.name]["sigma_local_Pa"]),
        "left_stress_difference_Pa": (
            metrics["LEFT_sigma_local_Pa"] -
            history_row["contacts"]["LEFT"]["sigma_local_Pa"]),
        "right_stress_difference_Pa": (
            metrics["RIGHT_sigma_local_Pa"] -
            history_row["contacts"]["RIGHT"]["sigma_local_Pa"]),
        "center_volume_difference_m3": (
            metrics["V_center_m3"] - history_row["center_volume_m3"]),
        "total_volume_difference_m3": (
            metrics["total_volume_m3"] -
            history_row["diagnostics"]["total_volume_m3"]),
        "topology_stop": bool(metrics["topology_stop"]),
    }


def overlap_pass(row):
    return bool(
        abs(row["selected_stress_difference_Pa"]) < 1e-3 and
        abs(row["left_stress_difference_Pa"]) < 1e-3 and
        abs(row["right_stress_difference_Pa"]) < 1e-3 and
        abs(row["center_volume_difference_m3"]) < 1e-32 and
        abs(row["total_volume_difference_m3"]) < 1e-32 and
        not row["topology_stop"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--event", type=int, action="append")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    history = json.loads((args.run / "history.json").read_text())
    launch = json.loads((args.run / "launch.json").read_text())
    rule = json.loads(Path(
        "docs/three_particle/cmc/angle_calibration.json").read_text())["rule"]
    events = args.event or list(range(1, 13))
    audit_path = args.out / "milestone_replay_audit.json"
    audit = ({"label": "DETERMINISTIC_EVENT_MILESTONE_RECONSTRUCTION",
              "run": str(args.run), "source": str(args.source), "events": []}
             if not audit_path.exists() else json.loads(audit_path.read_text()))
    for event_row in audit["events"]:
        event_row.setdefault("transport_recoveries_before_q0p75", 0)
        for milestone in event_row["milestones"]:
            if "q_over_b" in milestone:
                q = milestone.pop("q_over_b")
                milestone["nominal_q_over_b"] = q
                milestone["actual_accepted_q_over_b"] = q
                milestone["nominal_offset_over_b"] = 0.0
    completed = {int(row["event_number"]) for row in audit["events"]
                 if row.get("passed")}
    for event_number in events:
        if event_number in completed:
            print("MILESTONE_ALREADY_VERIFIED", event_number, flush=True)
            continue
        final_path = args.run / f"event_{event_number}_final.npz"
        with np.load(final_path) as data:
            base = tuple(value.copy() for value in data["base_fields"])
            contact = str(data["contact"])
        g, _, _ = load_mapped_sharp_state(args.source)
        pair = (0, 1) if contact == "LEFT" else (1, 2)
        event_dir = args.out / f"event_{event_number:02d}"
        event_dir.mkdir(exist_ok=True)
        saved = {}
        started = time.perf_counter()
        active_rows = [row for row in history
                       if int(row.get("event_number", 0)) == event_number and
                       row["phase"] == "ACTIVE_ONE_B"]
        actual_targets = {
            nominal: min((float(row["q_over_b"]) for row in active_rows
                          if float(row["q_over_b"]) >= nominal - 1e-12),
                         key=lambda q: q - nominal)
            for nominal in TARGETS}

        def callback(packet, fields, restart):
            q = float(packet["q_end_over_b"])
            for nominal, actual in actual_targets.items():
                if nominal in saved or abs(q - actual) > 1e-10:
                    continue
                path = event_dir / f"q_{nominal:.2f}b.npz"
                save_event_checkpoint(path, fields, restart, contact=contact,
                                      label="RECONSTRUCTED_EXACT_ACCEPTED_MILESTONE")
                comparison = overlap(
                    event, fields, recorded_row(history, event_number, actual),
                    nominal, actual)
                comparison["checkpoint"] = str(path)
                comparison["passed"] = overlap_pass(comparison)
                saved[nominal] = comparison
                print("MILESTONE", event_number, nominal, "actual", actual,
                      comparison["passed"], flush=True)

        state = base
        restart = None
        pause_count = 0
        while True:
            event = LargerDtBufferedContactEvent(g, pair, max_fast_blocks=512)
            result = event.run(
                state, target=max(actual_targets.values()), restart=restart,
                callback=callback,
                maximum_step_over_b=launch["accepted_event_increment_over_b"],
                initial_step_over_b=launch["accepted_event_increment_over_b"],
                minimum_step_over_b=launch["event_minimum_increment_over_b"],
                maximum_accepted_states=launch["event_accepted_state_cap"])
            state = result[:4]
            if result[4]:
                break
            if result[5].get("stop_reason") != "nonpositive_transport_affinity":
                raise RuntimeError(
                    f"event {event_number}: unexpected replay stop: {result[5].get('stop_reason')}")
            restart = result[5]["event_restart"]
            recovery_seconds = (MANIFEST["passive_hazard_quadrature_dt_model"] *
                                MANIFEST["seconds_per_model_time"])
            state = advance_source_window(
                state, recovery_seconds, g, pair, rule,
                reuse_small_step_preconditioner=True)
            restart["union_previous_fields"] = tuple(value.copy() for value in state)
            pause_count += 1
            print("REPLAY_TRANSPORT_RECOVERY", event_number, pause_count,
                  result[5]["event_progress_over_b"], flush=True)
        if not result[4] or set(saved) != set(TARGETS):
            raise RuntimeError(
                f"event {event_number}: milestone replay incomplete: {result[5]}")
        q1 = event_dir / "q_1.00b.npz"
        shutil.copy2(final_path, q1)
        with np.load(final_path) as data:
            final_fields = tuple(value.copy() for value in data["fields"])
        one = overlap(event, final_fields, recorded_row(history, event_number, 1.0),
                      1.0, 1.0)
        one["checkpoint"] = str(q1); one["passed"] = overlap_pass(one)
        row = {
            "event_number": event_number,
            "contact": contact,
            "wall_seconds": time.perf_counter() - started,
            "transport_recoveries_before_q0p75": pause_count,
            "milestones": [saved[q] for q in TARGETS] + [one],
        }
        row["passed"] = all(item["passed"] for item in row["milestones"])
        audit["events"] = [item for item in audit["events"]
                           if int(item["event_number"]) != event_number] + [row]
        audit["events"].sort(key=lambda item: int(item["event_number"]))
        audit["all_completed_events_passed"] = bool(
            len(audit["events"]) == 12 and all(item["passed"] for item in audit["events"]))
        atomic_json(audit_path, audit)
        if not row["passed"]:
            raise RuntimeError(f"event {event_number}: replay overlap failed")
        print("EVENT_MILESTONES_VERIFIED", event_number, row["wall_seconds"], flush=True)
    for event_row in audit["events"]:
        for milestone in event_row["milestones"]:
            checkpoint = Path(milestone["checkpoint"])
            milestone["checkpoint_size_bytes"] = checkpoint.stat().st_size
            milestone["checkpoint_sha256"] = file_sha256(checkpoint)
    atomic_json(audit_path, audit)
    print("ALL_EVENT_MILESTONES_VERIFIED", audit["all_completed_events_passed"], flush=True)


if __name__ == "__main__":
    main()
