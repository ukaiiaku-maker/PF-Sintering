"""Relabel premature q=1 event milestones while preserving every original byte."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from three_particle_forced_event import MANIFEST
from pf_sintering.three_particle_event import (
    load_event_checkpoint, save_event_checkpoint)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def q_over_b(restart):
    return float(restart["cumulative_q_m"]) / MANIFEST["b_event_m"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    repairs = []
    for path in sorted((args.run / "milestones").glob("event_*/q_1.00b.npz")):
        fields, restart, contact, label = load_event_checkpoint(path)
        actual = q_over_b(restart)
        if actual >= 1 - 1e-12:
            continue
        event = int(path.parent.name.split("_")[1])
        original_hash = sha256(path)
        relabeled = path.parent / f"partial_attempt_q_{actual:.8f}b.npz"
        if relabeled.exists():
            raise FileExistsError(relabeled)
        os.replace(path, relabeled)
        final_path = args.run / f"event_{event}_final.npz"
        final_fields, final_restart, final_contact, _ = load_event_checkpoint(final_path)
        final_q = q_over_b(final_restart)
        replacement = None
        if final_q >= 1 - 1e-12:
            save_event_checkpoint(
                path, final_fields, final_restart, contact=final_contact,
                label="GENUINE_STOCHASTIC_EVENT_IMMUTABLE_MILESTONE")
            replacement = {
                "path": str(path), "actual_q_over_b": final_q,
                "sha256": sha256(path)}
        repairs.append({
            "event_number": event,
            "incorrect_name": str(path),
            "preserved_partial_path": str(relabeled),
            "preserved_partial_actual_q_over_b": actual,
            "preserved_partial_sha256_before": original_hash,
            "preserved_partial_sha256_after": sha256(relabeled),
            "original_bytes_preserved_exactly": sha256(relabeled) == original_hash,
            "final_event_checkpoint_q_over_b": final_q,
            "correct_q1_replacement": replacement,
            "trajectory_or_stochastic_state_modified": False,
        })
    report = {
        "label": "IMMUTABLE_EVENT_MILESTONE_PREMATURE_Q1_LABEL_REPAIR",
        "cause": "q_1.00b milestone was written before checking a partial event return",
        "code_fix": "write q_1.00b only after completed=True",
        "repairs": repairs,
        "all_original_bytes_preserved": all(
            row["original_bytes_preserved_exactly"] for row in repairs),
        "physics_or_stochastic_state_modified": False,
    }
    target = args.run / "restart_boundaries" / "milestone_label_repair_audit.json"
    temporary = target.with_suffix(".writing.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, target)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
