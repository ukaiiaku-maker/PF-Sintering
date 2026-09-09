from __future__ import annotations

import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pr_avalanche_renewal_five import (  # noqa: E402
    ScalarOutput,
    load_continuation_history_prefix,
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_restart_history_drops_only_semantically_identical_boundary(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    fields = dict(
        cycle="1", avalanche_id="0", event_number="0",
        q_event_over_b="0.0", Q_avalanche_over_b="0.0",
        Q_cumulative_over_b="0.0", S_completed="0",
        source_alive="0", event_transport_active="0")
    _write(first, [
        dict(sample_id="0", t_model="0.0", frame_type="loading", **fields),
        dict(sample_id="1", t_model="1.0", frame_type="loading", **fields),
    ])
    _write(second, [
        dict(sample_id="0", t_model="1.0", frame_type="loading", **fields),
        dict(sample_id="1", t_model="1.0", frame_type="root_nucleation",
             **fields),
        dict(sample_id="2", t_model="2.0", frame_type="active_1b_transit",
             **fields),
    ])

    rows = load_continuation_history_prefix([first, second])

    assert [row["t_model"] for row in rows] == ["0.0", "1.0", "1.0", "2.0"]
    assert [row["frame_type"] for row in rows] == [
        "loading", "loading", "root_nucleation", "active_1b_transit"]
    assert [row["sample_id"] for row in rows] == [0, 1, 2, 3]


def test_scalar_output_continues_global_sample_ids(tmp_path):
    prefix = [dict(sample_id=0, t_model=0.0, marker="parent")]
    output = ScalarOutput(tmp_path, history_prefix_rows=prefix)
    branches = {
        "negative": {"z_m": [0.0], "r_m": [1.0]},
        "positive": {"z_m": [0.0], "r_m": [1.0]},
    }

    row = output.add(dict(t_model=1.0), branches, marker="child")
    output.flush()

    assert row["sample_id"] == 1
    with (tmp_path / "avalanche_renewal_history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ["0", "1"]
    assert [row["marker"] for row in rows] == ["parent", "child"]
