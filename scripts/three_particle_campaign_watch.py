"""Retain sparse full-field frames from a live renewal campaign."""
from pathlib import Path
import argparse
import json
import shutil
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.)
    args = parser.parse_args()
    frames = args.run/"frames"
    frames.mkdir(exist_ok=True)
    index_path = frames/"index.json"
    rows = json.loads(index_path.read_text()) if index_path.exists() else []
    last_phase = rows[-1]["phase"] if rows else None
    last_q_bin = rows[-1].get("q_bin") if rows else None
    last_reload_bin = rows[-1].get("reload_bin") if rows else None
    while True:
        try:
            history = json.loads((args.run/"history.json").read_text())
            row = history[-1]
            phase = row["phase"]
            q_bin = (int(float(row["q_over_b"])*10+1e-10)
                     if phase == "ACTIVE_ONE_B" else None)
            reload_bin = (int(float(row["time_s"])-float(history[0]["time_s"]))
                          if phase == "RELOAD" else None)
            keep = (not rows or phase != last_phase or
                    (q_bin is not None and q_bin != last_q_bin) or
                    (reload_bin is not None and reload_bin != last_reload_bin))
            if keep:
                number = len(rows)
                target = frames/f"frame_{number:05d}.npz"
                shutil.copyfile(args.run/"trajectory.npz", target)
                with np.load(target) as data:
                    metadata = json.loads(str(data["metadata"]))
                item = dict(
                    frame=number, path=str(target), time_s=row["time_s"],
                    phase=phase, event_number=row["event_number"],
                    completed_avalanches=row["completed_avalanches"],
                    q_over_b=row["q_over_b"], q_bin=q_bin,
                    reload_bin=reload_bin, status=metadata["status"])
                rows.append(item)
                index_path.write_text(json.dumps(rows, indent=2)+"\n")
                last_phase, last_q_bin, last_reload_bin = phase, q_bin, reload_bin
                print("FRAME", number, phase, row["time_s"], row["q_over_b"],
                      flush=True)
            if metadata["status"] != "RUNNING":
                break
        except (FileNotFoundError, json.JSONDecodeError, OSError, KeyError):
            pass
        time.sleep(max(1., args.poll_seconds))


if __name__ == "__main__":
    main()
