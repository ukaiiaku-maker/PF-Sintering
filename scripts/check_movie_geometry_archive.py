#!/usr/bin/env python3
"""Read-only structural validation of a PR movie-geometry HDF5 archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from pf_sintering.pr_movie_geometry import (
    FLOAT_STATE_FIELDS, FRAME_TYPES, INT_STATE_FIELDS,
)


def validate(path: Path) -> dict:
    failures = []
    with h5py.File(path, "r") as file:
        nbranch = int(file.attrs["N_branch"])
        t_model = np.asarray(file["time/t_model"])
        t_s = np.asarray(file["time/t_s"])
        nframe = t_model.size
        if nframe < 1:
            failures.append("archive has no frames")
        if nframe > 1 and not np.all(np.diff(t_model) > 0.0):
            failures.append("model times are not strictly increasing")
        if nframe > 1 and not np.all(np.diff(t_s) > 0.0):
            failures.append("physical times are not strictly increasing")
        geometry_names = (
            "z_negative_m", "r_negative_m", "z_positive_m", "r_positive_m")
        geometry = {name: np.asarray(file[f"geometry/{name}"])
                    for name in geometry_names}
        for name, values in geometry.items():
            if values.shape != (nframe, nbranch):
                failures.append(f"{name} has shape {values.shape}")
            if not np.all(np.isfinite(values)):
                failures.append(f"{name} contains nonfinite values")
        for name in ("r_negative_m", "r_positive_m"):
            if np.min(geometry[name], initial=0.0) < -1e-15:
                failures.append(f"{name} contains negative radii")
        required_state = (
            "frame_id", "frame_type", *FLOAT_STATE_FIELDS, *INT_STATE_FIELDS)
        for name in required_state:
            if file[f"state/{name}"].shape != (nframe,):
                failures.append(f"state/{name} length mismatch")
        ztj = np.asarray(file["state/z_TJ_m"])
        rtj = np.asarray(file["state/r_TJ_m"])
        tj_error_negative = np.hypot(
            geometry["z_negative_m"][:, 0] - ztj,
            geometry["r_negative_m"][:, 0] - rtj)
        tj_error_positive = np.hypot(
            geometry["z_positive_m"][:, 0] - ztj,
            geometry["r_positive_m"][:, 0] - rtj)
        max_tj_error = float(max(
            np.max(tj_error_negative, initial=0.0),
            np.max(tj_error_positive, initial=0.0)))
        if max_tj_error > 1e-12:
            failures.append(f"TJ/profile mismatch {max_tj_error:.6e} m")
        cycle = np.asarray(file["state/cycle"])
        avalanche_id = np.asarray(file["state/avalanche_id"])
        active = np.asarray(file["state/avalanche_active"])
        event = np.asarray(file["state/event_number"])
        if nframe > 1 and np.any(np.diff(cycle) < 0):
            failures.append("cycle identifiers decrease")
        if np.any((active == 1) & (avalanche_id <= 0)):
            failures.append("active avalanche has no positive ID")
        if np.any((event < 0) | (avalanche_id < 0)):
            failures.append("negative avalanche/event identifier")
        known_types = set(FRAME_TYPES.values())
        observed_types = set(np.asarray(file["state/frame_type"]).tolist())
        if not observed_types.issubset(known_types):
            failures.append("unknown frame-type code")
        result = dict(
            status="PASS" if not failures else "FAIL",
            path=str(path), nframe=int(nframe), N_branch=nbranch,
            time_model_range=[float(t_model[0]), float(t_model[-1])]
            if nframe else [],
            time_s_range=[float(t_s[0]), float(t_s[-1])] if nframe else [],
            avalanche_ids=sorted(set(int(x) for x in avalanche_id if x > 0)),
            event_numbers=sorted(set(int(x) for x in event if x > 0)),
            frame_type_counts={
                name: int(np.sum(np.asarray(file["state/frame_type"]) == code))
                for name, code in FRAME_TYPES.items()},
            maximum_TJ_profile_error_m=max_tj_error,
            failures=failures)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = validate(args.archive)
    text = json.dumps(result, indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(text)
    print(text, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
