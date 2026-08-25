"""Compact append-only geometry archive for axisymmetric PR renewal movies.

This is an output observer only.  Resampled contours are never returned to the
phase-field solver or used to update a physical state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


FRAME_TYPES = {
    "loading": 0,
    "root_nucleation": 1,
    "active_1b_transit": 2,
    "correlation_window": 3,
    "avalanche_extinction": 4,
    "child_completion": 5,
    "post_avalanche_reload": 6,
}

FLOAT_STATE_FIELDS = (
    "q_event_over_b",
    "Q_avalanche_over_b",
    "Q_cumulative_over_b",
    "sigma_local_Pa",
    "sigma_integral_Pa",
    "r_neck_m",
    "Vp_over_Vp_cycle",
    "z_TJ_m",
    "r_TJ_m",
    "z_GB_m",
    "r_GB_m",
    "root_hazard",
    "root_threshold",
    "descendant_hazard",
    "descendant_threshold",
)

INT_STATE_FIELDS = (
    "cycle",
    "avalanche_id",
    "event_number",
    "sink_state",
    "avalanche_active",
    "S_completed",
    "pending_children",
)


def resample_branch_by_arclength(
        z_m, r_m, *, npoint: int, z_TJ_m: float, r_TJ_m: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """Orient a physical branch from the TJ and resample it by arc length."""
    z = np.asarray(z_m, dtype=float).reshape(-1)
    r = np.asarray(r_m, dtype=float).reshape(-1)
    if npoint < 2:
        raise ValueError("npoint must be at least two")
    if z.size != r.size or z.size < 2:
        raise ValueError("branch coordinates must have equal length >= 2")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(r)):
        raise ValueError("branch coordinates must be finite")
    if np.min(r) < -1e-15:
        raise ValueError("branch radius cannot be negative")
    d_first = np.hypot(z[0] - z_TJ_m, r[0] - r_TJ_m)
    d_last = np.hypot(z[-1] - z_TJ_m, r[-1] - r_TJ_m)
    if d_last < d_first:
        z = z[::-1]
        r = r[::-1]
    ds = np.hypot(np.diff(z), np.diff(r))
    keep = np.r_[True, ds > np.finfo(float).eps]
    z = z[keep]
    r = r[keep]
    if z.size < 2:
        raise ValueError("branch has zero resolved arc length")
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(z), np.diff(r)))]
    target = np.linspace(0.0, s[-1], int(npoint))
    return np.interp(target, s, z), np.interp(target, s, r)


class MovieGeometryArchive:
    """Append one fixed-size contour frame at a time to a crash-safe HDF5."""

    def __init__(self, path: str | Path, *, nbranch: int = 256,
                 seconds_per_model_time: float, metadata: Mapping | None = None,
                 mode: str = "x") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.nbranch = int(nbranch)
        self.seconds_per_model_time = float(seconds_per_model_time)
        self.file = h5py.File(self.path, mode)
        self._create(metadata or {})
        self.last_t_model = -np.inf

    def _create(self, metadata: Mapping) -> None:
        file = self.file
        file.attrs["format"] = "axisymmetric_pr_movie_geometry_v1"
        file.attrs["N_branch"] = self.nbranch
        file.attrs["coordinate_units"] = "m"
        file.attrs["time_units"] = "model_time and physical_s"
        file.attrs["symmetry_axis"] = "r=0"
        file.attrs["branch_ordering"] = "each branch begins at TJ; proceeds away from TJ"
        file.attrs["negative_branch_role"] = "substrate_or_neighbor_side"
        file.attrs["positive_branch_role"] = "particle_side"
        file.attrs["negative_terminal_closure"] = "z-min planar terminal cap"
        file.attrs["positive_terminal_closure"] = "symmetry-axis cap at r=0"
        file.attrs["GB_convention"] = "circular internal disk at z_GB=z_TJ, r_GB=r_TJ"
        file.attrs["frame_type_codes_json"] = json.dumps(FRAME_TYPES, sort_keys=True)
        file.attrs["metadata_json"] = json.dumps(dict(metadata), sort_keys=True)
        geometry = file.create_group("geometry")
        for name in ("z_negative_m", "r_negative_m", "z_positive_m", "r_positive_m"):
            geometry.create_dataset(
                name, shape=(0, self.nbranch), maxshape=(None, self.nbranch),
                dtype="f4", chunks=(16, self.nbranch), compression="gzip",
                compression_opts=4, shuffle=True)
        time = file.create_group("time")
        for name in ("t_model", "t_s"):
            time.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8",
                                chunks=(256,), compression="gzip", compression_opts=4)
        state = file.create_group("state")
        state.create_dataset("frame_id", shape=(0,), maxshape=(None,), dtype="i8",
                             chunks=(256,), compression="gzip", compression_opts=4)
        state.create_dataset("frame_type", shape=(0,), maxshape=(None,), dtype="i1",
                             chunks=(256,), compression="gzip", compression_opts=4)
        for name in FLOAT_STATE_FIELDS:
            state.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8",
                                 chunks=(256,), compression="gzip", compression_opts=4)
        for name in INT_STATE_FIELDS:
            dtype = "i1" if name in ("sink_state", "avalanche_active") else "i4"
            state.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                 chunks=(256,), compression="gzip", compression_opts=4)
        file.flush()

    @property
    def nframe(self) -> int:
        return int(self.file["time/t_model"].shape[0])

    def append(self, branches: Mapping, frame: Mapping, *, frame_type: str,
               flush: bool = False) -> int:
        if frame_type not in FRAME_TYPES:
            raise ValueError(f"unknown frame type {frame_type!r}")
        t_model = float(frame["t_model"])
        if not np.isfinite(t_model) or t_model <= self.last_t_model:
            raise ValueError("movie frame model times must increase strictly")
        ztj = float(frame["z_TJ_m"])
        rtj = float(frame["r_TJ_m"])
        coordinates = {}
        for side in ("negative", "positive"):
            z, r = resample_branch_by_arclength(
                branches[side]["z_m"], branches[side]["r_m"],
                npoint=self.nbranch, z_TJ_m=ztj, r_TJ_m=rtj)
            coordinates[f"z_{side}_m"] = z.astype(np.float32)
            coordinates[f"r_{side}_m"] = r.astype(np.float32)
        index = self.nframe
        for name, values in coordinates.items():
            dataset = self.file[f"geometry/{name}"]
            dataset.resize((index + 1, self.nbranch))
            dataset[index] = values
        t_s = float(frame.get(
            "t_s", t_model * self.seconds_per_model_time))
        for name, value in (("t_model", t_model), ("t_s", t_s)):
            dataset = self.file[f"time/{name}"]
            dataset.resize((index + 1,))
            dataset[index] = value
        scalar_values = dict(frame)
        scalar_values.setdefault("z_GB_m", ztj)
        scalar_values.setdefault("r_GB_m", rtj)
        scalar_values.setdefault("root_hazard", np.nan)
        scalar_values.setdefault("root_threshold", np.nan)
        scalar_values.setdefault("descendant_hazard", np.nan)
        scalar_values.setdefault("descendant_threshold", np.nan)
        state = self.file["state"]
        for name, value in (("frame_id", index),
                            ("frame_type", FRAME_TYPES[frame_type])):
            state[name].resize((index + 1,))
            state[name][index] = value
        for name in FLOAT_STATE_FIELDS:
            state[name].resize((index + 1,))
            state[name][index] = float(scalar_values[name])
        for name in INT_STATE_FIELDS:
            state[name].resize((index + 1,))
            state[name][index] = int(scalar_values[name])
        self.last_t_model = t_model
        if flush:
            self.file.flush()
        return index

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.flush()
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
