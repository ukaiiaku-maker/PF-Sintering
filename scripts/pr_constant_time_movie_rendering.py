#!/usr/bin/env python3
"""Render strictly constant-simulation-time PR production movies."""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pf_sintering.pr_movie_geometry import FRAME_TYPES


LONG_DT_S = 1.0e-3
AVALANCHE_DT_S = 1.0e-4
EVENT_DT_S = 2.5e-7


def _uniform_times(start: float, end: float, dt: float) -> np.ndarray:
    count = int(math.floor((end-start)/dt + 1.0e-12)) + 1
    return start + np.arange(count, dtype=float)*dt


def _bracket(times: np.ndarray, target: float):
    upper = int(np.searchsorted(times, target, side="right"))
    upper = min(max(upper, 1), len(times)-1)
    lower = upper-1
    fraction = ((target-times[lower])/(times[upper]-times[lower])
                if times[upper] > times[lower] else 0.0)
    return lower, upper, float(np.clip(fraction, 0.0, 1.0))


def _interpolate(array: np.ndarray, lower: int, upper: int, fraction: float):
    return (1.0-fraction)*array[lower] + fraction*array[upper]


def _render_grid(data: dict, targets: np.ndarray, target: Path, *,
                 title_prefix: str, playback_s: float) -> dict:
    times = data["times"]
    xlim = data["xlim"]
    rmax = data["rmax"]
    with imageio.get_writer(
            target, mode="I", duration=playback_s, loop=0) as writer:
        for target_s in targets:
            lower, upper, fraction = _bracket(times, float(target_s))
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=90)
            for side, color in (("negative", "#b45309"),
                                ("positive", "#1d4ed8")):
                z = _interpolate(
                    data[f"z_{side}"], lower, upper, fraction)*1e9
                r = _interpolate(
                    data[f"r_{side}"], lower, upper, fraction)*1e9
                ax.plot(z, r, color=color, linewidth=1.8)
                ax.plot(z, -r, color=color, linewidth=1.8)
            ztj = float(_interpolate(
                data["z_TJ"], lower, upper, fraction))*1e9
            rtj = float(_interpolate(
                data["r_TJ"], lower, upper, fraction))*1e9
            sigma = float(_interpolate(
                data["sigma"], lower, upper, fraction))*1e-6
            discrete = lower
            source_alive = int(data["source_alive"][discrete])
            transport = int(data["event_transport_active"][discrete])
            avalanche = int(data["avalanche_id"][discrete])
            event = int(data["event_number"][discrete])
            ax.scatter([ztj, ztj], [rtj, -rtj], c="crimson", s=18)
            ax.set_xlim(*xlim)
            ax.set_ylim(-rmax, rmax)
            ax.set_aspect("equal")
            ax.set_title(
                f"{title_prefix}  t={target_s:.6f} s  "
                f"a={avalanche} e={event}  sigma={sigma:.3f} MPa  "
                f"source={source_alive} transport={transport}")
            ax.set_xlabel("z (nm)")
            ax.set_ylabel("r (nm)")
            ax.grid(alpha=.15)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)
    return dict(path=str(target), frames=len(targets),
                first_time_s=float(targets[0]),
                last_time_s=float(targets[-1]))


def render_constant_time_production_movies(movie_path: Path) -> dict:
    movie_path = Path(movie_path)
    out = movie_path.parent
    with h5py.File(movie_path, "r") as file:
        times = np.asarray(file["time/t_s"], dtype=float)
        data = dict(
            times=times,
            z_negative=np.asarray(file["geometry/z_negative_m"]),
            r_negative=np.asarray(file["geometry/r_negative_m"]),
            z_positive=np.asarray(file["geometry/z_positive_m"]),
            r_positive=np.asarray(file["geometry/r_positive_m"]),
            z_TJ=np.asarray(file["state/z_TJ_m"]),
            r_TJ=np.asarray(file["state/r_TJ_m"]),
            sigma=np.asarray(file["state/sigma_local_Pa"]),
            source_alive=np.asarray(file["state/source_alive"]),
            event_transport_active=np.asarray(
                file["state/event_transport_active"]),
            avalanche_id=np.asarray(file["state/avalanche_id"]),
            event_number=np.asarray(file["state/event_number"]),
            frame_type=np.asarray(file["state/frame_type"]),
        )
    all_z = np.concatenate([data["z_negative"], data["z_positive"]])*1e9
    all_r = np.concatenate([data["r_negative"], data["r_positive"]])*1e9
    data["xlim"] = (float(np.min(all_z)-5.0), float(np.max(all_z)+5.0))
    data["rmax"] = float(np.max(all_r)+5.0)
    long_times = _uniform_times(times[0], times[-1], LONG_DT_S)
    long_result = _render_grid(
        data, long_times, out / "production_morphology_constant_1ms.gif",
        title_prefix="PR production", playback_s=0.04)
    products = dict(long_timescale=long_result, events=[], avalanches=[])
    active_code = FRAME_TYPES["active_1b_transit"]
    event_pairs = sorted(set(zip(
        data["avalanche_id"][data["frame_type"] == active_code].tolist(),
        data["event_number"][data["frame_type"] == active_code].tolist())))
    for avalanche_id, event_number in event_pairs:
        active = np.flatnonzero(
            (data["frame_type"] == active_code)
            & (data["avalanche_id"] == avalanche_id)
            & (data["event_number"] == event_number))
        first = max(int(active[0])-1, 0)
        last = int(active[-1])
        targets = _uniform_times(times[first], times[last], EVENT_DT_S)
        products["events"].append(_render_grid(
            data, targets, out / (
                f"avalanche{avalanche_id}_event{event_number}_"
                "constant_0p25us.gif"),
            title_prefix=f"Avalanche {avalanche_id} event {event_number}",
            playback_s=0.06))
    for avalanche_id in sorted(set(
            value for value in data["avalanche_id"] if value > 0)):
        indices = np.flatnonzero(data["avalanche_id"] == avalanche_id)
        first, last = int(indices[0]), int(indices[-1])
        targets = _uniform_times(times[first], times[last], AVALANCHE_DT_S)
        products["avalanches"].append(_render_grid(
            data, targets,
            out / f"avalanche{avalanche_id}_constant_0p1ms.gif",
            title_prefix=f"Avalanche {avalanche_id}", playback_s=0.05))
    manifest = dict(
        source_archive=str(movie_path),
        long_timescale_grid_definition="t_n=t_0+n*0.001 s",
        long_timescale_dt_s=LONG_DT_S,
        long_timescale_special_frames_appended=False,
        exact_transition_states_remain_in_source_HDF5=True,
        event_dt_s=EVENT_DT_S,
        avalanche_overview_dt_s=AVALANCHE_DT_S,
        morphology_interpolation=(
            "linear between adjacent fixed-arclength archive contours"),
        discrete_state_interpolation="left-continuous",
        **products)
    (out / "constant_time_movie_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest
