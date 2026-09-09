#!/usr/bin/env python3
"""Read-only cumulative curvature monitor for current-state transfer events.

Production archives are opened read-only. Diagnostic products are written to
a separate analysis directory and are never fed back to the solver.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.signal import savgol_filter
from skimage.measure import find_contours
import re


FRAME_CHILD_COMPLETION = 5


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str, default=math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def branch_profile(z, r, *, z_tj: float, r_tj: float, W: float,
                   side: str) -> dict[str, np.ndarray]:
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    first = math.hypot(z[0] - z_tj, r[0] - r_tj)
    last = math.hypot(z[-1] - z_tj, r[-1] - r_tj)
    if last < first:
        z, r = z[::-1], r[::-1]
    ds0 = np.hypot(np.diff(z), np.diff(r))
    keep = np.r_[True, ds0 > 1.0e-15 * W]
    z, r = z[keep], r[keep]
    s0 = np.r_[0.0, np.cumsum(np.hypot(np.diff(z), np.diff(r)))]
    spacing = min(float(np.median(np.diff(s0))), W / 8.0)
    spacing = max(spacing, W / 20.0)
    s = np.arange(0.0, s0[-1] + 0.25 * spacing, spacing)
    z = np.interp(s, s0, z)
    r = np.interp(s, s0, r)
    target = max(7, int(round(W / spacing)))
    window = target if target % 2 else target + 1
    window = min(window, len(s) - (1 - len(s) % 2))
    if window < 7:
        raise RuntimeError("branch is too short for curvature monitor")
    poly = min(3, window - 2)
    z_s = savgol_filter(z, window, poly, mode="interp")
    r_s = savgol_filter(r, window, poly, mode="interp")
    dz = savgol_filter(z, window, poly, deriv=1, delta=spacing, mode="interp")
    dr = savgol_filter(r, window, poly, deriv=1, delta=spacing, mode="interp")
    d2z = savgol_filter(z, window, poly, deriv=2, delta=spacing, mode="interp")
    d2r = savgol_filter(r, window, poly, deriv=2, delta=spacing, mode="interp")
    speed = np.maximum(np.hypot(dz, dr), 1.0e-12)
    kappa_m = (dz * d2r - dr * d2z) / speed**3
    kappa_theta = np.divide(
        dz, np.maximum(r_s * speed, 1.0e-30),
        out=np.full_like(dz, np.nan), where=r_s > W / 20.0)
    kappa_total = kappa_m + kappa_theta
    dkappa_ds = np.gradient(kappa_m, s)
    distance_tj = np.hypot(z_s - z_tj, r_s - r_tj)
    distance_edge = np.minimum(
        np.abs(distance_tj - 3.0 * W),
        np.abs(distance_tj - 10.0 * W))
    return dict(
        side=np.full(len(s), side), s_m=s, z_m=z_s, r_m=r_s,
        distance_from_TJ_m=distance_tj,
        distance_to_nearest_mask_edge_m=distance_edge,
        kappa_m_per_m=kappa_m, kappa_theta_per_m=kappa_theta,
        kappa_total_per_m=kappa_total,
        dkappa_m_ds_per_m2=dkappa_ds)


def event_frames(run_dirs: list[Path]) -> list[dict]:
    found = {}
    for run in run_dirs:
        path = run / "avalanche_renewal_movie_geometry.h5"
        if not path.exists():
            continue
        try:
            archive_context = h5py.File(path, "r")
        except BlockingIOError:
            # A live HDF5 writer holds an exclusive lock. Atomic q=1 NPZ
            # checkpoints below provide the same event morphology without
            # touching or attaching to that writer.
            continue
        with archive_context as archive:
            frame_type = np.asarray(archive["state/frame_type"], dtype=int)
            for index in np.flatnonzero(frame_type == FRAME_CHILD_COMPLETION):
                avalanche = int(archive["state/avalanche_id"][index])
                event = int(archive["state/event_number"][index])
                key = (avalanche, event)
                found[key] = dict(
                    run=run, archive=path, index=int(index),
                    avalanche_id=avalanche, event_number=event,
                    t_model=float(archive["time/t_model"][index]),
                    t_s=float(archive["time/t_s"][index]),
                    z_tj=float(archive["state/z_TJ_m"][index]),
                    r_tj=float(archive["state/r_TJ_m"][index]),
                    z_negative=np.asarray(
                        archive["geometry/z_negative_m"][index], dtype=float),
                    r_negative=np.asarray(
                        archive["geometry/r_negative_m"][index], dtype=float),
                    z_positive=np.asarray(
                        archive["geometry/z_positive_m"][index], dtype=float),
                    r_positive=np.asarray(
                        archive["geometry/r_positive_m"][index], dtype=float))
    histories, _ = all_support_rows(run_dirs)
    pattern = re.compile(r"avalanche(\d+)_event(\d+)_q1\.00b\.npz$")
    for run in run_dirs:
        manifest_path = run / "launch_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        shape = tuple(manifest.get("grid_shape", ()))
        if len(shape) != 2:
            continue
        dz = float(manifest["dz_m"])
        dr = float(manifest["dr_m"])
        z_grid = (np.arange(shape[0]) + 0.5) * dz
        r_grid = (np.arange(shape[1]) + 0.5) * dr
        for checkpoint in (run / "checkpoints").glob(
                "avalanche*_event*_q1.00b.npz"):
            match = pattern.search(checkpoint.name)
            if not match:
                continue
            avalanche, event = map(int, match.groups())
            key = (avalanche, event)
            if key in found:
                continue
            stub = dict(avalanche_id=avalanche, event_number=event)
            scalar = nearest_row(histories, stub, q1=True)
            if not scalar:
                continue
            with np.load(checkpoint, allow_pickle=False) as saved:
                f = np.asarray(
                    saved["current_f"] if "current_f" in saved else saved["f"])
            z_tj = number(scalar, "z_TJ_m")
            r_tj = number(scalar, "r_n_m")
            branches = contour_branches_from_field(
                f, z_grid, r_grid, z_tj=z_tj, r_tj=r_tj)
            found[key] = dict(
                run=run, archive=checkpoint, index=-1,
                avalanche_id=avalanche, event_number=event,
                t_model=number(scalar, "t_model"),
                t_s=number(scalar, "t_s"), z_tj=z_tj, r_tj=r_tj,
                z_negative=branches["negative"][0],
                r_negative=branches["negative"][1],
                z_positive=branches["positive"][0],
                r_positive=branches["positive"][1])
    return [found[key] for key in sorted(found)]


def contour_branches_from_field(f, z_grid, r_grid, *, z_tj, r_tj):
    candidates = []
    for contour in find_contours(np.asarray(f, dtype=float), 0.5):
        z = np.interp(contour[:, 0], np.arange(len(z_grid)), z_grid)
        r = np.interp(contour[:, 1], np.arange(len(r_grid)), r_grid)
        distance = np.min(np.hypot(z - z_tj, r - r_tj))
        candidates.append((distance, len(z), z, r))
    if not candidates:
        raise RuntimeError("no f=0.5 contour in event checkpoint")
    _, _, z, r = min(candidates, key=lambda item: (item[0], -item[1]))
    index = int(np.argmin(np.hypot(z - z_tj, r - r_tj)))
    before = (z[index::-1], r[index::-1])
    after = (z[index:], r[index:])
    if np.mean(before[0][:min(20, len(before[0]))]) < z_tj:
        negative, positive = before, after
    else:
        negative, positive = after, before
    if len(negative[0]) < 8 or len(positive[0]) < 8:
        raise RuntimeError("TJ split did not resolve both exterior branches")
    return dict(negative=negative, positive=positive)


def all_support_rows(run_dirs: list[Path]):
    histories, subevents = [], []
    for run in run_dirs:
        histories.extend(read_csv(run / "avalanche_renewal_history.csv"))
        subevents.extend(read_csv(run / "one_b_subevents.csv"))
    return histories, subevents


def nearest_row(rows, frame, *, q1=False):
    candidates = [
        row for row in rows
        if int(number(row, "avalanche_id", -1)) == frame["avalanche_id"]
        and int(number(row, "event_number", -1)) == frame["event_number"]
        and (not q1 or number(row, "q_event_over_b", -1.0) >= 1.0 - 1e-9)
    ]
    if not candidates:
        return {}
    target_time = number(frame, "t_model")
    if not math.isfinite(target_time):
        # NPZ checkpoint discovery supplies only the event identity.  Prefer
        # the latest matching q=1 scalar record across continuation segments;
        # those rows are exact completion states and may be duplicated in
        # immutable history prefixes.
        return max(candidates, key=lambda row: number(row, "t_model"))
    return min(candidates,
               key=lambda row: abs(number(row, "t_model") - target_time))


def extrema_record(profile: dict, W: float) -> dict:
    s = profile["s_m"]
    # Exclude the physical 0--3W TJ/neck core and the terminal W near a branch
    # endpoint. The latter prevents an axis/end-cap closure from masquerading
    # as a transfer-mask feature.
    valid = ((profile["distance_from_TJ_m"] >= 3.0 * W)
             & (s <= s[-1] - W)
             & np.isfinite(profile["kappa_m_per_m"])
             & np.isfinite(profile["dkappa_m_ds_per_m2"]))
    if not np.any(valid):
        raise RuntimeError("no curvature samples outside physical TJ region")
    indices = np.flatnonzero(valid)
    ik = indices[np.argmax(np.abs(profile["kappa_m_per_m"][valid]))]
    idk = indices[np.argmax(np.abs(profile["dkappa_m_ds_per_m2"][valid]))]
    values = np.abs(profile["kappa_m_per_m"][valid])
    edge_zone = valid & (
        profile["distance_to_nearest_mask_edge_m"] <= 1.0 * W)
    edge_values = np.abs(profile["kappa_m_per_m"][edge_zone])
    edge_gradients = np.abs(profile["dkappa_m_ds_per_m2"][edge_zone])
    return dict(
        max_abs_kappa_per_m=float(abs(profile["kappa_m_per_m"][ik])),
        p95_abs_kappa_per_m=float(np.percentile(values, 95.0)),
        p99_abs_kappa_per_m=float(np.percentile(values, 99.0)),
        kappa_max_index=int(ik),
        max_abs_dkappa_ds_per_m2=float(
            abs(profile["dkappa_m_ds_per_m2"][idk])),
        dkappa_max_index=int(idk),
        max_abs_kappa_within_1W_of_mask_edge_per_m=(
            float(np.max(edge_values)) if edge_values.size else math.nan),
        max_abs_dkappa_ds_within_1W_of_mask_edge_per_m2=(
            float(np.max(edge_gradients))
            if edge_gradients.size else math.nan))


def analyze_event(frame, histories, subevents, *, W: float,
                  out: Path) -> tuple[dict, list[dict]]:
    profiles = [
        branch_profile(frame["z_negative"], frame["r_negative"],
                       z_tj=frame["z_tj"], r_tj=frame["r_tj"], W=W,
                       side="negative"),
        branch_profile(frame["z_positive"], frame["r_positive"],
                       z_tj=frame["z_tj"], r_tj=frame["r_tj"], W=W,
                       side="positive")]
    extrema = [extrema_record(profile, W) for profile in profiles]
    winner = max(range(2), key=lambda i: extrema[i]["max_abs_kappa_per_m"])
    gradient_winner = max(
        range(2), key=lambda i: extrema[i]["max_abs_dkappa_ds_per_m2"])
    p = profiles[winner]
    e = extrema[winner]
    ip = e["kappa_max_index"]
    pg = profiles[gradient_winner]
    eg = extrema[gradient_winner]
    ig = eg["dkappa_max_index"]
    scalar = nearest_row(histories, frame, q1=True)
    event = nearest_row(subevents, frame)
    combined_abs = np.concatenate([
        np.abs(profile["kappa_m_per_m"])[
            (profile["distance_from_TJ_m"] >= 3.0 * W)
            & (profile["s_m"] <= profile["s_m"][-1] - W)]
        for profile in profiles])
    edge_kappa = max(
        item["max_abs_kappa_within_1W_of_mask_edge_per_m"]
        for item in extrema)
    edge_gradient = max(
        item["max_abs_dkappa_ds_within_1W_of_mask_edge_per_m2"]
        for item in extrema)
    row = dict(
        avalanche_id=frame["avalanche_id"],
        event=frame["event_number"], t_model=frame["t_model"],
        t_s=frame["t_s"],
        max_abs_kappa_outside_TJ_per_m=e["max_abs_kappa_per_m"],
        p95_abs_kappa_per_m=float(np.percentile(combined_abs, 95.0)),
        p99_abs_kappa_per_m=float(np.percentile(combined_abs, 99.0)),
        max_abs_dkappa_ds_per_m2=eg["max_abs_dkappa_ds_per_m2"],
        max_abs_kappa_within_1W_of_mask_edge_per_m=edge_kappa,
        max_abs_dkappa_ds_within_1W_of_mask_edge_per_m2=edge_gradient,
        curvature_max_side=str(p["side"][ip]),
        curvature_max_s_nm=float(p["s_m"][ip] * 1e9),
        curvature_max_z_nm=float(p["z_m"][ip] * 1e9),
        curvature_max_r_nm=float(p["r_m"][ip] * 1e9),
        curvature_max_distance_from_TJ_nm=float(
            p["distance_from_TJ_m"][ip] * 1e9),
        curvature_max_distance_to_nearest_mask_edge_nm=float(
            p["distance_to_nearest_mask_edge_m"][ip] * 1e9),
        gradient_max_side=str(pg["side"][ig]),
        gradient_max_s_nm=float(pg["s_m"][ig] * 1e9),
        gradient_max_distance_to_nearest_mask_edge_nm=float(
            pg["distance_to_nearest_mask_edge_m"][ig] * 1e9),
        receiver_mask_edge_distance_from_TJ_nm=3.0 * W * 1e9,
        donor_outer_mask_edge_distance_from_TJ_nm=10.0 * W * 1e9,
        neck_radius_nm=frame["r_tj"] * 1e9,
        TJ_z_nm=frame["z_tj"] * 1e9,
        local_stress_MPa=number(scalar, "sigma_local_Pa") * 1e-6,
        integral_stress_MPa=number(scalar, "sigma_integral_Pa") * 1e-6,
        affinity_MPa=number(scalar, "transport_affinity_Pa") * 1e-6,
        local_stress_drop_MPa=number(event, "delta_sigma_local_Pa") * 1e-6,
        integral_stress_drop_MPa=(
            number(event, "delta_sigma_integral_Pa") * 1e-6),
        neck_radius_change_nm=number(event, "delta_r_TJ_m") * 1e9,
        total_volume_relative_error=number(
            scalar, "V_solid_relative_error"),
        curvature_edge_tied=int(
            p["distance_to_nearest_mask_edge_m"][ip] <= 0.75 * W),
        gradient_edge_tied=int(
            pg["distance_to_nearest_mask_edge_m"][ig] <= 0.75 * W),
        artifact_flag=0, artifact_reason="bounded/no progressive evidence")
    profile_rows = []
    for profile in profiles:
        for index in range(len(profile["s_m"])):
            profile_rows.append({
                "avalanche_id": frame["avalanche_id"],
                "event": frame["event_number"],
                **{key: value[index] for key, value in profile.items()}})
    profile_path = out / "profiles" / (
        f"avalanche{frame['avalanche_id']}_event{frame['event_number']}_curvature.csv")
    write_csv(profile_path, profile_rows)
    return row, profile_rows


def apply_progressive_flags(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        if index < 2:
            continue
        window = rows[index - 2:index + 1]
        same_avalanche = len({item["avalanche_id"] for item in window}) == 1
        edge_tied = all(
            item["curvature_edge_tied"] or item["gradient_edge_tied"]
            for item in window)
        maxima = [item[
            "max_abs_kappa_within_1W_of_mask_edge_per_m"] for item in window]
        gradients = [item[
            "max_abs_dkappa_ds_within_1W_of_mask_edge_per_m2"]
                     for item in window]
        grows = ((maxima[0] < maxima[1] < maxima[2]
                  and maxima[2] > 1.5 * maxima[0])
                 or (gradients[0] < gradients[1] < gradients[2]
                     and gradients[2] > 1.5 * gradients[0]))
        if same_avalanche and edge_tied and grows:
            row["artifact_flag"] = 1
            row["artifact_reason"] = (
                "three-event monotone >50% curvature/gradient growth tied "
                "within 0.75W of a transfer-mask edge")


def run_once(campaign_root: Path, out: Path) -> dict:
    run_dirs = sorted(
        path for path in campaign_root.parent.glob(campaign_root.name + "*")
        if path.is_dir() and "artifact_monitor" not in path.name)
    frames = event_frames(run_dirs)
    histories, subevents = all_support_rows(run_dirs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profiles").mkdir(exist_ok=True)
    rows = []
    for frame in frames:
        row, _ = analyze_event(
            frame, histories, subevents, W=10.0e-9, out=out)
        rows.append(row)
    apply_progressive_flags(rows)
    write_csv(out / "per_event_curvature_diagnostics.csv", rows)
    status = dict(
        status=("ARTIFACT_REVIEW_REQUIRED" if any(
            row["artifact_flag"] for row in rows)
                else "BOUNDED_NO_PROGRESSIVE_MASK_EDGE_ARTIFACT"),
        production_files_opened_read_only=True,
        production_physics_modified=False,
        analyzed_events=len(rows),
        analyzed_event_ids=[
            [row["avalanche_id"], row["event"]] for row in rows],
        TJ_exclusion_distance_W=3.0,
        terminal_exclusion_distance_W=1.0,
        mask_edges_distance_from_TJ_W=[3.0, 10.0],
        edge_tie_tolerance_W=0.75,
        progressive_flag_definition=(
            "three successive same-avalanche events, extrema tied to mask "
            "edges, monotone >50% growth in max curvature or gradient"),
        run_directories=[str(path.resolve()) for path in run_dirs],
        updated_unix_time=time.time())
    atomic_json(out / "monitor_status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    previous = None
    while True:
        status = run_once(args.campaign_root, args.out)
        signature = tuple(map(tuple, status["analyzed_event_ids"]))
        if signature != previous:
            print(json.dumps(status, sort_keys=True), flush=True)
            previous = signature
        if not args.watch:
            break
        time.sleep(max(args.poll_seconds, 10.0))


if __name__ == "__main__":
    main()
