"""Read-only forensic audit of the completed seed-20260910 campaign.

This script never calls an evolution operator.  It extracts serialized
production metrology from history.json and computes additional morphology
descriptors only from immutable retained fields.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.colors import ListedColormap
import numpy as np

from pf_sintering.current_state_mass_transfer import build_current_state_masks
from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
from pf_sintering.pr_avalanche import DescendantBarrier, descendant_rate_per_s


PHASES = [
    "POST_TRANSIENT_NEW_TRAJECTORY", "RELOAD", "ROOT_CROSSING",
    "ACTIVE_ONE_B", "ONE_B_COMPLETE", "SOURCE_WINDOW_OPEN",
    "CHILD_CROSSING", "FACILITATED_WINDOW", "EVENT_TRANSPORT_PAUSED",
    "EVENT_TRANSPORT_RECOVERY", "AVALANCHE_EXTINCT_REPINNED",
]
PHASE_COLORS = {
    "POST_TRANSIENT_NEW_TRAJECTORY": "#777777", "RELOAD": "#3b82a0",
    "ROOT_CROSSING": "#29a36a", "ACTIVE_ONE_B": "#d44e41",
    "ONE_B_COMPLETE": "#ed8b2c", "SOURCE_WINDOW_OPEN": "#8357b2",
    "CHILD_CROSSING": "#b43c6c", "FACILITATED_WINDOW": "#7151a0",
    "EVENT_TRANSPORT_PAUSED": "#8d1e2d",
    "EVENT_TRANSPORT_RECOVERY": "#e2775c",
    "AVALANCHE_EXTINCT_REPINNED": "#5d9f55",
}
SIDE_NAMES = [
    ("LEFT", "negative", "LEFT outer"),
    ("LEFT", "positive", "LEFT center"),
    ("RIGHT", "negative", "RIGHT center"),
    ("RIGHT", "positive", "RIGHT outer"),
]
WINDOW_PHASES = {
    "SOURCE_WINDOW_OPEN", "FACILITATED_WINDOW", "CHILD_CROSSING",
    "AVALANCHE_EXTINCT_REPINNED",
}


def finite(value, default=np.nan):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, out, name):
    fig.savefig(out/(name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out/(name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def phase_background(ax, rows, t0, alpha=.06):
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["phase"] != rows[start]["phase"]:
            x0 = rows[start]["time_s"] - t0
            x1 = rows[i-1]["time_s"] - t0
            if x1 == x0:
                x1 += max(1e-8, 1e-6*(rows[-1]["time_s"]-t0))
            ax.axvspan(x0, x1, color=PHASE_COLORS.get(
                rows[start]["phase"], "#aaaaaa"), alpha=alpha, lw=0)
            start = i


def source_extent(f, z):
    line = f[:, 0]
    inside = np.flatnonzero(line >= .5)
    if not len(inside):
        return np.nan, np.nan, np.nan
    i, j = int(inside[0]), int(inside[-1])
    if i == 0 or j == len(z)-1:
        return np.nan, np.nan, np.nan
    lo = z[i-1] + (.5-line[i-1])*(z[i]-z[i-1])/(line[i]-line[i-1])
    hi = z[j] + (.5-line[j])*(z[j+1]-z[j])/(line[j+1]-line[j])
    return float(lo), float(hi), float(hi-lo)


def radius_profile(f, r):
    out = np.full(f.shape[0], np.nan)
    for j, row in enumerate(f):
        edges = np.flatnonzero((row[:-1] >= .5) & (row[1:] < .5))
        if len(edges):
            i = int(edges[-1])
            out[j] = r[i] + (row[i]-.5)/(row[i]-row[i+1])*(r[i+1]-r[i])
    return out


def curvature_descriptors(z, radius, mask):
    ids = np.flatnonzero(mask & np.isfinite(radius))
    if len(ids) < 7:
        return dict(rms=np.nan, p95=np.nan, grad_max=np.nan,
                    grad_z=np.nan, kappa_max=np.nan, kappa_z=np.nan)
    zz, rr = z[ids], radius[ids]
    slope = np.gradient(rr, zz)
    second = np.gradient(slope, zz)
    kappa = -second/(1+slope*slope)**1.5
    grad = np.gradient(kappa, zz)
    edge = min(3, len(ids)//4)
    use = slice(edge, len(ids)-edge) if len(ids) > 2*edge+2 else slice(None)
    ku, gu, zu = kappa[use], grad[use], zz[use]
    ik = int(np.nanargmax(np.abs(ku))); ig = int(np.nanargmax(np.abs(gu)))
    return dict(
        rms=float(np.sqrt(np.nanmean(ku*ku))),
        p95=float(np.nanpercentile(np.abs(ku), 95)),
        grad_max=float(np.abs(gu[ig])), grad_z=float(zu[ig]),
        kappa_max=float(np.abs(ku[ik])), kappa_z=float(zu[ik]))


def nearest_history_index(times, value):
    i = int(np.searchsorted(times, value))
    candidates = [max(0, i-1), min(len(times)-1, i)]
    return min(candidates, key=lambda k: abs(times[k]-value))


def frame_record(fields, ownership, gb, time_s, z, r, dr, dz,
                 history_row, source_volume, source_extent_m, frame_number):
    f = fields[0]
    eta = np.asarray(fields[1:])
    cell = 2*np.pi*r[None, :]*dr*dz
    volumes = np.sum(eta*cell[None, :, :], axis=(1, 2))
    centroids = np.sum(eta*cell[None, :, :]*z[None, :, None], axis=(1, 2))/volumes
    equivalent = (3*volumes/(4*np.pi))**(1/3)
    radius = radius_profile(f, r)
    lo, hi, extent = source_extent(f, z)
    span = float(gb[1]-gb[0])
    center_mask = np.isfinite(radius) & (z > gb[0]) & (z < gb[1])
    center_radius = float(np.nanmax(radius[center_mask])) if np.any(center_mask) else np.nan
    center_aspect = 2*center_radius/span if span > 0 else np.nan
    total_centroid = np.sum(eta*cell[None, :, :]*z[None, :, None], axis=(1, 2))
    zvar = np.sum(eta*cell[None, :, :]
                  *(z[None, :, None]-centroids[:, None, None])**2,
                  axis=(1, 2))/volumes
    r2 = np.sum(eta*cell[None, :, :]*r[None, None, :]**2,
                axis=(1, 2))/volumes
    second_aspect = np.sqrt(np.maximum(r2[1]/2, 0)/max(zvar[1], 1e-60))
    valid = np.flatnonzero(np.isfinite(radius))
    if len(valid) > 2:
        surface_area = float(np.sum(
            2*np.pi*.5*(radius[valid][1:]+radius[valid][:-1])
            * np.hypot(np.diff(z[valid]), np.diff(radius[valid]))))
    else:
        surface_area = np.nan
    W = 4e-9
    regions = {
        "left_outer": z < gb[0]-3*W,
        "center": (z > gb[0]+3*W) & (z < gb[1]-3*W),
        "right_outer": z > gb[1]+3*W,
    }
    output = dict(
        frame=frame_number, time_s=time_s, phase=history_row["phase"],
        avalanche_id=history_row["avalanche_id"],
        event_number=history_row["event_number"], q_over_b=history_row["q_over_b"],
        active_contact=history_row.get("contact"),
        V_left_m3=volumes[0], V_center_m3=volumes[1], V_right_m3=volumes[2],
        R_left_m=equivalent[0], R_center_m=equivalent[1], R_right_m=equivalent[2],
        V_center_over_source=volumes[1]/source_volume,
        centroid_left_m=centroids[0], centroid_center_m=centroids[1],
        centroid_right_m=centroids[2],
        outer_centroid_separation_m=centroids[2]-centroids[0],
        GB_LEFT_m=float(gb[0]), GB_RIGHT_m=float(gb[1]), center_span_m=span,
        near_axis_low_m=lo, near_axis_high_m=hi, near_axis_extent_m=extent,
        near_axis_extent_strain=1-extent/source_extent_m,
        center_max_radius_m=center_radius, center_envelope_aspect_ratio=center_aspect,
        center_second_moment_aspect_ratio=float(second_aspect),
        surface_area_proxy_m2=surface_area,
        LEFT_stress_Pa=history_row["contacts"]["LEFT"]["sigma_local_Pa"],
        RIGHT_stress_Pa=history_row["contacts"]["RIGHT"]["sigma_local_Pa"],
        quota_strain=history_row["production_densification_strain"],
        centroid_strain=history_row["geometric_chain_strain"],
        mirror_error=float(np.max(np.abs(f-f[::-1]))),
    )
    for name, mask in regions.items():
        desc = curvature_descriptors(z, radius, mask)
        for key, value in desc.items():
            output[f"{name}_curvature_{key}"] = value
    for contact in ("LEFT", "RIGHT"):
        c = history_row["contacts"][contact]
        output[f"{contact}_TJ_radius_m"] = c["r_n_m"]
        output[f"{contact}_TJ_z_m"] = c["z_TJ_m"]
    return output


def flatten_history(history, frame_rows):
    ft = np.array([r["time_s"] for r in frame_rows])
    rows = []
    max_residual = 0.0
    for row in history:
        d = row["diagnostics"]
        out = dict(
            time_s=row["time_s"], elapsed_s=row["time_s"]-history[0]["time_s"],
            phase=row["phase"], active_contact=row.get("contact"),
            avalanche_id=row["avalanche_id"], event_number=row["event_number"],
            q_over_b=row["q_over_b"], cumulative_q_over_b=row["cumulative_event_quota_over_b"],
            quota_strain=row["production_densification_strain"],
            centroid_strain=row["geometric_chain_strain"],
            source_amplitude_h=row["source_amplitude"],
            descendant_barrier_lowering_eV=1.5*row["source_amplitude"],
            descendant_hazard=row["descendant_hazard"],
            descendant_threshold=row["descendant_threshold"],
            descendant_H_over_Hstar=row.get("descendant_H_over_Hstar"),
            root_hazard_LEFT=row["hazards"]["LEFT"],
            root_hazard_RIGHT=row["hazards"]["RIGHT"],
            root_threshold_LEFT=row["thresholds"]["LEFT"],
            root_threshold_RIGHT=row["thresholds"]["RIGHT"],
            root_H_over_Hstar_LEFT=row["H_over_Hstar"]["LEFT"],
            root_H_over_Hstar_RIGHT=row["H_over_Hstar"]["RIGHT"],
            transport_paused=row["phase"] == "EVENT_TRANSPORT_PAUSED",
            recovery_active=row["phase"] == "EVENT_TRANSPORT_RECOVERY",
            source_window_deadline_s=row["source_window_deadline_s"],
            energy_J=d["energy_J"], V_left_m3=d["V_left_m3"],
            V_center_m3=d["V_center_m3"], V_right_m3=d["V_right_m3"],
            R_left_m=d["R_left_m"], R_center_m=d["R_center_m"],
            R_right_m=d["R_right_m"], centroid_left_m=d["centroid_left_m"],
            centroid_center_m=d["centroid_center_m"], centroid_right_m=d["centroid_right_m"],
            outer_centroid_separation_m=d["centroid_right_m"]-d["centroid_left_m"],
            GB_LEFT_m=d["LEFT_gb_z_m"], GB_RIGHT_m=d["RIGHT_gb_z_m"],
            TJ_LEFT_m=d["LEFT_tj_z_m"], TJ_RIGHT_m=d["RIGHT_tj_z_m"],
            center_span_m=d["RIGHT_tj_z_m"]-d["LEFT_tj_z_m"],
            center_mean_local_Pa=row["center_particle_mean_local_Pa"],
            cluster_weighted_local_Pa=row["cluster_area_weighted_local_Pa"],
            cluster_weighted_integral_Pa=row["cluster_area_weighted_integral_Pa"],
            Cannon_Carter_LEFT_Pa=d["LEFT_CC_stress_Pa"],
            Cannon_Carter_RIGHT_Pa=d["RIGHT_CC_stress_Pa"],
            Cannon_Carter_LEFT_force_N=d["LEFT_CC_force_N"],
            Cannon_Carter_RIGHT_force_N=d["RIGHT_CC_force_N"],
        )
        deadline = finite(row["source_window_deadline_s"])
        out["source_lifetime_remaining_s"] = (
            max(0., deadline-row["time_s"]) if np.isfinite(deadline) else np.nan)
        out["source_window_age_s"] = (
            max(0., row["time_s"]-(deadline-.009))
            if np.isfinite(deadline) and row["phase"] in WINDOW_PHASES else np.nan)
        for contact in ("LEFT", "RIGHT"):
            c = row["contacts"][contact]
            out[f"{contact}_sigma_local_Pa"] = c["sigma_local_Pa"]
            out[f"{contact}_sigma_integral_continuous_Pa"] = c["sigma_integral_continuous_Pa"]
            out[f"{contact}_root_rate_per_s"] = c["root_rate_per_s"]
            out[f"{contact}_transport_affinity_Pa"] = c["transport_affinity_Pa"]
            out[f"{contact}_TJ_radius_m"] = c["r_n_m"]
            out[f"{contact}_neck_radius_m"] = d[f"{contact}_neck_r_m"]
            out[f"{contact}_contact_area_m2"] = np.pi*c["r_n_m"]**2
            out[f"{contact}_contact_angle_mean_rad"] = .5*(
                c["theta_negative_rad"]+c["theta_positive_rad"])
            for side in ("negative", "positive"):
                theta = c[f"theta_{side}_rad"]
                sigma_kappa = -c["gamma_J_per_m2"]*c[f"kappa1_{side}_per_m"]
                sigma_tj = 3*c["gamma_J_per_m2"]*np.sin(theta/2)/c["r_n_m"]
                sigma_sum = sigma_kappa+sigma_tj
                residual = sigma_sum-c[f"sigma_local_{side}_Pa"]
                max_residual = max(max_residual, abs(residual))
                prefix = f"{contact}_{side}"
                out[f"{prefix}_theta_rad"] = theta
                out[f"{prefix}_kappa_meridional_per_m"] = c[f"kappa1_{side}_per_m"]
                out[f"{prefix}_sigma_kappa_Pa"] = sigma_kappa
                out[f"{prefix}_sigma_TJ_Pa"] = sigma_tj
                out[f"{prefix}_sigma_local_Pa"] = c[f"sigma_local_{side}_Pa"]
                out[f"{prefix}_decomposition_residual_Pa"] = residual
                out[f"{prefix}_continuous_turning_rad"] = c[
                    f"delta_phi_f_continuous_{side}_rad"]
                out[f"{prefix}_branch_length_m"] = c[f"L_f_{side}_m"]
                out[f"{prefix}_continuous_curvature_per_m"] = (
                    c[f"delta_phi_f_continuous_{side}_rad"]/c[f"L_f_{side}_m"])
                out[f"{prefix}_sigma_integral_continuous_Pa"] = c[
                    f"sigma_integral_continuous_{side}_Pa"]
            max_residual = max(max_residual, abs(
                c["sigma_local_Pa"]-.5*(c["sigma_local_negative_Pa"]+
                                          c["sigma_local_positive_Pa"])))
        if len(ft):
            fi = nearest_history_index(ft, row["time_s"])
            out["nearest_frame"] = frame_rows[fi]["frame"]
            out["nearest_frame_time_error_s"] = ft[fi]-row["time_s"]
            out["near_axis_extent_strain_nearest_frame"] = frame_rows[fi][
                "near_axis_extent_strain"]
        rows.append(out)
    for i, out in enumerate(rows):
        reference = rows[0]
        for key in ("GB_LEFT_m","GB_RIGHT_m","TJ_LEFT_m","TJ_RIGHT_m",
                    "outer_centroid_separation_m"):
            out["delta_"+key] = out[key]-reference[key]
        if i == 0:
            out.update(transport_clock_increment_s=0., qdot_m_per_s=np.nan,
                       quota_increment_over_b=0.)
            continue
        previous = rows[i-1]
        dt = out["time_s"]-previous["time_s"]
        dq = out["cumulative_q_over_b"]-previous["cumulative_q_over_b"]
        is_transfer = dq > 1e-12 and out["event_number"] == previous["event_number"]
        out["transport_clock_increment_s"] = dt if is_transfer else 0.
        out["quota_increment_over_b"] = dq if is_transfer else 0.
        out["qdot_m_per_s"] = dq*0.25e-9/dt if is_transfer and dt > 0 else np.nan
    return rows, max_residual


def event_rows(history_flat, frame_rows, b_m):
    events = []
    avalanches = []
    for number in range(1, 14):
        segment = [r for r in history_flat if r["event_number"] == number]
        start = next(r for r in segment if r["phase"] == "ACTIVE_ONE_B" and r["q_over_b"] == 0)
        end = next(r for r in segment if r["phase"] == "ONE_B_COMPLETE")
        active = [r for r in segment if start["time_s"] <= r["time_s"] <= end["time_s"]]
        durations = interval_totals(active)
        intended_transfer = 0.
        for a, b in zip(active, active[1:]):
            dq = b["cumulative_q_over_b"]-a["cumulative_q_over_b"]
            if dq > 0:
                intended_transfer += np.pi*a["LEFT_TJ_radius_m"]**2*dq*b_m
        row = dict(
            avalanche_id=start["avalanche_id"], event_number=number,
            active_contact=start["active_contact"],
            event_kind="root" if number in (1, 6) else "descendant",
            start_time_s=start["time_s"], end_time_s=end["time_s"],
            event_duration_s=end["time_s"]-start["time_s"], q_start=0., q_end=1.,
            quota_strain_increment=end["quota_strain"]-start["quota_strain"],
            centroid_strain_increment=end["centroid_strain"]-start["centroid_strain"],
            extent_strain_increment_nearest_frames=(
                end["near_axis_extent_strain_nearest_frame"]-
                start["near_axis_extent_strain_nearest_frame"]),
            transport_affinity_min_Pa=min(r["LEFT_transport_affinity_Pa"] for r in active),
            transport_affinity_max_Pa=max(r["LEFT_transport_affinity_Pa"] for r in active),
            accepted_transfer_clock_s=durations["accepted_transfer_clock_s"],
            fixed_q_recovery_s=durations["fixed_q_recovery_s"],
            source_amplitude_h=start["source_amplitude_h"],
            descendant_barrier_lowering_eV=start["descendant_barrier_lowering_eV"],
            intended_conservative_transfer_volume_m3=intended_transfer,
        )
        for key in ("V_left_m3", "V_center_m3", "V_right_m3",
                    "LEFT_sigma_local_Pa", "RIGHT_sigma_local_Pa", "energy_J"):
            row[key+"_before"] = start[key]
            row[key+"_after"] = end[key]
            row["delta_"+key] = end[key]-start[key]
        for contact in ("LEFT", "RIGHT"):
            for side in ("negative", "positive"):
                key = f"{contact}_{side}_sigma_local_Pa"
                row[f"delta_{key}"] = end[key]-start[key]
            for key in ("GB_"+contact+"_m", "TJ_"+contact+"_m",
                        contact+"_TJ_radius_m", contact+"_neck_radius_m",
                        contact+"_contact_angle_mean_rad"):
                row[key+"_before"] = start[key]
                row[key+"_after"] = end[key]
                row["delta_"+key] = end[key]-start[key]
        row["delta_outer_centroid_separation_m"] = (
            end["outer_centroid_separation_m"]-start["outer_centroid_separation_m"])
        # The saved event records combine conservative source and fast PF relaxation.
        row["source_vs_fast_relaxation_separable"] = False
        end_index = history_flat.index(end)
        next_active = next((r for r in history_flat[end_index+1:]
                            if r["phase"] == "ACTIVE_ONE_B" and r["q_over_b"] == 0
                            and r["avalanche_id"] == start["avalanche_id"]), None)
        extinction = next((r for r in history_flat[end_index+1:]
                           if r["phase"] == "AVALANCHE_EXTINCT_REPINNED" and
                           r["avalanche_id"] == start["avalanche_id"]), None)
        window_end = next_active or extinction
        next_root = next((r for r in history_flat[end_index+1:]
                          if r["phase"] == "ROOT_CROSSING"), None)
        for grain in ("left", "center", "right"):
            key=f"V_{grain}_m3"
            row[f"delta_{key}_source_window"] = (
                window_end[key]-end[key] if window_end is not None else np.nan)
            row[f"delta_{key}_following_reload"] = (
                next_root[key]-extinction[key]
                if extinction is not None and next_root is not None and
                window_end is extinction else np.nan)
        events.append(row)
    for aid in (1, 2):
        group = [r for r in events if r["avalanche_id"] == aid]
        start, end = group[0], group[-1]
        avalanches.append(dict(
            avalanche_id=aid, event_count=len(group), first_event=group[0]["event_number"],
            last_event=group[-1]["event_number"], start_time_s=start["start_time_s"],
            end_time_s=end["end_time_s"], active_event_duration_s=sum(
                r["event_duration_s"] for r in group),
            accepted_transfer_clock_s=sum(r["accepted_transfer_clock_s"] for r in group),
            fixed_q_recovery_s=sum(r["fixed_q_recovery_s"] for r in group),
            delta_V_center_m3=sum(r["delta_V_center_m3"] for r in group),
            quota_strain_increment=sum(r["quota_strain_increment"] for r in group),
            centroid_strain_increment=sum(r["centroid_strain_increment"] for r in group)))
    return events, avalanches


def interval_totals(rows):
    totals = dict(accepted_transfer_clock_s=0., fixed_q_recovery_s=0.,
                  source_window_s=0., other_s=0.)
    for a, b in zip(rows, rows[1:]):
        dt = b["time_s"]-a["time_s"]
        dq = b["cumulative_q_over_b"]-a["cumulative_q_over_b"]
        if dq > 1e-12:
            key = "accepted_transfer_clock_s"
        elif b["phase"] == "EVENT_TRANSPORT_RECOVERY":
            key = "fixed_q_recovery_s"
        elif b["phase"] in WINDOW_PHASES:
            key = "source_window_s"
        else:
            key = "other_s"
        totals[key] += dt
    return totals


def pause_rows(history_flat):
    output = []
    i = 0
    while i < len(history_flat):
        row = history_flat[i]
        if row["phase"] != "EVENT_TRANSPORT_PAUSED":
            i += 1
            continue
        event, q = row["event_number"], row["q_over_b"]
        j = i+1
        while (j < len(history_flat) and history_flat[j]["event_number"] == event
               and abs(history_flat[j]["q_over_b"]-q) <= 1e-12
               and history_flat[j]["phase"] in {
                   "EVENT_TRANSPORT_PAUSED", "EVENT_TRANSPORT_RECOVERY"}):
            j += 1
        group = history_flat[i:j]
        start, end = group[0], group[-1]
        affinities = [r["LEFT_transport_affinity_Pa"] for r in group]
        record = dict(
            episode=len(output)+1, avalanche_id=start["avalanche_id"],
            event_number=event, q_over_b=q, start_time_s=start["time_s"],
            end_time_s=end["time_s"], duration_s=end["time_s"]-start["time_s"],
            recovery_steps=sum(r["phase"] == "EVENT_TRANSPORT_RECOVERY" for r in group),
            affinity_start_Pa=affinities[0], affinity_min_Pa=min(affinities),
            affinity_end_Pa=affinities[-1],
        )
        for key in ("LEFT_sigma_local_Pa", "RIGHT_sigma_local_Pa", "V_center_m3",
                    "center_span_m", "LEFT_TJ_radius_m", "RIGHT_TJ_radius_m",
                    "LEFT_neck_radius_m", "RIGHT_neck_radius_m",
                    "GB_LEFT_m", "GB_RIGHT_m", "centroid_strain", "quota_strain",
                    "energy_J"):
            record[key+"_start"] = start[key]
            record[key+"_end"] = end[key]
            record["delta_"+key] = end[key]-start[key]
        output.append(record)
        i = j
    return output


def decision_rows(history, manifest):
    p = manifest["root_barrier_slice"]
    barrier = DescendantBarrier(CompleteExpFloorParams(
        p["G0_eV"], p["Gfloor_eV"], p["a"], p["sigmahat_Pa"], p["n"]), 1.5)
    attempt = manifest["clock_scale"]/manifest["seconds_per_model_time"]
    output = []
    for row in history:
        if row["phase"] == "ROOT_CROSSING":
            output.append(dict(
                decision="root", time_s=row["time_s"], avalanche_id=row["avalanche_id"],
                event_number=row["event_number"]+1, selected=row["contact"],
                LEFT_rate_per_s=row["contacts"]["LEFT"]["root_rate_per_s"],
                RIGHT_rate_per_s=row["contacts"]["RIGHT"]["root_rate_per_s"],
                LEFT_hazard=row["hazards"]["LEFT"], RIGHT_hazard=row["hazards"]["RIGHT"],
                LEFT_threshold=row["thresholds"]["LEFT"],
                RIGHT_threshold=row["thresholds"]["RIGHT"],
                LEFT_eligible=True, RIGHT_eligible=True,
                selection_semantics="first independent root clock to cross"))
        elif row["phase"] == "CHILD_CROSSING":
            rates = {}
            for contact in ("LEFT", "RIGHT"):
                c = row["contacts"][contact]
                rates[contact] = descendant_rate_per_s(
                    c["sigma_local_Pa"], c["r_n_m"], barrier=barrier,
                    temperature_K=manifest["temperature_K"],
                    attempt_frequency_per_s=attempt, b_m=manifest["b_event_m"],
                    source_amplitude=row["source_amplitude"])
            output.append(dict(
                decision="descendant", time_s=row["time_s"],
                avalanche_id=row["avalanche_id"], event_number=row["event_number"]+1,
                selected=row["contact"], LEFT_rate_per_s=rates["LEFT"],
                RIGHT_rate_per_s=rates["RIGHT"],
                LEFT_hazard=row["descendant_hazard"], RIGHT_hazard=np.nan,
                LEFT_threshold=row["descendant_threshold"], RIGHT_threshold=np.nan,
                LEFT_eligible=row["contact"] == "LEFT",
                RIGHT_eligible=row["contact"] == "RIGHT",
                selection_semantics=(
                    "serialized descendant remains on root-selected contact; opposite "
                    "hypothetical rate is diagnostic only")))
    return output


def plot_history(rows, out):
    t0 = rows[0]["time_s"]
    t = np.array([r["elapsed_s"] for r in rows])
    def a(key, scale=1.): return np.array([finite(r.get(key))*scale for r in rows])
    roots = [r["elapsed_s"] for r in rows if r["phase"] == "ROOT_CROSSING"]
    events = [r["elapsed_s"] for r in rows if r["phase"] == "ONE_B_COMPLETE"]
    def marks(ax):
        for x in roots: ax.axvline(x, color="k", ls="--", lw=.8)
        for x in events: ax.axvline(x, color="#888", lw=.25, alpha=.5)
        phase_background(ax, rows, t0)
        ax.grid(alpha=.2)

    # A
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax[0].plot(t, a("LEFT_sigma_local_Pa", 1e-6), label="LEFT production")
    ax[0].plot(t, a("RIGHT_sigma_local_Pa", 1e-6), label="RIGHT production")
    ax[0].plot(t, a("LEFT_sigma_integral_continuous_Pa", 1e-6), "--", label="LEFT integral")
    ax[0].plot(t, a("RIGHT_sigma_integral_continuous_Pa", 1e-6), "--", label="RIGHT integral")
    for contact, side, label in SIDE_NAMES:
        ax[1].plot(t, a(f"{contact}_{side}_sigma_local_Pa", 1e-6), label=label)
    for key, label in [("center_mean_local_Pa", "center mean"),
                       ("cluster_weighted_local_Pa", "cluster weighted"),
                       ("Cannon_Carter_LEFT_Pa", "CC LEFT"),
                       ("Cannon_Carter_RIGHT_Pa", "CC RIGHT")]:
        ax[2].plot(t, a(key, 1e-6), label=label)
    for axis in ax: marks(axis); axis.legend(ncol=4, fontsize=8); axis.set_ylabel("MPa")
    ax[-1].set_xlabel("Time since production source (s)")
    ax[0].set_title("A — exact production stresses and non-activation diagnostics")
    savefig(fig, out, "figure_A_stresses")

    # B
    fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    for grain in ("left", "center", "right"):
        ax[0].plot(t, a(f"V_{grain}_m3", 1e21), label=grain)
        ax[1].plot(t, a(f"R_{grain}_m", 1e9), label=grain)
    ax[2].plot(t, a("center_span_m", 1e9), label="center span")
    ax[2].plot(t, a("outer_centroid_separation_m", 1e9), label="outer centroids")
    for key in ("GB_LEFT_m", "GB_RIGHT_m", "TJ_LEFT_m", "TJ_RIGHT_m"):
        ax[2].plot(t, a(key, 1e9), label=key)
    for contact in ("LEFT", "RIGHT"):
        ax[3].plot(t, a(f"{contact}_TJ_radius_m", 1e9), label=f"{contact} TJ/neck")
        for side in ("negative", "positive"):
            ax[3].plot(t, np.rad2deg(a(f"{contact}_{side}_theta_rad")), "--",
                       label=f"{contact} {side} angle (deg)")
    ax[0].set_ylabel("Volume (10$^{-21}$ m$^3$)"); ax[1].set_ylabel("Eq. radius (nm)")
    ax[2].set_ylabel("Axial position/length (nm)"); ax[3].set_ylabel("Radius (nm) / angle (deg)")
    for axis in ax: marks(axis); axis.legend(ncol=4, fontsize=7)
    ax[-1].set_xlabel("Time since production source (s)"); ax[0].set_title("B — geometry")
    savefig(fig, out, "figure_B_geometry")

    # C
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax[0].plot(t, 100*a("quota_strain"), label="quota strain")
    ax[0].plot(t, 100*a("centroid_strain"), label="centroid strain")
    ax[0].plot(t, 100*a("near_axis_extent_strain_nearest_frame"), label="extent (nearest frame)")
    for key in ("GB_LEFT_m", "GB_RIGHT_m", "TJ_LEFT_m", "TJ_RIGHT_m"):
        ax[1].plot(t, (a(key)-a(key)[0])*1e9, label="Δ"+key)
    ax[2].plot(t, (a("outer_centroid_separation_m")-
                         a("outer_centroid_separation_m")[0])*1e9,
               label="Δ outer-centroid spacing")
    for axis in ax: marks(axis); axis.legend(fontsize=8); axis.set_ylabel("%" if axis is ax[0] else "nm")
    ax[-1].set_xlabel("Time since production source (s)"); ax[0].set_title("C — strain and displacement")
    savefig(fig, out, "figure_C_strain_displacement")

    # D
    fig, ax = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    ax[0].plot(t, a("root_H_over_Hstar_LEFT"), label="root LEFT")
    ax[0].plot(t, a("root_H_over_Hstar_RIGHT"), label="root RIGHT")
    ax[0].plot(t, a("descendant_H_over_Hstar"), label="descendant")
    ax[0].axhline(1, color="k", lw=.7)
    ax[1].plot(t, a("source_amplitude_h"), label="h")
    ax[1].plot(t, a("source_lifetime_remaining_s")/.009, label="lifetime remaining / 9ms")
    ax[2].plot(t, a("q_over_b"), label="q/b")
    ax[2].plot(t, a("LEFT_transport_affinity_Pa", 1e-6), label="LEFT affinity (MPa)")
    phase_id = np.array([PHASES.index(r["phase"]) for r in rows])
    ax[3].step(t, phase_id, where="post", color="k")
    ax[3].set_yticks(range(len(PHASES)), PHASES, fontsize=6)
    for axis in ax:
        marks(axis)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    ax[-1].set_xlabel("Time since production source (s)"); ax[0].set_title("D — controller state")
    savefig(fig, out, "figure_D_controller")

    # E
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (contact, side, label) in zip(axes.ravel(), SIDE_NAMES):
        axis.plot(t, a(f"{contact}_{side}_sigma_kappa_Pa", 1e-6), label=r"$-\gamma k_m$")
        axis.plot(t, a(f"{contact}_{side}_sigma_TJ_Pa", 1e-6), label="TJ term")
        axis.plot(t, a(f"{contact}_{side}_sigma_local_Pa", 1e-6), label="sum/local")
        axis.set_title(label); axis.set_ylabel("MPa"); marks(axis); axis.legend(fontsize=8)
    for axis in axes[-1]: axis.set_xlabel("Time since production source (s)")
    fig.suptitle("E — exact one-sided production-stress decomposition")
    savefig(fig, out, "figure_E_stress_decomposition")

    # F
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for grain in ("left", "center", "right"):
        ax[0].plot(t, a(f"V_{grain}_m3", 1e21), label=grain)
        dv = np.r_[np.nan, np.diff(a(f"V_{grain}_m3"))]
        dt = np.r_[np.nan, np.diff(t)]
        ax[1].plot(t, np.divide(dv, dt, out=np.full_like(dv, np.nan), where=dt>0)*1e24,
                   label=grain)
    ax[2].plot(t, (a("V_center_m3")/a("V_center_m3")[0]-1)*100, label="center")
    for axis in ax: marks(axis); axis.legend(fontsize=8)
    ax[0].set_ylabel("Volume (10$^{-21}$ m$^3$)"); ax[1].set_ylabel("dV/dt (10$^{-24}$ m$^3$/s)")
    ax[2].set_ylabel("Center ΔV (%)"); ax[-1].set_xlabel("Time since production source (s)")
    ax[0].set_title("F — volume and phase-resolved mass flow")
    savefig(fig, out, "figure_F_mass_flow")

    # G
    loss = (1-a("V_center_m3")/a("V_center_m3")[0])*100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for phase in PHASES:
        keep = np.array([r["phase"] == phase for r in rows])
        if keep.any():
            axes[0].scatter(loss[keep], a("LEFT_sigma_local_Pa", 1e-6)[keep], s=5,
                            color=PHASE_COLORS[phase], label=phase)
            axes[1].scatter(loss[keep], a("RIGHT_sigma_local_Pa", 1e-6)[keep], s=5,
                            color=PHASE_COLORS[phase], label=phase)
    for axis, label in zip(axes, ("LEFT", "RIGHT")):
        axis.set(xlabel="Center-volume loss (%)", ylabel=f"{label} production stress (MPa)")
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("G — stress versus center-volume loss, colored by serialized phase")
    savefig(fig, out, "figure_G_stress_vs_center_volume")

    # Time-scale representations.
    for scale, name in (("linear", "timescale_full_linear"), ("symlog", "timescale_full_symlog")):
        fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        ax[0].plot(t, a("LEFT_sigma_local_Pa", 1e-6), label="LEFT")
        ax[0].plot(t, a("RIGHT_sigma_local_Pa", 1e-6), label="RIGHT")
        ax[1].plot(t, 100*a("quota_strain"), label="quota")
        ax[1].plot(t, 100*a("centroid_strain"), label="centroid")
        if scale == "symlog":
            for axis in ax: axis.set_xscale("symlog", linthresh=.02)
        for axis in ax: marks(axis); axis.legend(); axis.grid(alpha=.2)
        ax[-1].set_xlabel("Time since production source (s)")
        ax[0].set_title(f"Physical time ({scale})")
        savefig(fig, out, name)

    # Save every master-panel data view independently for reuse in a paper.
    panels = {
        "A1_production_and_integral": ("Production and integral stresses", [
            ("LEFT_sigma_local_Pa",1e-6,"LEFT production"),
            ("RIGHT_sigma_local_Pa",1e-6,"RIGHT production"),
            ("LEFT_sigma_integral_continuous_Pa",1e-6,"LEFT integral"),
            ("RIGHT_sigma_integral_continuous_Pa",1e-6,"RIGHT integral")], "MPa"),
        "A2_four_one_sided": ("Four one-sided production stresses", [
            (f"{c}_{s}_sigma_local_Pa",1e-6,label) for c,s,label in SIDE_NAMES], "MPa"),
        "A3_nonactivation_aggregates": ("Non-activation aggregate diagnostics", [
            ("center_mean_local_Pa",1e-6,"center mean"),
            ("cluster_weighted_local_Pa",1e-6,"cluster weighted"),
            ("Cannon_Carter_LEFT_Pa",1e-6,"CC LEFT"),
            ("Cannon_Carter_RIGHT_Pa",1e-6,"CC RIGHT")], "MPa"),
        "B1_grain_volumes": ("Grain volumes", [(f"V_{g}_m3",1e21,g) for g in ("left","center","right")], "10$^{-21}$ m$^3$"),
        "B2_equivalent_radii": ("Volume-equivalent radii", [(f"R_{g}_m",1e9,g) for g in ("left","center","right")], "nm"),
        "B3_axial_positions": ("Axial geometry", [(k,1e9,k) for k in ("GB_LEFT_m","GB_RIGHT_m","TJ_LEFT_m","TJ_RIGHT_m","center_span_m","outer_centroid_separation_m")], "nm"),
        "B4_contact_radii": ("Contact radii", [(f"{c}_TJ_radius_m",1e9,c) for c in ("LEFT","RIGHT")], "nm"),
        "B5_one_sided_angles": ("One-sided contact angles", [(f"{c}_{s}_theta_rad",180/np.pi,label) for c,s,label in SIDE_NAMES], "degrees"),
        "C1_strains": ("Quota and measured strains", [("quota_strain",100,"quota"),("centroid_strain",100,"centroid"),("near_axis_extent_strain_nearest_frame",100,"extent")], "%"),
        "C2_GB_TJ_displacements": ("GB and TJ displacements", [("delta_GB_LEFT_m",1e9,"GB LEFT"),("delta_GB_RIGHT_m",1e9,"GB RIGHT"),("delta_TJ_LEFT_m",1e9,"TJ LEFT"),("delta_TJ_RIGHT_m",1e9,"TJ RIGHT")], "nm"),
        "C3_outer_centroid_displacement": ("Outer-centroid separation change", [("delta_outer_centroid_separation_m",1e9,"outer-centroid spacing")], "nm"),
        "D1_hazards": ("Root and descendant hazards", [("root_H_over_Hstar_LEFT",1,"root LEFT"),("root_H_over_Hstar_RIGHT",1,"root RIGHT"),("descendant_H_over_Hstar",1,"descendant")], "H/H*"),
        "D2_source_state": ("Facilitated-source state", [("source_amplitude_h",1,"h"),("source_lifetime_remaining_s",1,"remaining lifetime")], "amplitude / s"),
        "D3_quota_affinity": ("Event quota and transport affinity", [("q_over_b",1,"q/b"),("LEFT_transport_affinity_Pa",1e-6,"LEFT affinity MPa")], "q/b or MPa"),
        "F1_center_volume": ("Center-volume evolution", [("V_center_m3",1e21,"center volume")], "10$^{-21}$ m$^3$"),
    }
    for name,(title,series,ylabel) in panels.items():
        fig,axis=plt.subplots(figsize=(10,4.8))
        for key,scale,label in series: axis.plot(t,a(key,scale),label=label,lw=1.2)
        marks(axis); axis.legend(fontsize=8,ncol=min(4,len(series)))
        axis.set(title=title,xlabel="Time since production source (s)",ylabel=ylabel)
        savefig(fig,out,"panel_"+name)
    for contact,side,label in SIDE_NAMES:
        fig,axis=plt.subplots(figsize=(10,4.8))
        for key,text_label in (("sigma_kappa_Pa",r"$-\gamma k_m$"),
                               ("sigma_TJ_Pa","TJ term"),("sigma_local_Pa","total")):
            axis.plot(t,a(f"{contact}_{side}_{key}",1e-6),label=text_label)
        marks(axis);axis.legend();axis.set(title=label+" stress decomposition",
            xlabel="Time since production source (s)",ylabel="MPa")
        savefig(fig,out,f"panel_E_{contact}_{side}_decomposition")
    for contact in ("LEFT","RIGHT"):
        fig,axis=plt.subplots(figsize=(7,5))
        y=a(f"{contact}_sigma_local_Pa",1e-6)
        for phase in PHASES:
            keep=np.array([r["phase"]==phase for r in rows])
            if keep.any(): axis.scatter(loss[keep],y[keep],s=5,color=PHASE_COLORS[phase],label=phase)
        axis.set(title=f"{contact} stress versus center-volume loss",
                 xlabel="Center-volume loss (%)",ylabel="MPa")
        axis.grid(alpha=.2);axis.legend(fontsize=5,bbox_to_anchor=(1.02,1),loc="upper left")
        savefig(fig,out,f"panel_G_{contact}_stress_vs_center_volume")
    fig,axis=plt.subplots(figsize=(11,5))
    phase_id=np.array([PHASES.index(r["phase"]) for r in rows])
    axis.step(t,phase_id,where="post",color="k")
    axis.set_yticks(range(len(PHASES)),PHASES,fontsize=7)
    axis.set(title="Serialized controller phase",xlabel="Time since production source (s)")
    axis.grid(alpha=.2);savefig(fig,out,"panel_D4_serialized_phase")
    fig,axis=plt.subplots(figsize=(10,4.8))
    dt=np.r_[np.nan,np.diff(t)]
    for grain in ("left","center","right"):
        dv=np.r_[np.nan,np.diff(a(f"V_{grain}_m3"))]
        rate=np.divide(dv,dt,out=np.full_like(dv,np.nan),where=dt>0)*1e24
        axis.plot(t,rate,label=grain,lw=.8)
    marks(axis);axis.legend();axis.set(title="Phase-wise grain volume rate",
        xlabel="Time since production source (s)",ylabel="10$^{-24}$ m$^3$/s")
    savefig(fig,out,"panel_F2_phase_volume_rates")


def plot_events(events, history_flat, out):
    fig, axes = plt.subplots(7, 2, figsize=(15, 23))
    keys = [
        ("LEFT_sigma_local_Pa", 1e-6, "LEFT stress (MPa)"),
        ("RIGHT_sigma_local_Pa", 1e-6, "RIGHT stress (MPa)"),
        ("V_center_m3", 1e21, "Center volume (10$^{-21}$ m$^3$)"),
        ("centroid_strain", 100, "Centroid strain (%)"),
        ("LEFT_TJ_radius_m", 1e9, "LEFT TJ radius (nm)"),
        ("LEFT_transport_affinity_Pa", 1e-6, "LEFT affinity (MPa)"),
        ("quota_strain", 100, "Cumulative quota strain (%)"),
    ]
    colors = plt.cm.viridis(np.linspace(0, 1, 13))
    for number, color in zip(range(1, 14), colors):
        rows = [r for r in history_flat if r["event_number"] == number and
                r["phase"] in {"ACTIVE_ONE_B", "EVENT_TRANSPORT_PAUSED",
                               "EVENT_TRANSPORT_RECOVERY", "ONE_B_COMPLETE"}]
        if not rows: continue
        q = np.array([r["q_over_b"] for r in rows])
        local_t = np.array([r["time_s"]-rows[0]["time_s"] for r in rows])
        for k, (key, scale, label) in enumerate(keys):
            v = np.array([r[key] for r in rows])*scale
            axes[k, 0].plot(q, v, color=color, lw=.8, label=str(number))
            axes[k, 1].plot(local_t, v, color=color, lw=.8, label=str(number))
            axes[k, 0].set_ylabel(label)
    for k in range(len(keys)):
        axes[k, 0].set_xlabel("q/b")
        axes[k, 1].set_xlabel("Event-local physical time (s)")
        axes[k, 1].set_xscale("symlog", linthresh=.01)
        axes[k, 0].grid(alpha=.2); axes[k, 1].grid(alpha=.2)
    axes[0, 1].legend(title="event", ncol=2, fontsize=7)
    fig.suptitle("H — all 13 events aligned by quota and event-local time")
    savefig(fig, out, "figure_H_event_aligned")
    for key, scale, label in keys:
        fig, panel = plt.subplots(1, 2, figsize=(12, 4.5))
        for number, color in zip(range(1, 14), colors):
            rows = [r for r in history_flat if r["event_number"] == number and
                    r["phase"] in {"ACTIVE_ONE_B", "EVENT_TRANSPORT_PAUSED",
                                   "EVENT_TRANSPORT_RECOVERY", "ONE_B_COMPLETE"}]
            if not rows: continue
            q=np.array([r["q_over_b"] for r in rows]); local=np.array([r["time_s"]-rows[0]["time_s"] for r in rows])
            values=np.array([r[key] for r in rows])*scale
            panel[0].plot(q,values,color=color,lw=.8,label=str(number))
            panel[1].plot(local,values,color=color,lw=.8,label=str(number))
        panel[0].set(xlabel="q/b",ylabel=label);panel[1].set(xlabel="Event-local time (s)",ylabel=label)
        panel[1].set_xscale("symlog",linthresh=.01)
        for axis in panel: axis.grid(alpha=.2)
        panel[1].legend(title="event",ncol=2,fontsize=6)
        savefig(fig,out,"panel_H_"+key)
    fig,axis=plt.subplots(figsize=(10,5))
    numbers=np.array([r["event_number"] for r in events]);width=.25
    for offset,grain,color in [(-width,"left","#3b82a0"),(0,"center","#ed8b2c"),(width,"right","#42a65a")]:
        values=np.array([r[f"delta_V_{grain}_m3"] for r in events])*1e24
        axis.bar(numbers+offset,values,width,label=grain,color=color)
    axis.axhline(0,color="k",lw=.7);axis.set_xticks(numbers)
    axis.set(title="Event-wise grain-volume redistribution",xlabel="Event number",
             ylabel="ΔV (10$^{-24}$ m$^3$)")
    axis.legend();axis.grid(axis="y",alpha=.2)
    savefig(fig,out,"panel_F3_event_volume_changes")


def render_full(field):
    return np.concatenate([field[:, ::-1], field], axis=1)


def morphology_panel(ax_total, ax_owner, fields, ownership, z, r, row, title):
    f = fields[0]
    rfull = np.r_[-r[::-1], r]
    ffull = render_full(f)
    extent = [z[0]*1e9, z[-1]*1e9, rfull[0]*1e9, rfull[-1]*1e9]
    ax_total.imshow(ffull.T, origin="lower", aspect="equal", extent=extent,
                    vmin=0, vmax=1, cmap="gray")
    ax_total.contour(z*1e9, rfull*1e9, ffull.T, levels=[.5], colors="#ef4444", linewidths=.7)
    colors = np.array([[.20, .45, .85], [.95, .60, .16], [.25, .70, .38]])
    rgb = np.einsum("kzr,kc->zrc", ownership, colors)
    rgb = np.clip(rgb, 0, 1)*np.clip(f[..., None], 0, 1)
    rgbfull = np.concatenate([rgb[:, ::-1], rgb], axis=1)
    ax_owner.imshow(np.transpose(rgbfull, (1, 0, 2)), origin="lower", aspect="equal", extent=extent)
    ax_owner.contour(z*1e9, rfull*1e9, ffull.T, levels=[.5], colors="white", linewidths=.6)
    d = row["diagnostics"]
    for contact in ("LEFT", "RIGHT"):
        c = row["contacts"][contact]
        ax_owner.axvline(c["z_TJ_m"]*1e9, color="white", lw=.7, ls=":", alpha=.8)
        ax_owner.scatter([c["z_TJ_m"]*1e9]*2,
                         [c["r_n_m"]*1e9, -c["r_n_m"]*1e9],
                         marker="x", color="white", s=18)
    for grain, color in zip(("left", "center", "right"), colors):
        ax_owner.axvline(d[f"centroid_{grain}_m"]*1e9, color=color,
                         lw=.8, ls="--", alpha=.8)
    if row.get("contact"):
        c = row["contacts"][row["contact"]]
        ax_owner.scatter([c["z_TJ_m"]*1e9], [c["r_n_m"]*1e9],
                         s=60, facecolors="none", edgecolors="yellow", linewidths=1.3)
    text = (f"{title}\nt={row['time_s']:.4f} s | A{row['avalanche_id']} E{row['event_number']} "
            f"q/b={row['q_over_b']:.3f} | {row['phase']}\n"
            f"Vc/Vc0={row['center_volume_m3']/morphology_panel.v0:.6f} | "
            f"σL/R={row['contacts']['LEFT']['sigma_local_Pa']/1e6:.2f}/"
            f"{row['contacts']['RIGHT']['sigma_local_Pa']/1e6:.2f} MPa | "
            f"εc/q={100*row['geometric_chain_strain']:.3f}/"
            f"{100*row['production_densification_strain']:.3f}%")
    ax_total.set_title(text, fontsize=8)
    for ax in (ax_total, ax_owner):
        ax.set_xlabel("z (nm)"); ax.set_ylabel("r (nm)")
    ax_owner.set_title("Ownership: LEFT blue, CENTER orange, RIGHT green", fontsize=8)


def make_movie_and_montage(run, source, history, frame_index, out):
    times = np.array([r["time_s"] for r in history])
    with np.load(source) as d:
        source_fields=d["fields"].copy(); source_ownership=d["ownership"].copy()
        z=d["z"].copy(); r=d["r_c"].copy()
    morphology_panel.v0 = history[0]["center_volume_m3"]
    mandatory = {0, len(frame_index)-1}
    retry_phases = {"EVENT_TRANSPORT_PAUSED", "EVENT_TRANSPORT_RECOVERY"}
    critical_phases = {"ROOT_CROSSING", "ONE_B_COMPLETE", "SOURCE_WINDOW_OPEN",
                       "CHILD_CROSSING", "AVALANCHE_EXTINCT_REPINNED"}
    for i in range(1, len(frame_index)):
        a, b = frame_index[i-1], frame_index[i]
        critical_boundary = (a["phase"] != b["phase"] and
                             bool({a["phase"],b["phase"]} & critical_phases))
        if critical_boundary or a["event_number"] != b["event_number"]:
            mandatory.update((i-1, i))
    for event in range(1,14):
        for phase in retry_phases:
            ids=[i for i,row in enumerate(frame_index)
                 if row["event_number"]==event and row["phase"]==phase]
            if ids: mandatory.update((ids[0],ids[-1]))
    uniform = set(np.linspace(0, len(frame_index)-1, 75).round().astype(int))
    selected = sorted(mandatory | uniform)
    # Cap while preserving all boundaries: the retained watcher already sparsified states.
    if len(selected) > 150:
        boundary = sorted(mandatory)
        extra = [i for i in selected if i not in mandatory]
        selected = sorted(boundary + extra[::max(1, math.ceil(len(extra)/(150-len(boundary))))])
    movie_items = [None, *selected]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    def draw(k):
        for ax in axes: ax.clear()
        item = movie_items[k]
        if item is None:
            fields, ownership, row, title = (
                source_fields, source_ownership, history[0], "production source")
        else:
            meta = frame_index[item]
            path = run/"frames"/Path(meta["path"]).name
            with np.load(path) as d:
                fields=d["fields"]; ownership=d["ownership"]
            hi = nearest_history_index(times, meta["time_s"])
            row, title = history[hi], f"retained frame {meta['frame']}"
        morphology_panel(axes[0], axes[1], fields, ownership, z, r, row, title)
    animation = FuncAnimation(fig, draw, frames=len(movie_items), interval=300)
    animation.save(out/"morphology_ownership_forensic.gif", writer=PillowWriter(fps=4))
    plt.close(fig)

    def near_time(value):
        return int(np.argmin(np.abs(np.array([x["time_s"] for x in frame_index])-value)))
    roots = [r for r in history if r["phase"] == "ROOT_CROSSING"]
    ext = [r for r in history if r["phase"] == "AVALANCHE_EXTINCT_REPINNED"]
    av2_pause = [i for i, x in enumerate(frame_index)
                 if x["completed_avalanches"] == 1 and x["phase"] in {
                     "EVENT_TRANSPORT_PAUSED", "EVENT_TRANSPORT_RECOVERY"}]
    pause_pick = [av2_pause[int(q*(len(av2_pause)-1))] for q in (.15, .5, .85)]
    choices = [("production source", None),
               ("pre avalanche 1", near_time(roots[0]["time_s"])),
               ("after avalanche 1", near_time(ext[0]["time_s"])),
               ("pre avalanche 2", near_time(roots[1]["time_s"]))]
    choices += [(f"avalanche 2 pause {label}", idx)
                for label, idx in zip(("early", "middle", "late"), pause_pick)]
    choices += [("after avalanche 2", near_time(ext[1]["time_s"])),
                ("final", len(frame_index)-1)]
    fig, axes = plt.subplots(len(choices), 2, figsize=(13, 4.2*len(choices)), constrained_layout=True)
    for row_axes, (title, idx) in zip(axes, choices):
        if idx is None:
            fields, ownership, hr = source_fields, source_ownership, history[0]
        else:
            meta=frame_index[idx]
            with np.load(run/"frames"/Path(meta["path"]).name) as d:
                fields=d["fields"].copy(); ownership=d["ownership"].copy()
            hr=history[nearest_history_index(times,meta["time_s"])]
        morphology_panel(row_axes[0],row_axes[1],fields,ownership,z,r,hr,title)
    savefig(fig, out, "morphology_montage")
    return movie_items, choices


def mask_figures(run, source, history, out):
    with np.load(source) as d: z=d["z"].copy(); r=d["r_c"].copy()
    for event in (1, 5, 9, 13):
        with np.load(run/f"event_{event}.npz") as d:
            fields=d["base_fields"].copy()
        start=next(x for x in history if x["event_number"]==event and
                   x["phase"]=="ACTIVE_ONE_B" and x["q_over_b"]==0)
        c=start["contacts"]["LEFT"]
        receiver, donor, _=build_current_state_masks(
            fields[0],z,r,z_TJ_m=c["z_TJ_m"],r_TJ_m=c["r_n_m"],W_m=4e-9)
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for ax,data,title,cmap in [(axes[0],receiver,"receiver near TJ","Blues"),
                                   (axes[1],donor,"donor 3W–10W","Reds")]:
            ax.imshow(data.T,origin="lower",aspect="equal",extent=[z[0]*1e9,z[-1]*1e9,r[0]*1e9,r[-1]*1e9],cmap=cmap)
            ax.contour(z*1e9,r*1e9,fields[0].T,levels=[.5],colors="k",linewidths=.6)
            ax.set(title=title,xlabel="z (nm)",ylabel="r (nm)")
        fig.suptitle(f"Event {event} LEFT current-state transfer supports")
        savefig(fig,out,f"event_{event:02d}_source_masks")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--skip-visuals",action="store_true")
    parser.add_argument("--skip-movie",action="store_true")
    parser.add_argument("--movie-only",action="store_true")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.movie_only:
        history=json.loads((args.run/"history.json").read_text())
        frame_index=json.loads((args.run/"frames/index.json").read_text())
        make_movie_and_montage(args.run,args.source,history,frame_index,args.out)
        return
    watched=[args.run/"history.json",args.run/"launch.json",args.run/"frames/index.json",args.source]
    watched += sorted(args.run.glob("event_*_final.npz"))
    source_stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in watched}
    history=json.loads((args.run/"history.json").read_text())
    launch=json.loads((args.run/"launch.json").read_text())
    manifest=json.loads(Path("docs/three_particle/production_screen/bicrystal_launch_manifest.json").read_text())
    frame_index=json.loads((args.run/"frames/index.json").read_text())
    ht=np.array([r["time_s"] for r in history])
    with np.load(args.source) as d:
        z=d["z"].copy(); r=d["r_c"].copy(); dr=float(r[1]-r[0]); dz=float(z[1]-z[0])
        sf=d["fields"].copy(); so=d["ownership"].copy(); sgb=d["gb"].copy()
    _,_,extent0=source_extent(sf[0],z)
    frame_rows=[]
    source_row=frame_record(sf,so,sgb,history[0]["time_s"],z,r,dr,dz,
                            history[0],history[0]["center_volume_m3"],extent0,-1)
    frame_rows.append(source_row)
    for meta in frame_index:
        with np.load(args.run/"frames"/Path(meta["path"]).name) as d:
            fields=d["fields"].copy(); ownership=d["ownership"].copy(); gb=d["gb"].copy()
        hi=nearest_history_index(ht,meta["time_s"])
        frame_rows.append(frame_record(fields,ownership,gb,meta["time_s"],z,r,dr,dz,
                                       history[hi],history[0]["center_volume_m3"],extent0,meta["frame"]))
    # Objective departure: maximum relative deviation among robust envelope
    # descriptors, saved alongside every retained frame.
    keys=["V_center_over_source","center_span_m","center_max_radius_m",
          "center_envelope_aspect_ratio","outer_centroid_separation_m","surface_area_proxy_m2"]
    base=frame_rows[0]
    for fr in frame_rows:
        deviations=[abs(fr[k]/base[k]-1) for k in keys if np.isfinite(fr[k]) and base[k] != 0]
        fr["morphology_departure_max_fraction"]=max(deviations) if deviations else np.nan
    history_flat,residual=flatten_history(history,frame_rows)
    events,avalanches=event_rows(history_flat,frame_rows,manifest["b_event_m"])
    pauses=pause_rows(history_flat)
    decisions=decision_rows(history,manifest)
    write_csv(args.out/"history_forensic.csv",history_flat)
    write_csv(args.out/"frame_geometry.csv",frame_rows)
    write_csv(args.out/"events.csv",events)
    write_csv(args.out/"avalanches.csv",avalanches)
    write_csv(args.out/"pause_recovery_episodes.csv",pauses)
    write_csv(args.out/"decision_audit.csv",decisions)
    if args.skip_visuals:
        from PIL import Image
        with Image.open(args.out/"morphology_ownership_forensic.gif") as movie:
            movie_frame_count=movie.n_frames
        montage_labels=["production source","pre avalanche 1","after avalanche 1",
                        "pre avalanche 2","avalanche 2 pause early",
                        "avalanche 2 pause middle","avalanche 2 pause late",
                        "after avalanche 2","final"]
    else:
        plot_history(history_flat,args.out); plot_events(events,history_flat,args.out)
        mask_figures(args.run,args.source,history,args.out)
        if args.skip_movie:
            from PIL import Image
            with Image.open(args.out/"morphology_ownership_forensic.gif") as movie:
                movie_frame_count=movie.n_frames
            montage_labels=["production source","pre avalanche 1","after avalanche 1",
                            "pre avalanche 2","avalanche 2 pause early",
                            "avalanche 2 pause middle","avalanche 2 pause late",
                            "after avalanche 2","final"]
        else:
            selected,montage=make_movie_and_montage(
                args.run,args.source,history,frame_index,args.out)
            movie_frame_count=len(selected);montage_labels=[x[0] for x in montage]
    strong=next((fr for fr in frame_rows if fr["morphology_departure_max_fraction"]>=.05),None)
    totals={}
    for aid in (1,2):
        root=next(r for r in history_flat if r["phase"]=="ROOT_CROSSING" and r["avalanche_id"]==aid)
        end=[r for r in history_flat if r["phase"]=="AVALANCHE_EXTINCT_REPINNED" and r["avalanche_id"]==aid][-1]
        segment=[r for r in history_flat if r["avalanche_id"]==aid and root["time_s"]<=r["time_s"]<=end["time_s"]]
        totals[str(aid)]=interval_totals(segment)
        totals[str(aid)]["duration_s"]=end["time_s"]-root["time_s"]
    phase_changes={}
    for phase in PHASES:
        values=dict(duration_s=0.,delta_V_left_m3=0.,delta_V_center_m3=0.,
                    delta_V_right_m3=0.,delta_centroid_strain=0.,delta_energy_J=0.)
        for a,b in zip(history_flat,history_flat[1:]):
            if b["phase"] != phase: continue
            values["duration_s"] += b["time_s"]-a["time_s"]
            for key in ("V_left_m3","V_center_m3","V_right_m3","centroid_strain","energy_J"):
                target = "delta_"+key
                values[target] += b[key]-a[key]
        phase_changes[phase]=values
    source_masks_note=(
        "Retained event checkpoints store the pre-event and accepted post-relaxation fields. "
        "They do not store the intermediate field immediately after source application, so direct "
        "source and fast-PF-relaxation volume changes are not separately identifiable post hoc.")
    summary=dict(
        label="IMMUTABLE_THREE_PARTICLE_FORENSIC_AUDIT", records=len(history),
        retained_frames=len(frame_index), completed_events=len(events),
        completed_avalanches=2, stress_decomposition_max_abs_residual_Pa=residual,
        event9_metrology=dict(legacy_change_MPa=1.0962394099565482,
                              corrected_change_MPa=.047448098666071886,
                              unchanged_limit_MPa=.1,physical_change=False),
        avalanche_time_decomposition=totals, pause_plateaus=len(pauses),
        phase_changes=phase_changes,
        initial=dict(center_equivalent_radius_nm=history_flat[0]["R_center_m"]*1e9,
                     center_span_nm=history_flat[0]["center_span_m"]*1e9,
                     left_TJ_radius_nm=history_flat[0]["LEFT_TJ_radius_m"]*1e9,
                     right_TJ_radius_nm=history_flat[0]["RIGHT_TJ_radius_m"]*1e9,
                     TJ_to_center_equivalent_radius_ratio=(
                         .5*(history_flat[0]["LEFT_TJ_radius_m"]+history_flat[0]["RIGHT_TJ_radius_m"])/history_flat[0]["R_center_m"]),
                     center_envelope_aspect_ratio=frame_rows[0]["center_envelope_aspect_ratio"]),
        final=dict(center_volume_change_fraction=(history_flat[-1]["V_center_m3"]/history_flat[0]["V_center_m3"]-1),
                   centroid_strain=history_flat[-1]["centroid_strain"],
                   quota_strain=history_flat[-1]["quota_strain"],
                   extent_strain=frame_rows[-1]["near_axis_extent_strain"],
                   mirror_error=history[-1]["mirror_error_diagnostic"]),
        event_mass_flow=dict(delta_V_left_m3=sum(e["delta_V_left_m3"] for e in events),
                             delta_V_center_m3=sum(e["delta_V_center_m3"] for e in events),
                             delta_V_right_m3=sum(e["delta_V_right_m3"] for e in events)),
        first_strong_morphology_departure=strong,
        maximum_morphology_departure=max(
            frame_rows,key=lambda row:row["morphology_departure_max_fraction"]),
        movie_frames=movie_frame_count, montage_states=montage_labels,
        source_fast_relaxation_separation=source_masks_note,
        all_left_semantics=(
            "Two root choices were competitions between independent LEFT and RIGHT clocks. "
            "Eleven descendants were serialized onto the root-selected LEFT source by construction; "
            "RIGHT was mechanically evaluated but not an eligible descendant site."),
        recommendation=4,
        recommendation_text="Both initial geometry and event mechanics need revision before another seed.",
        no_evolution=True,no_interpolation=True,no_symmetry_projection=True,
        source_hashes={str(p):sha256(p) for p in watched},
        git_head=os.popen("git rev-parse HEAD").read().strip())
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,default=float)+"\n")
    after={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in watched}
    if after != source_stats: raise RuntimeError("immutable input changed during audit")
    print(json.dumps({k:summary[k] for k in (
        "records","retained_frames","completed_events","stress_decomposition_max_abs_residual_Pa",
        "avalanche_time_decomposition","initial","final","event_mass_flow",
        "first_strong_morphology_departure","recommendation_text")},indent=2,default=float))


if __name__ == "__main__":
    main()
