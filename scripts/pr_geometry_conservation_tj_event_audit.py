#!/usr/bin/env python3
"""Field/contour conservation and tracker audit for the frozen root-65 run.

This script is deliberately post-processing only.  It reads immutable production
checkpoints and the movie contour archive and writes a separate diagnostic tree.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
import sys

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_final_balanced as balanced  # noqa: E402
from pf_sintering.axisym import axisym_volume  # noqa: E402
from pf_sintering.continuous_field_tj import (  # noqa: E402
    ContinuousFieldTJTracker, field_intersection_candidates,
)


SOURCE = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_five_eps0p25_Dgb_over_Ds0p5_root65_"
    "continuous_surface_bounded_3W")
OUT = ROOT / "runs/pr_geometry_conservation_tj_event_audit_20260826"
CHECKPOINTS = SOURCE / "checkpoints"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def outer_radius(f: np.ndarray, r_c: np.ndarray, level: float = 0.5) -> np.ndarray:
    radius = np.full(f.shape[0], np.nan)
    for j, row in enumerate(f):
        crossings = np.flatnonzero((row[:-1] - level) * (row[1:] - level) <= 0)
        usable = [int(i) for i in crossings if row[i + 1] != row[i]
                  and min(row[i], row[i + 1]) <= level <= max(row[i], row[i + 1])]
        if usable:
            i = usable[-1]
            a = (level - row[i]) / (row[i + 1] - row[i])
            radius[j] = r_c[i] + a * (r_c[i + 1] - r_c[i])
    return radius


def contour_volume(z: np.ndarray, radius: np.ndarray) -> float:
    finite = np.isfinite(radius)
    zz, rr = z[finite], radius[finite]
    if len(zz) < 2:
        return math.nan
    return float(math.pi * np.trapezoid(rr * rr, zz))


def packed_contour_volume(zn, rn, zp, rp) -> float:
    z = np.r_[np.asarray(zn)[::-1], np.asarray(zp)[1:]]
    r = np.r_[np.asarray(rn)[::-1], np.asarray(rp)[1:]]
    order = np.argsort(z)
    return float(math.pi * np.trapezoid(r[order] ** 2, z[order]))


def weighted_centroid(field, z, r_c) -> float:
    w = np.maximum(np.asarray(field), 0.0) * r_c[None, :]
    return float(np.sum(w * z[:, None]) / np.sum(w))


def contour_tj(f, particle, substrate, z, r_c, previous_z) -> tuple[float, float, int]:
    candidates = field_intersection_candidates(f, particle, substrate, z, r_c)
    if not candidates:
        return math.nan, math.nan, 0
    chosen = min(candidates, key=lambda p: abs(p[0] - previous_z))
    return float(chosen[0]), float(chosen[1]), len(candidates)


def neck_markers(radius, z, z_tj, W):
    finite = np.isfinite(radius)
    use = finite & (np.abs(z - z_tj) <= 6.0 * W)
    idx = np.flatnonzero(use)
    if len(idx) < 7:
        return dict(z_neck_raw_m=math.nan, r_neck_raw_m=math.nan,
                    z_neck_smooth_m=math.nan, r_neck_smooth_m=math.nan)
    raw_i = int(idx[np.argmin(radius[idx])])
    smoothed = radius.copy()
    values = radius[idx]
    smoothed[idx] = gaussian_filter1d(values, sigma=1.5, mode="nearest")
    smooth_i = int(idx[np.argmin(smoothed[idx])])
    return dict(
        z_neck_raw_m=float(z[raw_i]), r_neck_raw_m=float(radius[raw_i]),
        z_neck_smooth_m=float(z[smooth_i]),
        r_neck_smooth_m=float(smoothed[smooth_i]))


def load_state(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "f" in data:
            state = tuple(np.asarray(data[k]) for k in ("f", "particle", "substrate"))
        else:
            state = tuple(np.asarray(data[k]) for k in
                          ("current_f", "current_particle", "current_neighbor"))
        scalars = {k: np.asarray(data[k]).item() for k in data.files
                   if np.asarray(data[k]).shape == () and np.asarray(data[k]).dtype.kind != "U"}
    return state, scalars


def checkpoint_selection() -> list[Path]:
    names = ["initial_state.npz", "avalanche1_before_root.npz"]
    names += [f"avalanche1_event1_q{x}b.npz" for x in
              ("0.10", "0.25", "0.50", "0.75", "1.00")]
    names += ["avalanche1_event1_complete.npz",
              "avalanche1_event26_complete.npz", "avalanche1_extinct.npz",
              "avalanche2_validity_censored.npz"]
    return [CHECKPOINTS / n for n in names if (CHECKPOINTS / n).exists()]


def event_time_map() -> dict[int, tuple[float, float]]:
    result = {}
    with (SOURCE / "one_b_subevents.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["avalanche_id"]) == 1:
                result[int(row["event_number"])] = (
                    float(row["start_time_model"]), float(row["end_time_model"]))
    return result


def checkpoint_time(path: Path, scalars, events) -> float:
    name = path.name
    if "t_model" in scalars:
        return float(scalars["t_model"])
    match = re.search(r"event(\d+)_q", name)
    if match:
        start = events[int(match.group(1))][0]
        return start + float(scalars["event_time_model"])
    return math.nan


def audit_fields(setup, geom):
    events = event_time_map()
    tracker = ContinuousFieldTJTracker(
        setup["z"], setup["r_c"], geom["z1"], 200.0 * max(setup["dr"], setup["dz"]))
    rows, states = [], []
    previous_contour_z = geom["z1"]
    for path in checkpoint_selection():
        state, scalars = load_state(path)
        f, p, s = state
        t = checkpoint_time(path, scalars, events)
        radius = outer_radius(f, setup["r_c"])
        node = tracker.locate(f, p, s)
        zc, rc, count = contour_tj(f, p, s, setup["z"], setup["r_c"], previous_contour_z)
        previous_contour_z = zc if math.isfinite(zc) else previous_contour_z
        closure = p + s - f
        finite = np.isfinite(radius)
        max_r = float(np.nanmax(radius))
        z_cap = float(np.max(setup["z"][finite]))
        w = setup["r_c"][None, :]
        row = dict(
            label=path.stem, t_model=t,
            M_f_axisym_m3=axisym_volume(f, setup["r_c"], setup["dr"], setup["dz"]),
            M_eta_axisym_m3=axisym_volume(p + s, setup["r_c"], setup["dr"], setup["dz"]),
            V_particle_m3=axisym_volume(p, setup["r_c"], setup["dr"], setup["dz"]),
            V_substrate_m3=axisym_volume(s, setup["r_c"], setup["dr"], setup["dz"]),
            contour_volume_m3=contour_volume(setup["z"], radius),
            closure_max=float(np.max(np.abs(closure))),
            closure_rms=float(np.sqrt(np.mean(closure * closure))),
            closure_axisym_rms=float(np.sqrt(np.sum(w * closure * closure) / np.sum(w) / f.shape[0])),
            f_min=float(np.min(f)), f_max=float(np.max(f)),
            z_TJ_field_m=node["z_TJ_m"], r_TJ_field_m=node["r_TJ_m"],
            field_TJ_jump_m=node["jump_m"], z_TJ_contour_m=zc,
            r_TJ_contour_m=rc, TJ_candidate_count=count,
            radial_margin_m=float(setup["r_f"][-1] - max_r),
            positive_z_margin_m=float((setup["z"][-1] + 0.5 * setup["dz"]) - z_cap),
            radial_boundary_f_max=float(np.max(f[:, -1])),
            positive_boundary_f_max=float(np.max(f[-1, :])),
            negative_boundary_f_max=float(np.max(f[0, :])),
            z_centroid_particle_m=weighted_centroid(p, setup["z"], setup["r_c"]),
            z_centroid_substrate_m=weighted_centroid(s, setup["z"], setup["r_c"]),
            **neck_markers(radius, setup["z"], node["z_TJ_m"], setup["W"]))
        rows.append(row)
        states.append((row, state, radius))
    for key in ("M_f_axisym_m3", "M_eta_axisym_m3", "V_particle_m3",
                "V_substrate_m3", "contour_volume_m3"):
        base = rows[0][key]
        for row in rows:
            row[key.replace("_m3", "_relative")] = (row[key] - base) / base
    return rows, states


def audit_movie():
    path = SOURCE / "avalanche_renewal_movie_geometry.h5"
    rows = []
    with h5py.File(path) as h:
        n = len(h["time/t_model"])
        arrays = {name: np.asarray(h[name]) for name in (
            "time/t_model", "state/avalanche_id", "state/event_number",
            "state/q_event_over_b", "state/sink_state", "state/z_TJ_m",
            "state/r_TJ_m", "state/sigma_local_Pa", "state/sigma_integral_Pa")}
        zn = h["geometry/z_negative_m"]
        rn = h["geometry/r_negative_m"]
        zp = h["geometry/z_positive_m"]
        rp = h["geometry/r_positive_m"]
        previous = None
        for i in range(n):
            zni, rni, zpi, rpi = map(lambda x: np.asarray(x, dtype=float),
                                     (zn[i], rn[i], zp[i], rp[i]))
            # Each archived branch is resampled from the TJ outward.
            candidates = []
            for zb, rb in ((zni, rni), (zpi, rpi)):
                s_branch = np.r_[0.0, np.cumsum(np.hypot(np.diff(zb), np.diff(rb)))]
                use = np.flatnonzero(s_branch <= 6.0e-8)
                if len(use):
                    smooth_branch = gaussian_filter1d(rb, 1.5, mode="nearest")
                    raw_i = int(use[np.argmin(rb[use])])
                    smooth_i = int(use[np.argmin(smooth_branch[use])])
                    candidates.append((zb, rb, smooth_branch, raw_i, smooth_i))
            raw_choice = min(candidates, key=lambda x: x[1][x[3]])
            smooth_choice = min(candidates, key=lambda x: x[2][x[4]])
            tj = np.array([arrays["state/z_TJ_m"][i], arrays["state/r_TJ_m"][i]])
            jump = 0.0 if previous is None else float(np.linalg.norm(tj - previous))
            previous = tj
            rows.append(dict(
                frame_id=i, t_model=float(arrays["time/t_model"][i]),
                avalanche_id=int(arrays["state/avalanche_id"][i]),
                event_number=int(arrays["state/event_number"][i]),
                q_event_over_b=float(arrays["state/q_event_over_b"][i]),
                sink_state=int(arrays["state/sink_state"][i]),
                z_TJ_m=float(tj[0]), r_TJ_m=float(tj[1]), TJ_jump_m=jump,
                z_neck_raw_m=float(raw_choice[0][raw_choice[3]]),
                r_neck_raw_m=float(raw_choice[1][raw_choice[3]]),
                z_neck_smooth_m=float(smooth_choice[0][smooth_choice[4]]),
                r_neck_smooth_m=float(smooth_choice[2][smooth_choice[4]]),
                contour_volume_m3=packed_contour_volume(zni, rni, zpi, rpi),
                radial_margin_m=float(183.425e-9 - max(np.max(rni), np.max(rpi))),
                positive_z_margin_m=float(1282.806e-9 - np.max(zpi)),
                sigma_local_Pa=float(arrays["state/sigma_local_Pa"][i]),
                sigma_integral_Pa=float(arrays["state/sigma_integral_Pa"][i])))
    base = rows[0]["contour_volume_m3"]
    for row in rows:
        row["contour_volume_relative"] = (row["contour_volume_m3"] - base) / base
    return rows


def plot_audits(field_rows, movie_rows):
    tf = np.array([r["t_model"] for r in field_rows])
    tm = np.array([r["t_model"] for r in movie_rows])
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for key, label in (("M_f_axisym_relative", "field f"),
                       ("M_eta_axisym_relative", "eta sum"),
                       ("contour_volume_relative", "checkpoint contour")):
        ax[0].plot(tf, [r[key] for r in field_rows], "o-", label=label)
    ax[0].plot(tm, [r["contour_volume_relative"] for r in movie_rows], lw=1,
               label="movie contour")
    ax[0].set_ylabel("relative volume change"); ax[0].legend(ncol=2)
    ax[1].semilogy(tf, [max(r["closure_max"], 1e-20) for r in field_rows], "o-", label="max")
    ax[1].semilogy(tf, [max(r["closure_rms"], 1e-20) for r in field_rows], "o-", label="RMS")
    ax[1].set_ylabel("|e1+e2-f|"); ax[1].legend()
    ax[2].plot(tf, np.array([r["V_particle_m3"] for r in field_rows]) / field_rows[0]["V_particle_m3"], "o-", label="particle")
    ax[2].plot(tf, np.array([r["V_substrate_m3"] for r in field_rows]) / field_rows[0]["V_substrate_m3"], "o-", label="substrate")
    ax[2].set_ylabel("grain volume / initial"); ax[2].set_xlabel("model time"); ax[2].legend()
    fig.tight_layout(); fig.savefig(OUT / "volume_audit.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(tm, np.array([r["radial_margin_m"] for r in movie_rows]) * 1e9, label="radial")
    ax[0].plot(tm, np.array([r["positive_z_margin_m"] for r in movie_rows]) * 1e9, label="+z")
    ax[0].axhline(30, color="k", ls="--", lw=1, label="3W")
    ax[0].set_ylabel("interface margin (nm)"); ax[0].legend()
    ax[1].semilogy(tf, np.maximum([r["radial_boundary_f_max"] for r in field_rows], 1e-20), "o-", label="radial f")
    ax[1].semilogy(tf, np.maximum([r["positive_boundary_f_max"] for r in field_rows], 1e-20), "o-", label="+z f")
    ax[1].set_ylabel("boundary max f"); ax[1].set_xlabel("model time"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "boundary_margin_audit.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(tm, np.array([r["z_TJ_m"] for r in movie_rows]) * 1e9, label="field TJ")
    ax[0].plot(tm, np.array([r["z_neck_raw_m"] for r in movie_rows]) * 1e9, label="raw minimum")
    ax[0].plot(tm, np.array([r["z_neck_smooth_m"] for r in movie_rows]) * 1e9, label="smooth minimum")
    ax[0].set_ylabel("z (nm)"); ax[0].legend(ncol=3)
    ax[1].plot(tm, np.array([r["r_TJ_m"] for r in movie_rows]) * 1e9, label="field TJ")
    ax[1].plot(tm, np.array([r["r_neck_raw_m"] for r in movie_rows]) * 1e9, label="raw minimum")
    ax[1].plot(tm, np.array([r["r_neck_smooth_m"] for r in movie_rows]) * 1e9, label="smooth minimum")
    ax[1].set_ylabel("r (nm)"); ax[1].legend(ncol=3)
    ax[2].semilogy(tm, np.maximum(np.array([r["TJ_jump_m"] for r in movie_rows]) * 1e9, 1e-6))
    ax[2].axhline(1.25, color="r", ls="--", label="1 cell")
    ax[2].set_ylabel("TJ jump (nm)"); ax[2].set_xlabel("model time"); ax[2].legend()
    fig.tight_layout(); fig.savefig(OUT / "tj_neck_tracking_audit.png", dpi=180); plt.close(fig)


def render_field_gif(states, setup):
    frames = []
    z_nm = setup["z"] * 1e9; r_nm = setup["r_c"] * 1e9
    extent = [z_nm[0], z_nm[-1], r_nm[0], r_nm[-1]]
    for row, state, radius in states:
        f, p, s = state
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
        for a, field, title, cmap, limits in (
                (axes[0, 0], f, "conserved field f", "viridis", (0, 1)),
                (axes[0, 1], p, "particle e1", "magma", (0, 1)),
                (axes[1, 0], s, "substrate e2", "magma", (0, 1)),
                (axes[1, 1], p-s, "ownership e1-e2", "coolwarm", (-1, 1))):
            im = a.imshow(field.T, origin="lower", extent=extent, aspect="equal",
                          cmap=cmap, vmin=limits[0], vmax=limits[1])
            a.plot(z_nm, radius * 1e9, "w-", lw=0.8)
            a.plot(row["z_TJ_field_m"] * 1e9, row["r_TJ_field_m"] * 1e9,
                   "co", ms=4, label="field TJ")
            a.plot(row["z_neck_smooth_m"] * 1e9, row["r_neck_smooth_m"] * 1e9,
                   "yx", ms=5, label="smooth neck")
            a.axvspan((row["z_TJ_field_m"] - 3*setup["W"]) * 1e9,
                      (row["z_TJ_field_m"] + 3*setup["W"]) * 1e9,
                      color="white", alpha=0.06)
            a.set_title(title); a.set_xlabel("z (nm)"); a.set_ylabel("r (nm)")
            fig.colorbar(im, ax=a, fraction=0.025)
        fig.suptitle(f"{row['label']}   t={row['t_model']:.6g}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(OUT / "field_diagnostic.gif", frames, duration=0.7, loop=0)


def render_contour_overlay_gif(movie_rows):
    path = SOURCE / "avalanche_renewal_movie_geometry.h5"
    count = len(movie_rows)
    indices = np.unique(np.r_[np.linspace(0, count-1, 55, dtype=int),
                              np.flatnonzero([r["sink_state"] for r in movie_rows])[[0, -1]]])
    frames=[]
    with h5py.File(path) as h:
        for i in indices:
            row=movie_rows[int(i)]
            zn=np.asarray(h["geometry/z_negative_m"][i])*1e9
            rn=np.asarray(h["geometry/r_negative_m"][i])*1e9
            zp=np.asarray(h["geometry/z_positive_m"][i])*1e9
            rp=np.asarray(h["geometry/r_positive_m"][i])*1e9
            fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
            ax.plot(zn,rn,"tab:blue",lw=1.5,label="substrate branch")
            ax.plot(zp,rp,"tab:orange",lw=1.5,label="particle branch")
            ax.fill_between(np.r_[zn[::-1],zp[1:]],0,np.r_[rn[::-1],rp[1:]],
                            color="0.8",alpha=.35)
            ax.plot(row["z_TJ_m"]*1e9,row["r_TJ_m"]*1e9,"ko",ms=5,label="field TJ")
            ax.plot(row["z_neck_smooth_m"]*1e9,row["r_neck_smooth_m"]*1e9,
                    "rx",ms=6,label="smooth local neck")
            ax.axvspan((row["z_TJ_m"]-3e-8)*1e9,(row["z_TJ_m"]+3e-8)*1e9,
                       color="tab:green",alpha=.08,label="3W fit span")
            ax.set_xlim(0,1283);ax.set_ylim(0,183);ax.set_aspect("equal")
            ax.set_xlabel("z (nm)");ax.set_ylabel("r (nm)")
            ax.set_title(f"t={row['t_model']:.3f}  event={row['event_number']}  "
                         f"q/b={row['q_event_over_b']:.2f}  "
                         f"local={row['sigma_local_Pa']/1e6:.1f} MPa")
            ax.legend(loc="upper right",ncol=2,fontsize=8)
            fig.canvas.draw();frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy());plt.close(fig)
    imageio.mimsave(OUT/"contour_tracker_overlay.gif",frames,duration=.18,loop=0)


def plot_replay_local_profiles(setup, geom):
    replay = OUT / "one_b_replay/one_b_replay_fields.h5"
    if not replay.exists():
        return
    with h5py.File(replay) as h:
        q = np.asarray(h["q_over_b"])
        chosen = [int(np.argmin(np.abs(q-value))) for value in
                  (0.0, 0.25, 0.50, 0.75, 1.0)]
        fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
        base = outer_radius(np.asarray(h["f"][chosen[0]]), setup["r_c"])
        for index in chosen:
            radius = outer_radius(np.asarray(h["f"][index]), setup["r_c"])
            x = (setup["z"] - geom["z1"]) / setup["W"]
            use = np.isfinite(radius) & (np.abs(x) <= 6.0)
            ax[0].plot(x[use], radius[use]*1e9, label=f"q/b={q[index]:.2f}")
            ax[1].plot(x[use], (radius[use]-base[use])*1e9,
                       label=f"q/b={q[index]:.2f}")
        for a in ax:
            a.axvline(0, color="k", ls="--", lw=1, label="initial GB/TJ z")
            a.grid(alpha=.2); a.legend(ncol=3)
        ax[0].set_ylabel("f=0.5 radius (nm)")
        ax[1].set_ylabel("radius change (nm)"); ax[1].set_xlabel("(z-zTJ,0)/W")
        fig.tight_layout(); fig.savefig(OUT / "one_b_replay/local_TJ_profile_evolution.png", dpi=180)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geom, setup = balanced.build_balanced_case()
    field_rows, states = audit_fields(setup, geom)
    movie_rows = audit_movie()
    write_csv(OUT / "field_checkpoint_audit.csv", field_rows)
    write_csv(OUT / "movie_contour_audit.csv", movie_rows)
    plot_audits(field_rows, movie_rows)
    render_field_gif(states, setup)
    render_contour_overlay_gif(movie_rows)
    plot_replay_local_profiles(setup, geom)
    summary = dict(
        source=str(SOURCE), field_checkpoint_count=len(field_rows),
        contour_frame_count=len(movie_rows),
        field_mass_relative_change=(field_rows[-1]["M_f_axisym_m3"] /
                                    field_rows[0]["M_f_axisym_m3"] - 1.0),
        contour_volume_relative_change=(movie_rows[-1]["contour_volume_m3"] /
                                        movie_rows[0]["contour_volume_m3"] - 1.0),
        maximum_closure_error=max(r["closure_max"] for r in field_rows),
        minimum_radial_margin_m=min(r["radial_margin_m"] for r in movie_rows),
        minimum_positive_z_margin_m=min(r["positive_z_margin_m"] for r in movie_rows),
        maximum_TJ_jump_m=max(r["TJ_jump_m"] for r in movie_rows))
    (OUT / "archive_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
