"""Milestone 16I: production three-barrier-regime campaign driver.

Builds ONE canonical initial state (the M16G/16H-promoted crest-capped
psi=160 particle/asperity geometry), saves it once to
`<campaign_root>/initial_state.npz`, and runs THREE regimes -- off, zero,
finite -- each loading that exact saved state (verified via exact array
equality before the sim loop starts, per Section 6 of the handoff) so no
regime is exposed to a silently-different starting condition. Every
regime directory is self-contained, checkpointed, and append-only
(history.jsonl + history.csv incrementally, periodic .npz checkpoints,
periodic morphology PNG + raw-state NPZ snapshots, a running
`figures/progress.png`) -- nothing is written only at process exit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.axisym_interface_metrics import free_surface_area_of_revolution, surface_to_volume_metrics  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, calibrate_barrier, hazard_step, particle_com_z, rbm_step, update_tau_sink  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.plotting_style import REGIME_ORDER, mark_events, new_fig, regime_plot_kwargs, style_axes  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import build_particle_asperity_geometry_c3_crest, find_gb_trough, find_stable_dt, grain_volumes  # noqa: E402

CAMPAIGN_ROOT_BASE = os.path.join(os.path.dirname(__file__), "..", "runs")
W = 10e-9
HISTORY_FIELDS = [
    "step", "time", "barrier_regime", "barrier_eV", "sink_active", "n_events", "hazard",
    "a_contact_nm", "a_over_a0", "X_neck_nm", "A_GB_nm2", "z_GB_nm",
    "r_neck_1p5W_nm", "r_neck_2p5W_nm", "sigma_s_1p5W_MPa", "sigma_s_2p5W_MPa",
    "sigma_curvature_1p5W_MPa", "sigma_width_1p5W_MPa", "Q_1p5W",
    "particle_volume_nm3", "particle_volume_fraction", "total_solid_volume_nm3",
    "RBM_cumulative_displacement_nm", "particle_COM_shift_nm",
    "sintering_strain", "rbm_strain",
    "free_surface_area_nm2", "free_surface_area_per_volume_1_per_nm",
    "GB_area_per_volume_1_per_nm", "total_interface_area_per_volume_1_per_nm",
    "total_free_energy", "mass_drift",
]


# ---------------------------------------------------------------------------
# Section 6-7: canonical initial state + campaign directory
# ---------------------------------------------------------------------------

def build_canonical_state(psi_deg=160.0, R_cyl_nm=100.0, W_nm=10.0, dx_nm=1.25):
    R_cyl, Wd, dr, dz = R_cyl_nm * 1e-9, W_nm * 1e-9, dx_nm * 1e-9, dx_nm * 1e-9
    gamma_gb = psi_to_gamma_gb(psi_deg)
    gamma_s = 1.0
    p = P(gamma_s=gamma_s, gamma_gb=gamma_gb, W=Wd)
    Wc = gb_obstacle_coefficients(gamma_gb, Wd)["Wc"]
    geom = build_particle_asperity_geometry_c3_crest(R_cyl, Wd, dr, dz)
    M_s = 1e-33
    M_eta = 1e-33 / (Wd * (32.0 / 35.0))
    dt = find_stable_dt(geom["f"], geom["e1"], geom["e2"], p, Wc, dr, dz, geom["r_c"], geom["r_f"],
                         M_s, M_eta, Wd, n_check=100)
    dt *= 0.4
    return dict(geom=geom, p=p, Wc=Wc, dr=dr, dz=dz, M_s=M_s, M_eta=M_eta, dt=dt,
                gamma_s=gamma_s, gamma_gb=gamma_gb, psi_deg=psi_deg, R_cyl_nm=R_cyl_nm,
                W_nm=W_nm, dx_nm=dx_nm)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def git_info():
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=os.path.dirname(__file__), text=True).strip()
        except Exception:
            return "unknown"
    return dict(commit=_run(["git", "rev-parse", "HEAD"]), branch=_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]))


def make_campaign_root(tag=None):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = os.path.join(CAMPAIGN_ROOT_BASE, f"m16i_three_barrier_{tag or ts}")
    for sub in ("off", "zero", "finite"):
        for leaf in ("checkpoints", "snapshots", "figures"):
            os.makedirs(os.path.join(root, sub, leaf), exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Section 8-11: extended diagnostics row
# ---------------------------------------------------------------------------

def diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0, a0, com0,
                     gamma_s, gamma_gb, step, t, sink: AxisymSink, barrier_regime, barrier_eV):
    R_of_z = measure_R_of_z(f, r_c)
    z_gb, a = find_gb_trough(R_of_z, z, z_gb_prev, lam=lam)
    Vp, Vs = grain_volumes(e1, e2, r_c, dr, dz)
    V = axisym_volume(f, r_c, dr, dz)
    com = particle_com_z(e1, z, r_c, dr, dz)
    F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
    A_free = free_surface_area_of_revolution(R_of_z, z)

    X_neck = 2 * a if np.isfinite(a) else float("nan")
    A_GB = math.pi * a ** 2 if np.isfinite(a) else float("nan")
    sv = surface_to_volume_metrics(A_free, A_GB, V)

    com_shift = (com0 - com) * 1e9 if np.isfinite(com) and np.isfinite(com0) else float("nan")
    row = dict(
        step=step, time=t, barrier_regime=barrier_regime, barrier_eV=barrier_eV,
        sink_active=bool(sink.active), n_events=sink.n_events, hazard=sink.hazard,
        a_contact_nm=a * 1e9 if np.isfinite(a) else float("nan"),
        a_over_a0=a / a0 if np.isfinite(a) else float("nan"),
        X_neck_nm=X_neck * 1e9 if np.isfinite(X_neck) else float("nan"),
        A_GB_nm2=A_GB * 1e18 if np.isfinite(A_GB) else float("nan"),
        z_GB_nm=z_gb * 1e9 if np.isfinite(z_gb) else float("nan"),
        particle_volume_nm3=Vp * 1e27, particle_volume_fraction=Vp / Vp0,
        total_solid_volume_nm3=V * 1e27,
        RBM_cumulative_displacement_nm=sink.cumulative_disp * 1e9,
        particle_COM_shift_nm=com_shift,
        sintering_strain=com_shift * 1e-9 / (2 * a0) if np.isfinite(com_shift) else float("nan"),
        rbm_strain=sink.cumulative_disp / (2 * a0),
        free_surface_area_nm2=A_free * 1e18 if np.isfinite(A_free) else float("nan"),
        free_surface_area_per_volume_1_per_nm=sv["S_over_V"] * 1e-9 if np.isfinite(sv["S_over_V"]) else float("nan"),
        GB_area_per_volume_1_per_nm=sv["GB_over_V"] * 1e-9 if np.isfinite(sv["GB_over_V"]) else float("nan"),
        total_interface_area_per_volume_1_per_nm=sv["total_interface_over_V"] * 1e-9 if np.isfinite(sv["total_interface_over_V"]) else float("nan"),
        total_free_energy=F, mass_drift=(V - V0) / V0,
        r_neck_1p5W_nm=float("nan"), r_neck_2p5W_nm=float("nan"),
        sigma_s_1p5W_MPa=float("nan"), sigma_s_2p5W_MPa=float("nan"),
        # Barrier-ladder correction: explicit decomposition and Q=X_neck/
        # (C_GB*r_neck) (>1 <=> sigma_s>0, i.e. curvature term dominates
        # the contact-width term) -- primary live diagnostics for the
        # high-stress finite case (is the geometry sharpening r_neck fast
        # enough, relative to X_neck, to keep driving sigma_s upward?).
        sigma_curvature_1p5W_MPa=float("nan"), sigma_width_1p5W_MPa=float("nan"), Q_1p5W=float("nan"),
    )
    if np.isfinite(z_gb) and np.isfinite(a):
        wins = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5, 2.5))
        for w in wins:
            suffix = "1p5W" if abs(w["window_W"] - 1.5) < 1e-9 else "2p5W"
            row[f"r_neck_{suffix}_nm"] = w["r_neck"] * 1e9 if np.isfinite(w["r_neck"]) else float("nan")
            if np.isfinite(w["r_neck"]):
                sigma, sigma_curv, sigma_width, C_GB = hussein_eq1b_sigma(w["r_neck"], X_neck, gamma_s, gamma_gb)
                row[f"sigma_s_{suffix}_MPa"] = sigma / 1e6
                if suffix == "1p5W":
                    row["sigma_curvature_1p5W_MPa"] = sigma_curv / 1e6
                    row["sigma_width_1p5W_MPa"] = sigma_width / 1e6
                    if np.isfinite(C_GB) and C_GB > 0:
                        row["Q_1p5W"] = X_neck / (C_GB * w["r_neck"])
    return row, (z_gb if np.isfinite(z_gb) else z_gb_prev), R_of_z


def append_history(regime_dir, row):
    jsonl_path = os.path.join(regime_dir, "history.jsonl")
    csv_path = os.path.join(regime_dir, "history.csv")
    with open(jsonl_path, "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Section 13-15: morphology snapshots (PNG + raw NPZ)
# ---------------------------------------------------------------------------

def save_morphology_snapshot(regime_dir, tag, f, e1, e2, r_c, z, row, step):
    apply_kw = dict()
    fig, ax = new_fig(figsize=(8, 3.2))
    Z, Rg = np.meshgrid(z * 1e9, r_c * 1e9, indexing="ij")
    grain_map = np.where(f > 0.5, np.where(e1 >= e2, 1.0, -1.0), 0.0)
    ax.pcolormesh(Z, Rg, grain_map, cmap="RdBu", vmin=-1, vmax=1, shading="auto")
    R_of_z = measure_R_of_z(f, r_c)
    ax.plot(z * 1e9, R_of_z * 1e9, color="black", linewidth=1.2)
    ax.set_xlim(0, z[-1] * 1e9)
    ax.set_ylim(0, r_c[-1] * 1e9)
    style_axes(ax, xlabel="z (nm)", ylabel="r (nm)")
    txt = (f"{row['barrier_regime'].upper()}  t={row['time']:.2f}  a/a0={row['a_over_a0']:.4f}  "
           f"X_neck={row['X_neck_nm']:.2f}nm  sigma_s={row.get('sigma_s_1p5W_MPa', float('nan')):.4f}MPa  "
           f"strain={row['sintering_strain']:.4e}  n_events={row['n_events']}")
    ax.text(0.01, 1.02, txt, transform=ax.transAxes, fontsize=8.5, color="0.25", va="bottom")
    fig.savefig(os.path.join(regime_dir, "snapshots", f"morph_{tag}.png"))
    import matplotlib.pyplot as plt
    plt.close(fig)

    np.savez_compressed(os.path.join(regime_dir, "snapshots", f"state_{tag}.npz"),
                         f=f, e1=e1, e2=e2, step=step, time=row["time"],
                         sink_active=row["sink_active"], hazard=row["hazard"], n_events=row["n_events"])


# ---------------------------------------------------------------------------
# Section 16: running progress figure
# ---------------------------------------------------------------------------

def update_progress_figure(regime_dir, rows, event_times=()):
    import matplotlib.pyplot as plt
    from pf_sintering.plotting_style import apply_style
    apply_style()
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    t = [r["time"] for r in rows]
    sigma = [r.get("sigma_s_1p5W_MPa") for r in rows]
    X = [r["X_neck_nm"] for r in rows]
    strain = [r["sintering_strain"] for r in rows]
    sv = [r["free_surface_area_per_volume_1_per_nm"] for r in rows]

    axs[0, 0].plot(t, sigma, color="#B4530A", linewidth=1.4)
    mark_events(axs[0, 0], event_times)
    style_axes(axs[0, 0], "time", "sigma_s (MPa), 1.5W window")

    axs[0, 1].plot(t, X, color="#2E8B57", linewidth=1.4)
    mark_events(axs[0, 1], event_times)
    style_axes(axs[0, 1], "time", "X_neck (nm)")

    axs[1, 0].plot(t, strain, color="#4C4C9D", linewidth=1.4)
    mark_events(axs[1, 0], event_times)
    style_axes(axs[1, 0], "time", "sintering strain")

    axs[1, 1].plot(strain, sv, color="0.2", linewidth=1.4)
    style_axes(axs[1, 1], "sintering strain", "A_free/V (1/nm)")

    fig.tight_layout()
    fig.savefig(os.path.join(regime_dir, "figures", "progress.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 2-5: per-regime sink configuration
# ---------------------------------------------------------------------------

REGIME_BARRIER_EV = {"off": math.inf, "zero": 0.0, "finite": None}  # finite filled from hp.A0/eV_to_J


def make_hazard_params(GS, sigma_target_hazard=2.5e6, v0_mult=10000.0):
    hp = HazardParams(V0=v0_mult * 1e-29, GS=GS)
    return calibrate_barrier(sigma_target=sigma_target_hazard, hp=hp)


# ---------------------------------------------------------------------------
# Addendum A6-A10: live/ read-only monitoring outputs. Architecture: the
# background simulation process is the ONLY writer, always via atomic
# tmp+rename for status.json (Section A7) so a concurrent reader (Jupyter,
# the standalone CLI monitor) never observes a torn write. Cadence is
# WALL-CLOCK-gated (not PF-step-gated) per Section A8 -- expensive image
# regeneration must not scale with simulation step count.
# ---------------------------------------------------------------------------

def write_live_outputs(regime_dir, f, e1, e2, r_c, z, row, sink, regime, last_event_time, checkpoint_step):
    import matplotlib.pyplot as plt
    from pf_sintering.live_dashboard import atomic_write_json, build_status_dict, render_dashboard_figure

    live_dir = os.path.join(regime_dir, "live")
    status = build_status_dict(row, sink, regime, last_event_time, checkpoint_step, time.time())
    atomic_write_json(os.path.join(live_dir, "status.json"), status)

    # latest_morphology.png: same rendering as save_morphology_snapshot,
    # but overwritten in place (via tmp+rename) rather than accumulated
    # under snapshots/ -- this is a CONVENIENCE product (Section A6),
    # not a substitute for the permanent snapshots/ series.
    fig, ax = new_fig(figsize=(8, 3.2))
    Z, Rg = np.meshgrid(z * 1e9, r_c * 1e9, indexing="ij")
    grain_map = np.where(f > 0.5, np.where(e1 >= e2, 1.0, -1.0), 0.0)
    ax.pcolormesh(Z, Rg, grain_map, cmap="RdBu", vmin=-1, vmax=1, shading="auto")
    R_of_z = measure_R_of_z(f, r_c)
    ax.plot(z * 1e9, R_of_z * 1e9, color="black", linewidth=1.2)
    ax.set_xlim(0, z[-1] * 1e9)
    ax.set_ylim(0, r_c[-1] * 1e9)
    style_axes(ax, xlabel="z (nm)", ylabel="r (nm)")
    txt = (f"{regime.upper()}  t={row['time']:.2f}  a/a0={row['a_over_a0']:.4f}  "
           f"X_neck={row['X_neck_nm']:.2f}nm  sigma_s={row.get('sigma_s_1p5W_MPa', float('nan')):.4f}MPa  "
           f"strain={row['sintering_strain']:.4e}  events={row['n_events']}  "
           f"sink={'ACTIVE' if sink.active else 'inactive'}")
    ax.text(0.01, 1.02, txt, transform=ax.transAxes, fontsize=8.5, color="0.25", va="bottom")
    tmp_png = os.path.join(live_dir, "latest_morphology.png.tmp.png")
    final_png = os.path.join(live_dir, "latest_morphology.png")
    fig.savefig(tmp_png)
    plt.close(fig)
    os.replace(tmp_png, final_png)

    tmp_npz = os.path.join(live_dir, "latest_state.npz.tmp.npz")
    final_npz = os.path.join(live_dir, "latest_state.npz")
    np.savez_compressed(tmp_npz, f=f, e1=e1, e2=e2, step=row["step"], time=row["time"])
    os.replace(tmp_npz, final_npz)

    try:
        dash_fig = render_dashboard_figure(regime_dir)
        tmp_prog = os.path.join(live_dir, "progress.png.tmp.png")
        final_prog = os.path.join(live_dir, "progress.png")
        dash_fig.savefig(tmp_prog)
        plt.close(dash_fig)
        os.replace(tmp_prog, final_prog)
    except Exception as exc:  # dashboard rendering must never crash the sim
        print(f"  [live] progress.png render skipped: {exc}")


# ---------------------------------------------------------------------------
# Regime runner (Sections 7-15): checkpointed, append-only-history,
# periodic + event-local morphology snapshots.
# ---------------------------------------------------------------------------

def run_regime(campaign_root, regime, state0, t_target, n_history_samples=300, n_snapshots=30,
               sigma_target_hazard=2.5e6, v0_mult=10000.0, checkpoint_every_steps=20000,
               live_update_wall_s=60.0, live_update_dt=1.0, event_window_speedup=8):
    assert regime in ("off", "zero", "finite")
    regime_dir = os.path.join(campaign_root, regime)
    ckpt_path = os.path.join(regime_dir, "checkpoints", "state_checkpoint.npz")
    meta_path = os.path.join(regime_dir, "meta.json")
    events_path = os.path.join(regime_dir, "events.jsonl")
    live_dir = os.path.join(regime_dir, "live")
    os.makedirs(live_dir, exist_ok=True)

    geom, p, Wc, dr, dz = state0["geom"], state0["p"], state0["Wc"], state0["dr"], state0["dz"]
    M_s, M_eta, dt = state0["M_s"], state0["M_eta"], state0["dt"]
    gamma_s, gamma_gb = state0["gamma_s"], state0["gamma_gb"]
    z, r_c, r_f = geom["z"], geom["r_c"], geom["r_f"]
    z1, lam = geom["z1"], geom["lam"]

    diag_every_steps = max(1, int(t_target / dt / n_history_samples))
    snap_every_steps = max(1, int(t_target / dt / n_snapshots))
    sigma_update_every = 20

    if os.path.exists(ckpt_path) and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        data = np.load(ckpt_path)
        f, e1, e2 = data["f"], data["e1"], data["e2"]
        step = int(data["step"])
        a0, Vp0, Vs0, V0, com0 = meta["a0"], meta["Vp0"], meta["Vs0"], meta["V0"], meta["com0"]
        z_gb_prev = meta.get("z_gb_prev", z1)
        sink = AxisymSink(**meta["sink_state"])
        hp = HazardParams(**meta["hp_state"])
        rng = np.random.default_rng(0)
        rng.bit_generator.state = meta["rng_state"]
        pre_event_state = meta.get("pre_event_state")
        print(f"resumed [{regime}] from step={step}, t={step*dt:.4f}, n_events={sink.n_events}")
    else:
        f, e1, e2 = geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy()
        R_of_z0 = measure_R_of_z(f, r_c)
        a0 = float(np.interp(z1, z, R_of_z0))
        Vp0, Vs0 = grain_volumes(e1, e2, r_c, dr, dz)
        V0 = axisym_volume(f, r_c, dr, dz)
        com0 = particle_com_z(e1, z, r_c, dr, dz)

        hp = make_hazard_params(GS=2 * geom["R_z1"], sigma_target_hazard=sigma_target_hazard, v0_mult=v0_mult)
        rng = np.random.default_rng(0)
        sink = AxisymSink()
        if regime == "zero":
            sink.active = True  # continuous accommodation, see M16H's calibration note
        elif regime == "off":
            sink.active = False
            sink.threshold = math.inf  # never activates

        step = 0
        z_gb_prev = z1
        pre_event_state = None
        barrier_eV = 0.0 if regime == "zero" else (math.inf if regime == "off" else hp.A0 / 1.602176634e-19)
        meta = dict(a0=a0, Vp0=Vp0, Vs0=Vs0, V0=V0, com0=com0, z_gb_prev=z_gb_prev,
                    sink_state=sink.__dict__, hp_state=hp.__dict__, rng_state=rng.bit_generator.state,
                    barrier_eV=str(barrier_eV), pre_event_state=None)
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, default=str)

        barrier_eV_row = meta["barrier_eV"]
        row0, z_gb_prev, _ = diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam,
                                              Vp0, Vs0, V0, a0, com0, gamma_s, gamma_gb, 0, 0.0, sink,
                                              regime, barrier_eV_row)
        append_history(regime_dir, row0)
        save_morphology_snapshot(regime_dir, "t0000_init", f, e1, e2, r_c, z, row0, 0)
        write_live_outputs(regime_dir, f, e1, e2, r_c, z, row0, sink, regime, None, 0)  # A18: dashboard exists from step 0
        print(f"[init] [{regime}] a0={a0*1e9:.4f}nm dt={dt:.4e} barrier_eV={barrier_eV_row}")

    barrier_eV_row = meta["barrier_eV"]
    n_steps_total = max(1, int(t_target / dt))
    t_wall0 = time.time()
    last_ckpt = step
    last_diag = step
    last_snap = step
    sigma_now = 0.0
    history_rows_recent = []
    event_times = []
    if os.path.exists(events_path):
        with open(events_path) as fh:
            event_times = [json.loads(l)["t"] for l in fh if l.strip()]
    snap_counter = int(step / max(snap_every_steps, 1))
    last_event_time = event_times[-1] if event_times else None

    # Addendum A8: wall-clock-gated live-output cadence, ALSO requiring
    # at least live_update_dt of simulation time to have passed --
    # "whichever produces LESS frequent output" (Section A8) is
    # implemented as an AND of both thresholds, so a fast-stepping
    # early phase doesn't spam image regeneration and a slow-stepping
    # phase doesn't wait needlessly long past the wall-clock budget.
    last_live_wall = time.time()
    last_live_sim_t = 0.0

    # Addendum A24: event-window high-frequency diagnostics -- normal
    # cadence is diag_every_steps; during a short pre-activation lookback
    # + the sink-active window + a short post-completion relaxation tail,
    # sample event_window_speedup times more often (finite mode only).
    diag_every_normal = diag_every_steps
    event_window_until_step = -1
    EVENT_WINDOW_POST_STEPS = diag_every_normal * 2

    while step < n_steps_total:
        f, e1, e2, diag = axisym_gb_face_projected_step(
            f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W, bc_z="noflux")
        step += 1
        if not np.all(np.isfinite(f)):
            print(f"BLOWUP [{regime}] at step {step} -- stopping for review")
            break

        if step % sigma_update_every == 0:
            R_of_z = measure_R_of_z(f, r_c)
            z_gb_now, a_now = find_gb_trough(R_of_z, z, z_gb_prev, lam=lam)
            if np.isfinite(z_gb_now):
                z_gb_prev = z_gb_now
            if np.isfinite(a_now):
                X_neck = 2 * a_now
                wins = neck_curvature_windows(R_of_z, z, z_gb_prev, W, window_widths_in_W=(1.5,))
                sigma_now = 0.0
                if wins and np.isfinite(wins[0]["r_neck"]):
                    sigma_now, _, _, _ = hussein_eq1b_sigma(wins[0]["r_neck"], X_neck, gamma_s, gamma_gb)
            else:
                sigma_now = 0.0

        if regime == "off":
            pass  # sink never touched
        elif regime == "zero":
            update_tau_sink(sink, sigma_now, hp)
            f, e1, e2, completed = rbm_step(f, e1, e2, sink, hp, dt, dz, r_c, z, z_gb_prev)
            if completed:
                sink.active = True
        else:  # finite
            if not sink.active:
                # keep a rolling pre-event snapshot so the exact
                # pre-activation morphology can be written the instant
                # activation fires (Section 14B)
                pre_event_state = dict(f=f.copy(), e1=e1.copy(), e2=e2.copy(), step=step)
            activated = hazard_step(sink, sigma_now, dt, hp, rng)
            if activated:
                diag_every_steps = max(1, diag_every_normal // event_window_speedup)  # A24
                with open(events_path, "a") as fh:
                    fh.write(json.dumps(dict(step=step, t=step * dt, sigma_at_activation=sigma_now)) + "\n")
                event_times.append(step * dt)
                last_event_time = step * dt
                if pre_event_state is not None:
                    row_pre, _, _ = diagnostics_row(pre_event_state["f"], pre_event_state["e1"], pre_event_state["e2"],
                                                      p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0, a0,
                                                      com0, gamma_s, gamma_gb, pre_event_state["step"],
                                                      pre_event_state["step"] * dt, sink, regime, barrier_eV_row)
                    save_morphology_snapshot(regime_dir, f"event{len(event_times):02d}_pre", pre_event_state["f"],
                                              pre_event_state["e1"], pre_event_state["e2"], r_c, z, row_pre,
                                              pre_event_state["step"])
                row_act, _, _ = diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0,
                                                  a0, com0, gamma_s, gamma_gb, step, step * dt, sink, regime, barrier_eV_row)
                save_morphology_snapshot(regime_dir, f"event{len(event_times):02d}_activation", f, e1, e2, r_c, z,
                                          row_act, step)
                write_live_outputs(regime_dir, f, e1, e2, r_c, z, row_act, sink, regime, last_event_time, step)  # A9
                last_live_wall, last_live_sim_t = time.time(), step * dt
            f, e1, e2, completed = rbm_step(f, e1, e2, sink, hp, dt, dz, r_c, z, z_gb_prev)
            if completed:
                event_window_until_step = step + EVENT_WINDOW_POST_STEPS  # keep fast sampling through relaxation tail
                row_done, _, _ = diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0,
                                                   a0, com0, gamma_s, gamma_gb, step, step * dt, sink, regime, barrier_eV_row)
                save_morphology_snapshot(regime_dir, f"event{len(event_times):02d}_complete", f, e1, e2, r_c, z,
                                          row_done, step)
                write_live_outputs(regime_dir, f, e1, e2, r_c, z, row_done, sink, regime, last_event_time, step)  # A9
                last_live_wall, last_live_sim_t = time.time(), step * dt
            elif sink.active:
                event_window_until_step = step + EVENT_WINDOW_POST_STEPS  # still active -> keep the window open
            elif step >= event_window_until_step >= 0:
                diag_every_steps = diag_every_normal  # relaxation tail elapsed, return to normal cadence

        if step - last_diag >= diag_every_steps:
            last_diag = step
            row, z_gb_prev, _ = diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0,
                                                  a0, com0, gamma_s, gamma_gb, step, step * dt, sink, regime, barrier_eV_row)
            append_history(regime_dir, row)
            history_rows_recent.append(row)
            if len(history_rows_recent) > 2000:
                history_rows_recent = history_rows_recent[-2000:]
            print(f"  [{regime}] step={step} t={row['time']:.3f} a/a0={row['a_over_a0']:.5f} "
                  f"X_neck={row['X_neck_nm']:.3f}nm sigma_s={row.get('sigma_s_1p5W_MPa', float('nan')):.4f}MPa "
                  f"strain={row['sintering_strain']:.4e} n_events={sink.n_events} "
                  f"mass_drift={row['mass_drift']:.2e} wall={time.time()-t_wall0:.0f}s")
            if row["mass_drift"] > 0.002:
                print(f"MASS DRIFT {row['mass_drift']:.4f} EXCEEDS 0.2%% -- STOPPING FOR REVIEW")
                break
            update_progress_figure(regime_dir, history_rows_recent, event_times)

            # Addendum A8: periodic (wall-clock AND sim-time gated) live
            # update -- event-forced updates above already cover
            # activation/completion, this covers ordinary reload/relax
            # intervals so `live/progress.png` is never more than
            # ~live_update_wall_s stale even between events.
            wall_elapsed = time.time() - last_live_wall
            sim_elapsed = row["time"] - last_live_sim_t
            if wall_elapsed >= live_update_wall_s and sim_elapsed >= live_update_dt:
                write_live_outputs(regime_dir, f, e1, e2, r_c, z, row, sink, regime, last_event_time, last_ckpt)
                last_live_wall, last_live_sim_t = time.time(), row["time"]

        if step - last_snap >= snap_every_steps:
            last_snap = step
            snap_counter += 1
            row_s, _, _ = diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0, a0,
                                            com0, gamma_s, gamma_gb, step, step * dt, sink, regime, barrier_eV_row)
            save_morphology_snapshot(regime_dir, f"t{snap_counter:04d}", f, e1, e2, r_c, z, row_s, step)

        if step - last_ckpt >= checkpoint_every_steps:
            last_ckpt = step
            tmp_path = ckpt_path.replace(".npz", "_tmp.npz")
            np.savez_compressed(tmp_path, f=f, e1=e1, e2=e2, step=step)
            os.replace(tmp_path, ckpt_path)
            meta.update(z_gb_prev=z_gb_prev, sink_state=sink.__dict__, hp_state=hp.__dict__,
                        rng_state=rng.bit_generator.state)
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, default=str)

    np.savez_compressed(ckpt_path, f=f, e1=e1, e2=e2, step=step)
    meta.update(z_gb_prev=z_gb_prev, sink_state=sink.__dict__, hp_state=hp.__dict__,
                rng_state=rng.bit_generator.state)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, default=str)
    row_final, _, _ = diagnostics_row(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0, a0, com0,
                                        gamma_s, gamma_gb, step, step * dt, sink, regime, barrier_eV_row)
    save_morphology_snapshot(regime_dir, "final", f, e1, e2, r_c, z, row_final, step)
    print(f"stopped [{regime}] at step={step}, t={step*dt:.4f}, n_events={sink.n_events}")


# ---------------------------------------------------------------------------
# Section 20-21: event table + campaign manifest
# ---------------------------------------------------------------------------

def write_event_summary(campaign_root):
    regime_dir = os.path.join(campaign_root, "finite")
    events_path = os.path.join(regime_dir, "events.jsonl")
    hist_path = os.path.join(regime_dir, "history.jsonl")
    out_path = os.path.join(regime_dir, "event_summary.csv")
    if not os.path.exists(events_path):
        return
    events = [json.loads(l) for l in open(events_path) if l.strip()]
    rows = [json.loads(l) for l in open(hist_path) if l.strip()]
    fields = ["event_number", "activation_time", "sigma_activation_MPa", "a_pre_nm", "a_post_nm",
              "X_pre_nm", "X_post_nm", "sigma_pre_MPa", "sigma_post_MPa", "stress_drop_MPa",
              "RBM_displacement_nm", "sintering_strain_pre", "sintering_strain_post", "waiting_time_since_previous"]
    prev_t = 0.0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, e in enumerate(events):
            te = e["t"]
            before = max((r for r in rows if r["time"] < te), key=lambda r: r["time"], default=None)
            after = min((r for r in rows if r["time"] > te), key=lambda r: r["time"], default=None)
            if before is None or after is None:
                continue
            sp = before.get("sigma_s_1p5W_MPa", float("nan"))
            sa = after.get("sigma_s_1p5W_MPa", float("nan"))
            w.writerow(dict(
                event_number=i + 1, activation_time=te, sigma_activation_MPa=e["sigma_at_activation"] / 1e6,
                a_pre_nm=before["a_contact_nm"], a_post_nm=after["a_contact_nm"],
                X_pre_nm=before["X_neck_nm"], X_post_nm=after["X_neck_nm"],
                sigma_pre_MPa=sp, sigma_post_MPa=sa,
                stress_drop_MPa=(sp - sa) if np.isfinite(sp) and np.isfinite(sa) else float("nan"),
                RBM_displacement_nm=after["RBM_cumulative_displacement_nm"] - before["RBM_cumulative_displacement_nm"],
                sintering_strain_pre=before["sintering_strain"], sintering_strain_post=after["sintering_strain"],
                waiting_time_since_previous=te - prev_t))
            prev_t = te
    print(f"wrote {out_path}")


def write_manifest(campaign_root, state0, sigma_target_hazard, v0_mult, hp_finite):
    gi = git_info()
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(), platform=platform.platform(),
        git_commit=gi["commit"], git_branch=gi["branch"],
        psi_deg=state0["psi_deg"], R_cyl_nm=state0["R_cyl_nm"], W_nm=state0["W_nm"], dx_nm=state0["dx_nm"],
        gamma_s=state0["gamma_s"], gamma_gb=state0["gamma_gb"], dt=state0["dt"],
        M_s=state0["M_s"], M_eta=state0["M_eta"],
        sink_b=hp_finite.b, sink_Omega=hp_finite.Omega, sink_D_gb=hp_finite.D_gb, sink_r0=hp_finite.r0,
        sink_A0_J=hp_finite.A0, sink_A0_eV=hp_finite.A0 / 1.602176634e-19, sink_V0=hp_finite.V0,
        sink_tau_ex0=hp_finite.tau_ex0, sigma_target_hazard_calibration_MPa=sigma_target_hazard / 1e6,
        v0_mult=v0_mult, random_seed=0,
        # Addendum Section A27: monitoring/output configuration -- these
        # are OUTPUT parameters, not physics parameters, recorded here
        # for reproducibility of the produced artifacts, not of the
        # simulated trajectory itself.
        monitoring_config=dict(
            diagnostic_cadence_samples_over_run=300, periodic_snapshot_cadence_samples_over_run=30,
            live_update_wall_s=60.0, live_update_dt=1.0, event_window_diag_speedup=8,
            event_local_snapshot_policy="pre,activation,complete (rolling pre-event state kept until activation)",
            jupyter_refresh_default_s=60, plotting_module="pf_sintering.plotting_style + pf_sintering.live_dashboard",
        ),
        regimes=dict(
            off=dict(barrier_regime="off", barrier_eV="Infinity", activation_model="sink_disabled_infinite_barrier_limit"),
            zero=dict(barrier_regime="zero", barrier_eV=0.0, activation_model="continuous_zero_barrier_limit"),
            finite=dict(barrier_regime="finite", barrier_eV=hp_finite.A0 / 1.602176634e-19,
                        activation_model="arrhenius_first_passage_hazard"),
        ),
    )
    with open(os.path.join(campaign_root, "campaign_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote campaign_manifest.json")


# ---------------------------------------------------------------------------
# Section 17-18: final per-regime and comparison figures
# ---------------------------------------------------------------------------

def load_history(campaign_root, regime):
    path = os.path.join(campaign_root, regime, "history.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def load_events(campaign_root, regime):
    path = os.path.join(campaign_root, regime, "events.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l)["t"] for l in open(path) if l.strip()]


def final_regime_figures(campaign_root, regime):
    import matplotlib.pyplot as plt
    rows = load_history(campaign_root, regime)
    if not rows:
        return
    events = load_events(campaign_root, regime)
    fig_dir = os.path.join(campaign_root, regime, "figures")
    t = [r["time"] for r in rows]

    specs = [("sigma_s_t", "sigma_s_1p5W_MPa", "sigma_s (MPa)"),
             ("X_neck_t", "X_neck_nm", "X_neck (nm)"),
             ("a_over_a0_t", "a_over_a0", "a/a0"),
             ("sintering_strain_t", "sintering_strain", "sintering strain")]
    for name, field, ylabel in specs:
        fig, ax = new_fig()
        ax.plot(t, [r.get(field) for r in rows], **regime_plot_kwargs(regime))
        if regime == "finite":
            mark_events(ax, events)
        style_axes(ax, "time", ylabel)
        fig.savefig(os.path.join(fig_dir, f"{name}.png"))
        plt.close(fig)

    fig, ax = new_fig()
    ax.plot(t, [r.get("free_surface_area_per_volume_1_per_nm") for r in rows], **regime_plot_kwargs(regime))
    style_axes(ax, "time", "A_free/V (1/nm)")
    fig.savefig(os.path.join(fig_dir, "S_over_V_t.png"))
    plt.close(fig)

    fig, ax = new_fig()
    ax.plot([r["sintering_strain"] for r in rows], [r.get("free_surface_area_per_volume_1_per_nm") for r in rows],
            **regime_plot_kwargs(regime))
    style_axes(ax, "sintering strain", "A_free/V (1/nm)")
    fig.savefig(os.path.join(fig_dir, "S_over_V_vs_strain.png"))
    plt.close(fig)

    if regime == "finite" and events:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs[0].plot(t, [r.get("sigma_s_1p5W_MPa") for r in rows], color="#B4530A", linewidth=1.4)
        mark_events(axs[0], events)
        style_axes(axs[0], None, "sigma_s (MPa)")
        axs[1].plot(t, [r["X_neck_nm"] for r in rows], color="#2E8B57", linewidth=1.4)
        mark_events(axs[1], events)
        style_axes(axs[1], "time", "X_neck (nm)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "sawtooth_sigma_Xneck.png"))
        plt.close(fig)

        fig, ax = new_fig()
        ax.plot([r["X_neck_nm"] for r in rows], [r.get("sigma_s_1p5W_MPa") for r in rows],
                color="#8A2BE2", linewidth=0.9, alpha=0.8)
        style_axes(ax, "X_neck (nm)", "sigma_s (MPa)")
        fig.savefig(os.path.join(fig_dir, "phase_portrait_sigma_vs_Xneck.png"))
        plt.close(fig)


def comparison_figures(campaign_root):
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(campaign_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    data = {r: load_history(campaign_root, r) for r in REGIME_ORDER}
    events_finite = load_events(campaign_root, "finite")

    specs = [("compare_sigma_s_t", "sigma_s_1p5W_MPa", "sigma_s (MPa)"),
             ("compare_X_neck_t", "X_neck_nm", "X_neck (nm)"),
             ("compare_sintering_strain_t", "sintering_strain", "sintering strain")]
    for name, field, ylabel in specs:
        fig, ax = new_fig()
        for regime in REGIME_ORDER:
            rows = data[regime]
            if not rows:
                continue
            ax.plot([r["time"] for r in rows], [r.get(field) for r in rows], **regime_plot_kwargs(regime))
        style_axes(ax, "time", ylabel)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(fig_dir, f"{name}.png"))
        plt.close(fig)

    fig, ax = new_fig()
    for regime in REGIME_ORDER:
        rows = data[regime]
        if not rows:
            continue
        ax.plot([r["sintering_strain"] for r in rows], [r.get("free_surface_area_per_volume_1_per_nm") for r in rows],
                **regime_plot_kwargs(regime))
    style_axes(ax, "sintering strain", "A_free/V (1/nm)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(fig_dir, "compare_S_over_V_vs_strain.png"))
    plt.close(fig)

    # morphology montage: beginning/middle/end for each regime
    fig, axs = plt.subplots(3, 3, figsize=(13, 8))
    for row_i, regime in enumerate(REGIME_ORDER):
        snap_dir = os.path.join(campaign_root, regime, "snapshots")
        pngs = sorted(f for f in os.listdir(snap_dir) if f.startswith("morph_t") and f.endswith(".png"))
        if not pngs:
            continue
        picks = [pngs[0], pngs[len(pngs) // 2], pngs[-1]]
        for col_i, fn in enumerate(picks):
            img = plt.imread(os.path.join(snap_dir, fn))
            axs[row_i, col_i].imshow(img)
            axs[row_i, col_i].axis("off")
            if col_i == 0:
                axs[row_i, col_i].set_ylabel(regime)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "morphology_montage.png"))
    plt.close(fig)


def postprocess(campaign_root):
    for regime in REGIME_ORDER:
        final_regime_figures(campaign_root, regime)
    write_event_summary(campaign_root)
    comparison_figures(campaign_root)
    print("postprocessing complete")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", type=str, required=True)
    ap.add_argument("--regime", type=str, choices=["off", "zero", "finite", "init", "postprocess"], required=True)
    ap.add_argument("--t-target", type=float, default=250.0)
    ap.add_argument("--sigma-target-hazard-MPa", type=float, default=2.5)
    ap.add_argument("--v0-mult", type=float, default=10000.0)
    ap.add_argument("--live-update-wall-s", type=float, default=60.0)
    ap.add_argument("--live-update-dt", type=float, default=1.0)
    ap.add_argument("--n-snapshots", type=int, default=30)
    args = ap.parse_args()

    root = args.campaign_root
    if args.regime == "init":
        os.makedirs(root, exist_ok=True)
        for sub in ("off", "zero", "finite"):
            for leaf in ("checkpoints", "snapshots", "figures"):
                os.makedirs(os.path.join(root, sub, leaf), exist_ok=True)
        state0 = build_canonical_state()
        geom = state0["geom"]
        init_path = os.path.join(root, "initial_state.npz")
        np.savez_compressed(init_path, f=geom["f"], e1=geom["e1"], e2=geom["e2"], z=geom["z"])
        h = array_hash(geom["f"], geom["e1"], geom["e2"], geom["z"])
        print(f"initial_state.npz written, hash={h}")
        hp_finite = make_hazard_params(GS=2 * geom["R_z1"], sigma_target_hazard=args.sigma_target_hazard_MPa * 1e6,
                                        v0_mult=args.v0_mult)
        write_manifest(root, state0, args.sigma_target_hazard_MPa * 1e6, args.v0_mult, hp_finite)
    elif args.regime == "postprocess":
        postprocess(root)
    else:
        state0 = build_canonical_state()
        geom = state0["geom"]
        init_path = os.path.join(root, "initial_state.npz")
        if os.path.exists(init_path):
            saved = np.load(init_path)
            h_now = array_hash(geom["f"], geom["e1"], geom["e2"], geom["z"])
            h_saved = array_hash(saved["f"], saved["e1"], saved["e2"], saved["z"])
            if h_now != h_saved:
                raise RuntimeError(f"initial state hash mismatch: freshly-built={h_now} saved={h_saved} -- "
                                    f"refusing to run regime {args.regime!r} against a different starting state")
            print(f"initial state hash verified: {h_now}")
        else:
            raise RuntimeError(f"{init_path} missing -- run --regime init first")
        run_regime(root, args.regime, state0, args.t_target, n_snapshots=args.n_snapshots,
                   sigma_target_hazard=args.sigma_target_hazard_MPa * 1e6, v0_mult=args.v0_mult,
                   live_update_wall_s=args.live_update_wall_s, live_update_dt=args.live_update_dt)
