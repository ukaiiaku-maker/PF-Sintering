"""RECOVERY Sections 8-15: short TJ qualification run.

Restores the M16P-era (commit a4eccc6) RBM physics UNCHANGED --
event nucleates -> advection produces rigid-body motion -> displaced/
excess mass is removed -> that mass is redistributed at/near the TJ via
`active_sink_transport_step` -> PF relaxes the morphology -- and swaps
ONLY the TJ coordinate/stress measurement feeding that operator: instead
of the OLD cross-TJ single-circle-fit method (`m16p_barrier_run.
operative_sigma`, which fits ONE circle across the TJ corner, mixing
particle-side and neighbor-side points), it now uses the SAME
branch-resolved TJ locator brought over from commit 993a101
(`pf_sintering.m16q_branch_curvature.measure_branch_resolved_sigma`),
which fits each free-surface branch SEPARATELY and never mixes them
across the physical TJ corner.

The OLD TJ is ALSO computed here, for this diagnostic run ONLY, purely as
a visual overlay -- it drives no physics (RBM/hazard/stress all use the
NEW corrected coordinate exclusively; the OLD value is never passed to
active_sink_transport_step or the hazard/kinetics calls).
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.axisym import axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, active_sink_transport_step  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402
from pf_sintering.m16k_neck_tracking import NeckTracker  # noqa: E402
from pf_sintering.m16q_branch_curvature import measure_branch_resolved_sigma  # noqa: E402
from pf_sintering.m16r_movie_rendering import draw_mirrored_geometry, draw_neck_zoom  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402
from m16p_barrier_run import _refine_zgb_subgrid, operative_sigma as operative_sigma_legacy  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
CHI = 1.5
RATIO = 0.185
W_NM = 6.0
DX_NM = 0.85

B = 2.5e-10
V0_OVER_B3 = 12.5
GS = 201.74e-9
R0 = 1e12
T = 1000.0
RANDOM_SEED = 0

N_EVENTS_TARGET = 2  # "at least one complete event, preferably into the next"
OUT_DIR = "runs/recovery_tj_qualification"
FRAMES_DIR = os.path.join(OUT_DIR, "recovery_first_event_sequence")

R_MAX_PLOT_NM = 260.0
Z_RANGE_PLOT_NM = (-140.0, 140.0)
ZOOM_HALF_R_NM = 35.0
ZOOM_HALF_Z_NM = 35.0


def operative_sigma_branch_resolved(f, particle, substrate, r_c, z, z_gb_guess, W):
    """Authoritative TJ/stress measurement for THIS recovery: branch-
    resolved (particle/neighbor fit separately, never mixed across the
    TJ corner). Returns (sigma_avg, R_of_z, z_tj, r_tj, sigma_particle,
    sigma_neighbor) -- z_tj/r_tj here are the ONE coordinate used
    everywhere downstream (RBM operator, frame marker, TJ trace)."""
    R_of_z = measure_R_of_z(f, r_c)
    res = measure_branch_resolved_sigma(f, substrate, particle, r_c, z, R_of_z, z_gb_guess, W, GAMMA_S, GAMMA_GB)
    z_tj = res["z_tj"] if np.isfinite(res["z_tj"]) else z_gb_guess
    r_tj = res["r_tj"]
    sigma_avg = res["sigma_avg"] if np.isfinite(res["sigma_avg"]) else 0.0
    return sigma_avg, R_of_z, z_tj, r_tj, res["sigma_particle"], res["sigma_neighbor"]


def save_qual_frame(png_path, f, particle, substrate, z, r_c, r_tj, z_tj, old_r_tj, old_z_tj,
                     t, sigma_avg_MPa, sigma_p_MPa, sigma_n_MPa, n_active, n_completed, cum_b, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    draw_mirrored_geometry(axes[0], f, particle, substrate, z, r_c, r_tj, z_tj,
                            R_MAX_PLOT_NM, Z_RANGE_PLOT_NM, title="full mirrored geometry")
    if np.isfinite(old_r_tj) and np.isfinite(old_z_tj):
        axes[0].plot([old_r_tj * 1e9, -old_r_tj * 1e9], [old_z_tj * 1e9, old_z_tj * 1e9],
                     marker="o", mfc="none", mec="lime", ms=7, mew=1.5, linestyle="None",
                     label="OLD TJ (diagnostic only)")
        axes[0].legend(loc="upper right", fontsize=7)
    draw_neck_zoom(axes[1], f, particle, substrate, z, r_c, r_tj, z_tj, ZOOM_HALF_R_NM, ZOOM_HALF_Z_NM)
    if np.isfinite(old_r_tj) and np.isfinite(old_z_tj):
        axes[1].plot([old_r_tj * 1e9], [old_z_tj * 1e9], marker="o", mfc="none", mec="lime", ms=9, mew=1.5)
    fig.suptitle(f"[{label}] t={t:.5f} cum_b={cum_b:.3f} sig_avg={sigma_avg_MPa:.1f}MPa "
                 f"sig_p={sigma_p_MPa:.1f} sig_n={sigma_n_MPa:.1f} N_act={n_active} N_comp={n_completed}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=105)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=CHI * R_P_NM, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    roles = roles_from_m16j_geometry(e1, e2)
    particle, substrate = roles.particle, roles.substrate
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = W_NM * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    # zero barrier for this diagnostic (Section 8: "zero barrier is fine for this diagnostic only")
    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B, D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=0.0, V0=V0_OVER_B3 * B ** 3, tau_ex0=0.0)

    dt = find_stable_dt(f, particle, substrate, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    print(f"dt={dt:.4e}", flush=True)

    sink = AxisymSink(active=True, current_disp=0.0)
    legacy_tracker = NeckTracker()

    z_gb_guess = 0.0
    n_completed = 0
    step = 0
    trace_rows = []
    frame_idx = 0

    # target: dense coverage of the FIRST event's full lifecycle
    # (pre-event, activation, 25/50/75%b, completion, post-relaxation,
    # start of next event) -- Section 9/27.
    b_checkpoints_hit = set()
    saved_frame_paths = []

    def save_frame_now(label):
        nonlocal frame_idx
        R_of_z_old = measure_R_of_z(f, r_c)
        sigma_old_MPa, _, old_z_tj, old_a, _ = operative_sigma_legacy(f, r_c, z, legacy_tracker, W)
        sigma_avg, R_of_z, z_tj, r_tj, sig_p, sig_n = operative_sigma_branch_resolved(
            f, particle, substrate, r_c, z, z_gb_guess, W)
        png_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}_{label}.png")
        save_qual_frame(png_path, f, particle, substrate, z, r_c, r_tj, z_tj, old_a, old_z_tj,
                         step * dt, sigma_avg / 1e6, sig_p / 1e6 if np.isfinite(sig_p) else float("nan"),
                         sig_n / 1e6 if np.isfinite(sig_n) else float("nan"),
                         1 if sink.active else 0, n_completed, sink.cumulative_disp / B, label)
        saved_frame_paths.append((png_path, label))
        frame_idx += 1

    save_frame_now("initial")

    while n_completed < N_EVENTS_TARGET and step < 200000:
        if step > 0:
            f, particle, substrate, _ = axisym_gb_face_projected_step(f, particle, substrate, p, Wc, dr, dz, r_c, r_f,
                                                                        dt, M_s, M_eta, W, bc_z="noflux")
            if not np.all(np.isfinite(f)):
                print(f"BLOWUP at step {step}")
                break
        t = step * dt

        sigma_avg, R_of_z, z_tj, r_tj, sig_p, sig_n = operative_sigma_branch_resolved(
            f, particle, substrate, r_c, z, z_gb_guess, W)
        if np.isfinite(z_tj):
            z_gb_guess = z_tj

        # -- diagnostic-only legacy TJ, for the overlay marker, NEVER used for physics --
        _, _, old_z_tj, old_a, _ = operative_sigma_legacy(f, r_c, z, legacy_tracker, W)

        f, particle, substrate, completed, tdiag = active_sink_transport_step(
            f, particle, substrate, sink, hp, sigma_avg, dt, dz, r_c, z, z_tj)
        cum_b_after = sink.cumulative_disp / B  # sink.cumulative_disp already accumulates across ALL events

        trace_rows.append(dict(step=step, t=t, r_tj_nm=r_tj * 1e9 if np.isfinite(r_tj) else float("nan"),
                                z_tj_nm=z_tj * 1e9 if np.isfinite(z_tj) else float("nan"),
                                old_r_tj_nm=old_a * 1e9 if np.isfinite(old_a) else float("nan"),
                                old_z_tj_nm=old_z_tj * 1e9 if np.isfinite(old_z_tj) else float("nan"),
                                cum_b=cum_b_after, sigma_particle_MPa=sig_p / 1e6 if np.isfinite(sig_p) else float("nan"),
                                sigma_neighbor_MPa=sig_n / 1e6 if np.isfinite(sig_n) else float("nan"),
                                sigma_avg_MPa=sigma_avg / 1e6))

        # high-cadence checkpoints across the event lifecycle, dense enough for a visually
        # continuous sequence (Section 9: 50-100 frames spanning the full event lifecycle)
        for frac in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
            key = (n_completed, frac)
            if key not in b_checkpoints_hit and cum_b_after - n_completed >= frac:
                b_checkpoints_hit.add(key)
                save_frame_now(f"event{n_completed}_{int(frac*100)}pct_b")

        if step in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300, 330, 340, 350, 360, 400, 450,
                    500, 550, 600, 650, 700, 750, 780, 790, 795):
            save_frame_now(f"early_step{step}")

        if completed:
            n_completed += 1
            sink.active = True
            print(f"  event #{n_completed} step={step} t={t:.4f} sigma_avg={sigma_avg/1e6:.2f}MPa "
                  f"r_tj={r_tj*1e9:.2f}nm z_tj={z_tj*1e9:.2f}nm e1e2f_resid={tdiag['e1e2f_residual']:.3e}",
                  flush=True)
            save_frame_now(f"event{n_completed}_complete")
            save_frame_now(f"event{n_completed}_post_relax")

        step += 1

    save_frame_now("final")

    with open(os.path.join(OUT_DIR, "recovery_tj_trace.csv"), "w", newline="") as fh:
        if trace_rows:
            w = csv.DictWriter(fh, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            for r in trace_rows:
                w.writerow(r)
    print(f"\nWrote {len(trace_rows)} trace rows to recovery_tj_trace.csv, {frame_idx} frames to {FRAMES_DIR}")

    # -- tj trace plot (raw, not smoothed) --
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ts = [r["t"] for r in trace_rows]
    axes[0].plot(ts, [r["r_tj_nm"] for r in trace_rows], ".-", ms=2, lw=0.7, label="r_TJ (corrected)")
    axes[0].plot(ts, [r["old_r_tj_nm"] for r in trace_rows], ".-", ms=2, lw=0.7, alpha=0.6, label="r_TJ (legacy, diagnostic)")
    axes[0].set_ylabel("r_TJ (nm)")
    axes[0].legend(fontsize=8)
    axes[1].plot(ts, [r["z_tj_nm"] for r in trace_rows], ".-", ms=2, lw=0.7, label="z_TJ (corrected)")
    axes[1].plot(ts, [r["old_z_tj_nm"] for r in trace_rows], ".-", ms=2, lw=0.7, alpha=0.6, label="z_TJ (legacy, diagnostic)")
    axes[1].set_ylabel("z_TJ (nm)")
    axes[1].set_xlabel("t")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "recovery_tj_trace.png"), dpi=110)
    plt.close(fig)

    # -- Section 14: contact sheet (12-20 representative frames, principal qualification output) --
    import matplotlib.image as mpimg
    n_sheet = min(18, len(saved_frame_paths))
    sel_idx = np.linspace(0, len(saved_frame_paths) - 1, n_sheet).astype(int)
    sel_idx = sorted(set(sel_idx.tolist()))
    ncols = 4
    nrows = math.ceil(len(sel_idx) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.atleast_2d(axes)
    for k, idx in enumerate(sel_idx):
        r, c = divmod(k, ncols)
        png_path, label = saved_frame_paths[idx]
        img = mpimg.imread(png_path)
        axes[r, c].imshow(img)
        axes[r, c].set_title(label, fontsize=7)
        axes[r, c].axis("off")
    for k in range(len(sel_idx), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig("recovery_tj_contact_sheet.png", dpi=100)
    plt.close(fig)
    print(f"Wrote recovery_tj_contact_sheet.png with {len(sel_idx)} frames")

    print(f"n_completed={n_completed} final_step={step} total_frames_saved={len(saved_frame_paths)}")


if __name__ == "__main__":
    main()
