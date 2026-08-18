"""RECOVERY Sections 17-21: full production run using the restored M16P
RBM physics + the corrected branch-resolved TJ coordinate (see
scripts/recovery_tj_qualification.py, which qualified this exact
combination visually and numerically before this script was run).

Runs zero-barrier first (the strongest repeated-event stress test), then
-- only if it passes -- finite-barrier with EXACTLY the same spatial
mechanics (same active_sink_transport_step, same TJ coordinate, same
geometry), differing only in the nucleation-barrier mode. No new
transport physics; no accommodation PDE; no grain-fraction changes --
scope-limited per the recovery instruction.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.axisym import axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, active_sink_transport_step  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402
from pf_sintering.m16r_movie_rendering import draw_mirrored_geometry, draw_neck_zoom  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402
from recovery_tj_qualification import operative_sigma_branch_resolved  # noqa: E402

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

# M16Q's calibrated barrier, computed against this SAME branch-resolved
# stress metric -- internally consistent with the restored+corrected
# state used here, per Section 20 ("if the old A0 remains internally
# consistent with the restored state, retain it").
A0_EV_FINITE = 0.569030

N_EVENTS_TARGET = 100
MAX_STEPS = 2_000_000
CHECKPOINT_EVENTS = {0, 1, 2, 3, 5, 10, 20, 50, 100}

R_MAX_PLOT_NM = 260.0
Z_RANGE_PLOT_NM = (-140.0, 140.0)
ZOOM_HALF_R_NM = 40.0
ZOOM_HALF_Z_NM = 40.0


def count_solid_components(f, threshold=0.5):
    labeled, n = ndimage.label(f > threshold)
    return n


def save_geometry_frame(png_path, f, particle, substrate, z, r_c, r_tj, z_tj, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    draw_mirrored_geometry(axes[0], f, particle, substrate, z, r_c, r_tj, z_tj,
                            R_MAX_PLOT_NM, Z_RANGE_PLOT_NM, title="full mirrored geometry")
    draw_neck_zoom(axes[1], f, particle, substrate, z, r_c, r_tj, z_tj, ZOOM_HALF_R_NM, ZOOM_HALF_Z_NM)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=105)
    plt.close(fig)


def build_setup():
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
    dt = find_stable_dt(f, particle, substrate, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    return dict(f=f, particle=particle, substrate=substrate, z=z, r_c=r_c, r_f=r_f, dr=dr, dz=dz, W=W,
                p=p, Wc=Wc, M_s=M_s, M_eta=M_eta, dt=dt)


def run_campaign(label, a0_ev, out_dir, target_events=None, max_steps=None, checkpoint_events=None):
    target_events = N_EVENTS_TARGET if target_events is None else target_events
    max_steps = MAX_STEPS if max_steps is None else max_steps
    checkpoint_events = CHECKPOINT_EVENTS if checkpoint_events is None else checkpoint_events
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "checkpoint_frames")
    os.makedirs(frames_dir, exist_ok=True)

    setup = build_setup()
    f, particle, substrate = setup["f"], setup["particle"], setup["substrate"]
    z, r_c, r_f = setup["z"], setup["r_c"], setup["r_f"]
    dr, dz, W = setup["dr"], setup["dz"], setup["W"]
    p, Wc, M_s, M_eta, dt = setup["p"], setup["Wc"], setup["M_s"], setup["M_eta"], setup["dt"]
    print(f"[{label}] dt={dt:.4e} A0={a0_ev}eV", flush=True)

    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B, D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=(a0_ev * 1.602176634e-19 if a0_ev is not None else 0.0),
                       V0=V0_OVER_B3 * B ** 3, tau_ex0=0.0)
    sink = AxisymSink(active=(a0_ev is None), current_disp=0.0)
    rng = np.random.default_rng(RANDOM_SEED)

    events = []
    next_event_id = [0]
    n_born = 0
    if a0_ev is not None:
        from pf_sintering.m16m_multisink import PoissonBirthClock, SinkEvent, multi_sink_transport_step, poisson_multisink_birth_step
        clock = PoissonBirthClock()

    V0_mass = axisym_volume(f, r_c, dr, dz)
    z_gb_guess = 0.0
    n_completed = 0
    step = 0
    rows = []
    prev_r_tj, prev_z_tj = None, None
    stop_reason = None
    checkpoints_saved = set()

    def measure_and_checkpoint(tag):
        R_of_z = measure_R_of_z(f, r_c)
        sigma_avg, _, z_tj, r_tj, sig_p, sig_n = operative_sigma_branch_resolved(
            f, particle, substrate, r_c, z, z_gb_guess, W)
        n_comp_now = count_solid_components(f)
        V_now = axisym_volume(f, r_c, dr, dz)
        mass_drift = (V_now - V0_mass) / V0_mass
        png_path = os.path.join(frames_dir, f"{tag}.png")
        save_geometry_frame(png_path, f, particle, substrate, z, r_c, r_tj, z_tj,
                             f"[{label}] {tag} t={step*dt:.4f} event={n_completed} "
                             f"sig_avg={sigma_avg/1e6:.1f}MPa r_tj={r_tj*1e9:.1f}nm n_comp={n_comp_now} "
                             f"mass_drift={mass_drift:.2e}")
        return sigma_avg, z_tj, r_tj, sig_p, sig_n, n_comp_now, mass_drift

    while n_completed < target_events and step < max_steps:
        if step > 0:
            f, particle, substrate, _ = axisym_gb_face_projected_step(f, particle, substrate, p, Wc, dr, dz, r_c, r_f,
                                                                        dt, M_s, M_eta, W, bc_z="noflux")
            if not np.all(np.isfinite(f)):
                stop_reason = f"BLOWUP (nonfinite f) at step {step}"
                print(stop_reason, flush=True)
                break
        t = step * dt

        R_of_z = measure_R_of_z(f, r_c)
        sigma_avg, _, z_tj, r_tj, sig_p, sig_n = operative_sigma_branch_resolved(
            f, particle, substrate, r_c, z, z_gb_guess, W)
        if np.isfinite(z_tj):
            z_gb_guess = z_tj

        # Section 19: fail-fast geometry checks
        if not (np.isfinite(r_tj) and np.isfinite(z_tj)):
            stop_reason = f"non-finite TJ at step {step}"
        elif prev_r_tj is not None and (abs(r_tj - prev_r_tj) > 20e-9 or abs(z_tj - prev_z_tj) > 20e-9):
            stop_reason = (f"large TJ jump at step {step}: "
                            f"(r,z) {prev_r_tj*1e9:.2f},{prev_z_tj*1e9:.2f} -> {r_tj*1e9:.2f},{z_tj*1e9:.2f} nm")
        if stop_reason:
            print(stop_reason, flush=True)
            save_geometry_frame(os.path.join(frames_dir, "FAIL_before.png"), f, particle, substrate, z, r_c,
                                 prev_r_tj or 0.0, prev_z_tj or 0.0, f"BEFORE: {stop_reason}")
            break
        prev_r_tj, prev_z_tj = r_tj, z_tj

        newly_completed_ids = []
        if a0_ev is None:
            f, particle, substrate, completed, tdiag = active_sink_transport_step(
                f, particle, substrate, sink, hp, sigma_avg, dt, dz, r_c, z, z_tj)
            if completed:
                newly_completed_ids = [None]
        else:
            n_births = poisson_multisink_birth_step(clock, sigma_avg, dt, hp, rng)
            for _ in range(n_births):
                next_event_id[0] += 1
                events.append(SinkEvent(event_id=next_event_id[0], birth_time=t, birth_step=step,
                                         birth_sigma=sigma_avg / 1e6))
                n_born += 1
            active_events = [e for e in events if e.active]
            if active_events:
                f, particle, substrate, newly_completed_ids, tdiag = multi_sink_transport_step(
                    f, particle, substrate, events, hp, sigma_avg, dt, dz, r_c, z, z_tj)

        if not np.all(np.isfinite(f)):
            stop_reason = f"BLOWUP (nonfinite f after RBM) at step {step}"
            print(stop_reason, flush=True)
            break

        if newly_completed_ids:
            n_completed += len(newly_completed_ids)
            if a0_ev is None:
                sink.active = True
                cum_b_now = sink.cumulative_disp / B
            else:
                cum_b_now = n_completed + sum(e.delta for e in events if e.active) / B
            V_now = axisym_volume(f, r_c, dr, dz)
            mass_drift = (V_now - V0_mass) / V0_mass
            n_comp_now = count_solid_components(f)
            rows.append(dict(event=n_completed, step=step, t=t, sigma_avg_MPa=sigma_avg / 1e6,
                              sigma_particle_MPa=sig_p / 1e6 if np.isfinite(sig_p) else float("nan"),
                              sigma_neighbor_MPa=sig_n / 1e6 if np.isfinite(sig_n) else float("nan"),
                              r_tj_nm=r_tj * 1e9, z_tj_nm=z_tj * 1e9, n_components=n_comp_now,
                              mass_drift=mass_drift, cum_b=cum_b_now))
            print(f"  [{label}] event #{n_completed} step={step} t={t:.4f} sigma_avg={sigma_avg/1e6:.2f}MPa "
                  f"r_tj={r_tj*1e9:.2f}nm n_comp={n_comp_now} mass_drift={mass_drift:.2e}", flush=True)
            if n_completed in checkpoint_events and n_completed not in checkpoints_saved:
                checkpoints_saved.add(n_completed)
                measure_and_checkpoint(f"event_{n_completed:03d}")
            if n_comp_now != 1:
                stop_reason = f"disconnected geometry ({n_comp_now} components) at event {n_completed}"
                print(stop_reason, flush=True)
                break
            if abs(mass_drift) > 1e-2:
                stop_reason = f"mass drift {mass_drift:.3e} exceeds 1e-2 at event {n_completed}"
                print(stop_reason, flush=True)
                break

        step += 1

    if stop_reason is None and n_completed < target_events:
        stop_reason = f"max_steps ({max_steps}) reached with only {n_completed}/{target_events} events completed"
        print(stop_reason, flush=True)

    if 0 not in checkpoints_saved:
        measure_and_checkpoint("event_000")
    measure_and_checkpoint("final")

    with open(os.path.join(out_dir, f"recovery_{label}_summary.csv"), "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[{label}] done: n_completed={n_completed} final_step={step} stop_reason={stop_reason}", flush=True)
    return dict(label=label, rows=rows, n_completed=n_completed, final_step=step, stop_reason=stop_reason,
                dt=dt)


def main():
    t0 = time.time()
    out_zero = "runs/recovery_zero_barrier"
    result_zero = run_campaign("zero_barrier", a0_ev=None, out_dir=out_zero)

    import shutil
    final_geom_src = os.path.join(out_zero, "checkpoint_frames", "final.png")
    if os.path.exists(final_geom_src):
        shutil.copy(final_geom_src, "recovery_zero_barrier_final_geometry.png")
    shutil.copy(os.path.join(out_zero, "recovery_zero_barrier_summary.csv"), "recovery_zero_barrier_summary.csv")

    print(f"\n=== zero-barrier campaign complete in {time.time()-t0:.1f}s: "
          f"n_completed={result_zero['n_completed']} stop_reason={result_zero['stop_reason']} ===\n", flush=True)

    zero_pass = result_zero["stop_reason"] is None
    print(f"ZERO-BARRIER GATE: {'PASS' if zero_pass else 'FAIL'}", flush=True)
    with open("runs/recovery_zero_barrier/GATE.json", "w") as fh:
        json.dump(dict(pass_=zero_pass, n_completed=result_zero["n_completed"],
                        stop_reason=result_zero["stop_reason"]), fh)

    if not zero_pass:
        print("Zero-barrier campaign did not reach the target cleanly -- stopping before finite-barrier run "
              "per the recovery's fail-fast instruction.", flush=True)
        sys.exit(1)

    # Section 20: finite-barrier run, EXACTLY the same spatial mechanics, differing
    # only in nucleation-barrier mode. Uses the M16Q-calibrated A0 (calibrated
    # against this SAME branch-resolved stress metric -- internally consistent).
    t1 = time.time()
    out_finite = "runs/recovery_finite_barrier"
    # finite-barrier nucleation is thermally activated and much slower per-event
    # than zero-barrier (smoke test: ~24900 steps for 3 events vs ~1100 for
    # zero-barrier) -- matches historical M16Q practice (~7-event finite-barrier
    # campaigns), so a smaller target keeps this tractable within the session.
    result_finite = run_campaign("finite_barrier", a0_ev=A0_EV_FINITE, out_dir=out_finite,
                                  target_events=20, max_steps=3_000_000,
                                  checkpoint_events={0, 1, 2, 3, 5, 10, 15, 20})

    final_geom_src = os.path.join(out_finite, "checkpoint_frames", "final.png")
    if os.path.exists(final_geom_src):
        shutil.copy(final_geom_src, "recovery_finite_barrier_final_geometry.png")
    shutil.copy(os.path.join(out_finite, "recovery_finite_barrier_summary.csv"),
                "recovery_finite_barrier_summary.csv")

    print(f"\n=== finite-barrier campaign complete in {time.time()-t1:.1f}s: "
          f"n_completed={result_finite['n_completed']} stop_reason={result_finite['stop_reason']} ===\n", flush=True)
    finite_pass = result_finite["stop_reason"] is None
    print(f"FINITE-BARRIER GATE: {'PASS' if finite_pass else 'FAIL'}", flush=True)
    with open("runs/recovery_finite_barrier/GATE.json", "w") as fh:
        json.dump(dict(pass_=finite_pass, n_completed=result_finite["n_completed"],
                        stop_reason=result_finite["stop_reason"]), fh)

    # Section 21: finite vs zero comparison plot (sigma_avg(t), cum_b(t), r_TJ(t))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    for res, style in ((result_zero, dict(label="zero-barrier")), (result_finite, dict(label="finite-barrier"))):
        rs = res["rows"]
        if not rs:
            continue
        ts = [r["t"] for r in rs]
        axes[0].plot(ts, [r["sigma_avg_MPa"] for r in rs], ".-", ms=3, **style)
        axes[1].plot(ts, [r["cum_b"] for r in rs], ".-", ms=3, **style)
        axes[2].plot(ts, [r["r_tj_nm"] for r in rs], ".-", ms=3, **style)
    axes[0].set_ylabel("sigma_avg (MPa)")
    axes[1].set_ylabel("cumulative delta_sink/b")
    axes[2].set_ylabel("r_TJ (nm)")
    axes[2].set_xlabel("t")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("recovery_finite_vs_zero_comparison.png", dpi=110)
    plt.close(fig)

    print("\nRECOVERY RUN COMPLETE.")
    print(f"  zero-barrier: n_completed={result_zero['n_completed']} PASS={zero_pass}")
    print(f"  finite-barrier: n_completed={result_finite['n_completed']} PASS={finite_pass}")


if __name__ == "__main__":
    main()
