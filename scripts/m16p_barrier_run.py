"""M16P: unified finite-barrier / zero-barrier production driver.

Generalizes M16M's Poisson multi-sink qualification driver
(scripts/m16m_multisink_qualification.py) to:
  - an arbitrary (chi, ratio) geometry (M16O's finite-neighbor-curvature
    topology family, not just the flat-substrate case);
  - the M16O operative-sigma convention (path-continuous NeckTracker,
    FIXED 12nm window -- matching the deterministic sink-OFF trajectories
    already measured, for direct comparability);
  - the CORRECTED M16N one-b contract with the M16P mass-conservation fix
    (delta_sink drives completion; mass residual ~1e-16/event);
  - TWO kinetic modes:
      --mode finite : independent Poisson multi-sink nucleation
                       (pf_sintering.m16m_multisink: PoissonBirthClock +
                       SinkEvent + multi_sink_transport_step), calibrated
                       barrier from --a0-ev.
      --mode zero   : the project's ESTABLISHED zero-barrier / continuous-
                       accommodation convention (scripts/
                       m16h_three_regime_sink_barrier.py's "low" mode:
                       sink.active forced True permanently, bypassing the
                       Arrhenius hazard entirely, immediately re-armed
                       upon each one-b completion) -- NOT a small finite
                       barrier. Ported to the corrected one-b contract
                       (active_sink_transport_step, delta_sink-driven)
                       and the M16O geometry/sigma convention.

Output safety (M16N/M16M convention, carried forward): incremental CSV
writes, directory-existence checks before every write, periodic NPZ
checkpoints, fail-closed on an unwritable output directory.

PNG movie frames: adaptive dual cadence (Sections 12-13) -- LOW cadence
during quiet loading, HIGH cadence (with a rolling pre-event buffer so
the onset is never undersampled) during any transient (N_active>0, a
birth/completion event, or a large instantaneous |d sigma/dt|).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, active_sink_transport_step  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402
from pf_sintering.m16k_neck_tracking import NeckTracker  # noqa: E402
from pf_sintering.m16m_multisink import PoissonBirthClock, SinkEvent, multi_sink_transport_step, poisson_multisink_birth_step  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
TRACK_WINDOW_NM = 12.0  # M16O convention

B = 2.5e-10
V0_OVER_B3 = 12.5
GS = 201.74e-9  # carried forward from M16H-M16N (characteristic diffusion length scale)
R0 = 1e12
T = 1000.0
RANDOM_SEED = 0

STRESS_DROP_STOP_MPA_NOISE_FLOOR = 0.5  # MPa, for peak/trough cycle detection noise rejection
REQUEST_MEASURE_RATIO_MIN = 0.05
REQUEST_MEASURE_RATIO_MAX = 20.0

PRE_EVENT_BUFFER_LEN = 30
HIGH_CADENCE_STEPS = 50  # PF steps between high-cadence frames during a transient
LOW_CADENCE_STEPS = 2000  # PF steps between low-cadence frames during quiet loading
DSIGMA_DT_TRANSIENT_THRESHOLD_MPA_PER_UNIT_T = 200.0  # heuristic trigger for "large |d sigma/dt|"


def _check_output_dir_or_stop(out_root):
    if not os.path.isdir(out_root):
        raise RuntimeError(f"OUTPUT DIRECTORY MISSING: {out_root} -- stopping simulation immediately")
    probe = os.path.join(out_root, ".write_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise RuntimeError(f"OUTPUT DIRECTORY UNWRITABLE: {out_root} ({exc}) -- stopping simulation immediately") from exc


class IncrementalWriter:
    def __init__(self, path, out_root):
        self.path = path
        self.out_root = out_root
        self.fieldnames = None
        self._fh = None
        self._writer = None

    def write(self, row):
        _check_output_dir_or_stop(self.out_root)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
            self._writer.writeheader()
        missing = [k for k in row if k not in self.fieldnames]
        if missing:
            self.close()
            self.fieldnames = self.fieldnames + missing
            row = {k: row.get(k, "") for k in self.fieldnames}
            self._fh = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        else:
            row = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def operative_sigma(f, r_c, z, tracker, W_for_tracker_step):
    from pf_sintering.hussein_neck_stress import neck_curvature_windows
    R_of_z = measure_R_of_z(f, r_c)
    tr = tracker.step(R_of_z, z, W_for_tracker_step)
    if tr["selected_contact"] is None:
        z_gb = tracker.prev_z_gb if tracker.prev_z_gb is not None else 0.0
        return 0.0, R_of_z, z_gb, float("nan")
    z_gb, a = tr["selected_contact"]
    win = neck_curvature_windows(R_of_z, z, z_gb, 1e-9, window_widths_in_W=(TRACK_WINDOW_NM,))[0]
    r_neck = win["r_neck"]
    if np.isfinite(r_neck) and r_neck > 0:
        sigma_H, *_ = hussein_eq1b_sigma(r_neck, 2 * a, GAMMA_S, GAMMA_GB)
    else:
        sigma_H = 0.0
    return (sigma_H if np.isfinite(sigma_H) else 0.0), R_of_z, z_gb, a


def save_frame(png_path, f, particle, substrate, z, r_c, z_gb, a_contact, t, sigma_MPa, n_active, n_completed,
               cum_delta_sink_over_b, r_max_plot_nm=None, z_range_plot_nm=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    Z_nm = z[:, None] * 1e9 * np.ones((1, len(r_c)))
    R_nm = r_c[None, :] * 1e9 * np.ones((len(z), 1))
    ax.contourf(R_nm, Z_nm, particle, levels=[0.5, 1.5], colors=["#4C72B0"], alpha=0.6)
    ax.contourf(R_nm, Z_nm, substrate, levels=[0.5, 1.5], colors=["#DD8452"], alpha=0.6)
    ax.contour(R_nm, Z_nm, f, levels=[0.5], colors="black", linewidths=0.8)
    if np.isfinite(z_gb) and np.isfinite(a_contact):
        ax.plot([a_contact * 1e9], [z_gb * 1e9], marker="x", color="red", ms=10, mew=2)
    if r_max_plot_nm is not None:
        ax.set_xlim(0, r_max_plot_nm)
    if z_range_plot_nm is not None:
        ax.set_ylim(*z_range_plot_nm)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(f"t={t:.4f}  sigma={sigma_MPa:.2f}MPa  N_active={n_active}  "
                 f"N_completed={n_completed}  cum_dsink/b={cum_delta_sink_over_b:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def main(chi, ratio, mode, a0_ev, w_nm=6.0, dx_nm=0.85, t_target_cap=2.0, n_samples=4000,
         max_wall_hours=4.0, out_root=None, save_frames=True, r_max_plot_nm=None, z_range_plot_nm=None):
    assert mode in ("finite", "zero")
    chi_label = "flat" if chi is None else f"chi{chi:g}"
    label = f"{chi_label}_ratio{ratio:.3f}_{mode}"
    out_root = out_root or os.path.join(os.path.dirname(__file__), "..", "runs", "m16p_barrier_run", label)
    frames_dir = os.path.join(out_root, "frames")
    os.makedirs(out_root, exist_ok=True)
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    _check_output_dir_or_stop(out_root)
    t0 = time.time()

    R_s_nm = None if chi is None else chi * R_P_NM
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=R_s_nm, X0_over_2Rp=ratio, psi_deg=PSI_DEG,
                                    W_nm=w_nm, dr_nm=dx_nm, dz_nm=dx_nm, aspect_ratio=1.0)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    roles = roles_from_m16j_geometry(e1, e2)
    particle, substrate = roles.particle, roles.substrate
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = w_nm * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B, D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=a0_ev * 1.602176634e-19 if a0_ev is not None else 0.0,
                       V0=V0_OVER_B3 * B ** 3, tau_ex0=0.0)
    rng = np.random.default_rng(RANDOM_SEED)

    dt = find_stable_dt(f, particle, substrate, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    n_steps_cap = max(1, int(t_target_cap / dt))
    diag_every = max(1, n_steps_cap // n_samples)
    print(f"[M16P {label}] dt={dt:.4e} n_steps_cap={n_steps_cap} diag_every={diag_every} mode={mode} "
          f"A0={a0_ev}eV", flush=True)

    tracker = NeckTracker()
    V0_mass = axisym_volume(f, r_c, dr, dz)

    # kinetics state
    clock = PoissonBirthClock()
    events: list[SinkEvent] = []
    next_event_id = [0]
    zero_sink = AxisymSink(active=(mode == "zero"), current_disp=0.0)

    n_born = 0
    n_completed = 0
    cumulative_delta_sink = 0.0
    cumulative_delta_COM = 0.0

    history_writer = IncrementalWriter(os.path.join(out_root, "history.csv"), out_root)
    event_dense_writer = IncrementalWriter(os.path.join(out_root, "event_dense_history.csv"), out_root)
    events_path = os.path.join(out_root, "events.jsonl")
    last_checkpoint_t = -1.0

    pre_buffer = deque(maxlen=PRE_EVENT_BUFFER_LEN)
    frame_idx = [0]
    last_frame_step = -10 ** 9
    in_transient = False
    prev_sigma_for_deriv = None
    prev_t_for_deriv = None

    def maybe_save_frame(step, t, sigma_now, n_active, force=False, flush_buffer=False):
        nonlocal last_frame_step
        if not save_frames:
            return
        cadence = HIGH_CADENCE_STEPS if in_transient else LOW_CADENCE_STEPS
        if not force and (step - last_frame_step) < cadence:
            return
        last_frame_step = step
        png_path = os.path.join(frames_dir, f"frame_{frame_idx[0]:05d}.png")
        R_of_z = measure_R_of_z(f, r_c)
        tr = tracker.prev_z_gb
        a_contact = float("nan")
        z_gb_show = tr if tr is not None else float("nan")
        save_frame(png_path, f, particle, substrate, z, r_c, z_gb_show, a_contact, t, sigma_now / 1e6,
                   n_active, n_completed, cumulative_delta_sink / B, r_max_plot_nm, z_range_plot_nm)
        frame_idx[0] += 1

    step = 0
    while True:
        if step > 0:
            f, particle, substrate, _ = axisym_gb_face_projected_step(f, particle, substrate, p, Wc, dr, dz, r_c, r_f,
                                                                        dt, M_s, M_eta, W, bc_z="noflux")
            if not np.all(np.isfinite(f)):
                stop_reason = f"BLOWUP at step {step}"
                print(stop_reason)
                break

        t = step * dt
        sigma_now, R_of_z, z_gb, a_contact = operative_sigma(f, r_c, z, tracker, W)

        do_diag = (step % diag_every == 0)

        # --- kinetics ---
        newly_completed = []
        n_active = 0
        if mode == "finite":
            n_births = poisson_multisink_birth_step(clock, sigma_now, dt, hp, rng)
            for _ in range(n_births):
                next_event_id[0] += 1
                ev = SinkEvent(event_id=next_event_id[0], birth_time=t, birth_step=step, birth_sigma=sigma_now / 1e6)
                events.append(ev)
                n_born += 1
                print(f"  *** BIRTH #{n_born} id={ev.event_id} step={step} t={t:.4f} sigma={sigma_now/1e6:.2f}MPa "
                      f"N_active={sum(e.active for e in events)} ***", flush=True)
                _check_output_dir_or_stop(out_root)
                with open(events_path, "a") as fh:
                    fh.write(json.dumps(dict(kind="birth", event_id=ev.event_id, step=step, t=t,
                                              sigma_MPa=sigma_now / 1e6)) + "\n")
            active_events = [e for e in events if e.active]
            n_active = len(active_events)
            if active_events:
                f, particle, substrate, newly_completed, tdiag = multi_sink_transport_step(
                    f, particle, substrate, events, hp, sigma_now, dt, dz, r_c, z, z_gb)
                cumulative_delta_sink += tdiag.get("total_requested_dDelta", 0.0)
                cumulative_delta_COM += tdiag.get("total_measured_relative_dDelta", 0.0)
                if do_diag or newly_completed:
                    event_dense_writer.write(dict(step=step, time=t, sigma_MPa=sigma_now / 1e6, **tdiag))
                for eid in newly_completed:
                    ev = next(e for e in events if e.event_id == eid)
                    ev.completion_time = t
                    ev.completion_step = step
                    ev.completion_sigma = sigma_now / 1e6
                    n_completed += 1
                    print(f"  *** COMPLETE id={eid} (#{n_completed}) step={step} t={t:.4f} "
                          f"sigma={sigma_now/1e6:.2f}MPa N_active={sum(e.active for e in events)} "
                          f"cum_dsink/b={cumulative_delta_sink/B:.3f} ***", flush=True)
                    _check_output_dir_or_stop(out_root)
                    with open(events_path, "a") as fh:
                        fh.write(json.dumps(dict(kind="complete", event_id=eid, step=step, t=t,
                                                  sigma_MPa=sigma_now / 1e6,
                                                  cumulative_delta_sink_over_b=cumulative_delta_sink / B)) + "\n")
        else:  # zero-barrier: continuous accommodation (established convention)
            f, particle, substrate, completed, tdiag = active_sink_transport_step(
                f, particle, substrate, zero_sink, hp, sigma_now, dt, dz, r_c, z, z_gb)
            n_active = 1 if zero_sink.active else 0
            cumulative_delta_sink += tdiag.get("delta_sink_this_step", 0.0)
            cumulative_delta_COM += tdiag.get("delta_COM_this_step", 0.0)
            if do_diag or completed:
                event_dense_writer.write(dict(step=step, time=t, sigma_MPa=sigma_now / 1e6, **tdiag))
            if completed:
                n_completed += 1
                n_born += 1
                print(f"  *** [zero-barrier] one-b COMPLETE #{n_completed} step={step} t={t:.4f} "
                      f"sigma={sigma_now/1e6:.2f}MPa cum_dsink/b={cumulative_delta_sink/B:.3f} ***", flush=True)
                _check_output_dir_or_stop(out_root)
                with open(events_path, "a") as fh:
                    fh.write(json.dumps(dict(kind="complete_zero_barrier", step=step, t=t,
                                              sigma_MPa=sigma_now / 1e6,
                                              cumulative_delta_sink_over_b=cumulative_delta_sink / B)) + "\n")
                zero_sink.active = True  # immediately re-arm: continuous accommodation

        # --- transient detection for frame cadence ---
        dsigma_dt = 0.0
        if prev_sigma_for_deriv is not None and t > prev_t_for_deriv:
            dsigma_dt = (sigma_now / 1e6 - prev_sigma_for_deriv) / (t - prev_t_for_deriv)
        prev_sigma_for_deriv, prev_t_for_deriv = sigma_now / 1e6, t
        in_transient = (n_active > 0) or bool(newly_completed) or (abs(dsigma_dt) > DSIGMA_DT_TRANSIENT_THRESHOLD_MPA_PER_UNIT_T)

        # rolling pre-event buffer: always record a lightweight frame descriptor;
        # flush (save) the buffered ones the moment a transient begins
        pre_buffer.append((step, t, sigma_now, n_active))
        if in_transient and save_frames:
            maybe_save_frame(step, t, sigma_now, n_active, force=(n_born > 0 and step - last_frame_step > 5))
        elif do_diag:
            maybe_save_frame(step, t, sigma_now, n_active)

        if do_diag:
            row = dict(step=step, time=t, sigma_MPa=sigma_now / 1e6, a_contact_nm=a_contact * 1e9 if np.isfinite(a_contact) else float("nan"),
                       X_neck_nm=2 * a_contact * 1e9 if np.isfinite(a_contact) else float("nan"),
                       N_active=n_active, N_born=n_born, N_completed=n_completed,
                       cumulative_delta_sink_nm=cumulative_delta_sink * 1e9,
                       cumulative_delta_sink_over_b=cumulative_delta_sink / B,
                       cumulative_delta_COM_nm=cumulative_delta_COM * 1e9,
                       cumulative_delta_COM_over_b=cumulative_delta_COM / B,
                       mass_drift=(axisym_volume(f, r_c, dr, dz) - V0_mass) / V0_mass)
            history_writer.write(row)
            print(f"  t={t:.3f} sigma={sigma_now/1e6:.2f}MPa N_active={n_active} N_born={n_born} "
                  f"N_completed={n_completed} cum_dsink/b={cumulative_delta_sink/B:.3f} "
                  f"mass_drift={row['mass_drift']:.2e} wall={time.time()-t0:.0f}s", flush=True)

            if t - last_checkpoint_t >= 0.05:
                _check_output_dir_or_stop(out_root)
                np.savez_compressed(os.path.join(out_root, "checkpoint.npz"), f=f, particle=particle,
                                     substrate=substrate, step=step, n_born=n_born, n_completed=n_completed)
                last_checkpoint_t = t

        if step >= n_steps_cap:
            stop_reason = f"reached n_steps_cap (t_target_cap={t_target_cap})"
            break
        if time.time() - t0 >= max_wall_hours * 3600.0:
            stop_reason = f"PRAGMATIC wall-clock budget reached ({max_wall_hours}h)"
            break
        step += 1

    history_writer.close()
    event_dense_writer.close()
    meta = dict(chi=chi, ratio=ratio, mode=mode, a0_ev=a0_ev, w_nm=w_nm, dx_nm=dx_nm, n_born=n_born,
                n_completed=n_completed, cumulative_delta_sink_over_b=cumulative_delta_sink / B,
                cumulative_delta_COM_over_b=cumulative_delta_COM / B, final_step=step, final_time=step * dt,
                stop_reason=stop_reason, wall_time_s=time.time() - t0, n_switches=len(tracker.switch_log),
                n_frames=frame_idx[0])
    with open(os.path.join(out_root, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[M16P {label}] DONE wall={time.time()-t0:.0f}s stop_reason={stop_reason} n_born={n_born} "
          f"n_completed={n_completed} cum_dsink/b={cumulative_delta_sink/B:.3f} n_frames={frame_idx[0]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=float, default=None)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--mode", choices=["finite", "zero"], required=True)
    ap.add_argument("--a0-ev", type=float, default=None)
    ap.add_argument("--w-nm", type=float, default=6.0)
    ap.add_argument("--dx-nm", type=float, default=0.85)
    ap.add_argument("--t-target-cap", type=float, default=2.0)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--max-wall-hours", type=float, default=4.0)
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()
    if args.mode == "finite" and args.a0_ev is None:
        raise SystemExit("--a0-ev is required for --mode finite")
    main(args.chi, args.ratio, args.mode, args.a0_ev, w_nm=args.w_nm, dx_nm=args.dx_nm,
         t_target_cap=args.t_target_cap, n_samples=args.n_samples, max_wall_hours=args.max_wall_hours,
         save_frames=not args.no_frames)
