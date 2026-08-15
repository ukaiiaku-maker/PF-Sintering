"""Milestone 16L: CORRECTED one-event finite-hazard qualification run.

Two fixes relative to scripts/m16k_first_event_qualification_v2.py:

1. GRAIN-IDENTITY CORRECTION (the central M16L finding): the M16J
   geometry construction (pf_sintering.m16j_geometry.build_candidate_
   geometry) assigns e1=SUBSTRATE, e2=PARTICLE -- the OPPOSITE of what
   pf_sintering.axisym_sink_rbm.py's active_sink_transport_step/
   particle_com_z assume (their own contract, and the OLDER M16G
   geometry they were validated against, both use e1=particle). The M16K
   driver called `active_sink_transport_step(f, e1, e2, ...)` directly
   with the M16J fields, silently advecting/measuring the SUBSTRATE
   instead of the particle for the entire M16K investigation. Fixed here
   by calling `active_sink_transport_step(f, e2, e1, ...)` (swapped) and
   correspondingly unswapping the returned grain fields. Verified via
   scripts/m16l_relative_displacement_microtest.py: with the correct
   grain, a full b=0.25nm particle-relative-to-substrate displacement
   changes sigma_Hussein by <0.5% (45.12->44.91 MPa) -- a full Burgers
   vector is geometrically tiny relative to this ~20-280nm neck, a
   dramatically different (and more physically sensible) picture than
   the wrong-grain version's ~45->~0 MPa "collapse".

2. OUTPUT SAFETY (Section 18): history is now written incrementally
   (appended every diagnostic sample, not held in RAM until the end),
   the run directory's existence is checked before every write, and the
   simulation STOPS IMMEDIATELY if the directory becomes missing/
   unwritable, rather than continuing to compute with nowhere to save
   results (the M16K t=25 extension lost ~80 minutes of dense output
   this way).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.axisym_interface_metrics import free_surface_area_of_revolution, surface_to_volume_metrics  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, active_sink_transport_step, nucleation_hazard_step  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402
from pf_sintering.m16k_neck_tracking import NeckTracker, find_tj_from_contour  # noqa: E402
from pf_sintering.young_laplace_pressure import young_laplace_pressure as young_laplace_pressure_fn  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
RATIO = 0.10
W_NM = 10.0
DX_NM = 1.25

# frozen hazard parameterization -- UNCHANGED from M16K (Section 19: "do
# not retune the barrier")
B = 2.5e-10
V0 = 12.5 * B ** 3
A0_EV = 0.8589
A0_J = A0_EV * 1.602176634e-19
GS = 201.74e-9
R0 = 1e12
T = 1000.0
RANDOM_SEED = 0

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16l_first_event_qualification")


def _check_output_dir_or_stop(out_root):
    """Section 18: fail closed. If the run directory has vanished or
    become unwritable, STOP the simulation immediately rather than
    continuing to compute with nowhere to save results."""
    if not os.path.isdir(out_root):
        raise RuntimeError(f"OUTPUT DIRECTORY MISSING: {out_root} -- stopping simulation immediately (Section 18)")
    probe = os.path.join(out_root, ".write_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise RuntimeError(f"OUTPUT DIRECTORY UNWRITABLE: {out_root} ({exc}) -- stopping simulation immediately") from exc


def diagnostics(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, W, tracker, step, t, V0_mass, F0):
    """Note: e1=substrate, e2=particle (M16J convention) -- the
    diagnostics themselves (R(z), curvature, sigma_Hussein) don't care
    which is which, only the TRANSPORT physics does."""
    R_of_z = measure_R_of_z(f, r_c)
    tr = tracker.step(R_of_z, z, W, step_index=step, t=t)
    if tr["selected_contact"] is None:
        return None, tr
    z_gb, a = tr["z_gb"], tr["a_contact"]
    tj = find_tj_from_contour(f, e1, e2, r_c, z, R_of_z, z_gb)
    X_neck = 2 * a
    r_neck = tr["all_candidate_curvatures"].get(1.5, float("nan"))
    if np.isfinite(r_neck) and r_neck > 0:
        sigma_H, sc, sw, C_GB = hussein_eq1b_sigma(r_neck, X_neck, GAMMA_S, GAMMA_GB)
    else:
        sigma_H = sc = sw = float("nan")
    win = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5,))[0]
    p_YL, km, ka = young_laplace_pressure_fn(win["kappa_signed"], a, GAMMA_S)
    V = axisym_volume(f, r_c, dr, dz)
    A_free = free_surface_area_of_revolution(R_of_z, z)
    A_GB = math.pi * a ** 2
    sv = surface_to_volume_metrics(A_free, A_GB, V)
    F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
    row = dict(step=step, time=t, a_contact_nm=a * 1e9, X_neck_nm=X_neck * 1e9,
               r_neck_1p5W_nm=r_neck * 1e9 if np.isfinite(r_neck) else float("nan"),
               sigma_Hussein_MPa=sigma_H / 1e6, sigma_curvature_MPa=sc / 1e6, sigma_width_MPa=sw / 1e6,
               young_laplace_pressure_MPa=p_YL / 1e6, S_over_V_1_per_nm=sv["S_over_V"] * 1e-9, free_energy=F,
               mass_drift=(V - V0_mass) / V0_mass, n_candidates=len(tr["all_candidate_contacts"]),
               z_tj_from_contour_nm=tj["z_tj"] * 1e9 if np.isfinite(tj["z_tj"]) else float("nan"),
               tj_agreement_nm=tj["agreement_nm"])
    return row, tr


class IncrementalWriter:
    """Section 18: append-as-you-go CSV writer, checking the output
    directory exists before every write."""

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
        # tolerate new keys appearing later (event_dense rows gain keys
        # once transport starts) by re-opening with a superset header if needed
        missing = [k for k in row if k not in self.fieldnames]
        if missing:
            self.close()
            self.fieldnames = self.fieldnames + missing
            # rewrite header-only file is not attempted mid-run (would lose
            # prior rows); instead just widen the row to existing fields
            # and drop truly-new keys from CSV (still present in events.json)
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


def main(t_target=10.0, n_samples=200):
    os.makedirs(OUT_ROOT, exist_ok=True)
    _check_output_dir_or_stop(OUT_ROOT)
    t0 = time.time()

    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]  # e1=substrate, e2=particle (M16J convention)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = W_NM * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B, D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=A0_J, V0=V0, tau_ex0=0.0)
    sink = AxisymSink()
    rng = np.random.default_rng(RANDOM_SEED)

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    n_steps_total = max(1, int(t_target / dt))
    diag_every = max(1, n_steps_total // n_samples)
    sigma_update_every = 20
    print(f"regime=finite_barrier_sink_inactive [M16L, grain-identity CORRECTED] dt={dt:.4e} "
          f"n_steps_total={n_steps_total} V0/b^3=12.5 A0={A0_EV}eV GS={GS*1e9:.2f}nm", flush=True)

    tracker = NeckTracker()
    V0_mass = axisym_volume(f, r_c, dr, dz)
    F0 = None
    events = []
    activation_step = None
    event_completed_step = None
    sigma_now = None

    history_writer = IncrementalWriter(os.path.join(OUT_ROOT, "history.csv"), OUT_ROOT)
    event_writer = IncrementalWriter(os.path.join(OUT_ROOT, "event_dense_history.csv"), OUT_ROOT)
    events_path = os.path.join(OUT_ROOT, "events.jsonl")
    last_checkpoint_t = -1.0

    for step in range(0, n_steps_total + 1):
        if step > 0:
            f, e1, e2, _ = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                                           bc_z="noflux")
            if not np.all(np.isfinite(f)):
                print(f"BLOWUP at step {step}"); break

        do_diag = (step % diag_every == 0) or step == n_steps_total
        do_sigma_update = (step % sigma_update_every == 0) or do_diag
        if do_diag:
            row, tr = diagnostics(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, W, tracker, step, step * dt, V0_mass, F0)
            if row is not None:
                if F0 is None:
                    F0 = row["free_energy"]
                sigma_now = row["sigma_Hussein_MPa"] * 1e6
        elif do_sigma_update:
            R_of_z = measure_R_of_z(f, r_c)
            z_gb = tracker.prev_z_gb if tracker.prev_z_gb is not None else 0.0
            win = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5,))[0]
            if np.isfinite(win["r_neck"]) and win["r_neck"] > 0:
                a_now = float(np.interp(z_gb, z, R_of_z))
                sigma_now, _, _, _ = hussein_eq1b_sigma(win["r_neck"], 2 * a_now, GAMMA_S, GAMMA_GB)
            else:
                sigma_now = 0.0
            row = None
        else:
            row = None

        if not sink.active:
            activated = nucleation_hazard_step(sink, sigma_now, dt, hp, rng)
            if activated:
                activation_step = step
                print(f"  *** ACTIVATION at step={step} t={step*dt:.4f} sigma={sigma_now/1e6:.3f}MPa ***", flush=True)
                ev = dict(kind="activation", step=step, t=step * dt, sigma_MPa=sigma_now / 1e6)
                events.append(ev)
                _check_output_dir_or_stop(OUT_ROOT)
                with open(events_path, "a") as fh:
                    fh.write(json.dumps(ev) + "\n")
        else:
            R_of_z = measure_R_of_z(f, r_c)
            z_gb = tracker.prev_z_gb if tracker.prev_z_gb is not None else 0.0
            win = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5,))[0]
            if np.isfinite(win["r_neck"]) and win["r_neck"] > 0:
                a_now = float(np.interp(z_gb, z, R_of_z))
                sigma_now, _, _, _ = hussein_eq1b_sigma(win["r_neck"], 2 * a_now, GAMMA_S, GAMMA_GB)
            else:
                sigma_now = 0.0
            # *** GRAIN-IDENTITY FIX: pass (e2=particle, e1=substrate) into
            # the function's (particle, substrate) argument slots, then
            # unswap the returned fields back into (e1, e2). ***
            f, e2, e1, completed, transport_diag = active_sink_transport_step(
                f, e2, e1, sink, hp, sigma_now, dt, dz, r_c, z, z_gb)
            event_writer.write(dict(step=step, time=step * dt, sigma_Hussein_MPa=sigma_now / 1e6, **transport_diag))
            if completed:
                event_completed_step = step
                print(f"  *** EVENT COMPLETE at step={step} t={step*dt:.4f} "
                      f"RBM_disp={sink.cumulative_disp*1e9:.4f}nm ***", flush=True)
                ev = dict(kind="complete", step=step, t=step * dt,
                          RBM_cumulative_displacement_nm=sink.cumulative_disp * 1e9)
                events.append(ev)
                _check_output_dir_or_stop(OUT_ROOT)
                with open(events_path, "a") as fh:
                    fh.write(json.dumps(ev) + "\n")

        if do_diag and row is not None:
            row["sink_active"] = bool(sink.active)
            row["hazard"] = sink.hazard
            row["n_events"] = sink.n_events
            row["RBM_cumulative_displacement_nm"] = sink.cumulative_disp * 1e9
            row["delta_event_nm"] = sink.current_disp * 1e9
            row["rbm_strain"] = sink.cumulative_disp / (2 * (row["a_contact_nm"] * 1e-9))
            history_writer.write(row)
            print(f"  t={row['time']:.3f} sigma={row['sigma_Hussein_MPa']:.3f}MPa sink={'ACTIVE' if sink.active else 'inactive'} "
                  f"hazard={sink.hazard:.4e} n_events={sink.n_events} delta_event={row['delta_event_nm']:.6f}nm "
                  f"mass_drift={row['mass_drift']:.2e} wall={time.time()-t0:.0f}s", flush=True)

            # periodic checkpoint (Section 18: at least every 0.1-0.25
            # model-time units)
            if row["time"] - last_checkpoint_t >= 0.1:
                _check_output_dir_or_stop(OUT_ROOT)
                np.savez_compressed(os.path.join(OUT_ROOT, "checkpoint.npz"), f=f, e1=e1, e2=e2, step=step,
                                     sink_active=sink.active, hazard=sink.hazard, current_disp=sink.current_disp,
                                     cumulative_disp=sink.cumulative_disp, n_events=sink.n_events)
                last_checkpoint_t = row["time"]

        if event_completed_step is not None and step - event_completed_step > n_steps_total // 20:
            print(f"stopping after post-event relaxation window at step={step}")
            break

    history_writer.close()
    event_writer.close()

    meta = dict(regime="finite_barrier_sink_inactive_m16l_grain_corrected", R_p_nm=R_P_NM, ratio=RATIO,
                W_nm=W_NM, dx_nm=DX_NM, V0_over_b3=12.5, A0_eV=A0_EV, GS_nm=GS * 1e9, r0=R0, T=T,
                random_seed=RANDOM_SEED, n_events=sink.n_events, activation_step=activation_step,
                event_completed_step=event_completed_step, wall_time_s=time.time() - t0,
                n_switches=len(tracker.switch_log))
    with open(os.path.join(OUT_ROOT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(os.path.join(OUT_ROOT, "switch_log.json"), "w") as fh:
        json.dump(tracker.switch_log, fh, indent=2, default=str)
    print(f"DONE wall={time.time()-t0:.0f}s n_events={sink.n_events} event_completed={event_completed_step is not None}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-target", type=float, default=10.0)
    ap.add_argument("--n-samples", type=int, default=200)
    args = ap.parse_args()
    main(t_target=args.t_target, n_samples=args.n_samples)
