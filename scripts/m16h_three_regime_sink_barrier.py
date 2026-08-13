"""Milestone 16H: three-regime (OFF / LOW barrier / FINITE barrier) sink
demonstration on the promoted crest-capped psi=160 particle/asperity
geometry (Milestone 16G). Regime OFF reuses the already-run, extensively
validated c3_crest_psi160 / c3_long_psi160 trajectories (no sink/RBM
code at all) -- this script implements Regimes LOW and FINITE, using
pf_sintering/axisym_sink_rbm.py's hazard/RBM machinery.

Checkpointed, append-only-JSONL logging (mirroring
scripts/m16g_long_run_c3.py's established, restart-parity-qualified
pattern) -- a FINITE-barrier run demonstrating several cycles can take
many hours, and an earlier version of this script only wrote its output
once at the very end via a single json.dump, which would have lost all
data if interrupted. Fixed before any long run was committed to.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, calibrate_barrier, hazard_step, particle_com_z, rbm_step, update_tau_sink  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import build_particle_asperity_geometry_c3_crest, find_gb_trough, find_stable_dt, grain_volumes  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16h_campaign")
REQUIRED_META_KEYS = ("a0", "Vp0", "Vs0", "V0", "com0", "lam", "z1", "dt", "Nr", "Nz")


def build_case(psi_deg=160.0, R_cyl_nm=100.0, W_nm=10.0, dx_nm=1.25):
    R_cyl, W, dr, dz = R_cyl_nm * 1e-9, W_nm * 1e-9, dx_nm * 1e-9, dx_nm * 1e-9
    gamma_gb = psi_to_gamma_gb(psi_deg)
    gamma_s = 1.0
    p = P(gamma_s=gamma_s, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]
    geom = build_particle_asperity_geometry_c3_crest(R_cyl, W, dr, dz)
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))
    dt = find_stable_dt(geom["f"], geom["e1"], geom["e2"], p, Wc, dr, dz, geom["r_c"], geom["r_f"],
                         M_s, M_eta, W, n_check=100)
    dt *= 0.4
    return geom, p, Wc, dr, dz, M_s, M_eta, dt, gamma_s, gamma_gb


def diagnostics(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam, Vp0, Vs0, V0, a0, com0,
                 gamma_s, gamma_gb, W, step, t, sink: AxisymSink):
    R_of_z = measure_R_of_z(f, r_c)
    z_gb, a = find_gb_trough(R_of_z, z, z_gb_prev, lam=lam)
    Vp, Vs = grain_volumes(e1, e2, r_c, dr, dz)
    V = axisym_volume(f, r_c, dr, dz)
    com = particle_com_z(e1, z, r_c, dr, dz)
    row = dict(step=step, t=t, a_over_a0=a / a0 if np.isfinite(a) else float("nan"),
               a_nm=a * 1e9 if np.isfinite(a) else float("nan"),
               Vp_frac=Vp / Vp0, Vs_frac=Vs / Vs0, mass_drift=(V - V0) / V0,
               com_shift_nm=(com0 - com) * 1e9 if np.isfinite(com) and np.isfinite(com0) else float("nan"),
               densification_strain=(com0 - com) / (2 * a0) if np.isfinite(com) and np.isfinite(com0) else float("nan"),
               sink_active=sink.active, sink_hazard=sink.hazard, sink_n_events=sink.n_events,
               sink_cumulative_disp_nm=sink.cumulative_disp * 1e9)
    if np.isfinite(z_gb) and np.isfinite(a):
        X_neck = 2 * a
        wins = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(1.5, 2.5))
        for w in wins:
            tag = f"w{w['window_W']:g}W"
            row[f"r_neck_{tag}_nm"] = w["r_neck"] * 1e9 if np.isfinite(w["r_neck"]) else float("nan")
            if np.isfinite(w["r_neck"]):
                sigma, sc, sg, CGB = hussein_eq1b_sigma(w["r_neck"], X_neck, gamma_s, gamma_gb)
                row[f"sigma_s_{tag}"] = sigma
    return row, (z_gb if np.isfinite(z_gb) else z_gb_prev)


def run_regime(tag, mode, t_target, n_sample, sigma_target_hazard=2.5e6, v0_mult=10000.0,
               psi_deg=160.0, checkpoint_every_steps=20000, diag_every_steps=None,
               target_n_events=None, max_wall_s=None):
    """mode: 'low' (continuously-active sink, negligible barrier) or
    'finite' (Arrhenius hazard). `target_n_events`: if set (finite mode
    only), stop once this many sink events have completed, even before
    t_target -- avoids paying for the full (possibly very long) t_target
    once enough cycles are already visible."""
    W = 10e-9
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    ckpt_path = os.path.join(CAMPAIGN_DIR, f"{tag}_checkpoint.npz")
    log_path = os.path.join(CAMPAIGN_DIR, f"{tag}_log.jsonl")
    event_path = os.path.join(CAMPAIGN_DIR, f"{tag}_events.jsonl")
    meta_path = os.path.join(CAMPAIGN_DIR, f"{tag}_meta.json")

    diag_every_steps = diag_every_steps or max(1, int(t_target / n_sample / 4.8828e-5))

    if os.path.exists(ckpt_path) and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        missing = [k for k in REQUIRED_META_KEYS if k not in meta]
        if missing:
            raise RuntimeError(f"meta.json missing required keys {missing} -- refusing to resume")
        data = np.load(ckpt_path)
        f, e1, e2 = data["f"], data["e1"], data["e2"]
        step = int(data["step"])
        z = data["z"]
        geom, p, Wc, dr, dz, M_s, M_eta, dt, gamma_s, gamma_gb = build_case(psi_deg=psi_deg)
        r_c, r_f = geom["r_c"], geom["r_f"]
        lam, z1 = meta["lam"], meta["z1"]
        a0, Vp0, Vs0, V0, com0 = meta["a0"], meta["Vp0"], meta["Vs0"], meta["V0"], meta["com0"]
        z_gb_prev = meta.get("z_gb_prev", z1)
        sink = AxisymSink(**meta["sink_state"])
        hp = HazardParams(**meta["hp_state"])
        rng = np.random.default_rng(0)
        rng.bit_generator.state = meta["rng_state"]
        print(f"resumed [{tag}] from step={step}, t={step*dt:.4f}, n_events={sink.n_events}")
    else:
        geom, p, Wc, dr, dz, M_s, M_eta, dt, gamma_s, gamma_gb = build_case(psi_deg=psi_deg)
        f, e1, e2 = geom["f"], geom["e1"], geom["e2"]
        z, r_c, r_f = geom["z"], geom["r_c"], geom["r_f"]
        z1, lam = geom["z1"], geom["lam"]

        R_of_z0 = measure_R_of_z(f, r_c)
        a0 = float(np.interp(z1, z, R_of_z0))
        Vp0, Vs0 = grain_volumes(e1, e2, r_c, dr, dz)
        V0 = axisym_volume(f, r_c, dr, dz)
        com0 = particle_com_z(e1, z, r_c, dr, dz)

        hp = HazardParams(V0=v0_mult * 1e-29, GS=2 * geom["R_z1"])
        hp = calibrate_barrier(sigma_target=sigma_target_hazard, hp=hp)
        rng = np.random.default_rng(0)
        sink = AxisymSink()
        if mode == "low":
            # LOW barrier = negligible activation barrier: the sink is
            # ALWAYS active (continuous accommodation), never gated by
            # the stress-dependent Arrhenius hazard at all -- routing
            # "low barrier" through the SAME hazard_step used for the
            # finite case (just with a tiny threshold) was tried first
            # and found to still leave the sink essentially inert at
            # low stress (the hazard model's calibration targets a much
            # higher sigma_target than the initial ~0.85 MPa state, so
            # r_nuc is negligible there regardless of threshold) -- this
            # is correct for a FINITE barrier but not for "effectively
            # continuous" accommodation, so LOW mode bypasses
            # hazard_step entirely and forces sink.active=True
            # permanently (only tau_sink, computed the same physical
            # way, still responds to the instantaneous stress).
            sink.active = True

        step = 0
        z_gb_prev = z1
        meta = dict(lam=lam, z1=z1, a0=a0, Vp0=Vp0, Vs0=Vs0, V0=V0, com0=com0, dt=dt,
                    Nz=geom["Nz"], Nr=geom["Nr"], z_gb_prev=z_gb_prev, psi_deg=psi_deg,
                    sink_state=sink.__dict__, hp_state=hp.__dict__, rng_state=rng.bit_generator.state)
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, default=str)
        row0, z_gb_prev = diagnostics(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam,
                                       Vp0, Vs0, V0, a0, com0, gamma_s, gamma_gb, W, 0, 0.0, sink)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(row0, default=str) + "\n")
        print(f"[init] [{tag}] a0={a0*1e9:.4f}nm dt={dt:.4e}")

    n_steps_total = max(1, int(t_target / dt))
    t_wall0 = time.time()
    last_ckpt = step
    last_diag = step
    sigma_update_every = 20  # recompute the (expensive) neck-fit/stress every N PF steps,
    # holding sigma_now fixed in between -- sigma changes on the slow
    # coarsening timescale (secular drift over t~1-100), NOT on the PF
    # step scale (dt~5e-5), so this is a large wall-clock saving with
    # negligible accuracy cost for the hazard integral; the RBM velocity
    # field (tau_sink) updates on the same cadence.
    sigma_now = 0.0

    while step < n_steps_total:
        f, e1, e2, diag = axisym_gb_face_projected_step(
            f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W, bc_z="noflux")
        step += 1
        if not np.all(np.isfinite(f)):
            print(f"BLOWUP at step {step} -- stopping for review")
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

        if mode == "low":
            update_tau_sink(sink, sigma_now, hp)
        else:
            activated = hazard_step(sink, sigma_now, dt, hp, rng)
            if activated:
                with open(event_path, "a") as fh:
                    fh.write(json.dumps(dict(step=step, t=step * dt, sigma_at_activation=sigma_now)) + "\n")
        f, e1, e2, completed = rbm_step(f, e1, e2, sink, hp, dt, dz, r_c, z, z_gb_prev)
        if mode == "low" and completed:
            sink.active = True  # immediately re-arm (continuous accommodation)

        if step - last_diag >= diag_every_steps:
            last_diag = step
            row, z_gb_prev = diagnostics(f, e1, e2, p, Wc, dr, dz, r_c, r_f, z, z_gb_prev, lam,
                                          Vp0, Vs0, V0, a0, com0, gamma_s, gamma_gb, W, step, step * dt, sink)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            print(f"  [{tag}] step={step} t={row['t']:.3f} a/a0={row['a_over_a0']:.5f} "
                  f"Vp/Vp0={row['Vp_frac']:.6f} densif_strain={row['densification_strain']:.4e} "
                  f"n_events={sink.n_events} active={sink.active} mass_drift={row['mass_drift']:.2e} "
                  f"wall={time.time()-t_wall0:.0f}s")

        if step - last_ckpt >= checkpoint_every_steps:
            last_ckpt = step
            tmp_path = ckpt_path.replace(".npz", "_tmp.npz")
            np.savez_compressed(tmp_path, f=f, e1=e1, e2=e2, step=step, z=z)
            os.replace(tmp_path, ckpt_path)
            meta["z_gb_prev"] = z_gb_prev
            meta["sink_state"] = sink.__dict__
            meta["hp_state"] = hp.__dict__
            meta["rng_state"] = rng.bit_generator.state
            with open(meta_path, "w") as fh:
                json.dump(meta, fh, default=str)

        if target_n_events is not None and sink.n_events >= target_n_events and not sink.active:
            print(f"target_n_events={target_n_events} reached at step={step}, t={step*dt:.3f} -- stopping")
            break
        if max_wall_s is not None and (time.time() - t_wall0) > max_wall_s:
            print(f"max_wall_s={max_wall_s} exhausted at step={step}, t={step*dt:.3f} -- checkpointing and stopping")
            break

    np.savez_compressed(ckpt_path, f=f, e1=e1, e2=e2, step=step, z=z)
    meta["z_gb_prev"] = z_gb_prev
    meta["sink_state"] = sink.__dict__
    meta["hp_state"] = hp.__dict__
    meta["rng_state"] = rng.bit_generator.state
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, default=str)
    print(f"stopped [{tag}] at step={step}, t={step*dt:.4f}, n_events={sink.n_events}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["low", "finite"], required=True)
    ap.add_argument("--t-target", type=float, default=100.0)
    ap.add_argument("--n-sample", type=int, default=200)
    ap.add_argument("--sigma-target-hazard-MPa", type=float, default=2.5)
    ap.add_argument("--v0-mult", type=float, default=10000.0)
    ap.add_argument("--target-n-events", type=int, default=None)
    ap.add_argument("--max-wall-s", type=float, default=None)
    args = ap.parse_args()
    run_regime(args.tag, args.mode, args.t_target, args.n_sample,
               sigma_target_hazard=args.sigma_target_hazard_MPa * 1e6, v0_mult=args.v0_mult,
               target_n_events=args.target_n_events, max_wall_s=args.max_wall_s)
