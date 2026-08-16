"""M16N Section H: post-event blowup diagnostic.

Takes the field state IMMEDIATELY after one completed sink event
(constructed the same way as scripts/m16n_transport_only_microtest.py's
Phase 1, using the CORRECTED delta_sink-driven completion contract), then
runs PF-ONLY continuations (no hazard, no further RBM) at dt, dt/2,
dt/4, dt/8 -- the SAME dt the production driver was using before the
legacy-control run's post-event blowup (t=4.8678, ~9000 steps /
~0.446 time-units after the event completed) -- to determine whether that
blowup is simply the pre-event dt no longer being numerically stable
immediately after the RBM field remap.

Tracks max/min f, max|mu_f_gb| (chemical-potential proxy), free energy,
mass drift, and the first nonfinite step (if any) at each dt, out to a
MATCHED total time (0.5 model-time-units, comfortably past where the
real blowup occurred).
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_mu_f_gb, axisym_volume  # noqa: E402
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams, active_sink_transport_step  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402

sys.path.insert(0, "scripts")
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402

PSI_DEG, GAMMA_S = 160.0, 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM, RATIO, W_NM, DX_NM = 1000.0, 0.10, 10.0, 1.25
B = 2.5e-10
V0 = 12.5 * B ** 3
A0_J = 0.8589 * 1.602176634e-19
GS = 201.74e-9
R0 = 1e12
T = 1000.0
SIGMA_FIXED = 45.126e6
T_TARGET = 0.5  # comfortably past the real blowup's ~0.446 post-event interval

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16n_post_event_blowup_diagnostic")


def build_post_event_state():
    d = np.load("runs/m16k_prescribed_displacement_microtest/frozen_state.npz")
    f0, e1_0, e2_0 = d["f"], d["e1"], d["e2"]
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B, D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=A0_J, V0=V0, tau_ex0=0.0)
    roles = roles_from_m16j_geometry(e1_0, e2_0)
    f, particle, substrate = f0.copy(), roles.particle.copy(), roles.substrate.copy()
    sink = AxisymSink(active=True, current_disp=0.0)
    dt = 4.8828e-05
    completed = False
    while not completed:
        f, particle, substrate, completed, diag = active_sink_transport_step(
            f, particle, substrate, sink, hp, SIGMA_FIXED, dt, dz, r_c, z, 0.0)
    return f, particle, substrate


def run_dt(dt, f0, particle0, substrate0):
    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    W = W_NM * 1e-9
    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    f, particle, substrate = f0.copy(), particle0.copy(), substrate0.copy()
    V_ref = axisym_volume(f, r_c, dr, dz)
    n_steps = int(round(T_TARGET / dt))
    sample_every = max(1, n_steps // 40)
    rows = []
    first_nonfinite = None
    t0 = time.time()
    for step in range(n_steps + 1):
        if step > 0:
            f, particle, substrate, _ = axisym_gb_face_projected_step(f, particle, substrate, p, Wc, dr, dz, r_c,
                                                                        r_f, dt, M_s, M_eta, W, bc_z="noflux")
        if not np.all(np.isfinite(f)):
            first_nonfinite = step
            break
        if step % sample_every != 0:
            continue
        mu = axisym_mu_f_gb(f, particle, substrate, p, Wc, dr, dz, r_c, r_f, bc_z="noflux")
        F = axisym_free_energy_gb(f, particle, substrate, p, Wc, dr, dz, r_c, r_f, bc_z="noflux")
        V = axisym_volume(f, r_c, dr, dz)
        rows.append(dict(step=step, time=step * dt, f_max=float(np.max(f)), f_min=float(np.min(f)),
                          mu_abs_max=float(np.max(np.abs(mu))) if np.all(np.isfinite(mu)) else float("nan"),
                          free_energy=F, mass_drift=(V - V_ref) / V_ref))
    wall = time.time() - t0
    print(f"  [dt={dt:.4e}] n_steps={n_steps} first_nonfinite={first_nonfinite} wall={wall:.0f}s "
          f"final_mass_drift={rows[-1]['mass_drift'] if rows else float('nan'):.4e} "
          f"final_mu_max={rows[-1]['mu_abs_max'] if rows else float('nan'):.4e}", flush=True)
    return rows, first_nonfinite


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Constructing post-event state (corrected delta_sink-driven completion)...")
    f0, particle0, substrate0 = build_post_event_state()
    print("Post-event state ready. Running PF-only continuations at dt, dt/2, dt/4, dt/8...")

    dt_base = 4.8828e-05
    results = {}
    for div in [1, 2, 4, 8]:
        dt = dt_base / div
        print(f"\n--- dt = dt_base/{div} = {dt:.4e} ---")
        rows, first_nonfinite = run_dt(dt, f0, particle0, substrate0)
        results[div] = dict(dt=dt, rows=rows, first_nonfinite=first_nonfinite)
        out_path = os.path.join(OUT, f"dt_base_over_{div}.csv")
        if rows:
            with open(out_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    print("\n=== SUMMARY ===")
    for div, res in results.items():
        status = f"BLOWUP at step {res['first_nonfinite']} (t={res['first_nonfinite']*res['dt']:.4f})" \
            if res["first_nonfinite"] is not None else f"STABLE through t={T_TARGET}"
        print(f"  dt_base/{div} (dt={res['dt']:.4e}): {status}")
    print("\nDONE")


if __name__ == "__main__":
    main()
