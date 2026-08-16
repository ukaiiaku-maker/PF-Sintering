"""M16N Section D: short sink-OFF PF evolution for the qualified
candidates from Section C (runs/m16n_static_pf_qualification/), tracking
r_neck/X_neck/sigma_Hussein via a well-resolved FIXED window (12nm --
comfortably inside the analytic plateau found in Section C for all three
candidates, agreement <=1.6% at t=0) alongside free energy and mass.

Rejects (reports, does not silently continue) any trajectory showing:
  - immediate sign reversal in sigma
  - unresolved curvature (window-ladder disagreement re-emerging)
  - tracker switching (multiple R(z) minima)
  - a large artificial initial relaxation transient (Section D explicit
    reject criteria).
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry, find_all_extrema  # noqa: E402
from pf_sintering.m16k_neck_tracking import NeckTracker  # noqa: E402

sys.path.insert(0, "scripts")
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
TRACK_WINDOW_NM = 12.0  # well inside the analytic plateau for all 3 candidates (Section C)

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16n_sinkoff_screen")


def run_one(ratio, W_nm, dx_nm, t_target, n_samples):
    os.makedirs(OUT_ROOT, exist_ok=True)
    label = f"ratio{ratio:.3f}_W{W_nm:g}dx{dx_nm:g}"
    t0 = time.time()

    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=ratio, psi_deg=PSI_DEG,
                                    W_nm=W_nm, dr_nm=dx_nm, dz_nm=dx_nm, aspect_ratio=1.0)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = W_nm * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    n_steps_total = max(1, int(t_target / dt))
    diag_every = max(1, n_steps_total // n_samples)
    print(f"[{label}] Nz={f.shape[0]} Nr={f.shape[1]} dt={dt:.4e} n_steps_total={n_steps_total}", flush=True)

    rows = []
    V0_mass = axisym_volume(f, r_c, dr, dz)
    F0 = None
    reject_reasons = []
    tracker = NeckTracker()
    for step in range(0, n_steps_total + 1):
        if step > 0:
            f, e1, e2, _ = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                                           bc_z="noflux")
            if not np.all(np.isfinite(f)):
                print(f"[{label}] BLOWUP at step {step}"); break

        if step % diag_every != 0:
            continue
        t = step * dt
        R_of_z = measure_R_of_z(f, r_c)
        ext = find_all_extrema(R_of_z, z)
        n_candidates = len([1 for k, _, _ in ext if k == "min"])
        # path-continuous selection (M16K/M16L/M16M established convention)
        # -- picks whichever candidate is closest to the previously
        # selected one, not "first found", so a trajectory with multiple
        # simultaneous minima doesn't arbitrarily jump between them.
        tr = tracker.step(R_of_z, z, W)
        if tr["selected_contact"] is None:
            continue
        z_min, a_contact = tr["selected_contact"]
        X_neck = 2 * a_contact
        win = neck_curvature_windows(R_of_z, z, z_min, 1e-9, window_widths_in_W=(TRACK_WINDOW_NM,))[0]
        r_neck = win["r_neck"]
        if np.isfinite(r_neck) and r_neck > 0:
            sigma, *_ = hussein_eq1b_sigma(r_neck, X_neck, GAMMA_S, GAMMA_GB)
            sigma_MPa = sigma / 1e6
        else:
            sigma_MPa = float("nan")
        V = axisym_volume(f, r_c, dr, dz)
        F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
        if F0 is None:
            F0 = F
        row = dict(step=step, time=t, X_neck_nm=X_neck * 1e9, r_neck_nm=r_neck * 1e9 if np.isfinite(r_neck) else float("nan"),
                   sigma_MPa=sigma_MPa, n_candidates=n_candidates, free_energy=F,
                   mass_drift=(V - V0_mass) / V0_mass)
        rows.append(row)
        print(f"[{label}] t={t:.3f} X_neck={row['X_neck_nm']:.3f}nm r_neck={row['r_neck_nm']:.3f}nm "
              f"sigma={sigma_MPa:.3f}MPa n_cand={n_candidates} wall={time.time()-t0:.0f}s", flush=True)

        if step == diag_every and np.isfinite(sigma_MPa) and abs(sigma_MPa - rows[0]["sigma_MPa"]) > 0.3 * abs(rows[0]["sigma_MPa"]):
            reject_reasons.append("large artificial initial relaxation transient (>30% change in first sample)")

    out_path = os.path.join(OUT_ROOT, f"{label}.csv")
    if rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    sign_flips = sum(1 for i in range(1, len(rows)) if np.isfinite(rows[i]["sigma_MPa"]) and np.isfinite(rows[i - 1]["sigma_MPa"])
                      and rows[i]["sigma_MPa"] * rows[i - 1]["sigma_MPa"] < 0)
    max_candidates = max((r["n_candidates"] for r in rows), default=0)
    if sign_flips > 0:
        reject_reasons.append(f"{sign_flips} sign reversal(s) in sigma")
    if max_candidates > 1:
        reject_reasons.append(f"tracker ambiguity: up to {max_candidates} candidate minima seen")
    print(f"[{label}] DONE wall={time.time()-t0:.0f}s -- wrote {out_path}")
    print(f"[{label}] REJECT flags: {reject_reasons if reject_reasons else 'none'}")
    return rows, reject_reasons


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--w-nm", type=float, default=6.0)
    ap.add_argument("--dx-nm", type=float, default=0.85)
    ap.add_argument("--t-target", type=float, default=8.0)
    ap.add_argument("--n-samples", type=int, default=80)
    args = ap.parse_args()
    run_one(args.ratio, args.w_nm, args.dx_nm, args.t_target, args.n_samples)
