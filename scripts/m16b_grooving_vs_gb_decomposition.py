"""Milestone 16B Section 15: separate thermal grooving from GB
translation on the Hussein one-mode geometry (Section 13-14).

Three controls, same one-mode IC (build_hussein_one_mode, psi=120 deg
fixed) and same short time window:
  A: M_GB=0 (eta frozen -- pass M_eta=0 to axisym_gb_face_projected_step),
     surface diffusion active -- isolates PURE thermal grooving (the
     GB acts only as a fixed local energy/Wc anomaly, no GB motion).
  B: physical M_GB>0, surface diffusion active -- the full coupled
     production case (same as Section 14's run).
  C: GB-only short diagnostic (M_s=0, M_GB>0) -- isolates PURE GB
     migration with no surface transport at all (mostly a sanity check
     that eta kinetics alone cannot move mass/change R(z), since R(z)
     is entirely a property of the conserved f field, not eta).

Reports R_norm(t) for A and B side by side, and confirms C leaves R(z)
unchanged (mass_drift and R_of_z both static) -- i.e. any change in
R_norm under B beyond what A alone produces is attributable to the GB
migration's effect on the LOCAL Wc/coupling term feeding back into f's
own mu (not to any direct eta->f mass transfer, since there is none).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z, psi_to_gamma_gb  # noqa: E402
from m16b_hussein_benchmark import P, build_hussein_one_mode, find_stable_dt  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16b_campaign")


def run_control(label, M_s_scale, M_eta_scale, R0_nm=40.0, W_nm=20.0, dx_nm=2.5, psi_deg=120.0,
                 t_target=2.0, n_sample=15):
    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    Nr = max(24, round((R0 + 6 * W) / dr))
    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]
    f, e1, e2, L, lam = build_hussein_one_mode(10, Nr, dr, dz, R0, W, eps1_bar=0.2)
    Nz = f.shape[0]
    r_c, r_f = r_centers_faces(Nr, dr)

    M_s_base = 1e-33
    M_eta_base = 1e-33 / (W * (32.0 / 35.0))
    M_s = M_s_base * M_s_scale
    M_eta = M_eta_base * M_eta_scale

    # find_stable_dt needs at least one nonzero mobility to probe
    # against; for the (M_s=0) GB-only control, dt is instead governed
    # purely by the eta relaxation -- use the base M_s>0 dt-finder
    # result scaled up generously (halved further) as a safe fallback
    # if M_s=0 makes the f-step trivially stable at any dt.
    if M_s > 0:
        dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, max(M_eta, M_eta_base), W)
    else:
        dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s_base, max(M_eta, M_eta_base), W)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    R_of_z0 = measure_R_of_z(f, r_c)
    j_gb, j_mid = int(round(L / 4.0 / dz)), 0
    ratio0 = R_of_z0[j_gb] / R_of_z0[j_mid]

    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"[{label}] blew up at step {step}")
        R_of_z = measure_R_of_z(f, r_c)
        V = axisym_volume(f, r_c, dr, dz)
        ratio = R_of_z[j_gb] / R_of_z[j_mid]
        rows.append(dict(step=step, t=step * dt, R_norm=ratio / ratio0, mass_drift=(V - V0) / V0,
                          R_max_change_nm=(R_of_z[j_mid] - R_of_z0[j_mid]) * 1e9))
    final = rows[-1]
    print(f"[{label}] M_s={M_s:.2e} M_eta={M_eta:.2e} dt={dt:.3e} n_steps={n_steps_total} "
          f"R_norm(0)=1.0 -> R_norm(final)={final['R_norm']:.5f} mass_drift={final['mass_drift']:.2e} "
          f"R_mid_change={final['R_max_change_nm']:.4f}nm")
    return dict(label=label, M_s=M_s, M_eta=M_eta, dt=dt, n_steps_total=n_steps_total, rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "grooving_vs_gb.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        results = []
        results.append(run_control("A_MGB0_surfON", M_s_scale=1.0, M_eta_scale=0.0))
        results.append(run_control("B_physical", M_s_scale=1.0, M_eta_scale=1.0))
        results.append(run_control("C_GBonly", M_s_scale=0.0, M_eta_scale=1.0))
        with open(out_path, "w") as fh:
            json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 15 summary ---")
    for r in results:
        final = r["rows"][-1]
        print(f"{r['label']}: R_norm(final)={final['R_norm']:.5f} mass_drift={final['mass_drift']:.2e}")
