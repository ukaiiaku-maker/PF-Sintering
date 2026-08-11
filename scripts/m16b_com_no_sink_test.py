"""Milestone 16B Section 18: center-of-mass/no-sink test on the
axisymmetric M15-flat-substrate analogue (Section 17's
axisym_m15_flat_geometry).

Sink OFF, RBM (rigid-body-motion) OFF -- both are automatically true
for every function in pf_sintering/axisym.py: this module has never had
a sink term or a rigid-body-translation mechanism at any point in
Milestones 16A/16B, it is a fixed-grid conservative-PDE solver only.
This script's job is to VERIFY that meaningful neck/surface dynamics
(the physical process sink/RBM would otherwise be invoked to produce)
occurs from face-projected surface diffusion + tangent-cone GB kinetics
ALONE, while (a) total solid volume is conserved to machine precision
(already proven for axisym_gb_face_projected_step, re-confirmed here on
this specific geometry) and (b) the particle's center of mass moves by
only a small amount relative to its own size -- i.e. PR/de-sintering-
type shape evolution does not REQUIRE large-scale COM translation to
occur, distinguishing genuine capillary-driven neck evolution from an
artificial rigid shift.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_m15_com_z, axisym_m15_flat_geometry,
    axisym_m15_flat_volumes, axisym_reproject, axisym_volume, r_centers_faces,
)
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16b_campaign")


class P:
    def __init__(self, gamma_s=1.0, gamma_gb=0.6, W=10e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W
        self.k_eta = gb_obstacle_coefficients(gamma_gb, W)["k_eta"]


def measure_neck_radius(f, r_c, z, wall_z, dz):
    """f=0.5 crossing radius (in r) at the z-row closest to the
    substrate wall plane -- a simple, direct proxy for contact/neck
    size on this geometry (the analogue of `contact_width` diagnostics
    used throughout the Cartesian M15 series)."""
    j = int(round(wall_z / dz))
    row = f[j]
    idx = np.where((row[:-1] - 0.5) * (row[1:] - 0.5) < 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[-1]
    r0v, r1v = r_c[i], r_c[i + 1]
    f0v, f1v = row[i], row[i + 1]
    return r0v + (0.5 - f0v) * (r1v - r0v) / (f1v - f0v)


def find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=200):
    dt = 1.0
    for _ in range(300):
        f_t, e1_t, e2_t = f.copy(), e1.copy(), e2.copy()
        stable = True
        for _ in range(n_check):
            f_t, e1_t, e2_t, _ = axisym_gb_face_projected_step(f_t, e1_t, e2_t, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            if not np.all(np.isfinite(f_t)) or np.max(np.abs(f_t)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def run(Nz=200, Nr=120, dx_nm=2.0, W_nm=10.0, gamma_gb=0.6, t_target=3.0, n_sample=30):
    dr = dz = dx_nm * 1e-9
    W = W_nm * 1e-9
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]

    geom = axisym_m15_flat_geometry(Nz, Nr, dz, dr, W)
    f = geom["f"]
    e1, e2 = axisym_reproject(f, geom["e1_raw"].copy(), geom["e2_raw"].copy())
    r_c, r_f = r_centers_faces(Nr, dr)
    z = (np.arange(Nz) + 0.5) * dz
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    vols0 = axisym_m15_flat_volumes(geom, r_c, dr, dz)
    print(f"initial volumes: {vols0}")

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))

    V0 = axisym_volume(f, r_c, dr, dz)
    com_z0 = axisym_m15_com_z(e2, z, r_c, dr, dz)
    neck0 = measure_neck_radius(f, r_c, z, geom["wall_z"], dz)

    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))
    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"blew up at step {step}")
        V = axisym_volume(f, r_c, dr, dz)
        com_z = axisym_m15_com_z(e2, z, r_c, dr, dz)
        neck = measure_neck_radius(f, r_c, z, geom["wall_z"], dz)
        F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
        rows.append(dict(step=step, t=step * dt, mass_drift=(V - V0) / V0,
                          com_z_shift_nm=(com_z - com_z0) * 1e9, neck_r_nm=neck * 1e9, F=F))
        print(f"  step={step} t={rows[-1]['t']:.3e} mass_drift={rows[-1]['mass_drift']:.2e} "
              f"com_z_shift={rows[-1]['com_z_shift_nm']:.4f}nm neck_r={rows[-1]['neck_r_nm']:.4f}nm F={F:.4e}")

    Rr_nm = geom["Rr"] * 1e9
    final = rows[-1]
    com_shift_frac = abs(final["com_z_shift_nm"]) / Rr_nm
    neck_change_frac = abs(final["neck_r_nm"] - neck0 * 1e9) / (neck0 * 1e9)

    return dict(Nz=Nz, Nr=Nr, dx_nm=dx_nm, W_nm=W_nm, gamma_gb=gamma_gb, dt=dt,
                n_steps_total=n_steps_total, com_z0_nm=com_z0 * 1e9, neck0_nm=neck0 * 1e9,
                Rr_nm=Rr_nm, com_shift_frac=com_shift_frac, neck_change_frac=neck_change_frac,
                max_mass_drift=max(abs(r["mass_drift"]) for r in rows), rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "com_no_sink.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            result = json.load(fh)
    else:
        result = run()
        with open(out_path, "w") as fh:
            json.dump(result, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 18 summary ---")
    print(f"max |mass_drift| = {result['max_mass_drift']:.3e} (must be ~machine precision)")
    print(f"neck radius: {result['neck0_nm']:.4f}nm -> "
          f"{result['rows'][-1]['neck_r_nm']:.4f}nm ({result['neck_change_frac']*100:.2f}% change)")
    print(f"particle COM_z shift: {result['rows'][-1]['com_z_shift_nm']:.4f}nm "
          f"({result['com_shift_frac']*100:.3f}% of equatorial radius Rr={result['Rr_nm']:.2f}nm)")
    ok = result["max_mass_drift"] < 1e-8 and result["com_shift_frac"] < 0.05 and result["neck_change_frac"] > 0.01
    print(f"PASS (volume conserved, COM shift small, neck genuinely evolved without RBM/sink) = {ok}")
