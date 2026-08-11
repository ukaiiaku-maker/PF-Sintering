"""Milestone 16C Sections 9-12: PF dynamic cross-check of the sharp-
interface stability classification, on the one-particle-between-two-
substrates capped-topology geometry.

Uses the NEW `bc_z="noflux"` axisymmetric surface-diffusion path
(pf_sintering/axisym.py, Milestone 16C addition): exact no-flux
boundary at both z-ends instead of periodic wrap, conserving the
particle's own solid volume exactly (no reservoir, no sink, no RBM --
the two ends represent a genuinely rigid, immobile, impenetrable
substrate; the particle's own material can only redistribute within
[0,L]).

The two contacts carry NO GB energy in this specific dynamic check
(gamma_gb=0 at the substrate contacts) -- this is a deliberate, stated
simplification: adding an actively-evolving BOUNDARY GB energy term to
the PF mu field (a Robin-type boundary condition proportional to
gamma_gb*R at each end) is new physics beyond what any prior milestone
built, and is out of scope for this cross-check. Section
m16c_particle_between_substrates_map.py already showed (verified
directly, not assumed) that the SAME stable/unstable transition exists
from pure surface energy alone (classical liquid-bridge-between-plates
Rayleigh instability) -- so the gamma_gb=0 sharp-interface
classification and this gamma_gb=0 PF dynamic check use IDENTICAL
physics, and are directly, consistently comparable.

Section 10 (COM constraint): tracked explicitly -- no sink, no RBM are
present in this code path by construction (same as M16B); the particle
center of mass is tracked to confirm de-sintering does not require it
to move.

Section 12 ("quantify substantial"): reports a(t)/a(0) at each sample
time for the unstable case, requiring order-one recession (targeting
the 1.0/0.8/0.6/0.4 progression), not a few-percent wobble.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_face_projected_step, axisym_free_energy, axisym_volume, r_centers_faces  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16c_campaign")

GAMMA_S = 1.0
R_REF = 40e-9
W_NM = 6.0


class P:
    def __init__(self, gamma_s=1.0, W=W_NM * 1e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W


def build_f_from_profile(R_of_z, L, dz_sharp, dr_nm, W_nm, r_margin_mult=6.0):
    """Builds an axisymmetric (r,z) phase field from a sharp-interface
    R(z) profile (linear interpolation onto the finer PF grid), via the
    same tanh construction used throughout this project."""
    W = W_nm * 1e-9
    dr = dr_nm * 1e-9
    Nz_sharp = len(R_of_z)
    z_sharp = np.arange(Nz_sharp) * dz_sharp
    Nz = Nz_sharp  # keep the same z-resolution as the sharp-interface chain
    dz = dz_sharp
    Rmax = float(np.max(R_of_z))
    Nr = max(24, int(round((Rmax + r_margin_mult * W) / dr)))
    r_c, r_f = r_centers_faces(Nr, dr)
    z = (np.arange(Nz) + 0.0) * dz  # align exactly with the sharp-interface grid points
    R_of_z_grid = np.interp(z, z_sharp, R_of_z)
    Z, Rg = np.meshgrid(z, r_c, indexing="ij")
    Rprofile = np.broadcast_to(R_of_z_grid[:, None], (Nz, Nr))
    f = 0.5 * (1.0 - np.tanh((Rg - Rprofile) / W))
    return f, r_c, r_f, dr, dz, Nz, Nr


def measure_neck_and_com(f, r_c, z, dz):
    """neck radius a(t) = f=0.5 crossing at BOTH ends (z=0 and z=L),
    reported as their average (symmetric problem); particle COM_z via
    the SAME r-weighted measure used throughout this project."""
    def crossing(row):
        idx = np.where((row[:-1] - 0.5) * (row[1:] - 0.5) < 0)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[-1]
        r0v, r1v = r_c[i], r_c[i + 1]
        f0v, f1v = row[i], row[i + 1]
        return r0v + (0.5 - f0v) * (r1v - r0v) / (f1v - f0v)

    a0 = crossing(f[0])
    a1 = crossing(f[-1])
    w = r_c[None, :] * f
    denom = float(np.sum(w))
    com_z = float(np.sum(w * z[:, None])) / denom if denom > 0 else float("nan")
    return 0.5 * (a0 + a1), a0, a1, com_z


def find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, W, n_check=100):
    dt = 1.0
    for _ in range(300):
        f_t = f.copy()
        stable = True
        for _ in range(n_check):
            f_t, _, _ = axisym_face_projected_step(f_t, p, dr, dz, r_c, r_f, dt, M_s, W, bc_z="noflux")
            if not np.all(np.isfinite(f_t)) or np.max(np.abs(f_t)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def run_case(label, R_of_z, L, dz_sharp, t_target, n_sample=30, dr_nm=1.5):
    f, r_c, r_f, dr, dz, Nz, Nr = build_f_from_profile(R_of_z, L, dz_sharp, dr_nm, W_NM)
    p = P(gamma_s=GAMMA_S)
    M_s = 1e-33
    z = np.arange(Nz) * dz

    dt = find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, p.W)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    neck0, a0_left, a0_right, com0 = measure_neck_and_com(f, r_c, z, dz)

    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, mu, diag = axisym_face_projected_step(f, p, dr, dz, r_c, r_f, dt, M_s, p.W, bc_z="noflux")
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"[{label}] blew up at step {step}")
        neck, a_left, a_right, com_z = measure_neck_and_com(f, r_c, z, dz)
        V = axisym_volume(f, r_c, dr, dz)
        F = axisym_free_energy(f, p, dr, dz, r_c, r_f, bc_z="noflux")
        rows.append(dict(step=step, t=step * dt, neck_nm=neck * 1e9, a_left_nm=a_left * 1e9,
                          a_right_nm=a_right * 1e9, com_z_nm=com_z * 1e9,
                          com_shift_nm=(com_z - com0) * 1e9, a_over_a0=neck / neck0,
                          mass_drift=(V - V0) / V0, F=F))
        print(f"  [{label}] step={step} t={rows[-1]['t']:.3e} neck={neck*1e9:.4f}nm "
              f"a/a0={rows[-1]['a_over_a0']:.4f} com_shift={rows[-1]['com_shift_nm']:.4f}nm "
              f"mass_drift={rows[-1]['mass_drift']:.2e} F={F:.4e}")

    return dict(label=label, Nz=Nz, Nr=Nr, dr_nm=dr * 1e9, dz_nm=dz * 1e9, dt=dt,
                n_steps_total=n_steps_total, neck0_nm=neck0 * 1e9, com0_nm=com0 * 1e9, rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "two_substrate_pf_dynamics.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        with open(os.path.join(CAMPAIGN_DIR, "particle_between_substrates_map_gamma_gb0.json")) as fh:
            sharp = json.load(fh)
        by_L = {r["L_nm"]: r for r in sharp}

        results = []
        cases = [("stable_L60", 60.0, 6.0), ("near_neutral_L80", 80.0, 3.0), ("unstable_L120", 120.0, 3.0)]
        for label, L_nm, t_target in cases:
            r_sharp = by_L[L_nm]
            R_of_z = np.array(r_sharp["R_profile_nm"]) * 1e-9
            dz_sharp = r_sharp["dz_nm"] * 1e-9
            res = run_case(label, R_of_z, L_nm * 1e-9, dz_sharp, t_target=t_target, n_sample=30)
            res["L_nm"] = L_nm
            res["sharp_lowest_eig"] = r_sharp["lowest_eig"]
            results.append(res)
            with open(out_path, "w") as fh:
                json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Sections 9-12 summary ---")
    for r in results:
        final = r["rows"][-1]
        print(f"{r['label']}: sharp_lowest_eig={r['sharp_lowest_eig']:.4e} "
              f"neck: {r['neck0_nm']:.3f}nm -> {final['neck_nm']:.3f}nm "
              f"(a/a0={final['a_over_a0']:.4f}) com_shift={final['com_shift_nm']:.4f}nm "
              f"max_mass_drift={max(abs(rr['mass_drift']) for rr in r['rows']):.2e}")
