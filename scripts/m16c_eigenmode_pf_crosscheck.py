"""Milestone 16C Section 9: dynamic PF linear-stability cross-check.

For the lowest two Hessian eigenmodes of the L=100nm particle-between-
two-substrates base state (Section 11; mode 0 has eigenvalue -0.136,
UNSTABLE; mode 1 has eigenvalue +0.950, STABLE), initialize the
production-quality PF solver at `R_relaxed +/- epsilon*eigenmode`,
evolve, and project the PF-measured R(z,t) profile back onto the
eigenmode direction to get a scalar amplitude a_n(t). Requires
|a_n(t)| to GROW for the energy-negative mode and DECAY for the
energy-positive mode -- connecting the static sharp-interface
criterion to the kinetic PF implementation, as required.

(The base state at L=100nm was found, empirically, to have its LOWEST
eigenmode be an antisymmetric "seesaw" shape rather than a simple
symmetric neck-pinch -- confirmed directly by printing the eigenvector,
not assumed -- so a purely symmetric neck perturbation would not
excite it at all. This script perturbs along the ACTUAL computed
eigenvector, whatever its shape, which is the correct and general way
to do this cross-check regardless of mode symmetry.)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.sharp_interface_stability import (  # noqa: E402
    relax_to_equilibrium, total_volume, volume_constrained_eigenmodes,
)

sys.path.insert(0, os.path.dirname(__file__))
from m16c_two_substrate_pf_dynamics import P, build_f_from_profile, find_stable_dt  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16c_campaign")

GAMMA_S = 1.0
R_REF = 40e-9
W_NM = 6.0


def measure_R_profile(f, r_c, z, dz):
    Nz = len(z)
    out = np.full(Nz, np.nan)
    for j in range(Nz):
        row = f[j]
        idx = np.where((row[:-1] - 0.5) * (row[1:] - 0.5) < 0)[0]
        if len(idx) == 0:
            continue
        i = idx[-1]
        r0v, r1v = r_c[i], r_c[i + 1]
        f0v, f1v = row[i], row[i + 1]
        out[j] = r0v + (0.5 - f0v) * (r1v - r0v) / (f1v - f0v)
    return out


def run_mode_check(label, R_base, mode_vec, eig, dz_sharp, eps_frac, t_target, n_sample=20):
    L = (len(R_base) - 1) * dz_sharp
    eps = eps_frac * np.max(R_base)
    R_pert = R_base + eps * mode_vec
    f, r_c, r_f, dr, dz, Nz, Nr = build_f_from_profile(R_pert, L, dz_sharp, 1.5, W_NM)
    p = P(gamma_s=GAMMA_S)
    M_s = 1e-33
    z = np.arange(Nz) * dz

    dt = find_stable_dt(f, p, dr, dz, r_c, r_f, M_s, p.W)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    mode_hat = mode_vec / np.linalg.norm(mode_vec)
    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, mu, diag = axisym_face_projected_step(f, p, dr, dz, r_c, r_f, dt, M_s, p.W, bc_z="noflux")
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"[{label}] blew up at step {step}")
        R_now = measure_R_profile(f, r_c, z, dz)
        a_n = float(np.dot(R_now - R_base, mode_hat))
        rows.append(dict(step=step, t=step * dt, a_n_nm=a_n * 1e9))

    print(f"[{label}] eig={eig:.4f} eps={eps*1e9:.4f}nm  a_n(0)={rows[0]['a_n_nm']:.5f}nm "
          f"a_n(final)={rows[-1]['a_n_nm']:.5f}nm  ratio={rows[-1]['a_n_nm']/rows[0]['a_n_nm']:.4f}")
    return dict(label=label, eig=eig, eps_nm=eps * 1e9, rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "eigenmode_pf_crosscheck.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        gamma_gb = 0.0
        V_target = (4.0 / 3.0) * math.pi * R_REF ** 3
        Nz = 48
        L = 100e-9
        dz = L / (Nz - 1)
        z = np.arange(Nz) * dz
        a_guess = 0.4 * R_REF
        Rbody_guess = 1.3 * R_REF
        R = a_guess + (Rbody_guess - a_guess) * np.sin(math.pi * z / L)
        R = R * math.sqrt(V_target / total_volume(R, dz, "capped"))
        Req = relax_to_equilibrium(R, dz, GAMMA_S, gamma_gb, [0, Nz - 1], "capped", n_steps=4000, step_scale=1.0)
        evals, evecs, lam, gnorm = volume_constrained_eigenmodes(Req, dz, GAMMA_S, gamma_gb, [0, Nz - 1],
                                                                   "capped", n_modes=2)
        print(f"base state L={L*1e9}nm eigenvalues: {evals}")

        results = []
        results.append(run_mode_check("mode0_unstable_plus", Req, evecs[:, 0], evals[0], dz,
                                       eps_frac=0.01, t_target=6.0))
        results.append(run_mode_check("mode0_unstable_minus", Req, -evecs[:, 0], evals[0], dz,
                                       eps_frac=0.01, t_target=6.0))
        results.append(run_mode_check("mode1_stable_plus", Req, evecs[:, 1], evals[1], dz,
                                       eps_frac=0.01, t_target=6.0))
        with open(out_path, "w") as fh:
            json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 9 summary ---")
    for r in results:
        a0 = r["rows"][0]["a_n_nm"]
        af = r["rows"][-1]["a_n_nm"]
        grew = abs(af) > abs(a0)
        expect_grow = r["eig"] < 0
        ok = grew == expect_grow
        print(f"{r['label']}: eig={r['eig']:.4f} |a_n|: {abs(a0):.5f} -> {abs(af):.5f} "
              f"{'GREW' if grew else 'DECAYED'} (expected {'GROW' if expect_grow else 'DECAY'}) PASS={ok}")
