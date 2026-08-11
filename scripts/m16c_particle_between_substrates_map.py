"""Milestone 16C Section 11: sharp-interface stability map for the
one-particle-between-two-substrates sanity case.

A particle of FIXED volume V (set to the volume of an R=40nm sphere, an
arbitrary but concrete reference scale) is bonded to two flat substrates
separated by axial distance L, forming two GB contacts (the "capped"
topology in sharp_interface_stability.py: both ends of the R(z) chain
are GB disks, not free surface). For each L, an initial barrel-shaped
guess is relaxed to a genuine constrained energy critical point
(relax_to_equilibrium), then classified via the constrained-Hessian
eigenmode analysis (lambda_min<0 unstable, ~0 neutral, >0 stable).

Increasing L at fixed V thins the particle (larger L/V^(1/3), more
slender) -- analogous to the classical liquid-bridge-between-plates
slenderness instability, and to a bamboo grain becoming more
PR-unstable as its aspect ratio grows.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.sharp_interface_stability import (  # noqa: E402
    relax_to_equilibrium, total_volume, volume_constrained_eigenmodes,
)

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16c_campaign")

GAMMA_S = 1.0
GAMMA_GB = 0.6
R_REF = 40e-9
V_TARGET = (4.0 / 3.0) * math.pi * R_REF ** 3
NZ = 48


def build_and_relax(L, a_frac=0.4, n_steps=4000):
    dz = L / (NZ - 1)
    z = np.arange(NZ) * dz
    a_guess = a_frac * R_REF
    Rbody_guess = 1.3 * R_REF
    R = a_guess + (Rbody_guess - a_guess) * np.sin(math.pi * z / L)
    V0 = total_volume(R, dz, "capped")
    R = R * math.sqrt(V_TARGET / V0)
    Req = relax_to_equilibrium(R, dz, GAMMA_S, GAMMA_GB, [0, NZ - 1], "capped",
                                n_steps=n_steps, step_scale=1.0)
    return Req, dz


def classify(L, n_relax=4000):
    Req, dz = build_and_relax(L, n_steps=n_relax)
    evals, evecs, lam_lagrange, gnorm = volume_constrained_eigenmodes(
        Req, dz, GAMMA_S, GAMMA_GB, [0, NZ - 1], "capped", n_modes=3)
    neck_a = float(Req[0])
    body_R = float(np.max(Req))
    return dict(L_nm=L * 1e9, lowest_eig=float(evals[0]), second_eig=float(evals[1]),
                neck_a_nm=neck_a * 1e9, body_R_nm=body_R * 1e9,
                slenderness=L / V_TARGET ** (1.0 / 3.0), R_profile_nm=(Req * 1e9).tolist(),
                dz_nm=dz * 1e9)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "particle_between_substrates_map.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        L_values_nm = [40, 60, 80, 100, 120, 140, 160, 180, 200]
        results = []
        for L_nm in L_values_nm:
            r = classify(L_nm * 1e-9)
            results.append(r)
            print(f"L={r['L_nm']:.1f}nm slenderness={r['slenderness']:.4f} "
                  f"neck_a={r['neck_a_nm']:.3f}nm body_R={r['body_R_nm']:.3f}nm "
                  f"lowest_eig={r['lowest_eig']:.4e} second_eig={r['second_eig']:.4e}")
        with open(out_path, "w") as fh:
            json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 11 stability map ---")
    for r in results:
        cls = "UNSTABLE" if r["lowest_eig"] < -1e-3 else ("STABLE" if r["lowest_eig"] > 1e-3 else "NEUTRAL")
        print(f"L={r['L_nm']:6.1f}nm  neck_a={r['neck_a_nm']:6.3f}nm  "
              f"lowest_eig={r['lowest_eig']:+.4e}  {cls}")
