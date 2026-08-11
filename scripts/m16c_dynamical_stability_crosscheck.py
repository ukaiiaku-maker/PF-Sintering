"""Milestone 16C Section 9 (revised): dynamic PF cross-check of the
static Hessian classification -- the ROBUST version.

The first attempt (m16c_eigenmode_pf_crosscheck.py) tested whether
each BARE energy-Hessian eigenmode grows/decays under PF dynamics
individually. That test is not actually valid in general: the PF
dynamics is a MOBILITY-WEIGHTED gradient flow (d(delta_R)/dt =
-M*H*delta_R schematically, M highly non-uniform since q(f) is
concentrated at the interface), and the eigenmodes of `M*H` are NOT
the same as the eigenmodes of `H` alone unless M is a uniform scalar
multiple of the identity -- confirmed directly: mode 1 (bare-H
eigenvalue +0.95, "stable") in fact GREW substantially under PF
dynamics, while mode 0 ("unstable", eigenvalue -0.136) grew only
marginally. This is a genuine, documented finding, not swept under the
rug.

What DOES robustly transfer from the static to the dynamic picture
(Sylvester's law of inertia: for M SPD, M*H is congruent to
M^(1/2)*H*M^(1/2), a symmetric matrix with the SAME inertia/sign-count
as H) is the OVERALL classification: if H is positive-definite
everywhere (a genuine local energy minimum), NO nearby perturbation
can lower the energy below the base state under any positive-definite
mobility-weighted dynamics; if H has a negative eigenvalue (a saddle),
SOME direction exists that dynamically lowers the energy below the
base state. This script tests exactly that, comparing F(t) under a
small perturbation against the UNPERTURBED base state's own energy
F0_base, for one H-positive-definite case (L=60nm, all 5 lowest
eigenvalues positive) and one H-indefinite case (L=100nm, lowest
eigenvalue negative).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_face_projected_step, axisym_free_energy, r_centers_faces  # noqa: E402
from pf_sintering.sharp_interface_stability import (  # noqa: E402
    energy as sharp_energy, relax_to_equilibrium, total_volume, volume_constrained_eigenmodes,
)

sys.path.insert(0, os.path.dirname(__file__))
from m16c_two_substrate_pf_dynamics import P, build_f_from_profile, find_stable_dt  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16c_campaign")

GAMMA_S = 1.0
R_REF = 40e-9
W_NM = 6.0
NZ = 48


def base_state(L_nm, n_relax=4000):
    """`n_relax` (Milestone 16C Section 9 refinement): for L=120nm the
    FULL relaxation (4000 steps) drives the neck all the way to the
    numerical floor (~0.003nm, i.e. the sharp-interface calculation
    itself finds this configuration collapses to pinch-off) -- useful
    for the static stability MAP (Section 11) but not usable as a
    finite-neck PF initial condition. A PARTIAL relaxation (a smaller
    n_relax) gives an intermediate snapshot that is still clearly
    unstable (a sizable negative eigenvalue) but has a well-resolved,
    finite neck -- exactly what is needed here."""
    V_target = (4.0 / 3.0) * math.pi * R_REF ** 3
    L = L_nm * 1e-9
    dz = L / (NZ - 1)
    z = np.arange(NZ) * dz
    a_guess = 0.4 * R_REF
    Rbody_guess = 1.3 * R_REF
    R = a_guess + (Rbody_guess - a_guess) * np.sin(math.pi * z / L)
    R = R * math.sqrt(V_target / total_volume(R, dz, "capped"))
    Req = relax_to_equilibrium(R, dz, GAMMA_S, 0.0, [0, NZ - 1], "capped", n_steps=n_relax, step_scale=1.0)
    evals, evecs, lam, gnorm = volume_constrained_eigenmodes(Req, dz, GAMMA_S, 0.0, [0, NZ - 1],
                                                               "capped", n_modes=5)
    F0_sharp = sharp_energy(Req, dz, GAMMA_S, 0.0, [0, NZ - 1], "capped")
    return Req, dz, evals, evecs, F0_sharp


def evolve_trajectory(f0, p, dr, dz_pf, r_c, r_f, M_s, dt, sample_steps):
    f = f0.copy()
    Fs = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, mu, diag = axisym_face_projected_step(f, p, dr, dz_pf, r_c, r_f, dt, M_s, p.W, bc_z="noflux")
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"blew up at step {step}")
        Fs.append(axisym_free_energy(f, p, dr, dz_pf, r_c, r_f, bc_z="noflux"))
    return Fs


def run(label, L_nm, eps_frac, t_target, n_sample=25, n_relax=4000):
    """Differential comparison: evolve BOTH the unperturbed base
    profile and the mode0-perturbed profile over the IDENTICAL time
    window and grid, and compare F_perturbed(t)-F_baseline(t) -- this
    isolates the effect of the perturbation alone from the generic
    diffuse-interface-representation "settling" that affects both
    trajectories equally (found necessary after an initial version
    compared against a STATIC F0 reference that was not itself at the
    PF field's own converged local energy minimum -- confirmed
    directly: even the nominally-stable L=60nm case showed F dropping
    ~1.6% below that imprecise static reference, which is a
    discretization-mismatch artifact, not a real instability)."""
    Req, dz, evals, evecs, F0_sharp = base_state(L_nm, n_relax=n_relax)
    L = L_nm * 1e-9
    eps = eps_frac * np.max(Req)
    R_pert = Req + eps * evecs[:, 0]

    f_pert0, r_c, r_f, dr, dz_pf, Nz, Nr = build_f_from_profile(R_pert, L, dz, 1.5, W_NM)
    f_base0, _, _, _, _, _, _ = build_f_from_profile(Req, L, dz, 1.5, W_NM)
    p = P(gamma_s=GAMMA_S)
    M_s = 1e-33

    dt = find_stable_dt(f_base0, p, dr, dz_pf, r_c, r_f, M_s, p.W)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    F_base_traj = evolve_trajectory(f_base0, p, dr, dz_pf, r_c, r_f, M_s, dt, sample_steps)
    F_pert_traj = evolve_trajectory(f_pert0, p, dr, dz_pf, r_c, r_f, M_s, dt, sample_steps)
    ts = [s * dt for s in sample_steps]
    diff = [fp - fb for fp, fb in zip(F_pert_traj, F_base_traj)]

    below = any(d < -1e-20 for d in diff)
    print(f"[{label}] L={L_nm}nm lowest_eig={evals[0]:.4f} "
          f"F_pert-F_base range=[{min(diff):.4e}, {max(diff):.4e}] ever_below_baseline={below}")
    return dict(label=label, L_nm=L_nm, lowest_eig=float(evals[0]), eps_nm=eps * 1e9,
                ever_below_baseline=bool(below),
                rows=[dict(t=t, F_pert=fp, F_base=fb, diff=d)
                      for t, fp, fb, d in zip(ts, F_pert_traj, F_base_traj, diff)])


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "dynamical_stability_crosscheck.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        results = []
        results.append(run("stable_L60", 60.0, eps_frac=0.02, t_target=6.0))
        results.append(run("unstable_L120_partial", 120.0, eps_frac=0.02, t_target=6.0, n_relax=200))
        with open(out_path, "w") as fh:
            json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 9 summary (robust inertia-preservation test) ---")
    for r in results:
        expect_below = r["lowest_eig"] < 0
        ok = r["ever_below_baseline"] == expect_below
        print(f"{r['label']}: lowest_eig={r['lowest_eig']:.4f} ever_below_baseline={r['ever_below_baseline']} "
              f"(expected {expect_below}) PASS={ok}")
