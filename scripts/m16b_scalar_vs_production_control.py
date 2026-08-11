"""Milestone 16B Section 6: scalar-vs-production axisymmetric control.

Repeats the bounded M16A PR ladder (lambda/R0 in {5.5, 2*pi, 8}) with
BOTH the M16A scalar-mobility operator (axisym_surface_diffusion_step)
and the new Section 3-5 face-projected (tangentially-projected,
tensor-mobility) operator (axisym_face_projected_step), at the SAME
R0/W/dx/M_s/eps0. Requirement (Section 6): same stability SIGN, same
neutral location -- the kinetic prefactor (omega magnitude) is allowed
to differ since face-projection changes the effective mobility
geometry, but the underlying PR physics (governed by the free energy
and the conservative divergence, not by the tangential-projection
detail) must not flip which wavelengths grow vs decay.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_face_projected_step, axisym_free_energy, axisym_surface_diffusion_step, axisym_volume,
    r_centers_faces,
)

sys.path.insert(0, os.path.dirname(__file__))
from m16a_pr_benchmark import amplitude, build_rod  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16b_campaign")


class P:
    def __init__(self, gamma_s=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W


def find_stable_dt(step_fn, f, p, dr, dz, r_c, r_f, M_s, W, n_check=2000):
    """n_check=2000 (not M16A's original 60): a marginally-stable dt for
    the tensor-mobility face-projected operator can pass a short check
    but blow up after O(10^4) steps (found empirically -- an n_check=60
    threshold produced NaNs partway through a 51200-step trajectory)."""
    dt = 1.0
    for _ in range(300):
        f_try = f.copy()
        stable = True
        for _ in range(n_check):
            f_try, _, _ = step_fn(f_try, p, dr, dz, r_c, r_f, dt, M_s, W)
            if not np.all(np.isfinite(f_try)) or np.max(np.abs(f_try)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def run_one(step_fn, label, R0_nm, lam_over_R0, dx_nm, W_nm, eps0_frac, t_target, n_sample,
            fit_frac=0.5, dt_safety=0.5):
    """Sizes the step budget from a TARGET PHYSICAL TIME (not a fixed
    step count) -- a fixed count badly under-resolves the linear PR
    regime once dt varies across operators/resolutions (the same
    methodology bug M16A found and fixed in
    m16a_stage_convergence2.py). omega is fit on the FIRST `fit_frac`
    of the trajectory (matching M16A's own validated
    m16a_fit_growth.py convention: the EARLY window is the linear
    regime; late times saturate/go nonlinear once amp is no longer
    small, which is what produced sign flips in an earlier version of
    this script that fit the LAST quarter instead)."""
    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    lam = lam_over_R0 * R0
    Nz = max(16, round(lam / dz))
    lam = Nz * dz
    Nr = max(24, round((R0 + 6 * W) / dr))
    eps0 = eps0_frac * R0

    p = P(gamma_s=1.0, W=W)
    f, lam_actual = build_rod(Nz, Nr, dr, dz, R0, W, eps0, n_periods=1)
    r_c, r_f = r_centers_faces(Nr, dr)
    M_s = 1e-33

    dt = find_stable_dt(step_fn, f, p, dr, dz, r_c, r_f, M_s, W)
    dt *= dt_safety
    n_steps_total = max(1, int(t_target / dt))

    V0 = axisym_volume(f, r_c, dr, dz)
    amp0, _ = amplitude(f, r_c, dr)

    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))
    ts, amps, drifts = [], [], []
    step = 0
    blew_up = False
    for target in sample_steps:
        while step < target:
            f, _, _ = step_fn(f, p, dr, dz, r_c, r_f, dt, M_s, W)
            step += 1
            if not np.all(np.isfinite(f)) or np.max(np.abs(f)) > 5.0:
                blew_up = True
                break
        amp, _ = amplitude(f, r_c, dr)
        V = axisym_volume(f, r_c, dr, dz)
        ts.append(step * dt)
        amps.append(amp)
        drifts.append((V - V0) / V0)
        if blew_up:
            break

    # fit omega on the FIRST fit_frac of the trajectory (linear regime;
    # see run_one docstring)
    ts_a = np.array(ts)
    amps_a = np.array(amps)
    end = max(3, int(len(ts_a) * fit_frac))
    sel = np.isfinite(amps_a[:end]) & (amps_a[:end] > 0)
    r2 = float("nan")
    if np.sum(sel) >= 3:
        tt = ts_a[:end][sel]
        aa = np.log(amps_a[:end][sel])
        A = np.vstack([tt, np.ones_like(tt)]).T
        coef, res, _, _ = np.linalg.lstsq(A, aa, rcond=None)
        omega, intercept = coef
        pred = A @ coef
        ss_res = float(np.sum((aa - pred) ** 2))
        ss_tot = float(np.sum((aa - np.mean(aa)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        omega = float("nan")
    if blew_up:
        print(f"  [{label}] WARNING: blew up at step={step} t={ts[-1]:.3e} (before reaching "
              f"t_target={t_target}); dt likely still too large for the full horizon")

    print(f"  [{label}] lam/R0={lam_over_R0:.4f} n_steps={n_steps_total} t_final={ts[-1]:.3e} "
          f"omega={omega:.4e} r2={r2:.4f} amp0={amp0*1e9:.4f}nm ampf={amps[-1]*1e9:.4f}nm "
          f"max|mass_drift|={max(abs(d) for d in drifts):.2e}")
    return dict(label=label, R0_nm=R0_nm, lam_over_R0=lam_over_R0, lam_nm=lam_actual * 1e9,
                dx_nm=dx_nm, W_nm=W_nm, eps0_nm=eps0 * 1e9, dt=dt, n_steps_total=n_steps_total,
                omega=omega, r2=r2, amp0=amp0, ts=ts, amps=amps,
                mass_drift_max=max(abs(d) for d in drifts))


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "scalar_vs_production.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        cases = [("lam5.5", 5.5), ("lam2pi", 2 * math.pi), ("lam8", 8.0)]
        results = []
        for cid, lam_over_R0 in cases:
            r_scalar = run_one(axisym_surface_diffusion_step, f"scalar_{cid}", 40.0, lam_over_R0,
                                2.5, 20.0, 0.05, t_target=100.0, n_sample=40)
            r_face = run_one(axisym_face_projected_step, f"face_{cid}", 40.0, lam_over_R0,
                              2.5, 20.0, 0.05, t_target=100.0, n_sample=40)
            results.append(dict(case_id=cid, lam_over_R0=lam_over_R0, scalar=r_scalar, face=r_face))
        with open(out_path, "w") as fh:
            json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Section 6 summary: sign agreement (scalar vs face-projected) ---")
    all_ok = True
    for r in results:
        os_ = r["scalar"]["omega"]
        of_ = r["face"]["omega"]
        sign_s = "growth" if os_ > 0 else ("decay" if os_ < 0 else "neutral")
        sign_f = "growth" if of_ > 0 else ("decay" if of_ < 0 else "neutral")
        ok = sign_s == sign_f
        all_ok = all_ok and ok
        print(f"lam/R0={r['lam_over_R0']:.4f}: scalar omega={os_:.4e} ({sign_s}), "
              f"face omega={of_:.4e} ({sign_f}) -- {'MATCH' if ok else 'MISMATCH'}")
    print(f"\nALL SIGNS MATCH: {all_ok}")
