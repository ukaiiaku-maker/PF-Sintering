"""Milestone 15I Section 8: does the operator remainder R_L vanish under
TRUE dt refinement (not just fewer steps at the SAME dt, which M15H
used)?

Two separate scalings, both from the SAME canonical state
(theta=45deg/AN3, t=0.02s -- the checkpoint nearest the observed
crossover):
  (a) n_steps in {10,5,2,1} at the production dt (M15H's own scaling,
      repeated here for a consistency cross-check);
  (b) TRUE dt refinement: dt, dt/2, dt/4 (via dataclasses.replace(p,
      dt=...)), each run for n_steps chosen so n_steps*dt_trial is the
      SAME total Delta_t across the three trials (so (a) varies
      Delta_t at fixed dt per step; (b) varies dt per step at fixed
      total Delta_t -- the genuine Richardson-extrapolation axis).

If R_L/Delta_t -> a finite nonzero limit under (b) as dt_trial->0, the
M15H finding (a real, persistent nonlinear SURF/GB interaction) is
confirmed as a genuine continuum-limit effect, not finite-difference
error. If it -> 0, M15H's residual was finite-dt truncation error and
should be reclassified.
"""
from __future__ import annotations

import dataclasses
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y, build_state  # noqa: E402
from m15f_campaign_lib import M_GB_REF  # noqa: E402
from m15g_mechanism_lib import operator_audit, trial_evolve  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402

MORPH = dict(A_nm=120.0, lambda_nm=320.0, R2_nm=80.0, aspect_ratio=2.0, overlap_nm=5.0,
             gamma_gb=1.4, M_GB=30.0 * M_GB_REF, surface_mobility_scale=0.3 * 1.0, dx_nm=2.5, W_nm=20.0,
             use_aniso_surface=True, aniso_delta=1.0 / 15 - 0.002, theta_mis_deg=45.0)


def build_and_advance(t_ckpt):
    p, f, e1, e2, e3 = build_state(**MORPH)
    s = Sink(threshold=math.inf)
    dt = p.dt
    n_target = round(t_ckpt / dt)
    step = 0
    while step < n_target:
        f, e1, e2, e3 = trial_evolve(f, e1, e2, e3, s, p, "FULL", 1, bc_x=BC_X, bc_y=BC_Y)
        step += 1
    return p, f, e1, e2, e3, s


def operator_audit_with_dt(f, e1, e2, e3, s, p, dt_trial, n_steps, label):
    p_trial = dataclasses.replace(p, dt=dt_trial)
    return operator_audit(f, e1, e2, e3, s, p_trial, label, n_steps=n_steps, verbose=False)


if __name__ == "__main__":
    t_ckpt = 0.02
    p, f, e1, e2, e3, s = build_and_advance(t_ckpt)
    dt0 = p.dt
    print(f"canonical state at t={t_ckpt}, production dt={dt0:.6e}")

    print("\n(a) n_steps ladder at production dt (M15H-style, cross-check):")
    for n_steps in (10, 5, 2, 1):
        Delta_t = n_steps * dt0
        audit = operator_audit(f, e1, e2, e3, s, p, f"a_n{n_steps}", n_steps=n_steps, verbose=False)
        R_L = audit.get("R_coupled_L_contact")
        print(f"  n_steps={n_steps:2d} Delta_t={Delta_t:.3e} R_L={R_L} R_L/Delta_t={R_L/Delta_t if R_L is not None else None}")

    print("\n(b) TRUE dt refinement at FIXED total Delta_t (Richardson axis):")
    Delta_t_fixed = 10 * dt0  # match (a)'s n_steps=10 total window
    for divisor in (1, 2, 4):
        dt_trial = dt0 / divisor
        n_steps = round(Delta_t_fixed / dt_trial)
        audit = operator_audit_with_dt(f, e1, e2, e3, s, p, dt_trial, n_steps, f"b_div{divisor}")
        R_L = audit.get("R_coupled_L_contact")
        actual_Delta_t = n_steps * dt_trial
        print(f"  dt=dt0/{divisor} ({dt_trial:.3e}) n_steps={n_steps} Delta_t={actual_Delta_t:.3e} "
              f"R_L={R_L} R_L/Delta_t={R_L/actual_Delta_t if R_L is not None else None}")
