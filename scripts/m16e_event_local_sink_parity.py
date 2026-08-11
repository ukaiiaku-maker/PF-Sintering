"""Milestone 16E Sections 11-13: event-local sink OFF/ON clone
comparison, isolating the active-sink event itself from subsequent
ordinary coarsening (M16D evaluated the forced-sink effect over a
20ms-scale window when the sink quota completes by ~3ms -- diluting
the event's own signature with a much longer stretch of ordinary
post-event coarsening common to both clones).

One initial state (identical to M16D's corrected sink comparison:
literal `geometry="substrate"`, current tangent-cone eta pathway) is
cloned exactly into OFF and ON. OFF never calls `rbm()`. ON forces one
sink active from step 0 (same physical `tau_sink` formula
`hazard_step` itself uses -- no stochastic threshold). Both run only
until the ON clone's sink quota completes, plus a short fixed
post-completion tail, sampled at MATCHED steps.

V2 (grain-2/particle volume) is decomposed, at every sample, into the
contribution from ordinary CH/surface transport + eta migration
(present in BOTH clones, computed on the OFF clone as the reference)
versus the ADDITIONAL contribution unique to the ON clone's RBM/active-
sink transport (`V2_ON(t) - V2_OFF(t)`, the exact differential this
project's mass-accounting contract is built to keep visible) -- Section
13's explicit requirement that the active sink not become a silent,
unaccounted-for extra grain-volume-loss channel.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.ch_exact_energy import mu_isotropic  # noqa: E402
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update  # noqa: E402
from pf_sintering.model import Sink, compute_stress, rbm  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m15_gb_surface_rate_competition import BC_X, BC_Y  # noqa: E402
from m16d_sink_off_on_comparison import build_state, sample_row  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16e_campaign")


def run_clone(label, sink_on, f0, e1_0, e2_0, e3_0, p, tau_sink, n_steps_total, n_sample):
    f, e1, e2, e3 = f0.copy(), e1_0.copy(), e2_0.copy(), e3_0.copy()
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    dt = p.dt
    if sink_on:
        s.tau_sink = tau_sink
        s.active = True
        s.n_d = 1
        s.phi = 1

    V2_0 = float(e2.sum()) * p.dx * p.dx
    V_total_0 = float(f.sum()) * p.dx * p.dx
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    rows = [sample_row(f, e1, e2, e3, s, p, 0, 0.0, V2_0, V_total_0)]
    step = 0
    quota_step = None
    for target in sample_steps:
        if target == 0:
            continue
        while step < target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, _ = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s,
                                                        bc_x=BC_X, bc_y=BC_Y)
            e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt,
                                                                  use_eta3=False, bc_x=BC_X, bc_y=BC_Y)
            if sink_on:
                f, e1, e2, e3, completed = rbm(f, e1, e2, e3, s, p)
                if completed and quota_step is None:
                    quota_step = step
            step += 1
        rows.append(sample_row(f, e1, e2, e3, s, p, step, step * dt, V2_0, V_total_0))

    return dict(label=label, sink_on=sink_on, dt=dt, quota_step=quota_step, V2_0=V2_0,
                V_total_0=V_total_0, rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "event_local_sink_parity.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            result = json.load(fh)
    else:
        p, f0, e1_0, e2_0, e3_0 = build_state(dx_nm=2.0, W_nm=10.0, gamma_gb=0.6, R2_nm=80.0,
                                               aspect_ratio=2.0, overlap_nm=20.0, seed=42)
        s_probe = Sink(threshold=math.inf)
        st0, stop0, reason0 = compute_stress(f0, e1_0, e2_0, e3_0, s_probe, p)
        sigma0 = max(1e-6, st0.sigma)
        xd = 0.5 * st0.GS / 2
        tau_sink = (xd * xd * p.kB * p.T) / (sigma0 * p.Omega * p.D_gb)
        print(f"sigma0={sigma0:.4e}Pa GS={st0.GS*1e9:.2f}nm tau_sink={tau_sink:.4e}s dt={p.dt:.4e}s")

        # From M16D: quota completed at step 1041 with this exact tau_sink/dt.
        # Run to ~1.5x that (event + short post-completion tail), fine sampling.
        n_steps_total = 1600
        n_sample = 80

        off = run_clone("OFF", False, f0, e1_0, e2_0, e3_0, p, tau_sink, n_steps_total, n_sample)
        on = run_clone("ON", True, f0, e1_0, e2_0, e3_0, p, tau_sink, n_steps_total, n_sample)
        result = dict(sigma0=sigma0, tau_sink=tau_sink, dt=p.dt, n_steps_total=n_steps_total, off=off, on=on)
        with open(out_path, "w") as fh:
            json.dump(result, fh, default=str)
        print(f"saved to {out_path}")

    off, on = result["off"], result["on"]
    quota_step = on["quota_step"]
    print(f"\nquota_step (ON) = {quota_step}")

    def at_step(rows, step):
        best = min(rows, key=lambda r: abs(r["step"] - step))
        return best

    print("\n--- Section 11-12: event-local differential response ---")
    print(f"{'t':>12s} {'d_strain':>12s} {'d_separation':>14s} {'d_xneck':>10s} {'d_sigma':>12s} {'d_V2':>12s}")
    for r_off, r_on in zip(off["rows"], on["rows"]):
        d_strain = r_on["densification_strain"] - r_off["densification_strain"]
        d_sep = r_on["separation"] - r_off["separation"]
        d_xneck = r_on["x_neck"] - r_off["x_neck"]
        d_sigma = r_on["sigma"] - r_off["sigma"]
        d_V2 = r_on["V2"] - r_off["V2"]
        flag = " <-- quota" if quota_step is not None and abs(r_on["step"] - quota_step) < (on["rows"][1]["step"] - on["rows"][0]["step"]) else ""
        print(f"{r_on['t']:12.4e} {d_strain:12.4e} {d_sep*1e9:14.4f} {d_xneck*1e9:10.4f} {d_sigma:12.4e} {d_V2:12.4e}{flag}")
