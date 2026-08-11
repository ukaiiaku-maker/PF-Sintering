"""Milestone 16D Sections 13-16: matched sink-OFF / sink-ON comparison
on the actual project geometry (one particle, one flat substrate),
using the CURRENT, qualified (tangent-cone) eta pathway -- the same
`variational_surface_diffusion_step` + `constrained_tangent_cone_eta_
update` stepping pattern `m15_gb_surface_rate_competition.py` and
Milestone 16C's Cartesian-vs-axisymmetric comparison already use (NOT
the older, ownership-preserving `evolve_eta`+`project_eta_mass_
preserving` pathway `run_sinkoff_stress_matrix.py` (Milestone 1/2)
used -- see the Section 2 audit in the milestone report for why that
distinction matters).

Both cases start from the IDENTICAL initial state (flat substrate,
`A_nm=0`, matching Milestone 16C's own finding that this -- not the
sinusoidal substrate -- is the actual project geometry).

SINK-OFF: active sink=False, RBM=False throughout (by construction --
`rbm()` is simply never called). Ordinary coarsening (surface
diffusion + tangent-cone GB migration) proceeds.

SINK-ON: `s.active` is FORCED True from step 0 (bypassing
`hazard_step`'s stochastic threshold entirely -- no stochastic
nucleation, per this milestone's explicit requirement), `s.tau_sink`
computed once from the initial stress state via the SAME physical
formula `hazard_step` uses (`tau=(xd*xd*kB*T)/(sigma*Omega*D_gb)`,
`xd=GS/4`), then `rbm()` is called every step in addition to the same
ordinary coarsening. Runs for the SAME physical time as SINK-OFF (not
"until quota completes") so the two trajectories are directly, matched
comparable at every sampled time.
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
from pf_sintering.model import ModelConfig, Sink, build_params, center, compute_stress, initialize_fields, reproject, rbm  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m15_gb_surface_rate_competition import BC_X, BC_Y  # noqa: E402


def build_state(dx_nm, W_nm, gamma_gb, R2_nm, aspect_ratio, overlap_nm, seed):
    """geometry="substrate" LITERALLY (not "sinusoidal_substrate" with
    amplitude 0, which is what m15_gb_surface_rate_competition.
    build_state's build_config always constructs regardless of `A_nm`)
    -- a real distinction, not a cosmetic one: `compute_stress`
    branches explicitly on `p.geometry=="substrate"` to use the
    smoother, more sensitive `contact_width` formula; the
    "sinusoidal_substrate" (even at amplitude 0) path falls through to
    a cruder single-column solid-extent measure instead. Caught
    directly: the single-column measure showed `x_neck` EXACTLY
    unchanged (159.00000000nm to 8 decimals) over 30000 steps in an
    early version of this script, while `contact_width` on the
    identical trajectory showed a real, continuous change
    (`29.6605nm -> 29.6943nm` over just 3000 steps) -- confirming this
    is a measurement-choice issue, not genuine stagnation. Also matches
    Milestone 16C's own Section 17 finding that the literal flat
    `"substrate"` geometry -- not `"sinusoidal_substrate"` at zero
    amplitude -- is the physically well-posed one."""
    cfg = ModelConfig(preset="dev", geometry="substrate", dx=dx_nm * 1e-9, r2=R2_nm * 1e-9,
                       aspect_ratio=aspect_ratio, contact_orientation="short_plane",
                       initial_overlap=overlap_nm * 1e-9, t_total=1e-6,
                       interface_width_override=W_nm * 1e-9, gamma_gb_override=gamma_gb, seed=seed)
    p = build_params(cfg)
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16d_campaign")


def sample_row(f, e1, e2, e3, s, p, step, t, V2_0, V_total_0):
    """`separation` is `center(e2,p) - wall_x0` -- the SAME definition
    Milestone 1's own report used ("particle/substrate ... separation"),
    NOT `Stress.gb_col` (a raw column-index proxy for the GB's lateral
    position along the contact plane, not the particle-substrate
    separation along the substrate normal -- using it as "separation"
    would have been a real mislabeling, caught before it was reported,
    not after)."""
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    V2 = float(e2.sum()) * p.dx * p.dx
    V_total = float(f.sum()) * p.dx * p.dx
    wall_x0 = (p.substrate_wall_frac - 0.5) * p.Nx * p.dx
    separation = center(e2, p) - wall_x0
    return dict(step=step, t=t, V2=V2, V2_frac=V2 / V2_0, V_total=V_total,
                mass_drift=(V_total - V_total_0) / V_total_0, x_neck=st.x_neck,
                sigma=st.sigma, sigma_lt=st.sigma_lt, sigma_curv=st.sigma_curv,
                separation=separation,
                densification_strain=s.cumulative_strain, cumulative_disp=s.cumulative_disp,
                sink_active=bool(s.active), stop=bool(stop), reason=reason)


def run(label, sink_on, t_target, n_sample=40, dx_nm=2.0, W_nm=10.0, gamma_gb=0.6, seed=42):
    p, f, e1, e2, e3 = build_state(dx_nm=dx_nm, W_nm=W_nm, gamma_gb=gamma_gb,
                                    R2_nm=80.0, aspect_ratio=2.0, overlap_nm=20.0, seed=seed)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    dt = p.dt
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V2_0 = float(e2.sum()) * p.dx * p.dx
    V_total_0 = float(f.sum()) * p.dx * p.dx

    if sink_on:
        st0, stop0, reason0 = compute_stress(f, e1, e2, e3, s, p)
        sigma0 = max(1e-6, st0.sigma)
        xd = 0.5 * st0.GS / 2
        s.tau_sink = (xd * xd * p.kB * p.T) / (sigma0 * p.Omega * p.D_gb)
        s.active = True
        s.n_d = 1
        s.phi = 1
        print(f"[{label}] forced sink active: sigma0={sigma0:.4e}Pa GS={st0.GS*1e9:.2f}nm "
              f"tau_sink={s.tau_sink:.4e}s")

    rows = [sample_row(f, e1, e2, e3, s, p, 0, 0.0, V2_0, V_total_0)]
    step = 0
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
                if completed:
                    print(f"  [{label}] sink event quota completed at step {step}")
            step += 1
        row = sample_row(f, e1, e2, e3, s, p, step, step * dt, V2_0, V_total_0)
        rows.append(row)
        print(f"  [{label}] step={step} t={row['t']:.4e} V2_frac={row['V2_frac']:.6f} "
              f"x_neck={row['x_neck']*1e9:.3f}nm sigma={row['sigma']:.4e}Pa "
              f"strain={row['densification_strain']:.4e} sink_active={row['sink_active']} "
              f"mass_drift={row['mass_drift']:.2e}")

    return dict(label=label, sink_on=sink_on, dx_nm=dx_nm, W_nm=W_nm, gamma_gb=gamma_gb, dt=dt,
                n_steps_total=n_steps_total, V2_0=V2_0, V_total_0=V_total_0, rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "sink_off_on_comparison.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        t_target = 0.02
        off = run("SINK-OFF", sink_on=False, t_target=t_target)
        on = run("SINK-ON", sink_on=True, t_target=t_target)
        results = dict(off=off, on=on, t_target=t_target)
        with open(out_path, "w") as fh:
            json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Sections 14-16 summary ---")
    off, on = results["off"], results["on"]
    row0_off, rowf_off = off["rows"][0], off["rows"][-1]
    row0_on, rowf_on = on["rows"][0], on["rows"][-1]
    print(f"SINK-OFF: x_neck {row0_off['x_neck']*1e9:.3f}->{rowf_off['x_neck']*1e9:.3f}nm  "
          f"sigma {row0_off['sigma']:.4e}->{rowf_off['sigma']:.4e}Pa  "
          f"strain {row0_off['densification_strain']:.4e}->{rowf_off['densification_strain']:.4e}  "
          f"separation {row0_off['separation']*1e9:.3f}->{rowf_off['separation']*1e9:.3f}nm")
    print(f"SINK-ON:  x_neck {row0_on['x_neck']*1e9:.3f}->{rowf_on['x_neck']*1e9:.3f}nm  "
          f"sigma {row0_on['sigma']:.4e}->{rowf_on['sigma']:.4e}Pa  "
          f"strain {row0_on['densification_strain']:.4e}->{rowf_on['densification_strain']:.4e}  "
          f"separation {row0_on['separation']*1e9:.3f}->{rowf_on['separation']*1e9:.3f}nm")
