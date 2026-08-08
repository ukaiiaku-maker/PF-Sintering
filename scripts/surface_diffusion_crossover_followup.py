"""Milestone 11 follow-up: since the long S0 run found no g_CH sign
crossover (Outcome C), this computes the requested branch-profile/neck-
mass-balance/fork diagnostics at representative EARLY/MID/LATE states of
the characterized (monotonically slowing, never-reversing) trajectory
instead of at a before/neutral/after bracket that does not exist.

Re-runs S0 fresh from t=0 to each target step count (no persisted
snapshots from the original run) -- cheap, each run is a few minutes at
most given the measured ~3.4ms/step production-step cost.
"""

from __future__ import annotations

import argparse
import json
import math

from pf_sintering.ch_crossover_diagnostics import g_ch_probe, neck_ch_mass_balance, neck_region_mask, trace_branch_profile
from pf_sintering.ch_flux_diagnostics import evolve_f_diagnostic
from pf_sintering.differential_coarsening import _step_once, full_state_sample
from pf_sintering.diagnostics import wall_x0
from pf_sintering.model import Sink, build_params, compute_stress, initialize_fields
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact

from surface_diffusion_crossover_audit import build_config, run_fork, sinusoid_fourier_fit, _reduced


def run_to_step(p0, f0, e1_0, e2_0, e3_0, target_step):
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    for step in range(1, target_step + 1):
        f, e1, e2, e3 = _step_once(f, e1, e2, e3, s, p0)
    return f, e1, e2, e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--fixed-w-nm", type=float, default=20.0)
    ap.add_argument("--coarsening-rate-scale", type=float, default=3.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--eta-mobility-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 10319, 200000])
    ap.add_argument("--labels", type=str, nargs="+", default=["early", "mid", "late"])
    ap.add_argument("--branch-max-arclength-nm", type=float, default=150.0)
    ap.add_argument("--fork-target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--fork-max-steps", type=int, default=3000)
    ap.add_argument("--out", type=str, default="runs/surface_diffusion_crossover_followup.json")
    args = ap.parse_args()

    out = dict(args=vars(args))
    p0 = build_params(build_config(0.0, args))
    p1 = build_params(build_config(args.coarsening_rate_scale, args))
    assert p0.dt == p1.dt
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)
    y_excl = p0.Ry + 3 * p0.interface_width
    s = Sink(threshold=math.inf)

    profiles = {}
    forks = {}
    for label, target_step in zip(args.labels, args.steps):
        print(f"\n=== {label} (step={target_step}) ===")
        f_i, e1_i, e2_i, e3_i = run_to_step(p0, f0, e1_0, e2_0, e3_0, target_step)
        st, stop, reason = compute_stress(f_i, e1_i, e2_i, e3_i, s, p0)
        v20 = float(e2_0.sum()) * p0.dx * p0.dx
        row = _reduced(full_state_sample(f_i, e1_i, e2_i, e3_i, s, st, p0, target_step,
                                          target_step * p0.dt, v20, wall_x0(p0)))
        probe = g_ch_probe(f_i, e1_i, e2_i, e3_i, s, p0)
        fit = sinusoid_fourier_fit(f_i, p0, y_excl)
        print(f"  L_contact_TJ_sub={row['L_contact_TJ_sub']*1e9:.5f}nm g_CH={probe.g_ch}")
        print(f"  fourier: {fit}")

        f_after_ch, diag = evolve_f_diagnostic(f_i, e1_i, e2_i, e3_i, s, p0)
        rep = compute_neck_tj_forces(f_i, e1_i, e2_i, e3_i, s, p0)
        sub = compute_subgrid_contact(f_i, e1_i, e2_i, p0)
        mask = neck_region_mask(p0, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub)) if sub.resolved else None
        mb = neck_ch_mass_balance(f_i, f_after_ch, mask, p0) if mask is not None else math.nan
        print(f"  neck_mass_balance={mb:.4e} m^2")

        entry = dict(step=target_step, row=row, g_ch=probe.g_ch, fourier=fit, neck_mass_balance=mb)
        if rep.top and rep.top.resolved:
            entry["top"] = trace_branch_profile(f_i, diag.mu, diag.Jx, diag.Jy, p0, rep.top.tj_xy, rep.top.v_s1,
                                                 max_arclength=args.branch_max_arclength_nm * 1e-9)
        if rep.bottom and rep.bottom.resolved:
            entry["bottom"] = trace_branch_profile(f_i, diag.mu, diag.Jx, diag.Jy, p0, rep.bottom.tj_xy, rep.bottom.v_s1,
                                                    max_arclength=args.branch_max_arclength_nm * 1e-9)
        profiles[label] = entry

        forks[label] = run_fork(label, f_i, e1_i, e2_i, e3_i, p0, p1, args.fork_target_dv2_frac, args.fork_max_steps)

    out["profiles"] = profiles
    out["forks"] = forks
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
