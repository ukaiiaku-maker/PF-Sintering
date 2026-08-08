"""Milestone 11 -- natural surface-diffusion crossover in the sinusoidal contact.

DIAGNOSTIC ONLY. Produces the data behind
MILESTONE_11_NATURAL_SURFACE_DIFFUSION_CROSSOVER.md: a long, log-sampled S0
(Ostwald-off) trajectory on the sinusoidal substrate geometry, instantaneous
CH-only contact tendency g_CH(t), neck-region CH mass balance, free-surface
mu(s)/J_s(s) branch profiles at representative states, sinusoid/contact
multiscale tracking, F_before/F_neutral/F_after crossover-bracket states,
and S0/S1 forks from those states.

Does not modify production physics.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.ch_crossover_diagnostics import g_ch_probe, neck_ch_mass_balance, neck_region_mask, trace_branch_profile
from pf_sintering.ch_flux_diagnostics import evolve_f_diagnostic
from pf_sintering.differential_coarsening import _step_once, full_state_sample, run_single_trajectory
from pf_sintering.diagnostics import wall_x0
from pf_sintering.model import ModelConfig, Sink, build_params, compute_stress, evolve_eta, evolve_f, initialize_fields
from pf_sintering.ostwald_receiver_closures import ostwald_external_reservoir_step
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_config(rate, args, dt_override=None):
    return ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", nx=args.nx, ny=args.ny, dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=args.overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=rate, surface_mobility_scale=args.surface_mobility_scale,
        eta_mobility_scale=args.eta_mobility_scale,
        sinusoid_wavelength=args.wavelength_nm * 1e-9, sinusoid_amplitude=args.amplitude_nm * 1e-9,
        interface_width_override=args.fixed_w_nm * 1e-9, eta_diffusivity_fixed_physical=True,
        dt_override=dt_override,
    )


def log_step_schedule(max_steps, n_points):
    base = max_steps ** (1.0 / n_points)
    steps = sorted(set(max(1, int(round(base ** k))) for k in range(1, n_points + 1)))
    return [s for s in steps if 1 <= s <= max_steps]


def sinusoid_fourier_fit(f, p, y_exclude_half_width, n_harmonics=3):
    x = (np.arange(1, p.Nx + 1) - p.Nx / 2) * p.dx
    y = (np.arange(1, p.Ny + 1) - p.Ny / 2) * p.dx
    rows_used, x_cross = [], []
    for i in range(p.Ny):
        if abs(y[i]) < y_exclude_half_width:
            continue
        row = f[i, :]
        sign = row - 0.5
        idx = np.where(np.diff(np.sign(sign)) != 0)[0]
        if len(idx) == 0:
            continue
        j = idx[0]
        v0, v1 = sign[j], sign[j + 1]
        t = -v0 / (v1 - v0) if (v1 - v0) != 0 else 0.0
        x_c = x[j] + t * (x[j + 1] - x[j])
        rows_used.append(y[i])
        x_cross.append(x_c)
    if len(rows_used) < 2 * n_harmonics + 2:
        return None
    Y = np.array(rows_used)
    Xc = np.array(x_cross)
    cols = [np.ones_like(Y)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * 2 * math.pi * Y / p.sinusoid_wavelength))
        cols.append(np.sin(k * 2 * math.pi * Y / p.sinusoid_wavelength))
    A = np.stack(cols, axis=1)
    coeffs, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    out = dict(x_mean=float(coeffs[0]))
    for k in range(1, n_harmonics + 1):
        out[f"A{k}"] = float(coeffs[1 + 2 * (k - 1)])
        out[f"B{k}"] = float(coeffs[2 + 2 * (k - 1)])
        out[f"amp{k}"] = float(math.hypot(out[f"A{k}"], out[f"B{k}"]))
    return out


def _reduced(row):
    keys = ("step", "time_s", "V2", "L_contact_TJ_sub", "L_GB_geom_sub", "psi_deg_top", "psi_deg_bottom",
            "kappa_top_1pm", "kappa_bottom_1pm", "F_TJ_mag_top", "F_TJ_mag_bottom",
            "sigma_Pa", "sigma_lt_Pa", "sigma_curv_Pa", "F_gb_normal_top", "F_gb_normal_bottom",
            "tj_top_x_sub", "tj_top_y_sub", "tj_bottom_x_sub", "tj_bottom_y_sub")
    return {k: row.get(k) for k in keys}


def run_long_s0_with_probes(p0, f0, e1_0, e2_0, e3_0, schedule, y_excl, extra_after_crossover=8):
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    wall_x0_val = wall_x0(p0)
    v20 = float(e2.sum()) * p0.dx * p0.dx

    schedule_set = set(schedule)
    max_sched = max(schedule)
    samples = []
    snapshots = {}
    seen_positive = False
    seen_negative = False
    crossover_confirm_countdown = None

    step = 0
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p0)
    if 1 in schedule_set:
        pass  # step 0 handled specially below
    row0 = full_state_sample(f, e1, e2, e3, s, st, p0, 0, 0.0, v20, wall_x0_val)
    probe0 = g_ch_probe(f, e1, e2, e3, s, p0)
    fit0 = sinusoid_fourier_fit(f, p0, y_excl)
    samples.append(dict(row=_reduced(row0), g_ch=probe0.g_ch, fourier=fit0))
    snapshots[0] = (f.copy(), e1.copy(), e2.copy(), e3.copy())
    if probe0.g_ch.get(1.0, math.nan) > 0:
        seen_positive = True

    while step < max_sched:
        step += 1
        f, e1, e2, e3 = _step_once(f, e1, e2, e3, s, p0)
        if step in schedule_set:
            st, stop, reason = compute_stress(f, e1, e2, e3, s, p0)
            row = full_state_sample(f, e1, e2, e3, s, st, p0, step, step * p0.dt, v20, wall_x0_val)
            probe = g_ch_probe(f, e1, e2, e3, s, p0)
            fit = sinusoid_fourier_fit(f, p0, y_excl)
            samples.append(dict(row=_reduced(row), g_ch=probe.g_ch, fourier=fit))
            snapshots[step] = (f.copy(), e1.copy(), e2.copy(), e3.copy())
            g1 = probe.g_ch.get(1.0, math.nan)
            print(f"  step={step:6d} t={step*p0.dt:.4e}s L_contact_TJ_sub={row['L_contact_TJ_sub']*1e9:.5f}nm "
                  f"g_CH={g1:+.4e} 1/s stop={stop}")
            if math.isfinite(g1):
                if g1 > 0:
                    seen_positive = True
                elif g1 <= 0 and seen_positive:
                    seen_negative = True
            if seen_negative and crossover_confirm_countdown is None:
                crossover_confirm_countdown = extra_after_crossover
                print(f"  ** sign change detected at step={step}; continuing {extra_after_crossover} more samples to confirm **")
            elif crossover_confirm_countdown is not None:
                crossover_confirm_countdown -= 1
                if crossover_confirm_countdown <= 0:
                    print(f"  ** crossover confirmed; stopping early at step={step} (schedule max={max_sched}) **")
                    break
            if stop:
                print(f"  ** compute_stress stop condition reached: {reason} -- stopping **")
                break
    return samples, snapshots


def find_crossover_bracket(samples):
    g1_series = [smp["g_ch"].get(1.0, math.nan) for smp in samples]
    for i in range(len(g1_series) - 1):
        a, b = g1_series[i], g1_series[i + 1]
        if math.isfinite(a) and math.isfinite(b) and a > 0 and b <= 0:
            idx_neutral = i if abs(a) < abs(b) else i + 1
            return i, idx_neutral, i + 1
    return None


def run_fork(label, f0, e1_0, e2_0, e3_0, p0, p1, target_dv2_frac, max_steps):
    """S0 (continue no-coarsening) and S1 (external reservoir) forks from a
    shared state, matched physical time (same step count, set by S1
    reaching target_dv2_frac)."""
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    v20 = float(e2.sum()) * p1.dx * p1.dx
    step = 0
    while step < max_steps:
        step += 1
        ch_targets = eta_masses(e1, e2, e3, p1.use_eta3)
        f1 = evolve_f(f, e1, e2, e3, s, Sink(), p1)
        e1a, e2a, e3a = project_eta_mass_preserving(f1, e1, e2, e3, p1, target_masses=ch_targets)
        f2, e1b, e2b, e3b, _ = ostwald_external_reservoir_step(f1, e1a, e2a, e3a, p1, phi_local=0.0)
        ac_targets = eta_masses(e1b, e2b, e3b, p1.use_eta3)
        e1c, e2c, e3c = evolve_eta(e1b, e2b, e3b, p1)
        e1, e2, e3 = project_eta_mass_preserving(f2, e1c, e2c, e3c, p1, target_masses=ac_targets)
        f = f2
        v2 = float(e2.sum()) * p1.dx * p1.dx
        if abs(v2 - v20) / v20 >= target_dv2_frac:
            break
    n_ref = step
    st1, _, _ = compute_stress(f, e1, e2, e3, s, p1)
    row_s1 = _reduced(full_state_sample(f, e1, e2, e3, s, st1, p1, n_ref, n_ref * p1.dt, v20, wall_x0(p1)))

    rows_s0, stop0, _ = run_single_trajectory(p0, f0, e1_0, e2_0, e3_0, n_ref)
    row_s0 = _reduced(rows_s0[-1])
    delta = row_s1["L_contact_TJ_sub"] - row_s0["L_contact_TJ_sub"]
    print(f"  [{label}] n_ref={n_ref} dV2/V20={(v2-v20)/v20:+.4e} "
          f"S0.L_contact_TJ_sub={row_s0['L_contact_TJ_sub']*1e9:.5f}nm "
          f"S1.L_contact_TJ_sub={row_s1['L_contact_TJ_sub']*1e9:.5f}nm delta(S1-S0)={delta*1e9:+.6f}nm")
    return dict(n_ref=n_ref, dV2_over_V20=(v2 - v20) / v20, S0=row_s0, S1=row_s1, delta_S1_vs_S0=delta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--n-sample-points", type=int, default=90)
    ap.add_argument("--extra-after-crossover", type=int, default=8)
    ap.add_argument("--fork-target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--fork-max-steps", type=int, default=3000)
    ap.add_argument("--branch-max-arclength-nm", type=float, default=150.0)
    ap.add_argument("--out", type=str, default="runs/surface_diffusion_crossover_audit.json")
    args = ap.parse_args()

    out = dict(args=vars(args))
    p0 = build_params(build_config(0.0, args))
    p1 = build_params(build_config(args.coarsening_rate_scale, args))
    assert p0.dt == p1.dt
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)
    y_excl = p0.Ry + 3 * p0.interface_width

    print(f"geometry: Nx={p0.Nx} Ny={p0.Ny} dx={p0.dx*1e9:.2f}nm wavelength={p0.sinusoid_wavelength*1e9:.1f}nm "
          f"amplitude={p0.sinusoid_amplitude*1e9:.1f}nm dt={p0.dt:.4e}s")

    schedule = log_step_schedule(args.max_steps, args.n_sample_points)
    print(f"log-spaced schedule: {len(schedule)} points, first 10={schedule[:10]}, last 5={schedule[-5:]}")

    print("\n=== Section 2/3: long S0 trajectory with instantaneous g_CH probes ===")
    samples, snapshots = run_long_s0_with_probes(p0, f0, e1_0, e2_0, e3_0, schedule, y_excl,
                                                  extra_after_crossover=args.extra_after_crossover)
    out["samples"] = [dict(row=smp["row"], g_ch=smp["g_ch"], fourier=smp["fourier"]) for smp in samples]

    bracket = find_crossover_bracket(samples)
    out["crossover_found"] = bracket is not None
    if bracket is None:
        print("\n=== No g_CH sign crossover found within budget (Outcome C) ===")
        with open(args.out, "w") as fh:
            json.dump(out, fh)
        print(f"wrote {args.out}")
        return

    i_before, i_neutral, i_after = bracket
    step_before = samples[i_before]["row"]["step"]
    step_neutral = samples[i_neutral]["row"]["step"]
    step_after = samples[i_after]["row"]["step"]
    print(f"\n=== Section 9: crossover bracket -- before(step={step_before}) "
          f"neutral(step={step_neutral}) after(step={step_after}) ===")
    out["bracket_steps"] = dict(before=step_before, neutral=step_neutral, after=step_after)

    print("\n=== Section 6/7: branch profiles + neck mass balance at representative states ===")
    idx_early = 0
    idx_mid = len(samples) // 3
    reps = dict(early=idx_early, intermediate=idx_mid, near_crossover=i_neutral, late=i_after)
    profiles = {}
    for label, idx in reps.items():
        step_i = samples[idx]["row"]["step"]
        f_i, e1_i, e2_i, e3_i = snapshots.get(step_i, (None, None, None, None))
        if f_i is None:
            continue
        s = Sink(threshold=math.inf)
        f_after_ch, diag = evolve_f_diagnostic(f_i, e1_i, e2_i, e3_i, s, p0)
        rep = compute_neck_tj_forces(f_i, e1_i, e2_i, e3_i, s, p0)
        sub = compute_subgrid_contact(f_i, e1_i, e2_i, p0)
        mask = neck_region_mask(p0, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub)) if sub.resolved else None
        mb = neck_ch_mass_balance(f_i, f_after_ch, mask, p0) if mask is not None else math.nan
        entry = dict(step=step_i, neck_mass_balance=mb)
        if rep.top and rep.top.resolved:
            entry["top"] = trace_branch_profile(f_i, diag.mu, diag.Jx, diag.Jy, p0, rep.top.tj_xy, rep.top.v_s1,
                                                 max_arclength=args.branch_max_arclength_nm * 1e-9)
        if rep.bottom and rep.bottom.resolved:
            entry["bottom"] = trace_branch_profile(f_i, diag.mu, diag.Jx, diag.Jy, p0, rep.bottom.tj_xy, rep.bottom.v_s1,
                                                    max_arclength=args.branch_max_arclength_nm * 1e-9)
        profiles[label] = entry
        print(f"  {label} (step={step_i}): neck_mass_balance={mb:.4e}m^2  "
              f"top_profile={'ok' if entry.get('top') else 'unresolved'}  bottom_profile={'ok' if entry.get('bottom') else 'unresolved'}")
    out["profiles"] = profiles

    print("\n=== Section 11: S0/S1 forks from F_before/F_neutral/F_after ===")
    fork_out = {}
    for label, idx in (("F_before", i_before), ("F_neutral", i_neutral), ("F_after", i_after)):
        step_i = samples[idx]["row"]["step"]
        f_i, e1_i, e2_i, e3_i = snapshots[step_i]
        fork_out[label] = run_fork(label, f_i, e1_i, e2_i, e3_i, p0, p1,
                                    args.fork_target_dv2_frac, args.fork_max_steps)
    out["forks"] = fork_out

    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
