"""Milestone 7 -- differential-coarsening audit driver.

DIAGNOSTIC ONLY. Produces the data behind
MILESTONE_7_DIFFERENTIAL_COARSENING_CH_AUDIT.md:

1. primary paired C0 (coarsening off)/C1 (coarsening on) trajectory, sampled
   every step, extended to |dV2|/V20 = 3e-4, 1.5e-3, 3e-3 (single run, sliced
   at each milestone rather than three independent runs);
2. compact coarsening-rate series against the same C0 reference;
3. CH dt/dx stability audit numbers;
4. fixed-grid timestep-sensitivity check (dt, dt/2, dt/4) at dx=5nm and
   dx=2.5nm;
5. dt-converged coarse/fine grid paired comparison;
6. isotropic-mode exact free-energy functional trace (side check);
7. 5nm-overlap paired cross-check.

Does not modify production physics; only calls the existing validated
operators via pf_sintering.differential_coarsening.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.differential_coarsening import (
    PAIRED_KEYS,
    paired_delta_series,
    run_paired_trajectory,
    run_single_trajectory,
)
from pf_sintering.model import (
    ModelConfig,
    Sink,
    build_params,
    compute_stress,
    div,
    effective_gamma,
    evolve_eta,
    evolve_f,
    grad,
    initialize_fields,
    lap9,
    ostwald_substrate,
)
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving


def build_config(coarsening_rate_scale, args, overlap_nm=None, dx_nm=None, dt_override=None, nx=None, ny=None):
    return ModelConfig(
        preset="dev", geometry="substrate", nx=nx or args.nx, ny=ny or args.ny,
        dx=(dx_nm if dx_nm is not None else args.dx_nm) * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=(overlap_nm if overlap_nm is not None else args.overlap_nm) * 1e-9,
        t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=coarsening_rate_scale,
        surface_mobility_scale=args.surface_mobility_scale,
        eta_mobility_scale=args.eta_mobility_scale,
        dt_override=dt_override,
    )


def _find_milestone_index(rows_c1, v20, target):
    for i, row in enumerate(rows_c1):
        if abs(row["V2"] - v20) / v20 >= target:
            return i
    return len(rows_c1) - 1


def _reduced_row(row):
    return {k: row.get(k) for k in ("step", "time_s", "V2") + PAIRED_KEYS}


def run_primary(args):
    p_c0 = build_params(build_config(0.0, args))
    p_c1 = build_params(build_config(args.coarsening_rate_scale, args))
    assert p_c0.dt == p_c1.dt
    f0, e1_0, e2_0, e3_0 = initialize_fields(p_c0)
    print(f"\nprimary geometry: Nx={p_c0.Nx} Ny={p_c0.Ny} dx={p_c0.dx*1e9:.2f}nm "
          f"overlap={args.overlap_nm}nm dt={p_c0.dt:.4e}s")

    rows_c0, rows_c1, reason = run_paired_trajectory(
        p_c0, p_c1, f0, e1_0, e2_0, e3_0,
        target_dv2_frac=args.extended_dv2_frac_2, max_steps=args.max_steps,
    )
    print(f"primary paired run: n_steps={len(rows_c1)-1} stop_reason={reason!r}")
    v20 = rows_c0[0]["V2"]

    milestones = {}
    for target in (args.target_dv2_frac, args.extended_dv2_frac, args.extended_dv2_frac_2):
        idx = _find_milestone_index(rows_c1, v20, target)
        d = paired_delta_series(rows_c0[: idx + 1], rows_c1[: idx + 1])[-1]
        dv2 = (rows_c1[idx]["V2"] - v20) / v20
        print(f"\n=== milestone |dV2|/V20~{target:.1e}: idx={idx} actual dV2/V20={dv2:+.4e} ===")
        print(f"  C0: L_contact_TJ_sub={rows_c0[idx]['L_contact_TJ_sub']*1e9:+.5f}nm "
              f"L_GB_geom_sub={rows_c0[idx]['L_GB_geom_sub']*1e9:+.5f}nm "
              f"psi_top={rows_c0[idx]['psi_deg_top']:.3f}deg sigma={rows_c0[idx]['sigma_Pa']/1e6:.4f}MPa")
        print(f"  C1: L_contact_TJ_sub={rows_c1[idx]['L_contact_TJ_sub']*1e9:+.5f}nm "
              f"L_GB_geom_sub={rows_c1[idx]['L_GB_geom_sub']*1e9:+.5f}nm "
              f"psi_top={rows_c1[idx]['psi_deg_top']:.3f}deg sigma={rows_c1[idx]['sigma_Pa']/1e6:.4f}MPa")
        print(f"  delta_L_coarsening = L_contact_TJ_sub_C1 - L_contact_TJ_sub_C0 = "
              f"{d['L_contact_TJ_sub']*1e9:+.6f}nm")
        print(f"  delta_L_GB_geom_sub = {d['L_GB_geom_sub']*1e9:+.6f}nm")
        print(f"  delta_sigma = {d['sigma_Pa']/1e6:+.6f}MPa   delta_psi_top = "
              f"{rows_c1[idx]['psi_deg_top']-rows_c0[idx]['psi_deg_top']:+.4f}deg")
        milestones[f"{target:g}"] = dict(
            idx=idx, dV2_over_V20=dv2, delta=d,
            c0=_reduced_row(rows_c0[idx]), c1=_reduced_row(rows_c1[idx]),
        )

    deltas = paired_delta_series(rows_c0, rows_c1)
    return dict(
        n_steps=len(rows_c1) - 1, V20=v20, stop_reason=reason, milestones=milestones,
        rows_c0=[_reduced_row(r) for r in rows_c0],
        rows_c1=[_reduced_row(r) for r in rows_c1],
        deltas=deltas,
    ), p_c0, p_c1, (f0, e1_0, e2_0, e3_0)


def run_rate_series(args, p_c0_ref, init_state, n_ref):
    f0, e1_0, e2_0, e3_0 = init_state
    print(f"\n=== compact coarsening-rate series, N_ref={n_ref} steps (matched physical time) ===")
    out = {}
    for rate in args.rate_series:
        p = build_params(build_config(rate, args))
        assert p.dt == p_c0_ref.dt
        rows, stop, reason = run_single_trajectory(p, f0, e1_0, e2_0, e3_0, n_ref)
        v20 = rows[0]["V2"]
        dv2 = (rows[-1]["V2"] - v20) / v20
        L0, L1 = rows[0]["L_contact_TJ_sub"], rows[-1]["L_contact_TJ_sub"]
        print(f"  rate={rate:5g}: dV2/V20={dv2:+.4e}  L_contact_TJ_sub: {L0*1e9:.4f} -> {L1*1e9:.4f}nm "
              f"(d={((L1-L0)*1e9):+.5f}nm)  stop={stop}")
        out[f"{rate:g}"] = dict(
            n_steps=len(rows) - 1, stop=stop, reason=reason, V20=v20, dV2_over_V20=dv2,
            rows=[_reduced_row(r) for r in rows],
        )
    return out


def dt_dx_audit(args):
    print("\n=== CH dt/dx stability audit ===")
    out = {}
    for dx_nm in (5.0, 2.5, 1.25):
        p = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm))
        dt_raw = p.CFL * p.dx**4 / (p.M_f * p.k_f)
        clamp_hit = dt_raw > 1e-5
        print(f"  dx={dx_nm:5.2f}nm: dt_raw(pre-clamp,pre-aniso)={dt_raw:.4e}s  "
              f"1e-5-clamp active={clamp_hit}  final dt={p.dt:.6e}s  M_f*k_f={p.M_f*p.k_f:.6e}")
        out[f"{dx_nm:g}"] = dict(dt_raw=dt_raw, clamp_active=clamp_hit, dt_final=p.dt, M_f_k_f=p.M_f * p.k_f)
    # crossover dx solving CFL*dx^4/(M_f*k_f) == 1e-5, using the dx-independent M_f*k_f product
    p_ref = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=5.0))
    mfkf = p_ref.M_f * p_ref.k_f
    dx_crossover = (1e-5 * mfkf / p_ref.CFL) ** 0.25
    print(f"  crossover dx (CFL formula alone reaches the 1e-5 ceiling): {dx_crossover*1e9:.4f}nm")
    out["dx_crossover_nm"] = dx_crossover * 1e9
    return out


def _matched_time_bench(args, dx_nm, dt_fracs, n_bench):
    p_natural = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm))
    dt_natural = p_natural.dt
    f0, e1_0, e2_0, e3_0 = initialize_fields(p_natural)
    results = {}
    for frac_name, frac in dt_fracs:
        dt_i = dt_natural * frac
        n_i = round(n_bench / frac)
        p = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, dt_override=dt_i))
        rows, stop, reason = run_single_trajectory(p, f0, e1_0, e2_0, e3_0, n_i)
        last = rows[-1]
        total_f = None  # filled by caller if needed
        print(f"    dt={frac_name} (dt={dt_i:.4e}s, n_steps={n_i}, t_final={n_i*dt_i:.4e}s): "
              f"L_contact_TJ_sub={last['L_contact_TJ_sub']*1e9:.6f}nm "
              f"L_GB_geom_sub={last['L_GB_geom_sub']*1e9:.6f}nm "
              f"E_surf_J={last['E_surf_J']:.6e} E_gb_J={last['E_gb_J']:.6e} "
              f"G_interface_J={last['G_interface_J']:.6e} stop={stop}")
        results[frac_name] = dict(
            dt=dt_i, n_steps=n_i, t_final=n_i * dt_i, stop=stop, reason=reason,
            L_contact_TJ_sub=last["L_contact_TJ_sub"], L_GB_geom_sub=last["L_GB_geom_sub"],
            E_surf_J=last["E_surf_J"], E_gb_J=last["E_gb_J"], G_interface_J=last["G_interface_J"],
            V2=last["V2"],
        )
    return dict(dt_natural=dt_natural, n_bench=n_bench, results=results)


def _matched_time_paired_bench(args, dx_nm, dt_fracs, n_bench):
    """Same idea as _matched_time_bench but for the PAIRED (C0,C1) differential
    quantity delta_L_coarsening, to test directly whether the subtraction
    converges faster in dt than the raw C1 trajectory does (handoff Section
    14: 'subtraction may cancel some of the large common CH discretization
    error')."""
    p_c0_natural = build_params(build_config(0.0, args, dx_nm=dx_nm))
    dt_natural = p_c0_natural.dt
    f0, e1_0, e2_0, e3_0 = initialize_fields(p_c0_natural)
    results = {}
    for frac_name, frac in dt_fracs:
        dt_i = dt_natural * frac
        n_i = round(n_bench / frac)
        p_c0 = build_params(build_config(0.0, args, dx_nm=dx_nm, dt_override=dt_i))
        p_c1 = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, dt_override=dt_i))
        rows_c0, stop0, _ = run_single_trajectory(p_c0, f0, e1_0, e2_0, e3_0, n_i)
        rows_c1, stop1, _ = run_single_trajectory(p_c1, f0, e1_0, e2_0, e3_0, n_i)
        d_final = paired_delta_series(rows_c0[-1:], rows_c1[-1:])[0]
        print(f"    dt={frac_name} (dt={dt_i:.4e}s, n_steps={n_i}): "
              f"L_contact_TJ_sub C0={rows_c0[-1]['L_contact_TJ_sub']*1e9:.6f}nm "
              f"C1={rows_c1[-1]['L_contact_TJ_sub']*1e9:.6f}nm "
              f"delta_L_coarsening={d_final['L_contact_TJ_sub']*1e9:+.6f}nm stop={stop0 or stop1}")
        results[frac_name] = dict(dt=dt_i, n_steps=n_i, delta_L_coarsening=d_final["L_contact_TJ_sub"],
                                   L_contact_TJ_sub_C0=rows_c0[-1]["L_contact_TJ_sub"],
                                   L_contact_TJ_sub_C1=rows_c1[-1]["L_contact_TJ_sub"],
                                   stop=stop0 or stop1)
    return dict(dt_natural=dt_natural, n_bench=n_bench, results=results)


def timestep_sensitivity(args):
    print("\n=== fixed-grid timestep-sensitivity (matched physical time) ===")
    out = {}
    for dx_nm in (5.0, 2.5):
        print(f"  -- dx={dx_nm}nm (raw C1 trajectory) --")
        out[f"{dx_nm:g}"] = _matched_time_bench(
            args, dx_nm, [("dt", 1.0), ("dt/2", 0.5), ("dt/4", 0.25), ("dt/8", 0.125)],
            n_bench=args.dt_bench_steps,
        )
        print(f"  -- dx={dx_nm}nm (paired differential delta_L_coarsening) --")
        out[f"{dx_nm:g}_paired"] = _matched_time_paired_bench(
            args, dx_nm, [("dt", 1.0), ("dt/2", 0.5), ("dt/4", 0.25), ("dt/8", 0.125)],
            n_bench=args.dt_bench_steps,
        )
    return out


def dt_converged_grid_comparison(args, dt_frac_5nm, dt_frac_25nm):
    print(f"\n=== dt-converged coarse/fine grid paired comparison "
          f"(dx=5nm @ {dt_frac_5nm}x natural dt, dx=2.5nm @ {dt_frac_25nm}x natural dt) ===")
    out = {}
    for dx_nm, dt_frac in ((5.0, dt_frac_5nm), (2.5, dt_frac_25nm)):
        p_natural = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm))
        dt_i = p_natural.dt * dt_frac
        p_c0 = build_params(build_config(0.0, args, dx_nm=dx_nm, dt_override=dt_i))
        p_c1 = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, dt_override=dt_i))
        f0, e1_0, e2_0, e3_0 = initialize_fields(p_c0)
        rows_c0, rows_c1, reason = run_paired_trajectory(
            p_c0, p_c1, f0, e1_0, e2_0, e3_0,
            target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps * 4,
        )
        v20 = rows_c0[0]["V2"]
        d = paired_delta_series(rows_c0, rows_c1)[-1]
        dv2 = (rows_c1[-1]["V2"] - v20) / v20
        print(f"  dx={dx_nm}nm dt={dt_i:.4e}s: n_steps={len(rows_c1)-1} dV2/V20={dv2:+.4e} "
              f"delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm "
              f"dL_contact_TJ_sub_C1/dt~{(rows_c1[-1]['L_contact_TJ_sub']-rows_c1[0]['L_contact_TJ_sub'])/(len(rows_c1)-1)/dt_i:.4e} m/s")
        out[f"{dx_nm:g}"] = dict(
            dt=dt_i, n_steps=len(rows_c1) - 1, dV2_over_V20=dv2, delta_final=d,
            c0_final=_reduced_row(rows_c0[-1]), c1_final=_reduced_row(rows_c1[-1]),
        )
    return out


def _gl_wc(e1, e2, e3, s, p):
    pair = np.maximum(0.0, e1 * e2)
    gl = np.full_like(e1, p.gamma_gb_ref)
    mask = pair > 1e-20
    if np.any(mask):
        gl[mask] = effective_gamma(s, p)
    return 36.0 * gl / p.interface_width


def exact_free_energy_isotropic(f, e1, e2, e3, s, p):
    """Isotropic-mode (p.use_aniso_surface must be False) exact discrete free
    energy F = sum[(W_f/2) f^2 (1-f)^2 + Wc*eta2*(f^2/2 - f) + (k_f/2)|grad f|^2] * dx^2.
    Reconstructed so that its continuum functional derivative reproduces the
    production mu0 term (see differential_coarsening_audit.py docstring /
    MILESTONE_7 report Section 9 for the derivation): d/df[(W_f/2)f^2(1-f)^2]
    = W_f*f*(1-f)*(1-2f); d/df[Wc*eta2*(f^2/2-f)] = Wc*eta2*(f-1) =
    -Wc*eta2*(1-f), matching mu0 = W_f*f(1-f)(1-2f) - Wc*eta2*(1-fb) (fb~=f
    away from the [0,1] clip boundary). The gradient term here uses the
    model's own central-difference grad() for (k_f/2)|grad f|^2; production
    mu instead differentiates via the 9-point Laplacian stencil (lap9), which
    is NOT exactly the discrete Euler-Lagrange derivative of this particular
    discrete Dirichlet-energy sum, so dF/dt<=0 is expected to good
    approximation from the underlying continuum gradient-flow structure, not
    guaranteed to floating-point closure by discrete summation-by-parts."""
    fb = np.clip(f, 0.0, 1.0)
    eta2 = e1 * e1 + e2 * e2 + e3 * e3
    Wc = _gl_wc(e1, e2, e3, s, p)
    e_bulk = 0.5 * p.W_f * f * f * (1 - f) ** 2 + Wc * eta2 * (0.5 * f * f - f)
    gx, gy = grad(f, p.dx)
    e_grad = 0.5 * p.k_f * (gx * gx + gy * gy)
    return float(np.sum(e_bulk + e_grad)) * p.dx * p.dx


def free_energy_check(args, n_steps=80):
    print(f"\n=== isotropic-mode exact free-energy check (C0, n_steps={n_steps}) ===")
    cfg = build_config(0.0, args)
    cfg.use_aniso_surface = False
    p = build_params(cfg)
    f, e1, e2, e3 = initialize_fields(p)
    s = Sink(threshold=math.inf)
    F = [exact_free_energy_isotropic(f, e1, e2, e3, s, p)]
    total_f = [float(f.sum()) * p.dx * p.dx]
    for step in range(1, n_steps + 1):
        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, Sink(), p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)
        f, e1, e2, e3 = ostwald_substrate(f, e1, e2, e3, p)
        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)
        F.append(exact_free_energy_isotropic(f, e1, e2, e3, s, p))
        total_f.append(float(f.sum()) * p.dx * p.dx)
    increases = [i for i in range(1, len(F)) if F[i] > F[i - 1] + 1e-22]
    print(f"  F_exact[0]={F[0]:.8e}  F_exact[-1]={F[-1]:.8e}  "
          f"monotonic_non_increasing={len(increases)==0}  n_increasing_steps={len(increases)}")
    if increases:
        worst = max(increases, key=lambda i: F[i] - F[i - 1])
        print(f"  largest single-step increase at step {worst}: "
              f"{F[worst]-F[worst-1]:.4e} (vs F scale {abs(F[0]):.4e})")
    print(f"  total_f drift: {total_f[-1]-total_f[0]:.4e} (relative {(total_f[-1]-total_f[0])/total_f[0]:.4e})")
    return dict(F_exact=F, total_f=total_f, n_increasing_steps=len(increases))


def run_baseline(args):
    print(f"\n=== 5nm-overlap paired cross-check ===")
    p_c0 = build_params(build_config(0.0, args, overlap_nm=args.baseline_overlap_nm))
    p_c1 = build_params(build_config(args.coarsening_rate_scale, args, overlap_nm=args.baseline_overlap_nm))
    assert p_c0.dt == p_c1.dt
    f0, e1_0, e2_0, e3_0 = initialize_fields(p_c0)
    rows_c0, rows_c1, reason = run_paired_trajectory(
        p_c0, p_c1, f0, e1_0, e2_0, e3_0,
        target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps,
    )
    v20 = rows_c0[0]["V2"]
    d = paired_delta_series(rows_c0, rows_c1)[-1]
    dv2 = (rows_c1[-1]["V2"] - v20) / v20
    print(f"  n_steps={len(rows_c1)-1} dV2/V20={dv2:+.4e} "
          f"delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm "
          f"delta_L_GB_geom_sub={d['L_GB_geom_sub']*1e9:+.6f}nm stop_reason={reason!r}")
    return dict(
        n_steps=len(rows_c1) - 1, V20=v20, dV2_over_V20=dv2, delta_final=d, stop_reason=reason,
        c0_final=_reduced_row(rows_c0[-1]), c1_final=_reduced_row(rows_c1[-1]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--baseline-overlap-nm", type=float, default=5.0)
    ap.add_argument("--coarsening-rate-scale", type=float, default=3.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--eta-mobility-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--extended-dv2-frac", type=float, default=1.5e-3)
    ap.add_argument("--extended-dv2-frac-2", type=float, default=3e-3)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--rate-series", type=float, nargs="+", default=[0.0, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--dt-bench-steps", type=int, default=150)
    ap.add_argument("--dt-frac-5nm", type=float, default=1.0)
    ap.add_argument("--dt-frac-25nm", type=float, default=0.25)
    ap.add_argument("--skip-fine-grid", action="store_true")
    ap.add_argument("--out", type=str, default="runs/differential_coarsening_audit.json")
    args = ap.parse_args()

    out = dict(args=vars(args))

    primary, p_c0, p_c1, init_state = run_primary(args)
    out["primary"] = primary

    n_ref = primary["milestones"][f"{args.target_dv2_frac:g}"]["idx"]
    out["rate_series"] = run_rate_series(args, p_c0, init_state, n_ref)
    out["rate_series_n_ref"] = n_ref

    out["dt_dx_audit"] = dt_dx_audit(args)
    out["dt_sensitivity"] = timestep_sensitivity(args)

    # dt-converged comparison: chosen fractions filled in after inspecting
    # dt_sensitivity's convergence pattern; default to natural dt at both
    # grids first (frac=1.0), the report documents whether this is adequate.
    out["dt_converged_grid_comparison"] = dt_converged_grid_comparison(
        args, args.dt_frac_5nm, args.dt_frac_25nm)

    out["free_energy_isotropic"] = free_energy_check(args)

    out["baseline"] = run_baseline(args)

    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
