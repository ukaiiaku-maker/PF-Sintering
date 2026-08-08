"""Milestone 10 -- periodic sinusoidal substrate geometry audit.

DIAGNOSTIC ONLY. Produces the data behind
MILESTONE_10_PERIODIC_SINUSOIDAL_SUBSTRATE.md:

6. initial-geometry validation dump;
7. S0 (no coarsening, capillary-only) trajectory;
8. S1 (coarsening -> external reservoir) trajectory;
9. S2 (coarsening -> explicit local sinusoidal receiver, i.e. Model A/
   production Ostwald applied to the curved e1 field) trajectory;
10. S0/S1/S2 comparison at matched physical time;
11. sinusoid Fourier/amplitude evolution tracking;
13. small coarsening-rate series (S1 closure);
14. fixed-physics grid check (S1 and S2).

Does not modify production physics.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial

import numpy as np

from pf_sintering.differential_coarsening import full_state_sample, run_single_trajectory
from pf_sintering.diagnostics import wall_x0
from pf_sintering.model import ModelConfig, Sink, build_params, compute_stress, evolve_eta, evolve_f, initialize_fields
from pf_sintering.ostwald_receiver_closures import ostwald_external_reservoir_step
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving
from pf_sintering.tj_subgrid import compute_subgrid_contact
from pf_sintering.tj_force import compute_neck_tj_forces


def build_config(rate, args, dx_nm=None, w_nm=None, fixed_eta=False, dt_override=None):
    return ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", nx=args.nx, ny=args.ny,
        dx=(dx_nm if dx_nm is not None else args.dx_nm) * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=args.overlap_nm * 1e-9, t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=rate, surface_mobility_scale=args.surface_mobility_scale,
        eta_mobility_scale=args.eta_mobility_scale,
        sinusoid_wavelength=args.wavelength_nm * 1e-9, sinusoid_amplitude=args.amplitude_nm * 1e-9,
        interface_width_override=(w_nm if w_nm is not None else args.fixed_w_nm) * 1e-9,
        eta_diffusivity_fixed_physical=fixed_eta,
        dt_override=dt_override,
    )


def _step_with_reservoir(f, e1, e2, e3, s, p, phi_local):
    ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f1 = evolve_f(f, e1, e2, e3, s, Sink(), p)
    e1a, e2a, e3a = project_eta_mass_preserving(f1, e1, e2, e3, p, target_masses=ch_targets)

    f2, e1b, e2b, e3b, reservoir_delta = ostwald_external_reservoir_step(f1, e1a, e2a, e3a, p, phi_local=phi_local)

    ac_targets = eta_masses(e1b, e2b, e3b, p.use_eta3)
    e1c, e2c, e3c = evolve_eta(e1b, e2b, e3b, p)
    e1d, e2d, e3d = project_eta_mass_preserving(f2, e1c, e2c, e3c, p, target_masses=ac_targets)
    return f2, e1d, e2d, e3d, reservoir_delta


def run_s1_trajectory(p, f0, e1_0, e2_0, e3_0, n_steps):
    """S1: coarsening on, addition entirely diverted to an external
    reservoir (Milestone 9 Model B), applied to the sinusoidal geometry."""
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    wall_x0_val = wall_x0(p)
    v20 = float(e2.sum() * p.dx * p.dx)
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    rows = [full_state_sample(f, e1, e2, e3, s, st, p, 0, 0.0, v20, wall_x0_val)]
    reservoir = 0.0
    reservoir_series = [0.0]
    for step in range(1, n_steps + 1):
        f, e1, e2, e3, rd = _step_with_reservoir(f, e1, e2, e3, s, p, phi_local=0.0)
        reservoir += rd
        st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
        rows.append(full_state_sample(f, e1, e2, e3, s, st, p, step, step * p.dt, v20, wall_x0_val))
        reservoir_series.append(reservoir)
        if stop:
            return rows, reservoir_series, True, reason
    return rows, reservoir_series, False, ""


def _reduced(row):
    keys = ("step", "time_s", "V2", "V2_f_eta", "x_neck_m", "A_GB_m2",
            "L_contact_TJ_sub", "L_GB_geom_sub", "psi_deg_top", "psi_deg_bottom",
            "kappa_top_1pm", "kappa_bottom_1pm", "F_TJ_mag_top", "F_TJ_mag_bottom",
            "sigma_Pa", "sigma_lt_Pa", "sigma_curv_Pa", "tj_top_x_sub", "tj_top_y_sub",
            "tj_bottom_x_sub", "tj_bottom_y_sub")
    return {k: row.get(k) for k in keys}


def geometry_validation(args):
    print("\n=== Section 6: initial geometry validation ===")
    p = build_params(build_config(args.coarsening_rate_scale, args))
    f, e1, e2, e3 = initialize_fields(p)
    print(f"  Nx={p.Nx} Ny={p.Ny} dx={p.dx*1e9:.2f}nm wavelength={p.sinusoid_wavelength*1e9:.2f}nm "
          f"amplitude={p.sinusoid_amplitude*1e9:.2f}nm W={p.interface_width*1e9:.2f}nm dt={p.dt:.4e}s")
    s = Sink(threshold=math.inf)
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    print(f"  compute_stress: stop={stop} reason={reason!r} sigma={st.sigma/1e6:.4f}MPa "
          f"x_neck={st.x_neck*1e9:.4f}nm psi_legacy_deg={math.degrees(st.psi):.4f}")
    sub = compute_subgrid_contact(f, e1, e2, p)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    print(f"  sub.resolved={sub.resolved} L_contact_TJ_sub={sub.L_contact_TJ_sub*1e9:.4f}nm "
          f"L_GB_geom_sub={sub.L_GB_geom_sub*1e9:.4f}nm")
    print(f"  tj_force resolved n={rep.n_resolved} top_psi={rep.top.psi_deg if rep.top else None} "
          f"bottom_psi={rep.bottom.psi_deg if rep.bottom else None}")
    V1 = float(e1.sum()) * p.dx * p.dx
    V2 = float(e2.sum()) * p.dx * p.dx
    print(f"  V1={V1:.6e}m^2 V2={V2:.6e}m^2")
    return dict(
        Nx=p.Nx, Ny=p.Ny, dx=p.dx, wavelength=p.sinusoid_wavelength, amplitude=p.sinusoid_amplitude,
        W=p.interface_width, dt=p.dt, sigma_Pa=st.sigma, x_neck_m=st.x_neck, psi_legacy_deg=math.degrees(st.psi),
        sub_resolved=sub.resolved, L_contact_TJ_sub=sub.L_contact_TJ_sub, L_GB_geom_sub=sub.L_GB_geom_sub,
        V1=V1, V2=V2,
    )


def sinusoid_fourier_fit(f, p, y_exclude_half_width, n_harmonics=3):
    """Fit x_s(Y) = x_mean + sum_k [A_k cos(k*2*pi*Y/lambda) + B_k sin(...)]
    to the f=0.5 crossing column of each row with |Y| > y_exclude_half_width
    (away from the particle), via least squares."""
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
    ap.add_argument("--target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--rate-series", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--dx25-dt-frac", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="runs/sinusoidal_substrate_audit.json")
    args = ap.parse_args()

    out = dict(args=vars(args))
    out["geometry_validation"] = geometry_validation(args)

    p0 = build_params(build_config(0.0, args))
    p1 = build_params(build_config(args.coarsening_rate_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)
    assert p0.dt == p1.dt

    # find n_ref by running S2 (production Ostwald) to target -- also gives S2's trajectory directly
    print("\n=== Section 9: S2 (explicit local sinusoidal receiver, production Ostwald) ===")
    v20 = float(e2_0.sum()) * p1.dx * p1.dx
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    step = 0
    rows_s2 = [full_state_sample(f, e1, e2, e3, s, compute_stress(f, e1, e2, e3, s, p1)[0], p1, 0, 0.0, v20, wall_x0(p1))]
    while step < args.max_steps:
        step += 1
        ch_targets = eta_masses(e1, e2, e3, p1.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, Sink(), p1)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p1, target_masses=ch_targets)
        from pf_sintering.model import ostwald_substrate
        f, e1, e2, e3 = ostwald_substrate(f, e1, e2, e3, p1)
        ac_targets = eta_masses(e1, e2, e3, p1.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p1)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p1, target_masses=ac_targets)
        st, stop, reason = compute_stress(f, e1, e2, e3, s, p1)
        rows_s2.append(full_state_sample(f, e1, e2, e3, s, st, p1, step, step * p1.dt, v20, wall_x0(p1)))
        v2 = float(e2.sum()) * p1.dx * p1.dx
        if abs(v2 - v20) / v20 >= args.target_dv2_frac or stop:
            break
    n_ref = step
    d_s2 = rows_s2[-1]["L_contact_TJ_sub"] - rows_s2[0]["L_contact_TJ_sub"]
    print(f"  n_ref={n_ref} dV2/V20={(v2-v20)/v20:+.4e} dL_contact_TJ_sub_S2={d_s2*1e9:+.6f}nm")
    out["n_ref"] = n_ref
    out["S2_final"] = _reduced(rows_s2[-1])
    out["S2_trajectory"] = [_reduced(r) for r in rows_s2]

    print("\n=== Section 7: S0 (no coarsening, capillary-only control) ===")
    rows_s0, stop0, reason0 = run_single_trajectory(p0, f0, e1_0, e2_0, e3_0, n_ref)
    d_s0 = rows_s0[-1]["L_contact_TJ_sub"] - rows_s0[0]["L_contact_TJ_sub"]
    print(f"  n_steps={n_ref} dV2/V20={(rows_s0[-1]['V2']-v20)/v20:+.4e} dL_contact_TJ_sub_S0={d_s0*1e9:+.6f}nm stop={stop0}")
    out["S0_final"] = _reduced(rows_s0[-1])
    out["S0_trajectory"] = [_reduced(r) for r in rows_s0]

    print("\n=== Section 8: S1 (coarsening -> external reservoir) ===")
    rows_s1, reservoir_series, stop1, reason1 = run_s1_trajectory(p1, f0, e1_0, e2_0, e3_0, n_ref)
    d_s1 = rows_s1[-1]["L_contact_TJ_sub"] - rows_s1[0]["L_contact_TJ_sub"]
    print(f"  n_steps={n_ref} dV2/V20={(rows_s1[-1]['V2']-v20)/v20:+.4e} dL_contact_TJ_sub_S1={d_s1*1e9:+.6f}nm "
          f"reservoir_final={reservoir_series[-1]:.4e} stop={stop1}")
    out["S1_final"] = _reduced(rows_s1[-1])
    out["S1_trajectory"] = [_reduced(r) for r in rows_s1]
    out["S1_reservoir_final"] = reservoir_series[-1]

    print("\n=== Section 10: S0/S1/S2 comparison at matched physical time ===")
    delta_S1_vs_S0 = rows_s1[-1]["L_contact_TJ_sub"] - rows_s0[-1]["L_contact_TJ_sub"]
    delta_S2_vs_S0 = rows_s2[-1]["L_contact_TJ_sub"] - rows_s0[-1]["L_contact_TJ_sub"]
    print(f"  delta_L_coarsening (S1-S0) = {delta_S1_vs_S0*1e9:+.6f}nm")
    print(f"  delta_L_coarsening (S2-S0) = {delta_S2_vs_S0*1e9:+.6f}nm")
    out["delta_S1_vs_S0"] = delta_S1_vs_S0
    out["delta_S2_vs_S0"] = delta_S2_vs_S0

    print("\n=== Section 11: sinusoid Fourier tracking (S1 trajectory, start vs end) ===")
    y_excl = p1.Ry + 3 * p1.interface_width
    # need raw fields at start/end -- recompute end fields for S1 to fit contour (rows only carry scalars)
    f_end_s1, e1_end_s1, e2_end_s1, e3_end_s1 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    for step in range(1, n_ref + 1):
        f_end_s1, e1_end_s1, e2_end_s1, e3_end_s1, _ = _step_with_reservoir(
            f_end_s1, e1_end_s1, e2_end_s1, e3_end_s1, s, p1, phi_local=0.0)
    fit_start = sinusoid_fourier_fit(f0, p1, y_excl)
    fit_end = sinusoid_fourier_fit(f_end_s1, p1, y_excl)
    print(f"  y_exclude_half_width={y_excl*1e9:.2f}nm")
    print(f"  fit_start: {fit_start}")
    print(f"  fit_end (S1): {fit_end}")
    out["fourier_fit_start"] = fit_start
    out["fourier_fit_end_S1"] = fit_end

    print("\n=== Section 12: RBM/strain check ===")
    print(f"  S0/S1/S2 all used Sink(threshold=inf), hazard_step/rbm never called -- "
          f"cumulative_disp/cumulative_strain identically 0 throughout by construction.")
    out["rbm_strain_check"] = "hazard/rbm never called in any trajectory; strain=disp=0 by construction"

    print(f"\n=== Section 13: coarsening-rate series (S1 closure, N_ref={n_ref}, vs shared S0) ===")
    rate_out = {}
    for rate in args.rate_series:
        p_r = build_params(build_config(rate, args))
        rows_r, _, stop_r, _ = run_s1_trajectory(p_r, f0, e1_0, e2_0, e3_0, n_ref)
        d_abs = rows_r[-1]["L_contact_TJ_sub"] - rows_r[0]["L_contact_TJ_sub"]
        d_vs_s0 = rows_r[-1]["L_contact_TJ_sub"] - rows_s0[-1]["L_contact_TJ_sub"]
        dv2 = (rows_r[-1]["V2"] - v20) / v20
        print(f"  rate={rate:g}: dV2/V20={dv2:+.4e} dL_contact_TJ_sub(S1,abs)={d_abs*1e9:+.6f}nm "
              f"delta_vs_S0={d_vs_s0*1e9:+.6f}nm stop={stop_r}")
        rate_out[f"{rate:g}"] = dict(dV2_over_V20=dv2, dL_contact_TJ_sub_abs=d_abs, delta_vs_S0=d_vs_s0, stop=stop_r)
    # include rate=coarsening_rate_scale's already-computed S1 for a complete series point
    d_abs_primary = rows_s1[-1]["L_contact_TJ_sub"] - rows_s1[0]["L_contact_TJ_sub"]
    rate_out[f"{args.coarsening_rate_scale:g}"] = dict(
        dV2_over_V20=(rows_s1[-1]["V2"] - v20) / v20, dL_contact_TJ_sub_abs=d_abs_primary,
        delta_vs_S0=delta_S1_vs_S0, stop=stop1,
    )
    out["rate_series_S1"] = rate_out

    print("\n=== Section 14: fixed-physics grid check (S1 and S2) ===")
    grid_out = {}
    for dx_nm, dt_frac in ((5.0, 1.0), (2.5, args.dx25_dt_frac)):
        p_nat = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, fixed_eta=True))
        dt_i = p_nat.dt * dt_frac
        p0g = build_params(build_config(0.0, args, dx_nm=dx_nm, fixed_eta=True, dt_override=dt_i))
        p1g = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, fixed_eta=True, dt_override=dt_i))
        f0g, e1g, e2g, e3g = initialize_fields(p0g)
        v20g = float(e2g.sum()) * p0g.dx * p0g.dx

        # S1 at this grid
        rows_s1g, _, stopg, _ = None, None, None, None
        f, e1c, e2c, e3c = (a.copy() for a in (f0g, e1g, e2g, e3g))
        s = Sink(threshold=math.inf)
        step = 0
        while step < args.max_steps:
            step += 1
            f, e1c, e2c, e3c, _ = _step_with_reservoir(f, e1c, e2c, e3c, s, p1g, phi_local=0.0)
            v2g = float(e2c.sum()) * p1g.dx * p1g.dx
            if abs(v2g - v20g) / v20g >= args.target_dv2_frac:
                break
        n_refg = step
        sub_end_s1 = compute_subgrid_contact(f, e1c, e2c, p1g)
        sub_start = compute_subgrid_contact(f0g, e1g, e2g, p0g)

        rows_s0g, _, _ = run_single_trajectory(p0g, f0g, e1g, e2g, e3g, n_refg)
        d_s1g = sub_end_s1.L_contact_TJ_sub - rows_s0g[0]["L_contact_TJ_sub"]
        d_S1_vs_S0_g = sub_end_s1.L_contact_TJ_sub - rows_s0g[-1]["L_contact_TJ_sub"]

        # S2 at this grid
        f2, e1c2, e2c2, e3c2 = (a.copy() for a in (f0g, e1g, e2g, e3g))
        from pf_sintering.model import ostwald_substrate as _ostwald_prod
        for step in range(1, n_refg + 1):
            ch_targets = eta_masses(e1c2, e2c2, e3c2, p1g.use_eta3)
            f2 = evolve_f(f2, e1c2, e2c2, e3c2, s, Sink(), p1g)
            e1c2, e2c2, e3c2 = project_eta_mass_preserving(f2, e1c2, e2c2, e3c2, p1g, target_masses=ch_targets)
            f2, e1c2, e2c2, e3c2 = _ostwald_prod(f2, e1c2, e2c2, e3c2, p1g)
            ac_targets = eta_masses(e1c2, e2c2, e3c2, p1g.use_eta3)
            e1c2, e2c2, e3c2 = evolve_eta(e1c2, e2c2, e3c2, p1g)
            e1c2, e2c2, e3c2 = project_eta_mass_preserving(f2, e1c2, e2c2, e3c2, p1g, target_masses=ac_targets)
        sub_end_s2 = compute_subgrid_contact(f2, e1c2, e2c2, p1g)
        d_S2_vs_S0_g = sub_end_s2.L_contact_TJ_sub - rows_s0g[-1]["L_contact_TJ_sub"]

        print(f"  dx={dx_nm}nm dt={dt_i:.4e}s n_ref={n_refg}: "
              f"delta(S1-S0)={d_S1_vs_S0_g*1e9:+.6f}nm  delta(S2-S0)={d_S2_vs_S0_g*1e9:+.6f}nm")
        grid_out[f"{dx_nm:g}"] = dict(n_ref=n_refg, dt=dt_i, delta_S1_vs_S0=d_S1_vs_S0_g, delta_S2_vs_S0=d_S2_vs_S0_g)
    out["fixed_physics_grid"] = grid_out

    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
