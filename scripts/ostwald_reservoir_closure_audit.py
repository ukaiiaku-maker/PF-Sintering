"""Milestone 9 -- Ostwald receiver/reservoir closure audit.

DIAGNOSTIC ONLY. Produces the data behind
MILESTONE_9_OSTWALD_RESERVOIR_CLOSURE_AUDIT.md:

5. mass-closure audit (receiver advance vs. particle recession);
6. spatial partition of the exact production addition field, applied to a
   full trajectory (not renormalized);
7. one-step differential TJ response to each addition region and to
   removal-only/addition-only;
8. (derived from 6/7, not separately computed) R1 vs. R2 classification;
9/10. external-reservoir bookkeeping closures (Models A/B/C);
12. paired C0/C1 trajectory comparison of the closures;
13. corrected fixed-physics grid check (Milestone 8's eta-diffusivity bug
    fixed in model.py before this script is run).

Does not modify production physics.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial

import numpy as np
from skimage.measure import find_contours

from pf_sintering.differential_coarsening import (
    PAIRED_KEYS,
    full_state_sample,
    paired_delta,
    paired_delta_series,
    run_paired_trajectory,
    run_single_trajectory,
)
from pf_sintering.model import (
    ModelConfig,
    Sink,
    build_params,
    compute_stress,
    evolve_eta,
    evolve_f,
    initialize_fields,
)
from pf_sintering.ostwald_diagnostics import (
    ostwald_addition_only,
    ostwald_removal_only,
    ostwald_substrate_diagnostic,
)
from pf_sintering.ostwald_receiver_closures import (
    ostwald_external_reservoir_step,
    ostwald_partial_addition_step,
    region_masks_by_tj_distance,
)
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_config(rate, args, dx_nm=None, w_nm=None, fixed_eta=False, dt_override=None, overlap_nm=None):
    cfg = ModelConfig(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny,
        dx=(dx_nm if dx_nm is not None else args.dx_nm) * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        initial_overlap=(overlap_nm if overlap_nm is not None else args.overlap_nm) * 1e-9,
        t_total=1e-6, seed=args.seed,
        coarsening_rate_scale=rate, surface_mobility_scale=args.surface_mobility_scale,
        eta_mobility_scale=args.eta_mobility_scale, dt_override=dt_override,
    )
    if w_nm is not None:
        cfg.interface_width_override = w_nm * 1e-9
        cfg.eta_diffusivity_fixed_physical = fixed_eta
    return cfg


def _reduced(row):
    return {k: row.get(k) for k in ("step", "time_s", "V2") + PAIRED_KEYS}


def _contour_length(field, level, p):
    total = 0.0
    for rc in find_contours(field, level):
        pts = np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx]
        d = np.diff(pts, axis=0)
        total += float(np.sum(np.hypot(d[:, 0], d[:, 1])))
    return total


# ---------------------------------------------------------------------------
# Section 5: mass-closure audit
# ---------------------------------------------------------------------------

def mass_closure_audit(args):
    print("\n=== Section 5: mass-closure audit (one ordinary Ostwald step) ===")
    p = build_params(build_config(args.coarsening_rate_scale, args))
    f, e1, e2, e3 = initialize_fields(p)
    f2, e1n, e2n, e3n, diag = ostwald_substrate_diagnostic(f, e1, e2, e3, p)

    dV2 = -diag.actual_removed * p.dx * p.dx
    dV1 = diag.actual_added * p.dx * p.dx
    d_total_f = float((f2 - f).sum()) * p.dx * p.dx
    L_e1 = _contour_length(e1, 0.5, p)
    L_e2 = _contour_length(e2, 0.5, p)
    mean_advance_receiver = dV1 / L_e1 if L_e1 > 0 else math.nan
    mean_recession_particle = dV2 / L_e2 if L_e2 > 0 else math.nan

    print(f"  actual_removed={diag.actual_removed:.6e}  actual_added={diag.actual_added:.6e}  capped={diag.capped}")
    print(f"  dV2={dV2:.6e}m^2  dV1={dV1:.6e}m^2  d_total_local_f={d_total_f:.3e}m^2  dV1+dV2={dV1+dV2:.3e}")
    print(f"  receiving (e1) contour length={L_e1*1e9:.2f}nm  particle (e2) contour length={L_e2*1e9:.2f}nm")
    print(f"  mean normal advance on receiver (spread over full e1 contour): {mean_advance_receiver*1e9:.6e}nm")
    print(f"  mean normal recession on particle (spread over full e2 contour): {mean_recession_particle*1e9:.6e}nm")
    print(f"  R2={p.R2*1e9:.1f}nm  domain=({p.Nx*p.dx*1e9:.1f} x {p.Ny*p.dx*1e9:.1f})nm")

    return dict(
        actual_removed=diag.actual_removed, actual_added=diag.actual_added, capped=diag.capped,
        dV2=dV2, dV1=dV1, d_total_local_f=d_total_f,
        L_e1_contour_nm=L_e1 * 1e9, L_e2_contour_nm=L_e2 * 1e9,
        mean_advance_receiver_nm=mean_advance_receiver * 1e9,
        mean_recession_particle_nm=mean_recession_particle * 1e9,
        R2_nm=p.R2 * 1e9, domain_nm=(p.Nx * p.dx * 1e9, p.Ny * p.dx * 1e9),
    )


# ---------------------------------------------------------------------------
# Section 6: spatial partition of the addition field, trajectory-level
# ---------------------------------------------------------------------------

def _region_only_ostwald_fn(band_label, band_edges):
    def fn(f, e1, e2, e3, p):
        sub = compute_subgrid_contact(f, e1, e2, p)
        if not sub.resolved:
            f2, e1n, e2n, e3n, _ = ostwald_substrate_diagnostic(f, e1, e2, e3, p)
            return f2, e1n, e2n, e3n
        tj_top = (sub.top.x_sub, sub.top.y_sub)
        tj_bot = (sub.bottom.x_sub, sub.bottom.y_sub)
        masks = region_masks_by_tj_distance(p, tj_top, tj_bot, band_edges)
        f2, e1n, e2n, e3n, _ = ostwald_partial_addition_step(f, e1, e2, e3, p, masks[band_label])
        return f2, e1n, e2n, e3n
    return fn


def spatial_addition_trajectory(args, n_ref, band_edges=(0, 1, 3, 5)):
    print(f"\n=== Section 6: spatial addition-field partition, trajectory-level (N_ref={n_ref}) ===")
    p0 = build_params(build_config(0.0, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)
    rows_c0, _, _ = run_single_trajectory(p0, f0, e1_0, e2_0, e3_0, n_ref)

    labels = [f"{band_edges[i]:g}W_{band_edges[i+1]:g}W" for i in range(len(band_edges) - 1)] + [f">{band_edges[-1]:g}W"]
    out = {}
    for label in labels:
        p1 = build_params(build_config(args.coarsening_rate_scale, args))
        fn = _region_only_ostwald_fn(label, band_edges)
        rows_c1, stop, reason = run_single_trajectory(p1, f0, e1_0, e2_0, e3_0, n_ref, ostwald_fn=fn)
        d = paired_delta(rows_c0[-1], rows_c1[-1])
        v20 = rows_c0[0]["V2"]
        v2_c1 = rows_c1[-1]["V2"]
        print(f"  region {label:10s}: dV2/V20={((v2_c1-v20)/v20):+.4e} "
              f"delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm "
              f"delta_L_GB_geom_sub={d['L_GB_geom_sub']*1e9:+.6f}nm stop={stop}")
        out[label] = dict(dV2_over_V20=(v2_c1 - v20) / v20, delta_final=d, stop=stop)
    return out


# ---------------------------------------------------------------------------
# Section 7: one-step differential test
# ---------------------------------------------------------------------------

def one_step_differential_test(args, band_edges=(0, 1, 3, 5)):
    print("\n=== Section 7: one-step differential TJ response ===")
    p = build_params(build_config(args.coarsening_rate_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p)
    sub_before = compute_subgrid_contact(f0, e1_0, e2_0, p)
    assert sub_before.resolved
    tj_top = (sub_before.top.x_sub, sub_before.top.y_sub)
    tj_bot = (sub_before.bottom.x_sub, sub_before.bottom.y_sub)
    L_before = sub_before.L_contact_TJ_sub
    print(f"  before: L_contact_TJ_sub={L_before*1e9:.6f}nm  tj_top={tj_top}  tj_bottom={tj_bot}")

    out = {"L_before_nm": L_before * 1e9}

    variants = {
        "O0_normal": lambda: ostwald_substrate_diagnostic(f0, e1_0, e2_0, e3_0, p)[:4],
        "O1_removal_only": lambda: ostwald_removal_only(f0, e1_0, e2_0, e3_0, p)[:4],
        "O2_addition_only": lambda: ostwald_addition_only(f0, e1_0, e2_0, e3_0, p)[:4],
    }
    masks = region_masks_by_tj_distance(p, tj_top, tj_bot, band_edges)
    labels = [f"{band_edges[i]:g}W_{band_edges[i+1]:g}W" for i in range(len(band_edges) - 1)] + [f">{band_edges[-1]:g}W"]
    for label in labels:
        variants[f"region_{label}"] = (lambda lbl=label: ostwald_partial_addition_step(f0, e1_0, e2_0, e3_0, p, masks[lbl])[:4])

    for name, fn in variants.items():
        f1, e1_1, e2_1, e3_1 = fn()
        sub_after = compute_subgrid_contact(f1, e1_1, e2_1, p)
        if not sub_after.resolved:
            print(f"  {name}: TJ unresolved after this one-step perturbation")
            out[name] = dict(resolved=False)
            continue
        dL = sub_after.L_contact_TJ_sub - L_before
        d_top = math.hypot(sub_after.top.x_sub - tj_top[0], sub_after.top.y_sub - tj_top[1])
        d_bot = math.hypot(sub_after.bottom.x_sub - tj_bot[0], sub_after.bottom.y_sub - tj_bot[1])
        # signed displacement along the pre-step GB tangent, for direction not just magnitude
        top_disp = (sub_after.top.x_sub - tj_top[0], sub_after.top.y_sub - tj_top[1])
        bot_disp = (sub_after.bottom.x_sub - tj_bot[0], sub_after.bottom.y_sub - tj_bot[1])
        print(f"  {name:22s}: dL_contact_TJ_sub={dL*1e9:+.6e}nm  "
              f"|d_top|={d_top*1e9:.4e}nm  |d_bottom|={d_bot*1e9:.4e}nm")
        out[name] = dict(resolved=True, dL_contact_TJ_sub=dL, top_disp=top_disp, bottom_disp=bot_disp)
    return out


# ---------------------------------------------------------------------------
# Section 9/10: external-reservoir bookkeeping closures (Models A/B/C)
# ---------------------------------------------------------------------------

def _step_with_closure_and_reservoir(f, e1, e2, e3, s, p, phi_local, region_mask=None):
    """One full production step, but Ostwald is replaced with
    ostwald_external_reservoir_step(phi_local, region_mask); returns the new
    state AND the reservoir mass added this step."""
    ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f1 = evolve_f(f, e1, e2, e3, s, Sink(), p)
    e1a, e2a, e3a = project_eta_mass_preserving(f1, e1, e2, e3, p, target_masses=ch_targets)

    f2, e1b, e2b, e3b, reservoir_delta = ostwald_external_reservoir_step(
        f1, e1a, e2a, e3a, p, phi_local=phi_local, region_mask=region_mask,
    )

    ac_targets = eta_masses(e1b, e2b, e3b, p.use_eta3)
    e1c, e2c, e3c = evolve_eta(e1b, e2b, e3b, p)
    e1d, e2d, e3d = project_eta_mass_preserving(f2, e1c, e2c, e3c, p, target_masses=ac_targets)
    return f2, e1d, e2d, e3d, reservoir_delta


def run_closure_trajectory(p, f0, e1_0, e2_0, e3_0, phi_local, n_steps, region_mask=None):
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    wall_x0_val = __import__("pf_sintering.diagnostics", fromlist=["wall_x0"]).wall_x0(p)
    v20 = float(e2.sum() * p.dx * p.dx)

    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    rows = [full_state_sample(f, e1, e2, e3, s, st, p, 0, 0.0, v20, wall_x0_val)]
    reservoir = 0.0
    reservoir_series = [0.0]
    for step in range(1, n_steps + 1):
        f, e1, e2, e3, rd = _step_with_closure_and_reservoir(f, e1, e2, e3, s, p, phi_local, region_mask)
        reservoir += rd
        st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
        rows.append(full_state_sample(f, e1, e2, e3, s, st, p, step, step * p.dt, v20, wall_x0_val))
        reservoir_series.append(reservoir)
        if stop:
            return rows, reservoir_series, True, reason
    return rows, reservoir_series, False, ""


def external_reservoir_conservation_test(args, n_steps=50):
    print(f"\n=== Section 9: external-reservoir mass-conservation test (Model B, n_steps={n_steps}) ===")
    p1 = build_params(build_config(args.coarsening_rate_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p1)

    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    local_f = [float(f.sum()) * p1.dx * p1.dx]
    reservoir = 0.0
    reservoir_track = [0.0]
    for step in range(1, n_steps + 1):
        f, e1, e2, e3, rd = _step_with_closure_and_reservoir(f, e1, e2, e3, s, p1, phi_local=0.0)
        reservoir += rd
        local_f.append(float(f.sum()) * p1.dx * p1.dx)
        reservoir_track.append(reservoir)

    combined = [local_f[i] + reservoir_track[i] for i in range(len(local_f))]
    residual = [c - combined[0] for c in combined]
    print(f"  local_f: {local_f[0]:.8e} -> {local_f[-1]:.8e}  (change {local_f[-1]-local_f[0]:+.6e})")
    print(f"  reservoir: 0 -> {reservoir_track[-1]:.6e}")
    print(f"  combined residual (should be ~0): max|residual|={max(abs(r) for r in residual):.3e}  "
          f"relative to local_f0={max(abs(r) for r in residual)/local_f[0]:.3e}")
    return dict(local_f_initial=local_f[0], local_f_final=local_f[-1], reservoir_final=reservoir_track[-1],
                max_abs_residual=max(abs(r) for r in residual), n_steps=n_steps)


# ---------------------------------------------------------------------------
# Section 12: paired trajectory comparison of closures A/B/C
# ---------------------------------------------------------------------------

def paired_closure_comparison(args, target_dv2_frac):
    print(f"\n=== Section 12: paired closure comparison, target |dV2|/V20={target_dv2_frac:.1e} ===")
    p0 = build_params(build_config(0.0, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)

    out = {}
    for label, phi in (("ModelA_phi1.0", 1.0), ("ModelB_phi0.0", 0.0),
                        ("ModelC_phi0.1", 0.1), ("ModelC_phi0.5", 0.5)):
        p1 = build_params(build_config(args.coarsening_rate_scale, args))
        # step until C1 reaches target, tracking reservoir; C0 is the standard (phi=1, but rate=0 so Ostwald
        # is a no-op regardless of phi) run to the SAME step count for matched time.
        f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
        s = Sink(threshold=math.inf)
        v20 = float(e2.sum() * p1.dx * p1.dx)
        reservoir = 0.0
        step = 0
        while step < args.max_steps:
            step += 1
            f, e1, e2, e3, rd = _step_with_closure_and_reservoir(f, e1, e2, e3, s, p1, phi_local=phi)
            reservoir += rd
            v2 = float(e2.sum() * p1.dx * p1.dx)
            if abs(v2 - v20) / v20 >= target_dv2_frac:
                break
        n_ref = step

        rows_c0, stop0, _ = run_single_trajectory(p0, f0, e1_0, e2_0, e3_0, n_ref)
        d = paired_delta(rows_c0[-1], full_state_sample(
            f, e1, e2, e3, s, compute_stress(f, e1, e2, e3, s, p1)[0], p1, n_ref, n_ref * p1.dt, v20,
            __import__("pf_sintering.diagnostics", fromlist=["wall_x0"]).wall_x0(p1),
        ))
        local_f_final = float(f.sum()) * p1.dx * p1.dx
        local_f_initial = float(f0.sum()) * p1.dx * p1.dx
        print(f"  {label}: n_steps={n_ref} dV2/V20={(v2-v20)/v20:+.4e} "
              f"delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm "
              f"delta_L_GB_geom_sub={d['L_GB_geom_sub']*1e9:+.6f}nm "
              f"delta_sigma={d['sigma_Pa']/1e6:+.6f}MPa reservoir={reservoir:.4e} "
              f"local_f_change={(local_f_final-local_f_initial):.4e}")
        out[label] = dict(
            n_steps=n_ref, dV2_over_V20=(v2 - v20) / v20, delta=d, reservoir_final=reservoir,
            local_f_initial=local_f_initial, local_f_final=local_f_final,
            combined_residual=(local_f_final - local_f_initial) + reservoir,
        )
    return out


# ---------------------------------------------------------------------------
# Section 13: corrected fixed-physics grid check
# ---------------------------------------------------------------------------

def corrected_fixed_physics_grid_check(args):
    print("\n=== Section 13: corrected fixed-physics grid check ===")
    out = {}
    for dx_nm in (5.0, 2.5):
        p_natural = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm,
                                               w_nm=args.fixed_w_nm, fixed_eta=True))
        dt_natural = p_natural.dt
        f0, e1_0, e2_0, e3_0 = initialize_fields(p_natural)
        chosen_frac = args.dx5_dt_frac if dx_nm == 5.0 else args.dx25_dt_frac
        dt_chosen = dt_natural * chosen_frac
        p0 = build_params(build_config(0.0, args, dx_nm=dx_nm, w_nm=args.fixed_w_nm,
                                        fixed_eta=True, dt_override=dt_chosen))
        p1 = build_params(build_config(args.coarsening_rate_scale, args, dx_nm=dx_nm, w_nm=args.fixed_w_nm,
                                        fixed_eta=True, dt_override=dt_chosen))
        rows_c0, rows_c1, reason = run_paired_trajectory(
            p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps,
        )
        v20 = rows_c0[0]["V2"]
        d = paired_delta_series(rows_c0, rows_c1)[-1]
        dv2 = (rows_c1[-1]["V2"] - v20) / v20
        print(f"  dx={dx_nm}nm dt={dt_chosen:.4e}s ({chosen_frac}x natural): n_steps={len(rows_c1)-1} "
              f"dV2/V20={dv2:+.4e} delta_L_coarsening={d['L_contact_TJ_sub']*1e9:+.6f}nm")
        out[f"{dx_nm:g}"] = dict(dt_chosen=dt_chosen, n_steps=len(rows_c1) - 1, dV2_over_V20=dv2, delta_final=d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--coarsening-rate-scale", type=float, default=3.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--eta-mobility-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-dv2-frac", type=float, default=3e-4)
    ap.add_argument("--extended-dv2-frac", type=float, default=1.5e-3)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--fixed-w-nm", type=float, default=20.0)
    ap.add_argument("--dx5-dt-frac", type=float, default=1.0)
    ap.add_argument("--dx25-dt-frac", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="runs/ostwald_reservoir_closure_audit.json")
    args = ap.parse_args()

    out = dict(args=vars(args))
    out["mass_closure"] = mass_closure_audit(args)
    out["one_step"] = one_step_differential_test(args)

    # N_ref from a quick primary-case pass (matches earlier milestones' convention)
    p0 = build_params(build_config(0.0, args))
    p1 = build_params(build_config(args.coarsening_rate_scale, args))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p0)
    rows_c0, rows_c1, _ = run_paired_trajectory(p0, p1, f0, e1_0, e2_0, e3_0,
                                                 target_dv2_frac=args.target_dv2_frac, max_steps=args.max_steps)
    n_ref = len(rows_c1) - 1
    out["n_ref"] = n_ref

    out["spatial_addition_trajectory"] = spatial_addition_trajectory(args, n_ref)
    out["external_reservoir_conservation"] = external_reservoir_conservation_test(args, n_steps=min(50, n_ref))
    out["paired_closures_3e4"] = paired_closure_comparison(args, args.target_dv2_frac)
    out["paired_closures_1_5e3"] = paired_closure_comparison(args, args.extended_dv2_frac)
    out["corrected_fixed_physics_grid"] = corrected_fixed_physics_grid_check(args)

    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
