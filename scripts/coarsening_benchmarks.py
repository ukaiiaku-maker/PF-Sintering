"""Milestone 12 -- Gate C/D benchmarks: curvature-driven transfer without
spatial rules, and structural (grain-ownership) coarsening via the
constrained variational eta kinetics.

19. Curved bump on a broad substrate: single connected f field, isotropic
    mu, pure surface_transport (no eta) -- the high-curvature bump must
    lose mass, the low-curvature surroundings must gain it, with total f
    conserved and NO donor/receiver/removal/deposition region ever named.
20. Two-region connected coarsening: same geometry, now with two
    structural grains (e1=broad base, e2=bump) evolved by
    constrained_eta.constrained_variational_eta_update alongside f -- V2
    must decrease, V1 increase, total f unaffected, through the SAME
    curvature-driven flux, no Ostwald kernel anywhere.

Does not modify production physics.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.constrained_eta import constrained_variational_eta_update, f_weighted_ownership_volumes
from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update


def build_bump_geometry(dx_nm, W_nm, bump_height_nm, bump_sigma_nm, Nx=120, Ny=100):
    p = build_params(ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, use_aniso_surface=False, surface_mobility_scale=0.3,
    ))
    dx = p.dx; W = p.interface_width
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    baseline = x[Nx // 4]
    bump_height = bump_height_nm * 1e-9
    bump_sigma = bump_sigma_nm * 1e-9
    y0 = y.mean()
    h = baseline + bump_height * np.exp(-((Y - y0) ** 2) / (2 * bump_sigma ** 2))
    f = 0.5 * (1 + np.tanh((X - h) / W))
    return p, f, X, Y, y0, bump_sigma


def region_masks(Y, y0, bump_sigma, n_sigma=2.0):
    bump = np.abs(Y - y0) < n_sigma * bump_sigma
    return bump, ~bump


def bump_benchmark(args):
    print("\n=== Benchmark 19: curved bump on substrate (Gate C) ===")
    p, f, X, Y, y0, bump_sigma = build_bump_geometry(args.dx_nm, args.w_nm, args.bump_height_nm,
                                                       args.bump_sigma_nm, args.nx, args.ny)
    e1 = np.zeros_like(f); e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    bump_mask, far_mask = region_masks(Y, y0, bump_sigma)

    def region_area(field, mask):
        return float((field * mask).sum()) * p.dx * p.dx

    total0 = float(f.sum())
    bump0 = region_area(f, bump_mask)
    far0 = region_area(f, far_mask)
    print(f"  t=0: total_f={total0*p.dx*p.dx:.6e}  bump_region={bump0:.6e}  far_region={far0:.6e}")

    for step in range(1, args.bump_steps + 1):
        mu = mu_isotropic(f, e1, e1, e3, s, p)
        f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s,
                                             bc_x="reflecting", bc_y="reflecting")

    total1 = float(f.sum())
    bump1 = region_area(f, bump_mask)
    far1 = region_area(f, far_mask)
    mass_drift = abs(total1 - total0) / total0
    print(f"  t_final ({args.bump_steps} steps): total_f_drift={mass_drift:.3e}  "
          f"bump_region={bump1:.6e} (d={bump1-bump0:+.4e})  far_region={far1:.6e} (d={far1-far0:+.4e})")
    print(f"  bump lost mass: {bump1 < bump0}   far region gained mass: {far1 > far0}")
    return dict(mass_drift=mass_drift, bump0=bump0, bump1=bump1, far0=far0, far1=far1,
                bump_lost=bool(bump1 < bump0), far_gained=bool(far1 > far0))


def two_region_benchmark(args):
    print("\n=== Benchmark 20: two-region connected coarsening (Gate D) ===")
    # Reuse the EXISTING, already-validated flat-substrate initializer
    # (genuine GB between a low-curvature broad body (grain 1) and a
    # high-curvature small feature (grain 2), attached via a real
    # solid-solid interface) rather than a synthetic smooth-blend "bump" --
    # a smooth exp-weighted ownership split (tried first) has no genuine
    # GB for the structural free energy to act on and does not exercise
    # the intended two-grain coarsening mechanism (see MILESTONE_12
    # report for this finding). R2 must be properly resolved relative to W
    # (R2=80nm, W=20nm, matching every prior milestone's primary case) --
    # an under-resolved R2~30nm~1.5*W particle showed V2 decrease reversing
    # into an increase after ~1600 steps (a diffuse-interface artifact, not
    # a genuine physical crossover: confirmed absent at proper resolution
    # over 10x more steps -- see MILESTONE_12 report).
    from pf_sintering.model import ModelConfig, build_params, initialize_fields, reproject
    p = build_params(ModelConfig(
        preset="dev", geometry="substrate", dx=args.dx_nm * 1e-9, r2=args.gate_d_r2_nm * 1e-9,
        aspect_ratio=2.0, contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=args.w_nm * 1e-9, use_aniso_surface=False, surface_mobility_scale=0.3,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)  # pre-clean (see Benchmark 19/20's earlier finding)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)

    V1_0, V2_0, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    total_f_0 = float(f.sum()) * p.dx * p.dx
    print(f"  t=0: V1={V1_0:.6e} V2={V2_0:.6e} total_f={total_f_0:.6e}")

    sample_every = max(1, args.two_region_steps // 40)
    t_series, V1_series, V2_series = [0.0], [V1_0], [V2_0]
    var_change_total = 0.0
    proj_change_total = 0.0
    for step in range(1, args.two_region_steps + 1):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s,
                                             bc_x="reflecting", bc_y="periodic")
        e1, e2, e3, ediag = constrained_variational_eta_update(e1, e2, e3, f, s, p, bc_x="reflecting", bc_y="periodic")
        var_change_total += ediag["variational_change"]
        proj_change_total += ediag["projection_change"]
        if step % sample_every == 0:
            V1_i, V2_i, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
            t_series.append(step * p.dt)
            V1_series.append(V1_i)
            V2_series.append(V2_i)
            print(f"    step={step:6d} t={step*p.dt:.4e}s V1={V1_i:.8e} V2={V2_i:.8e}")

    V1_1, V2_1, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    total_f_1 = float(f.sum()) * p.dx * p.dx
    mass_drift = abs(total_f_1 - total_f_0) / total_f_0
    V2_arr = np.array(V2_series)
    V2_min_idx = int(np.argmin(V2_arr))
    print(f"  t_final ({args.two_region_steps} steps): V1={V1_1:.6e} (d={V1_1-V1_0:+.4e})  "
          f"V2={V2_1:.6e} (d={V2_1-V2_0:+.4e})  total_f={total_f_1:.6e}  mass_drift={mass_drift:.3e}")
    print(f"  cumulative |variational| change={var_change_total:.4e}  "
          f"cumulative |projection| change={proj_change_total:.4e}  "
          f"projection/variational={proj_change_total/(var_change_total+1e-300):.4e}")
    print(f"  V2 min={V2_arr.min():.8e} at t_series index {V2_min_idx} (t={t_series[V2_min_idx]:.4e}s), "
          f"i.e. V2 {'monotonically decreased then reversed' if 0 < V2_min_idx < len(V2_arr)-1 else ('monotonically decreased' if V2_min_idx==len(V2_arr)-1 else 'never net-decreased')}")
    print(f"  V2 decreased (endpoint): {V2_1 < V2_0}   V1 increased (endpoint): {V1_1 > V1_0}")
    return dict(V1_0=V1_0, V2_0=V2_0, V1_1=V1_1, V2_1=V2_1, total_f_0=total_f_0, total_f_1=total_f_1,
                mass_drift=mass_drift, var_change_total=var_change_total, proj_change_total=proj_change_total,
                V2_decreased=bool(V2_1 < V2_0), V1_increased=bool(V1_1 > V1_0),
                t_series=t_series, V1_series=V1_series, V2_series=V2_series,
                V2_min=float(V2_arr.min()), V2_min_t=t_series[V2_min_idx])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--ny", type=int, default=100)
    ap.add_argument("--bump-height-nm", type=float, default=40.0)
    ap.add_argument("--bump-sigma-nm", type=float, default=60.0)
    ap.add_argument("--gate-d-r2-nm", type=float, default=80.0)
    ap.add_argument("--bump-steps", type=int, default=4000)
    ap.add_argument("--two-region-steps", type=int, default=4000)
    ap.add_argument("--out", type=str, default="runs/coarsening_benchmarks.json")
    args = ap.parse_args()

    out = dict(args=vars(args))
    out["bump"] = bump_benchmark(args)
    out["two_region"] = two_region_benchmark(args)

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
