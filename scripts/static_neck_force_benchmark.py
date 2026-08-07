#!/usr/bin/env python3
"""Sections 8-10 of the force-balance handoff: static neck-force map, the
zero-force neck width x0(V2, L), a short restoring-direction test around x0,
and how x0 shifts as V2 shrinks at fixed L.

No stochastic hazard, RBM, or Ostwald exchange is used anywhere in this
script. The static force-map / x0 search uses `static_neck_geometry` states
directly (no time evolution at all). The restoring-direction test evolves a
handful of steps of ordinary capillary PF relaxation only (CH + mass-
preserving eta projection; no Ostwald, no hazard, no RBM), reusing the same
per-operator sequence qualified in pf_sintering/runner.py so mass conservation
holds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pf_sintering.diagnostics import contact_area, gb_energy, surface_energy, wall_x0
from pf_sintering.model import (
    ModelConfig,
    Sink,
    build_params,
    center,
    contact_width,
    evolve_eta,
    evolve_f,
)
from pf_sintering.static_neck_geometry import build_neck_state
from pf_sintering.structural_projection import eta_masses, project_eta_mass_preserving
from pf_sintering.tj_force import compute_neck_tj_forces


def force_map(p, s, v2_target, L_target, x_neck_scan_nm):
    rows = []
    for xn in x_neck_scan_nm:
        try:
            st = build_neck_state(p, xn * 1e-9, v2_target, L_target)
        except RuntimeError as e:
            rows.append(dict(x_neck_target_nm=xn, feasible=False, reason=str(e)))
            continue
        rep = compute_neck_tj_forces(st.f, st.e1, st.e2, st.e3, s, p)
        rows.append(dict(
            x_neck_target_nm=xn,
            feasible=True,
            x_neck_achieved_nm=st.x_neck_achieved * 1e9,
            V2_relerr=(st.V2_achieved - v2_target) / v2_target,
            L_relerr=(st.L_achieved - L_target) / L_target,
            resolved=rep.resolved,
            n_resolved=rep.n_resolved,
            F_drive=rep.F_drive if rep.resolved else None,
            psi_top_deg=rep.top.psi_deg if rep.top and rep.top.resolved else None,
            psi_bottom_deg=rep.bottom.psi_deg if rep.bottom and rep.bottom.resolved else None,
        ))
    return rows


def locate_x0(p, s, v2_target, L_target, x_scan_nm, tol_nm=0.05, max_bisect=40):
    resolved = [
        r for r in force_map(p, s, v2_target, L_target, x_scan_nm)
        if r.get("resolved") and r.get("F_drive") is not None
    ]
    bracket = None
    for a, b in zip(resolved, resolved[1:]):
        if a["F_drive"] == 0:
            return a["x_neck_achieved_nm"], resolved
        if a["F_drive"] * b["F_drive"] < 0:
            bracket = (a["x_neck_target_nm"], b["x_neck_target_nm"], a["F_drive"])
            break
    if bracket is None:
        return None, resolved

    lo, hi, F_lo_sign = bracket
    F_lo_sign = math.copysign(1.0, F_lo_sign)
    for _ in range(max_bisect):
        if hi - lo < tol_nm:
            break
        mid = 0.5 * (lo + hi)
        st = build_neck_state(p, mid * 1e-9, v2_target, L_target)
        rep = compute_neck_tj_forces(st.f, st.e1, st.e2, st.e3, s, p)
        if not rep.resolved:
            break
        if math.copysign(1.0, rep.F_drive) == F_lo_sign:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), resolved


def capillary_relax_steps(f, e1, e2, e3, p, n_steps):
    """Ordinary capillary PF relaxation only: CH + mass-preserving eta
    projection, structural relaxation + mass-preserving eta projection.
    No Ostwald, no hazard, no RBM (matches Section 9)."""
    s = Sink()  # never touched: no hazard call
    for _ in range(n_steps):
        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, Sink(), p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)

        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)
    return f, e1, e2, e3


def sample_state(f, e1, e2, e3, p, s):
    x_neck, _ = contact_width(e1, e2, p)
    a_gb = contact_area(e1, e2, p)
    e_surf = surface_energy(f, p)
    e_gb = gb_energy(e1, e2, s, p)
    v2 = float(e2.sum() * p.dx * p.dx)
    L = float(center(e2, p) - wall_x0(p))
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    return dict(
        x_neck_nm=x_neck * 1e9 if math.isfinite(x_neck) else None,
        A_GB_nm2=a_gb * 1e18, E_surf_J=e_surf, E_gb_J=e_gb,
        G_interface_J=e_surf + e_gb, V2=v2, L_nm=L * 1e9,
        F_drive=rep.F_drive if rep.resolved else None,
        resolved=rep.resolved,
    )


def restoring_direction_test(p, s, v2_target, L_target, x0_nm, fracs, n_steps):
    out = []
    for frac in fracs:
        st = build_neck_state(p, frac * x0_nm * 1e-9, v2_target, L_target)
        before = sample_state(st.f, st.e1, st.e2, st.e3, p, s)
        f1, e1_1, e2_1, e3_1 = capillary_relax_steps(st.f, st.e1, st.e2, st.e3, p, n_steps)
        after = sample_state(f1, e1_1, e2_1, e3_1, p, s)
        dt = n_steps * p.dt
        row = dict(
            frac_of_x0=frac, x_neck_start_target_nm=frac * x0_nm,
            before=before, after=after, dt_s=dt,
        )
        if before["x_neck_nm"] is not None and after["x_neck_nm"] is not None:
            row["dx_neck_dt_nm_per_s"] = (after["x_neck_nm"] - before["x_neck_nm"]) / dt
        if before["A_GB_nm2"] is not None and after["A_GB_nm2"] is not None:
            row["dA_GB_dt_nm2_per_s"] = (after["A_GB_nm2"] - before["A_GB_nm2"]) / dt
        if before["G_interface_J"] is not None and after["G_interface_J"] is not None:
            row["dG_interface_dt_J_per_s"] = (after["G_interface_J"] - before["G_interface_J"]) / dt
        if before["F_drive"] is not None and after["F_drive"] is not None:
            row["dF_drive_dt_per_s"] = (after["F_drive"] - before["F_drive"]) / dt
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--L-nm", type=float, default=100.0)
    ap.add_argument("--relax-steps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("runs/static_neck_force_benchmark.json"))
    args = ap.parse_args()

    p = build_params(ModelConfig(
        preset="dev", geometry="substrate", nx=args.nx, ny=args.ny, dx=args.dx_nm * 1e-9,
        r2=args.r2_nm * 1e-9, aspect_ratio=args.aspect_ratio, contact_orientation="short_plane",
        t_total=1e-6,
    ))
    s = Sink(threshold=1.0)
    v20 = math.pi * (args.r2_nm * 1e-9) ** 2
    L_target = args.L_nm * 1e-9

    print(f"base geometry: R2={args.r2_nm}nm dx={args.dx_nm}nm V20(=pi R2^2)={v20:.4e} L_target={args.L_nm}nm")

    coarse_scan_nm = [15, 18, 20, 22, 24, 26, 28, 30, 32, 34]
    x0_nm, rows = locate_x0(p, s, v20, L_target, coarse_scan_nm)
    print("\n--- Section 8: static force map (V2=V20, L=L_target) ---")
    for r in rows:
        print(f"  x_neck_target={r['x_neck_target_nm']:5.1f}nm ach={r['x_neck_achieved_nm']:6.2f}nm "
              f"F_drive={r['F_drive']:+.4f} psi_top={r['psi_top_deg']:.2f}deg" if r['F_drive'] is not None else r)
    print(f"\nx0 (V2=V20, L={args.L_nm}nm) = {x0_nm} nm" if x0_nm else "\nNO ZERO-FORCE CROSSING FOUND in scan range")

    result = dict(args=vars(args) | dict(out=str(args.out)), v20=v20, L_target_nm=args.L_nm,
                  force_map=rows, x0_nm=x0_nm)

    if x0_nm is not None:
        print("\n--- Section 9: restoring-direction test around x0 ---")
        fracs = [0.90, 0.95, 1.00, 1.05, 1.10]
        rd = restoring_direction_test(p, s, v20, L_target, x0_nm, fracs, args.relax_steps)
        for row in rd:
            print(f"  frac={row['frac_of_x0']:.2f} x_neck0={row['x_neck_start_target_nm']:.2f}nm "
                  f"dx_neck/dt={row.get('dx_neck_dt_nm_per_s')} nm/s  "
                  f"dG_interface/dt={row.get('dG_interface_dt_J_per_s')}")
        result["restoring_direction_test"] = rd

        print("\n--- Section 10: x0(V2, L) at fixed L ---")
        v2_fracs = [1.000, 0.999, 0.998, 0.995]
        x0_vs_v2 = []
        for vf in v2_fracs:
            v2t = vf * v20
            x0n, rows_v = locate_x0(p, s, v2t, L_target, coarse_scan_nm)
            print(f"  V2/V20={vf:.3f}  x0={x0n} nm")
            x0_vs_v2.append(dict(V2_over_V20=vf, x0_nm=x0n, force_map=rows_v))
        result["x0_vs_V2"] = x0_vs_v2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
