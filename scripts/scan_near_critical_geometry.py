#!/usr/bin/env python3
"""Cheap t=0 geometry/stress scan for selecting a near-critical sink-off
initial state (CODEX_PHYSICS_SEQUENCE.md Milestone 2 / handoff Section 8).

No time evolution is performed. For each candidate geometry this only builds
the initial fields and evaluates compute_stress once, so the whole scan is a
few seconds regardless of grid size. Candidates are ranked by how "resolved
but narrow" the neck is (x_neck / GS1) subject to: not disconnected, not
already burrowed, and a finite, non-degenerate stress decomposition
(dihedral angle measured directly rather than falling back to psi_eq).
"""

from __future__ import annotations

import argparse
import json
import math

from pf_sintering.diagnostics import contact_area
from pf_sintering.model import ModelConfig, Sink, build_params, compute_stress, initialize_fields


def scan(overlaps_nm, aspect_ratios, r2_nm, dx_nm, nx, ny, contact_orientation):
    rows = []
    for ar in aspect_ratios:
        for ov in overlaps_nm:
            cfg = ModelConfig(
                preset="dev", geometry="substrate",
                nx=nx, ny=ny, dx=dx_nm * 1e-9, r2=r2_nm * 1e-9,
                aspect_ratio=ar, contact_orientation=contact_orientation,
                initial_overlap=ov * 1e-9, t_total=1e-6,
            )
            p = build_params(cfg)
            f, e1, e2, e3 = initialize_fields(p)
            s = Sink(threshold=1.0)
            st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
            row = dict(
                aspect_ratio=ar, overlap_nm=ov,
                stop=stop, reason=reason,
                x_neck_nm=st.x_neck * 1e9 if math.isfinite(st.x_neck) else math.nan,
                x_neck_over_GS=st.x_neck / p.GS1 if math.isfinite(st.x_neck) and p.GS1 else math.nan,
                sigma_MPa=st.sigma / 1e6,
                sigma_lt_MPa=st.sigma_lt / 1e6 if math.isfinite(st.sigma_lt) else math.nan,
                sigma_curv_MPa=st.sigma_curv / 1e6 if math.isfinite(st.sigma_curv) else math.nan,
                psi_deg=math.degrees(st.psi) if math.isfinite(st.psi) else math.nan,
                psi_eq_deg=math.degrees(st.psi_eq) if math.isfinite(st.psi_eq) else math.nan,
                psi_measured=math.isfinite(st.psi) and not math.isclose(st.psi, st.psi_eq, rel_tol=1e-9),
                A_GB_nm2=contact_area(e1, e2, p) * 1e18,
                Nx=p.Nx, Ny=p.Ny, dx_nm=p.dx * 1e9,
            )
            rows.append(row)
    return rows


def rank(rows):
    ok = [
        r for r in rows
        if not r["stop"]
        and math.isfinite(r["x_neck_over_GS"])
        and r["sigma_MPa"] > 0
        and r["psi_measured"]
    ]
    # Prefer the narrowest resolved, non-degenerate neck: it is closest to the
    # anticipated critical (contact-starved) state while still being a clean
    # two-triple-junction geometry.
    ok.sort(key=lambda r: r["x_neck_over_GS"])
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlaps-nm", type=float, nargs="+", default=[3, 5, 8, 12, 16, 20])
    ap.add_argument("--aspect-ratios", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--dx-nm", type=float, default=5.0)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--contact-orientation", choices=["short_plane", "long_plane"], default="short_plane")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rows = scan(args.overlaps_nm, args.aspect_ratios, args.r2_nm, args.dx_nm, args.nx, args.ny, args.contact_orientation)
    ranked = rank(rows)

    print(f"{'ar':>5} {'ov_nm':>6} {'stop':>5} {'x_neck_nm':>10} {'x_neck/GS':>10} "
          f"{'sigma_MPa':>10} {'psi_deg':>8} {'psi_eq_deg':>10} {'measured':>9} {'reason':>28}")
    for r in rows:
        print(f"{r['aspect_ratio']:5.2f} {r['overlap_nm']:6.1f} {str(r['stop']):>5} "
              f"{r['x_neck_nm']:10.3f} {r['x_neck_over_GS']:10.4f} {r['sigma_MPa']:10.3f} "
              f"{r['psi_deg']:8.2f} {r['psi_eq_deg']:10.2f} {str(r['psi_measured']):>9} {r['reason']:>28}")

    print("\nTop candidates (narrowest resolved, non-degenerate neck):")
    for r in ranked[:5]:
        print(json.dumps(r, indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(all=rows, ranked=ranked), fh, indent=2)


if __name__ == "__main__":
    main()
