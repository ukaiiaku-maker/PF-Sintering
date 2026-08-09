"""Milestone 14C: capillary/sintering-stress analysis of saved checkpoint
states (Milestone 14B's A/B/C/D, or any state saved the same way by
m14b_save_checkpoint.py).

For each state: locate both TJs' free-surface branch directions, classify
particle vs. substrate branches, trace the particle-side free-surface arc
between the two TJs, compute the capillary resultant two independent ways
(curvature-form integral vs. endpoint-tangent-difference) and report their
relative discrepancy, derive the apparent 2-D sintering stress, and report
local capillary-pressure windows (particle side, both TJs) and substrate-
side curvature windows (both TJs).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m14_mechanism_screen import BC_X, BC_Y, build_config  # noqa: E402
from m12b_grid_convergence import classify_branches  # noqa: E402

from pf_sintering.capillary_stress import (
    apparent_sintering_stress,
    capillary_force_curvature_form,
    capillary_force_endpoint_form,
    force_validation_relative_error,
    oriented_contact_normal,
    substrate_curvature_windows,
    trace_particle_arc,
    window_mean_kappa,
    window_mean_kappa_from_end,
)
from pf_sintering.curvature_extraction import window_curvature
from pf_sintering.model import Sink, build_params
from pf_sintering.tj_force import compute_tj_force
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_p(args):
    return build_params(build_config(args))


def analyze_state(npz_path, args, max_arclength=560e-9):
    d = np.load(npz_path)
    f, e1, e2, e3, mu = d["f"], d["e1"], d["e2"], d["e3"], d["mu"]
    tj_top, tj_bottom = d["tj_top"], d["tj_bottom"]
    p = build_p(args)
    s = Sink(threshold=math.inf)
    gamma_s = p.gamma_s
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx

    top = compute_tj_force(f, e1, e2, tj_top, s, p)
    bottom = compute_tj_force(f, e1, e2, tj_bottom, s, p)
    out = dict(label=str(d["label"]), t=float(d["t"]), tau=float(d["tau"]),
               L_contact_saved=float(d["L_contact_TJ_sub"]), M_neck_f=float(d["M_neck_f"]),
               F=float(d["F"]), D_h=float(d["D_h"]),
               psi_top_deg=float(d["psi_top_deg"]), psi_bottom_deg=float(d["psi_bottom_deg"]),
               top_resolved=bool(top.resolved), bottom_resolved=bool(bottom.resolved),
               F_TJ_mag_top=top.F_TJ_mag if top.resolved else math.nan,
               F_TJ_mag_bottom=bottom.F_TJ_mag if bottom.resolved else math.nan)
    if not (top.resolved and bottom.resolved):
        out["resolved"] = False
        return out

    top_particle_dir, top_substrate_dir = classify_branches(f, p, tj_top, top.v_s1, top.v_s2, wall_mean)
    bot_particle_dir, bot_substrate_dir = classify_branches(f, p, tj_bottom, bottom.v_s1, bottom.v_s2, wall_mean)

    arc = trace_particle_arc(f, p, tj_top, tj_bottom, top_particle_dir, bot_particle_dir,
                              max_arclength=max_arclength)
    out["arc"] = arc
    if not arc.get("resolved", False):
        out["resolved"] = False
        return out

    F_curv = capillary_force_curvature_form(arc, gamma_s)
    F_ep = capillary_force_endpoint_form(top_particle_dir, bot_particle_dir, gamma_s)
    rel_err = force_validation_relative_error(F_curv, F_ep)
    out.update(F_cap_curvature_form=F_curv, F_cap_endpoint_form=F_ep, force_rel_err=rel_err,
               arc_length=arc["arc_length"], closest_approach_dist=arc["closest_approach_dist"])

    sub = compute_subgrid_contact(f, e1, e2, p)
    out["subgrid_resolved"] = bool(sub.resolved)
    if sub.resolved:
        n_GB = oriented_contact_normal(sub.n_GB_sub)
        sigma_curv, Fn_curv = apparent_sintering_stress(F_curv, n_GB, sub.L_contact_TJ_sub)
        sigma_ep, Fn_ep = apparent_sintering_stress(F_ep, n_GB, sub.L_contact_TJ_sub)
        out.update(n_GB=n_GB, L_contact=sub.L_contact_TJ_sub,
                   F_cap_n_curvature_form=Fn_curv, F_cap_n_endpoint_form=Fn_ep,
                   sigma_sint_app_curvature_form=sigma_curv, sigma_sint_app_endpoint_form=sigma_ep)

    W = p.interface_width
    windows = ((1.5, 3.0), (3.0, 5.0))
    p_gamma = {}
    for lo, hi in windows:
        key = f"{lo}W-{hi}W"
        kappa_top = window_mean_kappa(arc, lo * W, hi * W)
        kappa_bot = window_mean_kappa_from_end(arc, lo * W, hi * W)
        p_gamma[key] = dict(kappa_top=kappa_top, kappa_bottom=kappa_bot,
                             p_gamma_top=gamma_s * kappa_top, p_gamma_bottom=gamma_s * kappa_bot)
    out["p_gamma_particle_windows"] = p_gamma
    out["kappa_arc_max_abs"] = float(np.max(np.abs(arc["kappa"])))
    out["kappa_arc_max_signed"] = float(arc["kappa"][np.argmax(np.abs(arc["kappa"]))])

    # cross-check: independent Kasa-fit (window_curvature) at the near-TJ
    # window from the particle branch, using the pre-existing, previously
    # qualified windowed-curvature machinery (Section 10's "repeat with
    # slightly shifted windows"/independent-method spirit).
    cross = {}
    for label, tj_xy, branch_dir in (("top", tj_top, top_particle_dir), ("bottom", tj_bottom, bot_particle_dir)):
        wc_near = window_curvature(f, e1, e2, e3, s, p, tj_xy, branch_dir, 1.5 * W, 3.0 * W)
        wc_far = window_curvature(f, e1, e2, e3, s, p, tj_xy, branch_dir, 3.0 * W, 5.0 * W)
        cross[label] = dict(near_kappa_geom=wc_near.kappa_geom, far_kappa_geom=wc_far.kappa_geom)
    out["kasa_crosscheck_particle_windows"] = cross

    # substrate-side curvature, both TJs
    sub_curv = {}
    for label, tj_xy, branch_dir in (("top", tj_top, top_substrate_dir), ("bottom", tj_bottom, bot_substrate_dir)):
        sub_curv[label] = substrate_curvature_windows(f, e1, e2, e3, s, p, tj_xy, branch_dir, windows=windows)
    out["substrate_curvature_windows"] = sub_curv

    out["resolved"] = True
    return out


def _strip_arc(out):
    out = dict(out)
    out.pop("arc", None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=72.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.9)
    ap.add_argument("--gamma-gb-override", type=float, default=1.6)
    ap.add_argument("--dt-override", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--states", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    results = {}
    for state_path in args.states:
        label = os.path.splitext(os.path.basename(state_path))[0]
        print(f"=== analyzing {label} ({state_path}) ===")
        out = analyze_state(state_path, args)
        results[label] = _strip_arc(out)
        if out.get("resolved"):
            print(f"  t={out['t']:.4e}s tau={out['tau']:.4e}s L_contact={out['L_contact']*1e9:.3f}nm "
                  f"F_cap(curv)=({out['F_cap_curvature_form'][0]:.4f},{out['F_cap_curvature_form'][1]:.4f}) "
                  f"F_cap(ep)=({out['F_cap_endpoint_form'][0]:.4f},{out['F_cap_endpoint_form'][1]:.4f}) "
                  f"rel_err={out['force_rel_err']:.4f} "
                  f"sigma_sint_app(curv)={out['sigma_sint_app_curvature_form']:.4e}Pa "
                  f"sigma_sint_app(ep)={out['sigma_sint_app_endpoint_form']:.4e}Pa "
                  f"psi_top={out['psi_top_deg']:.2f} psi_bot={out['psi_bottom_deg']:.2f}")
        else:
            print(f"  NOT RESOLVED: {out}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, default=str, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
