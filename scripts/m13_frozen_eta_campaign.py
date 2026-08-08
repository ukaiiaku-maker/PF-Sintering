"""Milestone 13 Sections 3-7: pure surface-diffusion baseline.

Runs the SAME State-A sinusoidal-substrate contact and unified isotropic
transport as unified_sinusoidal_campaign.py / m12b_grid_convergence.py, but
with eta FROZEN at its initial values -- the eta update is never called at
all. Critically, eta is NOT removed from mu_f: `mu_isotropic(f, e1, e2, e3,
s, p)` is still evaluated every step with the (frozen) e1/e2/e3, so the
eta-f thermodynamic coupling term (Wc*eta2*(1-fb) inside mu0) stays exactly
as production has it. Only the GB/ownership KINETICS (d(eta_i)/dt) are
turned off, isolating the question Milestone 13 Section 3 poses: what does
the conserved surface-diffusion equation itself do to the morphology?

With eta frozen, f_weighted_ownership_volumes' V2(t) change is driven
ENTIRELY by conserved-f transport (the eta-migration contribution is
identically zero by construction) -- Section 5's dM_particle_f/dt IS just
dV2/dt in this baseline, no decomposition needed.

Section 4 morphology tracking (independent of V2/eta):
  - particle far-cap X position: max X of the f=0.5 contour.
  - A_particle_geom: sum(f * particle_side_mask)*dx^2, where
    particle_side_mask = (X - x_s(Y)) > n_W*W for a generous n_W -- a
    PURELY GEOMETRIC particle-size proxy (no eta anywhere in it), distinct
    from V2's ownership-weighted volume. This is the practical proxy used
    in place of a fully general particle-cap-area-bounded-by-free-surface-
    and-GB-chord construction (Section 4's "where possible"); the simpler
    version was chosen given this milestone's overall scope, and is
    documented as such in the Milestone 13 report.
  - kappa_geom (particle-far / substrate-far, both TJs) via
    curvature_extraction.window_curvature, wall_mean bug-fixed (Section 8).
  - TJ coordinates, L_contact_TJ_sub, L_GB_geom_sub.
  - dM_neck_f/dt: direct before/after f-mass change within
    ch_crossover_diagnostics.neck_region_mask.

Section 7 (less frequent, more expensive): full branch mu(s)/J_tangent(s)
profiles and neck_boundary_face_flux_balance's particle-side/substrate-side
split, at both TJs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from unified_sinusoidal_campaign import BC_X, BC_Y, build_config  # noqa: E402

from pf_sintering.ch_crossover_diagnostics import neck_ch_mass_balance, neck_region_mask
from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.constrained_eta import f_weighted_ownership_volumes
from pf_sintering.curvature_extraction import branch_mu_J_profile, window_curvature
from pf_sintering.flux_closure import neck_boundary_face_flux_balance
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update, surface_flux, surface_mobility_tensor
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build_state_a(args, dx_nm):
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def substrate_wall_mean(p):
    # Milestone 13 Section 8 fix -- see m12b_grid_convergence.py's run_campaign.
    return p.substrate_wall_frac * p.Nx * p.dx


def particle_side_mask(p, wall_mean, n_W=3.0):
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    x_s = wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)
    return (X - x_s) > n_W * p.interface_width


def particle_far_cap_x(f, p):
    from skimage.measure import find_contours
    pts = []
    for rc in find_contours(f, 0.5):
        pts.append((rc[:, 1] + 1) * p.dx)
    if not pts:
        return math.nan
    return float(np.max(np.concatenate(pts)))


def measure(f, e1, e2, e3, s, p, wall_mean, pmask, neck_mask):
    V1, V2, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    total_f = float(f.sum()) * p.dx * p.dx
    sub = compute_subgrid_contact(f, e1, e2, p)
    A_particle_geom = float(np.sum(f * pmask)) * p.dx * p.dx
    far_cap_x = particle_far_cap_x(f, p)
    M_neck = float(np.sum(f * neck_mask)) * p.dx * p.dx
    F = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    return dict(
        V1=V1, V2=V2, total_f=total_f, A_particle_geom=A_particle_geom,
        far_cap_x=far_cap_x, M_neck_f=M_neck, F=F,
        L_contact_TJ_sub=sub.L_contact_TJ_sub if sub.resolved else math.nan,
        L_GB_geom_sub=sub.L_GB_geom_sub if sub.resolved else math.nan,
        tj_top=(float(sub.top.x_sub), float(sub.top.y_sub)) if sub.resolved else None,
        tj_bottom=(float(sub.bottom.x_sub), float(sub.bottom.y_sub)) if sub.resolved else None,
        resolved=sub.resolved,
    )


def curvature_snapshot(f, e1, e2, e3, s, p, wall_mean):
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    from m12b_grid_convergence import classify_branches
    from pf_sintering.curvature_extraction import branch_window_report
    out = {}
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            out[label] = None
            continue
        pd, sdir = classify_branches(f, p, tj.tj_xy, tj.v_s1, tj.v_s2, wall_mean)
        wp = branch_window_report(f, e1, e2, e3, s, p, tj.tj_xy, pd)
        ws = branch_window_report(f, e1, e2, e3, s, p, tj.tj_xy, sdir)
        near_p, far_p = wp[0], wp[max(wp)]
        near_s, far_s = ws[0], ws[max(ws)]
        out[label] = dict(
            kappa_geom_particle_near=near_p.kappa_geom, kappa_geom_particle_far=far_p.kappa_geom,
            kappa_geom_substrate_near=near_s.kappa_geom, kappa_geom_substrate_far=far_s.kappa_geom,
        )
    return out


def asymmetric_flux_snapshot(f, e1, e2, e3, s, p, wall_mean, M_s):
    """Section 7: full mu(s)/J_tangent(s) branch profiles + neck boundary
    face-flux particle/substrate split."""
    from m12b_grid_convergence import classify_branches
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, p.dx, BC_X, BC_Y)

    out = dict(branches={}, neck_balance=None)
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            continue
        pd, sdir = classify_branches(f, p, tj.tj_xy, tj.v_s1, tj.v_s2, wall_mean)
        prof_p = branch_mu_J_profile(f, mu, Jx, Jy, p, tj.tj_xy, pd, 150e-9, n_samples=30)
        prof_s = branch_mu_J_profile(f, mu, Jx, Jy, p, tj.tj_xy, sdir, 150e-9, n_samples=30)
        out["branches"][label] = dict(particle=prof_p, substrate=prof_s)

    sub = compute_subgrid_contact(f, e1, e2, p)
    if sub.resolved:
        mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
        out["neck_balance"] = neck_boundary_face_flux_balance(Jx, Jy, mask, p, wall_mean, BC_X, BC_Y)
    return out


def run(dx_nm, args, t_end, sample_every_steps, flux_snapshot_every_samples):
    print(f"\n=== Frozen-eta pure surface-diffusion baseline, dx={dx_nm}nm, t_end={t_end:.4e}s ===")
    p, f, e1, e2, e3 = build_state_a(args, dx_nm)
    wall_mean = substrate_wall_mean(p)
    pmask = particle_side_mask(p, wall_mean)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    n_steps = round(t_end / p.dt)
    print(f"  Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s n_steps={n_steps}")

    sub0 = compute_subgrid_contact(f, e1, e2, p)
    neck_mask = neck_region_mask(p, (sub0.top.x_sub, sub0.top.y_sub), (sub0.bottom.x_sub, sub0.bottom.y_sub)) \
        if sub0.resolved else np.zeros_like(f, dtype=bool)

    rows = []
    m0 = measure(f, e1, e2, e3, s, p, wall_mean, pmask, neck_mask)
    m0["step"] = 0
    m0["time_s"] = 0.0
    m0["curvature"] = curvature_snapshot(f, e1, e2, e3, s, p, wall_mean)
    rows.append(m0)
    print(f"  step=0 V2={m0['V2']:.8e} A_particle_geom={m0['A_particle_geom']:.6e} "
          f"M_neck_f={m0['M_neck_f']:.6e} far_cap_x={m0['far_cap_x']*1e9:.3f}nm")

    sample_idx = 0
    for step in range(1, n_steps + 1):
        mu = mu_isotropic(f, e1, e2, e3, s, p)  # eta-f coupling STAYS; eta itself is frozen (never updated)
        f, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)

        if step % sample_every_steps == 0 or step == n_steps:
            m = measure(f, e1, e2, e3, s, p, wall_mean, pmask, neck_mask)
            m["step"] = step
            m["time_s"] = step * p.dt
            sample_idx += 1
            if sample_idx % flux_snapshot_every_samples == 0 or step == n_steps:
                m["curvature"] = curvature_snapshot(f, e1, e2, e3, s, p, wall_mean)
                m["asymmetric_flux"] = asymmetric_flux_snapshot(f, e1, e2, e3, s, p, wall_mean, M_s)
            else:
                m["curvature"] = None
                m["asymmetric_flux"] = None
            rows.append(m)
            if step % (sample_every_steps * 20) == 0 or step == n_steps:
                nb = m["asymmetric_flux"]["neck_balance"] if m.get("asymmetric_flux") else None
                nb_str = "" if nb is None else (
                    f" neck_particle_side={nb['dM_neck_dt_particle_side']:.3e} "
                    f"neck_substrate_side={nb['dM_neck_dt_substrate_side']:.3e}")
                print(f"  step={step:7d} t={m['time_s']:.4e}s V2={m['V2']:.8e} "
                      f"A_particle_geom={m['A_particle_geom']:.6e} M_neck_f={m['M_neck_f']:.6e} "
                      f"far_cap_x={m['far_cap_x']*1e9:.3f}nm total_f_drift="
                      f"{(m['total_f']-m0['total_f'])/m0['total_f']:.2e}{nb_str}")

    return dict(dx_nm=dx_nm, p_dt=p.dt, p_Nx=p.Nx, p_Ny=p.Ny, rows=rows)


def find_zero_crossing(t, y):
    t = np.asarray(t)
    y = np.asarray(y)
    sign = np.sign(y)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    t0, t1 = t[i], t[i + 1]
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        return float(t0)
    return float(t0 - y0 * (t1 - t0) / (y1 - y0))


def analyze(result):
    rows = result["rows"]
    t = np.array([r["time_s"] for r in rows])
    V2 = np.array([r["V2"] for r in rows])
    M_neck = np.array([r["M_neck_f"] for r in rows])
    far_cap_x = np.array([r["far_cap_x"] for r in rows])
    Lc = np.array([r["L_contact_TJ_sub"] for r in rows])
    dV2dt = np.gradient(V2, t)
    dM_neck_dt = np.gradient(M_neck, t)
    t_neck_star = find_zero_crossing(t, dM_neck_dt)
    return dict(
        t=t.tolist(), V2=V2.tolist(), M_neck_f=M_neck.tolist(), far_cap_x=far_cap_x.tolist(),
        L_contact=Lc.tolist(), dV2dt=dV2dt.tolist(), dM_neck_dt=dM_neck_dt.tolist(),
        t_neck_flux_sign_change=t_neck_star,
        dM_neck_dt_start=float(dM_neck_dt[0]), dM_neck_dt_end=float(dM_neck_dt[-1]),
        far_cap_x_start=float(far_cap_x[0]), far_cap_x_end=float(far_cap_x[-1]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--t-end", type=float, default=0.1)
    ap.add_argument("--sample-every-steps", type=int, default=50)
    ap.add_argument("--flux-snapshot-every-samples", type=int, default=20)
    ap.add_argument("--out", type=str, default="runs/m13_frozen_eta_baseline.json")
    args = ap.parse_args()

    result = run(args.dx_nm, args, args.t_end, args.sample_every_steps, args.flux_snapshot_every_samples)
    analysis = analyze(result)
    out = dict(args=vars(args), result=result, analysis=analysis)

    print(f"\n=== Summary dx={args.dx_nm}nm ===")
    print(f"  far_cap_x: {analysis['far_cap_x_start']*1e9:.3f}nm -> {analysis['far_cap_x_end']*1e9:.3f}nm")
    print(f"  dM_neck_f/dt: start={analysis['dM_neck_dt_start']:.4e} end={analysis['dM_neck_dt_end']:.4e}")
    print(f"  t at dM_neck_f/dt sign change: {analysis['t_neck_flux_sign_change']}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
