"""Milestone 12B -- Sections 8-20: fixed-physics grid+dt convergence study
for the unified isotropic transport law on the primary State-A sinusoidal-
substrate contact.

Physics is IDENTICAL to scripts/unified_sinusoidal_campaign.py (same
build_config/build_state_a, same constrained_variational_eta_update with
the Milestone 12B-corrected g_i, same surface_divergence_update transport,
use_aniso_surface=False throughout) -- this script only adds the grid/dt
convergence machinery and the curvature/mu/flux/energy diagnostics Section
8-20 require, reusing unified_sinusoidal_campaign.build_config/build_state_a
directly rather than re-deriving physics.

Section 8: audit_dx_dependent_quantities() prints every physical quantity
at each dx and confirms which ones are dx-independent by construction
(W, R2, wavelength, amplitude, gamma_s, gamma_gb_ref, k_f, k_eta, M_f,
M_eta, M_s -- all fixed given eta_diffusivity_fixed_physical=True and a
fixed surface_mobility_scale) vs. which necessarily scale with dx (Nx, Ny,
raw cell counts, and production's CFL-based dt formula).

Section 9: dt_convergence_test() runs a short common physical-time interval
at dt, dt/2, dt/4 (dt = production's own p.dt formula at that dx) from the
SAME initial state and compares V2, L_contact, total_f, F at the matched
end time -- does not assume a dx^4 rule, actually measures it.

Sections 10-18: run_campaign() produces dense V1(t)/V2(t)/L_contact(t)
time series plus periodic curvature (Method A+B, curvature_extraction.py),
mu/J_s branch profiles, and energy/dissipation snapshots.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math

import numpy as np

from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.constrained_eta import constrained_variational_eta_update, f_weighted_ownership_volumes
from pf_sintering.curvature_extraction import branch_window_report
from pf_sintering.model import Sink, initialize_fields, reproject
from pf_sintering.surface_transport import dissipation_density, m_s_ref, surface_divergence_update, surface_mobility_tensor
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from unified_sinusoidal_campaign import BC_X, BC_Y, build_config  # noqa: E402


def build_state_a(args, dx_nm):
    from pf_sintering.model import build_params
    p = build_params(build_config(argparse.Namespace(**{**vars(args), "dx_nm": dx_nm})))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def audit_dx_dependent_quantities(dx_list, args):
    print("\n=== Section 8: dx-dependent-quantity audit ===")
    rows = []
    for dx_nm in dx_list:
        p, f, e1, e2, e3 = build_state_a(args, dx_nm)
        M_s = m_s_ref(p.M_f, p.interface_width)
        natural_dt = p.CFL * p.dx ** 4 / (p.M_f * p.k_f)
        row = dict(
            dx_nm=dx_nm, Nx=p.Nx, Ny=p.Ny, dt=p.dt, dt_clamped=bool(p.dt < natural_dt - 1e-30),
            natural_cfl_dt=natural_dt,
            wavelength_nm=p.sinusoid_wavelength * 1e9, amplitude_nm=p.sinusoid_amplitude * 1e9,
            W_nm=p.interface_width * 1e9, R2_nm=p.R2 * 1e9,
            M_f=p.M_f, M_eta=p.M_eta, M_s=M_s, k_f=p.k_f, k_eta=p.k_eta,
            gamma_s=p.gamma_s, gamma_gb_ref=p.gamma_gb_ref,
        )
        rows.append(row)
        print(f"  dx={dx_nm}nm: Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s (natural_CFL_dt={natural_dt:.4e}s, "
              f"clamped={row['dt_clamped']}) wavelength={row['wavelength_nm']:.3f}nm "
              f"amplitude={row['amplitude_nm']:.3f}nm W={row['W_nm']:.3f}nm R2={row['R2_nm']:.3f}nm")
        print(f"           M_f={p.M_f:.6e} M_eta={p.M_eta:.6e} M_s={M_s:.6e} "
              f"k_f={p.k_f:.6e} k_eta={p.k_eta:.6e} gamma_s={p.gamma_s} gamma_gb_ref={p.gamma_gb_ref:.6e}")
    fixed_ok = all(math.isclose(r["wavelength_nm"], rows[0]["wavelength_nm"], rel_tol=1e-9) for r in rows) and \
        all(math.isclose(r["amplitude_nm"], rows[0]["amplitude_nm"], rel_tol=1e-9) for r in rows) and \
        all(math.isclose(r["W_nm"], rows[0]["W_nm"], rel_tol=1e-9) for r in rows) and \
        all(math.isclose(r["R2_nm"], rows[0]["R2_nm"], rel_tol=1e-9) for r in rows) and \
        all(math.isclose(r["M_f"], rows[0]["M_f"], rel_tol=1e-9) for r in rows) and \
        all(math.isclose(r["M_eta"], rows[0]["M_eta"], rel_tol=1e-9) for r in rows) and \
        all(math.isclose(r["M_s"], rows[0]["M_s"], rel_tol=1e-9) for r in rows)
    print(f"  all physical parameters (W, R2, wavelength, amplitude, M_f, M_eta, M_s) fixed across grids: {fixed_ok}")
    return rows, fixed_ok


def one_transport_step(f, e1, e2, e3, s, p, dt, M_s):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
    e1, e2, e3, ediag = constrained_variational_eta_update(e1, e2, e3, f_new, s, p, dt=dt, bc_x=BC_X, bc_y=BC_Y)
    return f_new, e1, e2, e3, diag, ediag


def measure(f, e1, e2, e3, s, p):
    V1, V2, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    total_f = float(f.sum()) * p.dx * p.dx
    sub = compute_subgrid_contact(f, e1, e2, p)
    F = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    return dict(V1=V1, V2=V2, total_f=total_f, F=F,
                L_contact_TJ_sub=sub.L_contact_TJ_sub if sub.resolved else math.nan,
                L_GB_geom_sub=sub.L_GB_geom_sub if sub.resolved else math.nan,
                resolved=sub.resolved)


def dt_convergence_test(dx_nm, args, n_steps_base=200, dt_fracs=(1.0, 0.5, 0.25)):
    print(f"\n=== Section 9: dt convergence at dx={dx_nm}nm ===")
    p0, f0, e10, e20, e30 = build_state_a(args, dx_nm)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p0.M_f, p0.interface_width)
    t_test = n_steps_base * p0.dt
    results = {}
    for frac in dt_fracs:
        p = dataclasses.replace(p0, dt=p0.dt * frac)
        n_steps = round(t_test / p.dt)
        f, e1, e2, e3 = f0.copy(), e10.copy(), e20.copy(), e30.copy()
        for _ in range(n_steps):
            f, e1, e2, e3, diag, ediag = one_transport_step(f, e1, e2, e3, s, p, p.dt, M_s)
        m = measure(f, e1, e2, e3, s, p)
        m["dt"] = p.dt
        m["n_steps"] = n_steps
        m["t_reached"] = n_steps * p.dt
        results[frac] = m
        print(f"  dt_frac={frac}: dt={p.dt:.4e}s n_steps={n_steps} t={m['t_reached']:.6e}s "
              f"V2={m['V2']:.8e} L_contact={m['L_contact_TJ_sub']*1e9:.4f}nm F={m['F']:.6e} total_f={m['total_f']:.6e}")

    base = results[1.0]
    fine = results[dt_fracs[-1]]
    rel_V2 = abs(fine["V2"] - base["V2"]) / abs(base["V2"])
    rel_Lc = abs(fine["L_contact_TJ_sub"] - base["L_contact_TJ_sub"]) / abs(base["L_contact_TJ_sub"])
    rel_F = abs(fine["F"] - base["F"]) / abs(base["F"])
    print(f"  |dV2|/V2 (dt vs {dt_fracs[-1]}*dt) = {rel_V2:.3e}   |dL_contact|/L_contact = {rel_Lc:.3e}   |dF|/F = {rel_F:.3e}")
    return dict(t_test=t_test, results={str(k): v for k, v in results.items()},
                rel_V2=rel_V2, rel_L_contact=rel_Lc, rel_F=rel_F)


def substrate_baseline_x(Y, p, wall_mean):
    return wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)


def classify_branches(f, p, tj_xy, v1, v2, wall_mean, probe_len=None):
    """Classify which of two TJ branch directions is the particle-cap
    branch vs. the substrate-surface branch: walk each ~1W and compare the
    mean X-deviation from the substrate's own analytic baseline x_s(Y) --
    the particle branch bulges well past x_s(Y) into +X, the substrate
    branch tracks close to it."""
    from pf_sintering.curvature_extraction import _walk_branch
    probe_len = probe_len or 2.0 * p.interface_width
    devs = []
    for v in (v1, v2):
        path, s_cum = _walk_branch(f, p, tj_xy, v, probe_len)
        if path is None:
            devs.append(-math.inf)
            continue
        baseline_x = substrate_baseline_x(path[:, 1], p, wall_mean)
        devs.append(float(np.mean(path[:, 0] - baseline_x)))
    if devs[0] >= devs[1]:
        return v1, v2  # particle, substrate
    return v2, v1


def curvature_snapshot(f, e1, e2, e3, s, p, wall_mean):
    """Section 11-12: Delta-kappa / Delta-mu between particle and substrate
    regions at both TJs, in the far window ([2W,3W]) and near-TJ window
    ([0,1W]), via curvature_extraction.branch_window_report.

    Milestone 13 Section 9: output keys renamed from the ambiguous
    "kappa"/"kappa_..._B" pair to explicit kappa_mu_effective_*/kappa_geom_*
    -- the former is a mu-derived proxy (not a validated curvature in the
    bicrystal case), the latter is the actual geometric curvature."""
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    out = {}
    for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            out[label] = None
            continue
        particle_dir, substrate_dir = classify_branches(f, p, tj.tj_xy, tj.v_s1, tj.v_s2, wall_mean)
        w_particle = branch_window_report(f, e1, e2, e3, s, p, tj.tj_xy, particle_dir)
        w_substrate = branch_window_report(f, e1, e2, e3, s, p, tj.tj_xy, substrate_dir)
        near_p, far_p = w_particle[0], w_particle[max(w_particle)]
        near_s, far_s = w_substrate[0], w_substrate[max(w_substrate)]
        out[label] = dict(
            kappa_mu_effective_particle_near_TJ=near_p.kappa_mu_effective, kappa_mu_effective_particle_far=far_p.kappa_mu_effective,
            kappa_mu_effective_substrate_near_TJ=near_s.kappa_mu_effective, kappa_mu_effective_substrate_far=far_s.kappa_mu_effective,
            kappa_geom_particle_near_TJ=near_p.kappa_geom, kappa_geom_particle_far=far_p.kappa_geom,
            kappa_geom_substrate_near_TJ=near_s.kappa_geom, kappa_geom_substrate_far=far_s.kappa_geom,
            mu_particle_far=far_p.mu_mean, mu_substrate_far=far_s.mu_mean,
            delta_kappa_mu_effective_far=(far_p.kappa_mu_effective - far_s.kappa_mu_effective) if (far_p.resolved and far_s.resolved) else math.nan,
            delta_mu_far=(far_p.mu_mean - far_s.mu_mean) if (far_p.resolved and far_s.resolved) else math.nan,
            delta_kappa_mu_effective_near=(near_p.kappa_mu_effective - near_s.kappa_mu_effective) if (near_p.resolved and near_s.resolved) else math.nan,
        )
    return out


def energy_dissipation_check(f, e1, e2, e3, s, p, M_s, dt):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    Mxx, Mxy, Myy = surface_mobility_tensor(f, p.dx, p.interface_width, M_s, BC_X, BC_Y, eps_n=1e-6 / p.interface_width)
    D_density = dissipation_density(Mxx, Mxy, Myy, mu, p.dx, BC_X, BC_Y)
    D_CH = float(np.sum(D_density)) * p.dx * p.dx
    F0 = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    f_new, _ = surface_divergence_update(f, mu, p.dx, dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
    F1 = exact_free_energy_isotropic(f_new, e1, e2, e3, s, p)
    return dict(F0=F0, F1_f_only=F1, dF=F1 - F0, D_CH=D_CH, D_CH_nonneg=bool(D_CH >= -1e-300),
                F_nonincreasing=bool(F1 <= F0 + 1e-6 * abs(F0)))


def run_campaign(dx_nm, args, t_end, sample_every_steps=20, curvature_every_samples=10, wall_mean=None):
    print(f"\n=== State-A unified campaign, dx={dx_nm}nm, t_end={t_end:.4e}s ===")
    p, f, e1, e2, e3 = build_state_a(args, dx_nm)
    if wall_mean is None:
        wall_mean = (p.substrate_wall_frac - 0.5) * p.Nx * p.dx
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    n_steps = round(t_end / p.dt)

    rows = []
    m0 = measure(f, e1, e2, e3, s, p)
    m0["step"] = 0
    m0["time_s"] = 0.0
    ediag_check = energy_dissipation_check(f, e1, e2, e3, s, p, M_s, p.dt)
    m0.update(ediag_check)
    m0["curvature"] = curvature_snapshot(f, e1, e2, e3, s, p, wall_mean)
    rows.append(m0)
    print(f"  step=0 V2={m0['V2']:.8e} L_contact={m0['L_contact_TJ_sub']*1e9:.4f}nm F={m0['F']:.6e}")

    sample_idx = 0
    for step in range(1, n_steps + 1):
        f, e1, e2, e3, diag, ediag = one_transport_step(f, e1, e2, e3, s, p, p.dt, M_s)
        if step % sample_every_steps == 0 or step == n_steps:
            m = measure(f, e1, e2, e3, s, p)
            m["step"] = step
            m["time_s"] = step * p.dt
            sample_idx += 1
            if sample_idx % curvature_every_samples == 0 or step == n_steps:
                m["curvature"] = curvature_snapshot(f, e1, e2, e3, s, p, wall_mean)
                edc = energy_dissipation_check(f, e1, e2, e3, s, p, M_s, p.dt)
                m.update(edc)
            else:
                m["curvature"] = None
            rows.append(m)
            if step % (sample_every_steps * 10) == 0 or step == n_steps:
                print(f"  step={step:6d} t={m['time_s']:.4e}s V2={m['V2']:.8e} "
                      f"L_contact={m['L_contact_TJ_sub']*1e9:.4f}nm total_f_drift="
                      f"{(m['total_f']-m0['total_f'])/m0['total_f']:.2e}")

    return dict(dx_nm=dx_nm, p_dt=p.dt, p_Nx=p.Nx, p_Ny=p.Ny, rows=rows)


def find_zero_crossing(t, y):
    """Locate t_V* where y (e.g. dV2/dt) crosses zero via linear interp
    between consecutive samples; returns None if no sign change."""
    t = np.asarray(t); y = np.asarray(y)
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


def analyze_campaign(result):
    rows = result["rows"]
    t = np.array([r["time_s"] for r in rows])
    V2 = np.array([r["V2"] for r in rows])
    V1 = np.array([r["V1"] for r in rows])
    Lc = np.array([r["L_contact_TJ_sub"] for r in rows])
    dV2dt = np.gradient(V2, t)
    dV1dt = np.gradient(V1, t)
    dLdt = np.gradient(Lc, t)
    t_star = find_zero_crossing(t, dV2dt)
    return dict(t=t.tolist(), V2=V2.tolist(), V1=V1.tolist(), L_contact=Lc.tolist(),
                dV2dt=dV2dt.tolist(), dV1dt=dV1dt.tolist(), dLdt=dLdt.tolist(),
                t_V_star=t_star, V2_sign_final=float(np.sign(V2[-1] - V2[0])),
                dV2dt_sign_final=float(np.sign(dV2dt[-1])))


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
    ap.add_argument("--dx-list-nm", type=str, default="5.0,2.5")
    ap.add_argument("--t-end", type=float, default=0.03)
    ap.add_argument("--sample-every-steps", type=int, default=20)
    ap.add_argument("--curvature-every-samples", type=int, default=10)
    ap.add_argument("--dt-test-steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="runs/m12b_grid_convergence.json")
    args = ap.parse_args()

    dx_list = [float(x) for x in args.dx_list_nm.split(",")]
    out = dict(args=vars(args))

    audit_rows, fixed_ok = audit_dx_dependent_quantities(dx_list, args)
    out["audit"] = audit_rows
    out["audit_fixed_ok"] = fixed_ok

    out["dt_convergence"] = {}
    out["campaigns"] = {}
    out["analysis"] = {}
    for dx_nm in dx_list:
        dtc = dt_convergence_test(dx_nm, args, n_steps_base=args.dt_test_steps)
        out["dt_convergence"][str(dx_nm)] = dtc

        result = run_campaign(dx_nm, args, args.t_end, args.sample_every_steps, args.curvature_every_samples)
        out["campaigns"][str(dx_nm)] = result
        analysis = analyze_campaign(result)
        out["analysis"][str(dx_nm)] = analysis
        print(f"\n  dx={dx_nm}nm summary: V2 sign(final-initial)={analysis['V2_sign_final']:+.0f}  "
              f"dV2/dt sign at t_end={analysis['dV2dt_sign_final']:+.0f}  t_V*={analysis['t_V_star']}")

    print("\n=== Cross-grid comparison ===")
    for dx_nm in dx_list:
        a = out["analysis"][str(dx_nm)]
        print(f"  dx={dx_nm}nm: V2(0)={a['V2'][0]:.6e} V2(end)={a['V2'][-1]:.6e} "
              f"dV2/dt(end)={a['dV2dt'][-1]:.4e} t_V*={a['t_V_star']}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
