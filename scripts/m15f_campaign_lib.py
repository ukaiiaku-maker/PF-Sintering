"""Milestone 15F: width-convergence + anisotropy campaign infrastructure.

Extends Milestone 15E's restartable campaign pattern
(scripts/m15e_campaign_lib.py) with two new axes: W_nm (interface_width,
via build_config's existing interface_width_override -- already
W-invariant-physics-correct, see m15f_width_invariant_audit.py) and
anisotropy (use_aniso_surface/aniso_delta/theta_mis_deg).

CRITICAL: `scripts/m15_gb_surface_rate_competition.py`'s `run_trajectory`
computes `mu` via `mu_isotropic` UNCONDITIONALLY -- p.use_aniso_surface
has NO EFFECT on the actual dynamics there (Section 4 audit finding).
`run_trajectory_f` below is the anisotropy-aware replacement (uses
pf_sintering.aniso_flux.mu_anisotropic when p.use_aniso_surface, else the
bit-identical original path) -- everything else (eta update, face-
projected flux step, sampling) is unchanged from the M15 series.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from m15_gb_surface_rate_competition import (  # noqa: E402
    BC_X, BC_Y, M_ETA_HISTORICAL_REF, build_config, build_state, energy_ledger, sample_state,
)
from pf_sintering.aniso_capillary import capillary_force_endpoint_form_aniso  # noqa: E402
from pf_sintering.aniso_flux import aniso_grad_energy_density, mu_anisotropic  # noqa: E402
from pf_sintering.ch_exact_energy import mu_isotropic  # noqa: E402
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update, local_wc  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step  # noqa: E402

M_GB_REF = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
BASELINE_MS_SCALE = 0.3

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15f_campaign")
MANIFEST_PATH = os.path.join(CAMPAIGN_DIR, "manifest.csv")
MANIFEST_FIELDS = [
    "case_id", "stage", "status", "reason",
    "A_nm", "lambda_nm", "R2_nm", "aspect_ratio", "overlap_nm",
    "gamma_gb_ratio", "gamma_s", "M_GB_scale", "M_s_scale", "dx_nm", "W_nm", "t_target",
    "use_aniso_surface", "aniso_delta", "theta_mis_deg", "aniso_label",
    "sigma_reset", "sigma_peak", "A_sigma", "Delta_sigma",
    "t_reset", "t_peak", "L_contact_reset", "L_contact_min", "L_contact_peak",
    "F_cap_n_reset", "F_cap_n_peak", "F_monotonic", "mass_drift_max",
    "wall_seconds",
]
CASE_KEYS = ("A_nm", "lambda_nm", "R2_nm", "aspect_ratio", "overlap_nm",
             "gamma_gb_ratio", "gamma_s", "M_GB_scale", "M_s_scale", "dx_nm", "W_nm", "t_target",
             "use_aniso_surface", "aniso_delta", "theta_mis_deg", "aniso_label")


def case_path(case_id):
    return os.path.join(CAMPAIGN_DIR, f"{case_id}.json")


def already_done(case_id):
    return os.path.exists(case_path(case_id))


def _ensure_manifest():
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS).writeheader()


def _append_manifest_row(row):
    _ensure_manifest()
    with open(MANIFEST_PATH, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS).writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def energy_ledger_f(f, e1, e2, e3, s, p, bc_x=BC_X, bc_y=BC_Y):
    """Anisotropy-aware energy_ledger: replaces the isotropic gradient
    term with aniso_flux.aniso_grad_energy_density when
    p.use_aniso_surface (else bit-identical to the M15 series'
    energy_ledger)."""
    if not p.use_aniso_surface:
        return energy_ledger(f, e1, e2, e3, s, p)
    fb = np.clip(f, 0.0, 1.0)
    Wc = local_wc(f, e1, e2, e3, s, p)
    eta2 = e1 * e1 + e2 * e2 + e3 * e3
    grad_energy = aniso_grad_energy_density(f, e1, e2, e3, p, bc_x, bc_y)
    E_surface = float(np.sum(0.5 * p.W_f * f * f * (1 - f) ** 2 + grad_energy)) * p.dx * p.dx
    E_coupling = float(np.sum(Wc * eta2 * (0.5 * fb * fb - fb))) * p.dx * p.dx
    from pf_sintering.bc_ops import lap9_bc
    grad_term = e1 * lap9_bc(e1, p.dx, bc_x=bc_x, bc_y=bc_y) + e2 * lap9_bc(e2, p.dx, bc_x=bc_x, bc_y=bc_y)
    if p.use_eta3:
        grad_term = grad_term + e3 * lap9_bc(e3, p.dx, bc_x=bc_x, bc_y=bc_y)
    E_GB = float(np.sum(-0.5 * p.k_eta * grad_term)) * p.dx * p.dx
    return dict(E_surface=E_surface, E_coupling=E_coupling, E_GB=E_GB, F_total=E_surface + E_coupling + E_GB)


def sample_state_f(f, e1, e2, e3, s, p, step, t, mass0, F0, cum_safety, cum_variational):
    """sample_state plus: anisotropic F_total (overriding the isotropic
    one when p.use_aniso_surface), anisotropic endpoint capillary force
    F_cap_endpoint_form_aniso + raw xi_start/xi_end vectors, and facet
    orientation diagnostics (Section 13) at both TJs' particle tangents."""
    out = sample_state(f, e1, e2, e3, s, p, step, t, mass0, F0, cum_safety, cum_variational)
    if p.use_aniso_surface:
        led = energy_ledger_f(f, e1, e2, e3, s, p)
        out.update(led)
        out["dF"] = led["F_total"] - F0
        if out.get("top_resolved") and out.get("bottom_resolved") and "top_particle_tangent" in out:
            tj_top = np.array(out["tj_top_xy"])
            tj_bot = np.array(out["tj_bottom_xy"])
            top_dir = np.array(out["top_particle_tangent"])
            bot_dir = np.array(out["bot_particle_tangent"])
            Fx, Fy, xi_s, xi_e = capillary_force_endpoint_form_aniso(f, tj_top, tj_bot, top_dir, bot_dir, p)
            out["F_cap_endpoint_form_aniso"] = [Fx, Fy]
            out["xi_start"] = [float(v) for v in xi_s]
            out["xi_end"] = [float(v) for v in xi_e]
            L_contact = out.get("L_contact")
            if L_contact is not None and math.isfinite(L_contact) and L_contact > 0:
                from pf_sintering.capillary_stress import apparent_sintering_stress, oriented_contact_normal
                from pf_sintering.tj_subgrid import compute_subgrid_contact
                sub = compute_subgrid_contact(f, e1, e2, p)
                if sub.resolved:
                    n_GB = oriented_contact_normal(sub.n_GB_sub)
                    sigma_a, Fn_a = apparent_sintering_stress((Fx, Fy), n_GB, L_contact)
                    out["sigma_sint_app_endpoint_form_aniso"] = sigma_a
                    out["F_cap_n_endpoint_form_aniso"] = Fn_a
            # facet diagnostics: how close is each end's local surface normal
            # to the anisotropy's OWN low-energy (easy) orientation theta0?
            theta0 = float(p.theta_grain[1])
            from pf_sintering.aniso_capillary import _tangent_normal_theta
            _n_top, th_top = _tangent_normal_theta(f, tj_top, top_dir, p)
            _n_bot, th_bot = _tangent_normal_theta(f, tj_bot, -bot_dir, p)
            ps_top = math.degrees((th_top - theta0) % (math.pi / 2))
            ps_bot = math.degrees((th_bot - theta0) % (math.pi / 2))
            out["facet_offset_top_deg"] = min(ps_top, 90.0 - ps_top)
            out["facet_offset_bottom_deg"] = min(ps_bot, 90.0 - ps_bot)
    return out


def run_trajectory_f(dx_nm, M_GB, surface_mobility_scale, t_target, sample_times, label,
                      W_nm=20.0, gamma_gb=1.0, A_nm=100.0, lambda_nm=320.0, R2_nm=80.0,
                      aspect_ratio=2.0, overlap_nm=20.0, verbose=True, dt_override=None,
                      use_aniso_surface=False, aniso_delta=None, theta_mis_deg=0.0):
    p, f, e1, e2, e3 = build_state(dx_nm=dx_nm, W_nm=W_nm, gamma_gb=gamma_gb, M_GB=M_GB,
                                    surface_mobility_scale=surface_mobility_scale, A_nm=A_nm,
                                    lambda_nm=lambda_nm, R2_nm=R2_nm, aspect_ratio=aspect_ratio,
                                    overlap_nm=overlap_nm, dt_override=dt_override,
                                    use_aniso_surface=use_aniso_surface, aniso_delta=aniso_delta,
                                    theta_mis_deg=theta_mis_deg)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    p._M_s_cache = M_s
    dt = p.dt
    n_target = round(t_target / dt)
    sample_steps = sorted(set(round(ts / dt) for ts in sample_times if ts <= t_target) | {0, n_target})

    mass0 = float(f.sum()) * p.dx * p.dx
    F0 = energy_ledger_f(f, e1, e2, e3, s, p)["F_total"]
    if verbose:
        print(f"=== {label}: dx={dx_nm}nm W={W_nm}nm aniso={use_aniso_surface}(d={aniso_delta}) "
              f"M_GB={M_GB} M_s={M_s:.4e} dt={dt:.4e}s n_target={n_target} Nx={p.Nx} Ny={p.Ny} "
              f"M_eta={p.M_eta:.4e} ===")

    rows = []
    cum_safety = 0.0
    cum_variational = 0.0
    t0_wall = time.time()

    def do_sample(step, t):
        row = sample_state_f(f, e1, e2, e3, s, p, step, t, mass0, F0, cum_safety, cum_variational)
        rows.append(row)
        if verbose:
            sig = row.get("sigma_sint_app_endpoint_form_aniso", row.get("sigma_sint_app_endpoint_form", math.nan))
            lgb = row.get("L_GB", math.nan)
            print(f"  step={step} t={t:.4e}s F={row['F_total']:.6e} dF={row['dF']:.3e} "
                  f"mass_drift={row['mass_drift']:.2e} L_GB={lgb*1e9 if math.isfinite(lgb) else float('nan'):.2f}nm "
                  f"sigma_app={sig if sig is not None else float('nan'):.4e}")

    do_sample(0, 0.0)
    step = 0
    for target in sample_steps:
        if target == 0:
            continue
        while step < target:
            if p.use_aniso_surface:
                mu = mu_anisotropic(f, e1, e2, e3, s, p, BC_X, BC_Y)
            else:
                mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, _fdiag = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s,
                                                             bc_x=BC_X, bc_y=BC_Y)
            e1, e2, e3, ediag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                                      bc_x=BC_X, bc_y=BC_Y)
            cum_safety += ediag["safety_correction"]
            cum_variational += ediag["variational_change"]
            step += 1
        do_sample(step, step * dt)

    wall = time.time() - t0_wall
    if verbose:
        print(f"  ({label} done in {wall:.1f}s wall)")

    return dict(label=label, dx_nm=dx_nm, W_nm=W_nm, M_GB=M_GB, M_s=M_s, dt=dt, M_eta=p.M_eta,
                Nx=p.Nx, Ny=p.Ny, n_target=n_target, mass0=mass0, F0=F0, rows=rows,
                wall_seconds=wall)


def validate_initial_state(case):
    """Section 3/11's reject-before-dynamics checks -- same substance as
    m15e_campaign_lib.validate_initial_state, generalized to W_nm and
    anisotropy (aniso itself doesn't change the initial f/e1/e2/e3
    construction, only the dynamics -- initial-state checks are physics-
    invariant)."""
    p, f, e1, e2, e3 = build_state(
        dx_nm=case["dx_nm"], W_nm=case.get("W_nm", 20.0), gamma_gb=case["gamma_gb_ratio"] * case["gamma_s"],
        M_GB=case["M_GB_scale"] * M_GB_REF, surface_mobility_scale=BASELINE_MS_SCALE * case["M_s_scale"],
        A_nm=case["A_nm"], lambda_nm=case["lambda_nm"], R2_nm=case["R2_nm"],
        aspect_ratio=case["aspect_ratio"], overlap_nm=case["overlap_nm"],
        use_aniso_surface=case.get("use_aniso_surface", False), aniso_delta=case.get("aniso_delta"),
        theta_mis_deg=case.get("theta_mis_deg", 0.0),
    )
    fb = np.clip(f, 0.0, 1.0)
    resid = np.abs((e1 + e2 + e3) - fb)
    solid_mask = fb > 0.005
    if solid_mask.any() and resid[solid_mask].max() > 1e-8:
        return False, f"eta_sum_mismatch_{resid[solid_mask].max():.2e}"
    if e1.min() < -1e-9 or e2.min() < -1e-9:
        return False, "negative_eta"
    if f.max() > 1.0 + 1e-6 or f.min() < -1e-6:
        return False, "f_out_of_range"
    s = Sink(threshold=math.inf)
    out = sample_state(f, e1, e2, e3, s, p, 0, 0.0, float(f.sum()) * p.dx * p.dx, 0.0, 0.0, 0.0)
    if not out.get("tj_resolved"):
        return False, "tj_not_resolved"
    if not (out.get("top_resolved") and out.get("bottom_resolved")):
        return False, "tj_force_not_resolved"
    if not out.get("subgrid_resolved"):
        return False, "subgrid_contact_not_resolved"
    if not out.get("arc_resolved"):
        return False, "particle_arc_not_resolved"
    L_contact = out.get("L_contact")
    if L_contact is None or not math.isfinite(L_contact) or L_contact <= 0:
        return False, "L_contact_invalid"
    if case["A_nm"] > 0.45 * case["lambda_nm"]:
        return False, "sinusoid_self_intersects"
    return True, ""


def amplification_metrics(rows, use_aniso):
    key = "sigma_sint_app_endpoint_form_aniso" if use_aniso else "sigma_sint_app_endpoint_form"
    fallback_key = "sigma_sint_app_endpoint_form"
    Fkey = "F_cap_n_endpoint_form_aniso" if use_aniso else "F_cap_n_endpoint_form"
    Ffallback = "F_cap_n_endpoint_form"
    ts, sigs, Lcs, Fns = [], [], [], []
    for r in rows:
        sig = r.get(key)
        if sig is None:
            sig = r.get(fallback_key)
        if sig is None or not math.isfinite(sig):
            continue
        ts.append(r["t"])
        sigs.append(sig)
        Lcs.append(r.get("L_contact", float("nan")))
        fn = r.get(Fkey, r.get(Ffallback, float("nan")))
        Fns.append(fn)
    if len(ts) < 2:
        return None
    ts = np.array(ts); sigs = np.array(sigs); Lcs = np.array(Lcs); Fns = np.array(Fns)
    post0 = ts > 0
    if not np.any(post0):
        return None
    idx_post0 = np.flatnonzero(post0)
    i_reset = idx_post0[int(np.argmin(sigs[post0]))]
    sigma_reset = float(sigs[i_reset]); t_reset = float(ts[i_reset])
    tail = sigs[i_reset:]
    i_peak = i_reset + int(np.argmax(tail))
    sigma_peak = float(sigs[i_peak]); t_peak = float(ts[i_peak])
    A_sigma = sigma_peak / sigma_reset if sigma_reset > 0 else float("nan")
    return dict(
        sigma_reset=sigma_reset, t_reset=t_reset, sigma_peak=sigma_peak, t_peak=t_peak,
        A_sigma=A_sigma, Delta_sigma=sigma_peak - sigma_reset,
        L_contact_reset=float(Lcs[i_reset]), L_contact_min=float(np.nanmin(Lcs[i_reset:])),
        L_contact_peak=float(Lcs[i_peak]),
        F_cap_n_reset=float(Fns[i_reset]), F_cap_n_peak=float(Fns[i_peak]),
    )


def run_case(case, verbose=True):
    case_id = case["case_id"]
    if already_done(case_id):
        with open(case_path(case_id)) as fh:
            return json.load(fh)

    ok, reason = validate_initial_state(case)
    if not ok:
        result = dict(case=case, status="rejected", reason=reason)
        with open(case_path(case_id), "w") as fh:
            json.dump(result, fh, default=str)
        _append_manifest_row(dict(case_id=case_id, stage=case.get("stage", ""), status="rejected",
                                   reason=reason, **{k: case.get(k, "") for k in CASE_KEYS}))
        if verbose:
            print(f"[{case_id}] REJECTED: {reason}")
        return result

    t0 = time.time()
    use_aniso = case.get("use_aniso_surface", False)
    traj = run_trajectory_f(
        case["dx_nm"], case["M_GB_scale"] * M_GB_REF, BASELINE_MS_SCALE * case["M_s_scale"],
        case["t_target"], case["sample_times"], case_id,
        W_nm=case.get("W_nm", 20.0), gamma_gb=case["gamma_gb_ratio"] * case["gamma_s"],
        A_nm=case["A_nm"], lambda_nm=case["lambda_nm"], R2_nm=case["R2_nm"],
        aspect_ratio=case["aspect_ratio"], overlap_nm=case["overlap_nm"], verbose=verbose,
        use_aniso_surface=use_aniso, aniso_delta=case.get("aniso_delta"),
        theta_mis_deg=case.get("theta_mis_deg", 0.0),
    )
    wall = time.time() - t0

    metrics = amplification_metrics(traj["rows"], use_aniso)
    F_series = [r["F_total"] for r in traj["rows"]]
    F_monotonic = all(F_series[i + 1] <= F_series[i] + 1e-18 for i in range(len(F_series) - 1))
    mass_drifts = [abs(r["mass_drift"]) for r in traj["rows"]]
    mass_drift_max = max(mass_drifts) if mass_drifts else float("nan")

    status = "ok" if metrics is not None else "unresolved"
    result = dict(case=case, status=status, wall_seconds=wall, F_monotonic=F_monotonic,
                  mass_drift_max=mass_drift_max, metrics=metrics, trajectory=traj)
    with open(case_path(case_id), "w") as fh:
        json.dump(result, fh, default=str)

    row = dict(case_id=case_id, stage=case.get("stage", ""), status=status, reason="",
               wall_seconds=wall, F_monotonic=F_monotonic, mass_drift_max=mass_drift_max,
               **{k: case.get(k, "") for k in CASE_KEYS})
    if metrics:
        row.update(metrics)
        row["sigma_peak"] = metrics["sigma_peak"]
    _append_manifest_row(row)
    if verbose:
        if metrics:
            print(f"[{case_id}] A_sigma={metrics['A_sigma']:.3f} sigma_reset={metrics['sigma_reset']/1e6:.2f}MPa "
                  f"sigma_peak={metrics['sigma_peak']/1e6:.2f}MPa L_contact_reset={metrics['L_contact_reset']*1e9:.2f}nm "
                  f"L_contact_min={metrics['L_contact_min']*1e9:.2f}nm F_monotonic={F_monotonic} "
                  f"mass_drift_max={mass_drift_max:.2e} wall={wall:.1f}s")
        else:
            print(f"[{case_id}] status={status} wall={wall:.1f}s")
    return result


def run_cases(cases, verbose=True):
    results = []
    for case in cases:
        if already_done(case["case_id"]):
            if verbose:
                print(f"[{case['case_id']}] already done, skipping")
            with open(case_path(case["case_id"])) as fh:
                results.append(json.load(fh))
            continue
        results.append(run_case(case, verbose=verbose))
    return results
