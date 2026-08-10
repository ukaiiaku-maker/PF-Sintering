"""Milestone 15E: overnight mechanism-discovery campaign infrastructure.

Shared library for the restartable, incrementally-written Stage A-D
screens. Each "case" is a parameter dict; `run_case` builds the state,
pre-validates it (Section 3's reject-before-dynamics list), runs the
coupled trajectory if valid, computes the stress-amplification metrics
(Section 1: sigma_reset, sigma_peak_after_reset, A_sigma, Delta_sigma),
and writes both a per-case JSON (full trajectory) and a manifest CSV row
(summary only) -- re-running the campaign skips any case_id whose JSON
already exists (Section 18 restartability).
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
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, build_state, run_trajectory, sample_state  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402

M_GB_REF = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
BASELINE_MS_SCALE = 0.3

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m15e_campaign")
MANIFEST_PATH = os.path.join(CAMPAIGN_DIR, "manifest.csv")
MANIFEST_FIELDS = [
    "case_id", "stage", "status", "reason",
    "A_nm", "lambda_nm", "R2_nm", "aspect_ratio", "overlap_nm",
    "gamma_gb_ratio", "gamma_s", "M_GB_scale", "M_s_scale", "dx_nm", "t_target",
    "sigma_reset", "sigma_peak", "A_sigma", "Delta_sigma",
    "t_reset", "t_peak", "L_contact_reset", "L_contact_min", "L_contact_peak",
    "F_cap_n_reset", "F_cap_n_peak", "F_monotonic", "mass_drift_max",
    "wall_seconds",
]


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
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def validate_initial_state(case):
    """Section 3's reject-before-dynamics checks. Returns (ok, reason)."""
    p, f, e1, e2, e3 = build_state(
        dx_nm=case["dx_nm"], W_nm=case.get("W_nm", 20.0), gamma_gb=case["gamma_gb_ratio"] * case["gamma_s"],
        M_GB=case["M_GB_scale"] * M_GB_REF, surface_mobility_scale=BASELINE_MS_SCALE * case["M_s_scale"],
        A_nm=case["A_nm"], lambda_nm=case["lambda_nm"], R2_nm=case["R2_nm"],
        aspect_ratio=case["aspect_ratio"], overlap_nm=case["overlap_nm"],
    )
    fb = np.clip(f, 0.0, 1.0)
    resid = np.abs((e1 + e2 + e3) - fb)
    # model.reproject's own "void" convention deliberately zeroes eta_i
    # wherever fb<=0.005 (avoids numerical noise deep in the diffuse
    # tail) -- a legitimate, pre-existing design choice (Milestones
    # 15/15B), not a construction error; only check the invariant where
    # it is actually supposed to hold.
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
    # trough clearance: the sinusoid's trough (opposite phase from the
    # particle crest) must stay clear of the particle -- amplitude must
    # not approach half the wavelength (self-intersecting substrate) or
    # exceed a generous fraction of R2 (particle swallowed by trough).
    if case["A_nm"] > 0.45 * case["lambda_nm"]:
        return False, "sinusoid_self_intersects"
    return True, ""


def amplification_metrics(rows):
    """Section 1: sigma_reset = min sigma over t>0 samples (the first
    resolved post-transient minimum, given every trajectory seen so far
    in this project has exactly one such minimum before any rebound);
    sigma_peak_after_reset = max sigma at or after that time."""
    ts, sigs, Lcs, Fns = [], [], [], []
    for r in rows:
        sig = r.get("sigma_sint_app_endpoint_form")
        if sig is None or not math.isfinite(sig):
            continue
        ts.append(r["t"])
        sigs.append(sig)
        Lcs.append(r.get("L_contact", float("nan")))
        Fns.append(r.get("F_cap_n_endpoint_form", float("nan")))
    if len(ts) < 2:
        return None
    ts = np.array(ts); sigs = np.array(sigs); Lcs = np.array(Lcs); Fns = np.array(Fns)
    post0 = ts > 0
    if not np.any(post0):
        return None
    i_reset_local = int(np.argmin(sigs[post0]))
    idx_post0 = np.flatnonzero(post0)
    i_reset = idx_post0[i_reset_local]
    sigma_reset = float(sigs[i_reset])
    t_reset = float(ts[i_reset])
    tail = sigs[i_reset:]
    i_peak_local = int(np.argmax(tail))
    i_peak = i_reset + i_peak_local
    sigma_peak = float(sigs[i_peak])
    t_peak = float(ts[i_peak])
    A_sigma = sigma_peak / sigma_reset if sigma_reset > 0 else float("nan")
    return dict(
        sigma_reset=sigma_reset, t_reset=t_reset, sigma_peak=sigma_peak, t_peak=t_peak,
        A_sigma=A_sigma, Delta_sigma=sigma_peak - sigma_reset,
        L_contact_reset=float(Lcs[i_reset]), L_contact_min=float(np.nanmin(Lcs[i_reset:])),
        L_contact_peak=float(Lcs[i_peak]),
        F_cap_n_reset=float(Fns[i_reset]), F_cap_n_peak=float(Fns[i_peak]),
    )


def run_case(case, verbose=True):
    """Idempotent: returns immediately (loading the saved JSON) if
    case_id already has a result on disk."""
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
                                   reason=reason, **{k: case.get(k, "") for k in
                                   ("A_nm", "lambda_nm", "R2_nm", "aspect_ratio", "overlap_nm",
                                    "gamma_gb_ratio", "gamma_s", "M_GB_scale", "M_s_scale", "dx_nm", "t_target")}))
        if verbose:
            print(f"[{case_id}] REJECTED: {reason}")
        return result

    t0 = time.time()
    traj, _ = run_trajectory(
        case["dx_nm"], case["M_GB_scale"] * M_GB_REF, BASELINE_MS_SCALE * case["M_s_scale"],
        case["t_target"], case["sample_times"], case_id,
        W_nm=case.get("W_nm", 20.0), gamma_gb=case["gamma_gb_ratio"] * case["gamma_s"],
        A_nm=case["A_nm"], lambda_nm=case["lambda_nm"], R2_nm=case["R2_nm"],
        aspect_ratio=case["aspect_ratio"], overlap_nm=case["overlap_nm"], verbose=verbose,
    )
    wall = time.time() - t0

    metrics = amplification_metrics(traj["rows"])
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
               **{k: case.get(k, "") for k in
                  ("A_nm", "lambda_nm", "R2_nm", "aspect_ratio", "overlap_nm",
                   "gamma_gb_ratio", "gamma_s", "M_GB_scale", "M_s_scale", "dx_nm", "t_target")})
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
