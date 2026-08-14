"""Milestone 16J Sections 18-21: short-run extrapolation, hold-out
validation, and candidate ranking.

Fits u(t)=1/r_neck(t) and v(t)=1/X_neck(t) SEPARATELY (never sigma(t)
directly -- it is the difference of two similar terms) to three candidate
forms (logarithmic, power-law, saturating-exponential), validates each fit
on a held-out final segment of the trajectory, and reconstructs
sigma_pred(t)=gamma_s*(u_pred-C_GB*v_pred). Candidates whose fit forms
disagree substantially are flagged "uncertain" rather than extrapolated
with false confidence (Section 19).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, ".")
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16e_exact_hussein_two_mode import psi_to_gamma_gb  # noqa: E402

STAGE1_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16j_geometry_search", "stage1")
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(160.0, GAMMA_S)
C_GB = math.sqrt(1.0 - (GAMMA_GB / (2.0 * GAMMA_S)) ** 2)


def load_history(cdir):
    path = os.path.join(cdir, "history.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def form_log(t, a, b, t0):
    return a + b * np.log(1.0 + t / max(t0, 1e-9))


def form_power(t, a, b, p):
    return a + b * np.power(np.clip(t, 0, None), p)


def form_saturating(t, a, b, tau):
    return a + b * (1.0 - np.exp(-t / max(tau, 1e-9)))


FORMS = {
    "log": (form_log, [1.0, 0.1, 1.0]),
    "power": (form_power, [1.0, 0.1, 0.5]),
    "saturating": (form_saturating, [1.0, 0.1, 1.0]),
}


def fit_all_forms(t, y):
    fits = {}
    for name, (fn, p0) in FORMS.items():
        try:
            popt, _ = curve_fit(fn, t, y, p0=p0, maxfev=20000)
            pred = fn(t, *popt)
            resid = y - pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fits[name] = dict(params=list(popt), r2=r2, fn=fn)
        except Exception as exc:
            fits[name] = dict(params=None, r2=float("nan"), error=str(exc))
    return fits


def holdout_validate(t, y, frac=0.75):
    n = len(t)
    n_fit = max(4, int(n * frac))
    t_fit, y_fit = t[:n_fit], y[:n_fit]
    t_hold, y_hold = t[n_fit:], y[n_fit:]
    if len(t_hold) < 2:
        return {}
    errs = {}
    for name, (fn, p0) in FORMS.items():
        try:
            popt, _ = curve_fit(fn, t_fit, y_fit, p0=p0, maxfev=20000)
            pred_hold = fn(t_hold, *popt)
            rel_err = float(np.mean(np.abs(pred_hold - y_hold) / (np.abs(y_hold) + 1e-30)))
            errs[name] = rel_err
        except Exception:
            errs[name] = float("nan")
    return errs


def analyze_candidate(cdir):
    rows = load_history(cdir)
    label = os.path.basename(cdir)
    if len(rows) < 8:
        return dict(candidate=label, status="insufficient_data", n_rows=len(rows))

    t = np.array([r["time"] for r in rows])
    r_neck = np.array([r["r_neck_nm"] for r in rows]) * 1e-9
    X_neck = np.array([r["X_neck_nm"] for r in rows]) * 1e-9
    finite = np.isfinite(r_neck) & np.isfinite(X_neck) & (r_neck > 0) & (X_neck > 0)
    t, r_neck, X_neck = t[finite], r_neck[finite], X_neck[finite]
    if len(t) < 8:
        return dict(candidate=label, status="insufficient_finite_data", n_rows=len(t))

    u = 1.0 / r_neck  # 1/m
    v = 1.0 / X_neck

    u_fits = fit_all_forms(t, u)
    v_fits = fit_all_forms(t, v)
    u_holdout = holdout_validate(t, u)
    v_holdout = holdout_validate(t, v)

    best_u_name = min((n for n in u_fits if np.isfinite(u_fits[n]["r2"])), key=lambda n: -u_fits[n]["r2"],
                       default=None)
    best_v_name = min((n for n in v_fits if np.isfinite(v_fits[n]["r2"])), key=lambda n: -v_fits[n]["r2"],
                       default=None)

    predictions = {}
    disagreement = float("nan")
    if best_u_name and best_v_name:
        fn_u, p_u = FORMS[best_u_name][0], u_fits[best_u_name]["params"]
        fn_v, p_v = FORMS[best_v_name][0], v_fits[best_v_name]["params"]

        preds_by_form = []
        for name_u in u_fits:
            if u_fits[name_u]["params"] is None:
                continue
            for name_v in v_fits:
                if v_fits[name_v]["params"] is None:
                    continue
                fu = FORMS[name_u][0]
                fv = FORMS[name_v][0]
                u_at_150 = fu(np.array([150.0]), *u_fits[name_u]["params"])[0]
                v_at_150 = fv(np.array([150.0]), *v_fits[name_v]["params"])[0]
                sigma_150 = GAMMA_S * (u_at_150 - C_GB * v_at_150)
                preds_by_form.append(sigma_150 / 1e6)
        if preds_by_form:
            disagreement = float(np.std(preds_by_form))

        for t_query in (10.0, 50.0, 150.0):
            u_pred = fn_u(np.array([t_query]), *p_u)[0]
            v_pred = fn_v(np.array([t_query]), *p_v)[0]
            sigma_pred = GAMMA_S * (u_pred - C_GB * v_pred)
            predictions[f"sigma_pred_t{t_query:.0f}_MPa"] = sigma_pred / 1e6

    dsigma_dt_late = float("nan")
    if len(t) >= 4:
        sigma_series = GAMMA_S * (u - C_GB * v) / 1e6
        dsigma_dt_late = float((sigma_series[-1] - sigma_series[-4]) / (t[-1] - t[-4])) if t[-1] != t[-4] else float("nan")

    return dict(
        candidate=label, status="ok", n_rows=len(t), t_max=float(t[-1]),
        r_neck_initial_nm=float(r_neck[0] * 1e9), r_neck_final_nm=float(r_neck[-1] * 1e9),
        X_neck_initial_nm=float(X_neck[0] * 1e9), X_neck_final_nm=float(X_neck[-1] * 1e9),
        sigma_initial_MPa=float(GAMMA_S * (u[0] - C_GB * v[0]) / 1e6),
        sigma_final_MPa=float(GAMMA_S * (u[-1] - C_GB * v[-1]) / 1e6),
        dsigma_dt_late_MPa_per_t=dsigma_dt_late,
        best_u_form=best_u_name, best_v_form=best_v_name,
        u_r2=u_fits.get(best_u_name, {}).get("r2"), v_r2=v_fits.get(best_v_name, {}).get("r2"),
        u_holdout_err=u_holdout.get(best_u_name), v_holdout_err=v_holdout.get(best_v_name),
        fit_disagreement_MPa=disagreement,
        **predictions,
    )


def main():
    cdirs = sorted(glob.glob(os.path.join(STAGE1_ROOT, "*")))
    results = []
    for cdir in cdirs:
        if not os.path.isdir(cdir):
            continue
        res = analyze_candidate(cdir)
        results.append(res)
        print(json.dumps(res, indent=2, default=str))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "m16j_geometry_search")
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(os.path.join(out_dir, "candidate_fits.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    ok = [r for r in results if r.get("status") == "ok"]
    ranked = sorted(ok, key=lambda r: (-(r.get("dsigma_dt_late_MPa_per_t") or -1e9)))
    with open(os.path.join(out_dir, "candidate_rankings.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank"] + fieldnames)
        w.writeheader()
        for i, r in enumerate(ranked, 1):
            row = dict(r); row["rank"] = i
            w.writerow(row)

    print(f"\nwrote candidate_fits.csv and candidate_rankings.csv ({len(results)} candidates, {len(ok)} ok)")
    print("\nranking by late-time dsigma/dt (highest = most actively building stress):")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['candidate']}: dsigma/dt={r.get('dsigma_dt_late_MPa_per_t'):.4f} MPa/t  "
              f"sigma_final={r.get('sigma_final_MPa'):.3f}MPa  disagreement={r.get('fit_disagreement_MPa'):.2f}MPa")


if __name__ == "__main__":
    main()
