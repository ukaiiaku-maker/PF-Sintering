"""Milestone 16A Section 8: fit exponential growth/decay rate omega from
eps(t)~eps0*exp(omega*t) in the (early, small-amplitude) linear regime,
per case, and report against the classical surface-diffusion dispersion
sign/zero-crossing structure."""
from __future__ import annotations

import json
import sys

import numpy as np

CAMPAIGN_DIR = "runs/m16a_campaign"


def fit_omega(case_id, frac_early=0.5):
    with open(f"{CAMPAIGN_DIR}/{case_id}.json") as fh:
        d = json.load(fh)
    rows = d["rows"]
    n_use = max(3, int(len(rows) * frac_early))
    t = np.array([r["t"] for r in rows[:n_use]])
    amp = np.array([r["amplitude"] for r in rows[:n_use]])
    logamp = np.log(amp)
    A = np.vstack([t, np.ones_like(t)]).T
    (omega, logamp0), *_ = np.linalg.lstsq(A, logamp, rcond=None)
    pred = A @ np.array([omega, logamp0])
    ss_res = np.sum((logamp - pred) ** 2)
    ss_tot = np.sum((logamp - np.mean(logamp)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(case_id=case_id, lam_over_R0=d.get("lam_over_R0"), lam_nm=d.get("lam_nm"),
                omega=float(omega), r2=float(r2), amp0=amp[0], amp_end=amp[-1])


if __name__ == "__main__":
    import sys
    case_ids = sys.argv[1:] if len(sys.argv) > 1 else [
        "pr_main_R0-40_lamR0-4", "pr_main_R0-40_lamR0-5.5", "pr_main_R0-40_lamR0-2pi",
        "pr_main_R0-40_lamR0-7", "pr_main_R0-40_lamR0-8",
    ]
    print(f"{'case':32s} {'lam/R0':>8s} {'omega':>14s} {'R2':>8s} {'amp0(nm)':>10s} {'amp_end(nm)':>12s}")
    for cid in case_ids:
        try:
            r = fit_omega(cid)
        except FileNotFoundError:
            print(f"{cid}: not found")
            continue
        print(f"{r['case_id']:32s} {r['lam_over_R0']:8.3f} {r['omega']:14.6e} {r['r2']:8.4f} "
              f"{r['amp0']*1e9:10.4f} {r['amp_end']*1e9:12.4f}")
