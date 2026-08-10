"""Milestone 15H Section 8: C0-C3 contact-width sign classification,
computed directly from L_contact(t)'s OWN trajectory (not anchored to
the sigma-reset time, which m15f_campaign_lib.amplification_metrics
uses and which can occur much LATER than an early transient L_contact
dip -- confirmed missed exactly this way for the X_th45_AN3_R3 case,
Section 7: L_contact dips from 32.84nm (t=0.002) to 32.32nm (t=0.01)
while sigma's own minimum doesn't occur until t=0.125, so the
sigma-anchored L_contact_reset/L_contact_min in the manifest showed
Lc_reset==Lc_min, hiding the dip entirely).

C0: L_contact(t) never decreases below its own running minimum-so-far
    after the initial transient -- i.e. monotonically broadening
    (allowing small numerical wobble).
C1: L_contact dips below a subsequent local value but the dip's
    duration (time under the pre-dip local max) is < 0.1s.
C2: a sustained narrowing interval (L_contact strictly below its
    pre-dip local reference) lasting >= 0.1s.
C3: C2, plus sigma accelerating (d^2 sigma/dt^2 > 0, i.e. the stress
    RAMP itself steepening) over the same interval.
"""
from __future__ import annotations

import json
import sys

import numpy as np

CASE_IDS = [
    "XL_th45_AN3_R3", "XL_AN0_R2.5",
    "X_th45_AN3_R1.5", "X_th45_AN3_R2", "X_th45_AN3_R2.5", "X_th45_AN3_R3",
    "X_th45_AN3_R4", "X_th45_AN3_R5", "X_th45_AN3_R6",
    "X_AN0_R1.5", "X_AN0_R2", "X_AN0_R2.5",
]


def classify(case_id, campaign_dir="runs/m15f_campaign"):
    with open(f"{campaign_dir}/{case_id}.json") as fh:
        d = json.load(fh)
    rows = [r for r in d["trajectory"]["rows"] if r.get("L_contact") is not None and r["t"] > 0]
    if len(rows) < 3:
        return dict(case_id=case_id, classification="UNRESOLVED")
    ts = np.array([r["t"] for r in rows])
    Lc = np.array([r["L_contact"] for r in rows])
    sig = np.array([r.get("sigma_sint_app_endpoint_form_aniso", r.get("sigma_sint_app_endpoint_form", np.nan))
                     for r in rows])

    running_max = np.maximum.accumulate(Lc)
    below = Lc < running_max - 1e-13  # strictly below the running local max so far -> narrowing episode
    if not np.any(below):
        return dict(case_id=case_id, classification="C0", max_dip_nm=0.0, dip_duration_s=0.0)

    # find contiguous "below" runs, compute each one's duration and depth
    idx = np.flatnonzero(below)
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))

    best_duration = 0.0
    best_depth_nm = 0.0
    best_run = None
    for (i0, i1) in runs:
        duration = ts[i1] - ts[max(0, i0 - 1)]
        ref = running_max[max(0, i0 - 1)]
        depth = float((ref - Lc[i0:i1 + 1].min()) * 1e9)
        if duration > best_duration:
            best_duration = duration
            best_depth_nm = depth
            best_run = (i0, i1)

    if best_duration >= 0.1:
        classification = "C2"
        if best_run is not None:
            i0, i1 = best_run
            seg_t, seg_s = ts[i0:i1 + 1], sig[i0:i1 + 1]
            if len(seg_t) >= 3 and np.all(np.isfinite(seg_s)):
                d1 = np.gradient(seg_s, seg_t)
                d2 = np.gradient(d1, seg_t)
                if np.mean(d2) > 0 and np.mean(d1) > 0:
                    classification = "C3"
    else:
        classification = "C1"
    return dict(case_id=case_id, classification=classification, max_dip_nm=best_depth_nm,
                dip_duration_s=best_duration)


if __name__ == "__main__":
    for cid in CASE_IDS:
        try:
            r = classify(cid)
        except FileNotFoundError:
            print(f"{cid}: not found")
            continue
        print(f"{r['case_id']:20s} {r['classification']:4s} max_dip={r.get('max_dip_nm', 0):.3f}nm "
              f"dip_duration={r.get('dip_duration_s', 0):.4f}s")
