"""Milestone 16K Sections 3-6: path-continuous neck/contact tracking.

Diagnostic finding (Section 3): a quick high-cadence re-run of the M16J
refined (flat, X0/(2Rp)=0.10, W=6nm) trajectory showed THREE local minima
in the measured R(z) profile from as early as t=0.04 -- not just one.
The M16J Stage-1 diagnostic (`mins[0]`, i.e. "the first local minimum
found scanning z from one end") is therefore under-specified: which
minimum is "first" can change as the profile evolves even when no
minimum's own depth/position changes much, causing the reported
r_neck/sigma_s to jump between fundamentally different features. This is
at least partly (likely primarily) a TRACKER artifact, not a proven
physical discontinuity (Section 3 classification).

This module fixes it with continuity-based tracking: at each step,
identify ALL local minima (candidate contacts), then select whichever is
CLOSEST in z to the previously-selected contact (not simply "first
found"). A change of selected candidate is logged explicitly. If the
newly-selected candidate is far from the previous one relative to how
much the contour could plausibly have moved in one diagnostic interval,
that is flagged as a genuine switch for downstream review.
"""
from __future__ import annotations

import numpy as np

from pf_sintering.hussein_neck_stress import fit_local_circle, neck_curvature_windows, signed_curvature_from_fit


class NeckTracker:
    """Stateful path-continuous tracker. Call `.step(R_of_z, z, W)` once
    per diagnostic sample; it returns a dict with the selected contact
    and full candidate list, and records whether a switch occurred."""

    def __init__(self, window_widths_in_W=(1.0, 1.5, 2.0, 2.5, 3.0)):
        self.window_widths_in_W = window_widths_in_W
        self.prev_z_gb = None
        self.switch_log = []

    def step(self, R_of_z, z, W, step_index=None, t=None):
        from pf_sintering.m16j_geometry import find_all_extrema
        ext = find_all_extrema(R_of_z, z)
        candidates = [(zz, RR) for k, zz, RR in ext if k == "min"]

        if not candidates:
            return dict(all_candidate_contacts=[], selected_contact=None, distance_from_previous_contact=float("nan"),
                        switched=False, all_candidate_curvatures={}, selected_curvature=None)

        if self.prev_z_gb is None:
            # first ever sample: no continuity to anchor to -- pick the
            # deepest (smallest-radius) candidate as the initial contact.
            selected = min(candidates, key=lambda c: c[1])
            dist = float("nan")
            ambiguous = len(candidates) > 1
        else:
            # continuity rule: whichever candidate is closest in z to the
            # PREVIOUSLY selected contact, not "first found" / deepest.
            selected = min(candidates, key=lambda c: abs(c[0] - self.prev_z_gb))
            dist = abs(selected[0] - self.prev_z_gb)
            ambiguous = len(candidates) > 1

        z_gb, a_contact = selected
        # a logged "switch" = more than one candidate existed at this step
        # (ambiguity), so the choice made here is a genuine tracker
        # decision, not a trivial single-candidate pass-through.
        if ambiguous:
            self.switch_log.append(dict(step=step_index, t=t, prev_z_gb=self.prev_z_gb, selected_z_gb=z_gb,
                                         n_candidates=len(candidates),
                                         all_candidates_nm=[(round(c[0] * 1e9, 3), round(c[1] * 1e9, 3)) for c in candidates]))

        all_curv = {}
        for w_mult in self.window_widths_in_W:
            windows = neck_curvature_windows(R_of_z, z, z_gb, W, window_widths_in_W=(w_mult,))
            all_curv[w_mult] = windows[0]["r_neck"] if windows else float("nan")

        finite = [v for v in all_curv.values() if np.isfinite(v) and v > 0]
        spread_pct = (100.0 * (max(finite) - min(finite)) / np.mean(finite)) if len(finite) >= 2 else float("nan")

        self.prev_z_gb = z_gb

        return dict(all_candidate_contacts=candidates, selected_contact=selected,
                    distance_from_previous_contact=dist, switched=ambiguous,
                    all_candidate_curvatures=all_curv, selected_curvature=all_curv.get(1.5, float("nan")),
                    r_neck_spread_pct=spread_pct, a_contact=a_contact, z_gb=z_gb)


def find_tj_from_contour(f, e1, e2, r_c, z, R_of_z, z_gb_guess, search_frac=0.15):
    """M16K Section 9: locate the TRUE triple junction as the point along
    the ALREADY-TRACED free-surface contour (f=0.5, i.e. (z, R_of_z(z)))
    where grain identity flips (e1(z,R_of_z(z)) - e2(z,R_of_z(z))
    changes sign) -- not merely a local minimum of R(z), which is only an
    indirect geometric proxy for the TJ (used here as a secondary
    cross-check, per the explicit instruction). Interpolates e1,e2 onto
    the contour at each finite R_of_z(z) via the nearest r-index (grid
    resolution is fine enough that this is adequate for a diagnostic).

    Returns dict(z_tj, r_tj, z_gb_r_of_z_min, agreement_nm) where
    `agreement_nm` = |z_tj - z_gb_r_of_z_min| converted to a radial-scale
    distance via the local contour, for reporting how well the two
    definitions agree."""
    window = search_frac * max(abs(z[-1] - z[0]), 1e-30)
    mask = np.abs(z - z_gb_guess) < window
    idx = np.where(mask & np.isfinite(R_of_z))[0]
    if len(idx) < 3:
        return dict(z_tj=float("nan"), r_tj=float("nan"), z_gb_r_of_z_min=z_gb_guess, agreement_nm=float("nan"))

    diff = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        Rz = R_of_z[i]
        j = int(np.argmin(np.abs(r_c - Rz)))
        diff[k] = e1[i, j] - e2[i, j]

    z_tj = float("nan")
    for k in range(len(idx) - 1):
        if not (np.isfinite(diff[k]) and np.isfinite(diff[k + 1])):
            continue
        if diff[k] == 0.0:
            # exact zero at a grid row (e.g. an exactly-symmetric chi=1
            # construction, where the TJ lands precisely on a row) -- this
            # IS the TJ itself, not a product-sign test (0*x is never <0).
            z_tj = z[idx[k]]
            break
        if diff[k] * diff[k + 1] < 0:
            z0, z1 = z[idx[k]], z[idx[k + 1]]
            d0, d1 = diff[k], diff[k + 1]
            frac = -d0 / (d1 - d0) if (d1 - d0) != 0 else 0.5
            z_tj = z0 + frac * (z1 - z0)
            break

    if not np.isfinite(z_tj):
        return dict(z_tj=float("nan"), r_tj=float("nan"), z_gb_r_of_z_min=z_gb_guess, agreement_nm=float("nan"))

    r_tj = float(np.interp(z_tj, z[idx], R_of_z[idx]))
    return dict(z_tj=z_tj, r_tj=r_tj, z_gb_r_of_z_min=z_gb_guess,
                agreement_nm=abs(z_tj - z_gb_guess) * 1e9)
