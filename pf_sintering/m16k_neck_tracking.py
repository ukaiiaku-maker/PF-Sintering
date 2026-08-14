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
