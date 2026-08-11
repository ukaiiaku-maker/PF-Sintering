"""Milestone 16C Sections 10 and 12: unstable PF de-sintering
demonstration with quantified order-one neck recession, and explicit
COM-constraint verification, on the one-particle-between-two-
substrates capped geometry.

Uses the L=140nm gamma_gb=0 sharp-interface relaxed profile (Section
11's stability map: lowest Hessian eigenvalue -1.560, i.e. clearly,
not marginally, unstable -- the sharp-interface relaxation itself
already drove the neck down to ~2nm) directly as the PF initial
condition (no additional eigenmode perturbation needed -- this shape
is already well inside the unstable regime). No sink, no RBM: this PF
code path has never had either. Volume is conserved by construction
(exact no-flux z-boundary, Milestone 16C's `bc_z="noflux"` addition to
axisym.py) -- verified numerically throughout, including through the
topological pinch-off event itself.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

sys.path.insert(0, os.path.dirname(__file__))
from m16c_two_substrate_pf_dynamics import run_case  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16c_campaign")

if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "pinchoff_demo.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            result = json.load(fh)
    else:
        with open(os.path.join(CAMPAIGN_DIR, "particle_between_substrates_map_gamma_gb0.json")) as fh:
            sharp = json.load(fh)
        by_L = {r["L_nm"]: r for r in sharp}
        r_sharp = by_L[140.0]
        R_of_z = np.array(r_sharp["R_profile_nm"]) * 1e-9
        dz_sharp = r_sharp["dz_nm"] * 1e-9
        result = run_case("pinchoff_L140", R_of_z, 140.0e-9, dz_sharp, t_target=0.5, n_sample=15, dr_nm=0.75)
        result["sharp_lowest_eig"] = r_sharp["lowest_eig"]
        result["sharp_neck0_nm"] = r_sharp["neck_a_nm"]
        with open(out_path, "w") as fh:
            json.dump(result, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Sections 10/12 summary ---")
    print(f"sharp-interface lowest_eig={result['sharp_lowest_eig']:.4f} (clearly unstable)")
    pinched = False
    for r in result["rows"]:
        a_over_a0 = r["a_over_a0"]
        status = "PINCHED OFF (neck below resolvable threshold)" if a_over_a0 != a_over_a0 else f"a/a0={a_over_a0:.4f}"
        if a_over_a0 != a_over_a0:
            pinched = True
        print(f"t={r['t']:.4f}  {status}  com_shift={r['com_shift_nm']:.5f}nm  mass_drift={r['mass_drift']:.2e}")
    max_com_shift = max(abs(r["com_shift_nm"]) for r in result["rows"])
    max_mass_drift = max(abs(r["mass_drift"]) for r in result["rows"])
    print(f"\nreached pinch-off: {pinched}")
    print(f"max |COM shift| over full run: {max_com_shift:.5f}nm (no RBM needed)")
    print(f"max |mass drift| over full run (through pinch-off): {max_mass_drift:.2e}")
