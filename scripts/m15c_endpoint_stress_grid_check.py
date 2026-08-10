"""Milestone 15C Section 5: absolute grid convergence of the ENDPOINT
apparent sintering stress (the promoted primary loading observable,
Section 4) at three matched physical states of the GB-FAST trajectory:

    A: t=0.03s  (post-transient stress minimum, Milestone 15B Section 6)
    B: t=0.125s (post-transient rebound peak)
    C: t=0.25s  (late recovery)

at dx=5, 2.5, 1.25nm. Reports F_cap_endpoint, L_contact,
sigma_sint_endpoint, psi, and the raw particle-side tangent vectors
directly -- convergence of the endpoint quantity itself, not inferred
from force_rel_err (which cross-checks against the curvature form, a
different, only diagnostic, construction).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, run_trajectory  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402

M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
M_GB_fast = 10.0 * M_GB_ref
sample_times = [0, 0.03, 0.125, 0.25]

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dx-nm", type=float, required=True)
ap.add_argument("--out", type=str, required=True)
args = ap.parse_args()

result, _ = run_trajectory(args.dx_nm, M_GB_fast, 0.3, 0.25, sample_times, f"fast_dx{args.dx_nm}")

out = {k: v for k, v in result.items()}
with open(args.out, "w") as fh:
    json.dump(out, fh, default=str)
print(f"wrote {args.out}")
