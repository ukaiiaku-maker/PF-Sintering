"""Milestone 15D Section 6: bounded M_GB/M_s rate-competition screen.

Extends the qualified G2 ladder (Milestones 15/15B/15C used up to
M_GB=10*M_GB_ref) to a few more extreme, but still bounded, mobility
ratios, to determine whether increasing GB/surface mismatch can drive
L_contact down and sigma_sint_endpoint up toward 75-100 MPa. Not an
unbounded search: M_GB in {10, 30, 100}*M_GB_ref at baseline M_s, plus
M_GB=100*M_GB_ref at M_s/3 and M_s/10 (the most mismatch-favoring
combination among those tested).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, run_trajectory  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402

M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
sample_times = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]

cases = [
    ("ultra30", 30.0, 0.3),
    ("extreme100", 100.0, 0.3),
    ("extreme100_Ms_over_3", 100.0, 0.1),
    ("extreme100_Ms_over_10", 100.0, 0.03),
]

out = {"M_GB_ref": M_GB_ref}
for name, M_GB_scale, ms_scale in cases:
    M_GB = M_GB_scale * M_GB_ref
    result, _ = run_trajectory(2.5, M_GB, ms_scale, 0.25, sample_times, name)
    out[name] = {k: v for k, v in result.items()}
    with open("/tmp/m15d_rate_screen.json", "w") as fh:
        json.dump(out, fh, default=str)

print("wrote /tmp/m15d_rate_screen.json")
