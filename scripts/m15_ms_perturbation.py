"""Milestone 15 Section 16: M_s perturbation at fixed M_GB=GB-FAST.

The primary M_GB ladder (m15_gb_surface_rate_competition.py) showed clear
stress buildup at GB-FAST (M_GB=10*M_GB_ref): sustained elevated
sigma_sint_app and a GB length (L_GB) that initially overshoots downward
past its eventual settling value before slowly recovering (a bounded,
transient coupled-relaxation signature, not runaway -- see the extended
0.6s GB-FAST run). This reruns that exact M_GB at M_s/3, M_s (repeated for
a clean same-script comparison), and 3*M_s, holding M_GB physically fixed,
comparing in the SAME physical time (Section 16's explicit requirement).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, run_trajectory  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402

M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
M_GB_fast = 10.0 * M_GB_ref
sample_times = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]

BASE_SCALE = 0.3
cases = [("Ms_over_3", BASE_SCALE / 3.0), ("Ms_ref", BASE_SCALE), ("Ms_times_3", BASE_SCALE * 3.0)]

out = {"M_GB_ref": M_GB_ref, "M_GB_fast": M_GB_fast}
for name, scale in cases:
    result, _ = run_trajectory(2.5, M_GB_fast, scale, 0.25, sample_times, f"fast_{name}")
    out[name] = {k: v for k, v in result.items()}

with open("/tmp/m15_ms_perturbation.json", "w") as fh:
    json.dump(out, fh, default=str)
print("wrote /tmp/m15_ms_perturbation.json")
