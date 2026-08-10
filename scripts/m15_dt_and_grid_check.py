"""Milestone 15 Section 18 (dt-halved check) and Section 19 (grid check) for
the most informative condition, GB-FAST (M_GB=10*M_GB_ref) at the baseline
M_s (surface_mobility_scale=0.3)."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, run_trajectory  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402

M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
M_GB_fast = 10.0 * M_GB_ref
sample_times = [0, 0.005, 0.01, 0.02, 0.03, 0.05]

out = {"M_GB_ref": M_GB_ref, "M_GB_fast": M_GB_fast}

# --- Section 18: dt-halved check, short horizon, dx=2.5nm ---
result_dt1, _ = run_trajectory(2.5, M_GB_fast, 0.3, 0.05, sample_times, "dt_baseline")
dt_half = result_dt1["dt"] / 2.0
result_dt2, _ = run_trajectory(2.5, M_GB_fast, 0.3, 0.05, sample_times, "dt_halved", dt_override=dt_half)
out["dt_baseline"] = dict(result_dt1)
out["dt_halved"] = dict(result_dt2)

# --- Section 19: grid check, dx=5nm (full horizon) ---
result_dx5, _ = run_trajectory(5.0, M_GB_fast, 0.3, 0.25,
                                [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
                                "dx5")
out["dx5"] = dict(result_dx5)

# --- Section 19: grid check, dx=1.25nm (bounded horizon) ---
result_dx125, _ = run_trajectory(1.25, M_GB_fast, 0.3, 0.05, sample_times, "dx1p25")
out["dx1p25"] = dict(result_dx125)

with open("/tmp/m15_dt_grid_check.json", "w") as fh:
    json.dump(out, fh, default=str)
print("wrote /tmp/m15_dt_grid_check.json")
