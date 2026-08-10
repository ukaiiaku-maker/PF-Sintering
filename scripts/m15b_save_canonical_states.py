"""Milestone 15B Section 12: save canonical states for the corrected
GB-FAST trajectory, labeled by the ACTUAL loading behavior (Section 6's
post-transient sigma_sint_app(t), not by GB-length peak timing):

    A_post_transient        t=0.005s (initial ownership-relaxation settled)
    B_stress_minimum        t=0.03s  (sigma_sint_app's post-transient minimum)
    C_post_transient_peak   t=0.125s (sigma_sint_app's post-transient rebound peak)
    D_late_recovery         t=0.25s  (partial relaxation from the peak)

determined from the Milestone 15B primary requalification campaign's own
sigma_sint_app(t) trajectory (/tmp/m15b_primary_campaign.json), not
hard-coded from Milestone 15's (pre-fix) GB-length-peak-based guess.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, gb_excess_energy, run_trajectory  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402
from pf_sintering.model import Sink  # noqa: E402
from pf_sintering.tj_subgrid import compute_subgrid_contact  # noqa: E402
import math  # noqa: E402

M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
M_GB_fast = 10.0 * M_GB_ref

# dx=2.5nm, dt=1e-5s (confirmed dt-converged Section 18/Milestone 15):
# step 500->t=0.005s, 3000->t=0.03s, 12500->t=0.125s, 25000->t=0.25s.
save_steps = {500, 3000, 12500, 25000}
sample_times = [0, 0.005, 0.03, 0.125, 0.25]

result, profiles = run_trajectory(2.5, M_GB_fast, 0.3, 0.25, sample_times, "fast_canonical_15b",
                                   save_profile_at=save_steps)

os.makedirs("runs/m15b_canonical", exist_ok=True)
meta = dict(M_GB=M_GB_fast, M_s=result["M_s"], dt=result["dt"], dx_nm=2.5, gamma_gb=1.0,
            W_nm=20.0, rows=result["rows"])
with open("runs/m15b_canonical/meta.json", "w") as fh:
    json.dump(meta, fh, default=str)

labels = {500: "A_post_transient_t0.005s", 3000: "B_stress_minimum_t0.03s",
          12500: "C_post_transient_peak_t0.125s", 25000: "D_late_recovery_t0.25s"}
s = Sink(threshold=math.inf)
p = None
for step, label in labels.items():
    if step not in profiles:
        print(f"WARNING: profile for step {step} ({label}) was not saved")
        continue
    prof = profiles[step]
    f, e1, e2, e3 = prof["f"], prof["e1"], prof["e2"], prof["e3"]
    out_path = f"runs/m15b_canonical/{label}.npz"
    np.savez(out_path, f=f, e1=e1, e2=e2, e3=e3,
              mu_J_top=json.dumps(prof["mu_J"]["top"]), mu_J_bottom=json.dumps(prof["mu_J"]["bottom"]))
    print(f"saved {out_path}")

print("done")
