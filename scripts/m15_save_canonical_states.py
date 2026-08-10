"""Milestone 15 Section 22: save canonical states for the most informative
(GB-FAST) trajectory -- initial post-transient, onset of measurable stress
buildup, and the strongest pre-recovery (near the L_GB overshoot minimum)
state. No runaway/depinning state occurred (the extended 0.6s GB-FAST run
showed L_GB and sigma_sint_app both beginning to relax back toward the
other cases' level, not diverging), so that fourth canonical state is
intentionally not produced."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m15_gb_surface_rate_competition import M_ETA_HISTORICAL_REF, run_trajectory  # noqa: E402
from pf_sintering.gb_obstacle_energy import m_gb_from_m_eta  # noqa: E402

M_GB_ref = m_gb_from_m_eta(M_ETA_HISTORICAL_REF, 20e-9)
M_GB_fast = 10.0 * M_GB_ref

# dx=2.5nm, dt=1e-5s (confirmed dt-converged): step 1000->t=0.01s (post-transient),
# step 3000->t=0.03s (near the L_GB/sigma_app peak, onset of measurable buildup),
# step 32500->t=0.325s (near the L_GB overshoot minimum from the extended run).
save_steps = {1000, 3000, 32500}
sample_times = [0, 0.01, 0.03, 0.325]

result, profiles = run_trajectory(2.5, M_GB_fast, 0.3, 0.35, sample_times, "fast_canonical",
                                   save_profile_at=save_steps)

os.makedirs("runs/m15_canonical", exist_ok=True)
meta = dict(M_GB=M_GB_fast, M_s=result["M_s"], dt=result["dt"], dx_nm=2.5, gamma_gb=1.0,
            W_nm=20.0, rows=result["rows"])
with open("runs/m15_canonical/meta.json", "w") as fh:
    json.dump(meta, fh, default=str)

labels = {1000: "post_transient_t0.01s", 3000: "stress_onset_t0.03s", 32500: "pre_recovery_t0.325s"}
for step, label in labels.items():
    if step not in profiles:
        print(f"WARNING: profile for step {step} ({label}) was not saved (diagnostics may have failed)")
        continue
    prof = profiles[step]
    out_path = f"runs/m15_canonical/{label}.npz"
    np.savez(out_path, f=prof["f"], e1=prof["e1"], e2=prof["e2"], e3=prof["e3"],
              mu_J_top=json.dumps(prof["mu_J"]["top"]), mu_J_bottom=json.dumps(prof["mu_J"]["bottom"]))
    print(f"saved {out_path}")

print("done")
