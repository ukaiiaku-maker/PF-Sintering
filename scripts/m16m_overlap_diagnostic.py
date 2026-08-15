"""Milestone 16M Section 15/16: Poisson-overlap kinetic diagnostic.

Pure calculation (no PF solver) using the SAME frozen hazard calibration
as M16L (b=2.5e-10, V0=12.5*b^3, A0=0.8589eV, GS=201.74nm, r0=1e12,
T=1000K) -- NOT tuned to obtain any particular overlap number.

Reports, at sigma in {20,30,40,45,50,60,75,100} MPa:
    Lambda(sigma)        total calibrated birth intensity (events/s)
    mean waiting time    1/Lambda
    tau_event(sigma)     time for one event to traverse b at fixed sigma
    B(sigma)=Lambda*tau_event   the overlap number

Also screens sensitivity to the UNVALIDATED SECONDS_PER_MODEL_TIME
mapping (Section 30): since Lambda and tau_event are computed here in
real SI seconds, B(sigma)=Lambda*tau_event is INDEPENDENT of
SECONDS_PER_MODEL_TIME by construction (both factors are in SI seconds,
and B is dimensionless) -- but the number of PF STEPS per event, and
the number of model-time-units per mean waiting time, DO depend on the
mapping (reported separately) via a representative dt_model, sourced
from the actual M16L first-event qualification run
(runs/m16l_first_event_qualification/meta.json: activation_step=90490,
t=4.4185 -> dt_model ~= 4.883e-5 model-time-units/PF-step, the same
adaptive dt the corrected M16L run settled into near activation).
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, ".")
from pf_sintering.axisym_sink_rbm import HazardParams  # noqa: E402
from pf_sintering.m16m_multisink import overlap_number_table  # noqa: E402

B_BURGERS = 2.5e-10
V0 = 12.5 * B_BURGERS ** 3
A0_EV = 0.8589
A0_J = A0_EV * 1.602176634e-19
GS = 201.74e-9
R0 = 1e12
T = 1000.0

DT_MODEL_REPRESENTATIVE = 4.883e-5  # model-time-units/PF-step, sourced from the M16L run near activation

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16m_overlap_diagnostic")
os.makedirs(OUT, exist_ok=True)


def main():
    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B_BURGERS,
                       D_gb=1e-3 * __import__("math").exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=A0_J, V0=V0, tau_ex0=0.0)

    sigma_list = [20, 30, 40, 45, 50, 60, 75, 100]
    rows = overlap_number_table(sigma_list, hp, seconds_per_model_time=1.0)

    print(f"{'sigma(MPa)':>10}  {'Lambda(1/s)':>13}  {'wait(s)':>10}  {'tau_event(s)':>13}  {'B=Lam*tau':>10}  "
          f"{'steps/event~':>13}  {'wait/dt_model~':>15}")
    for r in rows:
        n_steps_per_event = (r["tau_event_seconds"] / DT_MODEL_REPRESENTATIVE
                              if r["tau_event_seconds"] < float("inf") else float("inf"))
        wait_over_dt_model = (r["mean_wait_seconds"] / DT_MODEL_REPRESENTATIVE
                               if r["mean_wait_seconds"] < float("inf") else float("inf"))
        r["n_PF_steps_per_event_representative"] = n_steps_per_event
        r["mean_wait_over_dt_model_representative"] = wait_over_dt_model
        print(f"{r['sigma_MPa']:>10.0f}  {r['Lambda_per_second']:>13.4e}  {r['mean_wait_seconds']:>10.4e}  "
              f"{r['tau_event_seconds']:>13.4e}  {r['B']:>10.4e}  {n_steps_per_event:>13.2f}  {wait_over_dt_model:>15.2f}")

    # Section 30: B is dimensionless and computed in SI seconds throughout,
    # so it is by construction INDEPENDENT of SECONDS_PER_MODEL_TIME. State
    # this explicitly rather than silently relying on it.
    print("\nSection 30 time-scale caveat: B(sigma)=Lambda(1/s)*tau_event(s) uses only SI-second "
          "quantities and is therefore INDEPENDENT of the unvalidated SECONDS_PER_MODEL_TIME=1.0 "
          "mapping. The 'steps/event' and 'wait/dt_model' columns above DO depend on that mapping "
          "(via dt_model, which is itself a PF numerical-stability artifact, not a physical time) -- "
          "if SECONDS_PER_MODEL_TIME were, say, 0.01 or 100x different from 1.0, dt_model's SI-second "
          "value would scale accordingly and change how many PF steps are needed per event, but NOT "
          "change B(sigma) itself.")

    fieldnames = list(rows[0].keys())
    with open(os.path.join(OUT, "overlap_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nDONE -- wrote {OUT}/overlap_table.csv")


if __name__ == "__main__":
    main()
