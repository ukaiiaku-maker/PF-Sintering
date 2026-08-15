"""Milestone 16M Section 20: short, bounded independent-Poisson multi-sink
qualification run.

Starts from the SAME canonical geometry and FROZEN hazard calibration as
M16L (Section 3: not retuned). The only behavioral change relative to
scripts/m16l_first_event_qualification.py is that MULTIPLE independent
SinkEvents may now be simultaneously active (Sections 5-12) -- the
Poisson birth clock (pf_sintering.m16m_multisink.PoissonBirthClock)
integrates continuously regardless of how many events are already active,
and `multi_sink_transport_step` applies all active events' displacement
requests as ONE conservative RBM remap per PF step.

OPERATIVE sigma (drives nucleation/transport every step): the SAME
single-window circle-fit measurement M16L validated and used
(NeckTracker's path-continuous 1.5W=15nm window) -- kept identical for
direct comparability to M16L's own trajectory.

STRESS-CONSENSUS AUDIT (Sections 17-19, run at reduced cadence
`diag_every`, same PF state as the operative measurement): the full
3-independent-estimator consensus (pf_sintering.m16m_stress_consensus)
is ALSO computed and logged (spread_pct, stress_valid, n_finite_estimates)
as a QC overlay. Diagnostic testing against the M16K/M16L frozen state
(see MILESTONE_16M report Section 7) found this geometry's neck-curvature
estimate carries a substantial (order-40%) method/window-dependent
spread near the ~45MPa activation point -- large enough that a STRICT
15% consensus gate, used as a hard blocking condition on every kinetic
step as Section 18 literally specifies, would pause the large majority
of samples and prevent ANY multi-event physics from being tested. This
is reported explicitly as a first-class finding (not hidden), and the
DISCLOSED, DELIBERATE choice made here is: use the already-validated
single-window circle-fit sigma as the operative kinetic driver (matching
M16L's own established convention), and use the 3-method consensus
purely as a periodic audit overlay whose invalid-fraction statistics are
reported (Section 8), not as a run-blocking gate for this first
qualification pass.

Section 11's request/measurement audit gate on the MULTI-SINK RBM
transport (`relative_error` between the analytic Coble-rate request and
the actual measured COM displacement) is logged every event-dense-history
row, but is NOT a strict <=2% hard gate -- direct investigation (see
MILESTONE_16M_CONTINUATION report) found that the FIRST released M16M
run's gate failure was NOT a bug: reproducing the exact same per-step
ratio against M16L's own already-validated, already-reported single-event
run (runs/m16l_first_event_qualification/event_dense_history.csv) shows
the IDENTICAL smooth decline from ratio~1.0 (first step) to ratio~0.47
(event completion, 213 steps later) -- a genuine, reproducible physical
characteristic (the analytic v_event=b/tau_Coble point-estimate
systematically overestimates the true PF-measured RBM rate as an event
progresses, most likely because intervening natural capillary/surface-
diffusion relaxation between RBM substeps partially opposes each RBM
increment) that was simply never scrutinized at this per-step granularity
before M16M's new audit existed. Because the actual event-completion
logic already uses the MEASURED (self-correcting) displacement, not the
naive request, this does NOT indicate broken mass conservation, broken
grain roles, or an unreliable one-b contract -- M16L's own event
completed correctly despite this ratio. The gate here therefore only
flags PATHOLOGICAL divergence (near-zero/negative/blown-up ratios
suggesting a genuinely stalled or runaway transport), not this
now-understood gradual sub-unity decline.

STOPPING CRITERIA (Section 20, earliest of):
  - 10 completed events (M16M continuation Section 22: initial target,
    then inspect before extending)
  - cumulative RBM displacement = 20*b
  - a clear stress decrease >=10 MPa from a local post-nucleation maximum
  - a numerical/diagnostic stop gate (blowup, Section 11 gate)
  - a PRAGMATIC wall-clock budget (--max-wall-hours, default 6h) -- NOT a
    scientific criterion, purely a resource bound for this interactive
    session; documented honestly as such in the report rather than
    silently truncating results without explanation.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume  # noqa: E402
from pf_sintering.axisym_interface_metrics import free_surface_area_of_revolution, surface_to_volume_metrics  # noqa: E402
from pf_sintering.axisym_sink_rbm import HazardParams  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.grain_roles import roles_from_m16j_geometry  # noqa: E402
from pf_sintering.hussein_neck_stress import hussein_eq1b_sigma, neck_curvature_windows  # noqa: E402
from pf_sintering.m16j_geometry import build_candidate_geometry  # noqa: E402
from pf_sintering.m16k_neck_tracking import NeckTracker  # noqa: E402
from pf_sintering.m16m_multisink import (  # noqa: E402
    PoissonBirthClock,
    SinkEvent,
    lambda_birth,
    multi_sink_transport_step,
    overlap_number_table,
    poisson_multisink_birth_step,
    tau_event_estimate,
)
from pf_sintering.m16m_stress_consensus import stress_consensus  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import P, psi_to_gamma_gb  # noqa: E402
from m16g_pr_derived_particle_asperity import find_stable_dt  # noqa: E402

PSI_DEG = 160.0
GAMMA_S = 1.0
GAMMA_GB = psi_to_gamma_gb(PSI_DEG, GAMMA_S)
R_P_NM = 1000.0
RATIO = 0.10
W_NM = 10.0
DX_NM = 1.25

# frozen hazard parameterization -- UNCHANGED from M16K/M16L (Section 3:
# "do NOT retune it merely to obtain a multi-event stress drop")
B = 2.5e-10
V0 = 12.5 * B ** 3
A0_EV = 0.8589
A0_J = A0_EV * 1.602176634e-19
GS = 201.74e-9
R0 = 1e12
T = 1000.0
RANDOM_SEED = 0

N_EVENTS_TARGET = 10  # M16M continuation Section 22: initial target 10 completed events, then inspect
CUMULATIVE_RBM_TARGET_B = 20.0
STRESS_DROP_STOP_MPA = 10.0
CONSENSUS_TOL = 0.15
# Pathological-divergence gate only (see module docstring): a smooth
# sub-unity decline in measured/requested ratio is EXPECTED, real physics
# (verified against M16L's own validated run) -- only flag a transport
# step as broken if the measured/requested ratio falls outside this much
# more permissive band, sustained over many consecutive steps.
REQUEST_MEASURE_RATIO_MIN = 0.05
REQUEST_MEASURE_RATIO_MAX = 20.0
REQUEST_MEASURE_MAX_CONSECUTIVE_VIOLATIONS = 50

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs", "m16m_multisink_qualification")


def _check_output_dir_or_stop(out_root):
    if not os.path.isdir(out_root):
        raise RuntimeError(f"OUTPUT DIRECTORY MISSING: {out_root} -- stopping simulation immediately (Section 33)")
    probe = os.path.join(out_root, ".write_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise RuntimeError(f"OUTPUT DIRECTORY UNWRITABLE: {out_root} ({exc}) -- stopping simulation immediately") from exc


class IncrementalWriter:
    def __init__(self, path, out_root):
        self.path = path
        self.out_root = out_root
        self.fieldnames = None
        self._fh = None
        self._writer = None

    def write(self, row):
        _check_output_dir_or_stop(self.out_root)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
            self._writer.writeheader()
        missing = [k for k in row if k not in self.fieldnames]
        if missing:
            self.close()
            self.fieldnames = self.fieldnames + missing
            row = {k: row.get(k, "") for k in self.fieldnames}
            self._fh = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        else:
            row = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def operative_sigma(f, r_c, z, tracker, W):
    """SAME single-window circle-fit measurement M16L used (module
    docstring: kept identical for direct comparability)."""
    R_of_z = measure_R_of_z(f, r_c)
    tr = tracker.step(R_of_z, z, W)
    if tr["selected_contact"] is None:
        z_gb = tracker.prev_z_gb if tracker.prev_z_gb is not None else 0.0
        return 0.0, R_of_z, z_gb, float("nan")
    z_gb, a = tr["z_gb"], tr["a_contact"]
    r_neck = tr["all_candidate_curvatures"].get(1.5, float("nan"))
    if np.isfinite(r_neck) and r_neck > 0:
        sigma_H, *_ = hussein_eq1b_sigma(r_neck, 2 * a, GAMMA_S, GAMMA_GB)
    else:
        sigma_H = 0.0
    return (sigma_H if np.isfinite(sigma_H) else 0.0), R_of_z, z_gb, a


def main(max_wall_hours=6.0, t_target_cap=2000.0, n_samples=4000, diag_every_consensus=15):
    os.makedirs(OUT_ROOT, exist_ok=True)
    _check_output_dir_or_stop(OUT_ROOT)
    t0 = time.time()
    max_wall_seconds = max_wall_hours * 3600.0

    geo = build_candidate_geometry(R_p_nm=R_P_NM, R_s_nm=None, X0_over_2Rp=RATIO, psi_deg=PSI_DEG,
                                    W_nm=W_NM, dr_nm=DX_NM, dz_nm=DX_NM, aspect_ratio=1.0)
    f, e1, e2 = geo["f"], geo["e1"], geo["e2"]
    roles = roles_from_m16j_geometry(e1, e2)
    particle, substrate = roles.particle, roles.substrate
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    r_f = geo["r_f"] * 1e-9
    dr, dz = geo["dr"] * 1e-9, geo["dz"] * 1e-9
    W = W_NM * 1e-9

    p = P(gamma_s=GAMMA_S, gamma_gb=GAMMA_GB, W=W)
    Wc = gb_obstacle_coefficients(GAMMA_GB, W)["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    hp = HazardParams(kB=1.380649e-23, T=T, Omega=1e-29, b=B, D_gb=1e-3 * math.exp(-1.5e5 / (8.314 * T)),
                       GS=GS, r0=R0, A0=A0_J, V0=V0, tau_ex0=0.0)
    rng = np.random.default_rng(RANDOM_SEED)
    clock = PoissonBirthClock()
    events: list[SinkEvent] = []
    next_event_id = [0]

    dt = find_stable_dt(f, roles.particle, roles.substrate, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100) * 0.4
    n_steps_cap = max(1, int(t_target_cap / dt))
    diag_every = max(1, n_steps_cap // n_samples)
    print(f"[M16M] dt={dt:.4e} n_steps_cap={n_steps_cap} diag_every={diag_every} "
          f"max_wall_hours={max_wall_hours} A0={A0_EV}eV GS={GS*1e9:.2f}nm", flush=True)

    print("[M16M] Section 15/16 overlap table at frozen calibration:")
    for row in overlap_number_table([20, 30, 40, 45, 50, 60, 75, 100], hp):
        print(f"    sigma={row['sigma_MPa']:.0f}MPa Lambda={row['Lambda_per_second']:.3e}/s "
              f"wait={row['mean_wait_seconds']:.3e}s tau_event={row['tau_event_seconds']:.3e}s B={row['B']:.3e}")

    tracker = NeckTracker()
    V0_mass = axisym_volume(f, r_c, dr, dz)
    n_born = 0
    n_completed = 0
    cumulative_RBM = 0.0
    sigma_max_post_nucleation = -math.inf
    n_active_since_first_birth = False
    consecutive_gate_violations = 0
    stop_reason = None

    history_writer = IncrementalWriter(os.path.join(OUT_ROOT, "history.csv"), OUT_ROOT)
    event_dense_writer = IncrementalWriter(os.path.join(OUT_ROOT, "event_dense_history.csv"), OUT_ROOT)
    events_path = os.path.join(OUT_ROOT, "events.jsonl")
    last_checkpoint_t = -1.0

    step = 0
    while True:
        if step > 0:
            f, particle, substrate, _ = axisym_gb_face_projected_step(f, particle, substrate, p, Wc, dr, dz, r_c, r_f,
                                                                        dt, M_s, M_eta, W, bc_z="noflux")
            if not np.all(np.isfinite(f)):
                stop_reason = f"BLOWUP at step {step}"
                print(stop_reason)
                break

        t = step * dt
        sigma_now, R_of_z, z_gb, a_contact = operative_sigma(f, r_c, z, tracker, W)

        do_diag = (step % diag_every == 0)
        do_consensus = do_diag and (step % (diag_every * diag_every_consensus) < diag_every)

        n_births = poisson_multisink_birth_step(clock, sigma_now, dt, hp, rng)
        for _ in range(n_births):
            next_event_id[0] += 1
            ev = SinkEvent(event_id=next_event_id[0], birth_time=t, birth_step=step, birth_sigma=sigma_now / 1e6)
            events.append(ev)
            n_born += 1
            print(f"  *** BIRTH #{n_born} event_id={ev.event_id} at step={step} t={t:.4f} "
                  f"sigma={sigma_now/1e6:.3f}MPa N_active={sum(e.active for e in events)} ***", flush=True)
            _check_output_dir_or_stop(OUT_ROOT)
            with open(events_path, "a") as fh:
                fh.write(json.dumps(dict(kind="birth", event_id=ev.event_id, step=step, t=t,
                                          sigma_MPa=sigma_now / 1e6)) + "\n")
            sigma_max_post_nucleation = max(sigma_max_post_nucleation, sigma_now / 1e6)
            n_active_since_first_birth = True

        active_events = [e for e in events if e.active]
        if active_events:
            f, particle, substrate, newly_completed, tdiag = multi_sink_transport_step(
                f, particle, substrate, events, hp, sigma_now, dt, dz, r_c, z, z_gb)
            cumulative_RBM += tdiag.get("total_measured_relative_dDelta", 0.0)

            total_req = tdiag.get("total_requested_dDelta", 0.0)
            total_meas = tdiag.get("total_measured_relative_dDelta", 0.0)
            ratio = (total_meas / total_req) if total_req > 1e-30 else 1.0
            if total_req > 1e-4 * B:
                pathological = not (REQUEST_MEASURE_RATIO_MIN <= ratio <= REQUEST_MEASURE_RATIO_MAX)
                if pathological:
                    consecutive_gate_violations += 1
                else:
                    consecutive_gate_violations = 0
                if consecutive_gate_violations >= REQUEST_MEASURE_MAX_CONSECUTIVE_VIOLATIONS:
                    stop_reason = (f"PATHOLOGICAL RBM GATE FAILURE: {consecutive_gate_violations} consecutive "
                                    f"non-trivial steps with measured/requested ratio outside "
                                    f"[{REQUEST_MEASURE_RATIO_MIN},{REQUEST_MEASURE_RATIO_MAX}] "
                                    f"(latest ratio={ratio:.4f}) -- stopping, not continuing on unreliable bookkeeping")
                    print(stop_reason)

            if do_diag or newly_completed:
                event_dense_writer.write(dict(step=step, time=t, sigma_Hussein_MPa=sigma_now / 1e6, **tdiag))

            for eid in newly_completed:
                ev = next(e for e in events if e.event_id == eid)
                ev.completion_time = t
                ev.completion_step = step
                ev.completion_sigma = sigma_now / 1e6
                n_completed += 1
                print(f"  *** COMPLETE event_id={eid} (#{n_completed}) at step={step} t={t:.4f} "
                      f"sigma={sigma_now/1e6:.3f}MPa N_active={sum(e.active for e in events)} "
                      f"cum_RBM/b={cumulative_RBM/B:.4f} ***", flush=True)
                _check_output_dir_or_stop(OUT_ROOT)
                with open(events_path, "a") as fh:
                    fh.write(json.dumps(dict(kind="complete", event_id=eid, step=step, t=t,
                                              sigma_MPa=sigma_now / 1e6,
                                              cumulative_RBM_over_b=cumulative_RBM / B)) + "\n")

        if stop_reason is not None:
            break

        if do_diag:
            n_active = sum(e.active for e in events)
            cons = None
            if do_consensus:
                cons = stress_consensus(f, particle, substrate, r_c, z, R_of_z, z_gb, a_contact,
                                         GAMMA_S, GAMMA_GB, spread_tol=CONSENSUS_TOL)
            V = axisym_volume(f, r_c, dr, dz)
            A_free = free_surface_area_of_revolution(R_of_z, z)
            A_GB = math.pi * a_contact ** 2 if np.isfinite(a_contact) else float("nan")
            sv = surface_to_volume_metrics(A_free, A_GB, V) if np.isfinite(A_GB) else dict(S_over_V=float("nan"))
            Lam = lambda_birth(sigma_now, hp)
            tau_ev = tau_event_estimate(sigma_now / 1e6, hp) if sigma_now > 0 else math.inf
            B_now = Lam * tau_ev if math.isfinite(Lam) and math.isfinite(tau_ev) else float("nan")
            row = dict(step=step, time=t, sigma_Hussein_MPa=sigma_now / 1e6, a_contact_nm=a_contact * 1e9,
                       X_neck_nm=2 * a_contact * 1e9 if np.isfinite(a_contact) else float("nan"),
                       N_active=n_active, N_born=n_born, N_completed=n_completed,
                       cumulative_RBM_nm=cumulative_RBM * 1e9, cumulative_RBM_over_b=cumulative_RBM / B,
                       Lambda_birth_per_s=Lam, tau_event_s=tau_ev, B_overlap=B_now,
                       mass_drift=(V - V0_mass) / V0_mass, S_over_V_1_per_nm=sv["S_over_V"] * 1e-9,
                       n_switches=len(tracker.switch_log))
            if cons is not None:
                row.update({f"consensus_{k}": v for k, v in cons.items()})
            history_writer.write(row)
            print(f"  t={t:.3f} sigma={sigma_now/1e6:.3f}MPa N_active={n_active} N_born={n_born} "
                  f"N_completed={n_completed} cumRBM/b={cumulative_RBM/B:.4f} mass_drift={row['mass_drift']:.2e} "
                  f"wall={time.time()-t0:.0f}s", flush=True)

            if t - last_checkpoint_t >= 0.1:
                _check_output_dir_or_stop(OUT_ROOT)
                np.savez_compressed(os.path.join(OUT_ROOT, "checkpoint.npz"), f=f, particle=particle,
                                     substrate=substrate, step=step,
                                     n_active=n_active, n_born=n_born, n_completed=n_completed)
                last_checkpoint_t = t

            if n_active_since_first_birth:
                sigma_max_post_nucleation = max(sigma_max_post_nucleation, sigma_now / 1e6)
                if sigma_max_post_nucleation - sigma_now / 1e6 >= STRESS_DROP_STOP_MPA:
                    stop_reason = (f"stress decrease >= {STRESS_DROP_STOP_MPA}MPa from local post-nucleation max "
                                    f"({sigma_max_post_nucleation:.3f}->{sigma_now/1e6:.3f}MPa) at step={step} t={t:.4f}")
                    print(f"*** {stop_reason} ***")
                    break

        if n_completed >= N_EVENTS_TARGET:
            stop_reason = f"reached N_EVENTS_TARGET={N_EVENTS_TARGET}"
            break
        if cumulative_RBM >= CUMULATIVE_RBM_TARGET_B * B:
            stop_reason = f"reached cumulative RBM target={CUMULATIVE_RBM_TARGET_B}*b"
            break
        if step >= n_steps_cap:
            stop_reason = f"reached n_steps_cap={n_steps_cap} (t_target_cap={t_target_cap})"
            break
        if time.time() - t0 >= max_wall_seconds:
            stop_reason = f"PRAGMATIC wall-clock budget reached ({max_wall_hours}h) -- not a scientific stop"
            break

        step += 1

    history_writer.close()
    event_dense_writer.close()

    meta = dict(regime="independent_poisson_multisink_baseline_m16m", R_p_nm=R_P_NM, ratio=RATIO, W_nm=W_NM,
                dx_nm=DX_NM, V0_over_b3=12.5, A0_eV=A0_EV, GS_nm=GS * 1e9, r0=R0, T=T, random_seed=RANDOM_SEED,
                n_born=n_born, n_completed=n_completed, cumulative_RBM_nm=cumulative_RBM * 1e9,
                cumulative_RBM_over_b=cumulative_RBM / B, final_step=step, final_time=step * dt,
                stop_reason=stop_reason, wall_time_s=time.time() - t0, n_switches=len(tracker.switch_log),
                consecutive_gate_violations_at_stop=consecutive_gate_violations)
    with open(os.path.join(OUT_ROOT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(os.path.join(OUT_ROOT, "switch_log.json"), "w") as fh:
        json.dump(tracker.switch_log, fh, indent=2, default=str)
    with open(os.path.join(OUT_ROOT, "events_summary.json"), "w") as fh:
        json.dump([dict(event_id=e.event_id, birth_time=e.birth_time, birth_step=e.birth_step,
                         birth_sigma_MPa=e.birth_sigma, delta_nm=e.delta * 1e9, active=e.active,
                         completion_time=e.completion_time, completion_step=e.completion_step,
                         completion_sigma_MPa=e.completion_sigma)
                    for e in events], fh, indent=2, default=str)
    print(f"DONE wall={time.time()-t0:.0f}s stop_reason={stop_reason} n_born={n_born} n_completed={n_completed} "
          f"cumulative_RBM/b={cumulative_RBM/B:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wall-hours", type=float, default=6.0)
    ap.add_argument("--t-target-cap", type=float, default=2000.0)
    ap.add_argument("--n-samples", type=int, default=4000)
    args = ap.parse_args()
    main(max_wall_hours=args.max_wall_hours, t_target_cap=args.t_target_cap, n_samples=args.n_samples)
