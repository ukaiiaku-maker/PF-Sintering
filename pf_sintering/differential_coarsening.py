"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 7: paired-trajectory (C0/C1) differential-coarsening machinery.

Milestone 6C established that the physical TJ-to-TJ contact widens under the
production operator sequence, but that alone does not show coarsening is the
cause: the same initial (non-equilibrated) geometry can widen from ordinary
capillary relaxation with coarsening completely disabled. This module runs
two trajectories from an IDENTICAL initial state -- C0 (Ostwald/coarsening
off, everything else on) and C1 (normal Ostwald/coarsening on) -- sampled
every step, so the coarsening-induced perturbation can be isolated as
`delta = C1 - C0` rather than inferred from either trajectory in isolation.

Both trajectories use the SAME per-step operator sequence already validated
elsewhere (`operator_ledger.run_ledger_step_v3`,
`rate_competition.run_sinkoff_trajectory`): CH -> mass-preserving projection
-> Ostwald -> structural relaxation -> mass-preserving projection, sink/RBM
always off. `_step_once` here is that same sequence with only the final
state returned (no per-operator stage sampling -- Milestone 6C already
established the operator attribution for the C1-like case; this milestone
needs matched-time END-of-step snapshots for two trajectories, not another
six-stage ledger for both).

`ostwald_substrate` with `coarsening_rate_scale=0` (`tau_ripening = 20/0 =
inf`) is an exact, bit-for-bit no-op: `tr = min(V*dt/inf, .002*V) = 0.0`
exactly, so every downstream quantity derived from `tr` (`rem`, `act`,
`add`) is exactly zero and `e1,e2,f` are returned unchanged. C0 is therefore
genuinely "coarsening off", not "coarsening slow".

`p.dt` does not depend on `coarsening_rate_scale` (see `model.build_params`:
`p.dt = min(p.CFL*p.dx**4/(p.M_f*p.k_f), 1e-5)` and `p.M_f`/`p.k_f` are
functions of `surface_mobility_scale` and geometry only) -- so as long as
C0/C1 share every ModelConfig field except `coarsening_rate_scale`, their
`dt` is identical and stepping both the same number of times is sufficient
for "matched physical time"; `run_paired_trajectory` asserts this rather
than assuming it.
"""

from __future__ import annotations

import math

import numpy as np

from .diagnostics import wall_x0
from .model import Sink, compute_stress, evolve_eta, evolve_f, ostwald_substrate
from .operator_ledger import contact_rich_sample_v3
from .structural_projection import eta_masses, project_eta_mass_preserving
from .tj_force import compute_neck_tj_forces

PAIRED_KEYS = (
    "V2", "V2_f_eta", "x_neck_m", "A_GB_m2",
    "L_contact_TJ_sub", "L_GB_geom_sub", "L_contact_TJ", "L_GB_geom",
    "tj_top_x_sub", "tj_top_y_sub", "tj_bottom_x_sub", "tj_bottom_y_sub",
    "psi_deg_top", "psi_deg_bottom",
    "kappa_top_1pm", "kappa_bottom_1pm",
    "F_TJ_mag_top", "F_TJ_mag_bottom",
    "F_TJ_x_top", "F_TJ_x_bottom", "F_TJ_y_top", "F_TJ_y_bottom",
    "F_gb_normal_top", "F_gb_normal_bottom",
    "F_gb_parallel_top", "F_gb_parallel_bottom",
    "sigma_Pa", "sigma_lt_Pa", "sigma_curv_Pa",
    "E_surf_J", "E_gb_J", "G_interface_J",
)


def full_state_sample(f, e1, e2, e3, s, st, p, step, time_s, v20, wall_x0_val):
    """contact_rich_sample_v3() plus the F_TJ y-components it doesn't already
    carry (F_TJ_x/_mag/_normal/_parallel are already surfaced by
    rate_competition.rich_sample; F_TJ_y is not, so it is added here without
    modifying that module). Recomputes compute_neck_tj_forces a second time
    to get the raw TJForce objects -- acceptable diagnostic-only overhead,
    not the per-step cost bottleneck (CH stepping and the sub-grid contour
    tracing dominate)."""
    row = contact_rich_sample_v3(f, e1, e2, e3, s, st, p, step, time_s, v20, wall_x0_val)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    for name, tj in (("top", rep.top), ("bottom", rep.bottom)):
        row[f"F_TJ_y_{name}"] = float(tj.F_TJ[1]) if (tj is not None and tj.resolved) else math.nan
    return row


def _step_once(f, e1, e2, e3, s, p, ostwald_fn=ostwald_substrate):
    """One full production-ordering physical step (CH -> mass-preserving
    projection -> Ostwald -> structural relaxation -> mass-preserving
    projection), sink/RBM off. Identical operator sequence and order to
    operator_ledger.run_ledger_step_v3 / rate_competition.run_sinkoff_trajectory.

    `ostwald_fn` defaults to the production `ostwald_substrate`; Milestone 8's
    O0/O1/O2 mechanism-isolation study (ostwald_diagnostics.py) passes
    `ostwald_removal_only` / `ostwald_addition_only` instead -- diagnostic
    substitution only, the default preserves production behavior exactly."""
    ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f = evolve_f(f, e1, e2, e3, s, Sink(), p)
    e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)

    step_out = ostwald_fn(f, e1, e2, e3, p)
    f, e1, e2, e3 = step_out[0], step_out[1], step_out[2], step_out[3]

    ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
    e1, e2, e3 = evolve_eta(e1, e2, e3, p)
    e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)
    return f, e1, e2, e3


def run_single_trajectory(p, f0, e1_0, e2_0, e3_0, n_steps, ostwald_fn=ostwald_substrate):
    """Run one trajectory (any coarsening_rate_scale) for exactly n_steps,
    sampled every step (including step 0, the given initial state). Used
    both for the compact coarsening-rate series (Section 9) and for
    timestep-sensitivity checks (Section 13). `ostwald_fn` -- see
    `_step_once`; default preserves production behavior exactly."""
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)
    wall_x0_val = wall_x0(p)
    v20 = float(e2.sum() * p.dx * p.dx)

    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    rows = [full_state_sample(f, e1, e2, e3, s, st, p, 0, 0.0, v20, wall_x0_val)]
    if stop:
        return rows, True, reason

    for step in range(1, n_steps + 1):
        f, e1, e2, e3 = _step_once(f, e1, e2, e3, s, p, ostwald_fn=ostwald_fn)
        st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
        rows.append(full_state_sample(f, e1, e2, e3, s, st, p, step, step * p.dt, v20, wall_x0_val))
        if stop:
            return rows, True, reason
    return rows, False, ""


def run_paired_trajectory(p_c0, p_c1, f0, e1_0, e2_0, e3_0, target_dv2_frac, max_steps):
    """Run C0 (p_c0, coarsening off) and C1 (p_c1, coarsening on) from
    IDENTICAL copies of the same initial state, one full production step
    each per iteration, sampled every step (matched physical time by
    construction -- see module docstring). Stops once C1's |V2-V20|/V20
    reaches target_dv2_frac, at max_steps, or if either side's compute_stress
    signals a stop condition (e.g. neck/GB no longer resolved -- reported,
    not silently ignored, since the paired comparison stops being meaningful
    once that happens on either branch).

    Returns (rows_c0, rows_c1, stop_reason) where stop_reason is "" for a
    clean target/max_steps stop."""
    if p_c0.dt != p_c1.dt:
        raise ValueError(f"C0/C1 dt mismatch ({p_c0.dt!r} vs {p_c1.dt!r}): "
                          "paired trajectories require identical dt for matched physical time")

    f_c0, e1_c0, e2_c0, e3_c0 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    f_c1, e1_c1, e2_c1, e3_c1 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s0 = Sink(threshold=math.inf)
    s1 = Sink(threshold=math.inf)
    wall_x0_val = wall_x0(p_c0)  # identical geometry for C0/C1 -> identical constant

    v20 = float(e2_0.sum() * p_c0.dx * p_c0.dx)

    st0, stop0, r0 = compute_stress(f_c0, e1_c0, e2_c0, e3_c0, s0, p_c0)
    st1, stop1, r1 = compute_stress(f_c1, e1_c1, e2_c1, e3_c1, s1, p_c1)
    rows_c0 = [full_state_sample(f_c0, e1_c0, e2_c0, e3_c0, s0, st0, p_c0, 0, 0.0, v20, wall_x0_val)]
    rows_c1 = [full_state_sample(f_c1, e1_c1, e2_c1, e3_c1, s1, st1, p_c1, 0, 0.0, v20, wall_x0_val)]
    if stop0 or stop1:
        return rows_c0, rows_c1, (r0 or r1)

    step = 0
    while step < max_steps:
        step += 1
        time_s = step * p_c1.dt
        f_c0, e1_c0, e2_c0, e3_c0 = _step_once(f_c0, e1_c0, e2_c0, e3_c0, s0, p_c0)
        f_c1, e1_c1, e2_c1, e3_c1 = _step_once(f_c1, e1_c1, e2_c1, e3_c1, s1, p_c1)

        st0, stop0, r0 = compute_stress(f_c0, e1_c0, e2_c0, e3_c0, s0, p_c0)
        st1, stop1, r1 = compute_stress(f_c1, e1_c1, e2_c1, e3_c1, s1, p_c1)
        rows_c0.append(full_state_sample(f_c0, e1_c0, e2_c0, e3_c0, s0, st0, p_c0, step, time_s, v20, wall_x0_val))
        rows_c1.append(full_state_sample(f_c1, e1_c1, e2_c1, e3_c1, s1, st1, p_c1, step, time_s, v20, wall_x0_val))

        if stop0 or stop1:
            return rows_c0, rows_c1, (r0 or r1)

        v2_c1 = float(e2_c1.sum() * p_c1.dx * p_c1.dx)
        if abs(v2_c1 - v20) / v20 >= target_dv2_frac:
            break
    return rows_c0, rows_c1, ""


def paired_delta(row_c0, row_c1, keys=PAIRED_KEYS):
    """C1 - C0 for one matched-time pair of full_state_sample() rows."""
    out = {}
    for k in keys:
        a, b = row_c0.get(k), row_c1.get(k)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            out[k] = math.nan
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            out[k] = math.nan
            continue
        out[k] = b - a
    return out


def paired_delta_series(rows_c0, rows_c1, keys=PAIRED_KEYS):
    """delta(t) for every matched-time pair (rows_c0[i], rows_c1[i])."""
    n = min(len(rows_c0), len(rows_c1))
    return [paired_delta(rows_c0[i], rows_c1[i], keys=keys) for i in range(n)]
