"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 8: paired (C1-C0) operator-resolved ledger.

Milestone 6 (`operator_ledger.py`) decomposed ONE trajectory's per-step
change into its five constituent production operators (CH, post-CH
projection, Ostwald, eta_raw, post-eta projection). Milestone 7
(`differential_coarsening.py`) ran paired C0 (coarsening off)/C1
(coarsening on) trajectories and took their whole-step difference
`delta_L_coarsening = L_contact_TJ_sub_C1 - L_contact_TJ_sub_C0`. This
module combines them: for each physical step, run BOTH branches through the
SAME six-stage production sequence (`operator_ledger.run_ledger_step_v3`,
unmodified, called once per branch), take the paired (C1-C0) difference AT
EACH of the six stages, and then apply `operator_ledger._delta`'s own
generic dict-subtraction machinery to that already-paired stage series.

`_delta(a, b, keys)` is `{k: b[k]-a[k] for k in keys}` -- it has no
knowledge of what "a" and "b" mean. Feeding it two PAIRED (C1-C0)
stage-dicts instead of two single-branch stage-dicts therefore computes
`(C1_b[k]-C0_b[k]) - (C1_a[k]-C0_a[k])`, i.e. exactly "how much did the
(C1-C0) difference change across this operator" -- the per-operator
attribution of `d(delta_L_coarsening)` itself, not either branch's absolute
trajectory. Summed over a trajectory this telescopes to the total
`delta_L_coarsening` change exactly as `operator_ledger.summarize_ledger_v3`
already does for a single branch, so the same closure check applies here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .differential_coarsening import PAIRED_KEYS, paired_delta
from .diagnostics import wall_x0
from .model import Sink, compute_stress
from .operator_ledger import OPERATORS, STAGES, _delta, run_ledger_step_v3

# F_TJ_y is not surfaced by contact_rich_sample_v3 (only by
# differential_coarsening.full_state_sample, which run_ledger_step_v3 does
# not use) -- excluded here rather than silently reporting NaN for it.
LEDGER_PAIRED_KEYS = tuple(k for k in PAIRED_KEYS if not k.startswith("F_TJ_y"))


@dataclass
class PairedLedgerStep:
    step: int
    time_s: float
    paired_stage_samples: dict = field(default_factory=dict)   # stage -> {key: C1-C0}
    paired_operator_deltas: dict = field(default_factory=dict)  # operator -> {key: d(C1-C0)}
    stop: bool = False
    stop_reason: str = ""


def run_paired_operator_ledger_step(
    f0, e1_0, e2_0, e3_0, f1, e1_1, e2_1, e3_1, p0, p1, s0, s1, step, time_s, v20, wall_x0_val,
):
    """One physical step for both C0 and C1 branches (each via the unmodified
    operator_ledger.run_ledger_step_v3), plus the paired (C1-C0) stage series
    and its operator-resolved attribution. Returns
    (new_state_c0, new_state_c1, ledger0, ledger1, paired_step)."""
    f0n, e10n, e20n, e30n, ledger0 = run_ledger_step_v3(f0, e1_0, e2_0, e3_0, p0, s0, step, time_s, v20, wall_x0_val)
    f1n, e11n, e21n, e31n, ledger1 = run_ledger_step_v3(f1, e1_1, e2_1, e3_1, p1, s1, step, time_s, v20, wall_x0_val)

    if ledger0.stop or ledger1.stop:
        paired_step = PairedLedgerStep(step=step, time_s=time_s, stop=True,
                                        stop_reason=(ledger0.stop_reason or ledger1.stop_reason))
        return (f0n, e10n, e20n, e30n), (f1n, e11n, e21n, e31n), ledger0, ledger1, paired_step

    paired_stage_samples = {
        stage: paired_delta(ledger0.stage_samples[stage], ledger1.stage_samples[stage], keys=LEDGER_PAIRED_KEYS)
        for stage in STAGES
    }
    paired_operator_deltas = {
        "CH": _delta(paired_stage_samples["start"], paired_stage_samples["CH_raw"], keys=LEDGER_PAIRED_KEYS),
        "post_CH_projection": _delta(paired_stage_samples["CH_raw"], paired_stage_samples["CH_projected"], keys=LEDGER_PAIRED_KEYS),
        "Ostwald": _delta(paired_stage_samples["CH_projected"], paired_stage_samples["ostwald"], keys=LEDGER_PAIRED_KEYS),
        "eta_raw": _delta(paired_stage_samples["ostwald"], paired_stage_samples["eta_raw"], keys=LEDGER_PAIRED_KEYS),
        "post_eta_projection": _delta(paired_stage_samples["eta_raw"], paired_stage_samples["end"], keys=LEDGER_PAIRED_KEYS),
    }
    paired_step = PairedLedgerStep(step=step, time_s=time_s, paired_stage_samples=paired_stage_samples,
                                    paired_operator_deltas=paired_operator_deltas)
    return (f0n, e10n, e20n, e30n), (f1n, e11n, e21n, e31n), ledger0, ledger1, paired_step


def run_paired_operator_ledger_trajectory(p0, p1, f0, e1_0, e2_0, e3_0, target_dv2_frac, max_steps):
    """Runs both branches from identical copies of one initial state (never
    initialized separately), stops once C1's |V2-V20|/V20 reaches
    target_dv2_frac or at max_steps. Returns (paired_steps, ledgers0, ledgers1)."""
    if p0.dt != p1.dt:
        raise ValueError(f"C0/C1 dt mismatch ({p0.dt!r} vs {p1.dt!r})")

    fc0, e1c0, e2c0, e3c0 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    fc1, e1c1, e2c1, e3c1 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s0 = Sink(threshold=math.inf)
    s1 = Sink(threshold=math.inf)
    wall_x0_val = wall_x0(p0)
    v20 = float(e2_0.sum() * p0.dx * p0.dx)

    paired_steps, ledgers0, ledgers1 = [], [], []
    step = 0
    while step < max_steps:
        step += 1
        (fc0, e1c0, e2c0, e3c0), (fc1, e1c1, e2c1, e3c1), ledger0, ledger1, paired_step = \
            run_paired_operator_ledger_step(
                fc0, e1c0, e2c0, e3c0, fc1, e1c1, e2c1, e3c1, p0, p1, s0, s1,
                step, step * p1.dt, v20, wall_x0_val,
            )
        paired_steps.append(paired_step)
        ledgers0.append(ledger0)
        ledgers1.append(ledger1)
        if paired_step.stop:
            break
        v2_c1 = float(e2c1.sum() * p1.dx * p1.dx)
        if abs(v2_c1 - v20) / v20 >= target_dv2_frac:
            break
    return paired_steps, ledgers0, ledgers1


def summarize_paired_ledger(paired_steps, keys=LEDGER_PAIRED_KEYS):
    """Per-operator cumulative totals of d(delta_L_coarsening) [and every
    other paired key], the total observed paired-delta change (first step's
    paired 'start' to last step's paired 'end'), and the closure error
    between the two -- same telescoping-sum pattern as
    operator_ledger.summarize_ledger_v3, applied to the paired series."""
    resolved = [s for s in paired_steps if not s.stop]
    if not resolved:
        raise ValueError("no resolved paired steps in trajectory")

    totals = {op: {k: 0.0 for k in keys} for op in OPERATORS}
    for step in resolved:
        for op in OPERATORS:
            d = step.paired_operator_deltas[op]
            for k in keys:
                if d[k] is not None and math.isfinite(d[k]):
                    totals[op][k] += d[k]

    first = resolved[0].paired_stage_samples["start"]
    last = resolved[-1].paired_stage_samples["end"]
    total_observed = {k: (last[k] - first[k]) if (first.get(k) is not None and last.get(k) is not None
                          and math.isfinite(first[k]) and math.isfinite(last[k])) else None for k in keys}
    sum_reconstructed = {k: sum(totals[op][k] for op in OPERATORS) for k in keys}
    closure_error = {
        k: (sum_reconstructed[k] - total_observed[k]) if total_observed[k] is not None else None
        for k in keys
    }
    return dict(n_steps=len(resolved), totals=totals, total_observed=total_observed,
                sum_reconstructed=sum_reconstructed, closure_error=closure_error)
