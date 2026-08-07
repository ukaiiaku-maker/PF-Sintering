"""Operator-resolved geometry ledger -- DIAGNOSTIC ONLY, not wired into
production physics.

Decomposes ONE production-ordering physical step (CH -> mass-preserving
projection -> Ostwald -> structural relaxation -> mass-preserving
projection) into its five constituent operator stages and samples the full
rich diagnostic state (`rate_competition.rich_sample`, itself a composition
of the already-validated `diagnostics.py` + `tj_force.py` +
`signed_curvature.py`) between each one, so per-operator deltas can be
computed and attributed:

    STATE_n -> CH -> STATE_CH_raw -> post-CH projection -> STATE_CH_projected
      -> Ostwald -> STATE_Ostwald -> eta -> STATE_eta_raw
      -> post-eta projection -> STATE_{n+1}

The production ordering and formulas are unchanged -- this module only
inserts measurements between the existing operators. `hazard_step` and
`rbm` are never called (sink-off audit only, exactly as in
`rate_competition.run_sinkoff_trajectory`).

Per-operator deltas are true differences between immediately adjacent
operator states within the SAME step, so summing them over a trajectory is
a telescoping sum and reproduces the total observed change to floating-point
closure by construction -- `summarize_ledger` computes and reports this
closure explicitly rather than assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import Sink, compute_stress, evolve_eta, evolve_f, ostwald_substrate
from .rate_competition import rich_sample
from .structural_projection import eta_masses, project_eta_mass_preserving

STAGES = ("start", "CH_raw", "CH_projected", "ostwald", "eta_raw", "end")
OPERATORS = ("CH", "post_CH_projection", "Ostwald", "eta_raw", "post_eta_projection")

_LEDGER_KEYS = (
    "V2", "separation_m", "x_neck_m", "A_GB_m2",
    "sigma_Pa", "sigma_lt_Pa", "sigma_curv_Pa",
    "psi_deg_top", "psi_deg_bottom",
    "F_TJ_mag_top", "F_TJ_mag_bottom", "F_TJ_x_top", "F_TJ_x_bottom",
    "kappa_top_1pm", "kappa_bottom_1pm",
    "E_surf_J", "E_gb_J", "G_interface_J",
)


def _delta(a, b, keys=_LEDGER_KEYS):
    out = {}
    for k in keys:
        av, bv = a.get(k), b.get(k)
        if av is None or bv is None or not (math.isfinite(av) and math.isfinite(bv)):
            out[k] = None
        else:
            out[k] = bv - av
    return out


@dataclass
class LedgerStep:
    step: int
    time_s: float
    stage_samples: dict = field(default_factory=dict)
    operator_deltas: dict = field(default_factory=dict)
    stop: bool = False
    stop_reason: str = ""


def run_ledger_step(f, e1, e2, e3, p, s, step, time_s, v20):
    """Run one physical step (sink/RBM off) with full per-operator sampling.
    Returns (f_new, e1_new, e2_new, e3_new, LedgerStep)."""
    st0, stop0, reason0 = compute_stress(f, e1, e2, e3, s, p)
    stage_samples = dict(start=rich_sample(f, e1, e2, e3, s, st0, p, step, time_s, v20))
    if stop0:
        ledger = LedgerStep(step=step, time_s=time_s, stage_samples=stage_samples,
                             operator_deltas={}, stop=True, stop_reason=reason0)
        return f, e1, e2, e3, ledger

    ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f1 = evolve_f(f, e1, e2, e3, s, Sink(), p)
    st1, _, _ = compute_stress(f1, e1, e2, e3, s, p)
    stage_samples["CH_raw"] = rich_sample(f1, e1, e2, e3, s, st1, p, step, time_s, v20)

    e1a, e2a, e3a = project_eta_mass_preserving(f1, e1, e2, e3, p, target_masses=ch_targets)
    st2, _, _ = compute_stress(f1, e1a, e2a, e3a, s, p)
    stage_samples["CH_projected"] = rich_sample(f1, e1a, e2a, e3a, s, st2, p, step, time_s, v20)

    f2, e1b, e2b, e3b = ostwald_substrate(f1, e1a, e2a, e3a, p)
    st3, _, _ = compute_stress(f2, e1b, e2b, e3b, s, p)
    stage_samples["ostwald"] = rich_sample(f2, e1b, e2b, e3b, s, st3, p, step, time_s, v20)

    ac_targets = eta_masses(e1b, e2b, e3b, p.use_eta3)
    e1c, e2c, e3c = evolve_eta(e1b, e2b, e3b, p)
    st4, _, _ = compute_stress(f2, e1c, e2c, e3c, s, p)
    stage_samples["eta_raw"] = rich_sample(f2, e1c, e2c, e3c, s, st4, p, step, time_s, v20)

    e1d, e2d, e3d = project_eta_mass_preserving(f2, e1c, e2c, e3c, p, target_masses=ac_targets)
    st5, stop5, reason5 = compute_stress(f2, e1d, e2d, e3d, s, p)
    stage_samples["end"] = rich_sample(f2, e1d, e2d, e3d, s, st5, p, step, time_s, v20)

    operator_deltas = {
        "CH": _delta(stage_samples["start"], stage_samples["CH_raw"]),
        "post_CH_projection": _delta(stage_samples["CH_raw"], stage_samples["CH_projected"]),
        "Ostwald": _delta(stage_samples["CH_projected"], stage_samples["ostwald"]),
        "eta_raw": _delta(stage_samples["ostwald"], stage_samples["eta_raw"]),
        "post_eta_projection": _delta(stage_samples["eta_raw"], stage_samples["end"]),
    }
    ledger = LedgerStep(step=step, time_s=time_s, stage_samples=stage_samples,
                         operator_deltas=operator_deltas, stop=stop5, stop_reason=reason5)
    return f2, e1d, e2d, e3d, ledger


def run_ledger_trajectory(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=1e-4, max_steps=5000):
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)  # never touched: hazard integration suspended
    v20 = float(e2.sum() * p.dx * p.dx)

    steps = []
    step = 0
    while step < max_steps:
        step += 1
        f, e1, e2, e3, ledger = run_ledger_step(f, e1, e2, e3, p, s, step, step * p.dt, v20)
        steps.append(ledger)
        if ledger.stop:
            break
        v2 = float(e2.sum() * p.dx * p.dx)
        if abs(v2 - v20) / v20 >= target_dv2_frac:
            break
    return steps


def summarize_ledger(steps):
    """Per-operator cumulative totals over the trajectory, the total observed
    change (first step's start to last step's end), and the closure error
    between the two (should be at floating-point-roundoff level, since this
    is a telescoping sum of adjacent-state deltas by construction)."""
    resolved_steps = [s for s in steps if s.operator_deltas]
    if not resolved_steps:
        raise ValueError("no resolved steps in trajectory")

    totals = {op: {k: 0.0 for k in _LEDGER_KEYS} for op in OPERATORS}
    for ledger in resolved_steps:
        for op in OPERATORS:
            d = ledger.operator_deltas[op]
            for k in _LEDGER_KEYS:
                if d[k] is not None:
                    totals[op][k] += d[k]

    first = resolved_steps[0].stage_samples["start"]
    last = resolved_steps[-1].stage_samples["end"]
    total_observed = _delta(first, last)
    sum_reconstructed = {
        k: sum(totals[op][k] for op in OPERATORS) for k in _LEDGER_KEYS
    }
    closure_error = {
        k: (sum_reconstructed[k] - total_observed[k]) if total_observed[k] is not None else None
        for k in _LEDGER_KEYS
    }
    return dict(
        n_steps=len(resolved_steps), totals=totals, total_observed=total_observed,
        sum_reconstructed=sum_reconstructed, closure_error=closure_error,
        V20=first["V2"], dV2_over_V20=(last["V2"] - first["V2"]) / first["V2"] if first["V2"] else math.nan,
    )
