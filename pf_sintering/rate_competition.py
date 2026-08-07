"""Natural sink-off coarsening driven by the model's own physics, sampled
with the full diagnostic set (legacy sigma, direct TJ capillary vectors,
signed curvature) -- the rate-competition campaign.

Unlike the earlier constrained-relaxation work (now checkpointed as
EXPERIMENTAL/DIAGNOSTIC only, see constrained_relaxation.py), this module
does not prescribe V2, does not hold any degree of freedom other than the
sink/RBM being off, and does not attempt to find a converged minimum. It
simply runs the model's existing Ostwald + CH + structural-relaxation physics
forward and measures what happens, exactly as in Milestone 2
(scripts/run_sinkoff_stress_matrix.py) but with the richer per-sample
diagnostics from the direct TJ-force and signed-curvature work folded in, and
with the new coarsening_rate_scale / surface_mobility_scale / eta_mobility_scale
controls available to probe the rate competition among them.

Per-step operator sequence matches the qualified production runner (CH ->
mass-preserving eta projection -> Ostwald -> structural relaxation ->
mass-preserving eta projection). hazard_step and rbm are never called, so the
sink stays inactive and no RBM occurs by construction (not merely by
non-activation) -- separation is expected to stay fixed up to the same small
shape-driven centroid drift documented in MILESTONE_1_2_REPORT.md, not
enforced by any corrective projection.
"""

from __future__ import annotations

import math

import numpy as np

from .diagnostics import sample as base_sample
from .model import Sink, compute_stress, evolve_eta, evolve_f, ostwald_substrate
from .signed_curvature import signed_curvature_top_bottom
from .structural_projection import eta_masses, project_eta_mass_preserving
from .tj_force import compute_neck_tj_forces, locate_neck_tjs


def rich_sample(f, e1, e2, e3, s, st, p, step, time_s, v20) -> dict:
    """diagnostics.sample() plus the direct TJ-force and signed-curvature
    diagnostics requested for this campaign (measured psi at both TJs, xi
    vectors, F_TJ and its components, signed curvature) -- composed here
    without modifying any of the underlying, already-validated modules."""
    row = base_sample(f, e1, e2, e3, s, st, p, step, time_s, v20)

    tjs = locate_neck_tjs(f, e1, e2, p)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    row["tj_resolved"] = rep.resolved
    row["n_tj_resolved"] = rep.n_resolved
    row["F_drive"] = rep.F_drive

    kappa_top = kappa_bottom = math.nan
    if tjs is not None:
        kappa_top, kappa_bottom = signed_curvature_top_bottom(f, tjs["tj_top"], tjs["tj_bottom"], p)
    row["kappa_top_1pm"] = kappa_top
    row["kappa_bottom_1pm"] = kappa_bottom

    for name, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            for key in ("psi_deg", "F_TJ_mag", "F_TJ_x", "F_gb_normal", "F_gb_parallel"):
                row[f"{key}_{name}"] = math.nan
            row[f"xi_s1_{name}"] = row[f"xi_s2_{name}"] = row[f"xi_gb_{name}"] = None
            continue
        row[f"psi_deg_{name}"] = tj.psi_deg
        row[f"F_TJ_mag_{name}"] = tj.F_TJ_mag
        row[f"F_TJ_x_{name}"] = tj.F_TJ_x
        row[f"F_gb_normal_{name}"] = tj.F_gb_normal
        row[f"F_gb_parallel_{name}"] = tj.F_gb_parallel
        row[f"xi_s1_{name}"] = tuple(float(v) for v in tj.xi_s1)
        row[f"xi_s2_{name}"] = tuple(float(v) for v in tj.xi_s2)
        row[f"xi_gb_{name}"] = tuple(float(v) for v in tj.xi_gb)
    return row


def run_sinkoff_trajectory(
    p, f0, e1_0, e2_0, e3_0,
    target_dv2_frac=1e-3, max_steps=20000, sample_every=50, seed=42,
):
    """Sink-off, RBM-off natural coarsening trajectory: CH + mass-preserving
    projection -> Ostwald -> structural relaxation + mass-preserving
    projection, every step. hazard_step and rbm are never called (sink stays
    inactive and no RBM occurs by construction, not by non-activation).
    Stops once |V2(t) - V2(0)| / V2(0) reaches target_dv2_frac, or at
    max_steps. Returns the list of rich_sample() rows."""
    f, e1, e2, e3 = (a.copy() for a in (f0, e1_0, e2_0, e3_0))
    s = Sink(threshold=math.inf)  # never touched: hazard integration suspended

    v20 = float(e2.sum() * p.dx * p.dx)
    st0, stop0, reason0 = compute_stress(f, e1, e2, e3, s, p)
    if stop0:
        raise RuntimeError(f"initial state already stopped: {reason0}")

    samples = [rich_sample(f, e1, e2, e3, s, st0, p, 0, 0.0, v20)]
    step = 0
    while step < max_steps:
        step += 1

        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s, Sink(), p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)

        f, e1, e2, e3 = ostwald_substrate(f, e1, e2, e3, p)

        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)

        st, stop, reason = compute_stress(f, e1, e2, e3, s, p)

        v2 = float(e2.sum() * p.dx * p.dx)
        take_sample = (step % sample_every == 0) or stop
        if take_sample:
            samples.append(rich_sample(f, e1, e2, e3, s, st, p, step, step * p.dt, v20))
        if stop:
            break
        if abs(v2 - v20) / v20 >= target_dv2_frac:
            if not take_sample:
                samples.append(rich_sample(f, e1, e2, e3, s, st, p, step, step * p.dt, v20))
            break

    return samples
