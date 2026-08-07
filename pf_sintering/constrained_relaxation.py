"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not developed further, not wired into
production physics. Checkpointed as-is per the pivot documented in
MILESTONE_4_CONSTRAINED_RELAXATION_REPORT.md.

Constrained relaxation at fixed (V2, L): the Milestone-4 benchmark.

Relaxes the free surface, structural fields, and GB/TJ geometry toward the
lowest-interfacial-energy state reachable while holding the shrinking-grain
volume V2 and the particle/substrate separation L fixed exactly (to a stated
tolerance), with Ostwald exchange, the stochastic hazard, and RBM all off.
`x_neck` and `psi` are NOT constrained -- they are read off the converged
state, not dialed in.

MILESTONE_4_CONSTRAINED_RELAXATION_REPORT.md found that, within a practical
step budget, this explicit-time-stepping relaxation does not converge
tightly enough to treat G* as a well-defined function of (V2, L) (two
different starting shapes at the same (V2, L) land in measurably different
states), likely an intrinsic (system_size/reference_length)^4 stiffness of
explicit Cahn-Hilliard time-stepping at this particle size. The project has
since pivoted to letting the model's own coarsening dynamics run instead of
prescribing V2 changes (see `pf_sintering/rate_competition.py` and
`scripts/rate_competition_benchmark.py`), which is now the primary benchmark
for the coarsening/stress question. This module is retained for reference
and is not being developed further for now.

This module was written to supersede `static_neck_geometry.py` (which
artificially fixes x_neck too, and whose x0 was found not to be a physical
equilibrium -- see MILESTONE_3_FORCE_BALANCE_REPORT.md) as the source of
"equilibrium-like" geometry. That module is still used here only to generate
a convenient, already-precise (V2, L) starting guess; the real relaxation
dynamics (CH + structural relaxation, exactly as qualified in
pf_sintering/runner.py) then take over and are what actually determine the
converged shape.

Per-step operator sequence mirrors the qualified production runner (CH ->
mass-preserving eta projection -> structural relaxation -> mass-preserving
eta projection), with the joint (V2, L) corrector from
`separation_constraint.py` applied after that. No Ostwald, no hazard, no RBM
anywhere in this loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .diagnostics import contact_area, gb_energy, surface_energy, wall_x0
from .model import Sink, contact_width, evolve_eta, evolve_f
from .separation_constraint import enforce_V2_and_L
from .signed_curvature import signed_curvature_top_bottom
from .static_neck_geometry import build_neck_state
from .structural_projection import eta_masses, project_eta_mass_preserving
from .tj_force import compute_neck_tj_forces, locate_neck_tjs


@dataclass
class RelaxResult:
    f: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray
    steps_run: int
    converged: bool
    G_history: list = field(default_factory=list)  # (step, G_interface)
    dL_final: float = math.nan
    dV2_final: float = math.nan


def relax_at_fixed_V2_L(
    p, s, V2_target, L_target,
    init=None, x_neck_seed=None,
    n_steps=4000, tol_L=None, check_every=10,
    converge_window=20, converge_rtol=1e-7,
):
    """Run the constrained relaxation to (near-)convergence.

    `init`: optional (f, e1, e2, e3) warm start (e.g. a previously converged
    state, translated to a new L). If omitted, `x_neck_seed` (a physical
    length) selects the experimental static-neck-geometry family purely as
    an initial guess -- x_neck is free to move away from that seed value
    during relaxation.
    """
    w0 = wall_x0(p)
    if tol_L is None:
        tol_L = 1e-4 * p.dx

    if init is not None:
        f, e1, e2, e3 = (a.copy() for a in init)
    else:
        if x_neck_seed is None:
            raise ValueError("must supply either init or x_neck_seed")
        st = build_neck_state(p, x_neck_seed, V2_target, L_target)
        f, e1, e2, e3 = st.f, st.e1, st.e2, st.e3

    # Pin (V1, V2, V3) to their initial values for the whole run.
    v_targets = eta_masses(e1, e2, e3, p.use_eta3)
    f, e1, e2, e3, _ = enforce_V2_and_L(f, e1, e2, e3, p, w0, L_target, v_targets, tol_L)

    s_dummy = Sink()  # never touched: no hazard call anywhere in this loop
    G_history = []
    converged = False
    step = 0
    for step in range(1, n_steps + 1):
        ch_targets = eta_masses(e1, e2, e3, p.use_eta3)
        f = evolve_f(f, e1, e2, e3, s_dummy, Sink(), p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ch_targets)

        ac_targets = eta_masses(e1, e2, e3, p.use_eta3)
        e1, e2, e3 = evolve_eta(e1, e2, e3, p)
        e1, e2, e3 = project_eta_mass_preserving(f, e1, e2, e3, p, target_masses=ac_targets)

        f, e1, e2, e3, corr = enforce_V2_and_L(f, e1, e2, e3, p, w0, L_target, v_targets, tol_L)

        if step % check_every == 0 or step == 1:
            g = surface_energy(f, p) + gb_energy(e1, e2, s, p)
            G_history.append((step, g))
            if len(G_history) > converge_window:
                _, g_prev = G_history[-converge_window - 1]
                if g_prev != 0 and abs(g - g_prev) / abs(g_prev) < converge_rtol:
                    converged = True
                    break

    f, e1, e2, e3, final_corr = enforce_V2_and_L(f, e1, e2, e3, p, w0, L_target, v_targets, tol_L)
    return RelaxResult(
        f=f, e1=e1, e2=e2, e3=e3, steps_run=step, converged=converged,
        G_history=G_history, dL_final=final_corr.dL_final, dV2_final=final_corr.dV2_final,
    )


def summarize_constrained_state(f, e1, e2, e3, p, s):
    """All quantities requested for the constrained-state report."""
    x_neck, _ = contact_width(e1, e2, p)
    a_gb = contact_area(e1, e2, p)
    e_surf = surface_energy(f, p)
    e_gb = gb_energy(e1, e2, s, p)
    v2 = float(e2.sum() * p.dx * p.dx)

    tjs = locate_neck_tjs(f, e1, e2, p)
    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    kappa_top = kappa_bottom = math.nan
    if tjs is not None:
        kappa_top, kappa_bottom = signed_curvature_top_bottom(f, tjs["tj_top"], tjs["tj_bottom"], p)

    out = dict(
        V2=v2, x_neck_m=x_neck, A_GB_m2=a_gb,
        E_surf_J=e_surf, E_gb_J=e_gb, G_interface_J=e_surf + e_gb,
        tj_resolved=rep.resolved, n_tj_resolved=rep.n_resolved,
        kappa_top_1pm=kappa_top, kappa_bottom_1pm=kappa_bottom,
    )
    for name, tj in (("top", rep.top), ("bottom", rep.bottom)):
        if tj is None or not tj.resolved:
            out[f"psi_{name}_deg"] = math.nan
            out[f"F_TJ_{name}_mag"] = math.nan
            out[f"F_TJ_{name}_x"] = math.nan
            out[f"xi_s1_{name}"] = None
            out[f"xi_s2_{name}"] = None
            out[f"xi_gb_{name}"] = None
            continue
        out[f"psi_{name}_deg"] = tj.psi_deg
        out[f"F_TJ_{name}_mag"] = tj.F_TJ_mag
        out[f"F_TJ_{name}_x"] = tj.F_TJ_x
        out[f"xi_s1_{name}"] = tuple(float(x) for x in tj.xi_s1)
        out[f"xi_s2_{name}"] = tuple(float(x) for x in tj.xi_s2)
        out[f"xi_gb_{name}"] = tuple(float(x) for x in tj.xi_gb)
    return out
