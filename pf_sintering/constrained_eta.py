"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 12/12B: constrained variational structural (grain-ownership)
kinetics.

`structural_projection.project_eta_mass_preserving` rigidly holds each
grain's own `integral(eta_i)` fixed -- useful for isolating the old Ostwald
mechanism (Milestones 6-9), but it forbids genuine grain coarsening by
construction: `eta_i`'s are structural/ownership variables, not conserved
mass, and the physically relevant local constraint (inspected directly
from `model.py`) is `sum_i eta_i = f` pointwise, not `integral(eta_i) =
const` per grain.

**Milestone 12B correction**: Milestone 12's `g_i = -k_eta*lap(eta_i)`
(matching `model.evolve_eta`'s own gradient-only kinetics) is
**INCOMPLETE**. Inspecting `model.evolve_f`'s actual chemical potential
(`mu0 = W_f*f(1-f)(1-2f) - Wc*eta2*(1-fb)`, `eta2 = sum_i clip(eta_i,0,fb)^2`,
`Wc = 36*gl(local)/W`) shows the intended free energy contains an
eta-f BULK COUPLING term beyond the pure gradient term:

    F = integral [ (W_f/2)*f^2*(1-f)^2 + Wc*eta2*(f^2/2 - f)
                   + (k_eta/2)*sum_i |grad(eta_i)|^2 ] dV

(the bulk term's `f`-derivative reproduces `mu0` exactly -- Milestone 8's
`ch_exact_energy.py` derivation, reused verbatim here). Since
`eta2 = sum_i eta_i^2`, this term ALSO has a nonzero derivative with
respect to each `eta_i` individually (`d(eta2)/d(eta_i) = 2*eta_i`, no
cross-coupling to eta_j, i != j, since eta2 is a sum of squares, not a
product):

    g_i = delta F/delta eta_i = -k_eta*lap(eta_i) + 2*Wc*eta_i*(f^2/2 - f)

Production `model.evolve_eta` (`d(eta_i)/dt = M_eta*k_eta*lap9(eta_i)`)
implements only the FIRST term -- Outcome E2 of the Milestone 12B
handoff's Section 5 decision tree ("additional eta-dependent terms are
part of the intended F... implement the COMPLETE g_i... do not preserve
the old incomplete eta evolution merely for parity"). Verified via the
same finite-difference directional-derivative test Milestone 8 used for
`ch_exact_energy.py`: `[F(eta_i+eps*q)-F(eta_i-eps*q)]/(2*eps)` vs.
`dx^2*sum(g_i*q)`, using `-0.5*k_eta*eta_i*lap9(eta_i)` (not a naive
`|grad|^2` central-difference sum, for the same `lap9`-self-adjointness
reason `ch_exact_energy.py` documents) for the gradient term -- machine
precision agreement (`rel_err ~1e-12`, `tests/test_constrained_eta.py`).

Constrained update at fixed `f`:

    d(eta_i)/dt = -L_i * (g_i - lambda)

with `L_i = M_eta` for every active grain (equal mobility -- the only case
the current model provides) and `lambda = mean_i(g_i)` over the active
grains at each point -- an EXACT algebraic identity (`sum_i d(eta_i)/dt
= 0`), unaffected by which `g_i` is used. A local, minimally-corrective
projection (the EXISTING `model.reproject`, unmodified) then corrects any
residual numerical constraint violation; both deltas are tracked and
returned separately (variational vs. projection must stay a minor
correction, never a hidden coarsening driver in its own right).
"""

from __future__ import annotations

import numpy as np

from .bc_ops import lap9_bc
from .model import effective_gamma, reproject


def local_wc(f, e1, e2, e3, s, p):
    """Wc(local) = 36*gl/W, the SAME local-effective-gamma construction
    model.evolve_f uses for its eta-f coupling term (gl = gamma_gb_ref
    away from a GB, effective_gamma(s,p) where e1*e2>1e-20 -- reproduced
    here exactly, not re-derived, to guarantee identical Wc in both
    contexts)."""
    fb = np.clip(f, 0.0, 1.0)
    es = [np.clip(e, 0.0, fb) for e in (e1, e2, e3)]
    pair = np.maximum(0.0, es[0] * es[1])
    gl = np.full_like(f, p.gamma_gb_ref)
    mask = pair > 1e-20
    if np.any(mask):
        gl[mask] = effective_gamma(s, p)
    return 36.0 * gl / p.interface_width


def structural_thermodynamic_force(eta_i, f, Wc, dx, k_eta, bc_x="periodic", bc_y="reflecting"):
    """g_i = delta F/delta eta_i = -k_eta*lap(eta_i) + 2*Wc*eta_i*(f^2/2-f)
    -- the COMPLETE derivative (Milestone 12B Section 5; see module
    docstring), not just the gradient term Milestone 12 used."""
    grad_term = -k_eta * lap9_bc(eta_i, dx, bc_x=bc_x, bc_y=bc_y)
    coupling_term = 2.0 * Wc * eta_i * (0.5 * f * f - f)
    return grad_term + coupling_term


def constrained_variational_eta_update(e1, e2, e3, f, s, p, dt=None, use_eta3=False,
                                        bc_x="periodic", bc_y="reflecting", n_substeps=1):
    """One constrained-Allen-Cahn step (at fixed f) for the active eta
    fields (equal mobility L_i=M_eta), using the COMPLETE
    structural_thermodynamic_force, followed by model.reproject's existing
    local positivity/simplex correction. `s` (Sink) is required now
    (local_wc needs it for effective_gamma, matching model.evolve_f's own
    signature). Returns (e1_new, e2_new, e3_new, diag) with the variational
    and projection deltas tracked separately (summed over substeps).

    Milestone 12B Section 7: the bulk coupling term 2*Wc*eta_i*(f^2/2-f) is
    a genuine spinodal-type driving force (uneven ownership is energetically
    favored at fixed Wc, f -- the standard multi-order-parameter grain-growth
    mechanism) and is therefore unconditionally unstable under forward-Euler
    at any fixed dt: the linearized bulk dynamics is d(eta_i)/dt =
    +M_eta*Wc*(eta_i-mean(eta)), an exponentially growing mode with no
    stable dt. `n_substeps>1` runs n_substeps forward-Euler+reproject cycles
    of dt/n_substeps each (fixed f throughout) instead of one cycle of dt --
    this does not remove the instability but shrinks the per-step overshoot
    that reproject must correct, which is the mechanism by which projection
    reliance is reduced (see MILESTONE_12B report Section 5)."""
    dt = dt if dt is not None else p.dt
    if n_substeps > 1:
        dt_sub = dt / n_substeps
        var_total = proj_total = 0.0
        diag = None
        for _ in range(n_substeps):
            e1, e2, e3, diag = constrained_variational_eta_update(
                e1, e2, e3, f, s, p, dt=dt_sub, use_eta3=use_eta3, bc_x=bc_x, bc_y=bc_y, n_substeps=1)
            var_total += diag["variational_change"]
            proj_total += diag["projection_change"]
        diag = dict(diag)
        diag["variational_change"] = var_total
        diag["projection_change"] = proj_total
        diag["projection_fraction"] = proj_total / (var_total + 1e-300)
        return e1, e2, e3, diag

    Wc = local_wc(f, e1, e2, e3, s, p)
    g1 = structural_thermodynamic_force(e1, f, Wc, p.dx, p.k_eta, bc_x, bc_y)
    g2 = structural_thermodynamic_force(e2, f, Wc, p.dx, p.k_eta, bc_x, bc_y)
    if use_eta3:
        g3 = structural_thermodynamic_force(e3, f, Wc, p.dx, p.k_eta, bc_x, bc_y)
        lam = (g1 + g2 + g3) / 3.0
    else:
        g3 = np.zeros_like(e1)
        lam = (g1 + g2) / 2.0

    d1 = -p.M_eta * dt * (g1 - lam)
    d2 = -p.M_eta * dt * (g2 - lam)
    d3 = -p.M_eta * dt * (g3 - lam) if use_eta3 else np.zeros_like(e1)

    e1_var = e1 + d1
    e2_var = e2 + d2
    e3_var = e3 + d3 if use_eta3 else np.zeros_like(e1)

    if use_eta3:
        e1_proj, e2_proj, e3_proj = reproject(f, e1_var, e2_var, e3_var)
    else:
        e1_proj, e2_proj, _ = reproject(f, e1_var, e2_var, np.zeros_like(e1))
        e3_proj = e3

    variational_change = float(np.sum(np.abs(d1)) + np.sum(np.abs(d2)) + np.sum(np.abs(d3)))
    proj_corr_1 = e1_proj - e1_var
    proj_corr_2 = e2_proj - e2_var
    proj_corr_3 = (e3_proj - e3_var) if use_eta3 else np.zeros_like(e1)
    projection_change = float(np.sum(np.abs(proj_corr_1)) + np.sum(np.abs(proj_corr_2)) + np.sum(np.abs(proj_corr_3)))

    diag = dict(
        g1=g1, g2=g2, lam=lam, d1=d1, d2=d2,
        variational_change=variational_change, projection_change=projection_change,
        projection_fraction=projection_change / (variational_change + 1e-300),
    )
    return e1_proj, e2_proj, e3_proj, diag


def f_weighted_ownership_volumes(f, e1, e2, e3, dx, eps=1e-30):
    """Physical grain-volume measure (Milestone 12 Section 16): NOT raw
    integral(eta_i) (not conserved mass), but the f-weighted normalized
    ownership fraction w_i = eta_i/(sum_j eta_j + eps), V_i = integral(f*w_i)dV.
    For the two-grain substrate system, V1+V2 ~ integral(f)dV up to
    diffuse-interface/numerical tolerance."""
    denom = e1 + e2 + e3 + eps
    w1, w2, w3 = e1 / denom, e2 / denom, e3 / denom
    V1 = float(np.sum(f * w1)) * dx * dx
    V2 = float(np.sum(f * w2)) * dx * dx
    V3 = float(np.sum(f * w3)) * dx * dx
    return V1, V2, V3
