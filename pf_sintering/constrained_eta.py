"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 12 Commit 3: constrained variational structural (grain-ownership)
kinetics.

`structural_projection.project_eta_mass_preserving` rigidly holds each
grain's own `integral(eta_i)` fixed -- useful for isolating the old Ostwald
mechanism (Milestones 6-9), but it forbids genuine grain coarsening by
construction: `eta_i`'s are structural/ownership variables, not conserved
mass, and the physically relevant local constraint (inspected directly
from `model.py`) is `sum_i eta_i = f` pointwise, not `integral(eta_i) =
const` per grain.

`model.evolve_eta` already implements `d(eta_i)/dt = M_eta*k_eta*lap9(eta_i)`
for each `eta_i` independently -- the gradient-flow of the EXISTING
structural free energy `G = (k_eta/2) * sum_i integral |grad(eta_i)|^2 dV`
(sign-verified: `g_i = delta G/delta eta_i = -k_eta*lap(eta_i)`, and
`d(eta_i)/dt = -M_eta*g_i = +M_eta*k_eta*lap9(eta_i)` exactly matches).
This module does not invent a new free energy; it uses the SAME `g_i` but
projects out the component that would violate `sum_i eta_i = f`:

    d(eta_i)/dt = -L_i * (g_i - lambda)

with `L_i = M_eta` for every active grain (equal mobility -- the only case
the current model provides, since `M_eta` is a single shared parameter) and
`lambda = mean_i(g_i)` over the active grains at each point. This is an
EXACT algebraic identity, not an approximation: `sum_i d(eta_i)/dt =
-M_eta*(sum_i g_i - n*lambda) = 0` for `lambda = (1/n)*sum_i g_i`, so if
`sum_i eta_i = f` holds at the start of a step, the variational update
alone preserves it to the order of the time discretization -- BEFORE any
correction. A local, minimally-corrective projection (the EXISTING
`model.reproject`, already pointwise/local and not integral-preserving --
reused unmodified, not reimplemented) then corrects any residual numerical
constraint violation (negativity, `sum > f`, discretization drift). Both
deltas (variational vs. projection) are tracked and returned separately
(Milestone 12 Sections 14-15: the projection correction must stay small
relative to the variational change, never become a hidden coarsening
driver in its own right).
"""

from __future__ import annotations

import numpy as np

from .bc_ops import lap9_bc
from .model import reproject


def structural_thermodynamic_force(eta_i, dx, k_eta, bc_x="periodic", bc_y="reflecting"):
    """g_i = delta G/delta eta_i = -k_eta*lap(eta_i) for the existing
    production structural (gradient-only) free energy -- see module
    docstring for the sign derivation matching model.evolve_eta exactly."""
    return -k_eta * lap9_bc(eta_i, dx, bc_x=bc_x, bc_y=bc_y)


def constrained_variational_eta_update(e1, e2, e3, f, p, dt=None, use_eta3=False,
                                        bc_x="periodic", bc_y="reflecting"):
    """One constrained-Allen-Cahn step for the active eta fields (equal
    mobility L_i=M_eta), followed by model.reproject's existing local
    positivity/simplex correction. Returns (e1_new, e2_new, e3_new, diag)
    with the variational and projection deltas tracked separately."""
    dt = dt if dt is not None else p.dt
    g1 = structural_thermodynamic_force(e1, p.dx, p.k_eta, bc_x, bc_y)
    g2 = structural_thermodynamic_force(e2, p.dx, p.k_eta, bc_x, bc_y)
    if use_eta3:
        g3 = structural_thermodynamic_force(e3, p.dx, p.k_eta, bc_x, bc_y)
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
