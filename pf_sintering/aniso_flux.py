"""Milestone 15F Sections 4/9: anisotropic surface-energy chemical
potential for the M15-series production CH pathway.

CRITICAL FINDING (Section 4 audit): `p.use_aniso_surface`/`p.aniso_delta`
and the `a(psi)/ap(psi)` LUT (model.py build_params) are ONLY consumed by
the OLDER `model.evolve_f` Euler-step function and by diagnostic-only
code (`model.compute_stress`, `tj_force._aniso_gamma` usage) -- the
M15/M15E/M15F production pathway (`ch_exact_energy.mu_isotropic` +
`surface_transport.variational_surface_diffusion_step`) computes `mu`
via `mu_isotropic`, which is HARDCODED isotropic regardless of
`p.use_aniso_surface`. Simply setting `use_aniso_surface=True` in a
M15-series config would therefore be a SILENT NO-OP on the actual
dynamics. This module provides the anisotropic equivalent of
`mu_isotropic`, generalized to the BC-aware operators
(`bc_ops.grad_bc/div_bc`) the M15 series' `bc_x="reflecting",
bc_y="periodic"` convention requires (model.py's own anisotropic branch
hardcodes `grad`/`div`, which are periodic-x/one-sided-y only).

The gradient-energy law reproduced here (Kobayashi 1993 / McFadden et
al. weakly-anisotropic form, ALREADY implemented -- not invented here --
in model.py's `evolve_f` anisotropic branch) is

    E_grad(f) = integral (k_f/2) * a(theta)^2 * |grad f|^2 dA,
    theta = atan2(fy, fx), a(psi)=1-d*cos(4*psi), psi=(theta-theta0)%(pi/2)

whose Euler-Lagrange variational derivative is the standard anisotropic
result

    dE_grad/df = -div[ k_f*a*(a*fx - a'*fy), k_f*a*(a*fy + a'*fx) ]
               = -div[ k_f*(a^2*fx - a*a'*fy), k_f*(a^2*fy + a*a'*fx) ]

(verified algebraically: a*(a*fx-a'*fy) = a^2*fx - a*a'*fy).

DISCRETIZATION FIX (found empirically before any production use): a
first version of this module computed the WHOLE gradient term via
`div_bc(k_f*(a^2*fx-a*a'*fy), ...)` (the direct composition of two
independent `grad_bc`/`div_bc` central differences). Compared against
`mu_isotropic` AT DELTA=0 (a=1, a'=0, where the two should coincide) this
showed up to ~50% LOCAL disagreement deep inside the diffuse-interface
core (f~0.5, far from any domain edge) -- not roundoff, and not a small
perturbation: `div_bc(grad_bc(...))` is a naive 5-point Laplacian
composition, while `mu_isotropic`'s gradient term uses `model.lap9`, a
higher-order 9-point COMPACT stencil (Milestone 12B's validated exact
discrete adjoint) -- a genuinely different discretization of the same
continuum operator, and at W/dx~8 grid cells across the interface the
two are not close. Using it directly would have silently confounded
"does anisotropy change the neck" with "does switching stencils change
the neck", exactly for the AN0 vs AN1+ comparisons this milestone needs
to be clean.

Fix: split a^2 = 1 + (a^2-1) and distribute the divergence linearly,

    mu_grad = -k_f*lap9_bc(f)
            - div_bc[ k_f*((a^2-1)*fx - a*a'*fy), k_f*((a^2-1)*fy + a*a'*fx) ]

so the DOMINANT isotropic part reuses the exact same compact 9-point
stencil as the (already-validated) isotropic baseline, generalized to
the M15 series' own bc_x/bc_y convention via `lap9_bc` (bc_ops.py: exact
match to `model.lap9` at the legacy default bc_x=periodic/bc_y=
reflecting, so this is a strict BC generalization, not a different
operator). At delta=0, a=1 and a'=0 identically, so the correction term
is EXACTLY zero and `mu_anisotropic` reduces EXACTLY to
`-k_f*lap9_bc(f,...)` -- the only remaining difference from
`mu_isotropic` at AN0 is `lap9_bc`'s bc_x=reflecting/bc_y=periodic vs
`mu_isotropic`'s hardcoded bc_x=periodic/bc_y=reflecting (the M15
series' own physically-correct convention for this "sinusoidal_
substrate" geometry -- see BC_X/BC_Y in
scripts/m15_gb_surface_rate_competition.py -- vs `model.lap9`'s legacy
hardcoded opposite), confirmed inert in practice (Section 9's
qualification run) because the CH mobility q(f)=(12/W)f^2(1-f)^2
vanishes in bulk, so any boundary-treatment difference in `mu` outside
the diffuse interface never reaches the actual flux.

`theta0` (crystal-orientation reference) is the SAME locally-varying,
eta-fraction-weighted blend model.py's `evolve_f` uses --
`theta0(x) = sum_i(e_i(x)*theta_grain[i]) / sum_i(e_i(x))` -- so a single
global anisotropy law smoothly represents multiple differently-oriented
grains across one field.

Caveat (documented, not hidden): the anisotropic CORRECTION term itself
still uses `div_bc(grad_bc(...))`, which is NOT the exact discrete
adjoint of any single quadratic functional under a reflecting BC (see
bc_ops.py's own docstring warnings) -- unavoidable without a compact
9-point generalization of the anisotropic cross term, which does not
exist in this codebase. Since AN1-AN3 keep delta well inside the
weakly-anisotropic regime (delta<1/15), this correction is always a
small perturbation on top of the exact dominant term, not the whole
operator. Energy descent (`F_total` monotonicity, already tracked by
every M15-series trajectory) is the empirical check for whether this
approximation is adequate at the grid resolutions used here.
"""

from __future__ import annotations

import numpy as np

from .bc_ops import div_bc, grad_bc, lap9_bc
from .ch_exact_energy import mu0_bulk


def _local_theta0(e1, e2, e3, f, p):
    fb = np.clip(f, 0.0, 1.0)
    es = [np.clip(e, 0.0, fb) for e in (e1, e2, e3)]
    sm = sum(es) + 1e-30
    return (es[0] * p.theta_grain[0] + es[1] * p.theta_grain[1] + es[2] * p.theta_grain[2]) / sm


def aniso_a_ap(f, e1, e2, e3, p, bc_x, bc_y):
    """(a, ap, gx, gy) fields: a(theta)=gamma(theta)/gamma_s, ap=da/dtheta,
    at every grid point, using the SAME LUT model.py's build_params
    constructs (p.lut_psi/lut_a/lut_ap)."""
    gx, gy = grad_bc(f, p.dx, bc_x=bc_x, bc_y=bc_y)
    th0 = _local_theta0(e1, e2, e3, f, p)
    th = np.arctan2(gy, gx)
    ps = np.mod(th - th0, np.pi / 2)
    a = np.interp(ps, p.lut_psi, p.lut_a)
    ap = np.interp(ps, p.lut_psi, p.lut_ap)
    flat = gx * gx + gy * gy < (0.01 / p.interface_width) ** 2
    a = np.where(flat, 1.0, a)
    ap = np.where(flat, 0.0, ap)
    return a, ap, gx, gy


def mu_anisotropic(f, e1, e2, e3, s, p, bc_x, bc_y):
    """Anisotropic equivalent of ch_exact_energy.mu_isotropic. Requires
    p.use_aniso_surface and a built p.lut_psi/lut_a/lut_ap (i.e. p was
    built with aniso_delta set and use_aniso_surface=True). Splits the
    dominant isotropic part onto the compact 9-point `lap9_bc` stencil
    (matching mu_isotropic's discretization) and adds only the genuinely
    anisotropic correction via div_bc(grad_bc(...)) -- see module
    docstring; reduces EXACTLY to -k_f*lap9_bc(f) at delta=0."""
    a, ap, gx, gy = aniso_a_ap(f, e1, e2, e3, p, bc_x, bc_y)
    a2m1 = a * a - 1.0
    Jx = p.k_f * (a2m1 * gx - a * ap * gy)
    Jy = p.k_f * (a2m1 * gy + a * ap * gx)
    mu_grad = -p.k_f * lap9_bc(f, p.dx, bc_x=bc_x, bc_y=bc_y) - div_bc(Jx, Jy, p.dx, bc_x=bc_x, bc_y=bc_y)
    return mu0_bulk(f, e1, e2, e3, s, p) + mu_grad


def aniso_grad_energy_density(f, e1, e2, e3, p, bc_x, bc_y):
    """(k_f/2)*a(theta)^2*|grad f|^2 -- the anisotropic gradient-energy
    density, for F_total/energy-descent bookkeeping (replaces the
    isotropic (k_f/2)|grad f|^2 term when p.use_aniso_surface)."""
    a, _ap, gx, gy = aniso_a_ap(f, e1, e2, e3, p, bc_x, bc_y)
    return 0.5 * p.k_f * a * a * (gx * gx + gy * gy)
