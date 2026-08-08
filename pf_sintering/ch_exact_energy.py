"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 8, Section 12: the EXACT discrete free-energy functional for
isotropic CH (`p.use_aniso_surface=False`), fixing Milestone 7's naive
version (`differential_coarsening_audit.py::exact_free_energy_isotropic`),
which used a central-difference `|grad f|^2` gradient-energy term that is
NOT the exact discrete Euler-Lagrange conjugate of `model.lap9` (the 9-point
compact/rotated-Laplacian stencil production `evolve_f` actually
differentiates with).

Derivation: `lap9` is numerically confirmed symmetric under the plain dot
product (`sum(a*lap9(b,dx)) == sum(b*lap9(a,dx))` to ~1e-16 relative, despite
its periodic-x/clamped-y boundary treatment -- verified in
`tests/test_ch_exact_energy.py`), so it IS a valid discrete Laplacian for a
quadratic-energy variational derivative. For `E_grad(f) = -(k_f/2) * dx^2 *
sum(f * lap9(f,dx))`, linearity of `lap9` plus that symmetry gives, for any
perturbation direction q:

    d/deps[E_grad(f+eps*q)]|eps=0
        = -(k_f/2)*dx^2*[sum(q*lap9(f,dx)) + sum(f*lap9(q,dx))]
        = -k_f*dx^2*sum(q*lap9(f,dx))          (by the symmetry above)
        = dx^2 * sum((-k_f*lap9(f,dx)) * q)

exactly matching production's isotropic gradient term in `mu`. Combined
with the bulk term (already exact in Milestone 7's version -- local,
pointwise, no cross-cell coupling, so its discrete directional derivative
is trivially exact), `F_exact` below reproduces the FULL production
isotropic `mu` to finite-difference-truncation precision, verified directly
via `[F(f+eps*q)-F(f-eps*q)]/(2*eps)` vs. `dx^2*sum(mu*q)` in
`tests/test_ch_exact_energy.py` (O(eps^2) convergence down to the float64
roundoff floor).
"""

from __future__ import annotations

import numpy as np

from .model import effective_gamma, lap9


def _wc(e1, e2, e3, s, p):
    pair = np.maximum(0.0, e1 * e2)
    gl = np.full_like(e1, p.gamma_gb_ref)
    mask = pair > 1e-20
    if np.any(mask):
        gl[mask] = effective_gamma(s, p)
    return 36.0 * gl / p.interface_width


def mu0_bulk(f, e1, e2, e3, s, p):
    """Local (pointwise) part of mu -- identical formula to production's
    mu0 inside evolve_f / ch_flux_diagnostics.evolve_f_diagnostic."""
    fb = np.clip(f, 0.0, 1.0)
    eta2 = e1 * e1 + e2 * e2 + e3 * e3
    Wc = _wc(e1, e2, e3, s, p)
    return p.W_f * f * (1.0 - f) * (1.0 - 2.0 * f) - Wc * eta2 * (1.0 - fb)


def mu_isotropic(f, e1, e2, e3, s, p):
    """Full isotropic mu (bulk + gradient), matching production's evolve_f
    with use_aniso_surface=False exactly."""
    return mu0_bulk(f, e1, e2, e3, s, p) - p.k_f * lap9(f, p.dx)


def exact_free_energy_isotropic(f, e1, e2, e3, s, p):
    """The exact discrete free energy (isotropic mode only) whose discrete
    directional derivative reproduces mu_isotropic() exactly (see module
    docstring). F = sum[(W_f/2)f^2(1-f)^2 + Wc*eta2*(f^2/2-f)
    - (k_f/2)*f*lap9(f,dx)] * dx^2."""
    fb = np.clip(f, 0.0, 1.0)
    eta2 = e1 * e1 + e2 * e2 + e3 * e3
    Wc = _wc(e1, e2, e3, s, p)
    e_bulk = 0.5 * p.W_f * f * f * (1 - f) ** 2 + Wc * eta2 * (0.5 * f * f - f)
    e_grad = -0.5 * p.k_f * f * lap9(f, p.dx)
    return float(np.sum(e_bulk + e_grad)) * p.dx * p.dx
