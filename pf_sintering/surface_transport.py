"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 12 Commit 2: unified variational tangential surface-diffusion
transport. Replaces the special Ostwald removal/redistribution kernel with
ONE conserved flux driven by the SAME production chemical potential
`mu_f = delta F / delta f` used for reporting -- no spatial removal mask,
no prescribed volume decrement, no receiver fraction, no manual choice of
where material goes. See MILESTONE_12_UNIFIED_VARIATIONAL_SURFACE_COARSENING.md
for the full derivation; summarized here:

    n = grad(f) / sqrt(|grad(f)|^2 + eps_n^2)          (interface normal)
    P_t = I - n (x) n                                   (tangential projector)
    M_tensor = M_s * q(f) * P_t                          (rank-1, PSD, tangential-only)
    J = -M_tensor . grad(mu_f)                            (surface flux)
    df/dt = -div(J)                                      (conservation law)

`q(f) = (12/W) * f^2*(1-f)^2` is normalized so `integral q(f_0(n)) dn = 1`
for the analytic equilibrium tanh profile `f_0(x) = 0.5*(1+tanh(x/W))`
(derived from the SAME `k_f`/`W_f` the production bulk+gradient energy
already uses -- `L = 2*sqrt(k_f/W_f) = W` exactly, confirming `p.interface_
width` already IS this profile's own width parameter; the q-normalization
integral reduces to `integral f_0^2(1-f_0)^2 dx = W/12`, giving the 12/W
prefactor -- see `derive_q_normalization_analytic` and its numerical check
in tests/test_surface_transport.py). This makes the INTEGRATED tangential
mobility through the interface independent of dx or W by construction.

`M_s_ref` preserves the qualified production CH mobility's own integrated
interface conductance: the old (isotropic) `M_old(f) = min(M_f*(16f^2(1-f)^2)^2,
M_f)` evaluated on the same equilibrium profile is exactly `M_f*sech^8(x/W)`
(the min() never binds there, since sech<=1), and `integral sech^8(u)du =
32/35` (standard reduction-formula result), giving `M_s_ref = M_f*W*(32/35)`
-- dimensionally consistent (q has units 1/length, so M_s must carry the
same length dimension M_f itself lacks to match M_old's units). `surface_
mobility_scale` (already multiplying `M_f`) therefore propagates into
`M_s_ref` automatically -- no new, separate rate parameter is introduced.
"""

from __future__ import annotations

import math

import numpy as np

from .bc_ops import face_average, flux_divergence, grad_bc

SECH8_INTEGRAL = 32.0 / 35.0  # integral_{-inf}^{inf} sech^8(u) du


def interface_localization_q(f, W):
    """q(f) = (12/W) f^2(1-f)^2, normalized so integral q(f_0(n)) dn = 1
    for the analytic equilibrium profile of width W (see module docstring)."""
    return (12.0 / W) * f * f * (1.0 - f) ** 2


def q_normalization_numeric(W, dx, n_cells=4001):
    """Numerically integrate q(f_0(x)) over a fine 1-D discretization of
    the SAME analytic equilibrium profile (not the actual 2-D discrete
    field), to check the analytic 12/W prefactor independent of dx/grid --
    should return 1.0 to high precision regardless of dx (Section 5's
    'test the normalization at dx=5nm and 2.5nm with physical W=20nm')."""
    half = 8.0 * W
    x = np.linspace(-half, half, n_cells)
    f0 = 0.5 * (1.0 + np.tanh(x / W))
    q = interface_localization_q(f0, W)
    return float(np.trapezoid(q, x))


def m_s_ref(M_f, W):
    """Integrated tangential mobility preserving the qualified production
    CH mobility's own integrated interface conductance (see module
    docstring derivation). M_f already carries surface_mobility_scale."""
    return M_f * W * SECH8_INTEGRAL


def interface_normal(f, dx, bc_x, bc_y, eps_n):
    """n = grad(f)/sqrt(|grad f|^2 + eps_n^2). eps_n is a purely numerical
    regularization (Section 7): since q(f)->0 in the bulk (where grad(f)
    is undefined/noisy), eps_n must have no physical effect there -- only
    prevents division by ~0 where it is multiplied away by q(f) anyway.
    Not a physics parameter; sensitivity to its exact value within a
    reasonable numeric range is tested in tests/test_surface_transport.py."""
    gx, gy = grad_bc(f, dx, bc_x=bc_x, bc_y=bc_y)
    mag = np.sqrt(gx * gx + gy * gy + eps_n * eps_n)
    return gx / mag, gy / mag, mag


def tangential_projector(nx, ny):
    """P_t = I - n(x)n as its symmetric components (Pxx, Pxy, Pyy);
    Pyx == Pxy. Symmetric and positive-semidefinite by construction
    (I - n n^T for a unit vector n has eigenvalues {0, 1})."""
    Pxx = 1.0 - nx * nx
    Pxy = -nx * ny
    Pyy = 1.0 - ny * ny
    return Pxx, Pxy, Pyy


def surface_mobility_tensor(f, dx, W, M_s, bc_x, bc_y, eps_n):
    """M_tensor = M_s * q(f) * P_t, as its symmetric components. Cell-
    centered. Only tangential transport (Section 4: M_normal=M_bulk=
    M_vapor=conserved-GB-diffusion=0 in this first qualification -- there
    is no separate normal/bulk term anywhere in this construction to zero
    out; the tensor is rank-1 tangential-only by construction)."""
    q = interface_localization_q(f, W)
    nx, ny, _ = interface_normal(f, dx, bc_x, bc_y, eps_n)
    Pxx, Pxy, Pyy = tangential_projector(nx, ny)
    coef = M_s * q
    return coef * Pxx, coef * Pxy, coef * Pyy


def surface_flux(mu, Mxx, Mxy, Myy, dx, bc_x, bc_y):
    """J = -M_tensor . grad(mu_f), cell-centered (Myx == Mxy, symmetric tensor)."""
    gx_mu, gy_mu = grad_bc(mu, dx, bc_x=bc_x, bc_y=bc_y)
    Jx = -(Mxx * gx_mu + Mxy * gy_mu)
    Jy = -(Mxy * gx_mu + Myy * gy_mu)
    return Jx, Jy


def dissipation_density(Mxx, Mxy, Myy, mu, dx, bc_x, bc_y):
    """D_CH density = grad(mu) . M_tensor . grad(mu), pointwise -- manifestly
    non-negative wherever M_tensor is PSD (a quadratic form x^T M x with M
    PSD), which surface_mobility_tensor's construction guarantees at every
    point by construction (Section 9's D_CH>=0 requirement)."""
    gx_mu, gy_mu = grad_bc(mu, dx, bc_x=bc_x, bc_y=bc_y)
    return Mxx * gx_mu * gx_mu + 2.0 * Mxy * gx_mu * gy_mu + Myy * gy_mu * gy_mu


def surface_flux_divergence_conservative(Jx_cell, Jy_cell, dx, bc_x, bc_y):
    """Interpolates the cell-centered flux to faces (simple average) and
    applies bc_ops.flux_divergence, whose exact telescoping-sum mass
    conservation holds regardless of the face value's own accuracy (see
    bc_ops.flux_divergence docstring) -- this is what makes the update
    conservative even though the flux itself was built from a tensor
    mobility that does not fit the legacy face-centered-scalar pattern."""
    Jx_face = face_average(Jx_cell, axis=1, bc=bc_x)
    Jy_face = face_average(Jy_cell, axis=0, bc=bc_y)
    return flux_divergence(Jx_face, Jy_face, dx, bc_x=bc_x, bc_y=bc_y)


def surface_divergence_update(f, mu, dx, dt, W, M_s, bc_x, bc_y, eps_n=None):
    """One full conservative step of df/dt = -div(J), J = -M_s*q(f)*P_t.grad(mu_f).
    Returns (f_new, diagnostics dict with Jx/Jy/M components/D_CH density)."""
    if eps_n is None:
        eps_n = 1e-6 / W
    Mxx, Mxy, Myy = surface_mobility_tensor(f, dx, W, M_s, bc_x, bc_y, eps_n)
    Jx, Jy = surface_flux(mu, Mxx, Mxy, Myy, dx, bc_x, bc_y)
    div = surface_flux_divergence_conservative(Jx, Jy, dx, bc_x, bc_y)
    f_new = f - dt * div
    D_density = dissipation_density(Mxx, Mxy, Myy, mu, dx, bc_x, bc_y)
    return f_new, dict(Jx=Jx, Jy=Jy, Mxx=Mxx, Mxy=Mxy, Myy=Myy, div=div, D_density=D_density)
