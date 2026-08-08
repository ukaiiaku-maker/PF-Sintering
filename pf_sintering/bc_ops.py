"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 12 Commit 1: per-axis boundary-condition operators.

`model.py`'s `grad`/`div`/`lap9` hard-code X (axis=1) periodic, Y (axis=0)
reflecting/no-flux -- confirmed by direct inspection (Milestone 10 Section
2) and left completely UNCHANGED here (not imported, not modified, not
even called): every legacy caller keeps its exact existing behavior.

This module reimplements the same family of operators with the boundary
condition selectable per axis (`"periodic"` or `"reflecting"`/`"no_flux"`,
synonyms), so the new sinusoidal-substrate coarsening campaign can use
X=reflecting (substrate-normal, a true domain edge) / Y=periodic (lateral,
the sinusoid's actual repeat direction) -- the physically correct pairing
identified in Milestone 10 Section 2, as opposed to that milestone's
reflection-symmetry-equivalence workaround.

`_shift(a, 1, axis=0, bc="reflecting")` reproduces `model.py`'s legacy `N`
construction exactly (`np.vstack([a[:1], a[:-1]])`); `_shift(a, -1, axis=0,
"reflecting")` reproduces legacy `S` exactly; `_shift(a, ±1, axis=1,
"periodic")` reproduces legacy `E`/`W` exactly -- so `lap9_bc(a, dx,
bc_x="periodic", bc_y="reflecting")` is verified bit-identical to
`model.lap9` (`tests/test_bc_ops.py`).

`grad_bc`/`div_bc` do NOT attempt to reproduce `model.py`'s legacy
one-sided boundary difference at the reflecting axis (that scheme is not
the discrete adjoint of anything and was never required to be for a
diagnostic like `surface_energy`) -- they instead use a symmetric
ghost-cell central difference at every reflecting boundary. Checked
directly (`tests/test_bc_ops.py`): this ghost-cell pair is NOT an exact
discrete adjoint under a reflecting BC (a true self-adjoint Neumann pair
needs boundary-specific quadrature weighting, e.g. a staggered/finite-
volume formulation, which this simple cell-centered pair does not
implement) -- `grad_bc`/`div_bc` are therefore used here only for
non-conservative diagnostic quantities (the interface normal, `grad(mu)`
feeding the tensor flux below) where a reasonable central-difference
gradient with correct per-axis BC semantics is all that is needed, never
for a step that must itself conserve mass.

`flux_divergence`, used by `surface_transport.py` for every actual
conservative update, differences FACE-CENTERED fluxes (not `div_bc` of a
cell-centered vector) so that mass conservation holds by an exact
combinatorial telescoping-sum argument independent of the flux's own
accuracy -- see its docstring. This is the property Section 9/11 of the
milestone actually require, and it does not depend on `grad_bc`/`div_bc`
being adjoint.
"""

from __future__ import annotations

import numpy as np

_PERIODIC = "periodic"
_REFLECTING = {"reflecting", "no_flux"}


def _shift(a, shift, axis, bc):
    """a shifted by `shift` cells along `axis`, either wrapped (periodic)
    or edge-replicated (reflecting/no_flux -- a zero-gradient ghost cell)."""
    if bc == _PERIODIC:
        return np.roll(a, shift, axis=axis)
    if bc in _REFLECTING:
        k = abs(shift)
        if shift > 0:
            edge = np.take(a, [0], axis=axis)
            pad = np.repeat(edge, k, axis=axis)
            interior = np.take(a, range(0, a.shape[axis] - k), axis=axis)
            return np.concatenate([pad, interior], axis=axis)
        else:
            edge = np.take(a, [a.shape[axis] - 1], axis=axis)
            pad = np.repeat(edge, k, axis=axis)
            interior = np.take(a, range(k, a.shape[axis]), axis=axis)
            return np.concatenate([interior, pad], axis=axis)
    raise ValueError(f"unknown bc {bc!r}")


def face_average(a, axis, bc):
    """Cell-to-face average: result[i] = 0.5*(a[i]+a[i+1]) along `axis`
    (the value on the "+axis" face of cell i, same convention flux_divergence
    expects). Public helper for surface_transport.py's cell-centered-tensor
    flux, which must be interpolated to faces before flux_divergence can
    guarantee exact mass conservation (see flux_divergence docstring)."""
    return 0.5 * (a + _shift(a, -1, axis, bc))


def lap9_bc(a, dx, bc_x=_PERIODIC, bc_y="reflecting"):
    """Same 9-point compact/rotated Laplacian stencil as model.lap9,
    generalized to per-axis BC. Reduces to model.lap9 exactly at the
    legacy default (bc_x=periodic, bc_y=reflecting)."""
    E = _shift(a, 1, 1, bc_x); W = _shift(a, -1, 1, bc_x)
    N = _shift(a, 1, 0, bc_y); S = _shift(a, -1, 0, bc_y)
    NE = _shift(N, 1, 1, bc_x); NW = _shift(N, -1, 1, bc_x)
    SE = _shift(S, 1, 1, bc_x); SW = _shift(S, -1, 1, bc_x)
    return (4 * (N + S + E + W) + NE + NW + SE + SW - 20 * a) / (6 * dx * dx)


def grad_bc(a, dx, bc_x=_PERIODIC, bc_y="reflecting"):
    """Symmetric (ghost-cell) central-difference gradient, per-axis BC.
    NOT bit-identical to model.grad's legacy one-sided boundary treatment
    (see module docstring). NOT the exact discrete adjoint of div_bc under
    a reflecting BC (see module docstring) -- use only for diagnostic
    gradients (interface normal, grad(mu)), never in a conservative step."""
    gx = (_shift(a, -1, 1, bc_x) - _shift(a, 1, 1, bc_x)) / (2 * dx)
    gy = (_shift(a, -1, 0, bc_y) - _shift(a, 1, 0, bc_y)) / (2 * dx)
    return gx, gy


def div_bc(vx, vy, dx, bc_x=_PERIODIC, bc_y="reflecting"):
    """Symmetric (ghost-cell) central-difference divergence, per-axis BC,
    Provided for completeness/diagnostics; NOT the exact discrete adjoint
    of grad_bc under a reflecting BC (see module docstring) -- the
    mass-conservative primitive is flux_divergence, not div_bc(grad_bc(...))."""
    dvx = (_shift(vx, -1, 1, bc_x) - _shift(vx, 1, 1, bc_x)) / (2 * dx)
    dvy = (_shift(vy, -1, 0, bc_y) - _shift(vy, 1, 0, bc_y)) / (2 * dx)
    return dvx + dvy


def face_flux_x(M, mu, dx, bc_x=_PERIODIC):
    """Face-centered x-flux J[i,j] = flux on the RIGHT face of cell
    (i,j), mobility face-averaged (periodic-style construction; the actual
    boundary treatment for a non-periodic axis is applied afterward, in
    flux_divergence, by zeroing the two domain-edge faces -- matching
    model.evolve_f's own Jx/Jy pattern exactly for the legacy default)."""
    Mx = 0.5 * (M + np.roll(M, -1, axis=1))
    return -Mx * (np.roll(mu, -1, axis=1) - mu) / dx


def face_flux_y(M, mu, dx):
    My = 0.5 * (M + np.roll(M, -1, axis=0))
    return -My * (np.roll(mu, -1, axis=0) - mu) / dx


def _axis_efflux(J_periodic_style, axis, bc):
    """Net per-cell efflux J[i]-J[i-1] along `axis`, where J is a
    face-centered flux built in periodic style (J[i] = flux on the
    "+axis" face of cell i, wrapping at the domain edge). For a periodic
    axis this telescopes exactly around the ring. For a reflecting/
    no_flux axis, the wrap-around face value is discarded and BOTH domain-
    edge faces are forced to exactly zero flux instead -- reproducing
    model.evolve_f's Jy[Ny-1]=0 / Jyd[0]=0 no-flux convention exactly
    (verified in tests/test_bc_ops.py), so the telescoping sum over the
    whole (finite, non-wrapping) axis is exactly zero here too."""
    if bc == _PERIODIC:
        return J_periodic_style - np.roll(J_periodic_style, 1, axis=axis)
    if bc in _REFLECTING:
        Jf = J_periodic_style.copy()
        J_left = np.zeros_like(J_periodic_style)
        if axis == 0:
            Jf[-1, :] = 0.0
            J_left[1:, :] = J_periodic_style[:-1, :]
        else:
            Jf[:, -1] = 0.0
            J_left[:, 1:] = J_periodic_style[:, :-1]
        return Jf - J_left
    raise ValueError(f"unknown bc {bc!r}")


def flux_divergence(Jx_face, Jy_face, dx, bc_x=_PERIODIC, bc_y="reflecting"):
    """Exact discrete divergence of a pair of face-centered fluxes
    (periodic-style construction, see face_flux_x/y), per-axis BC.
    sum(flux_divergence(...)) is exactly 0.0 for any bc combination
    (each axis's contribution telescopes to exactly zero independently --
    periodic wraps around a closed ring, reflecting/no_flux has both its
    domain-edge faces forced to exactly zero) -- this is a combinatorial
    identity, true regardless of how Jx_face/Jy_face were computed, which
    is what makes it the right primitive for a provably mass-conserving
    update (see module docstring)."""
    return (_axis_efflux(Jx_face, axis=1, bc=bc_x) + _axis_efflux(Jy_face, axis=0, bc=bc_y)) / dx
