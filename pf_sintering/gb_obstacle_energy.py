"""Milestone 14G: single-free-energy obstacle-problem GB calibration.

Milestone 14E/14F found the DECLARED `gamma_gb` parameter and the eta
profile's actual implemented energy disagreed, and traced part of the gap
to an incorrect assumption: that the natural/equilibrium profile of the
constrained two-grain structural free energy is a TANH. It is not. This
module derives, and exposes, the correct equilibrium profile and the
resulting exact coefficient calibration.

## Obstacle reduction (Section 1)

At `f=1`, `eta1+eta2=1`, write `phi=eta2`, `eta1=1-phi`. The existing
structural free energy density `Wc*(eta1^2+eta2^2)*(f^2/2-f) +
(k_eta/2)*(|grad eta1|^2+|grad eta2|^2)` reduces (dropping the additive
constant `-0.5*Wc`, and using `|grad eta1|^2+|grad eta2|^2 =
2*|grad phi|^2`) to

    F_GB[phi] = integral[ k_eta*(phi')^2 + Wc*phi*(1-phi) ] dV

subject to the box constraint `0<=phi<=1` inherited from `0<=eta_i<=f`
(the tangent-cone-projected update, `constrained_eta.py`, enforces this
constraint directly, not via a smooth double well) -- an OBSTACLE
problem, not a standard Allen-Cahn double-well interface.

## Equilibrium profile (Section 2)

Inside the active interval (away from the obstacle), the Euler-Lagrange
equation is `-2*k_eta*phi'' + Wc*(1-2*phi) = 0`. With `ell =
sqrt(k_eta/Wc)`, substituting `psi=phi-0.5` gives the simple-harmonic
equation `psi'' = -psi/ell^2`, whose solution matching `phi(0)=0.5` (by
symmetry) is `psi(x)=A*sin(x/ell)`. The obstacle (free) boundary is where
`phi` first reaches 0 or 1 WITH C^1 (smooth-pasting) matching, i.e.
`phi'=0` there too: `phi(x) = 0.5*(1+sin(x/ell))`,
`phi'(x)=0.5/ell*cos(x/ell)`, both vanish/saturate exactly at
`x=+-pi*ell/2` -- confirming `A=0.5` and the compact-support width
`delta_GB = pi*ell`. Outside this interval `phi` is exactly 0 or 1
(zero energy density there, matching the obstacle solution exactly, not
just asymptotically as a tanh would).

Direct integration (verified in `tests/test_gb_obstacle_energy.py` both
symbolically-by-hand, reproduced in the derivation below, and numerically
against the exact analytic profile) gives, over one full active interval:

    integral(phi')^2 dx = pi/(8*ell)
    gamma_GB_PF = integral[k_eta*(phi')^2 + Wc*phi*(1-phi)] dx
                = (pi/4)*sqrt(k_eta*Wc)

(both by evaluating `k_eta*(phi')^2 = Wc*phi*(1-phi) = 0.25*Wc*cos^2(x/ell)`
identically over the active interval -- the two energy contributions are
EQUAL at every point, a standard equipartition feature of this class of
Euler-Lagrange solution).

## Physical coefficient calibration (Section 6)

Preferred convention: `delta_GB = W_GB` (the compact-support width equals
the SAME physical interface width `W` already used for the free-surface
`gamma_s`), `gamma_GB_PF = gamma_GB_physical` (the target physical GB
energy). Solving the two equations above for `k_eta`, `Wc`:

    Wc    = 4*gamma_GB/W_GB
    k_eta = 4*gamma_GB*W_GB/pi^2

(replacing the old, tanh-profile-derived `Wc=36*gamma_gb/W`,
`k_eta=3*gamma_gb*W`).

## Mobility mapping (Section 10)

The existing two-grain reduction of `constrained_eta.py`'s constrained
update, `d(eta1)/dt=-(M_eta/2)*(g1-g2)`, is EXACTLY `phi_dot =
-(M_eta/2)*delta_F_GB/delta_phi` (verified by direct substitution, see
module tests) -- i.e. the mobility mapping needs no change to the
integrator itself, only to how `M_eta` is interpreted physically. For a
traveling front driven by a small bulk bias (the planar driven-boundary
test), the standard translational-mode solvability projection gives
`v_GB = (M_eta/(2*I))*Delta_g` with `I=integral(phi')^2 dx=pi/(8*ell)`,
so

    M_GB = (M_eta/2)/I = 4*M_eta*ell/pi = 4*M_eta*delta_GB/pi^2

and, for the `delta_GB=W_GB` convention,

    M_GB = 4*M_eta*W_GB/pi^2   <=>   M_eta = pi^2*M_GB/(4*W_GB).
"""

from __future__ import annotations

import math

import numpy as np

TANH_PROFILE_EXCESS_FACTOR = 19.0
"""Renamed from Milestone 14F's `GB_ENERGY_CALIBRATION_FACTOR`. This is
the excess free energy of an IMPOSED TANH profile evaluated with the old
`Wc=36*gamma_declared/W`, `k_eta=3*gamma_declared*W` coefficients
(`18*gamma_declared` bulk + `1*gamma_declared` gradient, both exact and
W-independent -- Milestone 14E/14F's own derivation, still numerically
correct as a statement about the TANH profile specifically). It is NOT a
physical GB energy calibration: the tanh is not the equilibrium profile
of the actual constrained (obstacle) free energy (see module docstring).
Retained ONLY for regression/documentation -- do not use to parameterize
GB thermodynamics. Milestone 14G's `gb_obstacle_coefficients` (the
correct, obstacle-equilibrium-derived calibration) supersedes it."""


def gb_obstacle_coefficients(gamma_gb_physical: float, gb_width_physical: float,
                              width_convention: str = "compact_support") -> dict:
    """The single source of truth for `k_eta`/`Wc` given a PHYSICAL GB
    energy and width, replacing the old `k_eta=3*gamma_gb*W`,
    `Wc=36*gamma_gb/W`. `width_convention="compact_support"` (the only
    one implemented) takes `gb_width_physical` as the obstacle profile's
    full compact-support width `delta_GB` (Section 6's preferred
    convention: `delta_GB=W_GB`, i.e. the same interface width already
    used for `gamma_s`). Returns `k_eta`, `Wc`, and `W_cpl_f` (an alias
    for `Wc`, matching `Params.W_cpl_f`'s existing field name)."""
    if width_convention != "compact_support":
        raise ValueError(f"unknown width_convention {width_convention!r}")
    Wc = 4.0 * gamma_gb_physical / gb_width_physical
    k_eta = 4.0 * gamma_gb_physical * gb_width_physical / math.pi ** 2
    return dict(k_eta=k_eta, Wc=Wc, W_cpl_f=Wc)


def obstacle_ell(k_eta: float, Wc: float) -> float:
    return math.sqrt(k_eta / Wc)


def obstacle_compact_width(k_eta: float, Wc: float) -> float:
    return math.pi * obstacle_ell(k_eta, Wc)


def obstacle_gamma_gb(k_eta: float, Wc: float) -> float:
    return (math.pi / 4.0) * math.sqrt(k_eta * Wc)


def obstacle_grad_sq_integral(ell: float) -> float:
    """`integral(phi')^2 dx` over one full active interval."""
    return math.pi / (8.0 * ell)


def obstacle_profile(x, ell: float, x0: float = 0.0):
    """`phi(x)` for the equilibrium obstacle GB profile centered at `x0`,
    active interval `|x-x0|<=pi*ell/2`, saturating to 0/1 outside."""
    x = np.asarray(x, dtype=float)
    xi = (x - x0) / ell
    phi = 0.5 * (1.0 + np.sin(np.clip(xi, -math.pi / 2, math.pi / 2)))
    phi = np.where(xi < -math.pi / 2, 0.0, phi)
    phi = np.where(xi > math.pi / 2, 1.0, phi)
    return phi


def m_gb_from_m_eta(M_eta: float, gb_width_physical: float) -> float:
    """`M_GB = 4*M_eta*W_GB/pi^2` (the `delta_GB=W_GB` convention)."""
    return 4.0 * M_eta * gb_width_physical / math.pi ** 2


def m_eta_from_m_gb(M_gb: float, gb_width_physical: float) -> float:
    """Inverse of `m_gb_from_m_eta`: `M_eta = pi^2*M_GB/(4*W_GB)`."""
    return math.pi ** 2 * M_gb / (4.0 * gb_width_physical)
