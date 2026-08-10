"""Milestone 15E Section 13: diffuse configurational-force (Eshelby/
Korteweg) diagnostic, derived directly from the SAME unified free energy
`F[f,eta]` this project's coupled dynamics actually uses -- not from an
arbitrarily chosen local-curvature window.

Derivation
----------
The raw local free-energy density this project's `energy_ledger` sums is

    psi_raw = (W_f/2)*f^2*(1-f)^2 + (k_f/2)|grad f|^2      [[E_surface]]
            + Wc*(e1^2+e2^2)*(f^2/2-f)                      [[E_coupling]]
            + (k_eta/2)*(|grad e1|^2+|grad e2|^2)           [[E_GB]]

but `psi_raw` does NOT vanish in bulk solid even far from any real grain
boundary: at f=1, single grain (e1=1,e2=0), E_coupling=-0.5*Wc (nonzero!)
-- this is exactly the background `gb_excess_energy` (Milestone 15B
Section 11, `scripts/m15_gb_surface_rate_competition.py`) was written to
subtract off for the SAME reason: an isolated planar surface or a plain
single-grain bulk region has NO physical GB excess, so any local energy
density used to derive a configurational (material) force -- which must
vanish identically away from a genuine defect -- has to use the same
background-subtracted, single-grain-referenced excess density, not
psi_raw. Empirically confirmed: using psi_raw here made the "isolated
planar interface" contour force grow ~linearly with contour radius
(Milestone 15E benchmark run) instead of vanishing, tracing exactly to
this nonzero single-grain background.

Following `gb_excess_energy`'s own derivation (background = the same f
profile with a single grain e_single=f, eta2=f^2) and re-expressing its
already-validated discrete form in continuum-gradient language:

    E_coupling_excess = Wc*e1*e2*f*(2-f)      (vanishes wherever e1*e2=0)
    E_GB_excess       = (k_eta/2)*(|grad e1|^2+|grad e2|^2) - (k_eta/2)|grad f|^2

(the second line is the continuum form of `gb_excess_energy`'s
`excess_grad = -0.5*k_eta*(grad_term-bg_grad_term)`, via integration by
parts: -e*lap(e) <-> |grad e|^2 up to a boundary flux). The full
excess/physical local energy density used for the Eshelby tensor is
therefore

    psi = (W_f/2)*f^2*(1-f)^2 + (k_f/2)|grad f|^2
        + ((k_f-k_eta)/2)... collected below ...

which collects to (verified to reduce EXACTLY to the pure surface term
when e1=f,e2=0, i.e. a genuine single-grain surface, independent of
k_eta, as required for benchmark A to be meaningful):

    psi = (W_f/2)*f^2*(1-f)^2 + (k_f/2)|grad f|^2 - (k_eta/2)|grad f|^2
        + (k_eta/2)*(|grad e1|^2+|grad e2|^2) + Wc*e1*e2*f*(2-f)

IMPORTANT correction: e1+e2=f is an EXACT constraint in this project's
field representation (Milestone 15B), so {f,e1,e2} are NOT three
independent fields -- treating them as independent (as an earlier
version of this module did) silently drops a real cross-gradient term
and was empirically caught by benchmark C: the resulting F_conf had the
right order of magnitude but did not flip sign between a
narrower-than-equilibrium and a wider-than-equilibrium wedge the way the
true Young-Herring residual F_TJ does. Eliminating e2=f-e1 and treating
(f,e1) as the two independent fields gives, after substitution,

    psi = (W_f/2)f^2(1-f)^2 + (k_f/2)|grad f|^2
        + k_eta*|grad e1|^2 - k_eta*(grad f . grad e1)
        + Wc*e1*(f-e1)*f*(2-f)

(the cross term -k_eta*(grad f . grad e1) is new; it vanishes identically
under the earlier, incorrect independent-fields treatment). Its Eshelby
tensor (still symmetric) is

    C_ij = psi*delta_ij - k_f*(df/dx_i)(df/dx_j) - 2*k_eta*(de1/dx_i)(de1/dx_j)
                        + k_eta*[(df/dx_i)(de1/dx_j) + (de1/dx_i)(df/dx_j)]

Both this and the special case e1=f (single grain, e2=0) reduce to the
same, expected pure-surface-term tensor `C_ij=psi*delta_ij-k_f*df_i*df_j`
with `psi=(W_f/2)f^2(1-f)^2+(k_f/2)|grad f|^2` -- confirming benchmark A
is unaffected by the fix, only the two-grain (wedge/GB) case is.

For a region satisfying the bulk Euler-Lagrange stationarity condition,
`div(C)=0` except at singular points/defects enclosed by the integration
contour; the configurational (material) force driving such a defect is
the standard Eshelby contour integral

    F_conf = -oint_Gamma C . n ds     (n = outward normal)

matching the classical Peach-Koehler/Eshelby sign convention. This sign
(and the overall normalization) is verified, not merely assumed, against
the independently-known Young-Herring residual `F_TJ` on a controlled
off-equilibrium wedge (Section 14's benchmark C). Because `psi` is built
from the field configuration's ACTUAL local gradients (not assumed to
solve any particular Euler-Lagrange equation), a meaningful
(radius-independent) benchmark additionally requires the tested field
configuration to be a genuine near-equilibrium state of the real coupled
dynamics (`evolve_f`/`evolve_eta`/`reproject`) -- an arbitrarily
hand-constructed profile (e.g. a bare guessed tanh) is generally NOT
such a state and will show a spurious, non-plateauing residual driving
force even after the background-subtraction fix above. See
`scripts/m15e_conf_force_benchmarks.py`, which relaxes every synthetic
test profile through the real dynamics before evaluating F_conf.
"""

from __future__ import annotations

import math

import numpy as np

from .bc_ops import grad_bc
from .constrained_eta import local_wc
from .model import Sink


def eshelby_tensor(f, e1, e2, p, s=None, bc_x="reflecting", bc_y="periodic"):
    """C_xx, C_xy(=C_yx), C_yy, psi -- all (Ny,Nx) arrays. `s` (a Sink)
    is only used by `local_wc`'s effective_gamma branch; pass a permanently
    inactive Sink (the convention used throughout this project's
    diagnostic scripts) if the local GB energy should not vary with any
    hazard/sink state."""
    if s is None:
        s = Sink(threshold=math.inf)
    e3 = np.zeros_like(f)
    Wc = local_wc(f, e1, e2, e3, s, p)
    surface_bulk = 0.5 * p.W_f * f * f * (1 - f) ** 2
    excess_coupling = Wc * e1 * (f - e1) * f * (2.0 - f)  # e2 eliminated via e2=f-e1

    gfx, gfy = grad_bc(f, p.dx, bc_x=bc_x, bc_y=bc_y)
    g1x, g1y = grad_bc(e1, p.dx, bc_x=bc_x, bc_y=bc_y)

    cross = gfx * g1x + gfy * g1y
    grad_energy = 0.5 * p.k_f * (gfx ** 2 + gfy ** 2) + p.k_eta * (g1x ** 2 + g1y ** 2) - p.k_eta * cross
    psi = surface_bulk + excess_coupling + grad_energy

    Cxx = psi - p.k_f * gfx * gfx - 2 * p.k_eta * g1x * g1x + p.k_eta * (gfx * g1x + g1x * gfx)
    Cyy = psi - p.k_f * gfy * gfy - 2 * p.k_eta * g1y * g1y + p.k_eta * (gfy * g1y + g1y * gfy)
    Cxy = -p.k_f * gfx * gfy - 2 * p.k_eta * g1x * g1y + p.k_eta * (gfx * g1y + g1x * gfy)
    return Cxx, Cxy, Cyy, psi


def _sample_bilinear(field, xs, ys, p):
    from scipy.ndimage import map_coordinates
    ci = np.asarray(xs) / p.dx - 1.0
    ri = np.asarray(ys) / p.dx - 1.0
    ci = np.clip(ci, 0.0, p.Nx - 1.0)
    ri = np.clip(ri, 0.0, p.Ny - 1.0)
    return map_coordinates(field, [ri, ci], order=1, mode="nearest")


def configurational_force_contour(Cxx, Cxy, Cyy, p, center_xy, radius, n_theta=720):
    """F_conf = -oint C.n ds around a circle of `radius` centered at
    `center_xy`, via the trapezoid rule on `n_theta` samples (bilinearly
    interpolated). Returns (Fx, Fy)."""
    thetas = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    xs = center_xy[0] + radius * np.cos(thetas)
    ys = center_xy[1] + radius * np.sin(thetas)
    nx, ny = np.cos(thetas), np.sin(thetas)  # outward normal on a circle
    cxx = _sample_bilinear(Cxx, xs, ys, p)
    cxy = _sample_bilinear(Cxy, xs, ys, p)
    cyy = _sample_bilinear(Cyy, xs, ys, p)
    tx = cxx * nx + cxy * ny
    ty = cxy * nx + cyy * ny
    ds = radius * (2 * math.pi / n_theta)
    Fx = -float(np.sum(tx) * ds)
    Fy = -float(np.sum(ty) * ds)
    return Fx, Fy


def configurational_force_radius_scan(f, e1, e2, p, center_xy, radii, s=None,
                                       bc_x="reflecting", bc_y="periodic", n_theta=720):
    Cxx, Cxy, Cyy, psi = eshelby_tensor(f, e1, e2, p, s=s, bc_x=bc_x, bc_y=bc_y)
    out = {}
    for r in radii:
        Fx, Fy = configurational_force_contour(Cxx, Cxy, Cyy, p, center_xy, r, n_theta=n_theta)
        out[r] = (Fx, Fy)
    return out
