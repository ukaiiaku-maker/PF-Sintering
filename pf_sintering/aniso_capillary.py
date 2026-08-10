"""Milestone 15F Section 7: anisotropic generalization of
capillary_stress.py's `capillary_force_endpoint_form`, exactly along the
path that module's own docstring anticipated (Section 20 note there):
"replacing [gamma_s*t] with the anisotropic xi(theta) would not change
any function signature here, only how the isotropic gamma_s scalars are
provided upstream."

Derivation: `capillary_force_endpoint_form` computes the net capillary
resultant transmitted through the particle free-surface arc as
`gamma_s*(t_end - t_start)`, the two ENDS of the arc's Cahn-Hoffman
vector `xi = gamma_s*t` (isotropic: xi reduces to a bare scaled tangent).
For an anisotropic surface energy gamma(theta), the generalized surface-
tension / Cahn-Hoffman vector at a point with tangent `t` and outward
(vapor-pointing) normal `n` is the standard result

    xi(theta) = gamma(theta)*t + gamma'(theta)*n

(already implemented for the OLDER model.compute_stress path via
model._aniso_gamma, reused here unchanged -- not re-derived). The
anisotropic endpoint resultant is then the SAME telescoping difference,
now with `xi` evaluated at each end using its OWN local tangent/normal
and crystal-orientation reference:

    F_cap_aniso = xi_end - xi_start

`xi_start` uses `t_start=top_particle_dir` (departing the top TJ);
`xi_end` uses `t_end=-bot_particle_dir` (arriving at the bottom TJ, the
same sign convention `capillary_force_endpoint_form` uses). Both
TJs' particle-side flanks belong to the SAME grain (the particle, e2)
for this project's "substrate"/"sinusoidal_substrate" two-body geometry
(confirmed: `trace_particle_arc` walks continuously from the top TJ to
the bottom TJ entirely along the particle's own free surface, never
touching the substrate branch) -- so `theta0=p.theta_grain[1]` for BOTH
ends, not the position-based v[0]<0 heuristic `model.compute_stress`
uses (that heuristic exists there only because that function's `flanks`
can mix particle- and substrate-side branches; that ambiguity does not
arise here).

Isotropic-limit check (Benchmark A, Section 8): at aniso_delta=0,
gamma(theta)=gamma_s and gamma'(theta)=0 identically for every theta, so
xi(theta)=gamma_s*t exactly and F_cap_aniso reduces EXACTLY (not just in
a limit) to `capillary_force_endpoint_form`'s isotropic result.
"""

from __future__ import annotations

import math

import numpy as np

from .model import _aniso_gamma, _vapor_normal


def _tangent_normal_theta(f, tj_xy, t, p):
    """Outward (vapor-pointing) unit normal for tangent `t` at `tj_xy`,
    via model._vapor_normal, and its polar angle."""
    n1 = np.array([-t[1], t[0]])
    n2 = -n1
    n = _vapor_normal(f, tj_xy, n1, n2, p)
    th = math.atan2(n[1], n[0])
    return n, th


def xi_vector(f, tj_xy, t, theta0, p):
    """Generalized (Cahn-Hoffman) surface-tension vector xi(theta) =
    gamma(theta)*t + gamma'(theta)*n at a point with unit tangent `t`,
    located at `tj_xy` (used only to determine the vapor-pointing normal
    direction), crystal-orientation reference `theta0`."""
    t = np.asarray(t, dtype=float)
    n, th = _tangent_normal_theta(f, tj_xy, t, p)
    gam, gp = _aniso_gamma(th, theta0, p)
    return gam * t + gp * n


def capillary_force_endpoint_form_aniso(f, tj_top_xy, tj_bottom_xy, top_particle_dir, bot_particle_dir, p):
    """Anisotropic F_cap_aniso = xi_end - xi_start. Requires
    p.use_aniso_surface and a built p.lut_psi/lut_a/lut_ap. Returns
    (Fx, Fy, xi_start, xi_end) -- the raw xi vectors are kept for
    diagnostics (Section 12's "anisotropic endpoint xi vectors")."""
    theta0 = float(p.theta_grain[1])
    t_start = np.asarray(top_particle_dir, dtype=float)
    t_end = -np.asarray(bot_particle_dir, dtype=float)
    xi_start = xi_vector(f, tj_top_xy, t_start, theta0, p)
    xi_end = xi_vector(f, tj_bottom_xy, t_end, theta0, p)
    F = xi_end - xi_start
    return float(F[0]), float(F[1]), xi_start, xi_end
