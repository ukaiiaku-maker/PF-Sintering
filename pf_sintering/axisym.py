"""Milestone 16A: a SEPARATE, from-scratch axisymmetric (r,z) phase-field
surface-diffusion solver, for validating the Plateau-Rayleigh instability
against sharp-interface theory. Does NOT modify or wire into the
production Cartesian path (model.py, surface_transport.py) in any way.

Physical setup (Section 1/4 of the milestone): a solid-vapor interface
represented by f(r,z), axisymmetric about r=0 (no theta-dependence), with
the SAME calibrated bulk+gradient free-energy density used everywhere
else in this project (reused, not re-derived):

    psi(f, grad f) = (W_f/2)*f^2*(1-f)^2 + (k_f/2)*(f_r^2 + f_z^2)

but now the total free energy is the physically correct 3-D (surface-of-
revolution) integral, not a per-unit-depth Cartesian one:

    F = 2*pi * integral r*psi dr dz

Variational derivative (Section 4's exact formula, not an ad hoc 1/r
patch on the Cartesian mu):

    mu = dpsi/df - (1/r) d/dr[r*dpsi/d(f_r)] - d/dz[dpsi/d(f_z)]
       = W_f*f*(1-f)*(1-2f) - k_f*[ (1/r)*d/dr(r*f_r) + f_zz ]
       = mu0_bulk(f) - k_f*Laplacian_cylindrical(f)

(the SAME bulk term mu0_bulk as the Cartesian model -- it carries no
gradient/curvature dependence, so it is coordinate-system-independent by
construction; only the gradient term picks up the 1/r contribution).

Conservation law (Section 5): a SCALAR-mobility Cahn-Hilliard flux
J = -M_s*q(f)*grad(mu) (reusing the SAME q(f)=(12/W)f^2(1-f)^2
interface-localization weight the production module uses, and the SAME
integrated M_s_ref convention) with the CYLINDRICAL divergence

    df/dt = -div_cyl(J) = -(1/r) d(r*J_r)/dr - dJ_z/dz

This is a deliberately SIMPLER transport law than the production
Cartesian path's tangential-projected mobility tensor
(surface_transport.py's M_tensor=M_s*q(f)*P_t) -- a standard, still
mass-conservative, still dissipative scalar-mobility Cahn-Hilliard
choice, appropriate for this foundational geometric-correctness
validation (the axisymmetric-vs-Cartesian DISTINCTION is what is being
tested here, not a replication of every production refinement). See
the module report for the explicit discrete dissipation identity this
choice makes available.

Discretization (finite-volume, HALF-integer r-cell centers so the
r=0 axis is an EXACT cell face, not offset by half a grid cell):

    r_centers[i] = (i+0.5)*dr,  i=0..Nr-1
    r_faces[i]   = i*dr,        i=0..Nr   (r_faces[0]=0 exactly)

Radial flux is forced to exactly 0 at r_faces[0] (axis symmetry: an
axisymmetric smooth field has zero radial flux through r=0 by
construction) and at r_faces[Nr] (outer no-flux boundary -- the domain
must extend far enough into vapor that this is physically inert, a
design requirement of every benchmark below, not assumed). This gives
EXACT discrete conservation of V_f=2*pi*integral(r*f)dr dz: the
weighted divergence telescopes to a pure boundary term that vanishes
identically (see `axisym_volume`/tests).

z is periodic throughout (rod/ligament periodicity, matching every
benchmark in this milestone).
"""

from __future__ import annotations

import math

import numpy as np


def r_centers_faces(Nr, dr):
    r_c = (np.arange(Nr) + 0.5) * dr
    r_f = np.arange(Nr + 1) * dr
    return r_c, r_f


def face_grad_r(f, dr):
    """(Nz, Nr+1) gradient of f at r-faces. Exactly 0 at both r-boundary
    faces (axis symmetry df/dr=0 at r=0; no-flux at the outer edge)."""
    Nz, Nr = f.shape
    g = np.zeros((Nz, Nr + 1))
    g[:, 1:Nr] = (f[:, 1:] - f[:, :-1]) / dr
    return g


def face_grad_z(f, dz):
    """(Nz, Nr) gradient of f at z-faces (periodic): face[j] is the
    gradient between row j and row j+1 (row Nz-1 wraps to row 0)."""
    return (np.roll(f, -1, axis=0) - f) / dz


def div_cyl(Jr_face, Jz_face, r_c, r_f, dr, dz):
    """Exact finite-volume cylindrical divergence. Jr_face: (Nz,Nr+1),
    Jz_face: (Nz,Nr) (face[j] = flux between row j and row j+1, periodic).
    Telescopes exactly to a pure r-boundary term when integrated against
    r_c*dr (see module docstring) -- zero when Jr_face's boundary
    columns are zero."""
    div_r = (r_f[None, 1:] * Jr_face[:, 1:] - r_f[None, :-1] * Jr_face[:, :-1]) / (r_c[None, :] * dr)
    div_z = (Jz_face - np.roll(Jz_face, 1, axis=0)) / dz
    return div_r + div_z


def axisym_laplacian(f, dr, dz, r_c, r_f):
    """Cylindrical (axisymmetric) Laplacian f_rr + f_r/r + f_zz, via
    div_cyl(grad(f)) -- the exact discrete adjoint structure needed for
    mu's gradient term (Section 4)."""
    gr = face_grad_r(f, dr)
    gz = face_grad_z(f, dz)
    return div_cyl(gr, gz, r_c, r_f, dr, dz)


def axisym_mu(f, p, dr, dz, r_c, r_f):
    """mu = W_f*f*(1-f)*(1-2f) - k_f*Laplacian_cylindrical(f). Reuses
    the SAME calibrated p.W_f, p.k_f the Cartesian model uses (the bulk
    term is coordinate-independent; k_f is the same physical gradient-
    energy coefficient, since locally, for W<<R0, the profile shape is
    unaffected by the global curvature of the r-weighting to leading
    order)."""
    mu0 = p.W_f * f * (1 - f) * (1 - 2 * f)
    return mu0 - p.k_f * axisym_laplacian(f, dr, dz, r_c, r_f)


def axisym_q(f, W):
    """SAME interface-localization weight the Cartesian module uses."""
    return (12.0 / W) * f * f * (1 - f) ** 2


def axisym_volume(f, r_c, dr, dz):
    """V_f = 2*pi*integral(r*f)dr dz -- the conserved quantity (Section
    5), NOT sum(f)*dx^2."""
    return 2 * math.pi * float(np.sum(r_c[None, :] * f)) * dr * dz


def axisym_free_energy(f, p, dr, dz, r_c, r_f):
    """F = 2*pi*integral(r*psi)dr dz, using the EXACT discrete adjoint
    form -0.5*k_f*f*Laplacian_cyl(f) for the gradient term (matching
    ch_exact_energy.py's own pattern for the Cartesian case) so this is
    the exact energy whose variational derivative is axisym_mu (verified
    numerically in tests, not just asserted)."""
    bulk = 0.5 * p.W_f * f * f * (1 - f) ** 2
    grad_term = -0.5 * p.k_f * f * axisym_laplacian(f, dr, dz, r_c, r_f)
    return 2 * math.pi * float(np.sum(r_c[None, :] * (bulk + grad_term))) * dr * dz


def axisym_surface_diffusion_step(f, p, dr, dz, r_c, r_f, dt, M_s, W):
    """One conservative step: J = -M_s*q_face*grad(mu), df/dt=-div_cyl(J).
    q is face-averaged (simple arithmetic mean of the two adjacent cell
    values) to build a face-centered scalar mobility, matching standard
    finite-volume practice. Returns (f_new, mu, diagnostics)."""
    mu = axisym_mu(f, p, dr, dz, r_c, r_f)
    q = axisym_q(f, W)

    q_face_r = np.zeros((f.shape[0], f.shape[1] + 1))
    q_face_r[:, 1:-1] = 0.5 * (q[:, 1:] + q[:, :-1])
    mu_gr = face_grad_r(mu, dr)
    Jr_face = -M_s * q_face_r * mu_gr

    q_face_z = 0.5 * (q + np.roll(q, -1, axis=0))
    mu_gz = face_grad_z(mu, dz)
    Jz_face = -M_s * q_face_z * mu_gz

    div = div_cyl(Jr_face, Jz_face, r_c, r_f, dr, dz)
    f_new = f - dt * div
    # Section 6: dF/dt = 2*pi*sum(r_c*mu*df/dt)*dr*dz by the CHAIN RULE
    # (mu is defined as delta F/delta f, i.e. exactly the quantity that
    # pairs with df/dt this way -- see module docstring derivation).
    # df/dt=-div here, so this is a direct, unambiguous discrete
    # evaluation -- verified against an independent finite-difference
    # [F(t+dt)-F(t)]/dt in tests/scripts, not assumed.
    Fdot = -2 * math.pi * float(np.sum(r_c[None, :] * mu * div)) * dr * dz
    diag = dict(mu=mu, Jr_face=Jr_face, Jz_face=Jz_face, Fdot_chain=Fdot)
    return f_new, mu, diag


# ---------------------------------------------------------------------------
# Milestone 16A Sections 11-13: minimal two-grain (GB) extension.
#
# A deliberately SIMPLER GB kinetic law than the production Cartesian
# path's tangent-cone-constrained update (pf_sintering/constrained_eta.py)
# -- direct (unconstrained) Allen-Cahn relaxation of the SAME complete
# variational derivative g_i (Milestone 12B's structural_thermodynamic_
# force, cylindrically generalized here), followed by a simple clip/
# renormalize projection onto e_i>=0, sum(e_i)<=f. This is a scope
# reduction from the production machinery's tangent-cone projection,
# made explicitly and documented (not silently): it is sufficient to
# test the QUALITATIVE GB-destabilization physics this milestone's
# Sections 11-13 need (does a GB groove shift the neutral wavelength,
# does a lower dihedral angle pinch off faster), not to reproduce every
# refinement of the production Cartesian GB kinetics.
# ---------------------------------------------------------------------------

def axisym_mu_f_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f):
    """mu for the CONSERVED f field including the eta-f coupling term
    (matching model.py's mu0 bulk term exactly, cylindrically-weighted
    gradient part as in axisym_mu)."""
    fb = np.clip(f, 0.0, 1.0)
    eta2 = e1 * e1 + e2 * e2
    mu0 = p.W_f * f * (1 - f) * (1 - 2 * f) - Wc * eta2 * (1 - fb)
    return mu0 - p.k_f * axisym_laplacian(f, dr, dz, r_c, r_f)


def axisym_g_eta(e_i, f, Wc, p, dr, dz, r_c, r_f):
    """g_i = delta F/delta eta_i = -k_eta*Laplacian_cyl(e_i) +
    2*Wc*e_i*(f^2/2-f) -- the cylindrical generalization of
    constrained_eta.structural_thermodynamic_force."""
    coupling = 2.0 * Wc * e_i * (0.5 * f * f - f)
    return coupling - p.k_eta * axisym_laplacian(e_i, dr, dz, r_c, r_f)


def axisym_reproject(f, e1, e2):
    fb = np.clip(f, 0.0, 1.0)
    e1 = np.clip(e1, 0.0, fb)
    e2 = np.clip(e2, 0.0, fb)
    sm = e1 + e2
    over = sm > fb + 1e-12
    if np.any(over):
        sc = fb[over] / sm[over]
        e1[over] *= sc
        e2[over] *= sc
    void = fb <= 0.005
    e1[void] = 0.0
    e2[void] = 0.0
    return e1, e2


def axisym_gb_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W):
    """One FULL step: conservative f update (surface diffusion, now
    using the eta-coupled mu) + Allen-Cahn eta relaxation (sum-
    preserving direction, simple clip/renormalize projection)."""
    mu = axisym_mu_f_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
    q = axisym_q(f, W)

    q_face_r = np.zeros((f.shape[0], f.shape[1] + 1))
    q_face_r[:, 1:-1] = 0.5 * (q[:, 1:] + q[:, :-1])
    Jr_face = -M_s * q_face_r * face_grad_r(mu, dr)
    q_face_z = 0.5 * (q + np.roll(q, -1, axis=0))
    Jz_face = -M_s * q_face_z * face_grad_z(mu, dz)
    div = div_cyl(Jr_face, Jz_face, r_c, r_f, dr, dz)
    f_new = f - dt * div

    g1 = axisym_g_eta(e1, f, Wc, p, dr, dz, r_c, r_f)
    g2 = axisym_g_eta(e2, f, Wc, p, dr, dz, r_c, r_f)
    lam = 0.5 * (g1 + g2)
    e1_new = e1 - dt * M_eta * (g1 - lam)
    e2_new = e2 - dt * M_eta * (g2 - lam)
    e1_new, e2_new = axisym_reproject(f_new, e1_new, e2_new)
    return f_new, e1_new, e2_new, mu
