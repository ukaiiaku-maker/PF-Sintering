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


def face_grad_z(f, dz, bc_z="periodic"):
    """(Nz, Nr) gradient of f at z-faces: face[j] is the gradient
    between row j and row j+1. `bc_z="periodic"` (default, unchanged
    behavior): row Nz-1 wraps to row 0. `bc_z="noflux"` (Milestone 16C
    Section 11's capped/two-substrate topology): the wrap face (index
    Nz-1, "above" the last row / "below" the first) is forced to
    exactly 0 -- div_cyl's np.roll telescoping then gives an EXACT
    finite-volume no-flux boundary at both z-ends with no other change
    needed (div_cyl itself is unmodified; it only ever consumes
    whatever Jz_face values it is given)."""
    g = (np.roll(f, -1, axis=0) - f) / dz
    if bc_z == "noflux":
        g[-1, :] = 0.0
    elif bc_z != "periodic":
        raise ValueError(f"unknown bc_z {bc_z!r}")
    return g


def div_cyl(Jr_face, Jz_face, r_c, r_f, dr, dz):
    """Exact finite-volume cylindrical divergence. Jr_face: (Nz,Nr+1),
    Jz_face: (Nz,Nr) (face[j] = flux between row j and row j+1, periodic).
    Telescopes exactly to a pure r-boundary term when integrated against
    r_c*dr (see module docstring) -- zero when Jr_face's boundary
    columns are zero."""
    div_r = (r_f[None, 1:] * Jr_face[:, 1:] - r_f[None, :-1] * Jr_face[:, :-1]) / (r_c[None, :] * dr)
    div_z = (Jz_face - np.roll(Jz_face, 1, axis=0)) / dz
    return div_r + div_z


def axisym_laplacian(f, dr, dz, r_c, r_f, bc_z="periodic"):
    """Cylindrical (axisymmetric) Laplacian f_rr + f_r/r + f_zz, via
    div_cyl(grad(f)) -- the exact discrete adjoint structure needed for
    mu's gradient term (Section 4). `bc_z` (Milestone 16C Section 11):
    "periodic" (default, unchanged) or "noflux" (capped/two-substrate
    topology, see face_grad_z)."""
    gr = face_grad_r(f, dr)
    gz = face_grad_z(f, dz, bc_z=bc_z)
    return div_cyl(gr, gz, r_c, r_f, dr, dz)


def axisym_mu(f, p, dr, dz, r_c, r_f, bc_z="periodic"):
    """mu = W_f*f*(1-f)*(1-2f) - k_f*Laplacian_cylindrical(f). Reuses
    the SAME calibrated p.W_f, p.k_f the Cartesian model uses (the bulk
    term is coordinate-independent; k_f is the same physical gradient-
    energy coefficient, since locally, for W<<R0, the profile shape is
    unaffected by the global curvature of the r-weighting to leading
    order)."""
    mu0 = p.W_f * f * (1 - f) * (1 - 2 * f)
    return mu0 - p.k_f * axisym_laplacian(f, dr, dz, r_c, r_f, bc_z=bc_z)


def axisym_q(f, W):
    """SAME interface-localization weight the Cartesian module uses."""
    return (12.0 / W) * f * f * (1 - f) ** 2


def axisym_volume(f, r_c, dr, dz):
    """V_f = 2*pi*integral(r*f)dr dz -- the conserved quantity (Section
    5), NOT sum(f)*dx^2."""
    return 2 * math.pi * float(np.sum(r_c[None, :] * f)) * dr * dz


def axisym_free_energy(f, p, dr, dz, r_c, r_f, bc_z="periodic"):
    """F = 2*pi*integral(r*psi)dr dz, using the EXACT discrete adjoint
    form -0.5*k_f*f*Laplacian_cyl(f) for the gradient term (matching
    ch_exact_energy.py's own pattern for the Cartesian case) so this is
    the exact energy whose variational derivative is axisym_mu (verified
    numerically in tests, not just asserted)."""
    bulk = 0.5 * p.W_f * f * f * (1 - f) ** 2
    grad_term = -0.5 * p.k_f * f * axisym_laplacian(f, dr, dz, r_c, r_f, bc_z=bc_z)
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

def axisym_mu_f_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f, bc_z="periodic"):
    """mu for the CONSERVED f field including the eta-f coupling term
    (matching model.py's mu0 bulk term exactly, cylindrically-weighted
    gradient part as in axisym_mu). `bc_z` (Milestone 16G): "periodic"
    (default, unchanged) or "noflux" (capped/particle-on-asperity
    topology, see face_grad_z) -- threaded straight through to the
    gradient-energy Laplacian, matching axisym_mu's own pattern."""
    fb = np.clip(f, 0.0, 1.0)
    eta2 = e1 * e1 + e2 * e2
    mu0 = p.W_f * f * (1 - f) * (1 - 2 * f) - Wc * eta2 * (1 - fb)
    return mu0 - p.k_f * axisym_laplacian(f, dr, dz, r_c, r_f, bc_z=bc_z)


def axisym_g_eta(e_i, f, Wc, p, dr, dz, r_c, r_f, bc_z="periodic"):
    """g_i = delta F/delta eta_i = -k_eta*Laplacian_cyl(e_i) +
    2*Wc*e_i*(f^2/2-f) -- the cylindrical generalization of
    constrained_eta.structural_thermodynamic_force. `bc_z` (Milestone
    16G): threaded through to the Laplacian, same convention as
    axisym_mu_f_gb."""
    coupling = 2.0 * Wc * e_i * (0.5 * f * f - f)
    return coupling - p.k_eta * axisym_laplacian(e_i, dr, dz, r_c, r_f, bc_z=bc_z)


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


# ---------------------------------------------------------------------------
# Milestone 16B Sections 3-6: PRODUCTION-QUALITY (face-projected,
# tangentially-projected) axisymmetric surface transport, replacing
# M16A's scalar-mobility simplification. Mirrors
# surface_transport._face_projected_mobility_and_flux/
# surface_flux_face_projected EXACTLY (same q(f), same M_s convention,
# same n/P_t/grad(mu) "all constructed at the same face" design
# principle) -- the only physical difference is evaluating the
# divergence with the CYLINDRICAL (r-weighted) conservation law instead
# of the Cartesian one.
# ---------------------------------------------------------------------------

def axisym_face_gradient_r(a, dr, dz):
    """Both (gr,gz) AT the r-face between column i and i+1 (matching
    bc_ops.face_gradient_x's construction: the r-derivative is the exact
    one-sided difference across that face; the z-derivative is each
    neighboring column's own centered z-derivative, averaged onto the
    face). Interior faces 1..Nr-1 only; boundary faces (r=0 axis, outer
    edge) forced to (0,0), matching the no-flux BC."""
    Nz, Nr = a.shape
    gr_face = np.zeros((Nz, Nr + 1))
    gr_face[:, 1:Nr] = (a[:, 1:] - a[:, :-1]) / dr
    gz_cell = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2 * dz)
    gz_face = np.zeros((Nz, Nr + 1))
    gz_face[:, 1:Nr] = 0.5 * (gz_cell[:, :-1] + gz_cell[:, 1:])
    return gr_face, gz_face


def axisym_face_gradient_z(a, dr, dz, bc_z="periodic"):
    """Both (gr,gz) AT the z-face between row j and j+1 (periodic in z
    by default). z-derivative: exact one-sided difference across the
    face. r-derivative: each neighboring row's own centered r-
    derivative (one-sided at the two r-boundaries, matching a natural
    Neumann extrapolation there), averaged onto the face.
    `bc_z="noflux"` (Milestone 16C Section 11): the wrap face (index
    Nz-1) is forced to (0,0), matching face_grad_z's no-flux BC."""
    Nz, Nr = a.shape
    gz_face = (np.roll(a, -1, axis=0) - a) / dz
    gr_cell = np.zeros_like(a)
    if Nr > 2:
        gr_cell[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2 * dr)
    gr_cell[:, 0] = (a[:, 1] - a[:, 0]) / dr if Nr > 1 else 0.0
    gr_cell[:, -1] = (a[:, -1] - a[:, -2]) / dr if Nr > 1 else 0.0
    gr_face = 0.5 * (gr_cell + np.roll(gr_cell, -1, axis=0))
    if bc_z == "noflux":
        gz_face[-1, :] = 0.0
        gr_face[-1, :] = 0.0
    elif bc_z != "periodic":
        raise ValueError(f"unknown bc_z {bc_z!r}")
    return gr_face, gz_face


def axisym_face_projected_flux(f, mu, dr, dz, W, M_s, eps_n=None, bc_z="periodic"):
    """Axisymmetric equivalent of surface_transport.
    surface_flux_face_projected: n/P_t/grad(mu) all constructed AT the
    same face, SAME q(f)=(12/W)*f^2(1-f)^2 and M_s convention as the
    Cartesian production path. Returns face-centered Jr_face (Nz,Nr+1)
    and Jz_face (Nz,Nr) -- the r- and z-COMPONENTS of the flux vector
    at their own respective face families, matching the Cartesian
    Jx_face/Jy_face convention exactly (only the "own" component is
    used by the conservative divergence; both components are returned
    per face family for the tangentiality/dissipation diagnostics).
    `bc_z="noflux"` (Milestone 16C Section 11, capped/two-substrate
    topology): Jz_face's wrap entry is forced to exactly 0, matching
    face_grad_z/axisym_face_gradient_z's no-flux convention (already
    propagates automatically since the underlying gradients are zeroed
    there, but zeroed explicitly too for the same defensive-clarity
    reason Jr_face's boundary columns are zeroed explicitly above)."""
    if eps_n is None:
        eps_n = 1e-6 / W
    Nz, Nr = f.shape

    # r-face family
    gr_f, gz_f = axisym_face_gradient_r(f, dr, dz)
    gmag_r = np.sqrt(gr_f * gr_f + gz_f * gz_f + eps_n * eps_n)
    nr_r, nz_r = gr_f / gmag_r, gz_f / gmag_r
    f_face_r = np.zeros((Nz, Nr + 1))
    f_face_r[:, 1:Nr] = 0.5 * (f[:, :-1] + f[:, 1:])
    q_face_r = axisym_q(f_face_r, W)
    gr_mu_r, gz_mu_r = axisym_face_gradient_r(mu, dr, dz)
    coef_r = M_s * q_face_r
    Mrr_r = coef_r * (1.0 - nr_r * nr_r)
    Mrz_r = coef_r * (-nr_r * nz_r)
    Jr_face = -(Mrr_r * gr_mu_r + Mrz_r * gz_mu_r)
    # zero at the two r-boundaries, matching the axis/no-flux BC
    Jr_face[:, 0] = 0.0
    Jr_face[:, Nr] = 0.0

    # z-face family
    gr_f2, gz_f2 = axisym_face_gradient_z(f, dr, dz, bc_z=bc_z)
    gmag_z = np.sqrt(gr_f2 * gr_f2 + gz_f2 * gz_f2 + eps_n * eps_n)
    nr_z, nz_z = gr_f2 / gmag_z, gz_f2 / gmag_z
    f_face_z = 0.5 * (f + np.roll(f, -1, axis=0))
    q_face_z = axisym_q(f_face_z, W)
    gr_mu_z, gz_mu_z = axisym_face_gradient_z(mu, dr, dz, bc_z=bc_z)
    coef_z = M_s * q_face_z
    Mzz_z = coef_z * (1.0 - nz_z * nz_z)
    Mrz_z = coef_z * (-nr_z * nz_z)
    Jz_face = -(Mrz_z * gr_mu_z + Mzz_z * gz_mu_z)
    if bc_z == "noflux":
        Jz_face[-1, :] = 0.0

    return dict(Jr_face=Jr_face, Jz_face=Jz_face, nr_r=nr_r, nz_r=nz_r, gr_mu_r=gr_mu_r, gz_mu_r=gz_mu_r,
                Mrr_r=Mrr_r, Mrz_r=Mrz_r, nr_z=nr_z, nz_z=nz_z, gr_mu_z=gr_mu_z, gz_mu_z=gz_mu_z,
                Mzz_z=Mzz_z, Mrz_z=Mrz_z)


def axisym_exact_dissipation(fp):
    """Axisymmetric analogue of surface_transport.
    exact_dissipation_face_projected: D_h = ONLY the own-direction
    component of each face family's quadratic form (the discrete exact
    adjoint identity, Milestone 13E's derivation, reused unchanged --
    the algebra is per-face-family and does not depend on Cartesian vs.
    cylindrical, only the DIVERGENCE weighting differs, applied
    separately when integrating D_h against r dr dz)."""
    Dr = fp["Mrr_r"] * fp["gr_mu_r"] ** 2 + fp["Mrz_r"] * fp["gr_mu_r"] * fp["gz_mu_r"]
    Dz = fp["Mzz_z"] * fp["gz_mu_z"] ** 2 + fp["Mrz_z"] * fp["gr_mu_z"] * fp["gz_mu_z"]
    return Dr, Dz


def axisym_face_projected_step(f, p, dr, dz, r_c, r_f, dt, M_s, W, bc_z="periodic"):
    """Production-quality axisymmetric conservative step: tangentially-
    projected face flux + cylindrical divergence. Returns (f_new, mu,
    diag) with the SAME Fdot_chain chain-rule diagnostic as
    axisym_surface_diffusion_step, plus D_h (exact dissipation) for the
    audit. `bc_z="noflux"` (Milestone 16C Section 11): capped/two-
    substrate topology -- exact no-flux boundary at both z-ends
    instead of periodic wrap, conserving whatever solid material is
    within the domain exactly (no reservoir, no sink)."""
    mu = axisym_mu(f, p, dr, dz, r_c, r_f, bc_z=bc_z)
    fp = axisym_face_projected_flux(f, mu, dr, dz, W, M_s, bc_z=bc_z)
    div = div_cyl(fp["Jr_face"], fp["Jz_face"], r_c, r_f, dr, dz)
    f_new = f - dt * div
    Fdot = -2 * math.pi * float(np.sum(r_c[None, :] * mu * div)) * dr * dz

    Dr, Dz = axisym_exact_dissipation(fp)
    # D_h integrated: r-face family weighted by r_f (own location),
    # z-face family weighted by r_c (own location) -- derived via
    # discrete Abel summation (integration by parts) of
    # sum(r_c*mu*div_cyl(J))*dr*dz, matching div_cyl's own weighting
    # convention exactly. BOTH terms need the full dr*dz measure (found
    # by direct re-derivation after an initial single-factor version
    # failed the exact-identity check below).
    D_h = 2 * math.pi * (float(np.sum(r_f[None, :] * Dr)) + float(np.sum(r_c[None, :] * Dz))) * dr * dz

    diag = dict(mu=mu, fp=fp, Fdot_chain=Fdot, D_h=D_h)
    return f_new, mu, diag


# ---------------------------------------------------------------------------
# Milestone 16B Sections 9-11: production-quality axisymmetric GB physics,
# replacing M16A's simplified unconstrained Allen-Cahn eta kinetics
# (axisym_gb_step above) with the SAME architecture the Cartesian
# production path uses: M14G obstacle-equilibrium free-energy
# calibration (gb_obstacle_energy.gb_obstacle_coefficients, reused
# UNCHANGED -- it is pure algebra, not Cartesian-specific), the exact
# eta1+eta2=f constrained simplex update via tangent-cone projection
# (constrained_eta.tangent_cone_projected_velocity, reused UNCHANGED --
# it operates pointwise/elementwise on whatever-shaped arrays it is
# given, so it is coordinate-system-agnostic by construction), and only
# the VARIATIONAL DERIVATIVE (axisym_g_eta, already added in M16A using
# axisym_laplacian's exact cylindrical adjoint operator, NOT a copied
# Cartesian lap9 term) and the CONSERVATIVE f-TRANSPORT (face-projected
# + cylindrical divergence, Sections 3-6 above) are genuinely
# re-derived for cylindrical coordinates.
# ---------------------------------------------------------------------------

from .constrained_eta import tangent_cone_projected_velocity  # noqa: E402


def axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f, bc_z="periodic"):
    """F_total = F_surface + F_GB = 2*pi*integral r*[ (W_f/2)*f^2*(1-f)^2
    + Wc*(e1^2+e2^2)*(f^2/2-f) + (k_eta/2)*(|grad e1|^2+|grad e2|^2) ] dr dz,
    the axisymmetric port of constrained_eta.py's module-docstring free
    energy (Milestone 12B/14G), using the SAME exact discrete adjoint
    form -0.5*k*field*Laplacian_cyl(field) axisym_free_energy already
    uses for the pure-surface case (so this is the exact energy whose
    eta-derivative is axisym_g_eta and whose f-derivative is
    axisym_mu_f_gb). `bc_z` (Milestone 16G): threaded through to both
    Laplacian terms, same convention as axisym_mu_f_gb/axisym_g_eta."""
    fb = np.clip(f, 0.0, 1.0)
    bulk = 0.5 * p.W_f * f * f * (1 - f) ** 2
    grad_f_term = -0.5 * p.k_f * f * axisym_laplacian(f, dr, dz, r_c, r_f, bc_z=bc_z)
    eta2 = e1 * e1 + e2 * e2
    coupling = Wc * eta2 * (0.5 * fb * fb - fb)
    grad_eta_term = -0.5 * p.k_eta * (e1 * axisym_laplacian(e1, dr, dz, r_c, r_f, bc_z=bc_z)
                                       + e2 * axisym_laplacian(e2, dr, dz, r_c, r_f, bc_z=bc_z))
    density = bulk + grad_f_term + coupling + grad_eta_term
    return 2 * math.pi * float(np.sum(r_c[None, :] * density)) * dr * dz


def axisym_constrained_tangent_cone_eta_update(e1, e2, f, Wc, p, dr, dz, r_c, r_f, dt, M_eta,
                                                active_tol=1e-4, bc_z="periodic"):
    """Axisymmetric port of constrained_eta.constrained_tangent_cone_eta_
    update (N=2 grains only, matching axisym_reproject's existing
    scope): (a) f-tracking rescale so sum_i eta_i=f exactly (an
    operator-splitting bookkeeping step, not eta physics -- identical
    logic to the Cartesian version, coordinate-independent), (b) the
    variational velocity v0_i=-M_eta*g_i using axisym_g_eta (the
    cylindrical g_i, NOT structural_thermodynamic_force), (c) the exact
    least-squares projection onto the local tangent cone of {eta_i>=0,
    sum eta_i=f} via tangent_cone_projected_velocity (reused verbatim
    -- pointwise/coordinate-agnostic), (d) a safety-net clip+rescale for
    any residual finite-dt overshoot. Returns (e1_new, e2_new, diag)
    with the same variational/f_tracking/safety_correction breakdown
    the Cartesian version tracks. `bc_z` (Milestone 16G): threaded
    through to axisym_g_eta."""
    fb = np.clip(f, 0.0, 1.0)
    grains = [e1, e2]
    N = 2

    old_sum = e1 + e2
    has_mass = old_sum > 1e-30
    void = fb <= 0.005
    scale = np.where(has_mass, fb / np.where(has_mass, old_sum, 1.0), 0.0)
    fallback = np.where((~void) & (fb > 0.02), fb / N, 0.0)
    rescaled = [np.where(has_mass, g * scale, fallback) for g in grains]
    f_tracking_correction = float(sum(np.sum(np.abs(rescaled[i] - grains[i])) for i in range(N)))

    g_list = [axisym_g_eta(rescaled[i], fb, Wc, p, dr, dz, r_c, r_f, bc_z=bc_z) for i in range(N)]
    v0_list = [-M_eta * g for g in g_list]

    v_list = tangent_cone_projected_velocity(rescaled, v0_list, active_tol=active_tol)
    stepped = [rescaled[i] + dt * v_list[i] for i in range(N)]
    variational_change = float(sum(np.sum(np.abs(dt * v_list[i])) for i in range(N)))

    clipped = [np.clip(x, 0.0, None) for x in stepped]
    final_sum = clipped[0] + clipped[1]
    denom = np.where(final_sum > 1e-30, final_sum, 1.0)
    rescale2 = np.where(final_sum > 1e-30, fb / denom, 0.0)
    final = [x * rescale2 for x in clipped]
    safety_correction = float(sum(np.sum(np.abs(final[i] - stepped[i])) for i in range(N)))

    diag = dict(
        g1=g_list[0], g2=g_list[1],
        variational_change=variational_change,
        f_tracking_correction=f_tracking_correction,
        safety_correction=safety_correction,
        safety_fraction=safety_correction / (variational_change + 1e-300),
    )
    return final[0], final[1], diag


def axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                   active_tol=1e-4, bc_z="periodic"):
    """Full production-quality axisymmetric step with GB physics: (1)
    conservative f-transport using the face-projected/tangentially-
    projected flux (Sections 3-6) with the eta-coupled mu
    (axisym_mu_f_gb), (2) tangent-cone-constrained eta kinetics
    (axisym_constrained_tangent_cone_eta_update) at the POST-transport
    f, matching the Cartesian production operator-splitting order.
    Preserves: conserved f, physical M_GB mapping (via
    gb_obstacle_energy.m_eta_from_m_gb/m_gb_from_m_eta, applied by the
    caller when choosing M_eta), no independent M_TJ, eta_i>=0,
    sum_i eta_i=f exactly. `bc_z="noflux"` (Milestone 16G: particle-on-
    asperity topology with GB physics, extending the pure-surface
    bc_z="noflux" path from M16C Section 11/axisym_face_projected_step
    to the full GB-coupled step): exact no-flux boundary at both z-ends
    for f-transport, mu's gradient term, AND the eta kinetics' gradient
    term -- no reservoir, no sink, whatever solid is in the domain is
    conserved exactly, verified directly by mass-conservation checks in
    scripts/m16g_pr_derived_particle_asperity.py rather than assumed."""
    mu = axisym_mu_f_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f, bc_z=bc_z)
    fp = axisym_face_projected_flux(f, mu, dr, dz, W, M_s, bc_z=bc_z)
    div = div_cyl(fp["Jr_face"], fp["Jz_face"], r_c, r_f, dr, dz)
    f_new = f - dt * div

    e1_new, e2_new, eta_diag = axisym_constrained_tangent_cone_eta_update(
        e1, e2, f_new, Wc, p, dr, dz, r_c, r_f, dt, M_eta, active_tol=active_tol, bc_z=bc_z)

    Dr, Dz = axisym_exact_dissipation(fp)
    D_h = 2 * math.pi * (float(np.sum(r_f[None, :] * Dr)) + float(np.sum(r_c[None, :] * Dz))) * dr * dz

    diag = dict(mu=mu, fp=fp, D_h=D_h, eta_diag=eta_diag)
    return f_new, e1_new, e2_new, diag


# ---------------------------------------------------------------------------
# Milestone 16B Section 17: the EXACT axisymmetric analogue of the M15
# particle/substrate geometry -- NOT a visually-similar surrogate.
#
# model.py's "sinusoidal_substrate" geometry (the one actually used by
# every M15/M15E-M15I production run) is a 2-D Cartesian (X,Y)
# per-unit-depth cross-section of a FLAT-topped substrate wall whose
# free surface undulates as a COSINE OF Y (x_s(Y)=wall_mean+A*cos(2*pi*
# Y/lambda+phase)), with the model's implicit THIRD (out-of-plane, Z)
# direction translationally invariant (that is what "per-unit-depth
# Cartesian" already means -- see MILESTONE_16A's own precise
# statement). Physically this is a corrugated plate with STRAIGHT
# ridges running along Z, periodic in Y -- i.e. it has TRANSLATIONAL
# symmetry along Z and PERIODIC symmetry along Y, but NO rotational
# symmetry about any single axis. Revolving this cross-section about
# EITHER in-plane axis produces a DIFFERENT physical object (concentric
# ring-grooves if revolved about X, a corrugated tube if revolved about
# Y) than the corrugated-plate-with-straight-ridges the Cartesian model
# actually represents -- there is no axis of revolution that recovers
# the original geometry. Per Section 17's own explicit contingency
# ("If the current M15 sinusoidal substrate does not have a unique
# physical axisymmetric interpretation: STOP and define the physically
# intended 3-D geometry first"): the sinusoidal-substrate case is
# STOPPED here -- it is NOT given an axisymmetric analogue in this
# milestone, and any M15-derived contact-broadening/narrowing finding
# that depended specifically on the SUBSTRATE'S curvature (the M15G-M15I
# series) is correspondingly out of scope for the axisymmetric
# stability/COM/stress work below (Sections 18-21).
#
# model.py's FLAT "substrate" geometry (geometry="substrate", the
# degenerate sinusoid_amplitude=0 case) is different: e1 (substrate) is
# a flat half-space X<wall, Y-independent (already rotationally
# symmetric about any axis parallel to X); e2 (particle) is an ELLIPSE
# in (X,Y) centered at Y=0, semi-axes (Rx,Ry). Revolving this
# cross-section about the line Y=0 (the particle's own center line,
# parallel to X) is EXACT and UNIQUE: the ellipse becomes a spheroid of
# revolution (polar semi-axis Rx along the rotation axis, equatorial
# radius Ry), and the flat wall X=wall (already Y-independent, i.e.
# already rotationally symmetric) is UNCHANGED by the revolution. This
# is the axisymmetric M15-equivalent geometry used below: z (axial,
# the rotation axis) <-> Cartesian X (substrate-normal height), r
# (radial) <-> Cartesian Y (lateral, |Y| since revolution folds +-Y
# onto r>=0).
# ---------------------------------------------------------------------------

def axisym_m15_flat_geometry(Nz, Nr, dz, dr, W, Rz_nm=113.137, Rr_nm=56.569, overlap_nm=20.0,
                              wall_z_frac=0.2):
    """Builds (f, e1, e2, wall_z, cz) for the axisymmetric M15-flat-
    substrate analogue (Section 17). Defaults reproduce M15's own
    build_config defaults exactly: R2_nm=80, aspect_ratio=2,
    contact_orientation="short_plane" -> Rx=R2*sqrt(ar)=113.137nm
    (polar, along the rotation axis), Ry=R2/sqrt(ar)=56.569nm
    (equatorial radius), overlap_nm=20. `wall_z_frac` places the
    substrate's flat top at wall_z=wall_z_frac*(Nz*dz) (an analogue of
    model.py's substrate_wall_frac, but expressed as a fraction of the
    axisymmetric domain length since this grid has no direct Nx/dx
    correspondence to the Cartesian one)."""
    Rz = Rz_nm * 1e-9
    Rr = Rr_nm * 1e-9
    overlap = overlap_nm * 1e-9
    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")

    wall_z = wall_z_frac * Nz * dz
    e1 = 0.5 * (1.0 - np.tanh((Z - wall_z) / W))
    cz = wall_z + Rz - overlap
    rr = np.sqrt(((Z - cz) / Rz) ** 2 + (R / Rr) ** 2)
    e2 = 0.5 * (1.0 - np.tanh((rr - 1.0) * min(Rz, Rr) / W))
    f = np.maximum(e1, e2)
    return dict(f=f, e1_raw=e1, e2_raw=e2, wall_z=wall_z, cz=cz, Rz=Rz, Rr=Rr, overlap=overlap)


def axisym_m15_flat_volumes(geom, r_c, dr, dz):
    """Physical volume accounting (Section 17's explicit requirement),
    all via the SAME 2*pi*integral(r*field)dr*dz measure axisym_volume
    uses elsewhere -- no separate/inconsistent volume convention.
    V_particle_analytic is the EXACT spheroid-of-revolution volume
    (4/3*pi*Rz*Rr^2) minus the analytic spherical-cap volume buried
    below the substrate wall (the part of the ellipsoid with z<wall_z),
    for cross-checking the diffuse-interface V_f measurement -- exact
    only in the sharp-interface (W->0) limit, so agreement is expected
    to be close but not exact at finite W."""
    Rz, Rr, wall_z, cz = geom["Rz"], geom["Rr"], geom["wall_z"], geom["cz"]
    V_domain = 2 * math.pi * float(np.sum(r_c[None, :] * np.ones_like(geom["f"]))) * dr * dz
    V_substrate_diffuse = 2 * math.pi * float(np.sum(r_c[None, :] * geom["e1_raw"])) * dr * dz
    V_particle_diffuse = 2 * math.pi * float(np.sum(r_c[None, :] * geom["e2_raw"])) * dr * dz
    V_union = 2 * math.pi * float(np.sum(r_c[None, :] * geom["f"])) * dr * dz

    V_spheroid = (4.0 / 3.0) * math.pi * Rz * Rr * Rr
    # buried cap: the portion of the ellipsoid (centered at cz, polar
    # semi-axis Rz) with z<wall_z. In the ellipsoid's own local axial
    # coordinate u=z-cz (so the ellipsoid spans u in [-Rz,Rz]), the cap
    # is u<wall_z-cz=-overlap (since cz=wall_z+Rz-overlap =>
    # wall_z-cz=-Rz+overlap-...; recomputed directly below from the
    # stored overlap for clarity).
    overlap = geom["overlap"]
    u_cut = -Rz + overlap  # = wall_z - cz
    u_cut_clipped = max(-Rz, min(Rz, u_cut))
    # spherical-cap-of-ellipsoid volume below u=u_cut (standard
    # cap-of-spheroid formula, exact for a sharp interface):
    # V_cap = pi*Rr^2/Rz^2 * integral_{-Rz}^{u_cut} (Rz^2-u^2) du
    a = -Rz
    b = u_cut_clipped
    integral = (Rz * Rz * (b - a)) - (b ** 3 - a ** 3) / 3.0
    V_cap = math.pi * (Rr * Rr / (Rz * Rz)) * integral
    V_particle_analytic = V_spheroid - V_cap

    return dict(V_domain=V_domain, V_substrate_diffuse=V_substrate_diffuse,
                V_particle_diffuse=V_particle_diffuse, V_union=V_union,
                V_spheroid_full=V_spheroid, V_particle_analytic=V_particle_analytic)


def axisym_m15_com_z(e2_raw, z, r_c, dr, dz):
    """Particle center-of-mass z-coordinate, mass-weighted by the
    physical r*dr*dz measure and by the particle's own raw level set
    e2_raw (the grain-2/particle ownership field, consistent with
    model.py's convention that e_raw sets are the underlying grain
    identity before any eta-ownership correction) -- what Section 18's
    COM/no-sink test tracks. `z` is the 1-D (Nz,) axial cell-center
    array."""
    w = r_c[None, :] * e2_raw
    denom = float(np.sum(w))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * z[:, None])) / denom
