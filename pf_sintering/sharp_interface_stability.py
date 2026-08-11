"""Milestone 16C Sections 5-7: sharp-interface (no phase-field diffuseness)
axisymmetric energy functional and constrained-Hessian morphological
stability machinery.

An axisymmetric body of revolution is represented by a piecewise-LINEAR
radius profile `R(z)` on a uniform grid `z_i=i*dz`. Surface area and
volume are the EXACT closed-form values of the resulting frustum
(truncated-cone) stack -- not a differential-geometry Riemann sum
approximation -- so there is no discretization ambiguity in the energy
functional itself, only in how finely the frustum chain resolves the
true smooth shape.

    frustum i (between R_i and R_{i+1}, axial length dz):
        lateral area  = pi*(R_i+R_{i+1})*sqrt((R_{i+1}-R_i)^2 + dz^2)
        volume        = (pi/3)*dz*(R_i^2 + R_i*R_{i+1} + R_{i+1}^2)

A grain boundary at grid index `j` contributes a flat circular disk of
area `pi*R_j^2` (the cross-sectional bonded area at that axial
position), weighted by `gamma_gb` instead of `gamma_s`.

Two topologies are supported (Section 5's "fixed external/support
geometry" requirement, made explicit per case):

    "periodic"  -- R(z) wraps around a domain of length Nz*dz (the
                   classical PR-rod / Hussein-bicrystal-fiber topology).
                   Free-surface area covers the whole ring; GB areas at
                   the specified periodic indices.
    "capped"    -- R(z) on a finite, non-periodic domain [0, Nz*dz];
                   the two ENDS are GB contacts (particle-substrate
                   bonds), not free surface -- i.e. the free-surface
                   area is the lateral frustum-stack area only (no
                   extra end-cap free-surface term, matching a real
                   flat bonded contact, not an exposed flat free
                   surface).

Stability classification (Section 7): given a base profile R0(z), build
a basis of admissible (to leading order volume-conserving) perturbation
directions, compute the discrete Hessian of F by finite differences,
project it onto the exact-volume-conserving tangent subspace (the
hyperplane orthogonal to the volume gradient dV/dR), and diagonalize.
The lowest eigenvalue's sign is the primary classification:
`lambda_min>0` stable, `~0` neutral, `<0` unstable.
"""
from __future__ import annotations

import math

import numpy as np


def frustum_volume(R):
    """R: (Nz,) array. Returns per-segment volumes, length Nz-1 (open)
    or Nz (periodic, caller decides which segments to include)."""
    R = np.asarray(R, dtype=float)
    Rn = np.roll(R, -1)
    return (R * R + R * Rn + Rn * Rn)  # caller multiplies by pi*dz/3


def frustum_area(R, dz):
    R = np.asarray(R, dtype=float)
    Rn = np.roll(R, -1)
    return math.pi * (R + Rn) * np.sqrt((Rn - R) ** 2 + dz * dz)


def total_volume(R, dz, topology):
    R = np.asarray(R, dtype=float)
    seg = frustum_volume(R) * (math.pi * dz / 3.0)
    if topology == "periodic":
        return float(np.sum(seg))
    else:
        return float(np.sum(seg[:-1]))  # last "wrap" segment excluded (open chain)


def free_surface_area(R, dz, topology):
    R = np.asarray(R, dtype=float)
    seg = frustum_area(R, dz)
    if topology == "periodic":
        return float(np.sum(seg))
    else:
        return float(np.sum(seg[:-1]))


def gb_area(R, gb_indices):
    return float(sum(math.pi * R[j] ** 2 for j in gb_indices))


def energy(R, dz, gamma_s, gamma_gb, gb_indices, topology):
    """F = gamma_s*A_free + gamma_gb*sum(GB disk areas)."""
    return gamma_s * free_surface_area(R, dz, topology) + gamma_gb * gb_area(R, gb_indices)


def gradient_fd(func, R, eps=None):
    R = np.asarray(R, dtype=float)
    n = len(R)
    if eps is None:
        eps = 1e-6 * (np.max(np.abs(R)) if np.max(np.abs(R)) > 0 else 1.0)
    g = np.zeros(n)
    for i in range(n):
        Rp = R.copy(); Rp[i] += eps
        Rm = R.copy(); Rm[i] -= eps
        g[i] = (func(Rp) - func(Rm)) / (2 * eps)
    return g


def hessian_fd(func, R, eps=None):
    """Dense finite-difference Hessian of a scalar function of R.
    O(Nz^2) function evaluations -- fine for Nz of a few hundred, which
    is all this milestone's validation/classification cases need."""
    R = np.asarray(R, dtype=float)
    n = len(R)
    if eps is None:
        eps = 1e-5 * (np.max(np.abs(R)) if np.max(np.abs(R)) > 0 else 1.0)

    H = np.zeros((n, n))
    F0 = func(R)
    Fp = np.zeros(n)
    Fm = np.zeros(n)
    for i in range(n):
        Rp = R.copy(); Rp[i] += eps
        Rm = R.copy(); Rm[i] -= eps
        Fp[i] = func(Rp)
        Fm[i] = func(Rm)
    for i in range(n):
        H[i, i] = (Fp[i] - 2 * F0 + Fm[i]) / (eps * eps)
    for i in range(n):
        for j in range(i + 1, n):
            Rpp = R.copy(); Rpp[i] += eps; Rpp[j] += eps
            Rpm = R.copy(); Rpm[i] += eps; Rpm[j] -= eps
            Rmp = R.copy(); Rmp[i] -= eps; Rmp[j] += eps
            Rmm = R.copy(); Rmm[i] -= eps; Rmm[j] -= eps
            val = (func(Rpp) - func(Rpm) - func(Rmp) + func(Rmm)) / (4 * eps * eps)
            H[i, j] = H[j, i] = val
    return H


def volume_constrained_eigenmodes(R0, dz, gamma_s, gamma_gb, gb_indices, topology, n_modes=6,
                                   eps=None):
    """Section 7: the CORRECT constrained second-variation (not just a
    projected raw energy Hessian). A general base profile R0 is a
    critical point of the LAGRANGIAN L=F-lambda*V, not of F alone (a
    uniform cylinder, for instance, has nonzero mean curvature 1/R0,
    i.e. nonzero dF/dR at fixed R -- it is only stationary once the
    Laplace-pressure Lagrange multiplier lambda=dF/dR . dV/dR /
    |dV/dR|^2 is subtracted off). The classical Rayleigh-Plateau
    threshold lambda_c/R0=2*pi comes precisely from this Lagrangian
    second variation, not from the bare area functional's -- verified
    directly in Section 8's validation below.

    Returns (eigenvalues, eigenvectors [length Nz, full R-space],
    lambda_lagrange, volume_gradient_norm), eigenvalues sorted
    ascending (most unstable first)."""
    R0 = np.asarray(R0, dtype=float)
    n = len(R0)

    def F(Rv):
        return energy(Rv, dz, gamma_s, gamma_gb, gb_indices, topology)

    def V(Rv):
        return total_volume(Rv, dz, topology)

    g_F = gradient_fd(F, R0, eps=eps)
    g_V = gradient_fd(V, R0, eps=eps)
    lam = float(np.dot(g_F, g_V) / np.dot(g_V, g_V))

    H_F = hessian_fd(F, R0, eps=eps)
    H_V = hessian_fd(V, R0, eps=eps)
    H_L = H_F - lam * H_V

    g_hat = g_V / np.linalg.norm(g_V)
    A = np.eye(n) - np.outer(g_hat, g_hat)
    U, S, _ = np.linalg.svd(A)
    tol = 1e-9 * S[0]
    basis = U[:, S > tol]  # (n, n-1) orthonormal basis of the tangent hyperplane

    H_proj = basis.T @ H_L @ basis
    H_proj = 0.5 * (H_proj + H_proj.T)
    evals, evecs_reduced = np.linalg.eigh(H_proj)
    evecs_full = basis @ evecs_reduced

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs_full = evecs_full[:, idx]
    k = min(n_modes, len(evals))
    return evals[:k], evecs_full[:, :k], lam, float(np.linalg.norm(g_V))


def relax_to_equilibrium(R0, dz, gamma_s, gamma_gb, gb_indices, topology, n_steps=4000,
                          step_scale=0.2, eps=None):
    """Constrained gradient-descent relaxation to a genuine critical
    point of the Lagrangian L=F-lambda*V (needed once a GB's local
    pull-down force is present: a uniform R0 with a GB attached is NOT
    a critical point of the full functional -- the GB pulls a local
    groove, exactly the Young-Herring physics -- so the single-global-
    Lagrange-multiplier approximation `volume_constrained_eigenmodes`
    uses on a UNPRELAXED base state is only approximately correct at
    the GB. Relaxing the base profile first makes the subsequent
    Hessian analysis rigorous rather than approximate, for any
    topology/GB configuration -- not analytically pre-solved, just
    numerically walked downhill under the exact volume constraint at
    every step (projected gradient + exact volume renormalization)."""
    R = np.asarray(R0, dtype=float).copy()
    n = len(R)

    def F(Rv):
        return energy(Rv, dz, gamma_s, gamma_gb, gb_indices, topology)

    def V(Rv):
        return total_volume(Rv, dz, topology)

    V0 = V(R)
    R_scale = np.max(R)
    step_frac = 1e-4 * step_scale  # fraction of R_scale moved per iteration, in the
    # normalized steepest-descent direction (descent/max|descent| has max component 1)
    for _ in range(n_steps):
        g_F = gradient_fd(F, R, eps=eps)
        g_V = gradient_fd(V, R, eps=eps)
        lam = float(np.dot(g_F, g_V) / np.dot(g_V, g_V))
        descent = -(g_F - lam * g_V)  # tangent-projected steepest descent
        step = step_frac * R_scale * descent / (np.max(np.abs(descent)) + 1e-300)
        R = R + step
        R = np.clip(R, 1e-4 * R_scale, None)
        # exact volume renormalization (uniform rescale)
        scale = math.sqrt(V0 / V(R))
        R = R * scale
    return R


# ---------------------------------------------------------------------------
# Milestone 16C Section 14: the CARTESIAN (per-unit-depth, 2-D cross-
# section, translationally invariant out-of-plane) analogue of the same
# energy functional, for direct side-by-side comparison against the
# axisymmetric case above on IDENTICAL (L, cross-sectional-area,
# gamma_s, gamma_gb) parameters. `h(z)` is the HALF-width (the profile
# is symmetric about h=0, matching the Cartesian model's own y-centered
# convention); cross-sectional area = 2*integral(h)dz (both sides,
# trapezoidal on the piecewise-linear chain); free "surface" is now a
# LENGTH (per unit depth), not an area -- no r-weighting, the whole
# structural difference this comparison exists to isolate; GB "area"
# becomes a GB LENGTH, 2*h at the GB index (the full bonded width
# there, per unit depth).
# ---------------------------------------------------------------------------

def cartesian_segment_length(h, dz):
    h = np.asarray(h, dtype=float)
    hn = np.roll(h, -1)
    return np.sqrt((hn - h) ** 2 + dz * dz)


def cartesian_area(h, dz, topology):
    h = np.asarray(h, dtype=float)
    hn = np.roll(h, -1)
    seg = 0.5 * (h + hn) * dz
    if topology == "periodic":
        return 2.0 * float(np.sum(seg))
    return 2.0 * float(np.sum(seg[:-1]))


def cartesian_free_surface_length(h, dz, topology):
    seg = cartesian_segment_length(h, dz)
    if topology == "periodic":
        return 2.0 * float(np.sum(seg))
    return 2.0 * float(np.sum(seg[:-1]))


def cartesian_gb_length(h, gb_indices):
    return float(sum(2.0 * h[j] for j in gb_indices))


def cartesian_energy(h, dz, gamma_s, gamma_gb, gb_indices, topology):
    return (gamma_s * cartesian_free_surface_length(h, dz, topology)
            + gamma_gb * cartesian_gb_length(h, gb_indices))


def cartesian_constrained_eigenmodes(h0, dz, gamma_s, gamma_gb, gb_indices, topology, n_modes=6, eps=None):
    """Cartesian analogue of volume_constrained_eigenmodes, same
    Lagrangian-second-variation construction, constraint is
    cross-sectional AREA instead of volume."""
    h0 = np.asarray(h0, dtype=float)
    n = len(h0)

    def F(hv):
        return cartesian_energy(hv, dz, gamma_s, gamma_gb, gb_indices, topology)

    def A(hv):
        return cartesian_area(hv, dz, topology)

    g_F = gradient_fd(F, h0, eps=eps)
    g_A = gradient_fd(A, h0, eps=eps)
    lam = float(np.dot(g_F, g_A) / np.dot(g_A, g_A))

    H_F = hessian_fd(F, h0, eps=eps)
    H_A = hessian_fd(A, h0, eps=eps)
    H_L = H_F - lam * H_A

    g_hat = g_A / np.linalg.norm(g_A)
    B = np.eye(n) - np.outer(g_hat, g_hat)
    U, S, _ = np.linalg.svd(B)
    tol = 1e-9 * S[0]
    basis = U[:, S > tol]

    H_proj = basis.T @ H_L @ basis
    H_proj = 0.5 * (H_proj + H_proj.T)
    evals, evecs_reduced = np.linalg.eigh(H_proj)
    evecs_full = basis @ evecs_reduced

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs_full = evecs_full[:, idx]
    k = min(n_modes, len(evals))
    return evals[:k], evecs_full[:, :k], lam, float(np.linalg.norm(g_A))


def cartesian_relax_to_equilibrium(h0, dz, gamma_s, gamma_gb, gb_indices, topology, n_steps=4000,
                                    step_scale=0.2, eps=None):
    """Cartesian analogue of relax_to_equilibrium (area-constrained)."""
    h = np.asarray(h0, dtype=float).copy()

    def F(hv):
        return cartesian_energy(hv, dz, gamma_s, gamma_gb, gb_indices, topology)

    def A(hv):
        return cartesian_area(hv, dz, topology)

    A0 = A(h)
    h_scale = np.max(h)
    step_frac = 1e-4 * step_scale
    for _ in range(n_steps):
        g_F = gradient_fd(F, h, eps=eps)
        g_A = gradient_fd(A, h, eps=eps)
        lam = float(np.dot(g_F, g_A) / np.dot(g_A, g_A))
        descent = -(g_F - lam * g_A)
        step = step_frac * h_scale * descent / (np.max(np.abs(descent)) + 1e-300)
        h = h + step
        h = np.clip(h, 1e-4 * h_scale, None)
        scale = A0 / A(h)  # area ~ h linearly (NOT h^2 like axisymmetric volume)
        h = h * scale
    return h
