"""Performance milestone, Section 5: a Numba-compiled, memory-fused
implementation of EXACTLY the same discrete algebra as
axisym.axisym_gb_face_projected_step (pf_sintering/axisym.py:513-544),
bc_z="noflux" only (the only mode used in production). No equation is
changed; every stencil below is a direct scalar transcription of the
corresponding NumPy function it replaces, written out so the exact same
per-face/per-cell arithmetic is reproduced (see the docstring of each
kernel for the NumPy function it mirrors). Physical/numerical
equivalence -- NOT bit-identical output -- is required and verified
separately in scripts/perf_numba_equivalence.py; small floating-point
differences from parallel reduction order are expected and acceptable
per the milestone's own tolerance.

Two subtleties in the reference implementation are preserved exactly,
not "fixed" here, since this milestone is execution/acceleration only:

1. `axisym_face_gradient_r` (used for the r-face family's normal/
   tangential-projection construction) computes its z-derivative via
   `np.roll` UNCONDITIONALLY periodic in z, regardless of the `bc_z`
   passed to the caller -- i.e. even in "noflux" mode, the r-face
   family's z-slope estimate wraps circularly between row 0 and row
   Nz-1. This only affects the LOCAL tangent/normal direction estimate
   used to build the tangentially-projected mobility tensor at r-faces
   (not the z-transport itself, which is separately and correctly
   no-flux via `axisym_face_gradient_z`/explicit Jz_face zeroing) and is
   numerically inert here because f is ~0 at the domain's z-boundaries
   by construction (bounding-box margin).
2. `f_face_z = 0.5*(f + roll(f,-1,axis=0))` (the face-averaged f feeding
   q_face_z) is likewise always periodic, but multiplies into Jz_face,
   whose wrap row (index Nz-1) is forced to exactly 0 for bc_z="noflux"
   regardless of this factor's value there -- so the leak is inert.

The eta tangent-cone projection (constrained_eta.tangent_cone_projected_
velocity) is specialized to N=2 (the only grain count this project
uses) via its exact closed-form reduction: v1_0=(v0_1-v0_2)/2, and since
v1_0+v2_0=0 identically, at most one of the two grains can violate its
active-set bound per cell, so the general iterative active-set loop
reduces to a single 3-way branch. Derivation and general-N reference are
in constrained_eta.py; the N=2 closed form used here was hand-verified
against tangent_cone_projected_velocity's actual output for random
active/violating configurations before use (see
scripts/perf_numba_equivalence.py's dedicated tangent-cone unit check).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False)
def laplacian_cyl_noflux_kernel(f, r_c, r_f, dr, dz, out):
    """Mirrors axisym_laplacian(f, dr, dz, r_c, r_f, bc_z="noflux") via
    face_grad_r + face_grad_z(bc_z="noflux") + div_cyl, fused into one
    pass (no intermediate face arrays materialized)."""
    Nz, Nr = f.shape
    for j in range(Nz):
        gz_jm1_row = 0.0
        for i in range(Nr):
            if i == 0:
                jr_i = 0.0
            else:
                jr_i = (f[j, i] - f[j, i - 1]) / dr
            if i == Nr - 1:
                jr_ip1 = 0.0
            else:
                jr_ip1 = (f[j, i + 1] - f[j, i]) / dr
            div_r = (r_f[i + 1] * jr_ip1 - r_f[i] * jr_i) / (r_c[i] * dr)

            if j == Nz - 1:
                gz_j = 0.0
            else:
                gz_j = (f[j + 1, i] - f[j, i]) / dz
            if j == 0:
                gz_jm1 = 0.0
            else:
                gz_jm1 = (f[j, i] - f[j - 1, i]) / dz
            div_z = (gz_j - gz_jm1) / dz

            out[j, i] = div_r + div_z
    return out


@njit(cache=True, fastmath=False, parallel=True)
def mu_kernel(f, e1, e2, W_f, k_f, Wc, dr, dz, r_c, r_f, out):
    """Mirrors axisym_mu_f_gb: mu0 = W_f*f*(1-f)*(1-2f) - Wc*eta2*(1-fb),
    mu = mu0 - k_f*Laplacian_cyl(f, bc_z="noflux"), fused (bulk +
    coupling + Laplacian all written in the same per-cell pass)."""
    Nz, Nr = f.shape
    for j in prange(Nz):
        for i in range(Nr):
            fij = f[j, i]
            fb = min(max(fij, 0.0), 1.0)
            e1ij = e1[j, i]
            e2ij = e2[j, i]
            eta2 = e1ij * e1ij + e2ij * e2ij
            mu0 = W_f * fij * (1.0 - fij) * (1.0 - 2.0 * fij) - Wc * eta2 * (1.0 - fb)

            if i == 0:
                jr_i = 0.0
            else:
                jr_i = (f[j, i] - f[j, i - 1]) / dr
            if i == Nr - 1:
                jr_ip1 = 0.0
            else:
                jr_ip1 = (f[j, i + 1] - f[j, i]) / dr
            div_r = (r_f[i + 1] * jr_ip1 - r_f[i] * jr_i) / (r_c[i] * dr)

            if j == Nz - 1:
                gz_j = 0.0
            else:
                gz_j = (f[j + 1, i] - f[j, i]) / dz
            if j == 0:
                gz_jm1 = 0.0
            else:
                gz_jm1 = (f[j, i] - f[j - 1, i]) / dz
            div_z = (gz_j - gz_jm1) / dz

            lap = div_r + div_z
            out[j, i] = mu0 - k_f * lap
    return out


@njit(cache=True, fastmath=False)
def _gr_cell_r(a, j, i, Nr, dr):
    """Centered r-derivative at cell (j,i), one-sided at r-boundaries --
    the `gr_cell` used inside axisym_face_gradient_z."""
    if Nr <= 1:
        return 0.0
    if i == 0:
        return (a[j, 1] - a[j, 0]) / dr
    if i == Nr - 1:
        return (a[j, Nr - 1] - a[j, Nr - 2]) / dr
    return (a[j, i + 1] - a[j, i - 1]) / (2.0 * dr)


@njit(cache=True, fastmath=False, parallel=True)
def flux_kernel(f, mu, dr, dz, W, M_s, eps_n, Jr_out, Jz_out):
    """Mirrors axisym_face_projected_flux(f, mu, dr, dz, W, M_s,
    bc_z="noflux") exactly, including its two "always-periodic-in-z"
    subtleties (see module docstring). Jr_out: (Nz,Nr+1). Jz_out:
    (Nz,Nr)."""
    Nz, Nr = f.shape

    # ---- r-face family: Jr_out[:, 1:Nr] (boundary columns stay 0) ----
    for j in prange(Nz):
        jp1 = j + 1 if j < Nz - 1 else 0     # unconditional periodic wrap
        jm1 = j - 1 if j > 0 else Nz - 1     # (axisym_face_gradient_r's np.roll)
        for i in range(1, Nr):
            # gr_face (r-derivative across the face between col i-1,i)
            gr_f = (f[j, i] - f[j, i - 1]) / dr
            gr_mu = (mu[j, i] - mu[j, i - 1]) / dr
            # gz_face (periodic-centered z-derivative at cell i-1 and i,
            # averaged) -- ALWAYS periodic per the reference function.
            gz_cell_im1 = (f[jp1, i - 1] - f[jm1, i - 1]) / (2.0 * dz)
            gz_cell_i = (f[jp1, i] - f[jm1, i]) / (2.0 * dz)
            gz_f = 0.5 * (gz_cell_im1 + gz_cell_i)
            gz_mu_cell_im1 = (mu[jp1, i - 1] - mu[jm1, i - 1]) / (2.0 * dz)
            gz_mu_cell_i = (mu[jp1, i] - mu[jm1, i]) / (2.0 * dz)
            gz_mu = 0.5 * (gz_mu_cell_im1 + gz_mu_cell_i)

            gmag = np.sqrt(gr_f * gr_f + gz_f * gz_f + eps_n * eps_n)
            nr_ = gr_f / gmag
            nz_ = gz_f / gmag

            f_face = 0.5 * (f[j, i - 1] + f[j, i])
            q_face = (12.0 / W) * f_face * f_face * (1.0 - f_face) ** 2
            coef = M_s * q_face
            Mrr = coef * (1.0 - nr_ * nr_)
            Mrz = coef * (-nr_ * nz_)
            Jr_out[j, i] = -(Mrr * gr_mu + Mrz * gz_mu)
        Jr_out[j, 0] = 0.0
        Jr_out[j, Nr] = 0.0

    # ---- z-face family: Jz_out[j,:] = face between row j and row j+1 ----
    for j in prange(Nz):
        if j == Nz - 1:
            for i in range(Nr):
                Jz_out[j, i] = 0.0  # bc_z="noflux": wrap face forced 0
            continue
        jp1 = j + 1
        for i in range(Nr):
            gz_f = (f[jp1, i] - f[j, i]) / dz
            gz_mu = (mu[jp1, i] - mu[j, i]) / dz
            gr_cell_j = _gr_cell_r(f, j, i, Nr, dr)
            gr_cell_jp1 = _gr_cell_r(f, jp1, i, Nr, dr)
            gr_f = 0.5 * (gr_cell_j + gr_cell_jp1)
            gr_mu_cell_j = _gr_cell_r(mu, j, i, Nr, dr)
            gr_mu_cell_jp1 = _gr_cell_r(mu, jp1, i, Nr, dr)
            gr_mu = 0.5 * (gr_mu_cell_j + gr_mu_cell_jp1)

            gmag = np.sqrt(gr_f * gr_f + gz_f * gz_f + eps_n * eps_n)
            nr_ = gr_f / gmag
            nz_ = gz_f / gmag

            f_face = 0.5 * (f[j, i] + f[jp1, i])  # always-periodic average (jp1 never wraps here since j<Nz-1)
            q_face = (12.0 / W) * f_face * f_face * (1.0 - f_face) ** 2
            coef = M_s * q_face
            Mzz = coef * (1.0 - nz_ * nz_)
            Mrz = coef * (-nr_ * nz_)
            Jz_out[j, i] = -(Mrz * gr_mu + Mzz * gz_mu)
    return Jr_out, Jz_out


@njit(cache=True, fastmath=False, parallel=True)
def div_and_update_kernel(f, Jr_face, Jz_face, r_c, r_f, dr, dz, dt, f_out):
    """div_cyl(Jr_face, Jz_face) + f_new = f - dt*div, fused."""
    Nz, Nr = f.shape
    for j in prange(Nz):
        jm1 = j - 1 if j > 0 else Nz - 1
        for i in range(Nr):
            div_r = (r_f[i + 1] * Jr_face[j, i + 1] - r_f[i] * Jr_face[j, i]) / (r_c[i] * dr)
            div_z = (Jz_face[j, i] - Jz_face[jm1, i]) / dz
            f_out[j, i] = f[j, i] - dt * (div_r + div_z)
    return f_out


@njit(cache=True, fastmath=False, parallel=True)
def eta_update_kernel(e1, e2, f_new, Wc, k_eta, dr, dz, r_c, r_f, dt, M_eta, active_tol,
                       e1_out, e2_out):
    """Mirrors axisym_constrained_tangent_cone_eta_update (N=2 closed
    form -- see module docstring). Single fused pass: f-tracking
    rescale -> g_eta (coupling + Laplacian(rescaled_i)) -> N=2 tangent-
    cone velocity -> Euler step -> clip -> renormalize to fb."""
    Nz, Nr = e1.shape
    for j in prange(Nz):
        for i in range(Nr):
            fij = f_new[j, i]
            fb = min(max(fij, 0.0), 1.0)
            e1ij = e1[j, i]
            e2ij = e2[j, i]
            old_sum = e1ij + e2ij

            if old_sum > 1e-30:
                scale = fb / old_sum
                r1 = e1ij * scale
                r2 = e2ij * scale
            else:
                void = fb <= 0.005
                if (not void) and fb > 0.02:
                    r1 = fb / 2.0
                    r2 = fb / 2.0
                else:
                    r1 = 0.0
                    r2 = 0.0
            e1_out[j, i] = r1  # temporarily store rescaled_1 (overwritten below)
            e2_out[j, i] = r2  # temporarily store rescaled_2 (overwritten below)

    # second pass: Laplacian(rescaled_i) needs neighbor rescaled values,
    # so it must run after ALL cells' rescale is written (race-free).
    for j in prange(Nz):
        for i in range(Nr):
            fij = f_new[j, i]
            fb = min(max(fij, 0.0), 1.0)
            r1 = e1_out[j, i]
            r2 = e2_out[j, i]

            # Laplacian_cyl(r1), Laplacian_cyl(r2), bc_z="noflux"
            if i == 0:
                jr1_i = 0.0
                jr2_i = 0.0
            else:
                jr1_i = (r1 - e1_out[j, i - 1]) / dr
                jr2_i = (r2 - e2_out[j, i - 1]) / dr
            if i == Nr - 1:
                jr1_ip1 = 0.0
                jr2_ip1 = 0.0
            else:
                jr1_ip1 = (e1_out[j, i + 1] - r1) / dr
                jr2_ip1 = (e2_out[j, i + 1] - r2) / dr
            div_r1 = (r_f[i + 1] * jr1_ip1 - r_f[i] * jr1_i) / (r_c[i] * dr)
            div_r2 = (r_f[i + 1] * jr2_ip1 - r_f[i] * jr2_i) / (r_c[i] * dr)

            if j == Nz - 1:
                gz1_j = 0.0
                gz2_j = 0.0
            else:
                gz1_j = (e1_out[j + 1, i] - r1) / dz
                gz2_j = (e2_out[j + 1, i] - r2) / dz
            if j == 0:
                gz1_jm1 = 0.0
                gz2_jm1 = 0.0
            else:
                gz1_jm1 = (r1 - e1_out[j - 1, i]) / dz
                gz2_jm1 = (r2 - e2_out[j - 1, i]) / dz
            div_z1 = (gz1_j - gz1_jm1) / dz
            div_z2 = (gz2_j - gz2_jm1) / dz

            lap1 = div_r1 + div_z1
            lap2 = div_r2 + div_z2

            g1 = 2.0 * Wc * r1 * (0.5 * fb * fb - fb) - k_eta * lap1
            g2 = 2.0 * Wc * r2 * (0.5 * fb * fb - fb) - k_eta * lap2
            v0_1 = -M_eta * g1
            v0_2 = -M_eta * g2

            lam0 = 0.5 * (v0_1 + v0_2)
            v1_0 = v0_1 - lam0
            v2_0 = v0_2 - lam0

            active1 = r1 <= active_tol
            active2 = r2 <= active_tol
            if active1 and v1_0 < -1e-15:
                v1 = 0.0
                v2 = 0.0
            elif active2 and v2_0 < -1e-15:
                v1 = 0.0
                v2 = 0.0
            else:
                v1 = v1_0
                v2 = v2_0

            stepped1 = r1 + dt * v1
            stepped2 = r2 + dt * v2
            clipped1 = stepped1 if stepped1 > 0.0 else 0.0
            clipped2 = stepped2 if stepped2 > 0.0 else 0.0
            final_sum = clipped1 + clipped2
            if final_sum > 1e-30:
                rescale2 = fb / final_sum
                e1_out[j, i] = clipped1 * rescale2
                e2_out[j, i] = clipped2 * rescale2
            else:
                e1_out[j, i] = 0.0
                e2_out[j, i] = 0.0
    return e1_out, e2_out


class NumbaScratch:
    """Preallocated scratch buffers (Section 4: no per-step allocation),
    reused across steps by axisym_gb_face_projected_step_fast."""

    def __init__(self, Nz, Nr):
        self.mu = np.empty((Nz, Nr), dtype=np.float64)
        self.Jr = np.empty((Nz, Nr + 1), dtype=np.float64)
        self.Jz = np.empty((Nz, Nr), dtype=np.float64)
        self.f_new = np.empty((Nz, Nr), dtype=np.float64)
        self.e1_new = np.empty((Nz, Nr), dtype=np.float64)
        self.e2_new = np.empty((Nz, Nr), dtype=np.float64)


def axisym_gb_face_projected_step_fast(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
                                        scratch, active_tol=1e-4, eps_n=None):
    """Numba-accelerated equivalent of axisym_gb_face_projected_step,
    bc_z="noflux" ONLY (the only mode this recovery's production path
    uses). Returns NEW arrays (f_new, e1_new, e2_new) -- scratch buffers
    are reused internally for intermediates (mu, Jr, Jz) but the
    returned state arrays are the scratch.f_new/e1_new/e2_new buffers
    themselves (caller must treat the previous step's f/e1/e2 as
    consumed, matching the reference function's own return-new-array
    convention closely enough for a sequential step loop: swap
    scratch buffers step-to-step rather than reallocating)."""
    if eps_n is None:
        eps_n = 1e-6 / W
    mu_kernel(f, e1, e2, p.W_f, p.k_f, Wc, dr, dz, r_c, r_f, scratch.mu)
    flux_kernel(f, scratch.mu, dr, dz, W, M_s, eps_n, scratch.Jr, scratch.Jz)
    div_and_update_kernel(f, scratch.Jr, scratch.Jz, r_c, r_f, dr, dz, dt, scratch.f_new)
    eta_update_kernel(e1, e2, scratch.f_new, Wc, p.k_eta, dr, dz, r_c, r_f, dt, M_eta, active_tol,
                       scratch.e1_new, scratch.e2_new)
    return scratch.f_new, scratch.e1_new, scratch.e2_new
