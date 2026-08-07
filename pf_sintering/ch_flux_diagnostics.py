"""DIAGNOSTIC ONLY -- not wired into production physics.

Exact-formula mirror of the active production CH kernel
(`parity_kernels.evolve_f`, confirmed monkey-patched onto `model.evolve_f`
at package-import time), instrumented to expose the discrete chemical
potential `mu`, mobility `M`, and conserved flux `J = -M*grad(mu)/dx` *as
actually discretized* -- not a continuum expression -- without altering the
dynamics.

`evolve_f_diagnostic` is a byte-for-byte transcription of the active
kernel's formula (verified in tests/test_ch_flux_diagnostics.py to
reproduce identical `f` output to the production kernel for the same
input); nothing about the physics is changed.

Discretization note: `Jx` is face-centered on the *right* face of each cell
(`Jx[i,j]` is the flux between column j and column j+1, matching
`np.roll(mu,-1,axis=1)`); `Jy` is face-centered on the face between row i
and row i+1 for `Jy[i,:]` with `i < Ny-1` (the last row has no flux below
it, matching the production kernel's boundary treatment, which has no flux
in/out of row 0 or row Ny-1). `df` is the *exact* applied CH increment for
that step (`f_new - f_old`, before any subsequent mass-preserving
projection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import div, effective_gamma, grad, lap9


@dataclass
class CHFluxDiagnostics:
    mu: np.ndarray
    M: np.ndarray
    Jx: np.ndarray
    Jy: np.ndarray
    df: np.ndarray


def evolve_f_diagnostic(f, e1, e2, e3, s1, p):
    """Exact transcription of parity_kernels.evolve_f (dropping the unused
    second Sink argument the production signature carries) with mu/M/J/df
    captured. Returns (f_new, diagnostics)."""
    fb = np.clip(f, 0.0, 1.0)

    e1c = np.clip(e1, 0.0, fb)
    e2c = np.clip(e2, 0.0, fb)
    e3c = np.clip(e3, 0.0, fb) if p.use_eta3 else np.zeros_like(fb)

    esum = e1c + e2c + e3c
    over = esum > fb + 1e-12
    if np.any(over):
        scale = fb[over] / esum[over]
        e1c[over] *= scale
        e2c[over] *= scale
        e3c[over] *= scale

    esum = e1c + e2c + e3c
    eta_sq = e1c * e1c + e2c * e2c + e3c * e3c

    pair12 = np.maximum(0.0, e1c * e2c)
    gamma_local = np.full_like(f, p.gamma_gb_ref)
    mask = pair12 > 1e-20
    if np.any(mask):
        gamma_local[mask] = effective_gamma(s1, p)
    Wc = 36.0 * gamma_local / p.interface_width

    dF_df = p.W_f * f * (1.0 - f) * (1.0 - 2.0 * f) - Wc * eta_sq * (1.0 - fb)

    if p.use_aniso_surface:
        gx, gy = grad(f, p.dx)
        th0 = (
            e1c * p.theta_grain[0] + e2c * p.theta_grain[1] + e3c * p.theta_grain[2]
        ) / np.maximum(esum, 1e-12)
        theta = np.arctan2(gy, gx)
        psi_f = np.mod(theta - th0, math.pi / 2.0)

        n_lut = len(p.lut_psi)
        idx = np.rint(psi_f * (2.0 / math.pi) * (n_lut - 1)).astype(np.int64)
        idx = np.clip(idx, 0, n_lut - 1)
        a_f = p.lut_a[idx].copy()
        ap_f = p.lut_ap[idx].copy()

        flat = gx * gx + gy * gy < (0.01 / p.interface_width) ** 2
        a_f[flat] = 1.0
        ap_f[flat] = 0.0

        xi_x = p.k_f * (a_f * a_f * gx - a_f * ap_f * gy)
        xi_y = p.k_f * (a_f * a_f * gy + a_f * ap_f * gx)
        mu = dF_df - div(xi_x, xi_y, p.dx)
    else:
        mu = dF_df - p.k_f * lap9(f, p.dx)

    M = np.minimum(p.M_f * (16.0 * f * f * (1.0 - f) ** 2) ** 2, p.M_f)

    Mx = 0.5 * (M + np.roll(M, -1, axis=1))
    Jx = -Mx * (np.roll(mu, -1, axis=1) - mu) / p.dx

    Jy = np.zeros_like(f)
    My = 0.5 * (M[:-1, :] + M[1:, :])
    Jy[:-1, :] = -My * (mu[1:, :] - mu[:-1, :]) / p.dx
    Jy_down = np.zeros_like(f)
    Jy_down[1:, :] = Jy[:-1, :]

    df = -p.dt * ((Jx - np.roll(Jx, 1, axis=1)) + (Jy - Jy_down)) / p.dx
    f_new = f + df

    return f_new, CHFluxDiagnostics(mu=mu, M=M, Jx=Jx, Jy=Jy, df=df)


def surface_flux_near_tj(f, Jx, Jy, tj_xy, p, band_widths_in_W=(1.0, 3.0)):
    """Tangential/normal flux components of J, averaged over an annulus band
    around a TJ (inner..outer radius in interface widths), sampled at cell
    centers along the f=0.5 contour within that band. Returns None if too
    few contour points fall in the band."""
    from skimage.measure import find_contours

    pts = []
    for rc in find_contours(f, 0.5):
        pts.append(np.c_[(rc[:, 1] + 1) * p.dx, (rc[:, 0] + 1) * p.dx])
    if not pts:
        return None
    P = np.vstack(pts)
    d = np.hypot(P[:, 0] - tj_xy[0], P[:, 1] - tj_xy[1])
    lo, hi = band_widths_in_W[0] * p.interface_width, band_widths_in_W[1] * p.interface_width
    band = P[(d >= lo) & (d <= hi)]
    if len(band) < 3:
        return None

    # Local tangent via the principal direction of the local contour-point
    # cloud, oriented AWAY from the TJ (into the bulk of that branch) --
    # the same convention tj_force._branch_dir uses, so that a positive
    # J_tangent below means flux directed away from the TJ (out of the neck)
    # and negative means flux directed toward the TJ (into the neck). Raw
    # SVD alone has an arbitrary +/- sign and would make J_tangent's sign
    # meaningless from call to call.
    centered = band - band.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    away = band.mean(0) - np.asarray(tj_xy)
    if np.dot(tangent, away) < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])

    ci = np.clip(np.round(band[:, 0] / p.dx - 1).astype(int), 0, p.Nx - 1)
    ri = np.clip(np.round(band[:, 1] / p.dx - 1).astype(int), 0, p.Ny - 1)
    jx = Jx[ri, ci]
    jy = Jy[ri, ci]
    j_mean = np.array([float(np.mean(jx)), float(np.mean(jy))])

    return dict(
        n_points=len(band),
        J_mean=(float(j_mean[0]), float(j_mean[1])),
        J_tangent=float(np.dot(j_mean, tangent)),
        J_normal=float(np.dot(j_mean, normal)),
        tangent=(float(tangent[0]), float(tangent[1])),
        normal=(float(normal[0]), float(normal[1])),
    )
