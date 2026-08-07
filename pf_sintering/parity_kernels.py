from __future__ import annotations

import math

import numpy as np
from scipy.signal import convolve2d

from .model import div, effective_gamma, grad


def evolve_f(f, e1, e2, e3, s1, s2, p):
    """MATLAB-v64-parity CH kernel.

    Key parity details restored here:
      * bounded eta fields are collectively renormalized where sum(eta_i) > f
      * anisotropy LUT uses MATLAB-style nearest-index lookup
    """
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

    # Substrate path: only the 1|2 GB is active.
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
            e1c * p.theta_grain[0]
            + e2c * p.theta_grain[1]
            + e3c * p.theta_grain[2]
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
        # Import lazily to avoid a circular helper dependency at module import.
        from .model import lap9

        mu = dF_df - p.k_f * lap9(f, p.dx)

    M = np.minimum(p.M_f * (16.0 * f * f * (1.0 - f) ** 2) ** 2, p.M_f)

    Mx = 0.5 * (M + np.roll(M, -1, axis=1))
    Jx = -Mx * (np.roll(mu, -1, axis=1) - mu) / p.dx

    Jy = np.zeros_like(f)
    My = 0.5 * (M[:-1, :] + M[1:, :])
    Jy[:-1, :] = -My * (mu[1:, :] - mu[:-1, :]) / p.dx
    Jy_down = np.zeros_like(f)
    Jy_down[1:, :] = Jy[:-1, :]

    return f - p.dt * ((Jx - np.roll(Jx, 1, axis=1)) + (Jy - Jy_down)) / p.dx


def ostwald_substrate(f, e1, e2, e3, p):
    """MATLAB-v64-parity substrate Ostwald kernel.

    scipy.signal.convolve2d(..., boundary='fill') matches MATLAB conv2(...,'same')
    zero-padding at the domain boundary.
    """
    from .model import overlap_col

    V = float(e2.sum())
    fb = np.clip(f, 0.0, 1.0)
    surf = 16.0 * fb * fb * (1.0 - fb) ** 2
    ker = np.ones((3, 3), dtype=float) / 9.0
    cr = p.Ny // 2

    _, col = overlap_col(e1[cr, :] * e2[cr, :])
    if not math.isfinite(col):
        col = p.substrate_wall_frac * p.Nx

    if getattr(p, "reservoir_neck_unprotected", False):
        # H1 diagnostic control: the reservoir sets only the net grain-volume
        # loss/gain rate below; it does not exclude the neck/contact region as
        # a source or sink. Where the surface recedes is then determined
        # entirely by the surf*e2 / surf*e1 interface weighting.
        incl = 1.0
    else:
        CC, RR = np.meshgrid(np.arange(1, p.Nx + 1), np.arange(1, p.Ny + 1))
        ex_c = max(5, round(2.0 * p.interface_width / p.dx))
        ex_r = max(5, round(3.0 * p.interface_width / p.dx))
        incl = 1.0 - np.exp(
            -0.5 * (((CC - col) / ex_c) ** 2 + ((RR - (cr + 1)) / ex_r) ** 2)
        )

    src = convolve2d(surf * e2 * incl, ker, mode="same", boundary="fill")
    snk = convolve2d(surf * e1 * incl, ker, mode="same", boundary="fill")
    tot_src = float(src.sum())
    tot_snk = float(snk.sum())
    if tot_src < 1e-15 or tot_snk < 1e-15:
        return f, e1, e2, e3

    transfer = min(V * p.dt / p.tau_ripening, 0.002 * V)
    remove = np.minimum(src / tot_src * transfer, 0.9 * e2)
    remove = np.maximum(remove, 0.0)
    actual = float(remove.sum())

    add_try = snk / (tot_snk + 1e-30) * actual
    cap_trial = np.maximum(0.0, np.minimum(1.0 - f, 1.0 - e1))
    max_depositable = float(np.minimum(add_try, cap_trial).sum())

    if max_depositable < actual and actual > 1e-30:
        sf = max_depositable / actual
        remove *= sf
        add_try *= sf

    e2 = e2 - remove
    f = f - remove

    cap = np.maximum(0.0, np.minimum(1.0 - f, 1.0 - e1))
    add1 = np.maximum(0.0, np.minimum(add_try, cap))
    e1 = e1 + add1
    f = f + add1

    return f, e1, e2, e3
