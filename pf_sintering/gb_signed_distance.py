"""Milestone 15B Section 3: exact signed Euclidean distance from a point
(X, Y) to the sinusoidal substrate/GB curve x = x_s(y), for use as the
calibrated obstacle profile's coordinate (replacing the raw horizontal
difference `X - x_s(Y)`, which is NOT the normal distance wherever the
curve has nonzero slope -- at this milestone's primary geometry
(A=100nm, lambda=320nm) the slope reaches ~2.0 near the TJs, so the
horizontal-difference approximation is not a small correction there).

For a point (X, Y), the squared distance to the curve h(Y') =
(X-x_s(Y'))^2 + (Y-Y')^2 is minimized where its Y'-derivative vanishes:

    g(Y') = (X - x_s(Y')) * x_s'(Y') + (Y - Y') = 0

Newton's method on g (using g'(Y') = -x_s'(Y')^2 + (X-x_s(Y'))*x_s''(Y') - 1)
converges quadratically once started near the true minimum, but can
diverge or converge to the wrong (non-minimizing) root if started far
from it -- this curve's curvature is large enough that naive
Newton-from-Y'=Y often does diverge (checked empirically). A coarse
grid search over candidate Y' offsets first locates the right basin;
Newton then refines it to machine precision (checked against brute-force
fine-sampling: agreement to ~1e-15 m).
"""

from __future__ import annotations

import math

import numpy as np


def sinusoid_signed_distance(X, Y, amplitude, wavelength, phase, x_mean,
                              n_coarse=41, search_frac=0.75, n_newton=6):
    """Signed distance from (X, Y) to x = x_mean + amplitude*cos(2*pi*Y/
    wavelength + phase), positive on the +X side of the curve (matching
    the existing `X - x_s(Y)` sign convention used elsewhere for this
    geometry: positive = vapor/grain-2 side)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    k = 2.0 * math.pi / wavelength

    def x_s(Yp):
        return x_mean + amplitude * np.cos(k * Yp + phase)

    def dxs_dY(Yp):
        return -amplitude * k * np.sin(k * Yp + phase)

    def d2xs_dY2(Yp):
        return -amplitude * k * k * np.cos(k * Yp + phase)

    best_Y = None
    best_d2 = None
    for off in np.linspace(-search_frac * wavelength, search_frac * wavelength, n_coarse):
        Yc = Y + off
        xs = x_s(Yc)
        d2 = (X - xs) ** 2 + (Y - Yc) ** 2
        if best_d2 is None:
            best_d2, best_Y = d2, Yc
        else:
            better = d2 < best_d2
            best_d2 = np.where(better, d2, best_d2)
            best_Y = np.where(better, Yc, best_Y)

    Yc = best_Y
    for _ in range(n_newton):
        xs = x_s(Yc)
        dxs = dxs_dY(Yc)
        d2xs = d2xs_dY2(Yc)
        g = (X - xs) * dxs + (Y - Yc)
        gp = -dxs * dxs + (X - xs) * d2xs - 1.0
        step = g / np.where(np.abs(gp) < 1e-30, -1e-30, gp)
        step = np.clip(step, -0.05 * wavelength, 0.05 * wavelength)
        Yc = Yc - step

    xs_final = x_s(Yc)
    dist = np.sign(X - xs_final) * np.hypot(X - xs_final, Y - Yc)
    return dist
