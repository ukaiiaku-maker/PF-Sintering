"""Dynamic normalized-ownership phase-field operator.

This is a parallel implementation path for the promoted corrected energy.
It never calls the legacy eta-coupled chemical potential.  The conserved
field is advanced with the existing face-projected surface flux, while the
independent ownership fraction follows

    phi_dot = -(M_eta/2) f g_phi.

The factor one half exactly recovers the calibrated two-order-parameter
obstacle kinetics in dense material.  The extra factor ``f`` is a positive
mobility gate: it is unity at a planar dense GB and suppresses meaningless
ownership motion continuously in vacuum.  Reconstructing ``eta1=f*phi`` and
``eta2=f*(1-phi)`` enforces partition closure by construction.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .axisym_numba_kernel import flux_kernel, div_and_update_kernel


@njit(cache=True, fastmath=False, parallel=True)
def _normalized_phi_kernel(f, eta1, phi):
    nz, nr = f.shape
    for j in prange(nz):
        for i in range(nr):
            fij = f[j, i]
            value = eta1[j, i] / fij if abs(fij) > 1.0e-14 else 0.0
            phi[j, i] = min(max(value, 0.0), 1.0)


@njit(cache=True, fastmath=False, parallel=True)
def _corrected_mu_kernel(f, phi, W_f, k_f, Wc, k_eta,
                         dr, dz, r_c, r_f, out):
    nz, nr = f.shape
    for j in prange(nz):
        for i in range(nr):
            fij = f[j, i]
            phij = phi[j, i]

            gr_minus = 0.0 if i == 0 else (fij - f[j, i - 1]) / dr
            gr_plus = 0.0 if i == nr - 1 else (f[j, i + 1] - fij) / dr
            lap_r = (r_f[i + 1] * gr_plus - r_f[i] * gr_minus) / (r_c[i] * dr)
            gz_minus = 0.0 if j == 0 else (fij - f[j - 1, i]) / dz
            gz_plus = 0.0 if j == nz - 1 else (f[j + 1, i] - fij) / dz
            lap_z = (gz_plus - gz_minus) / dz

            gate_derivative = 0.0
            if i > 0:
                dphi = phi[j, i] - phi[j, i - 1]
                gate_derivative += (
                    k_eta * r_f[i] * dphi * dphi
                    / (2.0 * r_c[i] * dr * dr))
            if i < nr - 1:
                dphi = phi[j, i + 1] - phij
                gate_derivative += (
                    k_eta * r_f[i + 1] * dphi * dphi
                    / (2.0 * r_c[i] * dr * dr))
            if j > 0:
                dphi = phij - phi[j - 1, i]
                gate_derivative += k_eta * dphi * dphi / (2.0 * dz * dz)
            if j < nz - 1:
                dphi = phi[j + 1, i] - phij
                gate_derivative += k_eta * dphi * dphi / (2.0 * dz * dz)

            out[j, i] = (
                W_f * fij * (1.0 - fij) * (1.0 - 2.0 * fij)
                - k_f * (lap_r + lap_z)
                + Wc * phij * (1.0 - phij)
                + gate_derivative)


@njit(cache=True, fastmath=False, parallel=True)
def _phi_update_kernel(phi, f, Wc, k_eta, dr, dz, r_c, r_f,
                       dt, M_eta, phi_out, eta1_out, eta2_out):
    nz, nr = f.shape
    for j in prange(nz):
        for i in range(nr):
            fij = min(max(f[j, i], 0.0), 1.0)
            phij = phi[j, i]

            flux_r_minus = 0.0
            if i > 0:
                flux_r_minus = (
                    0.5 * (fij + min(max(f[j, i - 1], 0.0), 1.0))
                    * (phij - phi[j, i - 1]) / dr)
            flux_r_plus = 0.0
            if i < nr - 1:
                flux_r_plus = (
                    0.5 * (fij + min(max(f[j, i + 1], 0.0), 1.0))
                    * (phi[j, i + 1] - phij) / dr)
            div_r = (
                r_f[i + 1] * flux_r_plus - r_f[i] * flux_r_minus
            ) / (r_c[i] * dr)

            flux_z_minus = 0.0
            if j > 0:
                flux_z_minus = (
                    0.5 * (fij + min(max(f[j - 1, i], 0.0), 1.0))
                    * (phij - phi[j - 1, i]) / dz)
            flux_z_plus = 0.0
            if j < nz - 1:
                flux_z_plus = (
                    0.5 * (fij + min(max(f[j + 1, i], 0.0), 1.0))
                    * (phi[j + 1, i] - phij) / dz)
            div_z = (flux_z_plus - flux_z_minus) / dz

            gphi = fij * Wc * (1.0 - 2.0 * phij) - 2.0 * k_eta * (div_r + div_z)
            updated = phij - dt * 0.5 * M_eta * fij * gphi
            updated = min(max(updated, 0.0), 1.0)
            phi_out[j, i] = updated
            eta1_out[j, i] = fij * updated
            eta2_out[j, i] = fij * (1.0 - updated)


class CorrectedNumbaScratch:
    """Preallocated buffers for the normalized-ownership production step."""

    def __init__(self, nz: int, nr: int):
        self.phi = np.empty((nz, nr), dtype=np.float64)
        self.mu = np.empty((nz, nr), dtype=np.float64)
        self.Jr = np.empty((nz, nr + 1), dtype=np.float64)
        self.Jz = np.empty((nz, nr), dtype=np.float64)
        self.f_new = np.empty((nz, nr), dtype=np.float64)
        self.phi_new = np.empty((nz, nr), dtype=np.float64)
        self.eta1_new = np.empty((nz, nr), dtype=np.float64)
        self.eta2_new = np.empty((nz, nr), dtype=np.float64)


def axisym_corrected_normalized_ownership_step_fast(
        f, eta1, eta2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W,
        scratch: CorrectedNumbaScratch, eps_n=None):
    """One no-flux corrected passive step, returning new scratch arrays."""
    if eps_n is None:
        eps_n = 1.0e-6 / W
    _normalized_phi_kernel(f, eta1, scratch.phi)
    _corrected_mu_kernel(
        f, scratch.phi, p.W_f, p.k_f, Wc, p.k_eta,
        dr, dz, r_c, r_f, scratch.mu)
    flux_kernel(f, scratch.mu, dr, dz, W, M_s, eps_n,
                scratch.Jr, scratch.Jz)
    div_and_update_kernel(
        f, scratch.Jr, scratch.Jz, r_c, r_f, dr, dz, dt, scratch.f_new)
    _phi_update_kernel(
        scratch.phi, scratch.f_new, Wc, p.k_eta, dr, dz, r_c, r_f,
        dt, M_eta, scratch.phi_new, scratch.eta1_new, scratch.eta2_new)
    return scratch.f_new, scratch.eta1_new, scratch.eta2_new

