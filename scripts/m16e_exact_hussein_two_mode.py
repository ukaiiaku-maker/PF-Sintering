"""Milestone 16E: exact published Hussein two-mode geometry and
grain-elimination-driven stability transition.

R(z)/R_cyl = R0/R_cyl + eps1*cos(2*pi*z/lambda) + eps2*cos(pi*z/lambda),
domain 0<=z<2*lambda, eps1=eps2=0.4, R0/R_cyl=sqrt(1-(eps1^2+eps2^2)/2).
GBs at the exact analytic trough positions z1,z2 =
(lambda/pi)*acos(-eps2/(4*eps1)), 2*lambda-z1. Grain 1 (INNER, the
small grain) is z1<z<z2 (centered at z=lambda); grain 2 (OUTER) is
0<z<z1 and z2<z<2*lambda, connected through the periodic boundary --
note both troughs have the IDENTICAL minimum radius (verified:
R(z1)=R(z2) to 12 significant figures) -- the inner/outer asymmetry is
entirely a matter of ARC LENGTH (z2-z1 ~ 0.839*lambda for the inner
grain vs ~1.161*lambda for the outer), NOT trough depth. This corrects
a real construction error in the M16D benchmark, which used an ad hoc
phase-shifted harmonic to force unequal trough DEPTHS -- the wrong
mechanism entirely, now replaced with the exact published formula.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume, r_centers_faces  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402
from pf_sintering.sharp_interface_stability import volume_constrained_eigenmodes  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16e_campaign")

EPS1 = 0.4
EPS2 = 0.4
R0_OVER_RCYL = math.sqrt(1.0 - (EPS1 ** 2 + EPS2 ** 2) / 2.0)
LAM_OVER_RCYL = 1.14 * math.pi
Z1_OVER_LAM = (1.0 / math.pi) * math.acos(-EPS2 / (4.0 * EPS1))
Z2_OVER_LAM = 2.0 - Z1_OVER_LAM


def psi_to_gamma_gb(psi_deg, gamma_s=1.0):
    return 2.0 * gamma_s * math.cos(math.radians(psi_deg) / 2.0)


class P:
    def __init__(self, gamma_s=1.0, gamma_gb=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W
        self.k_eta = gb_obstacle_coefficients(gamma_gb, W)["k_eta"]


def R_profile_over_Rcyl(z_over_lam):
    return R0_OVER_RCYL + EPS1 * np.cos(2 * math.pi * z_over_lam) + EPS2 * np.cos(math.pi * z_over_lam)


def build_exact_two_mode(R_cyl, W, dr, dz):
    lam = LAM_OVER_RCYL * R_cyl
    L = 2 * lam
    Nz = max(48, round(L / dz))
    if Nz % 2:
        Nz += 1
    dz_actual = L / Nz
    z = (np.arange(Nz) + 0.5) * dz_actual
    z_over_lam = z / lam
    R_of_z = R_cyl * R_profile_over_Rcyl(z_over_lam)

    z1 = Z1_OVER_LAM * lam
    z2 = Z2_OVER_LAM * lam

    Nr = max(24, round((float(np.max(R_of_z)) + 6 * W) / dr))
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    Rprofile = np.broadcast_to(R_of_z[:, None], (Nz, Nr))
    f = 0.5 * (1.0 - np.tanh((R - Rprofile) / W))

    # grain 1 (inner, small): z1<z<z2; grain 2 (outer): elsewhere,
    # wrapping through the periodic boundary -- smooth indicator with a
    # tanh transition of a couple grid cells at each GB, not a hard cut.
    smooth = 1.5 * dz_actual
    ind_inner = 0.5 * (1 + np.tanh((z - z1) / smooth)) * 0.5 * (1 + np.tanh((z2 - z) / smooth))
    e1 = f * ind_inner[:, None] * np.ones((1, Nr))  # grain 1 = inner
    e2 = f - e1
    return dict(f=f, e1=e1, e2=e2, Nz=Nz, Nr=Nr, dz=dz_actual, dr=dr, lam=lam, L=L, z1=z1, z2=z2,
                z=z, R_cyl=R_cyl, R_of_z0=R_of_z)


def find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=200):
    dt = 1.0
    for _ in range(300):
        f_t, e1_t, e2_t = f.copy(), e1.copy(), e2.copy()
        stable = True
        for _ in range(n_check):
            f_t, e1_t, e2_t, _ = axisym_gb_face_projected_step(f_t, e1_t, e2_t, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            if not np.all(np.isfinite(f_t)) or np.max(np.abs(f_t)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def grain_volumes(e1, e2, r_c, dr, dz):
    V1 = 2 * math.pi * float(np.sum(r_c[None, :] * e1)) * dr * dz
    V2 = 2 * math.pi * float(np.sum(r_c[None, :] * e2)) * dr * dz
    return V1, V2


def hessian_eigenvalue(R_of_z, dz, gamma_s, gamma_gb, n_modes=1):
    Nz = len(R_of_z)
    if not np.all(np.isfinite(R_of_z)):
        return float("nan"), []
    is_min = (R_of_z < np.roll(R_of_z, 1)) & (R_of_z < np.roll(R_of_z, -1))
    gb_indices = list(np.where(is_min)[0])
    if len(gb_indices) == 0:
        gb_indices = [int(np.argmin(R_of_z))]
    try:
        evals, evecs, lam, gnorm = volume_constrained_eigenmodes(
            R_of_z, dz, gamma_s, gamma_gb, gb_indices, "periodic", n_modes=n_modes)
        return float(evals[0]), gb_indices
    except Exception:
        return float("nan"), gb_indices
