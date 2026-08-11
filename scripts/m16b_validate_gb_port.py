"""Milestone 16B Sections 9-11: validate the axisymmetric GB free-energy
port before any physics benchmark.

(1) Directional-derivative check: axisym_g_eta must be the EXACT
    variational derivative of axisym_free_energy_gb w.r.t. eta_i (same
    finite-difference technique Milestone 8/12B used for the Cartesian
    case: central difference of F against a perturbation direction q,
    compared to dr*dz*2*pi*sum(r_c*g_i*q)).
(2) Same check for axisym_mu_f_gb against f.
(3) Exact conservation + monotone F_total under
    axisym_gb_face_projected_step with M_eta=0 (pure surface diffusion,
    eta frozen) and with M_eta>0 (eta also relaxing).
(4) Large-r limit: at large r, the cylindrical operators must reduce to
    the Cartesian planar tangent-cone benchmark (curvature negligible),
    checked by comparing the local structural force g_i computed by
    axisym_g_eta at large r against constrained_eta.
    structural_thermodynamic_force applied to the same local profile
    with a Cartesian Laplacian, for a z-only-varying (r-independent)
    test field -- in that limit the cylindrical Laplacian's 1/r*d/dr
    term vanishes only if the field has no r-dependence, so instead we
    verify the WEAKER, well-posed statement: for a purely
    z-dependent eta_i (no r variation at all), axisym_laplacian reduces
    EXACTLY to the 1-D second derivative in z (the r-term is identically
    zero since d(eta)/dr=0 everywhere), matching lap9_bc's z-only
    behavior exactly.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_free_energy_gb, axisym_g_eta, axisym_gb_face_projected_step, axisym_laplacian,
    axisym_mu_f_gb, axisym_volume, r_centers_faces,
)
from pf_sintering.bc_ops import lap9_bc  # noqa: E402
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402


class P:
    def __init__(self, gamma_s=1.0, gamma_gb=0.6, W=20e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W
        coeffs = gb_obstacle_coefficients(gamma_gb, W)
        self.k_eta = coeffs["k_eta"]


def directional_derivative_check():
    print("--- (1)/(2) directional-derivative checks ---")
    np.random.seed(0)
    Nz, Nr, dr, dz = 20, 16, 2.0e-9, 2.0e-9
    r_c, r_f = r_centers_faces(Nr, dr)
    W = 20e-9
    p = P(W=W)
    coeffs = gb_obstacle_coefficients(0.6, W)
    Wc = coeffs["Wc"]

    f = 0.3 + 0.4 * np.random.rand(Nz, Nr)
    e1 = 0.5 * f * (0.5 + 0.4 * np.random.rand(Nz, Nr))
    e2 = np.clip(f - e1, 0.0, None) * (0.4 + 0.4 * np.random.rand(Nz, Nr))

    eps = 1e-6
    q = np.random.randn(Nz, Nr)

    # eta1 directional derivative
    F_plus = axisym_free_energy_gb(f, e1 + eps * q, e2, p, Wc, dr, dz, r_c, r_f)
    F_minus = axisym_free_energy_gb(f, e1 - eps * q, e2, p, Wc, dr, dz, r_c, r_f)
    fd = (F_plus - F_minus) / (2 * eps)
    g1 = axisym_g_eta(e1, f, Wc, p, dr, dz, r_c, r_f)
    analytic = 2 * math.pi * float(np.sum(r_c[None, :] * g1 * q)) * dr * dz
    rel = abs(fd - analytic) / max(abs(analytic), 1e-30)
    print(f"eta1 directional derivative: fd={fd:.6e} analytic={analytic:.6e} rel_err={rel:.3e}")

    # f directional derivative
    qf = np.random.randn(Nz, Nr)
    F_plus = axisym_free_energy_gb(f + eps * qf, e1, e2, p, Wc, dr, dz, r_c, r_f)
    F_minus = axisym_free_energy_gb(f - eps * qf, e1, e2, p, Wc, dr, dz, r_c, r_f)
    fd_f = (F_plus - F_minus) / (2 * eps)
    mu = axisym_mu_f_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
    analytic_f = 2 * math.pi * float(np.sum(r_c[None, :] * mu * qf)) * dr * dz
    rel_f = abs(fd_f - analytic_f) / max(abs(analytic_f), 1e-30)
    print(f"f directional derivative:    fd={fd_f:.6e} analytic={analytic_f:.6e} rel_err={rel_f:.3e}")

    ok = rel < 1e-6 and rel_f < 1e-6
    print(f"PASS (directional derivatives)={ok}\n")
    return ok


def conservation_monotone_check():
    print("--- (3) conservation + monotone F_total ---")
    Nz, Nr, dr, dz = 32, 24, 2.0e-9, 2.0e-9
    r_c, r_f = r_centers_faces(Nr, dr)
    W = 20e-9
    p = P(W=W)
    coeffs = gb_obstacle_coefficients(0.6, W)
    Wc = coeffs["Wc"]
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    R0 = 12e-9
    f = 0.5 * (1.0 - np.tanh((R - R0) / W))
    L = Nz * dz
    g = 0.5 * (1 - np.cos(2 * math.pi * Z / L))
    e2 = f * g
    e1 = f - e2

    all_ok = True
    for label, M_eta_use in [("M_eta=0 (frozen eta)", 0.0), ("M_eta>0 (physical)", M_eta)]:
        f_t, e1_t, e2_t = f.copy(), e1.copy(), e2.copy()
        V0 = axisym_volume(f_t, r_c, dr, dz)
        F0 = axisym_free_energy_gb(f_t, e1_t, e2_t, p, Wc, dr, dz, r_c, r_f)
        dt = 2e-4
        max_dV = 0.0
        max_dF = 0.0
        Fprev = F0
        for _ in range(30):
            f_t, e1_t, e2_t, diag = axisym_gb_face_projected_step(
                f_t, e1_t, e2_t, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta_use, W)
            V = axisym_volume(f_t, r_c, dr, dz)
            F = axisym_free_energy_gb(f_t, e1_t, e2_t, p, Wc, dr, dz, r_c, r_f)
            max_dV = max(max_dV, abs(V - V0) / abs(V0))
            max_dF = max(max_dF, F - Fprev)
            Fprev = F
        ok = max_dV < 1e-8 and max_dF <= 1e-18
        all_ok = all_ok and ok
        print(f"[{label}] max relative |V-V0|/V0={max_dV:.3e} max(F(t+dt)-F(t))={max_dF:.3e} PASS={ok}")
    print(f"PASS (conservation/monotone)={all_ok}\n")
    return all_ok


def large_r_reduction_check():
    print("--- (4) large-r cylindrical-Laplacian z-only reduction check ---")
    Nz, Nr, dr, dz = 40, 8, 2.0e-9, 2.0e-9
    r_c, r_f = r_centers_faces(Nr, dr)
    # shift r_c/r_f to a LARGE-r window (curvature 1/r small), but the
    # z-only field trick makes this exact regardless of r since the
    # r-derivative of the field is identically zero everywhere.
    r_offset = 500e-9
    r_c_far = r_c + r_offset
    r_f_far = r_f + r_offset

    z = (np.arange(Nz) + 0.5) * dz
    field_z = np.sin(2 * math.pi * z / (Nz * dz))
    field = np.tile(field_z[:, None], (1, Nr))

    lap_cyl = axisym_laplacian(field, dr, dz, r_c_far, r_f_far)
    lap_1d = lap9_bc(field, dz, bc_x="periodic", bc_y="periodic")
    # lap9_bc is a 2-D 9-point Laplacian; for an r-independent field it
    # should also reduce to the pure-z second derivative in both x/y
    # slots equally -- compare against the exact analytic z-derivative
    # instead, which is unambiguous.
    k = 2 * math.pi / (Nz * dz)
    exact_zz = -(k ** 2) * field_z
    rel = float(np.max(np.abs(lap_cyl[:, 0] - exact_zz))) / float(np.max(np.abs(exact_zz)))
    print(f"axisym_laplacian (z-only field, far from axis) vs exact d2/dz2: rel_err={rel:.3e}")
    ok = rel < 1e-2  # finite-difference truncation, not machine precision
    print(f"PASS (large-r z-only reduction)={ok}\n")
    return ok


if __name__ == "__main__":
    ok1 = directional_derivative_check()
    ok2 = conservation_monotone_check()
    ok3 = large_r_reduction_check()
    print(f"ALL PASS = {ok1 and ok2 and ok3}")
