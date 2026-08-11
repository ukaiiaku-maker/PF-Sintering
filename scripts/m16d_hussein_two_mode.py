"""Milestone 16D Section 4-9: Hussein et al. two-mode coarsening-induced
stability-transition benchmark.

Two-mode initial condition: eps1_bar=eps2_bar=0.4, lambda/R_cyl=1.14*pi.
`eps1,eps2` are interpreted as the amplitudes of the fundamental and
first-harmonic Fourier components of the radius perturbation (NOT two
separate epsilon-labeled grains -- "two-mode" refers to the R(z)
perturbation having two nonzero Fourier terms, unlike the M16B one-mode
benchmark's eps2=0):

    R(z,0) = R0 * [1 + eps1_bar*cos(2*pi*z/lambda) + eps2_bar*cos(4*pi*z/lambda + phi)]

This is a documented interpretive choice (the exact paper figure is not
directly available in this environment) -- adding the first harmonic to
the fundamental breaks the up-down symmetry of a pure single-mode
perturbation, producing an ASYMMETRIC bump/trough pattern. The domain
is ONE period (L=lambda, not 2*lambda -- see build_two_mode's own
docstring for why 2*lambda was tried first and rejected), with GBs
placed at the two DETECTED troughs of the actual constructed profile,
giving the two GB-bounded grains genuinely different sizes -- exactly
the "inner grain" vs "outer grain" asymmetry the milestone's Section 8
requires for the coarsening-induced mechanism to have something to act
on.

Tracks all Section 7 observables and evaluates the sharp-interface
constrained-Hessian stability eigenvalue (pf_sintering.
sharp_interface_stability) along the trajectory at each sample time,
using the CURRENT R(z) profile as the base state for a fresh
(un-relaxed -- Section 7 wants the INSTANTANEOUS eigenvalue of the
ACTUAL evolving state, not a separately-relaxed proxy) eigenmode
calculation.
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
from m16a_gb_benchmark import measure_R_of_z, psi_to_gamma_gb  # noqa: E402
from m16b_young_herring_test import measure_gb_position  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16d_campaign")


class P:
    def __init__(self, gamma_s=1.0, gamma_gb=1.0, W=20e-9):
        self.gamma_s = gamma_s
        self.W = W
        self.k_f = 3 * gamma_s * W
        self.W_f = 12 * gamma_s / W
        self.k_eta = gb_obstacle_coefficients(gamma_gb, W)["k_eta"]


def build_two_mode(Nz0, Nr, dr, dz, R0, W, eps1_bar=0.4, eps2_bar=0.4, lam_over_Rcyl=1.14 * math.pi,
                    harmonic_phase=math.pi / 4):
    """R(z,0)=R0*[1+eps1_bar*cos(2*pi*z/lam)+eps2_bar*cos(4*pi*z/lam+phi)].

    IMPORTANT CONSTRUCTION NOTE (a real issue found and fixed, not
    silently avoided): a bare fundamental+harmonic sum (`phi=0`) is
    EXACTLY periodic with period `lam` (both cosine terms individually
    repeat every `lam`), so evaluating it over a domain L=2*lam (this
    project's established two-GB-per-domain convention) gives two
    IDENTICAL troughs -- verified directly (both minima came out
    0.5503*R0 to 4 decimal places) -- defeating the entire point of a
    two-mode benchmark (an inner grain that can shrink relative to an
    outer one requires genuinely UNEQUAL grain sizes). A nonzero
    relative phase between the fundamental and harmonic
    (`harmonic_phase`, default pi/4) breaks this degeneracy and
    produces two clearly unequal minima (verified: pi/4 gives minima at
    ~0.706*R0 and ~0.408*R0) while keeping both eps1_bar and eps2_bar
    at the specified 0.4 amplitude. This IS a documented interpretive
    construction choice (the exact Hussein Figure-4 geometry is not
    directly available in this environment), not an attempt to
    reproduce the paper's numbers exactly -- see module docstring."""
    lam = lam_over_Rcyl * R0
    # L=lam (a SINGLE period, not 2*lam): the fundamental+phase-shifted-
    # harmonic profile is period-lam overall (harmonic period lam/2
    # divides lam exactly regardless of phase), so it has exactly TWO
    # troughs and TWO crests per lam -- exactly the "one inner grain +
    # one outer grain" bicrystal topology needed, with NO redundant
    # repeated copy (an earlier version used L=2*lam, this project's
    # established one-mode convention, but that silently duplicated
    # the SAME asymmetric pattern twice, putting both GBs at the SAME
    # trough type instead of one at each -- caught and fixed).
    L_target = lam
    Nz = max(24, round(L_target / dz))
    z = (np.arange(Nz) + 0.5) * dz
    r = (np.arange(Nr) + 0.5) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    L = Nz * dz
    Rprofile_1d = R0 * (1.0 + eps1_bar * np.cos(2 * math.pi * z / lam)
                         + eps2_bar * np.cos(4 * math.pi * z / lam + harmonic_phase))
    Rprofile = np.broadcast_to(Rprofile_1d[:, None], (Nz, Nr))
    f = 0.5 * (1.0 - np.tanh((R - Rprofile) / W))

    # GBs at the two DETECTED trough (local minimum) positions of the
    # actual constructed profile (Section 4: "GBs at trough
    # positions"), not an assumed fixed fraction of the domain --
    # nearest-GB (1-D Voronoi, periodic) ownership assignment, smoothed
    # over ~2 grid cells to avoid a sharp/aliased partition boundary.
    is_min = (Rprofile_1d < np.roll(Rprofile_1d, 1)) & (Rprofile_1d < np.roll(Rprofile_1d, -1))
    gb_idx = list(np.where(is_min)[0])
    if len(gb_idx) < 2:
        gb_idx = [int(np.argmin(Rprofile_1d)), int(np.argmin(Rprofile_1d[Nz // 2:]) + Nz // 2)]
    gb_idx = sorted(gb_idx[:2])
    idx = np.arange(Nz)
    d0 = np.minimum(np.abs(idx - gb_idx[0]), Nz - np.abs(idx - gb_idx[0]))
    d1 = np.minimum(np.abs(idx - gb_idx[1]), Nz - np.abs(idx - gb_idx[1]))
    smooth_cells = 1.5
    g_1d = 0.5 * (1.0 + np.tanh((d0 - d1) / smooth_cells))  # ->1 near GB1's neighbor grain, 0 near GB2's
    G = np.broadcast_to(g_1d[:, None], (Nz, Nr))
    e2 = f * G
    e1 = f - e2
    return f, e1, e2, L, lam, Rprofile_1d, gb_idx


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


def instantaneous_eigenvalue(R_of_z, dz, gamma_s, gamma_gb, n_modes=1):
    """Section 7's instantaneous lambda_min(t): apply the sharp-interface
    Hessian DIRECTLY to the current (measured, not separately relaxed)
    R(z) profile -- a periodic topology, GBs located at the two
    resolvable troughs of the CURRENT profile (nearest local minima),
    not fixed grid indices (since GB position moves)."""
    Nz = len(R_of_z)
    if not np.all(np.isfinite(R_of_z)):
        return float("nan"), []
    # find the (up to) two troughs: local minima of R_of_z (periodic)
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


def run_case(psi_deg, R0_nm=40.0, W_nm=20.0, dx_nm=2.5, t_target=60.0, n_sample=60):
    R0 = R0_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    Nr = max(24, round((R0 * 1.6 + 6 * W) / dr))

    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]
    f, e1, e2, L, lam, Rprofile0, gb_idx0 = build_two_mode(10, Nr, dr, dz, R0, W)
    Nz = f.shape[0]
    r_c, r_f = r_centers_faces(Nr, dr)
    z = (np.arange(Nz) + 0.5) * dz
    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    V1_0, V2_0 = grain_volumes(e1, e2, r_c, dr, dz)

    rows = []
    step = 0
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W)
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"[psi={psi_deg}] blew up at step {step}")
        R_of_z = measure_R_of_z(f, r_c)
        V1, V2 = grain_volumes(e1, e2, r_c, dr, dz)
        V = axisym_volume(f, r_c, dr, dz)
        F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)
        lam_min, gb_idx = instantaneous_eigenvalue(R_of_z, dz, 1.0, gamma_gb)
        crest = float(np.nanmax(R_of_z))
        trough = float(np.nanmin(R_of_z))
        rows.append(dict(step=step, t=step * dt, V1_nm3=V1 * 1e27, V2_nm3=V2 * 1e27,
                          V_total_nm3=V * 1e27, mass_drift=(V - V0) / V0,
                          R_crest_nm=crest * 1e9, R_trough_nm=trough * 1e9,
                          lambda_min=lam_min, n_gb_troughs=len(gb_idx), F=F))
        print(f"  [psi={psi_deg:.0f}] t={rows[-1]['t']:.3e} V1={V1*1e27:.4f}nm3 V2={V2*1e27:.4f}nm3 "
              f"R_crest={crest*1e9:.3f}nm R_trough={trough*1e9:.4f}nm lambda_min={lam_min:.4e} "
              f"n_troughs={len(gb_idx)} F={F:.4e} mass_drift={rows[-1]['mass_drift']:.2e}")

    return dict(psi_deg=psi_deg, R0_nm=R0_nm, W_nm=W_nm, dx_nm=dx_nm, lam_nm=lam * 1e9, L_nm=L * 1e9,
                dt=dt, n_steps_total=n_steps_total, V1_0_nm3=V1_0 * 1e27, V2_0_nm3=V2_0 * 1e27,
                gb_idx0=gb_idx0, Rprofile0_nm=(Rprofile0 * 1e9).tolist(), rows=rows)


if __name__ == "__main__":
    out_path = os.path.join(CAMPAIGN_DIR, "hussein_two_mode.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
        with open(out_path) as fh:
            results = json.load(fh)
    else:
        results = []
        for psi_deg in (160.0, 80.0):
            r = run_case(psi_deg, t_target=60.0, n_sample=60)
            results.append(r)
            with open(out_path, "w") as fh:
                json.dump(results, fh, default=str)
        print(f"saved to {out_path}")

    print("\n--- Sections 7-9 summary ---")
    for r in results:
        row0, rowf = r["rows"][0], r["rows"][-1]
        print(f"psi={r['psi_deg']:.0f}: lambda_min(0)={row0['lambda_min']:.4e} -> "
              f"lambda_min(final)={rowf['lambda_min']:.4e}  V2: {row0['V2_nm3']:.3f} -> {rowf['V2_nm3']:.3f} nm^3  "
              f"R_trough: {row0['R_trough_nm']:.3f} -> {rowf['R_trough_nm']:.3f} nm")
