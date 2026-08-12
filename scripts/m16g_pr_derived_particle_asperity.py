"""Milestone 16G: P-R-derived particle-on-asperity sink-off test.

Construction (Sections 1-5 of the handoff): start from the EXACT
Figure-4 two-mode geometry (e1bar=e2bar=0.4, lambda/Rcyl=1.14*pi, same
formula and GB positions as scripts/m16e_exact_hussein_two_mode.py /
scripts/m16f_sharp_figure4_dynamics.py), retain the profile EXACTLY from
the substrate crest (z=0, an exact reflection-symmetry point of the
two-mode profile: R'(0)=0 by construction since both cosine terms are
even about z=0) through the left GB trough (z1) and the full particle
body out to the right GB trough (z2, which the un-modified two-mode
formula gives IDENTICAL radius to z1 -- verified in M16E/M16F to 12+
significant figures), then REMOVE the second (right) outer-grain lobe
entirely and close the particle off with a smooth ellipsoidal free-end
cap starting at z2.

Topology: substrate (grain2, z in [0,z1]) | particle (grain1, z in
[z1, z_end]) | free surface, one GB at z1, one free end. No second GB,
no periodic replication -- domain is OPEN (bc_z="noflux" throughout,
Milestone 16G's extension of axisym_gb_face_projected_step, see
pf_sintering/axisym.py), giving an EXACT no-flux/reservoir-free
boundary at both z=0 (substrate crest, a true symmetry plane of the
retained profile) and z=z_end (deep in the capped free-surface tail).

The particle/substrate eta split uses a SINGLE tanh step at z1 only
(ind_inner=0.5*(1+tanh((z-z1)/smooth)), no second factor cutting it back
off at z2 the way the original two-mode-per-period construction did) --
grain1 (the particle) owns everything from z1 through the cap, grain2
(the substrate) owns everything from z=0 to z1. This is the direct
"remove the third grain" operation: material that used to belong to the
SECOND (right-hand, wrap-around) grain2 lobe is reassigned to grain1
instead of being deleted, so the only thing removed is the extra GB and
the extra crest -- not any solid mass.

Ellipsoidal cap (Section 4): for z>=z2,
    rho(z,r) = sqrt( ((z-z2)/a_cap)^2 + (r/R(z2))^2 )
    f_cap(z,r) = 0.5*(1 - tanh((rho-1)*R(z2)/W))
matches the R(z)-profile tanh construction EXACTLY in both value and
z-slope at the z=z2 seam (both sides have zero z-slope there: the
two-mode profile because z2 is one of its analytic critical points, the
ellipsoid because d(rho)/dz=0 at z=z2 by construction) -- no kink, no ad
hoc blending. `cap_frac` sets a_cap=cap_frac*R(z2); the primary
construction uses cap_frac=1.5, the Section-4-required sensitivity
control uses cap_frac=3.0 (a substantially more elongated cap, tested
once, not swept).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.axisym import (  # noqa: E402
    axisym_free_energy_gb, axisym_gb_face_projected_step, axisym_volume, r_centers_faces,
)
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from m16e_exact_hussein_two_mode import EPS1, EPS2, LAM_OVER_RCYL, P, R0_OVER_RCYL, R_profile_over_Rcyl, Z1_OVER_LAM, Z2_OVER_LAM, psi_to_gamma_gb  # noqa: E402
from m16f_sharp_figure4_dynamics import fit_two_mode_and_stability_ratio  # noqa: E402
from pf_sintering.hussein_eq4_reference import lambda_c_over_R_eq4  # noqa: E402

CAMPAIGN_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "m16g_campaign")


def build_particle_asperity_geometry(R_cyl, W, dr, dz, cap_frac=1.5, tail_margin_W=4.0):
    """Returns dict(f, e1, e2, z, r_c, r_f, Nz, Nr, dz, dr, z1, z2, a_cap,
    R_z1, ind_inner). e1=particle (grain1), e2=substrate (grain2)."""
    lam = LAM_OVER_RCYL * R_cyl
    z1 = Z1_OVER_LAM * lam
    z2 = Z2_OVER_LAM * lam
    R_z1 = R_cyl * float(R_profile_over_Rcyl(np.array([Z1_OVER_LAM]))[0])
    R_z2 = R_cyl * float(R_profile_over_Rcyl(np.array([Z2_OVER_LAM]))[0])
    assert abs(R_z1 - R_z2) < 1e-6 * R_cyl, "left/right trough radii must match (Figure-4 property)"

    a_cap = cap_frac * R_z2
    z_end = z2 + a_cap + tail_margin_W * W

    Nz = max(48, round(z_end / dz))
    dz_actual = z_end / Nz
    z = (np.arange(Nz) + 0.5) * dz_actual

    R_max = R_cyl * (R0_OVER_RCYL + EPS1 + EPS2)  # crest value at z=0
    Nr = max(24, round((R_max + 6 * W) / dr))
    r_c, r_f = r_centers_faces(Nr, dr)

    Z, Rg = np.meshgrid(z, r_c, indexing="ij")

    # R(z) profile branch (z<=z2): exact two-mode formula, unchanged.
    R_profile = R_cyl * R_profile_over_Rcyl(z / lam)
    f_profile = 0.5 * (1.0 - np.tanh((Rg - R_profile[:, None]) / W))

    # Ellipsoidal cap branch (z>=z2): exact value/slope match at z2 (see
    # module docstring derivation).
    rho = np.sqrt(((Z - z2) / a_cap) ** 2 + (Rg / R_z2) ** 2)
    f_cap = 0.5 * (1.0 - np.tanh((rho - 1.0) * R_z2 / W))

    is_cap = z >= z2
    f = np.where(is_cap[:, None], f_cap, f_profile)

    smooth = 1.5 * dz_actual
    ind_inner = 0.5 * (1.0 + np.tanh((z - z1) / smooth))  # single step at z1 ONLY
    e1 = f * ind_inner[:, None]
    e2 = f - e1

    return dict(f=f, e1=e1, e2=e2, z=z, r_c=r_c, r_f=r_f, Nz=Nz, Nr=Nr, dz=dz_actual, dr=dr,
                z1=z1, z2=z2, a_cap=a_cap, R_z1=R_z1, lam=lam, R_cyl=R_cyl)


# ---------------------------------------------------------------------------
# M16G revision: C3-smooth free end (the C1 ellipsoidal-seam construction
# above is retained ONLY as a historical/provisional control, per
# instruction -- its R'' changes sign discontinuously at z2, which is not
# acceptable for a surface-diffusion qualification run since the surface
# flux depends on grad(mu) and mu depends on curvature).
#
# Three-segment meridional contour, R(z) single-valued throughout (the
# terminal ellipsoid cap's own equatorial radius R3 is chosen well above
# the diffuse-interface width W, so "R(z)-r" tanh construction stays
# well-posed -- the true near-axis closure singularity of the OLD
# ellipsoid-at-the-trough construction is avoided entirely here because
# the terminal cap starts at a much smaller radius/steeper convergence
# scale chosen explicitly, not at the full trough radius):
#
#   Segment 1 (z in [0,z2]):  exact two-mode Figure-4 profile, unchanged.
#   Segment A (z in [z2,z3]): a SEPTIC (degree-7) Hermite polynomial in
#       u=(z-z2)/L, L=z3-z2, matching (R,R',R'',R''') of the exact
#       profile at z2 (computed analytically, closed form) on the left,
#       and (R,R',R'',R''') of the terminal ellipsoid cap's own analytic
#       near-equator Taylor expansion at z3 on the right -- an exact
#       symbolic match, not a numerical fit, so the whole R(z) exists in
#       C^3 (four continuous derivatives) by construction.
#   Segment B (z in [z3,z_end]): ellipsoidal terminal cap,
#       rho=sqrt(((z-z3)/a_cap)^2+(r/R3)^2), f=0.5*(1-tanh((rho-1)*R3/W)) --
#       same level-set device as the C1 cap, but now started from R3 (a
#       deliberately chosen SMALL equatorial radius, not the full trough
#       radius), so its own near-z3 Taylor expansion is
#       R(z)=R3*sqrt(1-((z-z3)/a_cap)^2) = R3 - R3/(2*a_cap^2)*(z-z3)^2 + ...
#       (even in z-z3, so R'(z3)=0 and R'''(z3)=0 automatically, and
#       R''(z3)=-R3/a_cap^2) -- these three values are exactly what
#       Segment A's right-hand boundary conditions are built from below.
# ---------------------------------------------------------------------------

def two_mode_R_derivs(z, lam, R_cyl):
    """R,R',R'',R''' of the exact two-mode profile at axial position z
    (absolute, meters), closed form (verified against central finite
    differences to 7 significant figures at z2 during development)."""
    w1, w2 = 2 * math.pi / lam, math.pi / lam
    R0 = R0_OVER_RCYL
    R = R_cyl * (R0 + EPS1 * math.cos(w1 * z) + EPS2 * math.cos(w2 * z))
    Rp = R_cyl * (-EPS1 * w1 * math.sin(w1 * z) - EPS2 * w2 * math.sin(w2 * z))
    Rpp = R_cyl * (-EPS1 * w1 ** 2 * math.cos(w1 * z) - EPS2 * w2 ** 2 * math.cos(w2 * z))
    Rppp = R_cyl * (EPS1 * w1 ** 3 * math.sin(w1 * z) + EPS2 * w2 ** 3 * math.sin(w2 * z))
    return R, Rp, Rpp, Rppp


def septic_hermite_coeffs(bc_left, bc_right, L):
    """bc_left/bc_right = (R,R',R'',R''') in ABSOLUTE (z, not u) units at
    u=0 and u=1 respectively, u=(z-z_start)/L. Returns the 8 coefficients
    c0..c7 of R(u)=sum c_k u^k, solved by exact 8x8 linear algebra (not a
    hand-derived closed form -- robust, easy to re-verify numerically)."""
    R0, Rp0, Rpp0, Rppp0 = bc_left
    R1, Rp1, Rpp1, Rppp1 = bc_right
    targets = np.array([R0, Rp0 * L, Rpp0 * L ** 2, Rppp0 * L ** 3,
                         R1, Rp1 * L, Rpp1 * L ** 2, Rppp1 * L ** 3])
    A = np.zeros((8, 8))
    ks = np.arange(8)
    A[0, 0] = 1.0  # R(0)=c0
    A[1, 1] = 1.0  # dR/du(0)=c1
    A[2, 2] = 2.0  # d2R/du2(0)=2*c2
    A[3, 3] = 6.0  # d3R/du3(0)=6*c3
    A[4, :] = 1.0  # R(1)=sum c_k
    A[5, :] = ks  # dR/du(1)=sum k*c_k
    A[6, :] = ks * (ks - 1)  # d2R/du2(1)=sum k(k-1)c_k
    A[7, :] = ks * (ks - 1) * (ks - 2)  # d3R/du3(1)=sum k(k-1)(k-2)c_k
    c = np.linalg.solve(A, targets)
    return c


def septic_hermite_eval(c, z, z_start, L):
    """Evaluates R(z) (and its z-derivatives, for diagnostics) of the
    Hermite segment at absolute z (array or scalar), z in [z_start,
    z_start+L]."""
    u = (np.asarray(z, dtype=float) - z_start) / L
    ks = np.arange(8)
    R = sum(c[k] * u ** k for k in range(8))
    Rp = sum(c[k] * k * u ** max(k - 1, 0) * (1 if k >= 1 else 0) for k in range(1, 8)) / L
    Rpp = sum(c[k] * k * (k - 1) * u ** max(k - 2, 0) for k in range(2, 8)) / L ** 2
    return R, Rp, Rpp


def build_particle_asperity_geometry_c3(R_cyl, W, dr, dz, R3_frac=0.5, a_cap_frac=4.0,
                                         L_frac=3.0, tail_margin_W=16.0):
    """C3-smooth free-end construction (see module-section docstring
    above). R3_frac sets the Hermite/ellipsoid handoff radius
    (R3=R3_frac*R(z2)); a_cap_frac sets the ellipsoid's own axial semi-
    axis (a_cap=a_cap_frac*R3); L_frac sets the Hermite transition's
    axial length (L=L_frac*R(z2)).

    Parameter choice (development note, not re-derived at call time): a
    three-stage audit drove this choice. (1) Shape: the Hermite
    segment's overshoot/undershoot relative to its own endpoint radii
    (an unavoidable feature of an 8-condition degree-7 polynomial forced
    through a curvature-SIGN REVERSAL with zero slope pinned at both
    ends -- classic Hermite/Runge-type ripple, not a bug) shrinks as
    L_frac decreases. (2) The PEAK CURVATURE MAGNITUDE reached during
    the transition (max|R''|) does the OPPOSITE -- it shrinks as L_frac
    INCREASES and (with rapidly diminishing returns past a_cap_frac~4)
    as a_cap_frac increases. Checking the actual mu/flux field this
    curvature produces showed shape overshoot alone is the wrong thing
    to minimize: at the shape-optimal L_frac=0.5, the transition-region
    flux exceeded the natural near-GB flux several-fold -- a "dominant
    flux pulse". (3) A large ellipsoid aspect ratio (a_cap/R3) also
    STRETCHES the diffuse-interface width near the elongated tip (the
    (rho-1)*R3/W level set is only width-W at the equator); a
    gradient-normalized ("true distance") correction was tried and
    rejected -- it fixed the far-field decay but made the near-axis tip
    itself ill-conditioned (a much worse flux artifact exactly where
    cleanliness matters most). The final choice instead caps the aspect
    ratio at a_cap_frac=4 (peak curvature already close to its
    a_cap_frac->infinity floor at that point, per point (2)) and widens
    tail_margin_W to 16 so the (now-understood, ~4x-widened) tip
    interface fully decays before the no-flux boundary (verified
    directly: f<1e-3 well before the domain edge, vs. 0.27 -- not
    decayed at all -- with the original tail_margin_W=4). Primary:
    R3_frac=0.5, a_cap_frac=4.0, L_frac=3.0 (max|R''|~1.7e7/m, ~1.5x the
    natural GB-trough scale ~1.15e7/m; transition |Jz| peak below the
    natural near-GB |Jz| peak). The Section-8 cap-independence control
    uses a "meaningfully longer" cap, L_frac=4.0 (same a_cap_frac)."""
    lam = LAM_OVER_RCYL * R_cyl
    z1 = Z1_OVER_LAM * lam
    z2 = Z2_OVER_LAM * lam
    R_z2, _, _, _ = two_mode_R_derivs(z2, lam, R_cyl)

    bc_left = two_mode_R_derivs(z2, lam, R_cyl)  # (R,R',R'',R''') at z2, from the exact profile
    R3 = R3_frac * R_z2
    a_cap = a_cap_frac * R3
    L = L_frac * R_z2
    z3 = z2 + L
    z_end = z3 + a_cap + tail_margin_W * W

    # Right-hand BCs at z3: exact near-equator Taylor expansion of the
    # terminal ellipsoid (R3, R'=0, R''=-R3/a_cap^2, R'''=0 -- see
    # section docstring derivation).
    bc_right = (R3, 0.0, -R3 / a_cap ** 2, 0.0)
    c = septic_hermite_coeffs(bc_left, bc_right, L)

    Nz = max(48, round(z_end / dz))
    dz_actual = z_end / Nz
    z = (np.arange(Nz) + 0.5) * dz_actual

    R_max = R_cyl * (R0_OVER_RCYL + EPS1 + EPS2)
    Nr = max(24, round((R_max + 6 * W) / dr))
    r_c, r_f = r_centers_faces(Nr, dr)
    Z, Rg = np.meshgrid(z, r_c, indexing="ij")

    R_profile = R_cyl * R_profile_over_Rcyl(z / lam)
    f_seg1 = 0.5 * (1.0 - np.tanh((Rg - R_profile[:, None]) / W))

    u = (z - z2) / L
    R_hermite = np.zeros_like(z)
    for k in range(8):
        R_hermite += c[k] * np.clip(u, 0.0, 1.0) ** k
    f_segA = 0.5 * (1.0 - np.tanh((Rg - R_hermite[:, None]) / W))

    # Ellipsoidal cap, same level-set device as the C1 construction
    # (rho-1)*R3/W. NOTE: a gradient-normalized ("true distance")
    # variant was tried and rejected during development -- it fixed the
    # slow axial decay far from the tip but produced a much WORSE
    # near-axis flux artifact (the gradient regularizer needed near
    # r=0/z=z3+a_cap made the normalization itself ill-conditioned right
    # at the tip, which is exactly the region that must be clean). The
    # simple (rho-1)*R3/W form is kept, with the aspect ratio a_cap/R3
    # capped at a_cap_frac=4 (not 8) and a generously widened
    # tail_margin_W (see build_particle_asperity_geometry_c3's caller
    # default) so the domain is large enough for the (correctly
    # understood, now accounted-for) elongation-widened interface to
    # decay before the no-flux boundary.
    rho = np.sqrt(((Z - z3) / a_cap) ** 2 + (Rg / R3) ** 2)
    f_segB = 0.5 * (1.0 - np.tanh((rho - 1.0) * R3 / W))

    is_A = (z >= z2) & (z < z3)
    is_B = z >= z3
    f = np.where(is_B[:, None], f_segB, np.where(is_A[:, None], f_segA, f_seg1))

    smooth = 1.5 * dz_actual
    ind_inner = 0.5 * (1.0 + np.tanh((z - z1) / smooth))
    e1 = f * ind_inner[:, None]
    e2 = f - e1

    return dict(f=f, e1=e1, e2=e2, z=z, r_c=r_c, r_f=r_f, Nz=Nz, Nr=Nr, dz=dz_actual, dr=dr,
                z1=z1, z2=z2, z3=z3, R3=R3, a_cap=a_cap, L=L, hermite_coeffs=c,
                R_z1=R_z2, lam=lam, R_cyl=R_cyl)


def contour_derivs_c3(geom, z_query):
    """Evaluates the C3 contour's R,R',R'' at arbitrary z (for the
    continuity audit, Section 6) by dispatching to whichever of the
    three segments z_query falls in."""
    z1, z2, z3, L, R3, a_cap = geom["z1"], geom["z2"], geom["z3"], geom["L"], geom["R3"], geom["a_cap"]
    lam, R_cyl, c = geom["lam"], geom["R_cyl"], geom["hermite_coeffs"]
    if z_query < z2:
        return two_mode_R_derivs(z_query, lam, R_cyl)
    elif z_query < z3:
        R, Rp, Rpp = septic_hermite_eval(c, z_query, z2, L)
        return float(R), float(Rp), float(Rpp), float("nan")
    else:
        # ellipsoid branch, equatorial-region Taylor-consistent formula
        dz = z_query - z3
        s = 1.0 - (dz / a_cap) ** 2
        if s <= 0:
            return 0.0, float("nan"), float("nan"), float("nan")
        R = R3 * math.sqrt(s)
        Rp = -R3 * (dz / a_cap ** 2) / math.sqrt(s)
        Rpp = -R3 / a_cap ** 2 * (1.0 / math.sqrt(s) + (dz / a_cap) ** 2 / s ** 1.5)
        return R, Rp, Rpp, float("nan")


def grain_volumes(e1, e2, r_c, dr, dz):
    V1 = 2 * math.pi * float(np.sum(r_c[None, :] * e1)) * dr * dz
    V2 = 2 * math.pi * float(np.sum(r_c[None, :] * e2)) * dr * dz
    return V1, V2


def find_gb_trough(R_of_z, z, z1_guess, search_frac=0.35, lam=None):
    """Locate the current GB (contact) position as the local minimum of
    R(z) nearest the ORIGINAL z1 (GB migration is expected to be a small
    perturbation on the mesh scale over the run durations used here, so
    "nearest" is unambiguous). search_frac*lam bounds the search window
    so the substrate-crest region (also technically a local max, not
    min) and the cap tail cannot be mistaken for the GB."""
    if lam is None:
        lam = z1_guess  # fallback, not expected to be hit
    window = search_frac * lam
    mask = np.abs(z - z1_guess) < window
    idx_local = np.where(mask)[0]
    if len(idx_local) < 3:
        return float("nan"), float("nan")
    Rw = R_of_z[idx_local]
    if not np.all(np.isfinite(Rw)):
        good = np.isfinite(Rw)
        if good.sum() < 3:
            return float("nan"), float("nan")
    j = int(np.nanargmin(Rw))
    idx = idx_local[j]
    return float(z[idx]), float(R_of_z[idx])


def find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100):
    dt = 1.0
    for _ in range(300):
        f_t, e1_t, e2_t = f.copy(), e1.copy(), e2.copy()
        stable = True
        for _ in range(n_check):
            f_t, e1_t, e2_t, _ = axisym_gb_face_projected_step(
                f_t, e1_t, e2_t, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W, bc_z="noflux")
            if not np.all(np.isfinite(f_t)) or np.max(np.abs(f_t)) > 2.0:
                stable = False
                break
        if stable:
            return dt
        dt *= 0.5
    raise RuntimeError("could not find stable dt")


def run_case(psi_deg=160.0, R_cyl_nm=100.0, W_nm=10.0, dx_nm=1.25, cap_frac=1.5,
             t_target=30.0, n_sample=40, tag=""):
    R_cyl = R_cyl_nm * 1e-9
    W = W_nm * 1e-9
    dr = dz = dx_nm * 1e-9
    gamma_gb = psi_to_gamma_gb(psi_deg)
    p = P(gamma_s=1.0, gamma_gb=gamma_gb, W=W)
    Wc = gb_obstacle_coefficients(gamma_gb, W)["Wc"]

    geom = build_particle_asperity_geometry(R_cyl, W, dr, dz, cap_frac=cap_frac)
    f, e1, e2 = geom["f"], geom["e1"], geom["e2"]
    z, r_c, r_f = geom["z"], geom["r_c"], geom["r_f"]
    Nz, Nr = geom["Nz"], geom["Nr"]
    z1, lam = geom["z1"], geom["lam"]
    a0 = geom["R_z1"]

    M_s = 1e-33
    M_eta = 1e-33 / (W * (32.0 / 35.0))

    dt = find_stable_dt(f, e1, e2, p, Wc, dr, dz, r_c, r_f, M_s, M_eta, W, n_check=100)
    dt *= 0.4
    n_steps_total = max(1, int(t_target / dt))
    sample_steps = sorted(set(round(k * n_steps_total / n_sample) for k in range(n_sample + 1)))

    V0 = axisym_volume(f, r_c, dr, dz)
    Vp0, Vs0 = grain_volumes(e1, e2, r_c, dr, dz)

    gamma_ratio = 2.0 * math.cos(math.radians(psi_deg) / 2.0)

    rows = []
    step = 0
    t_wall0 = time.time()
    z_gb_prev = z1
    for target in sample_steps:
        while step < target:
            f, e1, e2, diag = axisym_gb_face_projected_step(
                f, e1, e2, p, Wc, dr, dz, r_c, r_f, dt, M_s, M_eta, W, bc_z="noflux")
            step += 1
            if not np.all(np.isfinite(f)):
                raise RuntimeError(f"[psi={psi_deg}] blew up at step {step}")
        R_of_z = measure_R_of_z(f, r_c)
        z_gb, a = find_gb_trough(R_of_z, z, z_gb_prev, lam=lam)
        if np.isfinite(z_gb):
            z_gb_prev = z_gb
        Vp, Vs = grain_volumes(e1, e2, r_c, dr, dz)
        V = axisym_volume(f, r_c, dr, dz)
        F = axisym_free_energy_gb(f, e1, e2, p, Wc, dr, dz, r_c, r_f)

        # local two-mode fit restricted to the still-analytic body z<z2,
        # for the parent-geometry stability-ratio comparison (Section 10)
        try:
            fit = fit_two_mode_and_stability_ratio(R_of_z[np.isfinite(R_of_z)], z[np.isfinite(R_of_z)], R_cyl, psi_deg)
        except Exception:
            fit = dict(e1bar=float("nan"), e2bar=float("nan"), lam_over_lamc=float("nan"))

        rows.append(dict(step=step, t=step * dt, Vp_frac=Vp / Vp0, Vs_frac=Vs / Vs0,
                          mass_drift=(V - V0) / V0, z_gb_nm=z_gb * 1e9, a_nm=a * 1e9,
                          a_over_a0=a / a0, A_GB_nm2=math.pi * (a * 1e9) ** 2, F=F,
                          R_of_z_nm=(R_of_z * 1e9).tolist()))
        print(f"  [psi={psi_deg:.0f} cap={cap_frac:.2f}] t={rows[-1]['t']:.4f} "
              f"Vp/Vp0={rows[-1]['Vp_frac']:.6f} Vs/Vs0={rows[-1]['Vs_frac']:.6f} "
              f"a={a*1e9:.4f}nm a/a0={rows[-1]['a_over_a0']:.6f} z_gb={z_gb*1e9:.3f}nm "
              f"mass_drift={rows[-1]['mass_drift']:.2e} wall={time.time()-t_wall0:.1f}s")

    return dict(psi_deg=psi_deg, R_cyl_nm=R_cyl_nm, W_nm=W_nm, dx_nm=dx_nm, cap_frac=cap_frac,
                Nz=Nz, Nr=Nr, dt=dt, n_steps_total=n_steps_total, a0_nm=a0 * 1e9,
                Vp0=Vp0, Vs0=Vs0, z1_nm=z1 * 1e9, lam_nm=lam * 1e9, rows=rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi", type=float, default=160.0)
    ap.add_argument("--t-target", type=float, default=30.0)
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--cap-frac", type=float, default=1.5)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    out_path = os.path.join(CAMPAIGN_DIR, f"psi{args.psi:g}_cap{args.cap_frac:g}{args.tag}.json")
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    if os.path.exists(out_path):
        print(f"already done: {out_path}")
    else:
        result = run_case(args.psi, t_target=args.t_target, n_sample=args.n_sample, cap_frac=args.cap_frac, tag=args.tag)
        with open(out_path, "w") as fh:
            json.dump(result, fh, default=str)
        print(f"saved to {out_path}")
