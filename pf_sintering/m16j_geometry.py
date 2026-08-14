"""Milestone 16J: geometry-driven stress-concentration search.

Constructs an axisymmetric particle-on-neighbor (or particle-on-flat-
substrate) initial condition whose neck satisfies the EXACT Young-Herring
dihedral-angle relation at the triple junction (TJ) by construction, rather
than relying on phase-field relaxation to correct an arbitrary raw-overlap
angle (Section 8 of the M16J handoff: "the LOCAL TJ angle must not be an
artificial source of a large initial transient").

Physical picture (r,z meridian cross-section, axis of revolution at r=0,
TJ height taken as the z=0 reference):

  - particle (grain e2): sphere/spheroid of radius R_p, "resting" so its
    unperturbed pole would touch (r=0, z=0); connected to the TJ via a
    circular-arc FILLET (radius rho1, i.e. r_neck's particle-side
    contribution) that is analytically tangent (C1: matched position AND
    slope) to the sphere, and passes through the TJ point (a, 0) with the
    EXACT slope dR/dz=+m required by force balance (m=cot(psi/2),
    gamma_gb/gamma_s=2*cos(psi/2) -- same convention as
    pf_sintering.hussein_eq4_reference.psi_to_gamma_ratio and
    pf_sintering.hussein_neck_stress).

  - substrate (grain e1), FINITE neighbor curvature (chi=R_s/R_p finite):
    the mirror construction (sphere of radius R_s below, fillet radius
    rho2, slope -m at the TJ). Both fillet+sphere-cap pieces are expressed
    as a single-valued R(z) (radius as a function of height) -- valid
    because both bodies are locally near-vertical at the TJ (shallow
    dR/dz), which is exactly where an R(z) representation is well-behaved.

  - substrate, EXACT FLAT limit (chi=infinity): a true half-space has NO
    well-defined R(z) (at the flat height, the solid/vapor transition
    occurs at r=infinity, not at a finite radius) -- R(z) is fundamentally
    the wrong representation here. Instead the flat substrate is built as
    a HEIGHT FIELD H(r) (height as a function of radius, well-behaved near
    a near-HORIZONTAL surface): flat at H=h_flat for r>=r_transition, and
    a mirror-image circular-arc groove for a<=r<=r_transition passing
    through the TJ with slope dH/dr=-1/m (the same physical angle,
    expressed in the reciprocal-slope convention appropriate to a
    near-horizontal curve) and tangent to the flat asymptote. The groove
    fillet radius is a free parameter for the flat case (there is no
    second curvature scale to tangent-match against) and is set equal to
    the particle-side fillet radius by default (the natural, symmetric
    choice) via `substrate_flat_fillet_radius`.

Both the finite-R_s and flat constructions are then truncated a modest
margin beyond their own fillet-to-body tangent point (independent of
R_p/R_s, so the domain size stays governed by the neck/interface scale,
not by the particle's physical size -- Sections 7 and 26's "crop
aggressively" instruction) and capped off with a simple, C0-matched cosine
taper (Section 7: far-field capping details were already shown in M16G
not to affect near-neck dynamics -- the "B2" finding -- so an exact C2/C3
cap is not required this far from the neck, only VALUE continuity and no
new local minimum).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def young_herring_slope(psi_deg: float) -> float:
    """|dR/dz| at the TJ for an R(z)-parameterized branch: m=cot(psi/2).
    Matches gamma_gb/gamma_s=2*cos(psi/2) used throughout this codebase
    (pf_sintering.hussein_eq4_reference.psi_to_gamma_ratio)."""
    return 1.0 / math.tan(math.radians(psi_deg / 2.0))


def solve_body_fillet(a: float, m: float, R_body: float, zc_body: float, slope_sign: float):
    """Circular-arc fillet (in the r-z meridian plane) that (1) passes
    through the TJ (a,0) with slope slope_sign*m there, and (2) is
    externally tangent to a sphere of radius R_body centered at
    (0,zc_body). Two algebraic roots exist in general; the physical one
    is selected as the one with r_c > a (verified numerically: the other
    root produces a non-monotonic R(z), i.e. a spurious extra bump).
    Returns dict(r_c,z_c,rho,r_tangent,z_tangent)."""
    def resid(t):
        r_c = a - t
        z_c = slope_sign * m * t
        rho = abs(t) * math.sqrt(1 + m * m)
        return math.hypot(r_c, z_c - zc_body) - (R_body + rho)

    # bracket the r_c>a root: that branch has t<0 for the particle
    # (slope_sign=+1) case verified numerically; search a broad range and
    # keep the sign-change bracket nearest t=0 on the r_c>a (t<0) side.
    ts = np.linspace(-8 * a, -1e-6 * a, 20000)
    vals = np.array([resid(t) for t in ts])
    root = None
    for i in range(len(ts) - 1):
        if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0:
            root = brentq(resid, ts[i], ts[i + 1])
            break
    if root is None:
        raise RuntimeError(f"no r_c>a fillet root found (a={a}, m={m}, R_body={R_body}, zc_body={zc_body})")
    t = root
    r_c = a - t
    z_c = slope_sign * m * t
    rho = abs(t) * math.sqrt(1 + m * m)
    dz = z_c - zc_body
    dr = r_c
    dist = math.hypot(dr, dz)
    r_tangent = R_body * dr / dist
    z_tangent = zc_body + R_body * dz / dist
    return dict(r_c=r_c, z_c=z_c, rho=rho, r_tangent=r_tangent, z_tangent=z_tangent, t=t)


def solve_flat_groove_fillet(a: float, m: float, rho2: float):
    """Height-field H(r) groove fillet for an exact-flat substrate: a
    circular arc through the TJ (a,0) with slope dH/dr=-1/m, with fillet
    radius FIXED at rho2 (a free parameter for the flat case -- no second
    curvature scale exists to tangent-match against; the default caller
    sets rho2 equal to the particle-side fillet radius for a natural
    symmetric neck). Two conditions (passes through the TJ; correct
    slope there) fully determine the circle given rho2 -- a THIRD
    "tangent to h_flat" condition would over-constrain the same circle
    and is not imposed; instead the arc is followed (r monotonically
    increasing, single-valued H(r)) from the TJ around to the circle's
    OWN rightmost point (r_c+rho, vertical tangent -- the natural extent
    of a single-valued arc), where it is capped flat at H=h_c (C0-exact
    at the join, since that IS the arc's value there). This traces a
    genuine dip-then-rise groove (Mullins-type thermal-groove shape),
    verified numerically: H(a)=0, H reaches a minimum near r=r_c, rises
    back to H=h_c at r=r_c+rho.
    Returns dict(r_c,h_c,rho,h_flat,r_transition)."""
    u = rho2 / math.sqrt(1 + m * m)
    r_c = a + u
    h_c = m * u
    r_transition = r_c + rho2
    h_flat = h_c
    return dict(r_c=r_c, h_c=h_c, rho=rho2, h_flat=h_flat, r_transition=r_transition)


def cosine_cap(z, z0, R0, z_end):
    """Simple C0 (value-only) monotonic taper from R(z0)=R0 down to 0 at
    z=z_end, used to close off a truncated sphere/groove far from the
    neck (Section 7: far-cap details do not measurably affect near-neck
    dynamics -- the M16G "B2" precedent)."""
    frac = np.clip((z - z0) / (z_end - z0), 0.0, 1.0)
    return R0 * np.cos(0.5 * math.pi * frac)


def particle_R_of_z(z, a, m, R_p, cap_margin_W, W):
    """R(z) for the particle branch (z>=0): fillet, then sphere cap, then
    a cosine taper closing to 0. `R_p` is the (possibly osculating-sphere-
    equivalent, for AR!=1) particle radius."""
    zc_p = R_p
    fil = solve_body_fillet(a, m, R_p, zc_p, +1.0)
    z_T, r_T = fil["z_tangent"], fil["r_tangent"]
    crop_margin = max(6.0 * W, 4.0 * fil["rho"])
    z_cap0 = z_T + crop_margin
    R_cap0 = math.sqrt(max(R_p * R_p - (z_cap0 - zc_p) ** 2, 0.0))
    if not (R_cap0 > 0 and z_cap0 - zc_p < R_p):
        # crop point would already be past the sphere's own equator --
        # shrink the margin so it stays on the rising part of the sphere.
        z_cap0 = 0.5 * (z_T + zc_p + R_p)
        R_cap0 = math.sqrt(max(R_p * R_p - (z_cap0 - zc_p) ** 2, 0.0))
    z_end = z_cap0 + max(6.0 * W, 0.5 * R_cap0)

    R = np.full_like(z, np.nan)
    m_fillet = (z >= 0) & (z <= z_T)
    r_c1, z_c1, rho1 = fil["r_c"], fil["z_c"], fil["rho"]
    inside = rho1 ** 2 - (z[m_fillet] - z_c1) ** 2
    R[m_fillet] = r_c1 - np.sqrt(np.clip(inside, 0.0, None))

    m_sphere = (z > z_T) & (z <= z_cap0)
    R[m_sphere] = np.sqrt(np.clip(R_p ** 2 - (z[m_sphere] - zc_p) ** 2, 0.0, None))

    m_cap = (z > z_cap0) & (z <= z_end)
    R[m_cap] = cosine_cap(z[m_cap], z_cap0, R_cap0, z_end)

    return R, dict(fillet=fil, z_tangent=z_T, r_tangent=r_T, z_cap0=z_cap0, R_cap0=R_cap0, z_end=z_end)


def substrate_R_of_z_sphere(z, a, m, R_s, cap_margin_W, W):
    """Mirror of particle_R_of_z for a finite-radius substrate/neighbor
    (z<=0 branch)."""
    zc_s = -R_s
    fil = solve_body_fillet(a, m, R_s, zc_s, -1.0)
    z_T, r_T = fil["z_tangent"], fil["r_tangent"]
    crop_margin = max(6.0 * W, 4.0 * fil["rho"])
    z_cap0 = z_T - crop_margin
    R_cap0 = math.sqrt(max(R_s * R_s - (z_cap0 - zc_s) ** 2, 0.0))
    if not (R_cap0 > 0 and zc_s - z_cap0 < R_s):
        z_cap0 = 0.5 * (z_T + zc_s - R_s)
        R_cap0 = math.sqrt(max(R_s * R_s - (z_cap0 - zc_s) ** 2, 0.0))
    z_end = z_cap0 - max(6.0 * W, 0.5 * R_cap0)

    R = np.full_like(z, np.nan)
    m_fillet = (z <= 0) & (z >= z_T)
    r_c2, z_c2, rho2 = fil["r_c"], fil["z_c"], fil["rho"]
    inside = rho2 ** 2 - (z[m_fillet] - z_c2) ** 2
    R[m_fillet] = r_c2 - np.sqrt(np.clip(inside, 0.0, None))

    m_sphere = (z < z_T) & (z >= z_cap0)
    R[m_sphere] = np.sqrt(np.clip(R_s ** 2 - (z[m_sphere] - zc_s) ** 2, 0.0, None))

    m_cap = (z < z_cap0) & (z >= z_end)
    R[m_cap] = cosine_cap(-z[m_cap], -z_cap0, R_cap0, -z_end)

    return R, dict(fillet=fil, z_tangent=z_T, r_tangent=r_T, z_cap0=z_cap0, R_cap0=R_cap0, z_end=z_end)


def substrate_H_of_r_flat(r, a, m, rho2, no_surface_height=1e6):
    """Height field H(r) for an exact-flat substrate's FREE SURFACE only:
    groove fillet for a<=r<=r_transition, exactly flat (H=h_flat) beyond.
    For r<a (under the particle -- no free surface there at all, both
    grains are fully solid up to the internal GB), H is set to a very
    large value so the resulting tanh field reads as "fully solid, no
    nearby transition" rather than spuriously registering the internal
    GB height (z=0) as if it were a free surface -- which grain owns a
    solid point for r<a is decided separately (by z<0 vs z>=0), not by
    this height field."""
    g = solve_flat_groove_fillet(a, m, rho2)
    r_c, h_c, rho, h_flat, r_trans = g["r_c"], g["h_c"], g["rho"], g["h_flat"], g["r_transition"]
    H = np.full_like(r, np.nan)
    m_disk = r < a
    H[m_disk] = no_surface_height
    m_groove = (r >= a) & (r <= r_trans)
    inside = rho ** 2 - (r[m_groove] - r_c) ** 2
    H[m_groove] = h_c - np.sqrt(np.clip(inside, 0.0, None))
    m_flat = r > r_trans
    H[m_flat] = h_flat
    return H, g


def build_candidate_geometry(R_p_nm, R_s_nm, X0_over_2Rp, psi_deg, W_nm, dr_nm, dz_nm,
                              aspect_ratio=1.0, r_max_margin_W=8.0):
    """Builds (f,e1,e2,z,r_c,r_f) for one M16J candidate.

    R_s_nm=None selects the EXACT flat-substrate construction (chi=infinity);
    otherwise R_s_nm is the finite neighbor-curvature radius (nm).
    X0_over_2Rp = X0/(2*R_p) sets the initial contact half-width a=X0/2.
    aspect_ratio!=1 approximates the particle via its osculating-sphere
    radius at the contact pole (Rr^2/Rz) for the fillet-tangency solve --
    an approximation intended for Stage-0/1 screening; Stage-2/3 refines
    promoted candidates.
    """
    m = young_herring_slope(psi_deg)
    a = X0_over_2Rp * R_p_nm
    Rz_p = R_p_nm * math.sqrt(aspect_ratio)
    Rr_p = R_p_nm / math.sqrt(aspect_ratio)
    R_p_eff = Rr_p ** 2 / Rz_p  # osculating-sphere radius at the pole

    flat = R_s_nm is None

    # --- particle-side profile (z>=0), always R(z)-based ---
    z_hi_guess = R_p_eff * 0.6 + 20 * W_nm  # generous upper bound; refined below
    Np_probe = 4000
    z_probe = np.linspace(0.0, z_hi_guess, Np_probe)
    Rp_of_z, p_info = particle_R_of_z(z_probe, a, m, R_p_eff, r_max_margin_W, W_nm)
    z_top = p_info["z_end"]

    if not flat:
        R_s_eff = float(R_s_nm)
        z_lo_guess = -(R_s_eff * 0.6 + 20 * W_nm)
        z_probe_s = np.linspace(z_lo_guess, 0.0, Np_probe)
        Rs_of_z, s_info = substrate_R_of_z_sphere(z_probe_s, a, m, R_s_eff, r_max_margin_W, W_nm)
        z_bottom = s_info["z_end"]
    else:
        fil_p = solve_body_fillet(a, m, R_p_eff, R_p_eff, +1.0)
        rho1 = fil_p["rho"]
        s_info = solve_flat_groove_fillet(a, m, rho1)
        # groove minimum height is h_c-rho (bottom of the fillet circle);
        # crop a margin below that, not just below the TJ.
        groove_min = s_info["h_c"] - s_info["rho"]
        z_bottom = groove_min - max(6.0 * W_nm, 2.0 * rho1)

    Nz = max(64, int(round((z_top - z_bottom) / dz_nm)))
    dz_actual = (z_top - z_bottom) / Nz
    z = z_bottom + (np.arange(Nz) + 0.5) * dz_actual

    R_max_particle = max(p_info["R_cap0"], a) + r_max_margin_W * W_nm
    if not flat:
        R_max_sub = max(s_info["R_cap0"], a) + r_max_margin_W * W_nm
        r_max = max(R_max_particle, R_max_sub)
    else:
        r_max = max(R_max_particle, s_info["r_transition"]) + r_max_margin_W * W_nm

    Nr = max(32, int(round(r_max / dr_nm)))
    dr_actual = r_max / Nr
    r_c = (np.arange(Nr) + 0.5) * dr_actual
    r_f = np.arange(Nr + 1) * dr_actual

    Z, Rg = np.meshgrid(z, r_c, indexing="ij")

    R_particle_z, _ = particle_R_of_z(z, a, m, R_p_eff, r_max_margin_W, W_nm)
    e2 = 0.5 * (1.0 - np.tanh((Rg - np.nan_to_num(R_particle_z, nan=-1e9)[:, None]) / W_nm))
    e2 = np.where((z >= 0)[:, None], e2, 0.0)

    if not flat:
        R_sub_z, _ = substrate_R_of_z_sphere(z, a, m, R_s_eff, r_max_margin_W, W_nm)
        R_full = np.where(z >= 0, np.nan_to_num(R_particle_z, nan=-1e9), np.nan_to_num(R_sub_z, nan=-1e9))
        f = 0.5 * (1.0 - np.tanh((Rg - R_full[:, None]) / W_nm))
        smooth = 1.5 * dz_actual
        ind_inner = 0.5 * (1.0 + np.tanh((0.0 - z) / smooth))  # ~1 for z<0 (substrate)
        e1 = f * ind_inner[:, None]
        e2 = f - e1
    else:
        H_r, _ = substrate_H_of_r_flat(r_c, a, m, s_info["rho"])
        f_substrate = 0.5 * (1.0 - np.tanh((Z - H_r[None, :]) / W_nm))
        f = np.maximum(f_substrate, e2)
        smooth = 1.5 * dz_actual
        ind_inner = 0.5 * (1.0 + np.tanh((0.0 - z) / smooth))  # ~1 for z<0 (substrate identity)
        e1 = f * ind_inner[:, None]
        e2 = f - e1

    return dict(f=f, e1=e1, e2=e2, z=z, r_c=r_c, r_f=r_f, Nz=Nz, Nr=Nr, dz=dz_actual, dr=dr_actual,
                a_contact=a, m_target=m, R_p_nm=R_p_nm, R_s_nm=R_s_nm, flat=flat,
                aspect_ratio=aspect_ratio, R_p_eff=R_p_eff, psi_deg=psi_deg, W_nm=W_nm,
                particle_info=p_info, substrate_info=s_info)


def find_all_extrema(R_of_z, z):
    """Local minima/maxima of a 1D array (finite points only), for QC
    (Section 15: reject a candidate with more than one neck). Tolerant of
    a PLATEAU of tied values (e.g. a symmetric chi=1 construction whose
    grid cells straddle the neck exactly, giving two equal minimum-R
    cells rather than a single strict minimum) -- a plateau bounded by
    strictly larger (for a min) or smaller (for a max) neighbors on both
    sides still counts as exactly one extremum, reported at its
    midpoint."""
    finite = np.isfinite(R_of_z)
    idx = np.where(finite)[0]
    if len(idx) < 3:
        return []
    Rv = R_of_z[idx]
    extrema = []
    k = 1
    n = len(idx)
    while k < n - 1:
        R0, R1 = Rv[k - 1], Rv[k]
        if R1 == R0:
            k += 1
            continue
        # find the end of a plateau starting at k (if any)
        j = k
        while j + 1 < n and Rv[j + 1] == Rv[j]:
            j += 1
        if j + 1 >= n:
            break
        R2 = Rv[j + 1]
        if R1 < R0 and R1 < R2:
            i_mid = idx[(k + j) // 2]
            extrema.append(("min", z[i_mid], R1))
        elif R1 > R0 and R1 > R2:
            i_mid = idx[(k + j) // 2]
            extrema.append(("max", z[i_mid], R1))
        k = j + 1
    return extrema
