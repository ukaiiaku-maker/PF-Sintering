"""Curvature-compatible TJ mass deposition (recovery, geometry-aware
redistribution).

Replaces the historical Gaussian-in-z redistribution kernel (the
positive-feedback-prone construction M16S root-caused and this
recovery's own surface-relaxation test showed forces the production
driver into an impractically expensive PF-subcycling regime just to
keep up) with a deposition that is ALREADY geometrically smooth: the
required excess/deficit volume is placed as a small, C2-continuous
OUTWARD NORMAL DISPLACEMENT of the actual free-surface branches
(particle-side and neighbor-side, each starting at the physical TJ),
using a compact bump profile chosen (by a cheap 1-D search over width
and branch partition) to minimize the resulting curvature-gradient
heterogeneity `integral (dK/ds)^2 * 2*pi*r ds`.

Physics preserved unchanged: advection/RBM kinematics, the one-b event
quota, Coble kinetics, M_s/M_eta, barrier parameters, the C2 initial
geometry, the branch-resolved TJ locator. Only the SPATIAL SHAPE of the
excess/deficit-volume redistribution changes -- the volume itself
(`Delta_V = V_excess - V_deficit`) is exactly the same quantity the old
Gaussian kernel already computed.
"""
from __future__ import annotations

import math

import numpy as np

PHI_SUPPORT = 1.0  # phi(x) is supported on x in [0,1]


def phi_bump(x):
    """C2 bump: phi(0)=phi'(0)=phi''(0)=0, phi(1)=phi'(1)=phi''(1)=0,
    64*x^3*(1-x)^3 on [0,1], zero outside."""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    m = (x >= 0) & (x <= 1)
    out[m] = 64.0 * x[m] ** 3 * (1.0 - x[m]) ** 3
    return out


def smootherstep_down(x):
    """C2 monotonic decay from 1 (x<=0) to 0 (x>=1): 1 - (6x^5-15x^4+10x^3).
    Zero first AND second derivative at both x=0 and x=1 (Perlin's
    quintic smoothstep). Used only to taper the far (arbitrary-cutoff)
    end of a branch response so it decays smoothly, without touching the
    physically-solved interior shape."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 1.0 - (6.0 * x ** 5 - 15.0 * x ** 4 + 10.0 * x ** 3)


def _locate_contour_and_tj(f, particle, substrate, r_c, z, z_tj_guess):
    """Computes the f=0.5 contour and locates the physical TJ ONCE (both
    are relatively expensive full-grid scans) -- shared by both branch
    extractions below, instead of each branch redundantly recomputing
    them (a ~2x cost reduction, Section 13's performance requirement)."""
    import sys
    sys.path.insert(0, "scripts")
    from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
    from pf_sintering.m16k_neck_tracking import find_tj_from_contour

    R_of_z = measure_R_of_z(f, r_c)
    tj = find_tj_from_contour(f, particle, substrate, r_c, z, R_of_z, z_tj_guess, search_frac=0.3)
    z_tj_nm, r_tj_nm = tj["z_tj"] * 1e9, tj["r_tj"] * 1e9
    z_nm = z * 1e9
    finite = np.isfinite(R_of_z)
    idx = np.where(finite)[0]
    return dict(z_all_nm=z_nm[idx], R_all_nm=R_of_z[idx] * 1e9, z_tj_nm=z_tj_nm, r_tj_nm=r_tj_nm)


def extract_branch(f, particle, substrate, r_c, z, z_tj_guess, side, n_probe=None, contour=None):
    """Extracts one free-surface branch (side='particle' for z>=z_tj,
    'neighbor' for z<=z_tj) as arrays (s_nm, r_nm, z_nm) ordered by
    arclength from the TJ, using the existing measure_R_of_z contour
    (grid-resolution samples -- fine enough for the deposition-region
    length scales, L~4-12*W~24-72nm, spanning many grid cells).

    `contour` (optional): pre-computed `_locate_contour_and_tj(...)`
    result, to avoid re-scanning the full grid when called for both
    branches back-to-back (see `extract_both_branches`)."""
    if contour is None:
        contour = _locate_contour_and_tj(f, particle, substrate, r_c, z, z_tj_guess)
    z_tj_nm, r_tj_nm = contour["z_tj_nm"], contour["r_tj_nm"]
    z_all_nm, R_all_nm = contour["z_all_nm"], contour["R_all_nm"]
    if side == "particle":
        mask = z_all_nm >= z_tj_nm
        order = np.argsort(z_all_nm[mask])
    else:
        mask = z_all_nm <= z_tj_nm
        order = np.argsort(-z_all_nm[mask])
    z_b = z_all_nm[mask][order]
    R_b = R_all_nm[mask][order]
    if len(z_b) < 3:
        return None
    ds = np.hypot(np.diff(z_b), np.diff(R_b))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return dict(s_nm=s, r_nm=R_b, z_nm=z_b, z_tj_nm=z_tj_nm, r_tj_nm=r_tj_nm)


def extract_both_branches(f, particle, substrate, r_c, z, z_tj_guess):
    """Locates the contour/TJ once, extracts both branches from it."""
    contour = _locate_contour_and_tj(f, particle, substrate, r_c, z, z_tj_guess)
    br_p = extract_branch(f, particle, substrate, r_c, z, z_tj_guess, "particle", contour=contour)
    br_n = extract_branch(f, particle, substrate, r_c, z, z_tj_guess, "neighbor", contour=contour)
    return br_p, br_n


def branch_K_of_s(s_nm, r_nm, z_nm):
    """Total curvature K(s)=kappa_m(s)+kappa_a(s) (meridional + azimuthal,
    matching M16's `kappa_a=z'/r` convention) on a UNIFORM-in-s resample
    for stable finite differences. Returns (s_uniform, r_u, z_u, theta,
    kappa_m, kappa_a, K)."""
    s_u = np.linspace(s_nm[0], s_nm[-1], len(s_nm))
    r_u = np.interp(s_u, s_nm, r_nm)
    z_u = np.interp(s_u, s_nm, z_nm)
    dr = np.gradient(r_u, s_u)
    dz = np.gradient(z_u, s_u)
    norm = np.hypot(dr, dz)
    dr, dz = dr / np.maximum(norm, 1e-30), dz / np.maximum(norm, 1e-30)
    theta = np.arctan2(dz, dr)
    d2r = np.gradient(dr, s_u)
    d2z = np.gradient(dz, s_u)
    kappa_m = dr * d2z - dz * d2r
    kappa_a = np.where(r_u > 1e-9, dz / np.maximum(r_u, 1e-9), 0.0)
    K = kappa_m + kappa_a
    return s_u, r_u, z_u, theta, kappa_m, kappa_a, K


def displaced_branch(s_u, r_u, z_u, u_of_s):
    """Applies a small OUTWARD normal displacement u(s) (nm) to the
    branch, returning the displaced (r,z). Outward normal = tangent
    rotated -90deg (points away from the solid into vapor, verified by
    sign convention: for the particle branch r increasing with s moving
    away from the axis, outward should increase r for the free surface
    facing +r -- checked numerically in the caller's test)."""
    dr = np.gradient(r_u, s_u)
    dz = np.gradient(z_u, s_u)
    norm = np.hypot(dr, dz)
    dr, dz = dr / np.maximum(norm, 1e-30), dz / np.maximum(norm, 1e-30)
    # outward normal: rotate tangent (dr,dz) by -90deg -> (dz,-dr); sign
    # chosen so a positive u pushes the surface AWAY from the axis/solid
    # (increasing r for a vertically-oriented branch) -- matches "outward".
    n_r, n_z = dz, -dr
    r_new = r_u + u_of_s * n_r
    z_new = z_u + u_of_s * n_z
    return r_new, z_new


def solve_branch_response(s_u, r_u, K_pre, M_s, W_nm, taper_start_frac=0.8, beta=1.0, method="thomas"):
    """Physically-motivated replacement for the old fixed-shape C2 bump
    AND the arbitrary p/L search it required.

    Rather than choosing a deposition width/partition by discrete search,
    solve for the branch's own diffusive COMPLIANCE: the shape u_hat(s)
    (and its own swept volume V_hat) that a unit Lagrange-multiplier
    'pressure' at the TJ would produce, given this branch's actual
    mobility- and curvature-weighted resistance to smooth displacement.
    This is the Euler-Lagrange solution of

        minimize  integral_0^L [ c(s) * (du/ds)^2 ] ds
        subject to  u(0)=0, u(L)=0

    with  c(s) = (1 + beta*(K_pre(s)*W)^2) / (r(s)*M_s)  -- i.e. the
    standard 'flux = -M_s * d(mu)/ds', mu=gamma_s*Omega*K capillary
    picture translated into a diffusive smoothness COST: LOWER mobility
    or HIGHER existing curvature (mu is more sensitive to further
    bending there) makes a point more 'resistant' to carrying a steep
    displacement gradient, exactly matching M_s and K(s) as the ONLY
    inputs, with NO free/searched partition parameter. The Euler-Lagrange
    equation for this functional is the tridiagonal linear BVP
        d/ds[ c(s) du/ds ] = -r(s),   u(0)=u(L)=0
    (a uniform-multiplier source, i.e. every branch element is 'equally
    keen' to accept volume up to its own local cost c(s)) -- solved once
    per branch as a dense O(N) linear solve (N~100-150, matching the
    'O(100) unknowns, not a 2D PDE' requirement).

    The actual per-branch displacement for a target branch volume
    Delta_V_b is then u_b(s) = (Delta_V_b/V_hat) * u_hat(s) (exact by
    linearity). The CROSS-BRANCH volume partition is NOT imposed here --
    the caller derives it from V_hat_particle/V_hat_neighbor (see
    `select_and_apply_deposition`): the more 'compliant' branch (larger
    V_hat, i.e. more volume absorbed per unit shared TJ 'pressure') gets
    a proportionally larger share, exactly the emergent, non-arbitrary
    partition requested (no p={0.25,0.5,0.75} search).

    The far end (s=L, an arbitrary finite cutoff, not a real physical
    boundary) is tapered by a C2 `smootherstep_down` window over the last
    `1-taper_start_frac` of the branch so u_hat, u_hat', u_hat'' all -> 0
    there smoothly (the raw BVP solution only guarantees u_hat(L)=0, not
    a zero derivative, which would otherwise show up as a slope
    discontinuity/curvature spike at the arbitrary window edge).

    `method`: 'thomas' (default, production -- O(N) tridiagonal solve) or
    'dense' (O(N^3) `np.linalg.solve` on the same assembled system, kept
    only as a validation reference, see `scripts/*_thomas_equivalence*.py`).
    Both call the SAME `_assemble_tridiagonal_bvp` (identical diagonal/
    off-diagonal coefficients, RHS, and boundary rows) and apply the SAME
    taper/normalization afterward -- only the linear-algebra kernel
    differs."""
    N = len(s_u)
    if N < 5:
        return np.zeros(N), 0.0
    lower, diag, upper, rhs = _assemble_tridiagonal_bvp(s_u, r_u, K_pre, M_s, W_nm, beta)
    if method == "thomas":
        u_hat = _solve_thomas(lower, diag, upper, rhs)
    elif method == "dense":
        u_hat = _solve_dense(lower, diag, upper, rhs)
    else:
        raise ValueError(f"unknown method {method!r}")
    L_nm = s_u[-1] - s_u[0]
    window = smootherstep_down((s_u - s_u[0] - taper_start_frac * L_nm) / max((1 - taper_start_frac) * L_nm, 1e-30))
    u_hat = u_hat * window
    V_hat = 2.0 * math.pi * float(np.trapezoid(r_u * u_hat, s_u))
    return u_hat, V_hat


def gaussian_diffusion_response(s_u, r_u, sigma_nm):
    """SUPERSEDED as the production path (kept for provenance/comparison
    only) -- see `curvature_bump_response` below. Unit-amplitude Gaussian
    source-diffusion branch response: u_hat(s) = exp(-s^2/(2*sigma^2)) /
    r(s), maximal AT s=0, smooth everywhere.

    Why this alone is NOT enough (found by direct measurement, NOT a
    hypothetical concern): `measure_branch_resolved_sigma` does not
    average curvature over its window -- `branch_local_curvature` fits a
    spline through the window's points but evaluates and returns
    curvature ONLY at the branch's own innermost (TJ-nearest) sample,
    `_spline_curvature_at_end`. A perturbation this function's own
    u_hat(0)=1/r(0) offset creates is, at s=0, LOCALLY FLAT to leading
    order in s (u_hat''(0) = -u_hat(0)/sigma^2, which for sigma~50-300nm
    is negligible against the ~0.01-0.05/nm curvature changes actually
    observed within the first 1-2 grid cells) -- confirmed directly:
    disabling redistribution entirely vs. using this function changed
    sigma_avg by <2% at every dose tested (see
    MILESTONE_RIGID_RBM_CORRECTION_AND_PRODUCTION.md). Depositing volume
    broadly is not the same as relieving the LOCAL curvature the
    measurement actually reads.

    `solve_branch_response`/`_assemble_tridiagonal_bvp`/the Thomas and
    dense solvers below are kept UNCHANGED, unused by the default path,
    for provenance and as an explicit opt-in alternative
    (`method="compliance_bvp"` in `select_and_apply_deposition`)."""
    sigma_eff = max(float(sigma_nm), 1e-6)
    u_hat = np.exp(-0.5 * (s_u / sigma_eff) ** 2) / np.maximum(r_u, 1e-9)
    V_hat = 2.0 * math.pi * float(np.trapezoid(r_u * u_hat, s_u))
    return u_hat, V_hat


def curvature_bump_response(s_u, r_u, sigma_nm):
    """PRODUCTION deposition response (replaces `gaussian_diffusion_response`
    as the default -- see that function's docstring for why a broad
    displacement bump alone does not work).

    Directly prescribes the CURVATURE correction this deposit should
    produce, as a Gaussian bump in arclength centered at the TJ (s=0)
    with width sigma_nm=sqrt(2*D*t) (same physical diffusion-length
    definition as before -- D the physical diffusivity, t the real
    elapsed time of the step this deposit represents):

        d2u_hat/ds2 (s) = exp(-s^2/(2*sigma^2))

    i.e. a UNIT-amplitude curvature-relief profile, maximal exactly at
    s=0 (where `measure_branch_resolved_sigma`'s endpoint-curvature
    evaluation actually reads), decaying smoothly over the physical
    length sigma -- this is what "minimize the curvature locally...
    within the neck region at distances limited by x^2~Dt" means made
    concrete: the CORRECTION applied to curvature is itself localized to
    that lengthscale, not merely the volume.

    u_hat(s) itself is obtained by integrating twice (cumulative
    trapezoidal, numerically -- avoids an error-function closed form for
    no benefit) with the natural conditions u_hat(0)=0, u_hat'(0)=0 (a
    smooth, unforced start -- NOT a Dirichlet pin on volume, merely the
    arbitrary integration reference point; unlike the compliance BVP's
    u(0)=0, this does NOT suppress the SECOND derivative at s=0, which
    is exactly the quantity that matters here). The far end (beyond
    ~3*sigma, where the Gaussian curvature target is already numerically
    negligible) is tapered via the same C2 `smootherstep_down` window
    `solve_branch_response` already uses, so u_hat and its first
    derivative both settle smoothly to a constant at the window edge --
    no slope or curvature discontinuity at the arbitrary cutoff.

    Returns (u_hat, V_hat) -- same interface as `solve_branch_response`/
    `gaussian_diffusion_response`, drop-in compatible with the caller's
    `lam = dV_target/V_hat; u = lam*u_hat` linear scaling (exact, since
    the whole construction is linear in amplitude)."""
    sigma_eff = max(float(sigma_nm), 1e-6)
    d2u_hat = np.exp(-0.5 * (s_u / sigma_eff) ** 2)
    du_hat = np.concatenate([[0.0], np.cumsum(0.5 * (d2u_hat[:-1] + d2u_hat[1:]) * np.diff(s_u))])
    u_hat = np.concatenate([[0.0], np.cumsum(0.5 * (du_hat[:-1] + du_hat[1:]) * np.diff(s_u))])
    L_nm = s_u[-1] - s_u[0]
    # taper should START after the curvature bump has mostly decayed
    # (~3*sigma), tapering the REMAINING window smoothly to a constant --
    # NOT the reverse (an earlier version of this line was inverted,
    # starting the taper almost immediately and corrupting the intended
    # curvature-bump shape well before 3*sigma, caught by direct
    # numerical inspection of d2u_hat/ds2 against its target).
    taper_start_frac = min(0.9, max(0.05, 3.0 * sigma_eff / max(L_nm, 1e-30)))
    window = smootherstep_down((s_u - s_u[0] - taper_start_frac * L_nm) / max((1 - taper_start_frac) * L_nm, 1e-30))
    u_hat = u_hat * window
    V_hat = 2.0 * math.pi * float(np.trapezoid(r_u * u_hat, s_u))
    return u_hat, V_hat


def _assemble_tridiagonal_bvp(s_u, r_u, K_pre, M_s, W_nm, beta):
    """Assembles d/ds[c(s) du/ds] = -r(s), u(0)=u(L)=0 as a tridiagonal
    system. Returns (lower, diag, upper, rhs), each length N (lower[0]
    and upper[N-1] are structurally zero/unused, Dirichlet rows). Shared,
    UNCHANGED coefficient formulas for both the dense and Thomas solvers
    -- only this one place defines the physics of the BVP."""
    N = len(s_u)
    c = (1.0 + beta * (K_pre * W_nm) ** 2) / np.maximum(r_u * M_s, 1e-300)
    c_face = 0.5 * (c[:-1] + c[1:])
    lower = np.zeros(N)
    diag = np.zeros(N)
    upper = np.zeros(N)
    rhs = np.zeros(N)
    for i in range(1, N - 1):
        dsm, dsp = s_u[i] - s_u[i - 1], s_u[i + 1] - s_u[i]
        cm, cp = c_face[i - 1], c_face[i]
        lower[i] = cm / dsm
        upper[i] = cp / dsp
        diag[i] = -(cm / dsm + cp / dsp)
        rhs[i] = -r_u[i]
    diag[0] = 1.0
    rhs[0] = 0.0
    diag[-1] = 1.0
    rhs[-1] = 0.0
    return lower, diag, upper, rhs


def _solve_dense(lower, diag, upper, rhs):
    """Validation-only reference solve: builds the full N x N matrix from
    the SAME tridiagonal coefficients and solves with np.linalg.solve."""
    N = len(diag)
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = diag[i]
        if i > 0:
            A[i, i - 1] = lower[i]
        if i < N - 1:
            A[i, i + 1] = upper[i]
    return np.linalg.solve(A, rhs)


def _solve_thomas(lower, diag, upper, rhs, pivot_tol=1e-300):
    """Standard Thomas algorithm (tridiagonal Gaussian elimination,
    O(N)). Fails closed: raises FloatingPointError on a near-zero or
    non-finite pivot, or a non-finite solution, rather than silently
    returning a garbage result."""
    N = len(diag)
    cp = np.zeros(N)
    dp = np.zeros(N)
    piv = diag[0]
    if not np.isfinite(piv) or abs(piv) < pivot_tol:
        raise FloatingPointError(f"Thomas solve: singular/non-finite pivot at row 0 (piv={piv})")
    cp[0] = upper[0] / piv
    dp[0] = rhs[0] / piv
    for i in range(1, N):
        piv = diag[i] - lower[i] * cp[i - 1]
        if not np.isfinite(piv) or abs(piv) < pivot_tol:
            raise FloatingPointError(f"Thomas solve: singular/non-finite pivot at row {i} (piv={piv})")
        cp[i] = (upper[i] / piv) if i < N - 1 else 0.0
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / piv
    x = np.zeros(N)
    x[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Thomas solve: non-finite solution")
    return x


def curvature_heterogeneity_cost(s_u, r_u, K_new, s_min_nm=5.0):
    """J = integral (dK/ds)^2 * 2*pi*r ds, excluding the immediate TJ
    grid-noise zone (s<s_min_nm, matching the SAME exclusion used
    throughout this recovery's curvature diagnostics)."""
    dKds = np.gradient(K_new, s_u)
    mask = s_u >= s_min_nm
    if not np.any(mask):
        return float("inf")
    return float(np.trapezoid((dKds[mask] ** 2) * 2 * math.pi * r_u[mask], s_u[mask]))


def _n_sign_changes(x, tol):
    dx = np.diff(x)
    s = np.sign(np.where(np.abs(dx) < tol, 0.0, dx))
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(np.diff(s) != 0))


def has_new_extrema_or_reversal(r_pre, r_new, K_pre, K_new, s_u, s_min_nm=5.0):
    """QC per Section 6: reject candidates that INCREASE the number of
    r(s) local extrema (non-monotonic geometry beyond what the baseline
    geometry already has) or INCREASE the number of K(s) sign changes,
    outside the excluded near-TJ zone. Compared as COUNTS against the
    pre-displacement baseline (not pointwise/absolute-zero), since the
    baseline itself carries small grid-noise wiggles and a tiny
    displacement legitimately shifts an existing zero-crossing's exact
    location without being a new physical feature."""
    mask = s_u >= s_min_nm
    if not np.any(mask):
        return False
    r_tol = 1e-4 * max(float(np.ptp(r_pre[mask])), 1e-6)
    if _n_sign_changes(r_new[mask], r_tol) > _n_sign_changes(r_pre[mask], r_tol):
        return True
    K_tol = 1e-4 * max(float(np.ptp(K_pre[mask])), 1e-9)
    if _n_sign_changes(K_new[mask], K_tol) > _n_sign_changes(K_pre[mask], K_tol):
        return True
    return False


def select_and_apply_deposition(f, particle, substrate, r_c, z, z_tj_guess, V_net_correction, W_m, M_s,
                                 D_m2s=None, dt_seconds=None, n_sigma=4.0, L_window_nm=80.0, beta=1.0,
                                 method="curvature_bump"):
    """Top-level curvature-compatible deposition.

    CORRECTION (production default is now `method="curvature_bump"`,
    `curvature_bump_response`): the deposit directly prescribes the
    CURVATURE correction as a Gaussian in arclength peaked at the TJ
    (s=0) with width `sigma_nm=sqrt(2*D_m2s*dt_seconds)*1e9` -- the
    actual diffusion length x~sqrt(D*t) for the diffusivity (`D_m2s`,
    e.g. hp.D_gb) and real physical duration (`dt_seconds`) of the step
    this deposit represents, both REQUIRED (no default) for
    `method="curvature_bump"` or `method="gaussian_diffusion"`.

    Two earlier attempts are kept for provenance, both empirically ruled
    out by direct dose-test measurement (see
    MILESTONE_RIGID_RBM_CORRECTION_AND_PRODUCTION.md), NOT hypothetically:
    `method="compliance_bvp"` (`solve_branch_response`) imposes u(0)=0,
    structurally excluding the TJ point from ever receiving deposited
    volume; `method="gaussian_diffusion"` (`gaussian_diffusion_response`)
    deposits volume broadly but is locally FLAT at its own peak (s=0),
    so it barely changes the curvature `measure_branch_resolved_sigma`
    actually reads there (that measurement evaluates curvature AT the
    TJ-nearest sample, not a window average -- see
    `curvature_bump_response`'s docstring). Both left sigma_avg
    essentially unchanged (<2% difference from no redistribution at
    all) across every dose tested.

    `V_net_correction` = V_excess - V_deficit, EXACTLY the quantity
    `active_sink_transport_step` passes to `redistribution_fn` -- i.e. the
    UNSCALED `_axisym_weighted_sum` convention (`sum(r_c*field)`, the
    constant `2*pi*dr*dz` factor deliberately omitted there since that
    caller only ever takes ratios of these sums). This function converts
    it to a true physical volume (m^3) internally before use, since here
    the constant does NOT cancel (an absolute displaced-volume target is
    needed, not a ratio). Positive means net excess to remove FROM the
    vapor side by pushing the free surface OUTWARD (a net volume increase
    of solid); the sign/physical mapping matches the old kernel's own
    `f = f + dep/V_dep*V_net_correction` (adding positive `dep` weight
    where `V_net_correction>0`).

    The branch-extraction window is `n_sigma*sigma_nm` (default n_sigma=4)
    for `method="curvature_bump"`/`"gaussian_diffusion"` -- NOT the fixed
    `L_window_nm`, which is retained only for `method="compliance_bvp"`.

    Returns (f_new, particle_new, substrate_new, diag) where diag reports
    the EMERGENT volume partition (`Q_particle_frac`, `Q_neighbor_frac`
    -- NOT prescribed), the achieved volume closure, each branch's own
    response `V_hat`, and `sigma_nm`/`window_nm` actually used.
    """
    dr = float(r_c[1] - r_c[0])
    dz = float(z[1] - z[0])
    W_nm = W_m * 1e9
    dV_total_m3 = V_net_correction * 2.0 * math.pi * dr * dz
    dV_total_nm3 = dV_total_m3 * 1e27  # m^3 -> nm^3

    br_p, br_n = extract_both_branches(f, particle, substrate, r_c, z, z_tj_guess)
    if br_p is None or br_n is None or abs(dV_total_nm3) < 1e-6:
        return f, particle, substrate, dict(skipped=True)

    if method in ("curvature_bump", "gaussian_diffusion"):
        if D_m2s is None or dt_seconds is None:
            raise ValueError(f"method={method!r} requires D_m2s and dt_seconds "
                              "(the real diffusivity and physical step duration) -- no default; "
                              "an unphysical guess here would silently reintroduce an arbitrary "
                              "length scale, exactly what this correction removes.")
        sigma_nm = math.sqrt(2.0 * D_m2s * dt_seconds) * 1e9
        window_nm = n_sigma * sigma_nm
    else:
        sigma_nm = None
        window_nm = L_window_nm

    # restrict to the physically-motivated accommodation window before
    # resampling/solving
    def _windowed(br):
        m = br["s_nm"] <= window_nm
        if int(np.sum(m)) < 5:
            m = np.ones_like(br["s_nm"], dtype=bool)
        return br["s_nm"][m], br["r_nm"][m], br["z_nm"][m]

    s_p_raw, r_p_raw, z_p_raw = _windowed(br_p)
    s_n_raw, r_n_raw, z_n_raw = _windowed(br_n)
    s_pu, r_pu, z_pu, _, _, _, K_p_pre = branch_K_of_s(s_p_raw, r_p_raw, z_p_raw)
    s_nu, r_nu, z_nu, _, _, _, K_n_pre = branch_K_of_s(s_n_raw, r_n_raw, z_n_raw)

    if method == "curvature_bump":
        u_hat_p, V_hat_p = curvature_bump_response(s_pu, r_pu, sigma_nm)
        u_hat_n, V_hat_n = curvature_bump_response(s_nu, r_nu, sigma_nm)
    elif method == "gaussian_diffusion":
        u_hat_p, V_hat_p = gaussian_diffusion_response(s_pu, r_pu, sigma_nm)
        u_hat_n, V_hat_n = gaussian_diffusion_response(s_nu, r_nu, sigma_nm)
    elif method in ("compliance_bvp", "thomas", "dense"):
        # back-compat: old callers passed method="thomas"/"dense" meaning
        # the LINEAR-ALGEBRA KERNEL for the compliance BVP, not the
        # top-level deposition method -- both now mean "compliance_bvp"
        # here, using "thomas" as the BVP's own solver choice.
        bvp_solver = "dense" if method == "dense" else "thomas"
        u_hat_p, V_hat_p = solve_branch_response(s_pu, r_pu, K_p_pre, M_s, W_nm, beta=beta, method=bvp_solver)
        u_hat_n, V_hat_n = solve_branch_response(s_nu, r_nu, K_n_pre, M_s, W_nm, beta=beta, method=bvp_solver)
    else:
        raise ValueError(f"unknown deposition method {method!r}")

    # emergent (NOT imposed) cross-branch partition: each branch draws a
    # share of the shared TJ 'pressure' proportional to its own compliance
    V_hat_sum = V_hat_p + V_hat_n
    if abs(V_hat_sum) > 1e-30:
        q_p_frac = V_hat_p / V_hat_sum
    else:
        q_p_frac = 0.5
    q_n_frac = 1.0 - q_p_frac
    dV_p_nm3 = q_p_frac * dV_total_nm3
    dV_n_nm3 = q_n_frac * dV_total_nm3

    lam_p = (dV_p_nm3 / V_hat_p) if abs(V_hat_p) > 1e-30 else 0.0
    lam_n = (dV_n_nm3 / V_hat_n) if abs(V_hat_n) > 1e-30 else 0.0
    u_p = lam_p * u_hat_p
    u_n = lam_n * u_hat_n

    r_p_new, z_p_new = displaced_branch(s_pu, r_pu, z_pu, u_p)
    r_n_new, z_n_new = displaced_branch(s_nu, r_nu, z_nu, u_n)
    _, _, _, _, _, _, K_p_new = branch_K_of_s(s_pu, r_p_new, z_p_new)
    _, _, _, _, _, _, K_n_new = branch_K_of_s(s_nu, r_n_new, z_n_new)
    qc_failed = (has_new_extrema_or_reversal(r_pu, r_p_new, K_p_pre, K_p_new, s_pu) or
                 has_new_extrema_or_reversal(r_nu, r_n_new, K_n_pre, K_n_new, s_nu))
    J_sel = curvature_heterogeneity_cost(s_pu, r_pu, K_p_new) + curvature_heterogeneity_cost(s_nu, r_nu, K_n_new)

    L_sel_nm = window_nm

    def axisym_vol(field):
        return 2 * math.pi * float(np.sum(r_c[None, :] * field)) * dr * dz

    # map normal displacement -> delta_f via delta_f=(2/W)*u*f*(1-f),
    # applied to cells nearest each branch's own arclength parameterization.
    # Computed into a RAW (unscaled) accumulator first: the linearized
    # delta_f~-u*df/dn mapping is only approximate on the discrete grid, so
    # the achieved swept volume is measured directly and the whole
    # contribution is rescaled ONCE (Section 9) to close dV_total exactly,
    # rather than trusting the continuum integral estimate.
    delta_f_raw = np.zeros_like(f)
    branch_patches = []  # (tag, iz_lo, iz_hi, ir_lo, ir_hi, delta_f_local_raw)
    for br, s_u, u_of_s, tag in ((br_p, s_pu, u_p, "particle"), (br_n, s_nu, u_n, "neighbor")):
        z_tj_nm, r_tj_nm = br["z_tj_nm"], br["r_tj_nm"]
        L_nm = L_sel_nm
        if br is br_p:
            z_lo, z_hi = z_tj_nm - 2 * W_nm, z_tj_nm + L_nm + 2 * W_nm
        else:
            z_lo, z_hi = z_tj_nm - L_nm - 2 * W_nm, z_tj_nm + 2 * W_nm
        r_lo, r_hi = max(0.0, r_tj_nm - L_nm - 2 * W_nm), r_tj_nm + L_nm + 2 * W_nm
        iz_lo = max(0, int(np.searchsorted(z * 1e9, z_lo)) - 2)
        iz_hi = min(len(z), int(np.searchsorted(z * 1e9, z_hi)) + 2)
        ir_lo = max(0, int(np.searchsorted(r_c * 1e9, r_lo)) - 2)
        ir_hi = min(len(r_c), int(np.searchsorted(r_c * 1e9, r_hi)) + 2)
        if iz_hi <= iz_lo or ir_hi <= ir_lo:
            branch_patches.append((tag, iz_lo, iz_hi, ir_lo, ir_hi, None))
            continue
        Zc = z[iz_lo:iz_hi, None] * 1e9 * np.ones((1, ir_hi - ir_lo))
        Rc = r_c[None, ir_lo:ir_hi] * 1e9 * np.ones((iz_hi - iz_lo, 1))
        # nearest branch point (brute-force over the branch's own s-samples;
        # branch has ~L/dz points, small, cheap)
        d2 = (Zc[:, :, None] - br["z_nm"][None, None, :]) ** 2 + (Rc[:, :, None] - br["r_nm"][None, None, :]) ** 2
        nearest = np.argmin(d2, axis=2)
        s_at_cell = br["s_nm"][nearest]
        u_at_cell = np.interp(s_at_cell, s_u, u_of_s, left=0.0, right=0.0)
        f_local = f[iz_lo:iz_hi, ir_lo:ir_hi]
        delta_f_local = (2.0 / W_nm) * u_at_cell * f_local * (1.0 - f_local)
        delta_f_raw[iz_lo:iz_hi, ir_lo:ir_hi] += delta_f_local
        branch_patches.append((tag, iz_lo, iz_hi, ir_lo, ir_hi, delta_f_local))

    V_delta_raw = axisym_vol(delta_f_raw)
    scale = dV_total_m3 / V_delta_raw if abs(V_delta_raw) > 1e-40 else 0.0

    f_new = f.copy()
    particle_new = particle.copy()
    substrate_new = substrate.copy()
    for tag, iz_lo, iz_hi, ir_lo, ir_hi, delta_f_local in branch_patches:
        if delta_f_local is None:
            continue
        scaled = delta_f_local * scale
        f_new[iz_lo:iz_hi, ir_lo:ir_hi] += scaled
        if tag == "particle":
            particle_new[iz_lo:iz_hi, ir_lo:ir_hi] += scaled
        else:
            substrate_new[iz_lo:iz_hi, ir_lo:ir_hi] += scaled

    np.clip(f_new, 0.0, 1.0, out=f_new)
    np.clip(particle_new, 0.0, None, out=particle_new)
    np.clip(substrate_new, 0.0, None, out=substrate_new)

    # renormalize once so particle+substrate=f exactly (Section 9)
    ssum = particle_new + substrate_new
    denom = np.where(ssum > 1e-30, ssum, 1.0)
    scale_grain = np.where(ssum > 1e-30, f_new / denom, 0.0)
    particle_new = particle_new * scale_grain
    substrate_new = substrate_new * scale_grain

    V_before = axisym_vol(f)
    V_after_trial = axisym_vol(f_new)
    achieved_dV_m3 = V_after_trial - V_before
    closure_error = abs(achieved_dV_m3 - dV_total_m3) / max(abs(dV_total_m3), 1e-30)

    diag = dict(skipped=False, method=method, sigma_nm=sigma_nm, window_nm=L_sel_nm, L_window_nm=L_sel_nm,
                J_selected=J_sel, qc_failed=qc_failed,
                Q_particle_frac=q_p_frac, Q_neighbor_frac=q_n_frac,
                V_hat_particle=V_hat_p, V_hat_neighbor=V_hat_n, lam_particle=lam_p, lam_neighbor=lam_n,
                dV_requested_m3=dV_total_m3, dV_achieved_m3=achieved_dV_m3, closure_error=closure_error)
    return f_new, particle_new, substrate_new, diag


def make_redistribution_fn(W_m, M_s, D_m2s=None, verbose=False, fraction_log=None, method="curvature_bump"):
    """Binds W, M_s, and (for the production `method="gaussian_diffusion"`)
    the physical diffusivity `D_m2s`, adapting select_and_apply_deposition
    to the `redistribution_fn(f, e1, e2, r_c, z, GB_z_hint,
    V_net_correction, dt_seconds) -> (f_new, e1_new, e2_new, diag)`
    interface expected by `active_sink_transport_step`/
    `multi_sink_transport_step`/`rigid_translation_sink_step`'s opt-in
    `redistribution_fn` hook.

    CORRECTION: the interface gained a required `dt_seconds` argument
    (the real physical duration of the step this call's deposit
    represents) because the production deposition method now needs it
    to set its Gaussian's diffusion-length width -- `dt_seconds` varies
    every call (it is NOT a fixed binding like `W_m`/`M_s`/`D_m2s`), so
    it must come from the caller, which already computes it locally
    (`dt*seconds_per_model_time`) for its own Coble-kinetics use.

    `method` (default 'gaussian_diffusion'): the production deposition
    method (see `select_and_apply_deposition`'s docstring). 'compliance_bvp'
    (or the old 'thomas'/'dense' spellings, kept for back-compat) selects
    the earlier compliance-BVP method instead, in which case `D_m2s`/
    `dt_seconds` are accepted but unused.

    `fraction_log` (optional list): if given, `(Q_particle_frac,
    Q_neighbor_frac)` is appended to it on every call -- the emergent
    (not imposed) branch partition, for auditing whether it is
    physically-derived and varies with geometry rather than being a
    constant."""
    def _fn(f, e1, e2, r_c, z, GB_z_hint, V_net_correction, dt_seconds=None):
        f_new, e1_new, e2_new, diag = select_and_apply_deposition(
            f, e1, e2, r_c, z, GB_z_hint, V_net_correction, W_m, M_s,
            D_m2s=D_m2s, dt_seconds=dt_seconds, method=method)
        if fraction_log is not None and not diag.get("skipped"):
            fraction_log.append((diag.get("Q_particle_frac"), diag.get("Q_neighbor_frac")))
        if verbose:
            print(f"    [redistribution] V_net_correction={V_net_correction:.4e}m3 "
                  f"Q_p_frac={diag.get('Q_particle_frac')} Q_n_frac={diag.get('Q_neighbor_frac')} "
                  f"sigma_nm={diag.get('sigma_nm')} window_nm={diag.get('window_nm')} "
                  f"V_hat_p={diag.get('V_hat_particle')} V_hat_n={diag.get('V_hat_neighbor')} "
                  f"closure_err={diag.get('closure_error')} qc_failed={diag.get('qc_failed')} "
                  f"skipped={diag.get('skipped')}", flush=True)
        return f_new, e1_new, e2_new, diag
    return _fn
