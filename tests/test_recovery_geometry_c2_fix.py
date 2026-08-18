"""Recovery: regression test for the C2 quintic TJ-to-body transition
(`solve_body_c2_transition`), replacing the old C1-only circular fillet
(`solve_body_fillet`, kept for reference but no longer used by
`particle_R_of_z`/`substrate_R_of_z_sphere`).

See MILESTONE_RECOVERY_GEOMETRY_FIX.md for the full root-cause account:
the circular fillet left a discontinuous curvature jump (37.6x on the
particle side, 92.7x with a sign reversal on the substrate side, for the
production chi=1.5/ratio=0.185/psi=160deg parameters) at the fillet-to-
body tangent point, a plausible direct cause of the neck groove/bump
found in the zero-barrier recovery campaign.
"""
from __future__ import annotations

import math

import numpy as np

from pf_sintering.m16j_geometry import (
    _quintic_hermite_eval,
    solve_body_c2_transition,
    solve_body_fillet,
    young_herring_slope,
)

PSI_DEG = 160.0
R_P_NM = 1000.0
CHI = 1.5
RATIO = 0.185


def _geometric_curvature(coeffs, z):
    R, dR, d2R = _quintic_hermite_eval(coeffs, z)
    return R, dR, d2R / (1.0 + dR ** 2) ** 1.5


def test_c2_transition_matches_body_exactly_at_join():
    m = young_herring_slope(PSI_DEG)
    a = RATIO * R_P_NM
    for R_body, zc_body, slope_sign in ((R_P_NM, R_P_NM, +1.0), (CHI * R_P_NM, -CHI * R_P_NM, -1.0)):
        res = solve_body_c2_transition(a, m, R_body, zc_body, slope_sign)
        z1 = res["z1"]
        R_e, dR_e, kappa_e = _geometric_curvature(res["coeffs"], np.array([z1]))
        # exact sphere values at z1
        dz1 = z1 - zc_body
        R1_true = math.sqrt(R_body ** 2 - dz1 ** 2)
        dRdz1_true = -dz1 / R1_true
        kappa1_true = (-1.0 - dRdz1_true ** 2) / R1_true / (1.0 + dRdz1_true ** 2) ** 1.5
        assert abs(R_e[0] - R1_true) < 1e-6
        assert abs(dR_e[0] - dRdz1_true) < 1e-8
        assert abs(kappa_e[0] - kappa1_true) < 1e-8


def test_c2_transition_matches_tj_position_and_slope():
    m = young_herring_slope(PSI_DEG)
    a = RATIO * R_P_NM
    for R_body, zc_body, slope_sign in ((R_P_NM, R_P_NM, +1.0), (CHI * R_P_NM, -CHI * R_P_NM, -1.0)):
        res = solve_body_c2_transition(a, m, R_body, zc_body, slope_sign)
        R_e, dR_e, _ = _geometric_curvature(res["coeffs"], np.array([0.0]))
        assert abs(R_e[0] - a) < 1e-9
        assert abs(dR_e[0] - slope_sign * m) < 1e-9


def test_c2_transition_curvature_has_no_large_jump_or_sign_reversal_at_join():
    """The defect this fix targets, directly quantified: the OLD circular
    fillet's curvature jumped discontinuously (with a sign reversal on
    the substrate side) at the join. The NEW construction must not."""
    m = young_herring_slope(PSI_DEG)
    a = RATIO * R_P_NM
    for R_body, zc_body, slope_sign in ((R_P_NM, R_P_NM, +1.0), (CHI * R_P_NM, -CHI * R_P_NM, -1.0)):
        res = solve_body_c2_transition(a, m, R_body, zc_body, slope_sign)
        z1 = res["z1"]
        z_eval = np.linspace(0.0, z1, 500)
        R_e, dR_e, kappa_e = _geometric_curvature(res["coeffs"], z_eval)
        # R always INCREASES moving away from the TJ (index 0) toward the body
        # (index -1), regardless of branch: z_eval itself runs from 0 toward
        # z1 (positive for the particle, negative for the substrate).
        assert np.all(np.diff(R_e) > 0), "R(z) must stay monotonic along the transition"
        # no interior curvature sign change beyond what the two endpoints already have
        kappa0, kappa1 = kappa_e[0], kappa_e[-1]
        if kappa0 * kappa1 >= 0:
            assert not np.any(np.diff(np.sign(kappa_e)) != 0), "no spurious curvature sign reversal"
        # curvature must not overshoot far beyond the endpoint range (old fillet
        # overshot by 37.6x-92.7x; require well under 1 order of magnitude)
        k_lo, k_hi = min(kappa0, kappa1), max(kappa0, kappa1)
        span = max(k_hi - k_lo, 1e-8)
        overshoot = max(0.0, k_lo - kappa_e.min()) + max(0.0, kappa_e.max() - k_hi)
        assert overshoot < 0.5 * span, f"curvature overshoot too large: {overshoot} vs span {span}"


def test_old_circular_fillet_is_confirmed_discontinuous_at_join_for_regression_context():
    """Documents (does not require fixing) the OLD construction's known
    defect, so a future change to `solve_body_fillet` that accidentally
    makes it C2 (and thus safe to reintroduce) does not silently break
    this test's own assumption."""
    m = young_herring_slope(PSI_DEG)
    a = RATIO * R_P_NM
    fil = solve_body_fillet(a, m, R_P_NM, R_P_NM, +1.0)
    rho = fil["rho"]
    kappa_fillet = 1.0 / rho
    kappa_body = 1.0 / R_P_NM
    assert kappa_fillet / kappa_body > 10.0, "sanity check: the old fillet's known curvature mismatch"


def test_dihedral_angle_matches_psi_target():
    """Matches scripts/recovery_geometry_audit.py's directly-measured
    result (159.7deg via numerical arclength differentiation of the
    actual constructed branches): the outward tangent ray of each branch
    makes angle atan2(1,m) with the OTHER branch's own axis direction, so
    the included angle between the two rays is 2*atan2(1,m)."""
    m = young_herring_slope(PSI_DEG)
    included = 2.0 * math.degrees(math.atan2(1.0, m))
    assert abs(included - PSI_DEG) < 1.0
