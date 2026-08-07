"""Synthetic triple-junction validation for pf_sintering/tj_force.py.

Constructs a synthetic diffuse-interface wedge/groove with a prescribed
dihedral angle psi and a straight GB descending from the TJ, so branch
directions, the independent psi measurement, and the vector force balance can
all be checked against closed-form expectations (see module docstring in
tj_force.py for the physical picture).

Geometry: TJ at (x_tj, y_tj). For y <= y_tj the domain is fully solid, split
into e1 (x < x_tj) / e2 (x > x_tj) by a diffuse GB at x = x_tj. For y > y_tj
the solid/vapor boundary is exactly the two rays from the TJ at angles
+-psi/2 from vertical, so the region directly above the TJ is vapor (the
groove opening) and the two "flanks" (each grain's own free surface) are
straight lines -- ideal for a clean angle measurement. This matches a "top"
TJ in the real substrate model (vapor above, solid below).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pf_sintering.model import ModelConfig, Sink, build_params
from pf_sintering.signed_curvature import signed_curvature_at
from pf_sintering.tj_force import cahn_hoffman_vector, compute_tj_force


def _base_params(dx_nm=2.0, nx=64, ny=96, aniso=False, aniso_delta=0.15, theta_mis_deg=30.0):
    p = build_params(ModelConfig(
        preset="dev", nx=nx, ny=ny, dx=dx_nm * 1e-9, r2=40e-9, t_total=1e-6,
        use_aniso_surface=aniso, theta_mis_deg=theta_mis_deg,
    ))
    if aniso:
        p.aniso_delta = aniso_delta
    return p


def build_synthetic_wedge(p, psi_deg, tj_xy=None):
    """Return (f, e1, e2, e3, tj_xy, d1, d2, d_gb) for a synthetic TJ with the
    prescribed dihedral angle psi_deg (angle between the two free-surface
    branch tangents, measured as constructed)."""
    if tj_xy is None:
        tj_xy = np.array([0.5 * p.Nx * p.dx, 0.35 * p.Ny * p.dx])
    x_tj, y_tj = tj_xy
    W = p.interface_width
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)

    half = math.radians(psi_deg) / 2.0
    d1 = np.array([-math.sin(half), math.cos(half)])  # left flank branch tangent
    d2 = np.array([math.sin(half), math.cos(half)])   # right flank branch tangent
    d_gb = np.array([0.0, -1.0])                       # GB tangent, away from TJ (downward)

    dxp, dyp = X - x_tj, Y - y_tj
    s1 = d1[0] * dyp - d1[1] * dxp  # cross(d1, P-tj)
    s2 = d2[0] * dyp - d2[1] * dxp  # cross(d2, P-tj)
    level = np.maximum(s1, -s2)
    f = 0.5 * (1.0 + np.tanh(level / W))

    g = 0.5 * (1.0 + np.tanh((X - x_tj) / W))  # fraction assigned to e2
    e2 = f * g
    e1 = f * (1.0 - g)
    e3 = np.zeros_like(f)
    return f, e1, e2, e3, tj_xy, d1, d2, d_gb


def _gamma_gb_for_equilibrium(psi_deg, gamma_s):
    return 2.0 * gamma_s * math.cos(math.radians(psi_deg) / 2.0)


ANGLES = [80.0, 100.0, 120.0, 140.0, 160.0]


# ---------------------------------------------------------------------------
# A + B: branch directions and independent psi measurement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("psi_deg", ANGLES)
def test_branch_directions_and_psi_recovered(psi_deg):
    p = _base_params()
    s = Sink(threshold=1.0)
    p.gamma_gb_ref = _gamma_gb_for_equilibrium(120.0, p.gamma_s)  # arbitrary fixed GB
    f, e1, e2, e3, tj_xy, d1, d2, d_gb = build_synthetic_wedge(p, psi_deg)

    res = compute_tj_force(f, e1, e2, tj_xy, s, p)
    assert res.resolved, res.reason

    # branch directions recovered to within ~2 degrees
    def angle_err(v, d):
        return math.degrees(math.acos(np.clip(np.dot(v, d), -1, 1)))

    # v_s1/v_s2 order is not guaranteed; match by nearest.
    pairs = [(res.v_s1, d1), (res.v_s2, d2)] if angle_err(res.v_s1, d1) < angle_err(res.v_s1, d2) \
        else [(res.v_s1, d2), (res.v_s2, d1)]
    for v, d in pairs:
        assert angle_err(v, d) < 2.5, f"branch direction off by {angle_err(v,d):.2f} deg"
    assert angle_err(res.v_gb, d_gb) < 2.5

    # independent psi measurement matches the prescribed synthetic angle
    assert abs(res.psi_deg - psi_deg) < 2.5, f"measured psi={res.psi_deg}, expected {psi_deg}"


# ---------------------------------------------------------------------------
# C: equilibrium vector balance -> F_TJ near zero
# ---------------------------------------------------------------------------

def test_equilibrium_configuration_gives_near_zero_resultant():
    p = _base_params()
    s = Sink(threshold=1.0)
    psi_eq = 120.0
    p.gamma_gb_ref = _gamma_gb_for_equilibrium(psi_eq, p.gamma_s)
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, psi_eq)

    res = compute_tj_force(f, e1, e2, tj_xy, s, p)
    assert res.resolved, res.reason
    # Normalize by gamma_s (O(1) capillary vector scale) for a dimensionless tolerance.
    assert res.F_TJ_mag / p.gamma_s < 0.06, f"|F_TJ|/gamma_s = {res.F_TJ_mag/p.gamma_s:.4f}"


# ---------------------------------------------------------------------------
# D: perturbation away from equilibrium -> nonzero resultant, expected sign
# ---------------------------------------------------------------------------

def test_perturbation_sign_is_monotonic_through_equilibrium():
    p = _base_params()
    s = Sink(threshold=1.0)
    psi_eq = 120.0
    p.gamma_gb_ref = _gamma_gb_for_equilibrium(psi_eq, p.gamma_s)

    parallel = {}
    for psi_deg in ANGLES:
        f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, psi_deg)
        res = compute_tj_force(f, e1, e2, tj_xy, s, p)
        assert res.resolved, res.reason
        parallel[psi_deg] = res.F_gb_parallel

    # Narrower-than-equilibrium angles pull one way, wider the other, and the
    # trend is monotonic across the whole sweep (closed-form: F_gb_parallel =
    # gamma_gb - 2*gamma_s*cos(psi/2), strictly increasing in psi).
    ordered = [parallel[a] for a in ANGLES]
    assert all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1)), ordered
    assert parallel[80.0] < 0 < parallel[160.0]
    assert abs(parallel[120.0]) < 0.06 * p.gamma_s


# ---------------------------------------------------------------------------
# E: anisotropic Cahn-Hoffman term is actually exercised
# ---------------------------------------------------------------------------

def test_anisotropic_cahn_hoffman_branch_is_exercised():
    p_iso = _base_params(aniso=False)
    p_aniso = _base_params(aniso=True, aniso_delta=0.15)
    s = Sink(threshold=1.0)
    psi_deg = 100.0  # asymmetric enough to avoid a special-symmetry orientation

    f_iso, e1_iso, e2_iso, *_ = build_synthetic_wedge(p_iso, psi_deg)
    f_an, e1_an, e2_an, *_, tj_xy, d1, d2, _ = build_synthetic_wedge(p_aniso, psi_deg)

    res_iso = compute_tj_force(f_iso, e1_iso, e2_iso, tj_xy, s, p_iso)
    res_an = compute_tj_force(f_an, e1_an, e2_an, tj_xy, s, p_aniso)
    assert res_iso.resolved and res_an.resolved

    # The anisotropic xi should differ meaningfully from the isotropic
    # gamma_s * v prediction for at least one branch.
    dev1 = np.linalg.norm(res_an.xi_s1 - p_aniso.gamma_s * res_an.v_s1)
    dev2 = np.linalg.norm(res_an.xi_s2 - p_aniso.gamma_s * res_an.v_s2)
    assert max(dev1, dev2) > 0.02 * p_aniso.gamma_s, (dev1, dev2)

    # And the isotropic run should show ~no deviation from that prediction.
    dev1_iso = np.linalg.norm(res_iso.xi_s1 - p_iso.gamma_s * res_iso.v_s1)
    dev2_iso = np.linalg.norm(res_iso.xi_s2 - p_iso.gamma_s * res_iso.v_s2)
    assert max(dev1_iso, dev2_iso) < 1e-9


def test_cahn_hoffman_vector_isotropic_matches_gamma_s_times_tangent():
    p = _base_params(aniso=False)
    f, e1, e2, e3, tj_xy, d1, d2, _ = build_synthetic_wedge(p, 120.0)
    xi, n = cahn_hoffman_vector(d1, f, tj_xy, p)
    assert np.allclose(xi, p.gamma_s * d1, atol=1e-9)


# ---------------------------------------------------------------------------
# Signed curvature: convex vs concave arcs
# ---------------------------------------------------------------------------

def _circular_bump_field(p, R, convex=True, center=None):
    if center is None:
        center = np.array([0.5 * p.Nx * p.dx, 0.5 * p.Ny * p.dx])
    W = p.interface_width
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X - center[0], Y - center[1])
    if convex:
        f = 0.5 * (1.0 - np.tanh((r - R) / W))  # solid disk of radius R
    else:
        f = 0.5 * (1.0 + np.tanh((r - R) / W))  # solid everywhere except a disk hole of radius R
    return f, center


def test_signed_curvature_convex_positive_concave_negative():
    p = _base_params(nx=96, ny=96, dx_nm=2.0)
    R = 20 * p.dx
    f_convex, center = _circular_bump_field(p, R, convex=True)
    f_concave, _ = _circular_bump_field(p, R, convex=False, center=center)

    probe = np.array([center[0] + R, center[1]])  # point on the circle, +x side
    k_convex = signed_curvature_at(f_convex, probe, p)
    k_concave = signed_curvature_at(f_concave, probe, p)

    assert k_convex > 0, k_convex
    assert k_concave < 0, k_concave
    assert math.isclose(abs(k_convex), 1.0 / R, rel_tol=0.15)
    assert math.isclose(abs(k_concave), 1.0 / R, rel_tol=0.15)
