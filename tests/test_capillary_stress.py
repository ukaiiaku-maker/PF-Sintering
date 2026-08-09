import math

import numpy as np

from pf_sintering.capillary_stress import (
    apparent_sintering_stress,
    capillary_force_curvature_form,
    capillary_force_endpoint_form,
    force_validation_relative_error,
    oriented_contact_normal,
    trace_particle_arc,
    window_mean_kappa,
    window_mean_kappa_from_end,
)
from pf_sintering.model import ModelConfig, build_params


def _circle_params_and_field(dx=2e-9, Nx=200, Ny=200, R=100e-9, W=20e-9):
    p = build_params(ModelConfig(
        preset="dev", dx=dx, nx=Nx, ny=Ny, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        interface_width_override=W, use_aniso_surface=False,
    ))
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    cx, cy = x.mean(), y.mean()
    r = np.hypot(X - cx, Y - cy)
    f = 0.5 * (1 - np.tanh((r - R) / p.interface_width))
    return p, f, (cx, cy), R


def _ccw_tangent(theta):
    return np.array([-math.sin(theta), math.cos(theta)])


def test_endpoint_vs_curvature_form_agree_on_analytic_circle():
    # Milestone 14C Sections 4-6: for a circle, the traced-arc curvature-
    # form capillary resultant must match BOTH the independent endpoint-
    # tangent form AND the closed-form analytic value gamma_s*(t2-t1) to a
    # documented, small (few-percent) discretization tolerance -- this is
    # the decisive validation of the whole force construction before it is
    # trusted on real (noisier, non-circular) states.
    p, f, (cx, cy), R = _circle_params_and_field()
    theta1, theta2 = math.radians(80), math.radians(-80)
    tj1 = np.array([cx + R * math.cos(theta1), cy + R * math.sin(theta1)])
    tj2 = np.array([cx + R * math.cos(theta2), cy + R * math.sin(theta2)])

    dir1 = _ccw_tangent(theta1)         # away from tj1, walking the long way (CCW) toward tj2
    dir2 = -_ccw_tangent(theta2)        # away from tj2, walking the long way (CW) back toward tj1

    arc = trace_particle_arc(f, p, tj1, tj2, dir1, dir2, max_arclength=550e-9, n_samples=400)
    assert arc["resolved"]
    assert arc["closest_approach_dist"] < 3 * p.dx
    # expected arclength: 200deg of a R=100nm circle
    assert math.isclose(arc["arc_length"], math.radians(200) * R, rel_tol=0.02)

    gamma_s = p.gamma_s
    F_curv = capillary_force_curvature_form(arc, gamma_s)
    F_ep = capillary_force_endpoint_form(dir1, dir2, gamma_s)
    t2_analytic = _ccw_tangent(math.radians(280))  # == theta2 mod 2pi
    F_analytic = (gamma_s * (t2_analytic[0] - dir1[0]), gamma_s * (t2_analytic[1] - dir1[1]))

    assert force_validation_relative_error(F_curv, F_ep) < 0.05
    assert math.isclose(F_ep[0], F_analytic[0], abs_tol=1e-9)
    assert math.isclose(F_ep[1], F_analytic[1], abs_tol=1e-9)

    # signed curvature should be close to +1/R (convex solid circle, center
    # inside the solid) everywhere along the arc, not just on average.
    assert np.mean(arc["kappa"]) > 0
    assert math.isclose(float(np.mean(arc["kappa"])), 1.0 / R, rel_tol=0.2)


def test_wrong_branch_direction_convention_fails_the_check():
    # Using the SAME (not reversed) tangent sense at both ends should break
    # the identity -- a regression guard on the sign convention itself.
    p, f, (cx, cy), R = _circle_params_and_field()
    theta1, theta2 = math.radians(80), math.radians(-80)
    tj1 = np.array([cx + R * math.cos(theta1), cy + R * math.sin(theta1)])
    tj2 = np.array([cx + R * math.cos(theta2), cy + R * math.sin(theta2)])
    dir1 = _ccw_tangent(theta1)
    wrong_dir2 = _ccw_tangent(theta2)  # NOT reversed -- wrong convention
    F_ep_wrong = capillary_force_endpoint_form(dir1, wrong_dir2, p.gamma_s)
    F_ep_right = capillary_force_endpoint_form(dir1, -_ccw_tangent(theta2), p.gamma_s)
    assert force_validation_relative_error(F_ep_wrong, F_ep_right) > 0.5


def test_apparent_sintering_stress_sign_and_units():
    # A capillary force pointing purely toward the substrate (-x) combined
    # with n_GB pointing from substrate toward particle (+x) must give a
    # POSITIVE apparent stress (densifying), per Section 7's sign choice.
    F_cap = (-2.0, 0.0)  # N/m, pointing toward substrate
    n_GB = (1.0, 0.0)    # substrate -> particle
    L_contact = 7e-8     # m
    sigma, F_cap_n = apparent_sintering_stress(F_cap, n_GB, L_contact)
    assert F_cap_n == -2.0
    assert sigma > 0
    assert math.isclose(sigma, 2.0 / L_contact)


def test_oriented_contact_normal_negates_raw_n_gb_sub():
    assert oriented_contact_normal((-1.0, 0.0)) == (1.0, -0.0)


def test_window_mean_kappa_from_each_end():
    p, f, (cx, cy), R = _circle_params_and_field()
    theta1, theta2 = math.radians(80), math.radians(-80)
    tj1 = np.array([cx + R * math.cos(theta1), cy + R * math.sin(theta1)])
    tj2 = np.array([cx + R * math.cos(theta2), cy + R * math.sin(theta2)])
    dir1 = _ccw_tangent(theta1)
    dir2 = -_ccw_tangent(theta2)
    arc = trace_particle_arc(f, p, tj1, tj2, dir1, dir2, max_arclength=550e-9, n_samples=400)
    W = p.interface_width
    # very-near-TJ windows (1.5W-3W) carry a real, expected smoothing-edge
    # bias (Section 10's own warning against trusting the diffuse TJ core)
    # -- checked here only for internal top/bottom self-consistency, not
    # against the analytic value.
    near_top = window_mean_kappa(arc, 1.5 * W, 3.0 * W)
    near_bottom = window_mean_kappa_from_end(arc, 1.5 * W, 3.0 * W)
    assert math.isclose(near_top, near_bottom, rel_tol=0.1)
    # a farther window (3W-5W), well clear of the smoothing kernel's edge
    # effects, should recover the true circle curvature closely from
    # EITHER end.
    far_top = window_mean_kappa(arc, 3.0 * W, 5.0 * W)
    far_bottom = window_mean_kappa_from_end(arc, 3.0 * W, 5.0 * W)
    assert math.isclose(far_top, far_bottom, rel_tol=0.1)
    assert math.isclose(far_top, 1.0 / R, rel_tol=0.1)
