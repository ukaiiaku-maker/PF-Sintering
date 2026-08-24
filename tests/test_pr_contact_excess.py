from __future__ import annotations

import math

import numpy as np

from pf_sintering.pr_contact_excess import (
    build_volume_matched_winterbottom,
    project_original_pr_modes,
    sharp_radius_for_particle_volume,
    spherical_major_cap_volume,
)


def test_spherical_cap_radius_inverts_volume():
    radius = 117e-9
    psi = math.radians(160.0)
    volume = spherical_major_cap_volume(radius, psi)
    assert math.isclose(sharp_radius_for_particle_volume(volume, psi), radius, rel_tol=1e-14)


def test_diffuse_winterbottom_matches_requested_particle_volume_and_simplex():
    dr = dz = 1.25e-9
    z = (np.arange(420) + 0.5) * dz
    r_c = (np.arange(190) + 0.5) * dr
    target = spherical_major_cap_volume(110e-9, math.radians(160.0))
    result = build_volume_matched_winterbottom(
        target, math.radians(160.0), 10e-9, z, r_c, dr, dz, 210e-9)
    assert abs(result["particle_volume_relative_error"]) < 1e-10
    assert np.allclose(result["e1"] + result["e2"], result["f"], atol=1e-14)
    assert result["radius_diffuse_m"] < result["radius_sharp_m"]


def test_original_pr_projection_recovers_declared_modes():
    lam = 360e-9
    R_cyl = 100e-9
    z = (np.arange(288) + 0.5) * lam / 288
    R = R_cyl * (
        0.82 + 0.17 * np.cos(2.0 * math.pi * z / lam)
        - 0.06 * np.cos(math.pi * z / lam))
    result = project_original_pr_modes(R, z, lam, R_cyl)
    assert math.isclose(result["mean_radius_over_Rcyl"], 0.82, abs_tol=1e-13)
    assert math.isclose(result["mode_2pi_amplitude"], 0.17, abs_tol=1e-13)
    assert math.isclose(result["mode_pi_amplitude"], -0.06, abs_tol=1e-13)
    assert math.isclose(result["A_PR"], math.hypot(0.17, 0.06), abs_tol=1e-13)
