import math

import numpy as np

from pf_sintering.pr_activation_coordinates import (
    local_capillary_residual,
    localized_embryo_mu_residual,
    winterbottom_spherical_cap_reference,
)


def test_candidate_a_is_exactly_zero_on_winterbottom_spherical_cap():
    psi = math.radians(160.0)
    R = 100e-9
    ref = winterbottom_spherical_cap_reference(R, psi, gamma_s=1.0)
    assert abs(ref["young_relation_residual"]) < 1e-15
    assert abs(ref["X_A_local_capillary_residual_Pa"]) < 1e-8
    assert ref["X_B_equilibrium_mu_residual_Pa"] == 0.0
    assert ref["F_s_constrained_equilibrium"] == 0.0
    assert ref["sigma_H_legacy_absolute_Pa"] != 0.0
    assert ref["Sigma_contact_3D_absolute_Pa"] != 0.0


def test_candidate_a_uses_no_separate_line_force():
    R = 75e-9
    psi = math.radians(140.0)
    r_neck = R * math.sin(psi / 2.0)
    value = local_capillary_residual(1.0 / R, 1.0 / R, r_neck, psi)
    assert abs(value) < 1e-8


def test_localized_mu_coordinate_is_a_state_function_and_normalized():
    class Params:
        W_f = 2.0
        k_f = 0.5

    z = np.arange(8.0) + 0.5
    r_c = np.arange(6.0) + 0.5
    r_f = np.arange(7.0)
    f = np.full((8, 6), 0.5)
    particle = np.full_like(f, 0.25)
    neighbor = f - particle
    kwargs = dict(
        f=f, particle=particle, neighbor=neighbor, pf_params=Params(),
        Wc=0.0, dr=1.0, dz=1.0, r_c=r_c, r_f=r_f, z=z,
        z_gb=4.0, r_neck=3.0, W=1.0,
        source_width_factor=2.0, sink_width_factor=1.0)
    a = localized_embryo_mu_residual(**kwargs)
    b = localized_embryo_mu_residual(**kwargs)
    assert a == b
    assert a["X_B_embryo_mu_residual_Pa"] == 0.0
    np.testing.assert_allclose(a["source_support_weighted_norm"], 1.0)
    np.testing.assert_allclose(a["sink_support_weighted_norm"], 1.0)
