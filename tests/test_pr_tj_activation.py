import numpy as np

from pf_sintering.pr_tj_activation import (
    add_equilibrium_referenced_tj_stresses,
    tj_functionals_from_mu,
    tj_geometric_indicators,
)


def synthetic_tj(nz=41, nr=51, dx=0.2):
    z = (np.arange(nz) - nz // 2) * dx
    r = (np.arange(nr) + 0.5) * dx
    Z, R = np.meshgrid(z, r, indexing="ij")
    f = 0.5 * (1.0 - np.tanh((R - 6.0) / 0.7))
    owner = 0.5 * (1.0 + np.tanh(Z / 0.6))
    e1 = f * owner
    e2 = f - e1
    return f, e1, e2, r, dx


def test_intrinsic_tj_support_is_nonnegative_and_has_no_width_parameter():
    f, e1, e2, _, _ = synthetic_tj()
    gb, surface, tj = tj_geometric_indicators(f, e1, e2)
    assert np.all(gb >= 0.0)
    assert np.all(surface >= 0.0)
    np.testing.assert_allclose(tj, gb * surface)
    assert np.unravel_index(np.argmax(tj), tj.shape)[0] == f.shape[0] // 2


def test_constant_mu_has_correct_pa_and_n_per_m_scaling():
    f, e1, e2, r, dx = synthetic_tj()
    value = 7.5
    row = tj_functionals_from_mu(
        np.full_like(f, value), f, e1, e2, dx, dx, r)
    np.testing.assert_allclose(row["mu_PF_TJ_Pa"], value, rtol=1e-14)
    np.testing.assert_allclose(
        row["lambda_TJ_raw_N_per_m"],
        value * row["line_support_length_m"], rtol=1e-14)
    assert row["circumference_multiplier_applied"] is False


def test_equilibrium_reference_is_zero_and_line_mapping_uses_physical_b():
    f, e1, e2, r, dx = synthetic_tj()
    equilibrium = tj_functionals_from_mu(
        np.full_like(f, 3.0), f, e1, e2, dx, dx, r)
    zero = add_equilibrium_referenced_tj_stresses(equilibrium, equilibrium, 0.25)
    assert zero["sigma_act_TJ_mu_Pa"] == 0.0
    assert zero["lambda_TJ_excess_N_per_m"] == 0.0
    assert zero["sigma_act_TJ_lambda_Pa"] == 0.0

    loaded = tj_functionals_from_mu(
        np.full_like(f, 5.0), f, e1, e2, dx, dx, r)
    excess = add_equilibrium_referenced_tj_stresses(loaded, equilibrium, 0.25)
    np.testing.assert_allclose(excess["sigma_act_TJ_mu_Pa"], 2.0)
    np.testing.assert_allclose(
        excess["sigma_act_TJ_lambda_Pa"],
        excess["lambda_TJ_excess_N_per_m"] / 0.25)
    assert excess["physical_TJ_width_definition"].startswith("w_TJ=b")
