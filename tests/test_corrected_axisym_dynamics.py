from types import SimpleNamespace

import numpy as np

from pf_sintering.corrected_axisym_dynamics import (
    CorrectedNumbaScratch,
    axisym_corrected_normalized_ownership_step_fast,
)
from pf_sintering.corrected_interfacial_energy import (
    normalized_ownership_variational_derivatives_axisym,
)


def test_corrected_dynamic_step_closes_partition_and_uses_corrected_mu():
    rng = np.random.default_rng(97)
    nz, nr = 10, 9
    dr = dz = 1.0
    r_c = np.arange(nr, dtype=float) + 0.5
    r_f = np.arange(nr + 1, dtype=float)
    f = 0.2 + 0.6 * rng.random((nz, nr))
    phi = 0.1 + 0.8 * rng.random((nz, nr))
    eta1, eta2 = f * phi, f * (1.0 - phi)
    p = SimpleNamespace(W_f=2.0, k_f=0.7, k_eta=0.4)
    Wc = 1.1
    expected_mu, _ = normalized_ownership_variational_derivatives_axisym(
        f, eta1, eta2, p, Wc, dr, dz, r_c, r_f)
    scratch = CorrectedNumbaScratch(nz, nr)
    result = axisym_corrected_normalized_ownership_step_fast(
        f, eta1, eta2, p, Wc, dr, dz, r_c, r_f,
        1.0e-8, 1.0e-4, 1.0e-4, 2.0, scratch)
    assert np.allclose(scratch.mu, expected_mu, rtol=2.0e-13, atol=2.0e-13)
    assert np.max(np.abs(result[1] + result[2] - result[0])) < 1.0e-15
    assert np.min(scratch.phi_new) >= 0.0
    assert np.max(scratch.phi_new) <= 1.0


def test_dense_phi_velocity_recovers_calibrated_half_metric():
    nz, nr = 8, 7
    dr = dz = 1.0
    r_c = np.arange(nr, dtype=float) + 0.5
    r_f = np.arange(nr + 1, dtype=float)
    f = np.ones((nz, nr))
    z = np.arange(nz, dtype=float)[:, None]
    phi = np.broadcast_to(0.2 + 0.6 * z / (nz - 1), (nz, nr)).copy()
    eta1, eta2 = phi.copy(), 1.0 - phi
    p = SimpleNamespace(W_f=2.0, k_f=0.7, k_eta=0.4)
    Wc, M_eta, dt = 1.1, 2.0e-4, 1.0e-7
    _, gphi = normalized_ownership_variational_derivatives_axisym(
        f, eta1, eta2, p, Wc, dr, dz, r_c, r_f)
    scratch = CorrectedNumbaScratch(nz, nr)
    axisym_corrected_normalized_ownership_step_fast(
        f, eta1, eta2, p, Wc, dr, dz, r_c, r_f,
        dt, 0.0, M_eta, 2.0, scratch)
    expected = np.clip(phi - dt * 0.5 * M_eta * gphi, 0.0, 1.0)
    assert np.allclose(scratch.phi_new, expected, rtol=1.0e-13, atol=1.0e-13)
