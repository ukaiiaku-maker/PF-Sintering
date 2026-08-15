"""Milestone 16M Section 2: tests for the explicit particle/substrate role
mapping, guarding against a repeat of M16L's grain-identity-reversal bug."""
from __future__ import annotations

import numpy as np
import pytest

from pf_sintering.grain_roles import assert_particle_above_substrate, roles_from_m16j_geometry
from pf_sintering.m16j_geometry import build_candidate_geometry


def test_roles_from_m16j_geometry_swaps_e1_e2():
    e1 = np.array([[1.0, 1.0]])
    e2 = np.array([[0.0, 0.0]])
    roles = roles_from_m16j_geometry(e1, e2)
    np.testing.assert_array_equal(roles.particle, e2)
    np.testing.assert_array_equal(roles.substrate, e1)


def test_m16j_geometry_particle_above_substrate():
    geo = build_candidate_geometry(R_p_nm=1000.0, R_s_nm=None, X0_over_2Rp=0.10, psi_deg=160.0,
                                    W_nm=10.0, dr_nm=1.25, dz_nm=1.25, aspect_ratio=1.0)
    z, r_c = geo["z"] * 1e-9, geo["r_c"] * 1e-9
    e1, e2 = geo["e1"], geo["e2"]
    roles = roles_from_m16j_geometry(e1, e2)
    assert_particle_above_substrate(roles.particle, roles.substrate, z, r_c)


def test_assert_particle_above_substrate_catches_reversal():
    Nz, Nr = 40, 10
    z = (np.arange(Nz) + 0.5) * 1e-9
    r_c = (np.arange(Nr) + 0.5) * 1e-9
    z_mid = z[Nz // 2]
    particle = (z[:, None] > z_mid).astype(float) * np.ones((1, Nr))
    substrate = 1.0 - particle
    # correct orientation: should not raise
    assert_particle_above_substrate(particle, substrate, z, r_c)
    # swapped: must raise
    with pytest.raises(AssertionError):
        assert_particle_above_substrate(substrate, particle, z, r_c)
