import numpy as np
import pytest
from pf_sintering.three_particle_symmetry import reconstruct_full_state_from_paired_halves


def test_reconstructs_exact_full_domain_and_swaps_outer_labels():
    rng=np.random.default_rng(911);f=rng.uniform(.1,.9,(6,4));raw=rng.uniform(size=(3,*f.shape));phi=raw/raw.sum(axis=0)
    result,ownership,state=reconstruct_full_state_from_paired_halves(f,phi)
    np.testing.assert_array_equal(result,result[::-1])
    np.testing.assert_array_equal(ownership[0],ownership[2,::-1])
    np.testing.assert_array_equal(ownership[1],ownership[1,::-1])
    np.testing.assert_allclose(sum(state[1:]),result,rtol=0,atol=2e-16)
    np.testing.assert_allclose(result.sum(),f.sum(),rtol=0,atol=2e-15)


def test_reconstruction_validates_shape():
    with pytest.raises(ValueError):reconstruct_full_state_from_paired_halves(np.zeros((5,2)),np.zeros((3,5,2)))
