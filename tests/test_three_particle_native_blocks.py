import numpy as np
from test_three_particle_implicit import fixture
from pf_sintering.three_particle_event import update_ownership,ownership_pair_step
from pf_sintering.three_particle_native_blocks import native_block
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update


def test_buffered_block_matches_native_stencils_with_active_ownership_flow():
    f,op,_=fixture();g=op.g;rng=np.random.default_rng(54)
    phi=rng.uniform(.1,1.,(3,*f.shape));phi/=phi.sum(axis=0)
    update_ownership(op,phi);density=op.gb_density.copy();h=1e-8;mobility=1e-4
    expected_f=f.copy();expected_phi=phi.copy()
    for _ in range(3):
        expected_f=bounded_mobility_update(expected_f,op.potential(expected_f),op,h)
        expected_phi=ownership_pair_step(expected_phi,expected_f,op,(0,1),h,mobility)
        update_ownership(op,expected_phi)
    actual_f,actual_phi,actual_density=native_block(f.copy(),phi.copy(),density,0,1,h,mobility,op.Wc,op.k_eta,op.W_f,op.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],op.W,op.physics.M_s,steps=3)
    np.testing.assert_allclose(actual_f,expected_f,atol=2e-14,rtol=0)
    np.testing.assert_array_equal(actual_phi,expected_phi)
    np.testing.assert_array_equal(actual_density,op.gb_density)
    np.testing.assert_array_equal(actual_phi[2],phi[2])
    assert np.max(abs(actual_phi[0]-phi[0]))>1e-5
