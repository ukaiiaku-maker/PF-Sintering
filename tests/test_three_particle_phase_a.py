import numpy as np
from pf_sintering.three_particle_geometry import ThreeParticleConfig,build_three_particle,grain_volumes
from pf_sintering.three_particle_phase_a import PhaseAOperator,initialize_step
from pf_sintering.axisym import axisym_volume
from pf_sintering.corrected_interfacial_energy import normalized_ownership_variational_derivatives_axisym

def case():return build_three_particle(ThreeParticleConfig(spacing=2.5e-9))

def test_variational_derivative():
    g=case();op=PhaseAOperator(g);f=g['f'];rng=np.random.default_rng(37)
    d=rng.normal(size=f.shape)*f*(1-f);eps=1e-5
    finite=(op.energy(f+eps*d)-op.energy(f-eps*d))/(2*eps)
    exact=axisym_volume(op.potential(f)*d,g['r_c'],g['dr'],g['dz'])
    np.testing.assert_allclose(finite,exact,rtol=2e-6,atol=1e-24)

def test_binary_reduction():
    g=case();p=g['ownership'].copy();p[0]+=p[1];p[1]=0;g['ownership']=p
    op=PhaseAOperator(g);f=np.maximum(g['f'],1e-10)
    # Binary production divides eta/f only above 1e-14; compare away from that vacuum convention.
    mu,_=normalized_ownership_variational_derivatives_axisym(f,f*p[0],f*p[2],op,op.Wc,g['dr'],g['dz'],g['r_c'],g['r_f'])
    np.testing.assert_allclose(op.potential(f),mu,rtol=2e-13,atol=1e-6)

def test_initialization_preserves_volumes_centers_and_energy():
    g=case();op=PhaseAOperator(g);f=g['f'];v=grain_volumes(f,g);moment=grain_volumes(f*g['z'][:,None],g)
    energy=op.energy(f)
    for _ in range(10):f,_=initialize_step(f,op)
    np.testing.assert_allclose(grain_volumes(f,g),v,rtol=2e-14,atol=1e-35)
    np.testing.assert_allclose(grain_volumes(f*g['z'][:,None],g),moment,rtol=2e-14,atol=1e-41)
    assert op.energy(f)<energy
    np.testing.assert_allclose(f,f[::-1],atol=1e-13)

def test_native_flux_global_conservation_and_transport_off():
    g=case();op=PhaseAOperator(g);f=g['f'];v=grain_volumes(f,g)
    np.testing.assert_array_equal(op.step(f,1,surface_enabled=False),f)
    evolved=op.step(f,1e-5)
    assert abs(grain_volumes(evolved,g).sum()/v.sum()-1)<2e-14
    assert np.max(np.abs(evolved-f))>0
    # Connectivity is an operator property, independent of seed transients.
    op.mu[:]=g['z'][:,None]*1e14
    from pf_sintering.axisym_numba_kernel import flux_kernel
    flux_kernel(f,op.mu,g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
    for b in g['gb']:
        j=np.argmin(abs(g['z']+g['dz']/2-b))
        assert abs(np.sum(op.Jz[j]*g['r_c']))>1e-25
