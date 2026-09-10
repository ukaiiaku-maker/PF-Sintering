import numpy as np
from test_three_particle_implicit import fixture
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion,bounded_mobility_update,harmonic


def test_harmonic_flux_is_conservative_and_matches_implicit_matrix():
    f,op,_=fixture();it=HarmonicSurfaceDiffusion(op);mu=np.random.default_rng(19).normal(size=f.shape)*1e8
    changed=bounded_mobility_update(f,mu,op,1.)-f
    expected=(it.mobility(f)@mu.ravel()).reshape(f.shape)
    np.testing.assert_allclose(changed,expected,rtol=2e-12,atol=1e-9)
    assert abs(np.sum(changed*op.g['r_c']))<1e-12


def test_pure_cells_have_zero_semidiscrete_rate():
    f,op,_=fixture();f[6,5]=0.;f[10,5]=1.;it=HarmonicSurfaceDiffusion(op)
    mu=np.random.default_rng(9).normal(size=f.shape)*1e8
    changed=bounded_mobility_update(f,mu,op,1.)
    assert changed[6,5]==0.;assert changed[10,5]==1.
    rate=(it.mobility(f)@mu.ravel()).reshape(f.shape)
    assert rate[6,5]==0.;assert rate[10,5]==0.


def test_consistent_second_order_face_quadrature():
    exact=(.4**2)*(.6**2);errors=[]
    for dx in [.02,.01,.005]:
        f0=.4-.3*dx/2;f1=.4+.3*dx/2
        q0=np.array([f0*f0*(1-f0)**2]);q1=np.array([f1*f1*(1-f1)**2])
        errors.append(abs(float(harmonic(q0,q1)[0])-exact))
    assert 3.99<errors[0]/errors[1]<4.01
    assert 3.99<errors[1]/errors[2]<4.01
