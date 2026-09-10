import numpy as np
from pf_sintering.three_particle_null import virtual_work_multipliers,native_rhs


def geometry():
    z=np.linspace(-1,1,15)[:,None];r=np.linspace(0,1,9)[None,:]
    a=1/(1+np.exp(-12*(z+.3)));b=1/(1+np.exp(-12*(z-.3)))
    phi=np.broadcast_to(np.stack([1-a,a-b,b]),(3,15,9)).copy()
    cv=np.broadcast_to(.1+r,(15,9)).copy();return phi,cv


def test_equilibrium_multipliers_are_independent_of_variation_basis():
    phi,cv=geometry();lam=np.array([17e6,19e6,17e6]);mu=np.einsum('k,kji->ji',lam,phi)
    rng=np.random.default_rng(71)
    for w in [np.ones_like(mu),.1+rng.random(mu.shape),(.1+rng.random(mu.shape))**2]:
        got,residual=virtual_work_multipliers(mu,phi,w,cv)
        np.testing.assert_allclose(got,lam,rtol=1e-13)
        assert residual<1e-6


def test_nonequilibrium_virtual_work_is_not_a_unique_thermodynamic_mu():
    phi,cv=geometry();mu=1e6*(1+.5*phi[1])*np.broadcast_to(np.arange(9)**2,(15,9))
    a,res=virtual_work_multipliers(mu,phi,np.ones_like(mu),cv)
    b,_=virtual_work_multipliers(mu,phi,1+np.broadcast_to(np.arange(9),(15,9)),cv)
    assert abs((a[1]-a[0])-(b[1]-b[0]))>1e6 and res>1e6


def test_native_rhs_is_the_unchanged_bounded_euler_generator():
    from test_three_particle_implicit import fixture
    f,op,_=fixture();h=1e-7;F=native_rhs(f,op)
    np.testing.assert_allclose(f+h*F,op.step(f,h),rtol=0,atol=2e-16)
    assert abs(np.sum(F*op.g['r_c']))<1e-12
