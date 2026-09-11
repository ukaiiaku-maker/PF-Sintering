import numpy as np
from test_three_particle_implicit import fixture
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion

def test_full_jacobian_matches_native_directional_derivative():
 f,op,_=fixture();it=FullJacobianSurfaceDiffusion(op);v=np.random.default_rng(916).normal(size=f.shape);v/=np.max(abs(v))
 Kr,Kz=it.flux_jacobian(f,op.potential(f).copy());exact=(it.Dr@(Kr@v.ravel())+it.Dz@(Kz@v.ravel())).reshape(f.shape);errors=[]
 for eps in [1e-4,5e-5,2.5e-5]:
  fd=(it.rhs(f+eps*v)-it.rhs(f-eps*v))/(2*eps);errors.append(np.linalg.norm(fd-exact)/np.linalg.norm(exact))
 assert errors[-1]<1e-5 and errors[0]/errors[-1]>10

def test_full_jacobian_step_consistency_mass_and_reflection():
 f,op,_=fixture(mirror=True);it=FullJacobianSurfaceDiffusion(op);F=it.rhs(f);h=1e-8;a=it.step(f,h);b=it.step(f,h/2)
 ratio=np.max(abs(a-f-h*F))/np.max(abs(b-f-h/2*F));assert 3.8<ratio<4.2
 assert abs(np.sum((a-f)*op.g['r_c']))<1e-22
 np.testing.assert_allclose(a,a[::-1],rtol=0,atol=1e-13)
 np.testing.assert_array_equal(it.step(f,0),f)
