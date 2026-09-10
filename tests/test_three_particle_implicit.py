from types import SimpleNamespace
import numpy as np
import pytest
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_implicit import ImplicitSurfaceDiffusion
from pf_sintering.axisym_numba_kernel import flux_kernel,div_and_update_kernel


def fixture(mirror=False):
    nz,nr=16,11;dr=1e-9;dz=.8e-9
    z=(np.arange(nz)-(nz-1)/2)*dz;r=(np.arange(nr)+.5)*dr
    f=.4+.12*np.cos(z[:,None]/4e-9)+.07*np.sin(r[None,:]/3e-9)
    if not mirror:f+=.02*np.sin(z[:,None]/3e-9)
    p=np.zeros((3,nz,nr));p[0]=.25;p[1]=.5;p[2]=.25
    g=dict(f=f,ownership=p,z=z,r_c=r,r_f=np.arange(nr+1)*dr,dr=dr,dz=dz,config=SimpleNamespace(width=4e-9))
    op=PhaseAOperator(g);return f,op,ImplicitSurfaceDiffusion(op)


def test_sparse_flux_matches_native_for_arbitrary_potential_and_boundaries():
    f,op,it=fixture();g=op.g
    mu=np.random.default_rng(123).normal(size=f.shape)*1e8
    flux_kernel(f,mu,g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
    div_and_update_kernel(np.zeros_like(f),op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],1.,op.out)
    np.testing.assert_allclose((it.mobility(f)@mu.ravel()).reshape(f.shape),op.out,rtol=2e-13,atol=1e-9)
    assert abs(np.sum(op.out*g['r_c'])) < 1e-12


def test_implicit_is_consistent_and_conserves_weighted_mass():
    f,op,it=fixture();h=1e-8
    e=op.step(f,h);a=it.step(f,h);b=it.step(f,h/2)
    # First-order consistency: Euler/implicit difference scales quadratically.
    error=np.max(abs(a-e));half_error=np.max(abs(b-op.step(f,h/2)))
    assert 3.8<error/half_error<4.2
    for x in [a,b]:
        assert abs(np.sum((x-f)*op.g['r_c']))<1e-22


def test_mirror_and_zero_time_and_invalid_time():
    f,op,it=fixture(mirror=True)
    a=it.step(f,1e-6)
    np.testing.assert_allclose(a,a[::-1],atol=1e-13,rtol=0)
    np.testing.assert_array_equal(it.step(f,0),f)
    for h in [-1,np.nan]:
        with pytest.raises(ValueError):it.step(f,h)


def test_actual_field_checkpoint_continuation_is_reproducible(tmp_path):
    f,op,it=fixture();h=1e-7
    first=it.step(f,h)
    path=tmp_path/'current_state.npz'
    np.savez_compressed(path,f=first,ownership=op.g['ownership'],next_h=h)
    direct=it.step(first,h)
    with np.load(path,allow_pickle=False) as saved:
        g=dict(op.g,f=saved['f'].copy(),ownership=saved['ownership'].copy())
        restored=ImplicitSurfaceDiffusion(PhaseAOperator(g)).step(g['f'],float(saved['next_h']))
    np.testing.assert_array_equal(restored,direct)


def test_null_plateau_requires_two_complete_quiet_doubling_intervals():
    from scripts.three_particle_implicit_analysis import null_plateau
    times=np.array([.1,.2,.4,.8,1.6])
    d={'t_model':times}
    for side in ['LEFT','RIGHT']:
        for key in ['CC_stress_Pa','PF_geometric_stress_Pa','kappa_per_m','psi_deg']:
            d[side+'_'+key]=np.ones(len(times))
    assert null_plateau(d)['t_start_model']==.4
    for side in ['LEFT','RIGHT']:
        d[side+'_CC_stress_Pa']=np.arange(len(times))*100000.
    assert null_plateau(d)['t_start_model'] is None
    for side in ['LEFT','RIGHT']:
        d[side+'_CC_stress_Pa']=np.array([0.,0.,100000.,200000.,200000.])
    assert null_plateau(d)['t_start_model'] is None


def test_reused_large_step_preconditioner_does_not_change_solved_equation():
    f,op,it=fixture(mirror=True)
    f=.2+.001*(f-np.mean(f))
    it.step(f,.1);factor=it._preconditioner
    cached=it.step(f,.12)
    assert it._preconditioner is factor
    fresh=ImplicitSurfaceDiffusion(op).step(f,.12)
    np.testing.assert_allclose(cached,fresh,atol=2e-11,rtol=0)
    assert abs(np.sum((cached-f)*op.g['r_c']))<1e-21


@pytest.mark.parametrize('harmonic_faces', [False, True])
def test_optional_small_step_cache_retains_current_operator_and_mass(harmonic_faces):
    from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
    f,op,_=fixture(mirror=True)
    f=.2+.001*(f-np.mean(f))
    cls=HarmonicSurfaceDiffusion if harmonic_faces else ImplicitSurfaceDiffusion
    cached_solver=cls(op,reuse_small_step_preconditioner=True)
    first=cached_solver.step(f,.001);factor=cached_solver._preconditioner
    cached=cached_solver.step(first,.0012)
    assert cached_solver._preconditioner is factor
    fresh=cls(op).step(first,.0012)
    np.testing.assert_allclose(cached,fresh,atol=2e-11,rtol=0)
    assert abs(np.sum((cached-first)*op.g['r_c']))<1e-21
