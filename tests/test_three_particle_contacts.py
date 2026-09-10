import json
from pathlib import Path
import numpy as np
from pf_sintering.three_particle_contacts import contact_stresses,root_law
from pf_sintering.pr_experimental_geometry import experimental_geometry_record
from pf_sintering.experimental_pr_metrology import _side_first_stresses,_particle_silhouette


def test_absent_third_grain_reduces_to_bicrystal_field_evaluator():
    # Asymmetric two-grain field, nonzero TJ offset, no third ownership.
    z=np.linspace(-90e-9,110e-9,401);r=np.linspace(.25e-9,90.25e-9,181)
    zt=7.3e-9;W=4e-9
    R=52e-9+.15*np.abs(z-zt)-.002*(z-zt)**2/1e-9
    f=.5*(1+np.tanh((R[:,None]-r[None,:])/W))
    phi=.5*(1+np.tanh((z[:,None]-zt)/W))*np.ones_like(f)
    state=(f,f*(1-phi),f*phi)
    setup=dict(z=z,r_c=r,r_f=np.arange(182)*.5e-9,dr=.5e-9,dz=z[1]-z[0],W=W,
               gamma_s=1.,gamma_gb=.34729635533386083,interfacial_energy_evaluator=lambda state,setup:0.)
    rt=float(np.interp(zt,z,R))
    geometry,branches=experimental_geometry_record(state,setup,z_tj_m=zt,r_tj_m=rt,local_window_m=3*W)
    expected=_side_first_stresses(geometry,_particle_silhouette(branches['positive']))
    actual,_=contact_stresses(*state,setup,zt,rt)
    for key in ['sigma_local_Pa','sigma_integral_continuous_Pa','sigma_local_negative_Pa',
                'sigma_local_positive_Pa','sigma_integral_continuous_negative_Pa','sigma_integral_continuous_positive_Pa']:
        np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=0)


def test_root_law_matches_production_rate_and_formation_penalty():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import pr_coarsening_stochastic_two_event as production
    from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
    m=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text())
    p=m['root_barrier_slice'];old=(production.BARRIER,production.TEMPERATURE_K,production.B_M)
    try:
        production.TEMPERATURE_K=m['temperature_K']
        production.B_M=m['b_event_m']
        production.BARRIER=CompleteExpFloorParams(p['G0_eV'],p['Gfloor_eV'],p['a'],p['sigmahat_Pa'],p['n'])
        for stress in [-1e6,0.,20e6,56.5e6,100e6]:
            fit,rate=production.rate(stress,80e-9,m['clock_scale'])
            rate*=np.exp(-m['root_formation_penalty_eV']/(1.380649e-23/1.602176634e-19*m['temperature_K']))
            actual=root_law(stress,80e-9,m)
            np.testing.assert_allclose(actual['G_root_eV'],fit+m['root_formation_penalty_eV'],rtol=1e-15)
            np.testing.assert_allclose(actual['root_rate_per_model_time'],rate,rtol=1e-15)
    finally:production.BARRIER,production.TEMPERATURE_K,production.B_M=old


def test_other_contact_terminates_center_branch_and_excludes_third_grain():
    z=np.linspace(-80e-9,120e-9,401);r=np.linspace(.25e-9,100.25e-9,201);W=4e-9
    radius=55e-9+.1*np.abs(z)-.0005*z*z/1e-9
    f=.5*(1+np.tanh((radius[:,None]-r[None,:])/W));eta=f*.5
    setup=dict(z=z,r_c=r,dr=.5e-9,dz=z[1]-z[0],W=W,gamma_s=1.)
    upper=40e-9;endpoint=float(np.interp(upper,z,radius))
    first,branches=contact_stresses(f,eta,eta,setup,0.,55e-9,upper=upper,other_contacts=[(upper,endpoint)])
    changed=f.copy();changed[z>upper+W]=0.
    second,_=contact_stresses(changed,eta,eta,setup,0.,55e-9,upper=upper,other_contacts=[(upper,endpoint)])
    assert branches['positive']['z_m'][-1]==upper
    assert branches['positive']['r_m'][-1]==endpoint
    for key in first:np.testing.assert_allclose(first[key],second[key],rtol=0,atol=0)


def test_mirror_contacts_use_same_particle_positive_bicrystal_frame():
    from types import SimpleNamespace
    from pf_sintering.three_particle_contacts import evaluate_contacts
    z=(np.arange(400)-199.5)*.5e-9;r=(np.arange(180)+.5)*.5e-9;W=4e-9;L=20e-9
    radius=60e-9+.15*np.abs(np.abs(z)-L)-.0008*z*z/1e-9
    f=.5*(1+np.tanh((radius[:,None]-r[None,:])/W))
    left=.5*(1+np.tanh((z+L)/.4e-9));right=.5*(1+np.tanh((z-L)/.4e-9))
    phi=np.array([1-left,left-right,right])[:,:,None]*np.ones((1,1,len(r)))
    g=dict(z=z,r_c=r,r_f=np.arange(181)*.5e-9,dr=.5e-9,dz=.5e-9,gb=np.array([-L,L]),ownership=phi)
    op=SimpleNamespace(g=g,W=W,physics=SimpleNamespace(gamma_s=1.),potential=lambda f:np.zeros_like(f))
    m=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text())
    result=evaluate_contacts(f,op,m)
    for key in ['sigma_local_Pa','sigma_integral_Pa','sigma_integral_continuous_Pa','root_rate_per_s']:
        np.testing.assert_allclose(result['LEFT'][key],result['RIGHT'][key],rtol=2e-12)
    np.testing.assert_allclose(result['LEFT']['sigma_integral_continuous_positive_Pa'],
                               result['RIGHT']['sigma_integral_continuous_negative_Pa'],rtol=2e-12)
