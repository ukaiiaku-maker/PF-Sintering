import numpy as np
from functools import partial
from pf_sintering.three_particle_event import pair_transfer
from pf_sintering.current_state_mass_transfer import bounded_conservative_transfer


def fixture():
    f=np.full((9,8),.6);r=np.arange(8)+.5;receiver=np.zeros_like(f);donor=np.zeros_like(f)
    receiver[2:4,2:5]=1.;donor[5:7,2:5]=1.
    return f,r,receiver,donor


def test_exact_bicrystal_source_reduction():
    f,r,receiver,donor=fixture();a=.4*f;b=f-a;kwargs=dict(transfer_volume_m3=.05,r_c=r,dr=1.,dz=1.)
    expected,ed=bounded_conservative_transfer((f,a,b),receiver,donor,**kwargs)
    actual,ad=pair_transfer((f,a,b,np.zeros_like(f)),receiver,donor,pair=(0,1),**kwargs)
    for x,y in zip(actual[:3],expected):np.testing.assert_array_equal(x,y)
    assert ad['inactive_eta_bitwise_preserved']


def test_inactive_eta_unchanged_and_pair_reflection():
    f,r,receiver,donor=fixture();state=(f,.2*f,.5*f,.3*f)
    kwargs=dict(transfer_volume_m3=.05,r_c=r,dr=1.,dz=1.)
    out,diag=pair_transfer(state,receiver,donor,pair=(0,1),**kwargs)
    np.testing.assert_array_equal(out[3],state[3]);np.testing.assert_allclose(sum(out[1:]),out[0],atol=1e-16)
    reflected=(f[::-1],state[3][::-1],state[2][::-1],state[1][::-1])
    mirrored,_=pair_transfer(reflected,receiver[::-1],donor[::-1],pair=(2,1),**kwargs)
    for a,b in [(0,0),(1,3),(2,2),(3,1)]:np.testing.assert_allclose(out[a],mirrored[b][::-1],atol=1e-16)


def test_event_zero_and_midpoint_restart_exactly_once():
    from pf_sintering.production_mass_transfer_event import current_state_mass_transfer_event
    from pf_sintering.model_time_transport import ModelTimeGBTransport
    z=np.linspace(-8.,8.,65);r=np.linspace(.125,8.125,33);Z,R=np.meshgrid(z,r,indexing='ij')
    f=.5*(1-np.tanh((np.hypot(Z,R-3)-3)/.5));state=(f,.2*f,.5*f,.3*f)
    setup=dict(z=z,r_c=r,dr=r[1]-r[0],dz=z[1]-z[0],W=.5)
    transport=ModelTimeGBTransport(1.,1.,1.,1.,.001,1.)
    def metrics(state,q):
        return dict(transport_affinity_Pa=1.,contact_area_m2=1.,sigma_local_Pa=1.,sigma_local_MPa=1.,
                    sigma_integral_continuous_Pa=1.,transport_affinity_MPa=1.,r_neck_nm=3.,r_TJ_nm=3.,A1_nm=0.,z_TJ_m=0.)
    def evaluator(*state):return dict(z_TJ_m=0.,r_TJ_m=3.,contact_area_m2=1.)
    def relax(state,q):return state,metrics(state,q),dict(converged=True)
    def run(state,target=1.,restart=None,cap=400,callback=None,**options):
        return current_state_mass_transfer_event(state,setup,{},evaluator,transport,target,
            fast_relax_fn=relax,state_metrics_fn=metrics,transfer_fn=partial(pair_transfer,pair=(0,1)),
            event_restart=restart,maximum_accepted_states=cap,accepted_progress_callback=callback,**options)
    direct=run(state)
    assert direct[4]
    zero=run(state,cap=0);assert not zero[4];assert zero[5]['event_progress_over_b']==0.
    from_zero=run(zero[:4],restart=zero[5]['event_restart'])
    halfway=run(state,.5);resumed=run(halfway[:4],restart=halfway[5]['event_restart'])
    for reference,result in [(direct,from_zero),(direct,resumed)]:
        for x,y in zip(reference[:4],result[:4]):np.testing.assert_allclose(x,y,rtol=0,atol=1e-15)
        np.testing.assert_allclose(reference[5]['event_time_model'],result[5]['event_time_model'],rtol=1e-13)
    repeat=run(direct[:4],restart=direct[5]['event_restart'])
    assert len(repeat[5]['packets'])==0
    for x,y in zip(repeat[:4],direct[:4]):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(direct[3],state[3])

    # Four hundred 0.0025b additions must finish the quota before the cap.
    fine=run(state,initial_step_over_b=.0025,maximum_step_over_b=.0025)
    assert fine[4], fine[5]['stop_reason']
    assert fine[5]['event_progress_over_b']==1.
    assert fine[5]['event_restart']['accepted_steps_total']==400
    np.testing.assert_allclose(sum(p['dq_m'] for p in fine[5]['packets']),transport.b_m,rtol=1e-13)

    # Interrupt an adaptive path after a rejection and before its growth cycle.
    original_metrics=metrics
    def metrics(fields,q):
        row=original_metrics(fields,q);row['sigma_local_MPa']=19.*q
        return row
    adaptive=run(state)
    saved={}
    class Interrupted(Exception):pass
    def checkpoint(packet,fields,restart):
        if restart['accepted_steps_total']==5:
            saved.update(fields=fields,restart=restart)
            raise Interrupted()
    import pytest
    with pytest.raises(Interrupted):run(state,callback=checkpoint)
    resumed_adaptive=run(saved['fields'],restart=saved['restart'])
    for x,y in zip(adaptive[:4],resumed_adaptive[:4]):np.testing.assert_array_equal(x,y)
    assert adaptive[5]['event_time_model']==resumed_adaptive[5]['event_time_model']
    assert adaptive[5]['event_restart']['quasistatic_trial_rejections_total']==resumed_adaptive[5]['event_restart']['quasistatic_trial_rejections_total']


def test_checkpoint_roundtrip_and_pair_ownership_binary_reduction(tmp_path):
    from test_three_particle_implicit import fixture as operator_fixture
    from pf_sintering.three_particle_event import save_event_checkpoint,load_event_checkpoint,ownership_pair_step
    from pf_sintering.corrected_axisym_dynamics import _phi_update_kernel
    f,op,_=operator_fixture();g=op.g;phi=np.zeros((3,*f.shape));phi[0]=.4+.1*np.sin(g['z'][:,None]/3e-9);phi[1]=1-phi[0]
    result=ownership_pair_step(phi,f,op,(0,1),1e-6,1e-10)
    po=np.empty_like(f);e1=np.empty_like(f);e2=np.empty_like(f)
    _phi_update_kernel(phi[0],f,op.Wc,op.k_eta,g['dr'],g['dz'],g['r_c'],g['r_f'],1e-6,1e-10,po,e1,e2)
    np.testing.assert_allclose(result[0],po,rtol=0,atol=2e-16)
    np.testing.assert_array_equal(result[2],phi[2])
    state=(f,*(phi*f[None]));restart=dict(base_fields=state,union_previous_fields=state,cumulative_q_m=.125e-9,event_time_model=2.,accepted_steps_total=25)
    path=tmp_path/'event.npz';save_event_checkpoint(path,state,restart,contact='LEFT',label='FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION')
    restored,r,contact,label=load_event_checkpoint(path)
    for x,y in zip(restored,state):np.testing.assert_array_equal(x,y)
    assert r['cumulative_q_m']==restart['cumulative_q_m'];assert contact=='LEFT';assert label.startswith('FORCED')


def test_compiled_threegrain_kernels_match_reference():
    from test_three_particle_implicit import fixture as operator_fixture
    from pf_sintering.three_particle_event import ownership_pair_step,ownership_pair_step_reference,update_ownership
    from pf_sintering.corrected_interfacial_energy import _axisym_gate_derivative_of_weighted_gradient
    f,op,_=operator_fixture();g=op.g;rng=np.random.default_rng(81)
    phi=rng.uniform(.1,1.,(3,*f.shape));phi/=phi.sum(axis=0)
    expected=ownership_pair_step_reference(phi,f,op,(1,2),1e-6,1e-10)
    np.testing.assert_allclose(ownership_pair_step(phi,f,op,(1,2),1e-6,1e-10),expected,rtol=0,atol=2e-16)
    density=op.Wc*sum(phi[i]*phi[j] for i in range(3) for j in range(i+1,3))
    for p in phi:density+=_axisym_gate_derivative_of_weighted_gradient(p,op.k_eta/2,g['dr'],g['dz'],g['r_c'],g['r_f'])
    update_ownership(op,phi);np.testing.assert_allclose(op.gb_density,density,rtol=3e-16,atol=0.)


def test_fast_pair_ownership_roundtrip_does_not_accumulate_closure():
    from pf_sintering.three_particle_event import normalized_pair_ownership
    rng=np.random.default_rng(4);f=rng.uniform(.1,1.,(7,9));phi=rng.uniform(.1,1.,(3,*f.shape));phi/=phi.sum(axis=0)
    state=(f,*(phi*f[None]));inactive=state[3].copy()
    for _ in range(2000):
        phi=normalized_pair_ownership(state,phi,(0,1));state=(f,*(phi*f[None]))
    assert np.max(abs(sum(state[1:])-f))<=3e-16
    np.testing.assert_allclose(state[3],inactive,rtol=0,atol=3e-16)
    # With no third grain, reconstruct exactly the binary complementary phi.
    state=(f,.4*f,.6*f,np.zeros_like(f));phi=normalized_pair_ownership(state,phi,(0,1))
    np.testing.assert_array_equal(phi[1],1.-phi[0]);np.testing.assert_array_equal(phi[2],0.)
