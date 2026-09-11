import numpy as np
import pytest
from pf_sintering.three_particle_renewal import RootClocks,locate_first_root


def test_field_crossing_rollback_and_renewal_restart():
    clocks=RootClocks(np.random.default_rng(19));clocks.threshold={'LEFT':.3,'RIGHT':.8}
    before=clocks.snapshot();calls=[]
    def advance(f,dt):calls.append(dt);return f+dt
    def rates(f):return {'LEFT':1.+float(f[0]),'RIGHT':2.}
    f=np.array([0.]);field,t,inc,contact=locate_first_root(f,1.,advance,rates,clocks,1e-8)
    assert contact=='LEFT';assert abs(t-(-1+np.sqrt(1.6)))<1e-8
    assert len(calls)>10;np.testing.assert_array_equal(f,[0.]);assert clocks.snapshot()==before
    np.testing.assert_allclose(field,[t]);clocks.commit(inc,contact)
    with pytest.raises(RuntimeError):clocks.increments(rates(f),rates(f),.1)
    restored=RootClocks.restore(clocks.snapshot());right=clocks.hazard['RIGHT'];threshold=clocks.threshold['RIGHT']
    clocks.extinct();restored.extinct();assert clocks.snapshot()==restored.snapshot()
    assert clocks.hazard['RIGHT']==right and clocks.threshold['RIGHT']==threshold
    assert clocks.hazard['LEFT']==0 and clocks.threshold['LEFT']!=.3


def test_right_can_fire_first_and_no_crossing_preserves_rng():
    clocks=RootClocks(np.random.default_rng(3));clocks.threshold={'LEFT':10.,'RIGHT':.1}
    result=locate_first_root(np.array([0.]),1.,lambda f,t:f+t,lambda f:{'LEFT':1.,'RIGHT':1.},clocks,1e-7)
    assert result[-1]=='RIGHT';assert abs(result[1]-.1)<1e-7
    state=clocks.snapshot();inc=clocks.increments({'LEFT':1.,'RIGHT':1.},{'LEFT':1.,'RIGHT':1.},.01)
    clocks.commit(inc);assert clocks.rng.bit_generator.state==state['rng']


def test_microsecond_descendant_crossing_resolves_to_picosecond_tolerance():
    clocks=RootClocks(np.random.default_rng(71));clocks.threshold={'LEFT':.73,'RIGHT':100.}
    initial=np.array([0.]);rate=2e5;slope=1e4
    field,t,increment,contact=locate_first_root(initial,.0004,lambda f,dt:f+dt,
        lambda f:{'LEFT':rate*(1+slope*float(f[0])),'RIGHT':0.},clocks,1e-12)
    exact=(np.sqrt(1+2*slope*.73/rate)-1)/slope
    assert contact=='LEFT' and 0 <= t-exact <= 1e-12
    np.testing.assert_array_equal(initial,[0.]);np.testing.assert_allclose(field,[t],rtol=0,atol=0)


def test_counted_strain_quota_survives_windows_failure_and_resume():
    from pf_sintering.three_particle_renewal import cumulative_event_quota
    sequence=[(0,'POST_TRANSIENT_NEW_TRAJECTORY',0),(0,'ROOT_CROSSING',0),
              (1,'ACTIVE_ONE_B',.5),(1,'ONE_B_COMPLETE',1),
              (1,'SOURCE_WINDOW_OPEN',0),(1,'CHILD_CROSSING',0),
              (2,'ACTIVE_ONE_B',.5),(2,'ONE_B_FAILED',.5),
              (2,'ACTIVE_EVENT_RESUMED',.5),(2,'ONE_B_COMPLETE',1),
              (2,'AVALANCHE_EXTINCT_REPINNED',0),(2,'RELOAD',0)]
    np.testing.assert_array_equal([cumulative_event_quota(*r) for r in sequence],
                                 [0,0,.5,1,1,1,1.5,1.5,1.5,2,2,2])
    assert cumulative_event_quota(1,'FORCED_ROOT_COMPLETE',1)==1
    with pytest.raises(ValueError):cumulative_event_quota(0,'ACTIVE_ONE_B',.5)
    with pytest.raises(ValueError):cumulative_event_quota(1,'UNKNOWN',0)
