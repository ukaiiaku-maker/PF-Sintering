import numpy as np
from pf_sintering.three_particle_loading import decompose,plateau,trial_invariant_reason

def test_decomposition_keeps_one_sided_cancellation_and_mean():
    c=dict(gamma_J_per_m2=1.,r_n_m=1e-7,sigma_local_Pa=15e6)
    for side,k in [('negative',10e6),('positive',20e6)]:c.update({f'kappa1_{side}_per_m':k,f'theta_{side}_rad':np.pi,f'sigma_local_{side}_Pa':30e6-k})
    d=decompose(c);assert d['production']['curvature_Pa']==-15e6;assert np.isclose(d['production']['TJ_Pa'],30e6)

def test_plateau_requires_both_contacts_and_whole_volume_window():
    def rows(slope):return [dict(center_loss_fraction=x,contacts={'LEFT':{'sigma_local_Pa':18e6+1e6*x},'RIGHT':{'sigma_local_Pa':18e6+slope*x}}) for x in np.linspace(.009,.022,60)]
    assert plateau(rows(1e6))['reached']
    assert not plateau(rows(10e6))['reached']
    assert not plateau(rows(1e6)[-5:])['reached']


def test_trial_guard_rejects_asymmetry_without_projecting_fields():
    f=np.full((4,3),.5);f[0,1]+=2e-8;before=f.copy()
    assert trial_invariant_reason(f,1.,1.)=='reflection guard'
    np.testing.assert_array_equal(f,before)
    f[0,1]=.5+5e-9
    assert trial_invariant_reason(f,1.,1.) is None
    assert trial_invariant_reason(f,1.+2e-11,1.)=='material guard'


def test_trial_guard_can_leave_reflection_as_diagnostic_only():
    f=np.zeros((4,3));f[0,0]=1e-7;original=f.copy()
    assert trial_invariant_reason(f,1.,1.,enforce_reflection=False) is None
    np.testing.assert_array_equal(f,original)
