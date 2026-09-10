import numpy as np
from pf_sintering.three_particle_geometry import ThreeParticleConfig,build_three_particle
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_diagnostics import diagnostics,cannon_carter,curvature_watch

def test_cannon_carter_line_and_curvature_balance():
    r=30e-9;gamma=1.;psi=np.deg2rad(160)
    zero=cannon_carter(r,psi,2*np.sin(psi/2)/r,gamma)
    assert abs(zero['stress_Pa'])<1e-7
    line=cannon_carter(r,psi,0,gamma)
    np.testing.assert_allclose(line['force_N'],2*np.pi*r*gamma*np.sin(psi/2),rtol=1e-15)

def test_contact_symmetry_and_existing_watcher():
    g=build_three_particle(ThreeParticleConfig(spacing=1.25e-9));op=PhaseAOperator(g)
    d,_=diagnostics(g['f'],op)
    for suffix in ['neck_r_m','kappa_per_m','CC_force_N','PF_geometric_stress_Pa']:
        np.testing.assert_allclose(d['LEFT_'+suffix],d['RIGHT_'+suffix],rtol=1e-11)
    assert d['LEFT_gb_z_m']==g['gb'][0] and d['RIGHT_tj_z_m']==g['gb'][1]
    watched=curvature_watch(g['f'],op)
    assert len(watched)==4
    assert all(np.isfinite(r['max_abs_kappa_per_m']) for r in watched.values())
