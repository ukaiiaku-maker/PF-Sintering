"""Prescribed compatible-CMC bracket and finite-width null qualification."""
from pathlib import Path
import sys,json,time,argparse
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from scipy.optimize import brentq
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf,describe
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_null import inspect_state
from pf_sintering.three_particle_geometry import grain_volumes
from pf_sintering.axisym_numba_kernel import flux_kernel,div_and_update_kernel
from three_particle_cmc_cleanup import metrics,check
from three_particle_implicit_run import projected_mu

ROOT=Path('runs/three_particle_stationary_null');DOC=Path('docs/three_particle/stationary_null')


def candidate(ratio,save=True):
    c,o=compatible_chain(float(ratio));g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);f=g['f'].copy()
    rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
    before=metrics(f,g,rule);dt=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4
    faces=[int(np.argmin(abs(g['z']+g['dz']/2-b))) for b in g['gb']]
    for _ in range(200):
        flux_kernel(f,op.potential(f),g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
        for j in faces:op.Jz[j]=0.
        div_and_update_kernel(f,op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],dt,op.out)
        f=op.out.copy()
        if f.min() < -1e-8 or f.max()>1+1e-8:raise FloatingPointError('cleanup bounds')
    changes,gates=check(metrics(f,g,rule),before,g)
    d=inspect_state(f,op);mu=projected_mu(f,op);vol=grain_volumes(f,g)
    metadata=dict(case='pf_null_candidate',ratio=float(ratio),half_length_m=c.half_length,outer_radius_m=c.radius_scale,width_m=op.W,spacing_m=g['dr'],dz_m=g['dz'],dt_model=dt,physical_time=0.,model_time=0.,rule=rule,phase_a_qualified=False,events_enabled=False)
    report=dict(ratio=float(ratio),cmc=describe(c,o,width=op.W),cleanup_steps=200,cleanup_changes=changes,cleanup_gates=gates,diagnostics=d,projected_mu_Pa=mu.tolist(),projected_mu_contrast_Pa=float(mu[1]-.5*(mu[0]+mu[2])),initial_volumes_m3=vol.tolist(),center_relative_rate_per_model_time=d['grain_volume_rates_m3_per_model_time'][1]/vol[1],stationary_qualified=False)
    if save:
        out=ROOT/f'ratio_{ratio:.12f}';out.mkdir(parents=True,exist_ok=True)
        (out/'initial_report.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(out/'initial.npz',f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],gb=g['gb'],metadata_json=json.dumps(metadata),canonical=bool(all(gates.values())))
    return g,f,metadata,report


def main():
    ROOT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rows=[]
    def evaluate(ratio):
        _,_,_,d=candidate(ratio);rows.append(d)
        print(json.dumps(dict(ratio=ratio,rate=d['center_relative_rate_per_model_time'],projected_mu=d['projected_mu_contrast_Pa'],virtual_work_spread=d['diagnostics']['virtual_work_delta_spread_Pa'],wall_s=time.perf_counter()-start)),flush=True)
        return d['center_relative_rate_per_model_time']
    for r in [.72,.73,.7428902074688286,.755,.77]:evaluate(r)
    root=brentq(evaluate,.73,.755,xtol=1e-10,rtol=1e-12)
    _,_,_,final=candidate(root)
    report=dict(status='ZERO_INITIAL_FLUX_CANDIDATE_NOT_STATIONARY_QUALIFICATION',flux_root_ratio=root,root=final,bracket_and_iterations=rows,events_enabled=False,phase_b_authorized=False)
    (DOC/'bracket.json').write_text(json.dumps(report,indent=2)+'\n')
    print('root',root,flush=True)
if __name__=='__main__':main()
