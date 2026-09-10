"""Off-trajectory diagnosis of a minimum-increment fast-manifold stop.

Extending an iteration budget here does not promote the result to production.
All physical coefficients and convergence tolerances remain unchanged.
"""
from pathlib import Path
import sys,json,time,argparse
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,TOL,MANIFEST
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint,pair_transfer,ownership_pair_step
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update
from pf_sintering.current_state_mass_transfer import build_current_state_masks
from pf_sintering.quasistatic_event_continuation import fast_manifold_increment,fast_manifold_converged

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='fast_stop_audit');args=ap.parse_args()
    source=Path('runs/three_particle_production_065/forced_left_compiled/final.npz')
    state,restart,_,_=load_event_checkpoint(source);q=restart['cumulative_q_m']/MANIFEST['b_event_m']
    if q>=1:raise RuntimeError('no partial event stop to audit')
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);e=ContactEvent(g);e.metrics(restart['base_fields'],0.)
    before=e.metrics(state,q);receiver,donor,_=build_current_state_masks(state[0],g['z'],g['r_c'],z_TJ_m=before['z_TJ_m'],r_TJ_m=before['r_n_m'],W_m=e.op.W)
    trial,_=pair_transfer(state,receiver,donor,pair=(0,1),transfer_volume_m3=before['contact_area_m2']*.0025*MANIFEST['b_event_m'],r_c=g['r_c'],dr=g['dr'],dz=g['dz'])
    previous=e.metrics(trial,q+.0025);history=[];consecutive=0;start=time.perf_counter();accepted=False
    for block in range(1,241):
        for _ in range(10):
            e.bind(trial);f=trial[0];phi=g['ownership'];fn=bounded_mobility_update(f,e.op.potential(f).copy(),e.op,e.dt)
            if fn.min() < -1e-8 or fn.max()>1+1e-8:raise RuntimeError('unchanged fast field guard')
            pn=ownership_pair_step(phi,fn,e.op,(0,1),e.dt,1.0937500000000001e-25);trial=(fn,*(pn*fn[None]))
        row=e.metrics(trial,q+.0025);inc=fast_manifold_increment(previous,row)
        passed=fast_manifold_converged(inc,TOL);consecutive=consecutive+1 if block>=3 and passed else 0
        history.append(dict(block=block,increments=inc,passing=passed,consecutive=consecutive))
        if consecutive>=3:accepted=True;break
        previous=row
    result=dict(label='OFF_TRAJECTORY_NUMERICAL_BUDGET_AUDIT',source=str(source),q_start=q,trial_dq_over_b=.0025,
        original_max_blocks=60,audit_max_blocks=240,converged=accepted,blocks=block,history=history,tolerances=TOL,
        wall_s=time.perf_counter()-start,before=before,trial_after=row,physical_parameters_changed=False,production_qualified=False,
        field_bounds=[float(trial[0].min()),float(trial[0].max())],partition_closure=float(np.max(abs(sum(trial[1:])-trial[0]))))
    (Path('docs/three_particle/production_065')/(args.label+'.json')).write_text(json.dumps(result,indent=2,default=float)+'\n');print('audit',accepted,block,flush=True)
if __name__=='__main__':main()
