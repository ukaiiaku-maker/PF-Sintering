"""Release the independently determined zero-initial-flux candidate."""
from pathlib import Path
import sys,json,time,csv,os
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_implicit import ImplicitSurfaceDiffusion
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_diagnostics import curvature_watch
from pf_sintering.three_particle_null import inspect_state
from three_particle_implicit_run import advance,record


def main():
    root=Path('runs/three_particle_stationary_null');doc=Path('docs/three_particle/stationary_null')
    calibration=json.loads((doc/'bracket.json').read_text());ratio=calibration['flux_root_ratio']
    source=root/f'ratio_{ratio:.12f}'/'initial.npz'
    with np.load(source,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata_json']));f=d['f'].copy();phi=d['ownership'].copy();z=d['z'].copy()
        if not bool(d['canonical']):raise RuntimeError('cleanup guards failed')
    c,o=compatible_chain(ratio);g=map_to_pf(c,o,meta['width_m'],meta['spacing_m'])
    np.testing.assert_array_equal(z,g['z']);np.testing.assert_array_equal(phi,g['ownership']);g['f']=f;g['eta']=phi*f[None]
    op=PhaseAOperator(g);it=ImplicitSurfaceDiffusion(op);v0=grain_volumes(f,g);watch0=curvature_watch(f,op)
    # Independent native released response, without changing the candidate.
    native=f.copy();short=[]
    for i in range(1001):
        if i in [0,1,10,100,1000]:short.append(dict(steps=i,t_model=i*meta['dt_model'],relative_center_change=float(grain_volumes(native,g)[1]/v0[1]-1),field_change=float(np.max(abs(native-f)))))
        if i<1000:native=op.step(native,meta['dt_model'])
    out=root/'released_flux_root';out.mkdir(exist_ok=False)
    t=0.;end=.9/op.physics.seconds_per_model_time;h=100*meta['dt_model'];step=0;rejected=0;reasons={};start=time.perf_counter();last=start;status='running'
    rows=[record(f,op,meta,t,step)];frames=[f.copy()];times=[t];next_frame=1000*meta['dt_model']
    def save():
        with (out/'history.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
        temp=out/'checkpoint.tmp.npz'
        np.savez_compressed(temp,f=f,ownership=phi,z=z,r_c=g['r_c'],gb=g['gb'],t_model=t,next_h=h,metadata_json=json.dumps(meta),events_enabled=False)
        os.replace(temp,out/'checkpoint.npz')
        report=dict(status=status,ratio=ratio,t_model=t,t_s=t*op.physics.seconds_per_model_time,requested_seconds=.9,wall_s=time.perf_counter()-start,accepted_steps=step,rejected_steps=rejected,rejections=reasons,relative_grain_changes=(grain_volumes(f,g)/v0-1).tolist(),initial=rows[0],final=rows[-1],native_short_release=short,phase_b_authorized=False,stationary_qualified=False)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    while t<end*(1-1e-13):
        h=min(h,end-t,2.)
        reason='nonlinear_error_estimate'
        try:trial,err=advance(f,h,it,meta['rule'],2e-5)
        except (FloatingPointError,RuntimeError) as exc:err=np.inf;reason=str(exc)
        if err>1:
            rejected+=1;reasons[reason]=reasons.get(reason,0)+1;h*=max(.2,.8/np.sqrt(err))
            if h<meta['dt_model']/8:status='numerical_guard_stop';break
            continue
        f=trial;t+=h;step+=1;rows.append(record(f,op,meta,t,step))
        if t>=next_frame:frames.append(f.copy());times.append(t);next_frame=max(t*1.5,t+1000*meta['dt_model'])
        h*=min(2.,max(.5,.9/np.sqrt(max(err,1e-8))))
        if topology_status(f,g)['stop']:status='topology_stop';break
        if abs(rows[-1]['total_volume_m3']/v0.sum()-1)>1e-11 or rows[-1]['mirror_error']>1e-8:status='mass_or_symmetry_stop';break
        if time.perf_counter()-last>20:
            print(json.dumps(dict(t_s=t*op.physics.seconds_per_model_time,steps=step,dVc=grain_volumes(f,g)[1]/v0[1]-1,delta_mu=rows[-1]['delta_mu_projected_Pa'],CC=rows[-1]['LEFT_CC_stress_Pa'],wall_s=time.perf_counter()-start)),flush=True);save();last=time.perf_counter()
        if time.perf_counter()-start>900:status='wall_budget_stop';break
    if status=='running':status='requested_time_reached'
    save();np.savez_compressed(out/'frames.npz',f=np.array(frames),t_model=times)
    report=json.loads((out/'report.json').read_text());report.update(initial_curvature_watch=watch0,final_curvature_watch=curvature_watch(f,op),final_field_diagnostics=inspect_state(f,op),topology=topology_status(f,g))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(doc/'released_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(status,flush=True)
if __name__=='__main__':main()
