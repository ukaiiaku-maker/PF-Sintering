"""Adaptive current-field PF continuation; separate outputs from native evidence."""
from pathlib import Path
import sys,os,json,time,argparse,csv
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_cmc_ripening import load_initial,observe
from three_particle_cmc_calibrate import measure
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_implicit import ImplicitSurfaceDiffusion
from pf_sintering.three_particle_geometry import grain_volumes,topology_status


def projected_mu(f,op):
    g=op.g;mu=op.potential(f);w=np.maximum(-np.gradient(f,g['dr'],axis=1),0.)
    den=w.sum(axis=1);p=np.divide((mu*w).sum(axis=1),den,out=np.full(len(den),np.nan),where=den>1.)
    z=g['z'];L=g['gb'][1];W=op.W;full=(f[:,0]>.999)&np.isfinite(p)
    return np.array([np.mean(p[m&full]) for m in [z<-L-2*W,abs(z)<L-2*W,z>L+2*W]])


def record(f,op,meta,t,step):
    row,_=observe(f,op,meta['rule'],t,step)
    mu=projected_mu(f,op)
    row.update(mu_projected_left_Pa=mu[0],mu_projected_center_Pa=mu[1],mu_projected_right_Pa=mu[2],
               delta_mu_projected_Pa=mu[1]-.5*(mu[0]+mu[2]),
               center_GB_separation_over_W=float(np.diff(op.g['gb'])[0]/op.W),
               center_solid_core_f=float(f[np.argmin(abs(op.g['z'])),0]))
    return row


def advance(f,h,it,rule,field_tol=2e-5):
    full=it.step(f,h);half=it.step(f,h/2);fine=it.step(half,h/2)
    # Richardson extrapolation cancels the first-order time error. All three
    # increments are conservative native face-divergences, hence so is this.
    result=2*fine-full
    if result.min() < -1e-8 or result.max()>1+1e-8 or not np.isfinite(result).all():
        raise FloatingPointError('extrapolated bounds')
    a=measure(full,it.op.g,rule);b=measure(fine,it.op.g,rule)
    angle_error=max(abs(x['psi_deg']-y['psi_deg']) for x,y in zip(a,b))
    curvature_error=max(np.max(abs(np.array(x['meridional_curvatures'])-y['meridional_curvatures'])) for x,y in zip(a,b))
    err=max(np.max(abs(fine-full))/field_tol,angle_error/.03,curvature_error/30000.,np.max(abs(result-f))/.025)
    return result,float(err)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['unequal','null'],required=True)
    ap.add_argument('--end-native-steps',type=float,default=10000);ap.add_argument('--label',default='overlap')
    ap.add_argument('--field-tol',type=float,default=2e-5);ap.add_argument('--max-wall-s',type=float,default=900)
    ap.add_argument('--resume',type=Path);ap.add_argument('--max-step-native',type=float,default=1e6)
    args=ap.parse_args()
    if args.end_native_steps>10000:
        gate=Path('docs/three_particle/implicit/overlap.json')
        if not gate.exists() or json.loads(gate.read_text())['status']!='PASS':
            raise RuntimeError('both native overlap cases must pass before extension')
    g,meta=load_initial(args.case);op=PhaseAOperator(g);it=ImplicitSurfaceDiffusion(op)
    f=g['f'].copy();v0=grain_volumes(f,g);t=0.;h=meta['dt_model']*100;step=0
    if args.resume:
        with np.load(args.resume,allow_pickle=False) as d:
            np.testing.assert_array_equal(d['z'],g['z']);np.testing.assert_array_equal(d['ownership'],g['ownership'])
            f=d['f'].copy();t=float(d['t_model']);h=float(d['next_h']) if 'next_h' in d else h
    out=Path(f'runs/three_particle_implicit/{args.label}_{args.case}');out.mkdir(parents=True,exist_ok=False)
    end=args.end_native_steps*meta['dt_model'];start=time.perf_counter();rows=[record(f,op,meta,t,step)]
    rejected=0;rejection_reasons={};status='running';last_print=start;next_native=(np.floor(t/(1000*meta['dt_model']))+1)*1000*meta['dt_model']
    native_end=10000*meta['dt_model'];frames=[f.copy()];frame_t=[t];next_frame=max(native_end,t*1.25)
    def save():
        with (out/'history.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        temp=out/'checkpoint.tmp.npz'
        np.savez_compressed(temp,f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],gb=g['gb'],t_model=t,next_h=h,metadata_json=json.dumps(meta),events_enabled=False)
        os.replace(temp,out/'checkpoint.npz')
        report=dict(case=args.case,status=status,wall_s=time.perf_counter()-start,accepted=step,rejected=rejected,rejection_reasons=rejection_reasons,t_model=t,t_s=t*op.physics.seconds_per_model_time,initial=rows[0],final=rows[-1],grain_volume_relative_changes=(grain_volumes(f,g)/v0-1).tolist(),settings=vars(args)|{'resume':str(args.resume)},phase_b_authorized=False)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    while t<end*(1-1e-13):
        h=min(h,end-t,args.max_step_native*meta['dt_model'])
        if t<native_end*(1-1e-12):h=min(h,next_native-t)
        reason='nonlinear_error_estimate'
        try:
            trial,err=advance(f,h,it,meta['rule'],args.field_tol)
        except (FloatingPointError,RuntimeError) as exc:
            err=np.inf;reason=str(exc)
        if err>1:
            rejection_reasons[reason]=rejection_reasons.get(reason,0)+1
            rejected+=1;h*=max(.2,.8/np.sqrt(err))
            if h<meta['dt_model']/8:status='failed_timestep_floor';break
            continue
        f=trial;t+=h;step+=1
        row=record(f,op,meta,t,step);rows.append(row)
        if t<=native_end*(1+1e-12) and t>=next_native*(1-1e-12):
            frames.append(f.copy());frame_t.append(t);next_native+=1000*meta['dt_model']
        elif t>native_end and t>=next_frame:
            frames.append(f.copy());frame_t.append(t);next_frame=t*1.25
        h*=min(2.,max(.5,.9/np.sqrt(max(err,1e-8))))
        if topology_status(f,g)['stop']:status='topology_stop';break
        if abs(row['total_volume_m3']/v0.sum()-1)>1e-11 or row['mirror_error']>1e-8:
            status='conservation_or_symmetry_stop';break
        if time.perf_counter()-last_print>20:
            print(json.dumps(dict(case=args.case,t_model=t,steps=step,rejected=rejected,h=h,dVc=row['V_center_m3']/v0[1]-1,CC_MPa=row['LEFT_CC_stress_Pa']/1e6,wall_s=time.perf_counter()-start)),flush=True);save();last_print=time.perf_counter()
        if time.perf_counter()-start>args.max_wall_s:status='wall_budget_stop';break
        if abs(row['V_center_m3']/v0[1]-1)>=.001:status='target_volume_change';break
    if status=='running':status='end_time'
    save();np.savez_compressed(out/'frames.npz',f=np.array(frames),t_model=frame_t)
    print((out/'report.json').read_text(),flush=True)
if __name__=='__main__':main()
