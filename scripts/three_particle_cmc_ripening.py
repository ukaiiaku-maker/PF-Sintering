"""Events-OFF native-PF trajectory from a guarded CMC initial checkpoint."""
from pathlib import Path
import sys,json,time,argparse,csv,resource
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from scipy.signal import savgol_filter
from pf_sintering.three_particle_cmc import compatible_chain,chemical_potential_null,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_diagnostics import diagnostics,cannon_carter,curvature_watch
from pf_sintering.pr_stress_metrology import local_3d_contact_stress
from three_particle_cmc_calibrate import measure,contour


def load_initial(case):
    source=Path(f'runs/three_particle_cmc/{case}_short_aligned/canonical_initial.npz')
    with np.load(source,allow_pickle=False) as saved:
        if not bool(saved['canonical']):raise RuntimeError('canonical initialization required')
        metadata=json.loads(str(saved['metadata_json']));f=saved['f'].copy();phi=saved['ownership'].copy();z=saved['z'].copy()
    c,o=chemical_potential_null() if case=='null' else compatible_chain(.7 if case=='unequal' else 1.)
    g=map_to_pf(c,o,metadata['width_m'],metadata['spacing_m'])
    np.testing.assert_array_equal(z,g['z']);g['f']=f;g['ownership']=phi;g['eta']=phi*f[None,:,:]
    return g,metadata


def observe(f,op,rule,t,step):
    row,profile=diagnostics(f,op);g=op.g;R=contour(f,g,.5)
    for name,b,m in zip(['LEFT','RIGHT'],g['gb'],measure(f,g,rule)):
        rb=float(np.interp(b,g['z'][np.isfinite(R)],R[np.isfinite(R)]));sl=np.array(m['slopes'])
        km=np.array(m['meridional_curvatures']);K=float(np.mean(km+1/(rb*np.sqrt(1+sl*sl))))
        psi=np.deg2rad(m['psi_deg']);cc=cannon_carter(rb,psi,K,op.physics.gamma_s)
        pf=local_3d_contact_stress(km[0],km[1],rb,psi,op.physics.gamma_s)
        row.update({name+'_'+k:v for k,v in dict(psi_deg=m['psi_deg'],kappa_per_m=K,neck_r_m=rb,CC_force_N=cc['force_N'],CC_stress_Pa=cc['stress_Pa'],PF_geometric_stress_Pa=pf['sigma_3D_local_Pa']).items()})
    mu=op.potential(f);mu_surface=np.full(len(R),np.nan);smooth_K=np.full(len(R),np.nan)
    for j,r in enumerate(R):
        if np.isfinite(r):mu_surface[j]=np.interp(r,g['r_c'],mu[j])
    boundaries=[-np.inf,*g['gb'],np.inf]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        ids=np.flatnonzero(np.isfinite(R)&(g['z']>lo)&(g['z']<hi))
        win=2*int(np.ceil(op.W/g['dz']/2))+1
        if len(ids)<win:continue
        rr=savgol_filter(R[ids],win,3);rp=savgol_filter(R[ids],win,3,deriv=1,delta=g['dz']);rpp=savgol_filter(R[ids],win,3,deriv=2,delta=g['dz'])
        smooth_K[ids]=-rpp/(1+rp*rp)**1.5+1/(rr*np.sqrt(1+rp*rp))
    row.update(t_model=t,t_s=t*op.physics.seconds_per_model_time,step=step)
    return row,dict(r=R,kappa_raw=profile['kappa_per_m'],kappa_smooth=smooth_K,mu_surface=mu_surface)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['unequal','null','equal'],required=True);ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--surface-off',action='store_true');args=ap.parse_args()
    g,meta=load_initial(args.case);op=PhaseAOperator(g);f=g['f'].copy();dt=meta['dt_model'];rule=meta['rule']
    out=Path('runs/three_particle_cmc/ripening_'+args.case+('_off' if args.surface_off else ''));out.mkdir(parents=True,exist_ok=True)
    rows=[];profiles=[];frames=[];watch=[];start=time.perf_counter();v0=grain_volumes(f,g);failure=None
    for i in range(args.steps+1):
        if i%1000==0 or i==args.steps:
            row,p=observe(f,op,rule,i*dt,i);rows.append(row);profiles.append(p);frames.append(f.copy())
            watch.append(curvature_watch(f,op))
            print(json.dumps(dict(case=args.case,step=i,wall_s=time.perf_counter()-start,dV_center_relative=row['V_center_m3']/v0[1]-1,total_volume_relative=row['total_volume_m3']/v0.sum()-1,psi_left_deg=row['LEFT_psi_deg'])),flush=True)
            if topology_status(f,g)['stop']:failure='topology resolution stop';break
        if i<args.steps:f=op.step(f,dt,surface_enabled=not args.surface_off)
    wall=time.perf_counter()-start
    with (out/'history.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'profiles.npz',z=g['z'],t_model=[r['t_model'] for r in rows],**{key:np.array([p[key] for p in profiles]) for key in profiles[0]})
    np.savez_compressed(out/'frames.npz',f=np.array(frames),t_s=[r['t_s'] for r in rows],z=g['z'],r=g['r_c'],ownership=g['ownership'])
    checkpoint=out/'final_checkpoint.npz'
    np.savez_compressed(checkpoint,f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],gb=g['gb'],t_model=i*dt,t_s=i*dt*op.physics.seconds_per_model_time,metadata_json=json.dumps(meta),events_enabled=False)
    with np.load(checkpoint,allow_pickle=False) as restored:restart_exact=bool(np.array_equal(restored['f'],f) and np.array_equal(restored['ownership'],g['ownership']))
    report=dict(status='DETERMINISTIC_TRAJECTORY_REQUIRES_JOINT_QUALIFICATION',case=args.case,surface_enabled=not args.surface_off,failure=failure,steps=i,t_model=i*dt,t_s=i*dt*op.physics.seconds_per_model_time,wall_s=wall,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint_bytes=checkpoint.stat().st_size,restart_roundtrip_exact=restart_exact,initial=rows[0],final=rows[-1],grain_volume_relative_changes=(grain_volumes(f,g)/v0-1).tolist(),total_volume_error=float(grain_volumes(f,g).sum()/v0.sum()-1),initial_checkpoint=meta,watch=watch,phase_b_authorized=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    Path(f'docs/three_particle/cmc/{out.name}_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
