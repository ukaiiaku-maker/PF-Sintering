"""Bounded native-PF cleanup; no global projection or macroscopic shape descent."""
from pathlib import Path
import sys,json,time,argparse,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,chemical_potential_null,map_to_pf,chain_radius,describe
from pf_sintering.three_particle_phase_a import PhaseAOperator,dilute_material_fraction
from pf_sintering.three_particle_geometry import grain_volumes
from pf_sintering.axisym_numba_kernel import flux_kernel,div_and_update_kernel
from three_particle_cmc_calibrate import measure,contour

# Preregistered before cleanup: relative amounts, centroid/Ro, neck radius,
# low-f storage, angle, and exterior displacement. No tolerance is fitted to a run.
LIMITS=dict(grain_volume_relative=1e-4,centroid_over_Ro=1e-4,neck_relative=.005,
            dilute_fraction_relative=.05,angle_error_deg=1.,contour_error_over_W=.1,contact_curvature_relative=.05)

def metrics(f,g,rule):
    v=grain_volumes(f,g);z=g['z'];r=g['r_c'];R=contour(f,g,.5)
    centroid=np.array([np.sum(f*p*r[None,:]*z[:,None])/np.sum(f*p*r[None,:]) for p in g['ownership']])
    neck=np.interp(g['gb'],z[np.isfinite(R)],R[np.isfinite(R)])
    exact=chain_radius(z,g['cmc_center'],g['cmc_cap'])
    valid=np.isfinite(R)&(R>3*g['config'].width)
    measured=measure(f,g,rule)
    expected=[[g['cmc_cap']['K_per_m'],g['cmc_center'].K],[g['cmc_center'].K,g['cmc_cap']['K_per_m']]]
    curvature_error=0.
    for side,rb,targets in zip(measured,neck,expected):
        K=np.array(side['meridional_curvatures'])+1/(rb*np.sqrt(1+np.array(side['slopes'])**2))
        curvature_error=max(curvature_error,float(np.max(abs(K/targets-1))))
    return dict(contact_curvature_relative=curvature_error,volumes_m3=v.tolist(),centroids_m=centroid.tolist(),neck_radii_m=neck.tolist(),
                dilute_fraction=dilute_material_fraction(f,g),mirror_error=float(np.max(abs(f-f[::-1]))),
                angles_deg=[x['psi_deg'] for x in measure(f,g,rule)],
                max_contour_error_over_W=float(np.max(abs(R[valid]-exact[valid]))/g['config'].width))

def check(now,initial,g):
    def relative(key):return float(np.max(abs(np.array(now[key])/np.array(initial[key])-1)))
    values=dict(grain_volume_relative=relative('volumes_m3'),
                centroid_over_Ro=float(np.max(abs(np.array(now['centroids_m'])-initial['centroids_m']))/g['config'].outer_radius),
                neck_relative=relative('neck_radii_m'),
                dilute_fraction_relative=abs(now['dilute_fraction']/initial['dilute_fraction']-1),
                angle_error_deg=max(abs(a-160) for a in now['angles_deg']),
                contour_error_over_W=now['max_contour_error_over_W'],contact_curvature_relative=now['contact_curvature_relative'])
    return values,{k:values[k]<=limit for k,limit in LIMITS.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['unequal','null','equal'],default='unequal');ap.add_argument('--steps',type=int,default=200);ap.add_argument('--ownership-cleanup',action='store_true');ap.add_argument('--canonical-study',action='store_true');args=ap.parse_args()
    center,cap=chemical_potential_null() if args.case=='null' else compatible_chain(.7 if args.case=='unequal' else 1.)
    g=map_to_pf(center,cap,4e-9,.5e-9);op=PhaseAOperator(g);f=g['f'].copy()
    rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
    label=args.case+('_ownership' if args.ownership_cleanup else '')+('_short' if args.canonical_study else '')+'_aligned'
    out=Path(f'runs/three_particle_cmc/{label}');out.mkdir(parents=True,exist_ok=True)
    initial=metrics(f,g,rule);history=[];snapshots=[f.copy()];steps=[0]
    ownership_records=[]
    if args.ownership_cleanup:
        from pf_sintering.fixed_geometry_profile_equilibration import equilibrate_ownership_variational_fixed_geometry
        setup={**g,'p':op,'Wc':op.Wc}
        updated=[]
        for k in [0,2]:
            state,record=equilibrate_ownership_variational_fixed_geometry((f,f*g['ownership'][k],f*(1-g['ownership'][k])),setup,maximum_iterations=40)
            phi=np.divide(state[1],f,out=g['ownership'][k].copy(),where=f>1e-14)
            updated.append(phi);ownership_records.append(record)
        g['ownership']=np.array([updated[0],1-updated[0]-updated[1],updated[1]])
        if g['ownership'].min() < -1e-12:raise RuntimeError('ownership overlap')
        g['eta']=g['ownership']*f[None,:,:]
        op=PhaseAOperator(g)
    np.savez_compressed(out/'mapped_cmc_seed.npz',f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],canonical=False)
    dt=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4
    faces=[int(np.argmin(abs(g['z']+g['dz']/2-b))) for b in g['gb']]
    start=time.perf_counter();failure=None
    for i in range(args.steps):
        flux_kernel(f,op.potential(f),g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
        # Initialization only: impermeable GB faces; all remaining flux is native.
        for j in faces:op.Jz[j]=0.
        div_and_update_kernel(f,op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],dt,op.out)
        f=op.out.copy()
        if not np.all(np.isfinite(f)) or f.min() < -1e-8 or f.max()>1+1e-8:
            failure='field bounds';break
        if (i+1)%200==0 or i+1==args.steps:
            m=metrics(f,g,rule);values,gates=check(m,initial,g)
            history.append(dict(step=i+1,model_cleanup_time=(i+1)*dt,metrics=m,changes=values,gates=gates))
            snapshots.append(f.copy());steps.append(i+1)
            print(json.dumps(dict(case=args.case,step=i+1,wall_s=time.perf_counter()-start,changes=values)),flush=True)
            if not all(gates.values()):failure='cleanup guard exceeded';break
    final=metrics(f,g,rule);values,gates=check(final,initial,g)
    # Morphology guards alone are insufficient to establish a canonical PF state.
    report=dict(status='SHORT_CLEANUP_GUARDS_PASS' if not failure and all(gates.values()) else 'CLEANUP_REJECTED',failure=failure,
                canonical=bool(args.canonical_study and not failure and all(gates.values())),ownership_cleanup=ownership_records,cmc=describe(center,cap,width=op.W),rule=rule,limits=LIMITS,initial=initial,final=final,changes=values,gates=gates,
                wall_s=time.perf_counter()-start,dt_model=dt,steps=i+1,history=history,
                sharp_target_volumes_m3=(4*np.pi/3*g['radii']**3).tolist(),width_m=op.W,spacing_m=g['dr'],dz_m=g['dz'])
    np.savez_compressed(out/'cleanup_candidate.npz',f=f,ownership=g['ownership'],canonical=False)
    np.savez_compressed(out/'cleanup_frames.npz',f=np.array(snapshots),steps=np.array(steps),dt_model=dt)
    (out/'cleanup_report.json').write_text(json.dumps(report,indent=2)+'\n')
    Path(f'docs/three_particle/cmc/{label}_cleanup_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.canonical_study and report['status']=='SHORT_CLEANUP_GUARDS_PASS':
        # Numerical initial checkpoint only; Phase-A scientific qualification is separate.
        metadata=dict(case=args.case,ratio=center.ratio,half_length_m=center.half_length,outer_radius_m=center.radius_scale,width_m=op.W,spacing_m=g['dr'],dz_m=g['dz'],dt_model=dt,physical_time=0.,model_time=0.,rule=rule,phase_a_qualified=False,events_enabled=False)
        np.savez_compressed(out/'canonical_initial.npz',f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],gb=g['gb'],metadata_json=json.dumps(metadata),canonical=True)
    print(report['status'],flush=True)
if __name__=='__main__':main()
