"""Events-off current-field viability continuation; no null qualification gate."""
from pathlib import Path
import sys,json,time,argparse,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_implicit import ImplicitSurfaceDiffusion
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from three_particle_implicit_run import advance
D=Path('docs/three_particle/production_screen')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ratio',type=float,required=True);ap.add_argument('--outer-nm',type=float,required=True);ap.add_argument('--harmonic',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    source=Path(f'runs/three_particle_production_screen/ratio_{args.ratio:.2f}_Ro_{args.outer_nm:.3f}nm.npz')
    c,o=compatible_chain(args.ratio,args.outer_nm*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=ImplicitSurfaceDiffusion(op)
    if args.harmonic:
        from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
        it=HarmonicSurfaceDiffusion(op)
    with np.load(source) as d:
        np.testing.assert_array_equal(g['z'],d['z']);np.testing.assert_array_equal(g['ownership'],d['ownership']);f=d['f_initial'].copy()
    m=json.loads((D/'bicrystal_launch_manifest.json').read_text());rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
    dt=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4;h=100*dt;t=0.;H=dict(LEFT=0.,RIGHT=0.);v0=grain_volumes(f,g);rows=[];rejected={};start=time.perf_counter();last=start;status='running'
    label=('harmonic_' if args.harmonic else '')+f'continuation_{args.ratio:.2f}_{args.outer_nm:.3f}'
    out=source.parent/label;out.mkdir(exist_ok=args.resume)
    def row():
        return dict(time_s=t*op.physics.seconds_per_model_time,contacts=evaluate_contacts(f,op,m),
                    hazard=H.copy(),center_relative_change=float(grain_volumes(f,g)[1]/v0[1]-1),
                    mass_relative_error=float(grain_volumes(f,g).sum()/v0.sum()-1),mirror_error=float(np.max(abs(f-f[::-1]))),energy_J=op.energy(f),f_min=float(f.min()),f_max=float(f.max()))
    rows.append(row());prior_wall=0.
    if args.resume:
        saved=json.loads((D/(label+'.json')).read_text());prior_wall=saved['wall_s'];rows=saved['history'];H=rows[-1]['hazard'].copy();rejected=saved['rejections']
        with np.load(out/'checkpoint.npz') as d:f=d['f'].copy();t=float(d['t_model'])
        assert abs(rows[-1]['time_s']-t*op.physics.seconds_per_model_time)<1e-12
        h=2.
    next_snapshot=t*op.physics.seconds_per_model_time+.5
    def save():
        report=dict(status=status,ratio=args.ratio,outer_radius_nm=args.outer_nm,history=rows,rejections=rejected,
          wall_s=prior_wall+time.perf_counter()-start,events_enabled=False,stationary_null_is_gate=False,
          source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          probability_any_root=1-np.exp(-sum(H.values())/m['root_threshold_multiplier']),
          hazard_is_deterministic_quadrature_not_localized_stochastic_event=True)
        (D/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(out/'checkpoint.npz',f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],t_model=t,gb=g['gb'],events_enabled=False)
    # A mean threshold is a viability diagnostic, never an imposed root trigger.
    while t*op.physics.seconds_per_model_time<30:
        h=min(h,20. if args.harmonic and t*op.physics.seconds_per_model_time>1 else 2.,30/op.physics.seconds_per_model_time-t)
        try:
            trial,err=advance(f,h,it,rule)
            if err>1:raise RuntimeError('nonlinear error estimator')
        except (FloatingPointError,RuntimeError) as e:
            reason=str(e);rejected[reason]=rejected.get(reason,0)+1;h*=.2
            if h<dt/8:status='FIELD_OR_SOLVER_GUARD';break
            continue
        f=trial;t+=h
        current=row();previous=rows[-1]
        for name in H:H[name]+=.5*(previous['contacts'][name]['root_rate_per_s']+current['contacts'][name]['root_rate_per_s'])*(current['time_s']-previous['time_s'])
        current['hazard']=H.copy();rows.append(current)
        if args.harmonic and (len(rows)==2 or previous['time_s']<.4<=current['time_s']):
            np.savez_compressed(out/'post_transient.npz',f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],t_model=t,gb=g['gb'],events_enabled=False)
        if args.harmonic and current['time_s']>=next_snapshot:
            np.savez_compressed(out/f"state_{current['time_s']:.9f}s.npz",f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],t_model=t,gb=g['gb'],events_enabled=False)
            next_snapshot=current['time_s']+.5
        if topology_status(f,g)['stop']:status='TOPOLOGY_GUARD';break
        if abs(current['mass_relative_error'])>1e-11 or current['mirror_error']>1e-8:status='CONSERVATION_GUARD';break
        if not args.harmonic and sum(H.values())>=m['root_threshold_multiplier']:status='APPRECIABLE_HAZARD_REQUIRES_EVENT_VALIDATION';break
        if time.perf_counter()-start>(3600 if args.harmonic else 900):status='WALL_BUDGET';break
        h*=min(2.,max(1.05,.8/max(err,1e-12)**.5))
        if time.perf_counter()-last>30:save();print(status,t*op.physics.seconds_per_model_time,H,flush=True);last=time.perf_counter()
    if status=='running':status='TIME_LIMIT'
    save();print(status,t*op.physics.seconds_per_model_time,H,flush=True)
if __name__=='__main__':main()
