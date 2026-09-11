"""Events-OFF full-field continuation to center-volume-loss milestones."""
from pathlib import Path
import sys,json,time,os,hashlib,argparse,shutil
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_loading import decompose,plateau,PLATEAU,trial_invariant_reason
from three_particle_implicit_run import advance
OUT=Path('runs/three_particle_volume_loading_065');DOC=Path('docs/three_particle/volume_loading_065')
SOURCE=Path('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999/checkpoint.npz')
BASE=Path('runs/three_particle_production_screen/ratio_0.65_Ro_119.999nm.npz')

def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--source',type=Path,default=SOURCE);ap.add_argument('--history-source',type=Path);ap.add_argument('--full-jacobian',action='store_true');ap.add_argument('--reuse-preconditioner',action='store_true');ap.add_argument('--max-step-model',type=float,default=20.);args=ap.parse_args();OUT=args.out;source=args.source
 c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g)
 if not 0 < args.max_step_model <= 20. and not args.full_jacobian:raise ValueError('Unqualified frozen-mobility ceiling')
 if args.reuse_preconditioner and not args.full_jacobian:raise ValueError('Reuse option requires full Jacobian')
 audit_path=Path('docs/three_particle/volume_loading_065/full_jacobian_refinement_32.json')
 if args.full_jacobian:
  audit=json.loads(audit_path.read_text())
  if not audit['passed'] or not 0 < args.max_step_model <= audit['qualified_max_step_model']:raise ValueError('Unqualified full-Jacobian ceiling')
  from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
  solver=FullJacobianSurfaceDiffusion(op,reuse_preconditioner=args.reuse_preconditioner)
 else:solver=HarmonicSurfaceDiffusion(op)
 engine='full_native_flux_jacobian' if args.full_jacobian else 'frozen_mobility_jacobian'
 m=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text());rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
 with np.load(BASE) as d:v0=grain_volumes(d['f_initial'],g)
 with np.load(OUT/'checkpoint.npz' if args.resume else source) as d:
  assert not bool(d['events_enabled']);np.testing.assert_array_equal(g['z'],d['z']);np.testing.assert_array_equal(g['ownership'],d['ownership']);np.testing.assert_array_equal(g['gb'],d['gb']);f=d['f'].copy();t=float(d['t_model']);h=float(d['next_h']) if 'next_h' in d else 2.
 if args.full_jacobian and not args.resume:h=min(20.,args.max_step_model)
 OUT.mkdir(exist_ok=args.resume);DOC.mkdir(parents=True,exist_ok=True);milestones=[.01,.02,.03,.05,.075,.10];saved=[];rows=[];rejections={};status='RUNNING';start=time.perf_counter();last=start
 def observe():
  contacts=evaluate_contacts(f,op,m);vol=grain_volumes(f,g);actual=dict(g,gb=np.array([contacts[n]['z_TJ_m'] for n in ['LEFT','RIGHT']]))
  return dict(time_s=t*op.physics.seconds_per_model_time,center_loss_fraction=float(1-vol[1]/v0[1]),grain_volumes_m3=vol.tolist(),contacts=contacts,decomposition={n:decompose(v) for n,v in contacts.items()},center_span_over_W=float(np.diff(actual['gb'])[0]/op.W),topology=topology_status(f,actual),energy_J=op.energy(f),f_min=float(f.min()),f_max=float(f.max()),mass_relative_error=float(vol.sum()/v0.sum()-1),mirror_error=float(np.max(abs(f-f[::-1]))),two_contact_wait_s=m['root_threshold_multiplier']/sum(v['root_rate_per_s'] for v in contacts.values()))
 if args.resume:
  launch=json.loads((OUT/'launch.json').read_text());assert launch.get('solver','frozen_mobility_jacobian')==engine and launch.get('reuse_preconditioner',False)==args.reuse_preconditioner and launch['max_step_model']==args.max_step_model
  report=json.loads((OUT/'report.json').read_text());rows=report['history'];saved=report['saved_milestones'];rejections=report['rejections'];assert abs(rows[-1]['time_s']-t*op.physics.seconds_per_model_time)<1e-10
 else:
  rows=[observe()];(OUT/'launch.json').write_text(json.dumps(dict(label='EVENTS_OFF_VOLUME_CONTROLLED_LOADING',events_enabled=False,stochastic_thresholds_drawn=False,hazards_evaluated=False,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),trial_guards_checked_before_commit=True,history_source=str(args.history_source),history_source_sha256=hashlib.sha256(args.history_source.read_bytes()).hexdigest() if args.history_source else None,baseline=str(BASE),baseline_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),baseline_volumes_m3=v0.tolist(),milestones=milestones,milestone_tolerance_loss_fraction=1e-5,plateau=PLATEAU,solver=engine,reuse_preconditioner=args.reuse_preconditioner,solver_source_sha256=hashlib.sha256(Path('pf_sintering/three_particle_full_jacobian.py' if args.full_jacobian else 'pf_sintering/three_particle_bounded_mobility.py').read_bytes()).hexdigest(),qualification_audit=str(audit_path) if args.full_jacobian else None,qualification_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest() if args.full_jacobian else None,max_step_model=args.max_step_model,field_tolerance=2e-5,physical_manifest=m),indent=2)+'\n')
 if args.history_source and not args.resume:
  parent=json.loads(args.history_source.read_text());rows=[r for r in parent['history'] if r['time_s']<=t*op.physics.seconds_per_model_time+1e-12]
  assert abs(rows[-1]['time_s']-t*op.physics.seconds_per_model_time)<1e-12
  assert abs(rows[-1]['center_loss_fraction']-observe()['center_loss_fraction'])<1e-14
  for target in parent['saved_milestones']:
   path=args.history_source.parent/f'loss_{100*target:g}pct.npz'
   with np.load(path) as data:past=float(data['t_model'])<=t+1e-12
   if past:shutil.copyfile(path,OUT/path.name);saved.append(target)
 def checkpoint(path):
  temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],gb=g['gb'],t_model=t,next_h=h,events_enabled=False);os.replace(temp,path)
 def save():
  checkpoint(OUT/'checkpoint.npz');payload=json.dumps(dict(status=status,history=rows,saved_milestones=saved,rejections=rejections,plateau=plateau(rows),events_enabled=False,stochastic_thresholds_drawn=False,phase_b_enabled=False,wall_s_this_launch=time.perf_counter()-start),indent=2)+'\n';temp=OUT/'report.tmp.json';temp.write_text(payload);os.replace(temp,OUT/'report.json')
 save();native=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4
 while status=='RUNNING':
  x=rows[-1]['center_loss_fraction'];target=next((v for v in milestones if v not in saved),None)
  if target is None:status='TEN_PERCENT_MILESTONE';break
  h=min(h,args.max_step_model)
  try:
   trial,error=advance(f,h,solver,rule)
   if error>1:raise RuntimeError('nonlinear error estimator')
   trial_volumes=grain_volumes(trial,g);reason=trial_invariant_reason(trial,float(trial_volumes.sum()),float(v0.sum()))
   if reason:raise RuntimeError(reason)
   tx=1-trial_volumes[1]/v0[1]
   if tx>target+1e-5 and x<target-1e-5:
    h*=max(.05,min(.95,(target-x)/(tx-x)));continue
   if op.energy(trial)>rows[-1]['energy_J']*(1+1e-12):raise RuntimeError('energy increase')
  except (FloatingPointError,RuntimeError) as e:
   rejections[str(e)]=rejections.get(str(e),0)+1;h*=.2
   if h<native/8:status='NUMERICAL_GUARD';break
   continue
  f=trial;t+=h;rows.append(observe());h*=min(2.,max(1.05,.8/max(error,1e-12)**.5));now=rows[-1]
  if abs(now['center_loss_fraction']-target)<=1e-5 or now['center_loss_fraction']>=target:
   checkpoint(OUT/f'loss_{100*target:g}pct.npz');saved.append(target);print('MILESTONE',target,now['time_s'],now['contacts']['LEFT']['sigma_local_Pa']/1e6,flush=True)
  if now['topology']['stop']:status='TOPOLOGY_RESOLUTION_LIMIT'
  elif abs(now['mass_relative_error'])>1e-11 or now['mirror_error']>1e-8:status='CONSERVATION_SYMMETRY_GUARD'
  elif plateau(rows)['reached']:status='STRESS_PLATEAU'
  save()
  if time.perf_counter()-last>30:print(status,'loss_pct',100*now['center_loss_fraction'],'t_s',now['time_s'],'sigma_MPa',now['contacts']['LEFT']['sigma_local_Pa']/1e6,flush=True);last=time.perf_counter()
 save();print('FINAL',status,rows[-1]['center_loss_fraction'],flush=True)
if __name__=='__main__':main()
