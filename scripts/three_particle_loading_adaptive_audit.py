"""Discarded-copy check of a larger implicit macrostep ceiling; unchanged error tests."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from pf_sintering.three_particle_loading import trial_invariant_reason
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes
from three_particle_implicit_run import advance
source=Path('docs/three_particle/volume_loading_065/adaptive_full_jacobian_source.npz');D=Path('docs/three_particle/volume_loading_065');m=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text());rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
limits=dict(field_Linf=2e-5,local_stress_difference_Pa=5000.,volume_relative_difference=1e-5,angle_difference_rad=float(np.radians(.03)),energy_relative_difference=5e-6)
(D/'adaptive_full_jacobian_predeclared.json').write_text(json.dumps(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),duration_s=3.,ceilings_model=[32.,128.],limits=limits,unchanged_embedded_error_tests=True,discarded_copies_only=True),indent=2)+'\n')
fields=[];rows=[]
for cap in [32.,128.]:
 c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=FullJacobianSurfaceDiffusion(op,reuse_preconditioner=True)
 with np.load(source) as d:f=d['f'].copy();np.testing.assert_array_equal(g['z'],d['z'])
 v0=grain_volumes(f,g);t=0.;h=20.;end=3/op.physics.seconds_per_model_time;accepted=0;rejected=0;wall=time.perf_counter()
 while t<end-1e-12:
  h=min(h,cap,end-t)
  try:
   trial,error=advance(f,h,it,rule)
   if error>1:raise RuntimeError('embedded error')
   reason=trial_invariant_reason(trial,float(grain_volumes(trial,g).sum()),float(v0.sum()))
   if reason:raise RuntimeError(reason)
   if op.energy(trial)>op.energy(f)*(1+1e-12):raise RuntimeError('energy increase')
  except (FloatingPointError,RuntimeError):
   rejected+=1;h*=.2
   if h<1e-7:
    (D/'adaptive_full_jacobian_audit.json').write_text(json.dumps(dict(passed=False,status='ABORTED_BASELINE_GUARD',ceiling_model=cap,last_rejection='invariant/error rejection; inspect trace',accepted=accepted,rejected=rejected,advanced_s=t*op.physics.seconds_per_model_time,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),no_equal_duration_comparison=True,not_an_accepted_trajectory=True),indent=2)+'\n')
    raise RuntimeError('audit numerical floor; no promotion')
   continue
  f=trial;t+=h;accepted+=1;h*=min(2.,max(1.05,.8/max(error,1e-12)**.5))
 fields.append(f);rows.append(dict(ceiling_model=cap,accepted=accepted,rejected=rejected,wall_s=time.perf_counter()-wall,contacts=evaluate_contacts(f,op,m),volumes=grain_volumes(f,g).tolist(),energy_J=op.energy(f),f_min=float(f.min()),f_max=float(f.max()),mass_error=float(grain_volumes(f,g).sum()/v0.sum()-1)));print('cap finished',cap,accepted,rejected,flush=True)
a,b=rows;diff=dict(field_Linf=float(np.max(abs(fields[0]-fields[1]))),local_stress_difference_Pa=max(abs(a['contacts'][n]['sigma_local_Pa']-b['contacts'][n]['sigma_local_Pa']) for n in ['LEFT','RIGHT']),volume_relative_difference=float(np.max(abs(np.array(a['volumes'])/b['volumes']-1))),angle_difference_rad=max(abs(a['contacts'][n][f'theta_{s}_rad']-b['contacts'][n][f'theta_{s}_rad']) for n in ['LEFT','RIGHT'] for s in ['negative','positive']),energy_relative_difference=abs(a['energy_J']/b['energy_J']-1))
report=dict(passed=all(diff[k]<=limits[k] for k in limits),differences=diff,limits=limits,rows=rows,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),not_an_accepted_trajectory=True)
(D/'adaptive_full_jacobian_audit.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(D/'adaptive_full_jacobian_audit.npz',fields=np.array(fields));print(report['passed'],diff,flush=True)
