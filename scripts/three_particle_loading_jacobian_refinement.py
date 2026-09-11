"""Predeclared equal-duration full-Jacobian temporal refinement, discarded only."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes
from three_particle_implicit_run import advance
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--total',type=float,default=64.);args=ap.parse_args();total=args.total;stem=f'full_jacobian_refinement_{total:g}'
D=Path('docs/three_particle/volume_loading_065');source=D/'late_stiffness_state.npz';rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];m=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text())
limits=dict(field_Linf=2e-5,local_stress_Pa=5000.,volume_relative=1e-5,angle_rad=float(np.radians(.03)),energy_relative=5e-6,mirror_error=1e-8,mass_error=1e-11)
(D/f'{stem}_predeclared.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),total_model_time=total,parts=[1,2,4],limits=limits),indent=2)+'\n')
with np.load(source) as d:initial=d['f'].copy()
rows=[];fields=[]
for parts in [1,2,4]:
 c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=FullJacobianSurfaceDiffusion(op);f=initial.copy();v0=grain_volumes(f,g);errors=[];energies=[op.energy(f)];wall=time.perf_counter();status='COMPUTED_DISCARDED'
 try:
  for _ in range(parts):
   f,error=advance(f,total/parts,it,rule);errors.append(error);energies.append(op.energy(f))
 except (FloatingPointError,RuntimeError) as e:status='FAILED: '+str(e)
 fields.append(f);rows.append(dict(parts=parts,status=status,embedded_errors=errors,energy_monotone=bool(np.all(np.diff(energies)<=np.array(energies[:-1])*1e-12)),energy_J=energies[-1],contacts=evaluate_contacts(f,op,m),volumes=grain_volumes(f,g).tolist(),mirror_error=float(np.max(abs(f-f[::-1]))),mass_error=float(grain_volumes(f,g).sum()/v0.sum()-1),f_min=float(f.min()),f_max=float(f.max()),wall_s=time.perf_counter()-wall));print(parts,status,errors,rows[-1]['mirror_error'],flush=True)
diffs=[]
for i in [0,1]:
 a,b=rows[i],rows[-1];diffs.append(dict(field_Linf=float(np.max(abs(fields[i]-fields[-1]))),local_stress_Pa=max(abs(a['contacts'][n]['sigma_local_Pa']-b['contacts'][n]['sigma_local_Pa']) for n in ['LEFT','RIGHT']),volume_relative=float(np.max(abs(np.array(a['volumes'])/b['volumes']-1))),angle_rad=max(abs(a['contacts'][n][f'theta_{s}_rad']-b['contacts'][n][f'theta_{s}_rad']) for n in ['LEFT','RIGHT'] for s in ['negative','positive']),energy_relative=abs(a['energy_J']/b['energy_J']-1)))
passed=all(r['status']=='COMPUTED_DISCARDED' and len(r['embedded_errors'])==r['parts'] and max(r['embedded_errors'])<=1 and r['energy_monotone'] and r['mirror_error']<=limits['mirror_error'] and abs(r['mass_error'])<=limits['mass_error'] for r in rows) and all(diffs[0][k]<=limits[k] for k in diffs[0]) and diffs[1]['field_Linf']<diffs[0]['field_Linf']
report=dict(passed=passed,limits=limits,differences_to_finest=diffs,rows=rows,discarded_copies_only=True,qualified_max_step_model=total if passed else None)
(D/f'{stem}.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(D/f'{stem}.npz',fields=np.array(fields));print('PASSED',passed,diffs,flush=True)
