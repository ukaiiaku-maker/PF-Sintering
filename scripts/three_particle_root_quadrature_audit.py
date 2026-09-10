"""Deterministic current-field root-hazard refinement; no RNG or source events."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from three_particle_implicit_run import advance
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_contacts import evaluate_contacts
D=Path('docs/three_particle/production_065');report_path=D/'root_quadrature_refinement.json'
assert not report_path.exists()
source=Path('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999/post_transient.npz')
with np.load(source) as d:initial=d['f'].copy();ownership=d['ownership'].copy();source_t=float(d['t_model'])*MANIFEST['seconds_per_model_time']
rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];rows=[];outputs=[];wall=time.perf_counter()
for macro in [.2,.1,.05]:
 c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);g['ownership']=ownership.copy();e=ContactEvent(g);it=HarmonicSurfaceDiffusion(e.op)
 f=initial.copy();t=0.;hazard={k:0. for k in ['LEFT','RIGHT']};history=[]
 def rates(field):return {k:v['root_rate_per_s'] for k,v in evaluate_contacts(field,e.op,MANIFEST).items()}
 r0=rates(f)
 while t<1.-1e-13:
  interval=min(macro,1.-t);elapsed=0.;h=min(interval,.1)
  while elapsed<interval-1e-14:
   h=min(h,interval-elapsed)
   try:
    trial,error=advance(f,h/MANIFEST['seconds_per_model_time'],it,rule)
    if error>1:raise RuntimeError('embedded field error')
   except (RuntimeError,FloatingPointError):
    h*=.2
    if h<1e-10:raise RuntimeError('root audit timestep floor')
    continue
   f=trial;elapsed+=h;h*=min(2.,max(.5,.8/max(error,1e-12)**.5))
  r1=rates(f)
  for k in hazard:hazard[k]+=.5*(r0[k]+r1[k])*interval
  t+=interval;history.append(dict(time_s=t,hazard=hazard.copy(),rates=r1));r0=r1
 outputs.append(f.copy());rows.append(dict(macro_step_s=macro,hazard=hazard,history=history));print(macro,hazard,flush=True)
for row,field in zip(rows,outputs):
 row['field_Linf_to_finest']=float(np.max(abs(field-outputs[-1])))
 row['max_relative_hazard_difference_to_finest']=max(abs(row['hazard'][k]/rows[-1]['hazard'][k]-1) for k in row['hazard'])
report=dict(label='OFF_TRAJECTORY_ROOT_QUADRATURE_REFINEMENT',source=str(source),source_time_s=source_t,duration_s=1.,random_thresholds_drawn=False,declared_hazard_relative_tolerance=1e-4,declared_field_Linf_tolerance=2e-5,rows=rows,wall_s=time.perf_counter()-wall)
report['coarse_quadrature_pass']=bool(rows[0]['max_relative_hazard_difference_to_finest']<1e-4 and rows[0]['field_Linf_to_finest']<2e-5)
report_path.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed('runs/three_particle_production_065/root_quadrature_refinement.npz',initial=initial,coarse=outputs[0],medium=outputs[1],fine=outputs[2]);print(report['coarse_quadrature_pass'],flush=True)
