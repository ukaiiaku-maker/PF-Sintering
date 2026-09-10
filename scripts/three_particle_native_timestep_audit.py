"""Local native timestep refinement at fixed physical convergence-check cadence."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,TOL,MANIFEST
from pf_sintering.three_particle_native_blocks import native_block
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
from pf_sintering.quasistatic_event_continuation import fast_manifold_increment,fast_manifold_converged

class RefinedEvent(ContactEvent):
 def __init__(self,g,factor):
  super().__init__(g,max_fast_blocks=1024);self.dt/=factor;self.block_steps=10*factor
 def relax(self,state,q):
  current=tuple(x.copy() for x in state);previous=self.metrics(current,q);f=current[0];consecutive=0;g=self.g;op=self.op
  for block in range(1,self.max_fast_blocks+1):
   f,phi,density=native_block(f,g['ownership'],op.gb_density,*self.pair,self.dt,1.0937500000000001e-25,
    op.Wc,op.k_eta,op.W_f,op.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],op.W,op.physics.M_s,steps=self.block_steps)
   g['ownership']=phi;op.gb_density=density;current=(f,*(phi*f[None]));row=self.metrics(current,q)
   if row['topology_stop']:raise RuntimeError('three-grain topology guard')
   increment=fast_manifold_increment(previous,row);consecutive=consecutive+1 if block>=3 and fast_manifold_converged(increment,TOL) else 0
   if consecutive>=3:return current,row,dict(converged=True,iterations=block*self.block_steps,blocks=block)
   previous=row
  return current,row,dict(converged=False,iterations=block*self.block_steps,blocks=block)

source=Path('runs/three_particle_production_065/forced_descendant_family/event_2_q0.250000.npz')
state,restart,_,_=load_event_checkpoint(source);rows=[];fields=[]
for factor in [1,2,4]:
 c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=RefinedEvent(g,factor);start=time.perf_counter()
 result=event.run(state,target=.0025,initial_step_over_b=.0025,maximum_step_over_b=.0025);m=event.metrics(result[:4],result[5]['event_progress_over_b']);fields.append(np.array(result[:4]))
 rows.append(dict(refinement_factor=factor,dt_model=event.dt,steps_per_block=event.block_steps,check_interval_model=event.dt*event.block_steps,
  completed=result[4],stop_reason=result[5]['stop_reason'],stop_detail=result[5].get('stop_detail'),wall_s=time.perf_counter()-start,
  blocks=[p['fast_relax_blocks'] for p in result[5]['packets']],clock_s=result[5]['event_time_model']*MANIFEST['seconds_per_model_time'],
  sigma_Pa={k:m[k+'_sigma_local_Pa'] for k in ['LEFT','RIGHT']},field_min=float(result[0].min()),field_max=float(result[0].max())))
 print(rows[-1],flush=True)
for row,field in zip(rows,fields):
 row['field_Linf_to_finest']=float(np.max(abs(field-fields[-1]))) if row['completed'] and rows[-1]['completed'] else None
 row['max_local_stress_difference_Pa']=max(abs(row['sigma_Pa'][k]-rows[-1]['sigma_Pa'][k]) for k in ['LEFT','RIGHT'])
 row['relative_clock_difference']=row['clock_s']/rows[-1]['clock_s']-1 if rows[-1]['clock_s'] else None
report=dict(label='OFF_TRAJECTORY_NATIVE_TIMESTEP_REFINEMENT',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 total_advanced_over_b=.0025,fixed_physical_convergence_check_cadence=True,unchanged_convergence_tolerances=True,rows=rows,
 note='Local relaxation increment only; this does not establish refinement of the previously accumulated morphology or a complete event.',production_changed=False)
Path('docs/three_particle/production_065/native_timestep_audit.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed('runs/three_particle_production_065/native_timestep_audit.npz',initial=np.array(state),coarse=fields[0],medium=fields[1],fine=fields[2]);print(report,flush=True)
