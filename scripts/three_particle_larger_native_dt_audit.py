"""Discarded-copy audit of a 2x native dt at fixed check cadence."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_buffered_event_probe import BufferedContactEvent
from three_particle_forced_event import ContactEvent,MANIFEST,TOL
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
from pf_sintering.three_particle_native_blocks import native_block
from pf_sintering.quasistatic_event_continuation import fast_manifold_increment,fast_manifold_converged

class LargerDtEvent(ContactEvent):
 def __init__(self,g,pair,factor):super().__init__(g,pair,max_fast_blocks=1024);self.factor=factor;self.steps_per_check=int(round(10/factor));assert abs(self.steps_per_check*factor-10)<1e-12;self.dt*=factor
 def relax(self,state,q):
  current=tuple(x.copy() for x in state);previous=self.fast_metrics(current,q);f=current[0];consecutive=0;g=self.g;op=self.op
  for block in range(1,self.max_fast_blocks+1):
   f,phi,density=native_block(f,g['ownership'],op.gb_density,*self.pair,self.dt,1.0937500000000001e-25,
    op.Wc,op.k_eta,op.W_f,op.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],op.W,op.physics.M_s,steps=self.steps_per_check)
   g['ownership']=phi;op.gb_density=density;current=(f,*(phi*f[None]));row=self.fast_metrics(current,q)
   if row['topology_stop']:raise RuntimeError('three-grain topology guard')
   inc=fast_manifold_increment(previous,row);consecutive=consecutive+1 if block>=3 and fast_manifold_converged(inc,TOL) else 0
   if consecutive>=3:return current,self.metrics(current,q),dict(converged=True,iterations=block*self.steps_per_check,blocks=block)
   previous=row
  return current,self.metrics(current,q),dict(converged=False,iterations=block*self.steps_per_check,blocks=block)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 state,restart,contact,label=load_event_checkpoint(a.source);pair=(0,1) if contact=='LEFT' else (1,2);rows=[];fields=[]
 for factor,klass in [(2,LargerDtEvent),(2.5,LargerDtEvent)]:
  c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=klass(g,pair,factor);start=time.perf_counter()
  result=event.run(state,target=.005,initial_step_over_b=.005,minimum_step_over_b=.005,maximum_step_over_b=.005);m=event.metrics(result[:4],result[5]['event_progress_over_b']);fields.append(np.array(result[:4]))
  rows.append(dict(dt_factor=factor,dt_model=event.dt,steps_per_check=event.steps_per_check,wall_s=time.perf_counter()-start,completed=result[4],clock_s=result[5]['event_time_model']*MANIFEST['seconds_per_model_time'],metrics=m,f_min=float(result[0].min()),f_max=float(result[0].max())));print('DONE',factor,flush=True)
 reference=rows[0]
 for candidate,field in zip(rows[1:],fields[1:]):
  candidate['field_linf_to_reference']=float(np.max(np.abs(field-fields[0])));candidate['stress_difference_Pa']=candidate['metrics'][contact+'_sigma_local_Pa']-reference['metrics'][contact+'_sigma_local_Pa'];candidate['relative_clock_difference']=(candidate['clock_s']/reference['clock_s']-1 if reference['clock_s'] else (0. if candidate['clock_s']==0 else float('inf')))
  candidate['passed']=bool(candidate['completed'] and reference['completed'] and candidate['f_min']>=-1e-8 and candidate['f_max']<=1+1e-8 and candidate['field_linf_to_reference']<2e-8 and abs(candidate['stress_difference_Pa'])<10 and abs(candidate['relative_clock_difference'])<1e-7)
 passed=rows[-1]['passed']
 report=dict(label='LIVE_DISCARDED_COPY_LARGER_NATIVE_DT_AUDIT',source=str(a.source),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),source_q_over_b=restart['cumulative_q_m']/MANIFEST['b_event_m'],fixed_physical_check_interval=True,unchanged_convergence_tolerances=True,rows=rows,passed=passed,production_state_modified=False)
 a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k!='metrics'} for r in rows]},indent=2))
if __name__=='__main__':main()
