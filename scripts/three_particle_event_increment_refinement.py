"""Discarded-copy event increment refinement; no accepted trajectory mutation."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
source=Path('runs/three_particle_production_065/forced_left_minimum_increment/watch_q0.800000.npz')
state,restart,_,_=load_event_checkpoint(source);results=[];fields=[];wall=time.perf_counter()
for n in [1,2,4]:
 c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g,max_fast_blocks=240);dq=.0025/n
 result=event.run(state,target=.0025,initial_step_over_b=dq,minimum_step_over_b=dq,maximum_step_over_b=dq)
 row=event.metrics(result[:4],result[5]['event_progress_over_b']);fields.append(np.array(result[:4]))
 results.append(dict(subincrements=n,increment_over_b=dq,completed=result[4],stop_reason=result[5]['stop_reason'],clock_s=result[5]['event_time_model']*MANIFEST['seconds_per_model_time'],metrics=row))
 print('subincrements',n,'completed',result[4],'sigma',row['LEFT_sigma_local_Pa'],flush=True)
for row,field in zip(results,fields):
 row['field_Linf_to_finest']=float(np.max(abs(field-fields[-1])))
 row['sigma_difference_to_finest_Pa']=row['metrics']['LEFT_sigma_local_Pa']-results[-1]['metrics']['LEFT_sigma_local_Pa']
 row['relative_clock_difference_to_finest']=row['clock_s']/results[-1]['clock_s']-1
report=dict(label='OFF_TRAJECTORY_EVENT_INCREMENT_REFINEMENT',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_q=restart['cumulative_q_m']/MANIFEST['b_event_m'],total_advanced_over_b=.0025,unchanged_fast_tolerances=True,rows=results,wall_s=time.perf_counter()-wall)
Path('docs/three_particle/production_065/event_increment_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed('runs/three_particle_production_065/event_increment_refinement.npz',initial=np.array(state),one=fields[0],two=fields[1],four=fields[2])
print([(r['subincrements'],r['field_Linf_to_finest'],r['sigma_difference_to_finest_Pa'],r['relative_clock_difference_to_finest']) for r in results],flush=True)
