"""Discarded-copy native continuation batching audit; no live trajectory changes."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_buffered_event_probe import BufferedContactEvent
from three_particle_forced_event import MANIFEST
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint

source=Path('runs/three_particle_production_065/forced_left_minimum_increment/watch_q0.800000.npz')
state,restart,_,_=load_event_checkpoint(source)
limits=dict(field_Linf=2e-5,local_stress_Pa=30000.,relative_clock=1e-3)
rows=[];fields=[]
for dq in [.01,.005,.0025]:
 c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9)
 event=BufferedContactEvent(g,max_fast_blocks=1024);start=time.perf_counter()
 result=event.run(state,target=.01,initial_step_over_b=dq,minimum_step_over_b=dq,maximum_step_over_b=dq)
 field=np.array(result[:4]);fields.append(field);metrics=event.metrics(result[:4],result[5]['event_progress_over_b'])
 rows.append(dict(increment_over_b=dq,completed=result[4],stop_reason=result[5]['stop_reason'],stop_detail=result[5].get('stop_detail'),wall_s=time.perf_counter()-start,
  clock_s=result[5]['event_time_model']*MANIFEST['seconds_per_model_time'],sigma_Pa={k:metrics[k+'_sigma_local_Pa'] for k in ['LEFT','RIGHT']},
  blocks=[p['fast_relax_blocks'] for p in result[5]['packets']]))
 print(rows[-1],flush=True)
for row,field in zip(rows,fields):
 if not row['completed']:
  row.update(field_Linf_to_reference=None,max_local_stress_difference_Pa=None,relative_clock_difference=None,overlap_pass=False)
  continue
 row['field_Linf_to_reference']=float(np.max(abs(field-fields[-1])))
 row['max_local_stress_difference_Pa']=max(abs(row['sigma_Pa'][k]-rows[-1]['sigma_Pa'][k]) for k in ['LEFT','RIGHT'])
 row['relative_clock_difference']=row['clock_s']/rows[-1]['clock_s']-1
 row['overlap_pass']=bool(row['completed'] and rows[-1]['completed'] and row['field_Linf_to_reference']<=limits['field_Linf'] and row['max_local_stress_difference_Pa']<=limits['local_stress_Pa'] and abs(row['relative_clock_difference'])<=limits['relative_clock'])
report=dict(label='OFF_TRAJECTORY_NATIVE_BATCH_AUDIT',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),declared_limits=limits,
 max_fast_blocks=1024,unchanged_fast_convergence_tolerances=True,production_promoted=False,rows=rows)
Path('docs/three_particle/production_065/event_batch_audit.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed('runs/three_particle_production_065/event_batch_audit.npz',initial=np.array(state),coarse=fields[0],medium=fields[1],reference=fields[2])
print(report,flush=True)
