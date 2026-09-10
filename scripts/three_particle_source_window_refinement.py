"""Off-trajectory full 9 ms active-ownership window refinement, without RNG."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from three_particle_source_window import advance_source_window
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
source=Path('runs/three_particle_production_065/forced_left_minimum_increment/watch_q0.800000.npz');state,restart,_,_=load_event_checkpoint(source)
rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];rows=[];outputs=[];wall=time.perf_counter()
for n in [1,2,4]:
 c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g);before=event.metrics(state,0.);current=state;done=0.;status='COMPLETE'
 try:
  for _ in range(n):current=advance_source_window(current,.009/n,g,(0,1),rule);done+=.009/n
 except (RuntimeError,ValueError,FloatingPointError) as error:status='STOPPED: '+str(error)
 after=event.metrics(current,0.);outputs.append(np.array(current));rows.append(dict(partitions=n,status=status,completed_seconds=done,after=after,mass_relative_error=after['total_volume_m3']/before['total_volume_m3']-1,energy_relative_change=after['G_phasefield_J']/before['G_phasefield_J']-1,field_min=float(current[0].min()),field_max=float(current[0].max()),ownership_closure=float(np.max(abs(sum(current[1:])-current[0])))))
 print(n,status,done,after['LEFT_sigma_local_Pa'],flush=True)
for row,field in zip(rows,outputs):
 row['field_Linf_to_finest']=float(np.max(abs(field-outputs[-1])))
 row['sigma_difference_to_finest_Pa']=row['after']['LEFT_sigma_local_Pa']-rows[-1]['after']['LEFT_sigma_local_Pa']
report=dict(label='OFF_TRAJECTORY_NINE_MS_SOURCE_WINDOW_AUDIT',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_q=restart['cumulative_q_m']/MANIFEST['b_event_m'],not_a_completed_root_or_descendant=True,no_random_draws=True,rows=rows,wall_s=time.perf_counter()-wall)
Path('docs/three_particle/production_065/source_window_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed('runs/three_particle_production_065/source_window_refinement.npz',initial=np.array(state),one=outputs[0],two=outputs[1],four=outputs[2])
print([(r['partitions'],r['field_Linf_to_finest'],r['sigma_difference_to_finest_Pa']) for r in rows],flush=True)
