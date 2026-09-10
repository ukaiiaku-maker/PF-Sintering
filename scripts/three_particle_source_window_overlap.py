"""Native versus accelerated active-source PF overlap on a saved event state."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from three_particle_source_window import advance_source_window
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint,ownership_pair_step
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update
source=Path('runs/three_particle_production_065/forced_left_extended_fast_budget/checkpoint.npz')
state,restart,_,_=load_event_checkpoint(source);c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);e=ContactEvent(g)
rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];start=time.perf_counter();native=tuple(x.copy() for x in state)
for _ in range(100):
    e.bind(native);fn=bounded_mobility_update(native[0],e.op.potential(native[0]).copy(),e.op,e.dt)
    phi=ownership_pair_step(g['ownership'],fn,e.op,(0,1),e.dt,1.0937500000000001e-25);native=(fn,*(phi*fn[None]))
implicit=advance_source_window(state,100*e.dt*MANIFEST['seconds_per_model_time'],g,(0,1),rule)
a=e.metrics(native,0.);b=e.metrics(implicit,0.);difference=float(np.max(abs(np.array(native)-np.array(implicit))))
result=dict(source=str(source),q=restart['cumulative_q_m']/.25e-9,native_steps=100,duration_s=100*e.dt*MANIFEST['seconds_per_model_time'],
    fields_Linf=difference,LEFT_stress_difference_Pa=b['LEFT_sigma_local_Pa']-a['LEFT_sigma_local_Pa'],
    partition_closure=float(np.max(abs(sum(implicit[1:])-implicit[0]))),
    relative_mass_difference=b['total_volume_m3']/a['total_volume_m3']-1,wall_s=time.perf_counter()-start)
result['passed']=difference<2e-5 and abs(result['LEFT_stress_difference_Pa'])<5e4 and result['partition_closure']<5e-15 and abs(result['relative_mass_difference'])<1e-11
Path('docs/three_particle/production_065/source_window_overlap.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed('runs/three_particle_production_065/source_window_overlap.npz',initial=np.array(state),native=np.array(native),implicit=np.array(implicit),q=restart['cumulative_q_m']/.25e-9)
print(result,flush=True)
