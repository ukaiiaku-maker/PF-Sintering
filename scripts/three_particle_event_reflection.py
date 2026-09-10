"""Actual-field LEFT/RIGHT reflection of one identical accepted event increment."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
source=Path('runs/three_particle_production_065/forced_left_fine_increment/checkpoint.npz');state,restart,_,_=load_event_checkpoint(source)
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);gm=map_to_pf(c,o,4e-9,.5e-9)
mirrored=(state[0][::-1].copy(),state[3][::-1].copy(),state[2][::-1].copy(),state[1][::-1].copy());start=time.perf_counter()
left=ContactEvent(g,max_fast_blocks=240).run(state,target=.0025)
right=ContactEvent(gm,pair=(1,2),max_fast_blocks=240).run(mirrored,target=.0025)
differences=[float(np.max(abs(left[a]-right[b][::-1]))) for a,b in [(0,0),(1,3),(2,2),(3,1)]]
result=dict(label='OFF_TRAJECTORY_REFLECTION_REGRESSION',source=str(source),source_q=restart['cumulative_q_m']/.25e-9,
    increment_over_b=.0025,left_completed=left[4],right_completed=right[4],field_Linf_by_component=differences,
    relative_clock_difference=left[5]['event_time_model']/right[5]['event_time_model']-1,wall_s=time.perf_counter()-start)
result['passed']=bool(left[4] and right[4] and max(differences)<1e-10 and abs(result['relative_clock_difference'])<1e-9)
Path('docs/three_particle/production_065/event_reflection.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
