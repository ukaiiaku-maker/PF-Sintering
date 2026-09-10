"""Preconditioner-only performance probe; original operator and residual tests."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent
from three_particle_source_window import advance_source_window
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
source=Path('runs/three_particle_production_065/forced_left_minimum_increment/watch_q0.800000.npz');state,restart,_,_=load_event_checkpoint(source)
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g);before=event.metrics(state,0.)
rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];wall=time.perf_counter()
current=advance_source_window(state,.009,g,(0,1),rule,reuse_small_step_preconditioner=True);after=event.metrics(current,0.)
report=dict(label='OFF_TRAJECTORY_SMALL_STEP_PRECONDITIONER_REUSE',source=str(source),duration_s=.009,wall_s=time.perf_counter()-wall,after=after,mass_relative_error=after['total_volume_m3']/before['total_volume_m3']-1,field_min=float(current[0].min()),field_max=float(current[0].max()),ownership_closure=float(np.max(abs(sum(current[1:])-current[0]))),linear_operator_changed=False,linear_residual_tolerance_changed=False,qualified=False)
Path('docs/three_particle/production_065/source_window_cache_probe.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed('runs/three_particle_production_065/source_window_cache_probe.npz',initial=np.array(state),cached=np.array(current))
print({k:v for k,v in report.items() if k!='after'},flush=True)
