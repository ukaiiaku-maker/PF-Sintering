"""Recheck the discarded implicit-fast prototype with preconditioner reuse."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import three_particle_fast_implicit_probe as probe
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
class NativeStartupProbe(probe.ImplicitFastProbe):
    def relax(self,state,q):
        from three_particle_forced_event import ContactEvent
        budget=self.max_fast_blocks
        self.max_fast_blocks=5
        try:current,row,info=ContactEvent.relax(self,state,q)
        finally:self.max_fast_blocks=budget
        if info['converged']:return current,row,info
        self.max_fast_blocks=budget-5
        try:current,row,result=probe.ImplicitFastProbe.relax(self,current,q)
        finally:self.max_fast_blocks=budget
        result['iterations']+=info['iterations'];result['blocks']+=info['blocks']
        return current,row,result

original=probe.HarmonicSurfaceDiffusion
probe.HarmonicSurfaceDiffusion=lambda op:original(op,reuse_small_step_preconditioner=True)
source=Path('runs/three_particle_production_065/forced_left_minimum_increment/watch_q0.800000.npz');state,restart,_,_=load_event_checkpoint(source)
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=NativeStartupProbe(g,max_fast_blocks=240);wall=time.perf_counter();result=event.run(state,target=.0025)
with np.load('runs/three_particle_production_065/event_increment_refinement.npz') as d:native=d['one'].copy()
reference=json.loads(Path('docs/three_particle/production_065/event_increment_refinement.json').read_text())['rows'][0]
metrics=event.metrics(result[:4],result[5]['event_progress_over_b'])
r=dict(label='OFF_TRAJECTORY_NATIVE_STARTUP_IMPLICIT_FAST_PROBE',source=str(source),completed=result[4],stop_reason=result[5]['stop_reason'],stop_detail=result[5].get('stop_detail'),wall_s=time.perf_counter()-wall,fields_Linf_to_native=float(np.max(abs(np.array(result[:4])-native))),sigma_difference_to_native_Pa=metrics['LEFT_sigma_local_Pa']-reference['metrics']['LEFT_sigma_local_Pa'],event_info=result[5],production_promoted=False)
r['event_info']={k:v for k,v in r['event_info'].items() if k!='event_restart'}
Path('docs/three_particle/production_065/fast_implicit_startup_probe.json').write_text(json.dumps(r,indent=2,default=float)+'\n')
np.savez_compressed('runs/three_particle_production_065/fast_implicit_startup_probe.npz',initial=np.array(state),native=native,implicit=np.array(result[:4]));print({k:v for k,v in r.items() if k!='event_info'},flush=True)
