"""Off-trajectory accelerated-fast-flow prototype, always checked against native PF."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from three_particle_implicit_run import advance
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from pf_sintering.three_particle_event import ownership_pair_step,load_event_checkpoint
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf

class ImplicitFastProbe(ContactEvent):
    def relax(self,state,q):
        current=tuple(x.copy() for x in state);solver=HarmonicSurfaceDiffusion(self.op)
        rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
        native_equivalent=0.;macro_steps=0;h=400*self.dt;limit=self.max_fast_blocks*10
        while native_equivalent+50<limit:
            self.bind(current);h=min(h,(limit-native_equivalent-50)*self.dt)
            try:
                fn,error=advance(current[0],h,solver,rule,field_tol=2e-7)
                if error>1:raise RuntimeError('fast implicit embedded error')
            except (RuntimeError,FloatingPointError) as error:
                h*=.5
                if h<self.dt:raise RuntimeError('fast implicit probe timestep floor: '+str(error))
                continue
            phi=ownership_pair_step(self.g['ownership'],fn,self.op,self.pair,h,1.0937500000000001e-25)
            current=(fn,*(phi*fn[None]));native_equivalent+=h/self.dt;macro_steps+=1
            # Acceptance always uses the ORIGINAL three consecutive passing
            # native ten-step blocks, not a rescaled macrostep tolerance.
            maximum=self.max_fast_blocks;self.max_fast_blocks=5
            try:current,row,info=super().relax(current,q)
            finally:self.max_fast_blocks=maximum
            native_equivalent+=info['iterations']
            if info['converged']:
                return current,row,dict(converged=True,iterations=int(native_equivalent),blocks=int(np.ceil(native_equivalent/10)),implicit_macros=macro_steps)
            h=min(400*self.dt,h*1.5)
        return current,self.metrics(current,q),dict(converged=False,iterations=int(native_equivalent),blocks=int(np.ceil(native_equivalent/10)))

def main():
    source=Path('runs/three_particle_production_065/forced_left_fine_increment/checkpoint.npz');state,restart,_,_=load_event_checkpoint(source)
    c,o=compatible_chain(.65,119.999e-9);outputs=[];rows=[]
    for cls in [ContactEvent,ImplicitFastProbe]:
        g=map_to_pf(c,o,4e-9,.5e-9);event=cls(g,max_fast_blocks=240);start=time.perf_counter();result=event.run(state,target=.005)
        outputs.append(result);rows.append(dict(method=cls.__name__,wall_s=time.perf_counter()-start,completed=result[4],stop_reason=result[5]['stop_reason'],after=event.metrics(result[:4],result[5]['event_progress_over_b'])))
        print(cls.__name__,rows[-1]['wall_s'],result[4],flush=True)
    native,implicit=outputs
    report=dict(label='OFF_TRAJECTORY_FAST_SOLVER_PROTOTYPE',source_q=restart['cumulative_q_m']/.25e-9,
        fields_Linf=float(np.max(abs(np.array(native[:4])-np.array(implicit[:4])))),runs=rows,
        local_stress_difference_Pa=rows[1]['after']['LEFT_sigma_local_Pa']-rows[0]['after']['LEFT_sigma_local_Pa'],
        relative_clock_difference=implicit[5]['event_time_model']/native[5]['event_time_model']-1,
        production_promoted=False)
    report['native_overlap_pass']=bool(native[4] and implicit[4] and report['fields_Linf']<2e-5 and abs(report['local_stress_difference_Pa'])<5000 and abs(report['relative_clock_difference'])<.001)
    Path('docs/three_particle/production_065/fast_implicit_probe.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
    np.savez_compressed('runs/three_particle_production_065/fast_implicit_probe.npz',initial=np.array(state),native=np.array(native[:4]),implicit=np.array(implicit[:4]),q=report['source_q'])
    print({k:v for k,v in report.items() if k!='runs'},flush=True)
if __name__=='__main__':main()
