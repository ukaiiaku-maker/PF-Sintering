"""Native PF with f/phi retained internally between identical diagnostic blocks."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,TOL
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update
from pf_sintering.three_particle_event import ownership_pair_step,update_ownership,load_event_checkpoint
from pf_sintering.quasistatic_event_continuation import fast_manifold_increment,fast_manifold_converged
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf

class RetainedOwnershipContactEvent(ContactEvent):
    def relax(self,state,q):
        current=tuple(x.copy() for x in state);previous=self.metrics(current,q);f=current[0];consecutive=0
        for block in range(1,self.max_fast_blocks+1):
            for _ in range(10):
                fn=bounded_mobility_update(f,self.op.potential(f),self.op,self.dt)
                if fn.min() < -1e-8 or fn.max()>1+1e-8:raise RuntimeError('fast field bounds')
                pn=ownership_pair_step(self.g['ownership'],fn,self.op,self.pair,self.dt,1.0937500000000001e-25)
                update_ownership(self.op,pn);f=fn
            current=(f,*(self.g['ownership']*f[None]));row=self.metrics(current,q)
            if row['topology_stop']:raise RuntimeError('three-grain topology guard')
            inc=fast_manifold_increment(previous,row);consecutive=consecutive+1 if block>=3 and fast_manifold_converged(inc,TOL) else 0
            if consecutive>=3:return current,row,dict(converged=True,iterations=block*10,blocks=block)
            previous=row
        return current,row,dict(converged=False,iterations=block*10,blocks=block)

def main():
    source=Path('runs/three_particle_production_065/forced_left_fine_increment/checkpoint.npz');state,restart,_,_=load_event_checkpoint(source)
    c,o=compatible_chain(.65,119.999e-9);outputs=[];rows=[]
    for cls in [ContactEvent,RetainedOwnershipContactEvent]:
        g=map_to_pf(c,o,4e-9,.5e-9);event=cls(g,max_fast_blocks=240);start=time.perf_counter();result=event.run(state,target=.0025)
        outputs.append(result);rows.append(dict(method=cls.__name__,wall_s=time.perf_counter()-start,completed=result[4],stop_reason=result[5]['stop_reason'],after=event.metrics(result[:4],result[5]['event_progress_over_b'])))
        print(cls.__name__,rows[-1]['wall_s'],result[4],flush=True)
    native,retained=outputs
    report=dict(label='OFF_TRAJECTORY_NATIVE_STORAGE_PARITY',source_q=restart['cumulative_q_m']/.25e-9,
        fields_Linf=float(np.max(abs(np.array(native[:4])-np.array(retained[:4])))),runs=rows,
        local_stress_difference_Pa=rows[1]['after']['LEFT_sigma_local_Pa']-rows[0]['after']['LEFT_sigma_local_Pa'],
        relative_clock_difference=retained[5]['event_time_model']/native[5]['event_time_model']-1,
        native_block_counts=[r[5]['packets'][-1]['fast_relax_blocks'] if r[5]['packets'] else None for r in outputs],production_promoted=False)
    report['native_overlap_pass']=bool(native[4] and retained[4] and report['fields_Linf']<1e-12 and abs(report['local_stress_difference_Pa'])<1e-3 and abs(report['relative_clock_difference'])<1e-10)
    Path('docs/three_particle/production_065/fast_storage_probe.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
    np.savez_compressed('runs/three_particle_production_065/fast_storage_probe.npz',initial=np.array(state),native=np.array(native[:4]),retained=np.array(retained[:4]),q=report['source_q'])
    print({k:v for k,v in report.items() if k!='runs'},flush=True)
if __name__=='__main__':main()
