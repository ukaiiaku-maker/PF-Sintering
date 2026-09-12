"""Exact native stencils with private reusable arrays between diagnostic blocks."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,TOL,MANIFEST
from pf_sintering.three_particle_native_blocks import native_block,native_block_reuse
from pf_sintering.three_particle_event import load_event_checkpoint
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.quasistatic_event_continuation import fast_manifold_increment,fast_manifold_converged

class AllocatingBufferedContactEvent(ContactEvent):
    def relax(self,state,q):
        current=tuple(x.copy() for x in state);previous=self.metrics(current,q);f=current[0];consecutive=0;g=self.g;op=self.op
        for block in range(1,self.max_fast_blocks+1):
            f,phi,density=native_block(f,g['ownership'],op.gb_density,*self.pair,self.dt,1.0937500000000001e-25,
                op.Wc,op.k_eta,op.W_f,op.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],op.W,op.physics.M_s)
            g['ownership']=phi;op.gb_density=density;current=(f,*(phi*f[None]));row=self.metrics(current,q)
            if row['topology_stop']:raise RuntimeError('three-grain topology guard')
            inc=fast_manifold_increment(previous,row);consecutive=consecutive+1 if block>=3 and fast_manifold_converged(inc,TOL) else 0
            if consecutive>=3:return current,row,dict(converged=True,iterations=block*10,blocks=block)
            previous=row
        return current,row,dict(converged=False,iterations=block*10,blocks=block)


class BufferedContactEvent(ContactEvent):
    """Exact native blocks with a persistent per-event scratch workspace."""
    def relax(self,state,q):
        current=tuple(x.copy() for x in state);previous=self.fast_metrics(current,q);f=current[0];consecutive=0;g=self.g;op=self.op
        shape=f.shape
        if not hasattr(self,'_native_workspace'):
            self._native_workspace=(np.empty_like(f),g['ownership'].copy(),np.empty_like(op.gb_density),
                np.empty_like(f),np.zeros((shape[0],shape[1]+1)),np.zeros_like(f))
        fn,pn,gbn,mu,jr,jz=self._native_workspace
        for block in range(1,self.max_fast_blocks+1):
            f,phi,density=native_block_reuse(f,g['ownership'],op.gb_density,*self.pair,self.dt,1.0937500000000001e-25,
                op.Wc,op.k_eta,op.W_f,op.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],op.W,op.physics.M_s,
                fn,pn,gbn,mu,jr,jz)
            g['ownership']=phi;op.gb_density=density;current=(f,*(phi*f[None]));row=self.fast_metrics(current,q)
            if row['topology_stop']:raise RuntimeError('three-grain topology guard')
            inc=fast_manifold_increment(previous,row);consecutive=consecutive+1 if block>=3 and fast_manifold_converged(inc,TOL) else 0
            if consecutive>=3:
                complete=self.metrics(current,q)
                return current,complete,dict(converged=True,iterations=block*10,blocks=block)
            previous=row
        return current,row,dict(converged=False,iterations=block*10,blocks=block)


class LargerDtBufferedContactEvent(ContactEvent):
    """Refinement-qualified 2x native dt with rotating five-step buffers."""
    def __init__(self,g,pair=(0,1),max_fast_blocks=60):
        super().__init__(g,pair,max_fast_blocks);self.dt*=2
    def relax(self,state,q):
        current=tuple(x.copy() for x in state);previous=self.fast_metrics(current,q);f=current[0];consecutive=0;g=self.g;op=self.op
        shape=f.shape
        if not hasattr(self,'_native_workspace'):
            self._native_workspace=(np.empty_like(f),g['ownership'].copy(),np.empty_like(op.gb_density),
                np.empty_like(f),np.zeros((shape[0],shape[1]+1)),np.zeros_like(f))
        fn,pn,gbn,mu,jr,jz=self._native_workspace
        for block in range(1,self.max_fast_blocks+1):
            prior_f,prior_phi,prior_density=f,g['ownership'],op.gb_density
            f,phi,density=native_block_reuse(f,g['ownership'],op.gb_density,*self.pair,self.dt,1.0937500000000001e-25,
                op.Wc,op.k_eta,op.W_f,op.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],op.W,op.physics.M_s,
                fn,pn,gbn,mu,jr,jz,steps=5)
            fn,pn,gbn=prior_f,prior_phi,prior_density
            self._native_workspace=(fn,pn,gbn,mu,jr,jz)
            g['ownership']=phi;op.gb_density=density;current=(f,*(phi*f[None]));row=self.fast_metrics(current,q)
            if row['topology_stop']:raise RuntimeError('three-grain topology guard')
            inc=fast_manifold_increment(previous,row);consecutive=consecutive+1 if block>=3 and fast_manifold_converged(inc,TOL) else 0
            if consecutive>=3:return current,self.metrics(current,q),dict(converged=True,iterations=block*5,blocks=block)
            previous=row
        return current,self.metrics(current,q),dict(converged=False,iterations=block*5,blocks=block)

def main():
    source=Path('runs/three_particle_production_065/forced_left_minimum_increment/watch_q0.800000.npz');state,restart,_,_=load_event_checkpoint(source)
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=BufferedContactEvent(g,max_fast_blocks=240);event.bind(state);op=event.op
    native_block(state[0].copy(),g['ownership'].copy(),op.gb_density.copy(),0,1,event.dt,1.0937500000000001e-25,op.Wc,op.k_eta,op.W_f,op.k_f,g['dr'],g['dz'],g['r_c'],g['r_f'],op.W,op.physics.M_s,steps=0)
    wall=time.perf_counter();result=event.run(state,target=.0025)
    with np.load('runs/three_particle_production_065/event_increment_refinement.npz') as d:native=d['one'].copy()
    reference=json.loads(Path('docs/three_particle/production_065/event_increment_refinement.json').read_text())['rows'][0]
    metrics=event.metrics(result[:4],result[5]['event_progress_over_b'])
    r=dict(label='OFF_TRAJECTORY_BUFFERED_NATIVE_EVENT',source=str(source),completed=result[4],stop_reason=result[5]['stop_reason'],wall_s=time.perf_counter()-wall,
        fields_Linf_to_native=float(np.max(abs(np.array(result[:4])-native))),sigma_difference_to_native_Pa=metrics['LEFT_sigma_local_Pa']-reference['metrics']['LEFT_sigma_local_Pa'],
        relative_clock_difference=result[5]['event_time_model']*MANIFEST['seconds_per_model_time']/reference['clock_s']-1,
        native_blocks=result[5]['packets'][-1]['fast_relax_blocks'] if result[5]['packets'] else None,production_promoted=False)
    r['native_overlap_pass']=bool(r['completed'] and r['fields_Linf_to_native']<1e-12 and abs(r['sigma_difference_to_native_Pa'])<1e-3 and abs(r['relative_clock_difference'])<1e-10)
    Path('docs/three_particle/production_065/buffered_event_probe.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed('runs/three_particle_production_065/buffered_event_probe.npz',initial=np.array(state),native=native,buffered=np.array(result[:4]));print(r,flush=True)
if __name__=='__main__':main()
