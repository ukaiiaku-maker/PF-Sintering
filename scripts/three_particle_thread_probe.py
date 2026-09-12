"""Run one discarded-copy event segment for thread-count overlap."""
from pathlib import Path
import argparse,json,sys,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from numba import get_num_threads
from three_particle_buffered_event_probe import BufferedContactEvent,AllocatingBufferedContactEvent
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allocating',action='store_true');p.add_argument('--target-over-b',type=float,default=.01);a=p.parse_args()
 state,_,contact,label=load_event_checkpoint(a.source)
 if label!='GENUINE_STOCHASTIC_EVENT':raise ValueError('requires genuine event copy')
 c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);pair=(0,1) if contact=='LEFT' else (1,2)
 event=(AllocatingBufferedContactEvent if a.allocating else BufferedContactEvent)(g,pair,max_fast_blocks=512);start=time.perf_counter()
 result=event.run(state,target=a.target_over_b,initial_step_over_b=.005,minimum_step_over_b=.005,maximum_step_over_b=.005)
 wall=time.perf_counter()-start;metrics=event.metrics(result[:4],result[5]['event_progress_over_b'])
 np.savez_compressed(a.out,fields=np.array(result[:4]),metrics=json.dumps(metrics),wall_s=wall,threads=get_num_threads(),allocating=a.allocating)
 print('threads',get_num_threads(),'wall',wall,'completed',result[4],flush=True)
if __name__=='__main__':main()
