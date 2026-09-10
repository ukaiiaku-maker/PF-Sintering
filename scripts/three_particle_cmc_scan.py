"""Cheap CMC scan: disconnected independently solved caps are never accepted."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pf_sintering.three_particle_cmc import outer_cap,solve_center,compatible_chain,describe

def main():
    start=time.perf_counter();rows=[];compatible=[];cap=outer_cap()
    for ratio in [.4,.5,.6,.7,.8,1.]:
        for factor in [.2,.4,.6,.8,1.,1.2,1.5]:
            L=ratio*100e-9*factor
            try:row=describe(solve_center(ratio,L),cap)
            except ValueError as e:row=dict(ratio=ratio,center_half_length_nm=L*1e9,status='NO_REGULAR_SOLUTION',reason=str(e))
            rows.append(row)
        c,o=compatible_chain(ratio);compatible.append(describe(c,o))
    result=dict(status='ANALYTICAL_SCAN',wall_s=time.perf_counter()-start,outer_cap=cap,spacing_scan=rows,compatible_chains=compatible,
        convention='symmetric Young-balanced one-sided slopes; sum-principal K; radii are volume-equivalent')
    Path('docs/three_particle/cmc/geometry_scan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'wall_s':result['wall_s'],'compatible':compatible},indent=2))
if __name__=='__main__':main()
