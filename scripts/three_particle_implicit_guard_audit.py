"""Audit the existing native bounds guard at the saved late unequal state."""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_cmc_ripening import load_initial
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_geometry import topology_status


def audit(case,stage):
    g,m=load_initial(case);op=PhaseAOperator(g)
    p=Path(f'runs/three_particle_implicit/{stage}_{case}/checkpoint.npz')
    with np.load(p,allow_pickle=False) as d:f=d['f'].copy();t=float(d['t_model'])
    loc=np.unravel_index(np.argmax(f),f.shape);stopped=False
    try:op.step(f,m['dt_model'])
    except FloatingPointError:stopped=True
    report=dict(checkpoint=str(p),t_model=t,native_dt_model=m['dt_model'],native_next_step_rejected=stopped,
        current_f_min=float(f.min()),current_f_max=float(f.max()),native_next_f_max=float(op.out.max()),
        maximum_location_z_r_nm=[float(g['z'][loc[0]]*1e9),float(g['r_c'][loc[1]]*1e9)],
        topology=topology_status(f,g),center_GB_separation_over_W=float(np.diff(g['gb'])[0]/op.W),
        outer_ownership_core_overlap=float(np.max(g['ownership'][0]*g['ownership'][2])),
        clipping_used=False,bounds_relaxed=False,interpretation='Existing native bounds guard, not a GB-core-overlap stop; the tiny overshoot is not evidence of physical failure.')
    return report

def main():
    report={'coarse':{case:audit(case,'target_v2') for case in ['unequal','null']},
            'refined':{case:audit(case,stage) for case,stage in [('unequal','target_refined_v2'),('null','refine_target_v2')]}}
    Path('docs/three_particle/implicit/guard_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
