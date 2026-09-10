"""Native-time-step audit of rejected trials, never accepted continuation.

All trials start from the SAME terminal field. The h->0 one-step epsilon
necessarily tends to the inherited epsilon; that alone cannot establish a
spatial equilibrium error. A fixed physical window separates this tautology
from time-integration convergence. No trial is saved as a restart state.
"""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_cmc_ripening import load_initial
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_null import native_rhs
from pf_sintering.three_particle_geometry import grain_volumes


def main():
    g,m=load_initial('unequal');op=PhaseAOperator(g)
    source=Path('runs/three_particle_implicit/target_refined_v2_unequal/checkpoint.npz')
    with np.load(source,allow_pickle=False) as d:f0=d['f'].copy();t0=float(d['t_model'])
    loc=np.unravel_index(np.argmax(f0),f0.shape);F=native_rhs(f0,op);dt=m['dt_model'];horizon=128*dt
    rows=[];final_fields=[];start=time.perf_counter();v0=grain_volumes(f0,g).sum()
    for factor in [1,2,4,8,16]:
        h=dt/factor;one=f0+h*F;f=f0.copy()
        for _ in range(128*factor):f=f+h*native_rhs(f,op)
        final_fields.append(f)
        rows.append(dict(refinement_factor=factor,dt_model=h,steps=128*factor,one_step_epsilon=float(one.max()-1),fixed_window_epsilon=float(f.max()-1),fixed_window_growth_at_original_max=float(f[loc]-f0[loc]),maximum_location_index=list(map(int,np.unravel_index(np.argmax(f),f.shape))),total_volume_relative_error=float(grain_volumes(f,g).sum()/v0-1)))
        print(rows[-1],flush=True)
    for r,f in zip(rows,final_fields):r['max_field_difference_from_finest']=float(np.max(abs(f-final_fields[-1])))
    report=dict(status='REJECTED_TRIAL_TIME_REFINEMENT_AUDIT',source=str(source),start_model=t0,initial_epsilon=float(f0.max()-1),maximum_location_z_r_nm=[float(g['z'][loc[0]]*1e9),float(g['r_c'][loc[1]]*1e9)],initial_rhs_at_max_per_model_time=float(F[loc]),fixed_window_model=horizon,fixed_window_seconds=horizon*op.physics.seconds_per_model_time,rows=rows,wall_s=time.perf_counter()-start,guard_relaxed=False,clipping_used=False,accepted_continuation=False,conclusion='Finite-time native trials converge to a nonzero overshoot from the terminal field; this is not proof that a spatially converged PF equilibrium exceeds one. Both an inherited overshoot and an outward semidiscrete derivative are present.')
    out=Path('docs/three_particle/stationary_null');out.mkdir(parents=True,exist_ok=True)
    (out/'native_bound_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
