"""Reproduce discarded bound and preconditioner-cost probes from a saved field."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from three_particle_implicit_run import advance
D=Path('docs/three_particle/volume_loading_065')
with np.load(D/'full_jacobian_bound_probe_source.npz') as d:f=d['f'].copy();t=float(d['t_model'])
c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=FullJacobianSurfaceDiffusion(op);rows=[];fields=[]
for h in [32.,16.,8.]:
 try:trial=it.step(f,h);status='COMPUTED_DISCARDED'
 except FloatingPointError as e:trial=op.out.copy();status=str(e)
 ij=np.unravel_index(np.argmax(trial),trial.shape);low=np.unravel_index(np.argmin(trial),trial.shape);fields.append(trial)
 rows.append(dict(h=h,status=status,fmax=float(trial.max()),fmin=float(trial.min()),max_index=list(map(int,ij)),max_initial=float(f[ij]),min_index=list(map(int,low)),min_initial=float(f[low]),r_max_nm=float(g['r_c'][ij[1]]*1e9),z_max_nm=float(g['z'][ij[0]]*1e9)))
(D/'full_jacobian_bound_probe.json').write_text(json.dumps(dict(time_s=t*op.physics.seconds_per_model_time,rows=rows,discarded_copies_only=True),indent=2)+'\n');np.savez_compressed(D/'full_jacobian_bound_probe.npz',fields=np.array(fields))
rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];rows=[];fields=[]
for reuse in [False,True]:
 it=FullJacobianSurfaceDiffusion(PhaseAOperator(g),reuse_preconditioner=reuse);wall=time.perf_counter();trial,error=advance(f,4.,it,rule);rows.append(dict(reuse=reuse,error=error,wall_s=time.perf_counter()-wall));fields.append(trial)
r=dict(rows=rows,field_difference=float(np.max(abs(fields[0]-fields[1]))),discarded_copies_only=True);(D/'preconditioner_reuse_probe.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(D/'preconditioner_reuse_probe.npz',fields=np.array(fields));print(r,flush=True)
