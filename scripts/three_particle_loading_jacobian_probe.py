"""Audit native Frechet derivative and discarded full-Jacobian steps."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from three_particle_implicit_run import advance
D=Path('docs/three_particle/volume_loading_065');source=D/'late_stiffness_state.npz';rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
with np.load(source) as d:f=d['f'].copy()
c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=FullJacobianSurfaceDiffusion(op);mode=f-f[::-1];mode/=np.max(abs(mode));mu=op.potential(f).copy();Kr,Kz=it.flux_jacobian(f,mu);exact=(it.Dr@(Kr@mode.ravel())+it.Dz@(Kz@mode.ravel())).reshape(f.shape);derivatives=[];print("analytic Rayleigh",float(np.sum(mode*exact*g["r_c"][None,:])/np.sum(mode*mode*g["r_c"][None,:])),flush=True)
for eps in [1e-6,1e-7,1e-8]:
 fd=(it.rhs(f+eps*mode)-it.rhs(f-eps*mode))/(2*eps);derivatives.append(dict(epsilon=eps,relative_L2_difference=float(np.linalg.norm(fd-exact)/np.linalg.norm(exact)),max_difference=float(np.max(abs(fd-exact)))));print(derivatives[-1],flush=True)
rows=[];fields=[]
for h in [20.,10.,5.]:
 wall=time.perf_counter()
 try:
  result,error=advance(f,h,it,rule);fields.append(result);row=dict(h_model=h,status='COMPUTED_DISCARDED',error=error,mirror_error=float(np.max(abs(result-result[::-1]))),field_change=float(np.max(abs(result-f))),energy_relative_change=op.energy(result)/op.energy(f)-1)
 except (RuntimeError,FloatingPointError) as e:row=dict(h_model=h,status='FAILED',reason=str(e));fields.append(f.copy())
 row['wall_s']=time.perf_counter()-wall;rows.append(row);print(row,flush=True)
(D/'full_jacobian_probe.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),derivative_checks=derivatives,rows=rows,discarded_copies_only=True),indent=2)+'\n');np.savez_compressed(D/'full_jacobian_probe.npz',fields=np.array(fields),native_mode_derivative=exact)
