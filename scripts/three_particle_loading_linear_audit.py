"""Discarded-copy linear-solve accuracy audit of the dilute-tail odd mode."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import pf_sintering.three_particle_implicit as implicit
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from three_particle_implicit_run import advance
source=Path('docs/three_particle/volume_loading_065/late_stiffness_state.npz');D=source.parent;rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
with np.load(source) as d:initial=d['f'].copy()
original=implicit.gmres;rows=[];fields=[]
for tolerance in [1e-9,1e-12,1e-14]:
 calls=[]
 def solve(A,b,**kwargs):
  kwargs.update(rtol=tolerance,maxiter=6,restart=80);x,info=original(A,b,**kwargs);calls.append(dict(info=int(info),relative_residual=float(np.linalg.norm(A@x-b)/max(np.linalg.norm(b),1e-300))));return x,info
 implicit.gmres=solve
 c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=HarmonicSurfaceDiffusion(op);start=time.perf_counter()
 try:
  f,error=advance(initial,20.,it,rule);fields.append(f);row=dict(rtol=tolerance,status='COMPUTED_DISCARDED',embedded_error=error,mirror_error=float(np.max(abs(f-f[::-1]))),field_change=float(np.max(abs(f-initial))),energy_relative_change=op.energy(f)/op.energy(initial)-1)
 except (FloatingPointError,RuntimeError) as e:fields.append(initial.copy());row=dict(rtol=tolerance,status='FAILED',reason=str(e))
 row.update(calls=calls,wall_s=time.perf_counter()-start);rows.append(row);print(row,flush=True)
implicit.gmres=original
(D/'linear_accuracy_audit.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_mirror_error=float(np.max(abs(initial-initial[::-1]))),h_model=20.,rows=rows,discarded_copies_only=True),indent=2)+'\n');np.savez_compressed(D/'linear_accuracy_audit.npz',fields=np.array(fields))
