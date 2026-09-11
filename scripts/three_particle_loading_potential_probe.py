"""Discarded algebraic formulation probe of the identical linearly implicit step."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu,gmres,LinearOperator
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from three_particle_implicit_run import advance
class PotentialStep(HarmonicSurfaceDiffusion):
 def __init__(self,op,rtol=1e-9):super().__init__(op);self.solves=[];self.rtol=rtol
 def step(self,f,h):
  A=self.mobility(f);H=self.Hgradient+sparse.diags(self.op.W_f*(1-6*f.ravel()+6*f.ravel()**2));mu=self.op.potential(f).copy();lhs=(self.identity-h*(H@A)).tocsc();lhs.eliminate_zeros()
  pre=lhs.copy();pre.data[abs(pre.data)<1e-7]=0;pre.eliminate_zeros();lu=splu(pre)
  effective,info=gmres(lhs,mu.ravel(),x0=mu.ravel(),M=LinearOperator(lhs.shape,lu.solve),rtol=self.rtol,atol=0.,restart=80,maxiter=6)
  residual=float(np.linalg.norm(lhs@effective-mu.ravel())/np.linalg.norm(mu));self.solves.append(dict(info=int(info),residual=residual))
  if info or residual>1e-8:raise FloatingPointError('potential linear solve')
  result=self.conservative_update(f,effective.reshape(f.shape),h)
  if not np.isfinite(result).all() or result.min() < -1e-8 or result.max()>1+1e-8:raise FloatingPointError('unchanged field guard')
  return result
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--strict',action='store_true');args=ap.parse_args();rtol=1e-11 if args.strict else 1e-9;stem='potential_formulation_strict_tolerance' if args.strict else 'potential_formulation_probe'
 D=Path('docs/three_particle/volume_loading_065');source=D/'late_stiffness_state.npz';rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];rows=[];fields=[]
 with np.load(source) as d:initial=d['f'].copy()
 for h in [20.,10.,5.]:
  c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=PotentialStep(op,rtol);wall=time.perf_counter()
  try:
   f,error=advance(initial,h,it,rule);fields.append(f);row=dict(h_model=h,status='COMPUTED_DISCARDED',embedded_error=error,mirror_error=float(np.max(abs(f-f[::-1]))),field_change=float(np.max(abs(f-initial))),energy_relative_change=op.energy(f)/op.energy(initial)-1)
  except (RuntimeError,FloatingPointError) as e:row=dict(h_model=h,status='FAILED',reason=str(e));fields.append(initial.copy())
  row.update(solves=it.solves,wall_s=time.perf_counter()-wall);rows.append(row);print(row,flush=True)
 (D/f'{stem}.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,linear_rtol=rtol,identity='(I-h H A) mu_eff = mu; f_new = f + h A mu_eff',physical_operator_unchanged=True,discarded_copies_only=True),indent=2)+'\n');np.savez_compressed(D/f'{stem}.npz',fields=np.array(fields))
