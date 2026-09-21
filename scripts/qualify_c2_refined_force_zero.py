#!/usr/bin/env python3
import argparse,json,runpy,sys,types
from pathlib import Path
import numpy as np
n=types.ModuleType('numba');n.njit=lambda *a,**k:(a[0] if a and callable(a[0]) else lambda f:f);n.prange=range;n.get_num_threads=lambda:1;n.set_num_threads=lambda _:None
sys.modules.setdefault('numba',n);sys.modules.setdefault('h5py',types.ModuleType('h5py'));sk=types.ModuleType('skimage');me=types.ModuleType('skimage.measure');me.find_contours=lambda *a,**k:None;sk.measure=me;sys.modules.setdefault('skimage',sk);sys.modules.setdefault('skimage.measure',me)
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from pf_sintering.constrained_densification_relaxation import volume_targets,multigrain_energy
from pf_sintering.constrained_newton_krylov import solve_constrained_stationary
from pf_sintering.three_particle_geometry import topology_status
D=runpy.run_path(str(ROOT/'scripts/qualify_c2_single_work_conjugate_event.py'));OUT=D['OUT']
def main():
 a=argparse.ArgumentParser();a.add_argument('--q',type=float,required=True);a.add_argument('--initial',required=True);x=a.parse_args();q=x.q
 g,f0,phi0,gb0,own,sm,src=D['load_problem']();phi=own(q);target=volume_targets(f0,phi0,g['z'],g['r_c'],g['config'].outer_radius)
 final=OUT/f'refined_q_{q:.6f}b.npz';cp=OUT/f'refined_q_{q:.6f}b_checkpoint.npz'
 if final.exists():print('ALREADY_COMPLETE',final);return
 source=cp if cp.exists() else Path(x.initial)
 with np.load(source) as d:f=d['f'].copy()
 def cb(field,row):
  print(json.dumps(row),flush=True)
  if row['stage']=='newton' or int(row['iteration'])%10==0:np.savez(cp,f=field)
 r=solve_constrained_stationary(f,phi,g,target,active_mask=np.ones_like(f,bool),lbfgs_max_iterations=300,newton_max_iterations=2000,include_moments=False,kkt_tolerance=1e-7,callback=cb,lbfgs_box_active_step=.01,newton_box_active_step=.01,gmres_maxiter=8,newton_minimum_damping=1e-4)
 top=topology_status(r.f,g);E=multigrain_energy(r.f,phi,g);np.savez_compressed(final,f=r.f,ownership=phi,active_mask=np.ones_like(f,bool),q_over_b=q,energy_J=E,converged=r.converged,projected_KKT_residual=r.projected_kkt_residual,normalized_constraint_residual=r.normalized_constraint_residual)
 meta=dict(q_over_b=q,converged=r.converged,KKT=r.projected_kkt_residual,volume_residual=r.normalized_constraint_residual,energy_J=E,topology_stop=bool(top['stop']),initial=str(source));(OUT/f'refined_q_{q:.6f}b.json').write_text(json.dumps(meta,indent=2)+'\n');print('RESULT',json.dumps(meta),flush=True)
 if not r.converged or top['stop']:raise SystemExit(2)
if __name__=='__main__':main()
