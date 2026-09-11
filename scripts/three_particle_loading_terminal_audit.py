"""Discarded native-mode and step-refinement probes at the reflection stop."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from three_particle_implicit_run import advance
D=Path('docs/three_particle/volume_loading_065');source=Path('runs/three_particle_volume_loading_065_cached/checkpoint.npz')
with np.load(source) as d:f=d['f'].copy();t=float(d['t_model'])
c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=FullJacobianSurfaceDiffusion(op,reuse_preconditioner=True);F=it.rhs(f);odd=f-f[::-1];oddF=F-F[::-1];i=np.unravel_index(np.argmax(abs(odd)),f.shape);mode=odd/np.max(abs(odd));Kr,Kz=it.flux_jacobian(f,op.potential(f).copy());jmode=(it.Dr@(Kr@mode.ravel())+it.Dz@(Kz@mode.ravel())).reshape(f.shape)
mode_rows=[]
for eps in [1e-8,1e-9,1e-10]:
 fd=(it.rhs(f+eps*mode)-it.rhs(f-eps*mode))/(2*eps);mode_rows.append(dict(epsilon=eps,rayleigh_per_model_time=float(np.sum(fd*mode)/np.sum(mode*mode))))
# A symmetrized COPY tests equivariance; it is never an accepted/restart state.
sym=.5*(f+f[::-1]);symF=it.rhs(sym)
rows=[];fields=[]
for h in [.001,.0005,.00025]:
 trial,error=advance(f,h,it,rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']);fields.append(trial);rows.append(dict(h_model=h,embedded_error=error,mirror_error=float(np.max(abs(trial-trial[::-1]))),mirror_growth_per_model_time=float((np.max(abs(trial-trial[::-1]))-np.max(abs(odd)))/h),f_min=float(trial.min()),f_max=float(trial.max()),energy_relative_change=float(op.energy(trial)/op.energy(f)-1)));print(rows[-1],flush=True)
r=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),time_s=t*op.physics.seconds_per_model_time,index=list(map(int,i)),z_nm=float(g['z'][i[0]]*1e9),r_nm=float(g['r_c'][i[1]]*1e9),field_value=float(f[i]),mirror_field_value=float(f[::-1][i]),mirror_error=float(np.max(abs(odd))),native_pointwise_mirror_growth_per_model_time=float(np.sign(odd[i])*oddF[i]),native_rayleigh_per_model_time=float(np.sum(odd*oddF)/np.sum(odd*odd)),analytic_jacobian_rayleigh_per_model_time=float(np.sum(mode*jmode)/np.sum(mode*mode)),finite_difference_modes=mode_rows,symmetric_copy_native_rhs_mirror_error=float(np.max(abs(symF-symF[::-1]))),gb_density_mirror_error_Pa=float(np.max(abs(op.gb_density-op.gb_density[::-1]))),rows=rows,discarded_copies_only=True,no_accepted_symmetrization=True,no_physical_instability_claim=True)
(D/'terminal_native_mode_audit.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(D/'terminal_native_mode_audit.npz',fields=np.array(fields));print(r,flush=True)
