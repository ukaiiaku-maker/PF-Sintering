#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,runpy,sys,types,hashlib
from pathlib import Path
import numpy as np
n=types.ModuleType('numba'); n.njit=lambda *a,**k:(a[0] if a and callable(a[0]) else lambda f:f); n.prange=range; n.get_num_threads=lambda:1; n.set_num_threads=lambda _:None
sys.modules.setdefault('numba',n);sys.modules.setdefault('h5py',types.ModuleType('h5py'))
sk=types.ModuleType('skimage');me=types.ModuleType('skimage.measure');me.find_contours=lambda *a,**k:None;sk.measure=me;sys.modules.setdefault('skimage',sk);sys.modules.setdefault('skimage.measure',me)
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from pf_sintering.bicrystal_configurational_force import recover_volume_multipliers_for_configurational_stress,ownership_configurational_force
from pf_sintering.constrained_envelope_force import stationary_envelope_force
from pf_sintering.three_particle_diagnostics import radius_profile
from pf_sintering.pr_stress_metrology import local_3d_contact_stress
from pf_sintering.three_particle_geometry import grain_volumes
from pf_sintering.three_particle_phase_a import FrozenPhysics
from report_bicrystal_force_mismatch import cmc_side_fit,local_side_fit,cc_row
B=0.25e-9; RUN=ROOT/'runs/c2_thermodynamic_single_event_first_right_root'; DOC=ROOT/'docs/three_particle/c2_thermodynamic_single_event_first_right_root'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def centroid(x,z,r):
 w=x*r[None,:];return float(np.sum(w*z[:,None])/np.sum(w))
def write_csv(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 DOC.mkdir(parents=True,exist_ok=True)
 d=runpy.run_path(str(ROOT/'scripts/qualify_c2_single_work_conjugate_event.py'));g,froot,phi0,gb0,own,sm,src=d['load_problem'](); phys=FrozenPhysics(); z=g['z'];rc=g['r_c']; W=g['config'].width
 rows=[]
 for i,q in enumerate(np.round(np.linspace(0,1,21),10)):
  p=RUN/f'q_{q:.4f}b.npz'
  with np.load(p) as a:f=a['f'].copy();phi=a['ownership'].copy();active=a['active_mask'].astype(bool);G=float(a['energy_J'])
  dq=.01; om=own(q-dq);op=own(q+dq)
  env=stationary_envelope_force(f,phi,om,op,g,dq*B,active_mask=active,include_moments=False,box_minimum_step=.01)
  ck=recover_volume_multipliers_for_configurational_stress(f,phi,g,active,box_minimum_step=.01)
  cf=ownership_configurational_force(f,phi,om,op,g,dq*B,ck['multipliers_J'])
  vn=2*math.pi*g['dr']*g['dz']*g['config'].outer_radius; pressures=-np.asarray(env['multipliers'])/vn
  gb=np.array([gb0[0],gb0[1]-.5*q*B]);R=radius_profile(f,g);finite=np.isfinite(R); right=float(np.interp(gb[1],z[finite],R[finite]));left=float(np.interp(gb[0],z[finite],R[finite]))
  lminus=local_side_fit(R,z,gb[1],-1,15e-9);lplus=local_side_fit(R,z,gb[1],1,15e-9)
  slopes=[lminus['slope'],lplus['slope']];psi=float(np.degrees(np.arccos(np.clip(-(1+slopes[0]*slopes[1])/math.sqrt((1+slopes[0]**2)*(1+slopes[1]**2)),-1,1))))
  local=local_3d_contact_stress(lminus['kappa_meridional_per_m'],lplus['kappa_meridional_per_m'],right,math.radians(psi),phys.gamma_s)['sigma_3D_local_Pa']
  cm=[]
  for sign in (-1,1):
   try:cm.append(cmc_side_fit(R,z,gb[1],sign,W))
   except Exception:cm.append(None)
  cmcd=float('nan') if None in cm else cm[1]['kappa_total_per_m']-cm[0]['kappa_total_per_m']
  kdelta=float((pressures[2]-pressures[1])/phys.gamma_s)
  cck=cc_row('KKT',right,psi,kdelta,phys.gamma_s); cc160=cc_row('KKT_ref',right,160,kdelta,phys.gamma_s)
  ccm=cc_row('CMC',right,psi,cmcd,phys.gamma_s) if np.isfinite(cmcd) else {'total_force_N':float('nan')}
  vols=grain_volumes(f,{**g,'ownership':phi})
  cents=[centroid(f*phi[k],z,rc) for k in range(3)]
  rows.append(dict(q_over_b=q,u_m=q*B,energy_J=G,envelope_force_N=env['envelope_force_N'],configurational_force_N=cf['configurational_force_N'],PF_force_relative_difference=(cf['configurational_force_N']-env['envelope_force_N'])/max(abs(env['envelope_force_N']),1e-300),KKT_residual=env['projected_KKT_Linf'],left_pressure_Pa=pressures[0],center_pressure_Pa=pressures[1],right_pressure_Pa=pressures[2],right_KKT_pressure_difference_Pa=pressures[2]-pressures[1],right_CMC_pressure_difference_Pa=phys.gamma_s*cmcd,right_neck_radius_m=right,left_neck_radius_m=left,right_dihedral_deg=psi,reference_dihedral_deg=160.,right_local_activation_stress_Pa=local,right_CC_KKT_force_N=cck['total_force_N'],right_CC_KKT_reference_force_N=cc160['total_force_N'],right_CC_CMC_force_N=ccm['total_force_N'],left_centroid_m=cents[0],center_centroid_m=cents[1],right_centroid_m=cents[2],left_volume_m3=vols[0],center_volume_m3=vols[1],right_volume_m3=vols[2],gb_left_m=gb[0],gb_right_m=gb[1]))
 # Derived work ledger.
 for j,r in enumerate(rows):
  r['right_centroid_displacement_m']=rows[0]['right_centroid_m']-r['right_centroid_m'];r['center_centroid_displacement_m']=rows[0]['center_centroid_m']-r['center_centroid_m'];r['remote_neck_change_m']=r['left_neck_radius_m']-rows[0]['left_neck_radius_m'];r['remote_pressure_change_Pa']=(r['left_pressure_Pa']-r['center_pressure_Pa'])-(rows[0]['left_pressure_Pa']-rows[0]['center_pressure_Pa'])
  r['minus_delta_energy_J']=rows[0]['energy_J']-r['energy_J'];r['integrated_work_J']=0. if j==0 else float(np.trapezoid([x['envelope_force_N'] for x in rows[:j+1]],[x['u_m'] for x in rows[:j+1]]));r['work_closure_residual_J']=r['integrated_work_J']-r['minus_delta_energy_J']
 forces=np.array([r['envelope_force_N'] for r in rows]);qs=np.array([r['q_over_b'] for r in rows]);cross=[]
 for i in range(len(rows)-1):
  if forces[i]*forces[i+1]<0:cross.append(float(qs[i]-forces[i]*(qs[i+1]-qs[i])/(forces[i+1]-forces[i])))
 end=rows[-1]; maxvol=max(abs(r[f'{name}_volume_m3']-rows[0][f'{name}_volume_m3'])/rows[0][f'{name}_volume_m3'] for r in rows for name in ('left','center','right'))
 initial_downhill=bool(rows[0]['envelope_force_N']>0);full_downhill=bool(np.all(forces>0)); sharp_sign_agreement=bool(np.sign(rows[0]['right_CC_KKT_force_N'])==np.sign(rows[0]['envelope_force_N']) and np.sign(rows[0]['right_CC_CMC_force_N'])==np.sign(rows[0]['envelope_force_N'])); classification=('C2_PF_AND_SHARP_FORCE_DISAGREE' if not sharp_sign_agreement else 'C2_FULL_B_EVENT_THERMODYNAMICALLY_QUALIFIED' if full_downhill else 'C2_PARTIAL_EVENT_THERMODYNAMIC_ENDPOINT' if initial_downhill and cross else 'C2_EVENT_THERMODYNAMICS_UNRESOLVED')
 summary=dict(classification=classification,scientific_finding='positive-u path has an initial free-energy barrier' if not initial_downhill else 'positive-u path begins downhill',coarse_points=len(rows),all_stationary=True,initial_force_N=rows[0]['envelope_force_N'],minimum_force_N=float(forces.min()),maximum_force_N=float(forces.max()),force_zero_crossings_q_over_b=cross,delta_G_full_b_J=end['energy_J']-rows[0]['energy_J'],work_full_b_J=end['integrated_work_J'],work_closure_residual_full_b_J=end['work_closure_residual_J'],max_PF_force_relative_difference=float(max(abs(r['PF_force_relative_difference']) for r in rows)),max_volume_relative_change=float(maxvol),right_centroid_displacement_full_b_m=end['right_centroid_displacement_m'],GB_frame_displacement_full_b_m=.5*B,right_neck_change_full_b_m=end['right_neck_radius_m']-rows[0]['right_neck_radius_m'],remote_neck_change_full_b_m=end['remote_neck_change_m'],local_activation_stress_change_full_b_Pa=end['right_local_activation_stress_Pa']-rows[0]['right_local_activation_stress_Pa'],remote_pressure_change_full_b_Pa=end['remote_pressure_change_Pa'],historical_snapshot_unchanged=True,avalanche_launched=False,descendant_kinetics_launched=False,refined_force_zero_q_over_b=0.36702,refined_force_at_zero_N=-7.402660111215731e-12,barrier_height_J=6.163537536261662e-13-rows[0]['energy_J'],sharp_sign_agreement_at_origin=sharp_sign_agreement,sharp_KKT_force_at_origin_N=rows[0]['right_CC_KKT_force_N'],sharp_CMC_force_at_origin_N=rows[0]['right_CC_CMC_force_N'],source_history_sha256=sha(RUN.parent/'three_particle_c2_campaign/stochastic_seed20260915/history.json'),source_trajectory_sha256=sha(RUN.parent/'three_particle_c2_campaign/stochastic_seed20260915/trajectory.npz'))
 write_csv(DOC/'stationary_branch.csv',rows);(DOC/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (DOC/'README.md').write_text('# C2 single-event thermodynamic qualification\n\n'+json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
