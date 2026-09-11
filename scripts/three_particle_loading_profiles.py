"""Actual events-off surface profiles; no event-state or geometry replay."""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_diagnostics import radius_profile
from monitor_current_state_transfer_curvature import branch_profile
D=Path('docs/three_particle/volume_loading_065')
paths=[Path('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999/checkpoint.npz'),Path('runs/three_particle_volume_loading_065_guarded/checkpoint.npz'),Path('runs/three_particle_volume_loading_065_full_jacobian/checkpoint.npz'),D/'automatic_snapshots/loss_1pct/checkpoint.npz']
c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);z=g['z'];left,right=g['gb'];fig,aa=plt.subplots(2,3,figsize=(13,8));data={}
for path in paths:
 with np.load(path) as d:f=d['f'];t=float(d['t_model'])*.01557994316955921;np.testing.assert_array_equal(d['z'],z)
 R=radius_profile(f,g)
 for row,mask in enumerate([np.isfinite(R)&(z>=left)&(z<=right),np.isfinite(R)&(z<=left)]):
  p=branch_profile(z[mask],R[mask],z_tj=left,r_tj=np.interp(left,z[np.isfinite(R)],R[np.isfinite(R)]),W=4e-9,side='center' if row==0 else 'outer');x=p['distance_from_TJ_m']/4e-9
  for key,value in p.items():data[f't{t:.8f}_{row}_{key}']=value
  for col,(key,scale) in enumerate([('r_m',1e9),('kappa_m_per_m',1e-6),('dkappa_m_ds_per_m2',1e-14)]):
   keep=(x>=1)&(x<=(7 if row==0 else 14));aa[row,col].plot(x[keep],p[key][keep]*scale,label=f'{t:.2f} s')
for row in range(2):
 for col,label in enumerate(['Surface radius (nm)','Signed profile curvature (1/µm)','Signed profile gradient (10¹⁴/m²)']):
  a=aa[row,col];a.set_ylabel(('Center: ' if row==0 else 'LEFT outer: ')+label);a.set_xlabel('Distance from LEFT TJ / W');a.grid(alpha=.2);a.legend(fontsize=7)
fig.suptitle('Events OFF: actual surface profiles during volume loading');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(D/'loading_profiles.pdf',bbox_inches='tight');fig.savefig(D/'loading_profiles.png',dpi=130,bbox_inches='tight');np.savez_compressed(D/'loading_profiles.npz',**data)
(D/'loading_profiles_sources.json').write_text(json.dumps([str(p) for p in paths],indent=2)+'\n')
