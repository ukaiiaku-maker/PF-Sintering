"""Diagnose, without clearing, the within-event mask-edge review flag."""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from three_particle_forced_event import ContactEvent
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_diagnostics import radius_profile
from monitor_current_state_transfer_curvature import branch_profile
D=Path('docs/three_particle/production_065');run=Path('runs/three_particle_production_065/forced_left_minimum_increment')
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);e=ContactEvent(g);rows=[]
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for path in sorted(run.glob('watch_q*.npz')):
 with np.load(path) as d:state=tuple(d['fields']);q=json.loads(str(d['restart_json']))['cumulative_q_m']/.25e-9
 m=e.metrics(state,q);left=m['LEFT_z_TJ_m'];right=m['RIGHT_z_TJ_m'];R=radius_profile(state[0],g);z=g['z'];valid=np.isfinite(R)&(z>=left)&(z<=right)
 p=branch_profile(z[valid],R[valid],z_tj=left,r_tj=m['r_TJ_m'],W=4e-9,side='center')
 x=p['distance_from_TJ_m']/4e-9;other=np.hypot(p['z_m']-right,p['r_m']-np.interp(right,z[np.isfinite(R)],R[np.isfinite(R)]))/4e-9
 monitored=(x>=3)&(other>=3);ids=np.flatnonzero(monitored);grad=abs(p['dkappa_m_ds_per_m2']);peak_ids=find_peaks(grad)[0];physical=(x[peak_ids]>=1)&(other[peak_ids]>=1);peak_ids=peak_ids[physical]
 imax=ids[np.argmax(grad[ids])];near=[int(i) for i in peak_ids if abs(x[i]-3)<=.75]
 rows.append(dict(q_over_b=q,masked_gradient_max_distance_W=float(x[imax]),masked_max_is_first_valid_sample=bool(imax==ids[0]),
   unmasked_local_gradient_peaks_distance_W=x[peak_ids].tolist(),unmasked_gradient_local_peaks_within_075W_of_3W=len(near),
   absolute_gradient_at_distances_225_3_375W=[float(np.interp(s,x,grad)) for s in [2.25,3.,3.75]],
   note='A boundary maximum alone does not distinguish a mask artifact from a smooth TJ-core tail; the original warning remains retained.'))
 for ax,y in zip(axes,[p['r_m']*1e9,p['kappa_m_per_m']/1e6,p['dkappa_m_ds_per_m2']/1e14]):ax.plot(x,y,label=f'{q:.3f}')
for ax in axes:ax.set_xlim(1,7);ax.axvline(3,color='k',ls='--');ax.axvspan(2.25,3.75,color='gray',alpha=.1);ax.set_xlabel('Distance from LEFT TJ / W');ax.grid(alpha=.2)
axes[0].set_ylabel('Surface radius (nm)');axes[0].set_ylim(138,140);axes[0].legend(title='q/b',fontsize=8)
axes[1].set_ylabel('Meridional curvature (1/µm)');axes[1].set_ylim(-27,8)
axes[2].set_ylabel('Curvature gradient (10¹⁴/m²)');axes[2].set_ylim(-10,40)
fig.suptitle('Unmasked center profiles near the LEFT 3W transfer boundary');fig.tight_layout();fig.savefig(D/'mask_edge_detail.pdf');fig.savefig(D/'mask_edge_detail.png',dpi=140)
(D/'mask_edge_detail.json').write_text(json.dumps(dict(rows=rows,clears_original_flag=False),indent=2)+'\n');print(rows,flush=True)
