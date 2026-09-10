"""Read-only morphology of retained conditional-descendant checkpoints."""
from pathlib import Path
import argparse,sys,json,re,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from scipy.signal import find_peaks,peak_prominences
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from three_particle_forced_event import ContactEvent,MANIFEST
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
from pf_sintering.three_particle_diagnostics import radius_profile,curvature_watch
from monitor_current_state_transfer_curvature import branch_profile

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
 frames=[]
 for path in args.run.glob('event_*_q*.npz'):
  match=re.fullmatch(r'event_(\d+)_q([0-9.]+)\.npz',path.name)
  if match:frames.append((int(match[1]),float(match[2]),path))
 if not frames:raise ValueError('no retained event checkpoints')
 frames.sort();c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g)
 fig,axes=plt.subplots(2,3,figsize=(13,7));rows=[];payload={};visible_y=[[[] for _ in range(3)] for _ in range(2)]
 for index,(number,_,path) in enumerate(frames):
  state,restart,contact,label=load_event_checkpoint(path)
  if label!='DESCENDANT_AFTER_FORCED_ROOT' or contact!='LEFT':raise ValueError('requires conditional LEFT descendant checkpoints')
  q=restart['cumulative_q_m']/MANIFEST['b_event_m'];m=event.metrics(state,q);left=m['LEFT_z_TJ_m'];right=m['RIGHT_z_TJ_m'];z=g['z'];R=radius_profile(state[0],g)
  watch=curvature_watch(state[0],event.op,two_contact_center=True,gb_positions=[left,right]);color=plt.cm.viridis(index/max(len(frames)-1,1));details={}
  for row,valid in enumerate([np.isfinite(R)&(z>=left)&(z<=right),np.isfinite(R)&(z<=left)]):
   p=branch_profile(z[valid],R[valid],z_tj=left,r_tj=m['LEFT_r_n_m'],W=4e-9,side='center' if row==0 else 'outer')
   x=p['distance_from_TJ_m']/4e-9;grad=np.abs(p['dkappa_m_ds_per_m2']);peaks=find_peaks(grad)[0]
   physical=x[peaks]>=1
   if row==0:
    other=np.hypot(p['z_m']-right,p['r_m']-np.interp(right,z[np.isfinite(R)],R[np.isfinite(R)]))/4e-9
    physical&=other[peaks]>=1
   peaks=peaks[physical];near=peaks[np.abs(x[peaks]-3)<=.75]
   details['center' if row==0 else 'outer']=dict(unmasked_local_gradient_peaks_distance_W=x[peaks].tolist(),local_gradient_peaks_within_075W_of_3W=int(len(near)),near_3W_peak_gradients_per_m2=grad[near].tolist(),near_3W_peak_prominences_per_m2=peak_prominences(grad,near)[0].tolist())
   for key,value in p.items():payload[f'event{number}_q{q:.6f}_{row}_{key}']=value
   for col,(ax,y) in enumerate(zip(axes[row],[p['r_m']*1e9,p['kappa_m_per_m']/1e6,p['dkappa_m_ds_per_m2']/1e14])):
    ax.plot(x,y,color=color,label=f'E{number} q={q:.3f}')
    visible_y[row][col].extend(y[(x>=1)&(x<=(7 if row==0 else 14))&np.isfinite(y)].tolist())
  rows.append(dict(event_number=number,descendant_number=number-1,q_over_b=q,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
   sigma_LEFT_Pa=m['LEFT_sigma_local_Pa'],sigma_RIGHT_Pa=m['RIGHT_sigma_local_Pa'],center_span_m=m['actual_center_span_m'],topology_stop=m['topology_stop'],watch=watch,unmasked_peak_diagnostics=details))
 for row in range(2):
  for col,ax in enumerate(axes[row]):
   lo,hi=min(visible_y[row][col]),max(visible_y[row][col]);padding=max((hi-lo)*.1,1e-6);ax.set_ylim(lo-padding,hi+padding)
   ax.set_xlim((1,7) if row==0 else (1,14));ax.set_xlabel('Distance from LEFT TJ / W');ax.grid(alpha=.2)
   for edge in [3,10]:ax.axvline(edge,color='gray',ls='--',lw=.7)
  axes[row,0].set_ylabel(('Center' if row==0 else 'LEFT outer')+' surface radius (nm)')
  axes[row,1].set_ylabel('Meridional curvature (1/µm)');axes[row,2].set_ylabel('Curvature gradient (10¹⁴/m²)')
 axes[0,0].legend(fontsize=7)
 fig.suptitle('Conditional descendants after a forced root — unmasked profiles; no qualification override')
 fig.tight_layout();fig.savefig(args.run/'family_morphology.pdf');fig.savefig(args.run/'family_morphology.png',dpi=130);plt.close(fig)
 report=dict(label='CONDITIONAL_DESCENDANT_MORPHOLOGY',genuine_stochastic_result=False,qualification_changed=False,rows=rows,
  note='Unmasked local peaks supplement the unchanged three-completed-event monitor; partial-event snapshots do not establish completed-event qualification.')
 (args.run/'family_morphology.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(args.run/'family_morphology_profiles.npz',**payload)
 print([(r['event_number'],r['q_over_b'],r['sigma_LEFT_Pa']/1e6,r['unmasked_peak_diagnostics']['center']) for r in rows],flush=True)
if __name__=='__main__':main()
