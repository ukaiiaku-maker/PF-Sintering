"""Discarded fixed-q relaxation sensitivity; no live tolerance changes."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from scipy.signal import find_peaks,peak_prominences
import three_particle_buffered_event_probe as buffered
from three_particle_forced_event import MANIFEST,TOL
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
from pf_sintering.three_particle_diagnostics import radius_profile
from monitor_current_state_transfer_curvature import branch_profile
source=Path('runs/three_particle_production_065/forced_descendant_family/event_2_q0.500000.npz')
state,restart,_,_=load_event_checkpoint(source);q=restart['cumulative_q_m']/MANIFEST['b_event_m'];rows=[];fields=[]
def peak(fields,event):
 m=event.metrics(fields,q);g=event.g;z=g['z'];r=radius_profile(fields[0],g);left=m['LEFT_z_TJ_m'];right=m['RIGHT_z_TJ_m'];valid=np.isfinite(r)&(z>=left)&(z<=right)
 p=branch_profile(z[valid],r[valid],z_tj=left,r_tj=m['LEFT_r_n_m'],W=4e-9,side='center');x=p['distance_from_TJ_m']/4e-9;y=np.abs(p['dkappa_m_ds_per_m2']);ids=find_peaks(y)[0];ids=ids[(x[ids]>=3)&(x[ids]<=3.75)]
 return dict(distance_W=x[ids].tolist(),gradient_per_m2=y[ids].tolist(),prominence_per_m2=peak_prominences(y,ids)[0].tolist())
for scale in [1.,.5,.25]:
 c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=buffered.BufferedContactEvent(g,max_fast_blocks=2048)
 before=event.metrics(state,q);before_peak=peak(state,event);buffered.TOL={k:v*scale for k,v in TOL.items()};start=time.perf_counter()
 result,metrics,info=event.relax(state,q);fields.append(np.array(result))
 rows.append(dict(tolerance_scale=scale,tolerances=buffered.TOL,converged=info['converged'],blocks=info['blocks'],wall_s=time.perf_counter()-start,
  native_relaxation_time_s=info['blocks']*10*event.dt*MANIFEST['seconds_per_model_time'],
  field_Linf_from_input=float(np.max(abs(np.array(result)-np.array(state)))),relative_energy_change=metrics['G_phasefield_J']/before['G_phasefield_J']-1,
  sigma_LEFT_Pa=metrics['LEFT_sigma_local_Pa'],sigma_RIGHT_Pa=metrics['RIGHT_sigma_local_Pa'],relative_center_volume_change=metrics['V_center_m3']/before['V_center_m3']-1,
  relative_mass_error=metrics['total_volume_m3']/before['total_volume_m3']-1,ownership_closure=float(np.max(abs(sum(result[1:])-result[0]))),
  field_min=float(result[0].min()),field_max=float(result[0].max()),outside_3W_peak=peak(result,event)))
 print(rows[-1],flush=True)
report=dict(label='OFF_TRAJECTORY_FIXED_Q_RELAXATION_SENSITIVITY',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),q_over_b=q,
 original_peak=before_peak,rows=rows,live_tolerances_changed=False,additional_event_quota=0.,additional_stochastic_draws=0,
 note='Pseudo-time native relaxation at fixed quota; no transport-event clock or stochastic hazard is advanced. This is not an accepted continuation or a source-window trajectory.')
Path('docs/three_particle/production_065/fixed_q_relaxation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed('runs/three_particle_production_065/fixed_q_relaxation_audit.npz',initial=np.array(state),nominal=fields[0],half_tolerance=fields[1],quarter_tolerance=fields[2]);print(report,flush=True)
