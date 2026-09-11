"""Loading versus actual center-volume loss; no fitted activation correction."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pf_sintering.three_particle_loading import decompose,PLATEAU
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=Path('runs/three_particle_volume_loading_065'));args=ap.parse_args();R=args.run;D=Path('docs/three_particle/volume_loading_065');D.mkdir(parents=True,exist_ok=True)
report=json.loads((R/'report.json').read_text());launch=json.loads((R/'launch.json').read_text());extension=report['history']
old=json.loads(Path('docs/three_particle/production_screen/harmonic_continuation_0.65_119.999.json').read_text())['history']
rows=[dict(r,center_loss_fraction=-r['center_relative_change'],decomposition={n:decompose(c) for n,c in r['contacts'].items()}) for r in old if r['time_s']<extension[0]['time_s']]+extension
x=np.array([r['center_loss_fraction'] for r in rows]);xp=100*x
fig,axs=plt.subplots(3,3,figsize=(15,11));ax=axs.ravel();colors={'LEFT':'tab:blue','RIGHT':'tab:orange'};derivative={}
for name,color in colors.items():
 c=[r['contacts'][name] for r in rows];sigma=np.array([v['sigma_local_Pa'] for v in c]);ax[0].plot(xp,sigma/1e6,color=color,label=name)
 for side,style in [('negative',':'),('positive','--')]:ax[0].plot(xp,[v[f'sigma_local_{side}_Pa']/1e6 for v in c],style,color=color,alpha=.45,label=name+' '+side)
 ax[1].plot(xp,[v['root_rate_per_s'] for v in c],color=color,label=name)
 # Centered secant over 0.05 percentage points; descriptive derivative only.
 grid=np.arange(max(.001,x[0])+.00025,x[-1]-.00025,.00025)
 slopes=(np.interp(grid+.00025,x,sigma)-np.interp(grid-.00025,x,sigma))/.0005
 derivative[name]=dict(loss_fraction=grid.tolist(),d_sigma_Pa_per_loss_fraction=slopes.tolist(),window_loss_fraction=.0005)
 ax[2].plot(100*grid,slopes/1e8,color=color,label=name)
 for key,style in [('curvature_Pa','-'),('TJ_Pa','--')]:ax[3].plot(xp,[r['decomposition'][name]['production'][key]/1e6 for r in rows],style,color=color,label=name+' '+key)
 for side,style in [('negative','-'),('positive','--')]:
  ax[4].plot(xp,[v[f'kappa1_{side}_per_m']/1e6 for v in c],style,color=color,label=name+' '+side)
  ax[6].plot(xp,[np.degrees(v[f'theta_{side}_rad']) for v in c],style,color=color,label=name+' '+side)
 ax[5].plot(xp,[v['r_n_m']*1e9 for v in c],color=color,label=name)
ex=100*np.array([r['center_loss_fraction'] for r in extension]);v0=np.array(launch['baseline_volumes_m3'])
for k,name in enumerate(['LEFT outer','Center','RIGHT outer']):ax[7].plot(ex,[100*(r['grain_volumes_m3'][k]/v0[k]-1) for r in extension],label=name)
ax[8].plot(ex,[r['center_span_over_W'] for r in extension]);ax[8].axhline(8,color='r',ls=':',label='Existing resolution bound')
labels=['Local stress (MPa)','Root rate Γ (1/s)','dσ/d(loss percentage point) (MPa)','Production stress contributions (MPa)','Measured one-sided k_m (1/µm)','TJ radius (nm)','Measured one-sided angle (degrees)','Grain volume change (%)','Actual center span / W']
for a,label in zip(ax,labels):a.set_ylabel(label);a.set_xlabel('Center volume loss from post-cleanup baseline (%)');a.grid(alpha=.2);a.legend(fontsize=6)
fig.suptitle('Events OFF: current-field loading versus center-volume loss — '+report['status']);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(D/'loading_vs_volume.pdf',bbox_inches='tight');fig.savefig(D/'loading_vs_volume.png',dpi=130,bbox_inches='tight');plt.close(fig)
fig,aa=plt.subplots(3,3,figsize=(14,11));aa=aa.ravel()
for a,key,label in zip(aa[:4],['time_s','energy_J','mass_relative_error','mirror_error'],['Physical time (s)','PF free energy (J)','Relative material error','Reflection error']):a.plot(ex,[r[key] for r in extension]);a.set_ylabel(label)
aa[4].plot(ex,[r['f_max'] for r in extension],label='max f');aa[4].plot(ex,[r['f_min'] for r in extension],label='min f');aa[4].legend();aa[4].set_ylabel('Field extrema')
aa[5].plot(ex,[r['two_contact_wait_s'] for r in extension]);aa[5].set_ylabel('Instantaneous two-contact wait scale (s)')
for name,color in colors.items():
 aa[6].plot(ex,[r['contacts'][name]['sigma_integral_continuous_Pa']/1e6 for r in extension],color=color,label=name)
 for side,style in [('negative','-'),('positive','--')]:
  aa[7].plot(ex,[r['decomposition'][name][side]['curvature_Pa']/1e6 for r in extension],style,color=color,label=name+' '+side)
  aa[8].plot(ex,[r['decomposition'][name][side]['TJ_Pa']/1e6 for r in extension],style,color=color,label=name+' '+side)
for a,label in zip(aa[6:],['Continuous-integral diagnostic (MPa)','One-sided curvature term (MPa)','One-sided TJ term (MPa)']):a.set_ylabel(label);a.legend(fontsize=7)
for a in aa:a.set_xlabel('Center volume loss (%)');a.grid(alpha=.2)
fig.suptitle('Events-OFF continuation diagnostics');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(D/'loading_diagnostics.pdf',bbox_inches='tight');fig.savefig(D/'loading_diagnostics.png',dpi=130,bbox_inches='tight');plt.close(fig)
# Separate actual component scales expose small variations without changing stress.
fig,aa=plt.subplots(2,2,figsize=(12,8));mask=x>=.001
for name,color in colors.items():
 for j,(key,label) in enumerate([('curvature_Pa','Curvature contribution (MPa)'),('TJ_Pa','TJ contribution (MPa)')]):
  values=np.array([r['decomposition'][name]['production'][key] for r in rows]);aa[0,j].plot(xp[mask],values[mask]/1e6,color=color,label=name+' production')
  for side,style in [('negative','-'),('positive','--')]:
   values=np.array([r['decomposition'][name][side][key] for r in rows]);aa[1,j].plot(xp[mask],values[mask]/1e6,style,color=color,label=name+' '+side)
  aa[0,j].set_ylabel(label);aa[1,j].set_ylabel(label)
for a in aa.ravel():a.set_xlabel('Center volume loss (%)');a.legend(fontsize=7);a.grid(alpha=.2)
fig.suptitle('Exact stress decomposition, separate scales — loss ≥0.1%');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(D/'loading_decomposition.pdf',bbox_inches='tight');fig.savefig(D/'loading_decomposition.png',dpi=130,bbox_inches='tight');plt.close(fig)
checkpoints=[]
for target in report['saved_milestones']:
 path=R/f'loss_{100*target:g}pct.npz'
 with np.load(path) as data:t=float(data['t_model'])*launch['physical_manifest']['seconds_per_model_time']
 row=min(extension,key=lambda r:abs(r['time_s']-t));assert abs(row['time_s']-t)<1e-9
 checkpoints.append(dict(target_loss_fraction=target,actual_loss_fraction=row['center_loss_fraction'],time_s=t,source=str(path),center_span_over_W=row['center_span_over_W'],topology=row['topology'],local_stress_Pa={n:row['contacts'][n]['sigma_local_Pa'] for n in colors},root_rates_per_s={n:row['contacts'][n]['root_rate_per_s'] for n in colors},two_contact_wait_s=row['two_contact_wait_s']))
candidates=[]
for checkpoint in checkpoints:
 if not .02<=checkpoint['target_loss_fraction']<=.05:continue
 end=checkpoint['actual_loss_fraction'];start=end-.0005
 xs=np.array([r['center_loss_fraction'] for r in extension]);ts=np.array([r['time_s'] for r in extension]);time_window=float(np.interp(end,xs,ts)-np.interp(start,xs,ts))
 slopes={n:float((np.interp(end,xs,[r['contacts'][n]['sigma_local_Pa'] for r in extension])-np.interp(start,xs,[r['contacts'][n]['sigma_local_Pa'] for r in extension]))/.0005) for n in colors}
 healthy=not checkpoint['topology']['stop'] and checkpoint['center_span_over_W']>=8
 candidates.append(dict(checkpoint,loading_candidate_only=True,production_qualified=False,healthy=healthy,backward_window_loss_fraction=.0005,d_sigma_Pa_per_loss_fraction=slopes,center_loss_fraction_per_s=.0005/time_window,passes_operational_loading_screen=bool(healthy and min(slopes.values())>PLATEAU['max_abs_slope_Pa_per_fraction']),selection='Candidate for review only; no thresholds drawn or production launch'))
summary=dict(candidate_checkpoints=candidates,source_run=str(R),status=report['status'],events_enabled=False,thresholds_drawn=False,baseline=launch['baseline'],derivative_definition='Centered secant over 0.05 percentage points of volume loss, excluding x<0.1%; diagnostic only, never an activation correction.',derivative=derivative,checkpoints=checkpoints,plateau=report['plateau'],final=extension[-1],decision=('B: clear operational plateau' if report['status']=='STRESS_PLATEAU' else 'A: loading persists at candidate checkpoints; separate production qualification remains disabled' if report['status']=='TEN_PERCENT_MILESTONE' and any(c['passes_operational_loading_screen'] for c in candidates) else 'INCOMPLETE: trajectory running; no A/B decision' if report['status']=='RUNNING' else 'INCONCLUSIVE: trajectory stopped before a qualified A/B conclusion; inspect stop reason'))
(D/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print('status',report['status'],'loss_pct',100*x[-1],'checkpoints',len(checkpoints),flush=True)
