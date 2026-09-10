"""Report the null search without promoting a flux root to PF equilibrium."""
from pathlib import Path
import sys,json,csv,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DOC=Path('docs/three_particle/stationary_null');ROOT=Path('runs/three_particle_stationary_null')


def rows(path):
    with Path(path).open() as stream:
        rr=[{k:float(v) for k,v in x.items() if v not in ['True','False']} for x in csv.DictReader(stream)]
    return {k:np.array([r[k] for r in rr]) for k in rr[0]}


def main():
    bracket=json.loads((DOC/'bracket.json').read_text());run=json.loads((DOC/'released_report.json').read_text());bound=json.loads((DOC/'native_bound_refinement.json').read_text())
    candidate=rows(ROOT/'released_flux_root'/'history.csv')
    old={case:rows(Path('docs/three_particle/implicit')/f'history_{case}.csv') for case in ['unequal','null']}
    start=.39884654514071577;end=min(candidate['t_s'][-1],old['unequal']['t_s'][-1])
    def at(d,key,t):return np.interp(t,d['t_s'],d[key])
    dvc=candidate['V_center_m3']/candidate['V_center_m3'][0]-1
    dm=at(candidate,'delta_mu_projected_Pa',end)-at(candidate,'delta_mu_projected_Pa',start)
    stress_changes={key:float(at(candidate,key,end)-at(candidate,key,start)) for key in ['LEFT_CC_stress_Pa','RIGHT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa','RIGHT_PF_geometric_stress_Pa']}
    total=float(max(abs(candidate['total_volume_m3']/candidate['total_volume_m3'][0]-1)));mirror=float(max(candidate['mirror_error']))
    gates=dict(reached_0p9_seconds=bool(candidate['t_s'][-1]>=.9*(1-1e-12)),center_drift_below_1ppm=bool(max(abs(dvc))<1e-6),post_transient_projected_mu_change_below_5000Pa=bool(abs(dm)<5000),post_transient_stress_changes_below_5000Pa=bool(max(abs(x) for x in stress_changes.values())<5000),total_mass=bool(total<1e-11),mirror=bool(mirror<1e-8),resolved_topology=not run['topology']['stop'],no_numerical_guard_stop=run['status']=='requested_time_reached')
    result=dict(status='PF_STATIONARY_NULL_NOT_QUALIFIED',flux_root_ratio=bracket['flux_root_ratio'],initial_relative_center_rate_per_model_time=bracket['root']['center_relative_rate_per_model_time'],initial_rhs_max=bracket['root']['diagnostics']['native_rhs_max_per_model_time'],initial_virtual_work_delta_spread_Pa=bracket['root']['diagnostics']['virtual_work_delta_spread_Pa'],gates=gates,end_seconds=float(candidate['t_s'][-1]),center_relative_drift=float(dvc[-1]),common_start_seconds=start,common_end_seconds=float(end),post_transient_projected_mu_change_Pa=float(dm),post_transient_stress_changes_Pa=stress_changes,total_mass_error=total,mirror_error=mirror,curvature_qualification='Initial/final native watcher archived; no stationary-null qualification is claimed.',phase_b_authorized=False)
    result['descriptive_unequal_minus_failed_candidate_excess_Pa']={key:float(at(old['unequal'],key,end)-at(old['unequal'],key,start)-stress_changes[key]) for key in stress_changes}
    (DOC/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    figures=[]
    samples=bracket['bracket_and_iterations'][:5];ratios=np.array([x['ratio'] for x in samples])
    fig,ax=plt.subplots(2,1,figsize=(9,8))
    ax[0].plot(ratios,[x['center_relative_rate_per_model_time'] for x in samples],'o-');ax[0].axhline(0,color='gray');ax[0].axvline(bracket['flux_root_ratio'],color='red',ls=':');ax[0].set(ylabel='Initial relative center rate / model time',title='Compatible CMC bracket, identical 200-step cleanup')
    for basis in ['profile','localization','normal_displacement']:
        ax[1].plot(ratios,[x['diagnostics']['virtual_work'][basis]['delta_Pa']/1e6 for x in samples],'o-',label=basis)
    ax[1].plot(ratios,[x['projected_mu_contrast_Pa']/1e6 for x in samples],'k--',label='far-TJ normal projection')
    ax[1].axhline(0,color='gray');ax[1].set(xlabel='CMC volume-equivalent radius ratio',ylabel='Potential contrast (MPa)',title='Basis-dependent virtual work: not equilibrium chemical potentials');ax[1].legend()
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();figures.append(fig)
    colors={'unequal':'#236f9c','null':'#ca8631','candidate':'#8640a1'}
    datasets=old|{'candidate':candidate};labels={'unequal':'unchanged 0.70','null':'analytical null','candidate':'initial-flux root (failed null)'}
    fig,axes=plt.subplots(3,1,figsize=(9,10))
    for case,d in datasets.items():
        axes[0].plot(d['t_s'],100*(d['V_center_m3']/d['V_center_m3'][0]-1),label=labels[case],color=colors[case],ls='--' if case=='candidate' else '-')
        axes[1].plot(d['t_s'],d['delta_mu_projected_Pa']/1e6,label=labels[case],color=colors[case],ls='--' if case=='candidate' else '-')
        axes[2].plot(d['t_s'],d['LEFT_CC_stress_Pa']/1e6,label=labels[case],color=colors[case],ls='--' if case=='candidate' else '-')
    for a,y in zip(axes,['Center-volume change (%)','Projected potential contrast (MPa)','CC stress (MPa)']):
        a.set(xlabel='Physical time (s)',ylabel=y,xscale='symlog');a.set_xscale('symlog',linthresh=.0002);a.axvline(start,color='gray',ls=':');a.grid(alpha=.2);a.legend()
    fig.suptitle('Zero initial net flux does not produce a stationary finite-width control');fig.tight_layout();figures.append(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,7));tt=np.linspace(start,end,301)
    for ax,key,label in zip(axes,['LEFT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa'],['CC increment (MPa)','PF increment (MPa)']):
        for case,d in datasets.items():ax.plot(tt,(at(d,key,tt)-at(d,key,start))/1e6,label=labels[case],color=colors[case],ls='--' if case=='candidate' else '-')
        ax.set(xlabel='Physical time (s)',ylabel=label);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Descriptive comparison only: the new candidate also drifts');fig.tight_layout();figures.append(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,7));h=np.array([r['dt_model'] for r in bound['rows']])
    for key,label in [('one_step_epsilon','one native trial step'),('fixed_window_epsilon','same finite time window')]:axes[0].plot(h,np.array([r[key] for r in bound['rows']])*1e8,'o-',label=label)
    axes[0].axhline(bound['initial_epsilon']*1e8,color='gray',ls=':',label='inherited epsilon');axes[0].set(xscale='log',ylabel='(max f − 1) × 10⁸');axes[0].legend()
    axes[1].plot(h,[r['max_field_difference_from_finest'] for r in bound['rows']],'o-');axes[1].set(xscale='log',xlabel='Native dt (model time)',ylabel='Max field difference from finest')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Rejected native-trial audit; bound unchanged, no clipping');fig.tight_layout();figures.append(fig)
    # Inspect the actual fields outside fixed TJ/pole exclusions.
    from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
    from pf_sintering.three_particle_phase_a import PhaseAOperator
    from three_particle_cmc_ripening import observe
    ratio=bracket['flux_root_ratio'];c,o=compatible_chain(ratio);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g)
    with np.load(ROOT/f'ratio_{ratio:.12f}'/'initial.npz') as saved:
        first=saved['f'];meta=json.loads(str(saved['metadata_json']))
    with np.load(ROOT/'released_flux_root'/'checkpoint.npz') as saved:last=saved['f']
    fig,axes=plt.subplots(2,1,figsize=(9,7));roughness={}
    for label,f in [('initial',first),('terminal',last)]:
        _,profile=observe(f,op,meta['rule'],0,0)
        valid=np.isfinite(profile['kappa_smooth'])&(profile['r']>6*op.W)
        for b in g['gb']:valid&=abs(g['z']-b)>3*op.W
        raw=profile['kappa_raw'];smooth=profile['kappa_smooth']
        valid&=np.isfinite(raw)
        roughness[label]=float(np.max(op.W*abs(raw[valid]-smooth[valid])))
        for side in [-1,1]:
            mask=valid&(np.sign(g['z'])==side)
            axes[0].plot(g['z']*1e9,np.where(mask,smooth*1e-9,np.nan),color='tab:blue' if label=='initial' else 'tab:purple',label=label if side==-1 else None)
            axes[1].plot(g['z']*1e9,np.where(mask,op.W*(raw-smooth),np.nan),color='tab:blue' if label=='initial' else 'tab:purple',label=label if side==-1 else None)
    axes[0].set(ylabel='Smoothed K (1/nm)');axes[1].set(ylabel='W × (raw K − smoothed K)',xlabel='z (nm)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Actual-field curvature: fixed 3W TJ and r<6W pole exclusions');fig.tight_layout();figures.append(fig)
    result['curvature_qualification']={'outside_core_max_W_times_raw_minus_smoothed_K':roughness,'scope':'Descriptive mesh-scale roughness and archived native watcher; this does not override failed stationarity gates.'}
    (DOC/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    with PdfPages(DOC/'study_plots.pdf') as pdf:
        for i,fig in enumerate(figures):pdf.savefig(fig);fig.savefig(DOC/f'study_{i+1}.png',dpi=150);plt.close(fig)
    (DOC/'candidate_history.csv').write_bytes((ROOT/'released_flux_root'/'history.csv').read_bytes())
    manifest=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (DOC/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
