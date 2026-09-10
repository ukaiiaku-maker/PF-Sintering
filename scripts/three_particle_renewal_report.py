"""Phase-resolved volume/strain analysis, with explicit conditional-study labels."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--conditional',action='store_true');args=ap.parse_args()
    launch=json.loads((args.run/'launch.json').read_text())
    if not args.conditional and launch.get('stochastic_result') is not True:raise ValueError('genuine stochastic launch required; label conditional data explicitly')
    history=json.loads((args.run/'history.json').read_text());records=[]
    for row in history:
        conditional=args.conditional;m=row.get('metrics',{})
        records.append(dict(time=row['time_since_forced_root_s'] if conditional else row['time_s'],
            volume=m['V_center_m3'] if conditional else row['center_volume_m3'],strain=m['chain_strain'] if conditional else row['chain_strain'],
            contacts=row['contacts'],diagnostics=row['diagnostics'],phase=row['phase'],event=row['event_number'],avalanche=row['avalanche_id'],q=row['q_over_b'],
            center_stress=m['center_particle_mean_local_Pa'] if conditional else row['center_particle_mean_local_Pa'],cluster_stress=row['cluster_area_weighted_local_Pa'],
            root_ratio=row.get('H_over_Hstar'),descendant_hazard=row.get('descendant_hazard'),descendant_threshold=row.get('descendant_threshold'),source_amplitude=row.get('source_amplitude')))
    t=np.array([r['time'] for r in records]);v=np.array([r['volume'] for r in records]);strain=np.array([r['strain'] for r in records]);elapsed=t-t[0]
    dt=np.diff(t);dv=-np.diff(v);de=np.diff(strain);phase_map={'ACTIVE_ONE_B':'event','RELOAD':'reload','ROOT_CROSSING':'reload','SOURCE_WINDOW':'source_window','FACILITATED_WINDOW':'source_window','CHILD_CROSSING':'source_window'}
    phase=np.array([phase_map.get(r['phase'],'boundary') for r in records[1:]])
    totals={};colors={'reload':'tab:blue','event':'tab:orange','source_window':'tab:green','boundary':'gray'}
    for label in colors:
        mask=(phase==label)&(dt>1e-12);x=dv[mask]/dt[mask];y=de[mask]/dt[mask]
        correlation=float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else None
        totals[label]=dict(intervals=int(mask.sum()),duration_s=float(dt[mask].sum()),center_volume_loss_m3=float(dv[mask].sum()),chain_strain_change=float(de[mask].sum()),descriptive_rate_correlation=correlation)
    events=[]
    for event in sorted({r['event'] for r in records if r['phase']=='ACTIVE_ONE_B'}):
        ids=[i for i,r in enumerate(records) if r['event']==event and r['phase'] in ['ACTIVE_ONE_B','ONE_B_COMPLETE','ONE_B_FAILED']]
        first=max(0,ids[0]-1);last=ids[-1];a,b=records[first],records[last]
        item=dict(event_number=event,avalanche_id=b['avalanche'],completed=b['phase']=='ONE_B_COMPLETE',q_over_b=b['q'],duration_s=b['time']-a['time'],center_volume_loss_m3=a['volume']-b['volume'],chain_strain_change=b['strain']-a['strain'],local_stress_before_Pa={k:a['contacts'][k]['sigma_local_Pa'] for k in ['LEFT','RIGHT']},local_stress_after_Pa={k:b['contacts'][k]['sigma_local_Pa'] for k in ['LEFT','RIGHT']})
        events.append(item)
    completed=[r for r in events if r['completed']]
    title='Conditional descendants after a forced root' if args.conditional else 'Genuine stochastic three-particle renewal'
    fig,axes=plt.subplots(4,3,figsize=(14,13));ax=axes.ravel()
    ax[0].plot(elapsed,100*(v/v[0]-1));ax[0].set_ylabel('Center volume change (%)')
    for name in ['LEFT','RIGHT']:ax[1].plot(elapsed,[r['contacts'][name]['sigma_local_Pa']/1e6 for r in records],label=name)
    ax[1].set_ylabel('Local activation stress (MPa)');ax[1].legend()
    if args.conditional:
        ratios=[r['descendant_hazard']/r['descendant_threshold'] if r['descendant_threshold'] and np.isfinite(r['descendant_threshold']) else np.nan for r in records]
        ax[2].plot(elapsed,ratios,label='Descendant H/H*');ax[2].text(.02,.08,'Genuine root thresholds not drawn',transform=ax[2].transAxes,fontsize=8)
    else:
        for name in ['LEFT','RIGHT']:ax[2].plot(elapsed,[r['root_ratio'][name] for r in records],label=name)
    ax[2].axhline(1,color='k',ls=':');ax[2].set_ylabel('Hazard / threshold');ax[2].legend()
    ax[3].plot(elapsed,100*strain);ax[3].set_ylabel('Global chain strain (%)')
    for key,label in [('center_stress','Center mean'),('cluster_stress','Area-weighted cluster')]:ax[4].plot(elapsed,[r[key]/1e6 for r in records],label=label)
    ax[4].set_ylabel('Aggregate diagnostics (MPa)');ax[4].legend()
    for name in ['LEFT','RIGHT']:ax[5].plot(elapsed,[r['contacts'][name]['sigma_integral_continuous_Pa']/1e6 for r in records],label=name)
    ax[5].set_ylabel('Integral diagnostic stress (MPa)');ax[5].legend()
    for label,color in colors.items():
        mask=(phase==label)&(dt>1e-12)
        if np.any(mask):ax[6].scatter(100*dv[mask]/v[0],100*de[mask],s=10,label=label,color=color,alpha=.6)
    ax[6].set_xlabel('Interval center volume loss (%)');ax[6].set_ylabel('Interval strain change (%)');ax[6].legend(fontsize=8)
    for e in events:ax[7].scatter(100*e['center_volume_loss_m3']/v[0],100*e['chain_strain_change'],marker='o' if e['completed'] else 'x');ax[7].annotate(str(e['event_number']),(100*e['center_volume_loss_m3']/v[0],100*e['chain_strain_change']))
    ax[7].set_xlabel('Per-event center volume loss (%)');ax[7].set_ylabel('Per-event strain change (%)');ax[7].set_title('x = incomplete event')
    labels=list(colors)[:3];locations=np.arange(3)
    ax[8].bar(locations-.18,[100*totals[k]['center_volume_loss_m3']/v[0] for k in labels],width=.36,label='Volume loss (%)')
    ax[8].bar(locations+.18,[100*totals[k]['chain_strain_change'] for k in labels],width=.36,label='Strain change (%)')
    ax[8].set_xticks(locations,labels);ax[8].legend(fontsize=8)
    for name in ['LEFT','RIGHT']:
        ax[9].plot(elapsed,[r['diagnostics'][name+'_CC_stress_Pa']/1e6 for r in records],label=name)
    ax[9].set_ylabel('Cannon–Carter diagnostic (MPa)');ax[9].legend()
    if args.conditional:ax[9].set_title('Fits about original pinned planes',fontsize=9)
    ax[10].plot(elapsed,[r['q'] if r['phase'] in ['ACTIVE_ONE_B','ONE_B_COMPLETE'] else np.nan for r in records]);ax[10].set_ylabel('Event progress q/b')
    ax[11].step(elapsed,[r['source_amplitude'] for r in records],where='post');ax[11].set_ylabel('Descendant source amplitude h')
    for a in list(ax[:6])+list(ax[9:]):a.set_xlabel('Elapsed recorded time (s)')
    for a in ax:a.grid(alpha=.2)
    fig.suptitle(title+' — '+('not genuine root statistics' if args.conditional else 'predeclared seed, no threshold selection'));fig.tight_layout()
    fig.savefig(args.run/'volume_strain_analysis.pdf');fig.savefig(args.run/'volume_strain_analysis.png',dpi=130);plt.close(fig)
    report=dict(label=title,genuine_stochastic_result=not args.conditional,recorded_duration_s=float(t[-1]-t[0]),records=len(records),completed_events=len(completed),events=events,phase_totals=totals,
        notes=['Rate correlations are descriptive; samples are time-correlated and are not independent statistical replicates.','Raw interval increments have unequal durations; phase totals and complete-event changes are reported separately.','Positive volume loss denotes shrinking; negative volume loss denotes center growth.','No aggregate diagnostic enters activation.'])
    (args.run/'volume_strain_analysis.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['events','phase_totals']},flush=True)
if __name__=='__main__':main()
