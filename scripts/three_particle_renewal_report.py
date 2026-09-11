"""Phase-resolved volume/strain analysis, with explicit conditional-study labels."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pf_sintering.three_particle_renewal import cumulative_event_quota
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--conditional',action='store_true');args=ap.parse_args()
    launch=json.loads((args.run/'launch.json').read_text())
    if not args.conditional and launch.get('stochastic_result') is not True:raise ValueError('genuine stochastic launch required; label conditional data explicitly')
    history=json.loads((args.run/'history.json').read_text());records=[]
    manifest=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text());b_event=manifest['b_event_m']
    reference_length=launch.get('densification_reference_length_m')
    if reference_length is None and args.conditional:
        with np.load(launch['root_checkpoint']) as data:base=data['base_fields'].copy()
        with np.load('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999/post_transient.npz') as data:z=data['z'].copy();r=data['r_c'].copy()
        weights=base[1:]*r[None,None,:];centers=np.sum(weights*z[None,:,None],axis=(1,2))/np.sum(weights,axis=(1,2));reference_length=float(centers[2]-centers[0])
    if reference_length is None or reference_length<=0:raise ValueError('initial centroid reference length required')
    for row in history:
        conditional=args.conditional;m=row.get('metrics',{})
        records.append(dict(time=row['time_since_forced_root_s'] if conditional else row['time_s'],
            volume=m['V_center_m3'] if conditional else row['center_volume_m3'],strain=m['chain_strain'] if conditional else row['chain_strain'],
            counted_strain=cumulative_event_quota(row['event_number'],row['phase'],row['q_over_b'])*b_event/reference_length,quota=cumulative_event_quota(row['event_number'],row['phase'],row['q_over_b']),contacts=row['contacts'],diagnostics=row['diagnostics'],phase=row['phase'],event=row['event_number'],avalanche=row['avalanche_id'],q=row['q_over_b'],
            center_stress=m['center_particle_mean_local_Pa'] if conditional else row['center_particle_mean_local_Pa'],cluster_stress=row['cluster_area_weighted_local_Pa'],
            root_ratio=row.get('H_over_Hstar'),descendant_hazard=row.get('descendant_hazard'),descendant_threshold=row.get('descendant_threshold'),source_amplitude=row.get('source_amplitude')))
    t=np.array([r['time'] for r in records]);v=np.array([r['volume'] for r in records]);strain=np.array([r['strain'] for r in records]);elapsed=t-t[0]
    counted=np.array([r['counted_strain'] for r in records]);dc=np.diff(counted)
    dt=np.diff(t);dv=-np.diff(v);de=np.diff(strain);phase_map={'ACTIVE_ONE_B':'event','RELOAD':'reload','ROOT_CROSSING':'reload','SOURCE_WINDOW':'source_window','FACILITATED_WINDOW':'source_window','CHILD_CROSSING':'source_window'}
    phase=np.array([phase_map.get(r['phase'],'boundary') for r in records[1:]])
    totals={};colors={'reload':'tab:blue','event':'tab:orange','source_window':'tab:green','boundary':'gray'}
    for label in colors:
        mask=(phase==label)&(dt>1e-12);x=dv[mask]/dt[mask];y=de[mask]/dt[mask]
        correlation=float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else None
        totals[label]=dict(intervals=int(mask.sum()),duration_s=float(dt[mask].sum()),center_volume_loss_m3=float(dv[mask].sum()),chain_strain_change=float(de[mask].sum()),descriptive_rate_correlation=correlation,production_densification_strain_change=float(dc[mask].sum()))
    events=[]
    for event in sorted({r['event'] for r in records if r['phase']=='ACTIVE_ONE_B'}):
        ids=[i for i,r in enumerate(records) if r['event']==event and r['phase'] in ['ACTIVE_ONE_B','ONE_B_COMPLETE','ONE_B_FAILED']]
        first=max(0,ids[0]-1);last=ids[-1];a,b=records[first],records[last]
        item=dict(event_number=event,avalanche_id=b['avalanche'],completed=b['phase']=='ONE_B_COMPLETE',q_over_b=b['q'],duration_s=b['time']-a['time'],center_volume_loss_m3=a['volume']-b['volume'],chain_strain_change=b['strain']-a['strain'],production_densification_strain_change=b['counted_strain']-a['counted_strain'],local_stress_before_Pa={k:a['contacts'][k]['sigma_local_Pa'] for k in ['LEFT','RIGHT']},local_stress_after_Pa={k:b['contacts'][k]['sigma_local_Pa'] for k in ['LEFT','RIGHT']})
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
    ax[3].plot(elapsed,100*strain);ax[3].set_ylabel('Geometric centroid strain (%)')
    for key,label in [('center_stress','Center mean'),('cluster_stress','Area-weighted cluster')]:ax[4].plot(elapsed,[r[key]/1e6 for r in records],label=label)
    ax[4].set_ylabel('Aggregate diagnostics (MPa)');ax[4].legend()
    for name in ['LEFT','RIGHT']:ax[5].plot(elapsed,[r['contacts'][name]['sigma_integral_continuous_Pa']/1e6 for r in records],label=name)
    ax[5].set_ylabel('Integral diagnostic stress (MPa)');ax[5].legend()
    for label,color in colors.items():
        mask=(phase==label)&(dt>1e-12)
        if np.any(mask):ax[6].scatter(100*dv[mask]/v[0],100*de[mask],s=10,label=label,color=color,alpha=.6)
    ax[6].set_xlabel('Interval center volume loss (%)');ax[6].set_ylabel('Interval centroid strain change (%)');ax[6].legend(fontsize=8)
    for e in events:ax[7].scatter(100*e['center_volume_loss_m3']/v[0],100*e['chain_strain_change'],marker='o' if e['completed'] else 'x');ax[7].annotate(str(e['event_number']),(100*e['center_volume_loss_m3']/v[0],100*e['chain_strain_change']))
    ax[7].set_xlabel('Per-event center volume loss (%)');ax[7].set_ylabel('Per-event centroid strain change (%)');ax[7].set_title('x = incomplete event')
    labels=list(colors)[:3];locations=np.arange(3)
    ax[8].bar(locations-.18,[100*totals[k]['center_volume_loss_m3']/v[0] for k in labels],width=.36,label='Volume loss (%)')
    ax[8].bar(locations+.18,[100*totals[k]['chain_strain_change'] for k in labels],width=.36,label='Centroid strain change (%)')
    ax[8].set_xticks(locations,labels);ax[8].legend(fontsize=8)
    for name in ['LEFT','RIGHT']:
        ax[9].plot(elapsed,[r['diagnostics'][name+'_CC_stress_Pa']/1e6 for r in records],label=name)
    ax[9].set_ylabel('Cannon–Carter diagnostic (MPa)');ax[9].legend()
    if args.conditional:ax[9].set_title('Fits about stored GB planes',fontsize=9)
    ax[10].plot(elapsed,[r['q'] if r['phase'] in ['ACTIVE_ONE_B','ONE_B_COMPLETE'] else np.nan for r in records]);ax[10].set_ylabel('Event progress q/b')
    ax[11].step(elapsed,[r['source_amplitude'] for r in records],where='post');ax[11].set_ylabel('Descendant source amplitude h')
    for a in list(ax[:6])+list(ax[9:]):a.set_xlabel('Elapsed recorded time (s)')
    for a in ax:a.grid(alpha=.2)
    fig.suptitle(title+' — '+('not genuine root statistics' if args.conditional else 'predeclared seed, no threshold selection'));fig.tight_layout()
    fig.savefig(args.run/'volume_strain_analysis.pdf');fig.savefig(args.run/'volume_strain_analysis.png',dpi=130);plt.close(fig)
    accounting,aa=plt.subplots(2,2,figsize=(11,8));aa=aa.ravel()
    aa[0].plot(elapsed,100*counted);aa[0].set_ylabel('Bicrystal quota-based strain (%)')
    aa[1].plot(elapsed,100*strain);aa[1].set_ylabel('Geometric centroid strain (%)')
    for a in aa[:2]:a.set_xlabel('Elapsed recorded time (s)')
    for label,color in colors.items():
        mask=(phase==label)&(dt>1e-12)
        if np.any(mask):
            aa[2].scatter(100*dv[mask]/v[0],100*dc[mask],s=10,label=label,color=color,alpha=.6)
            aa[3].scatter(100*dv[mask]/v[0],100*de[mask],s=10,label=label,color=color,alpha=.6)
    for a in aa[2:]:a.set_xlabel('Interval center volume loss (%)');a.legend(fontsize=8)
    aa[2].set_ylabel('Quota-based strain increment (%)');aa[3].set_ylabel('Centroid strain increment (%)')
    for a in aa:a.grid(alpha=.2)
    accounting.suptitle(title+' — strain accounting and independent geometry')
    accounting.text(.5,.005,'Quota-based strain advances during events by definition; that timing is not independent evidence of densification.',ha='center',fontsize=9)
    accounting.tight_layout(rect=[0,.025,1,.97]);accounting.savefig(args.run/'strain_accounting.pdf');accounting.savefig(args.run/'strain_accounting.png',dpi=130);plt.close(accounting)
    (args.run/'strain_accounting.json').write_text(json.dumps(dict(reference_length_m=reference_length,b_event_m=b_event,production_definition='cumulative accepted event quota times b / initial outer-grain centroid separation',geometric_definition='1 - current outer-grain centroid separation / initial separation',rows=[dict(time_s=r['time'],phase=r['phase'],event_number=r['event'],avalanche_id=r['avalanche'],cumulative_event_quota_over_b=r['quota'],production_densification_strain=r['counted_strain'],geometric_chain_strain=r['strain'],center_volume_m3=r['volume']) for r in records]),indent=2)+'\n')
    report=dict(reference_length_m=reference_length,production_densification_strain_change=float(counted[-1]-counted[0]),label=title,genuine_stochastic_result=not args.conditional,recorded_duration_s=float(t[-1]-t[0]),records=len(records),completed_events=len(completed),events=events,phase_totals=totals,
        notes=['Rate correlations are descriptive; samples are time-correlated and are not independent statistical replicates.','Raw interval increments have unequal durations; phase totals and complete-event changes are reported separately.','Positive volume loss denotes shrinking; negative volume loss denotes center growth.','No aggregate diagnostic enters activation.','chain_strain is the geometric centroid measure; production_densification_strain uses the bicrystal cumulative-quota definition.','Event confinement of quota-based strain follows its definition and is not independent evidence of mechanical densification.'])
    (args.run/'volume_strain_analysis.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['events','phase_totals']},flush=True)
if __name__=='__main__':main()
