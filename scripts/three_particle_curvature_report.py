"""Read-only curvature profiles across a forced event's immutable checkpoints."""
from pathlib import Path
import sys,json,argparse
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from three_particle_forced_event import ContactEvent,MANIFEST
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_diagnostics import radius_profile,curvature_watch
from monitor_current_state_transfer_curvature import branch_profile,apply_progressive_flags
D=Path('docs/three_particle/production_065')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='forced_left_fine_increment');args=ap.parse_args()
    run=Path('runs/three_particle_production_065')/args.run
    with np.load(run/'checkpoint.npz') as d:base=tuple(d['base_fields'])
    frames={0.:base}
    for path in [Path('runs/three_particle_production_065/forced_left/watch_q0.275000.npz'),*sorted(run.glob('watch_q*.npz')),*run.glob('final.npz')]:
        with np.load(path) as d:
            q=json.loads(str(d['restart_json']))['cumulative_q_m']/MANIFEST['b_event_m'] if 'restart_json' in d else float(d['q'])
            frames[q]=tuple(d['fields'])
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g)
    fig,axes=plt.subplots(2,3,figsize=(13,7));rows=[];payload={}
    for q,state in sorted(frames.items()):
        metrics=event.metrics(state,q);positions=np.array([metrics[k+'_z_TJ_m'] for k in ['LEFT','RIGHT']]);R=radius_profile(state[0],g);z=g['z'];color=plt.cm.viridis(q)
        watch=curvature_watch(state[0],event.op,two_contact_center=True,gb_positions=positions)
        rows.append(dict(q_over_b=q,watch=watch))
        for row,valid in enumerate([np.isfinite(R)&(z>=positions[0])&(z<=positions[1]),np.isfinite(R)&(z<=positions[0])]):
            profile=branch_profile(z[valid],R[valid],z_tj=positions[0],r_tj=np.interp(positions[0],z[np.isfinite(R)],R[np.isfinite(R)]),W=4e-9,side='center' if row==0 else 'outer')
            prefix=f'q{q:.6f}_{row}_'
            for key,value in profile.items():payload[prefix+key]=value
            axes[row,0].plot(profile['z_m']*1e9,profile['r_m']*1e9,color=color,label=f'{q:.3f}')
            axes[row,1].plot(profile['z_m']*1e9,profile['kappa_m_per_m']/1e6,color=color)
            axes[row,2].plot(profile['z_m']*1e9,profile['dkappa_m_ds_per_m2']/1e14,color=color)
    for row in range(2):
        for col in range(3):
            ax=axes[row,col];ax.set_xlabel('z (nm)');ax.grid(alpha=.2)
            ax.set_xlim((-20,20) if row==0 else (-70,-10))
            for b in positions:
                ax.axvline(b*1e9,color='k',ls=':',lw=.7)
            for edge in [-3,3,-10,10]:ax.axvline((positions[0]+edge*4e-9)*1e9,color='gray',ls='--',lw=.6)
    axes[0,0].legend(title='q/b',fontsize=7);axes[0,0].set_ylabel('Center surface radius (nm)');axes[1,0].set_ylabel('LEFT outer surface radius (nm)')
    for row in range(2):axes[row,1].set_ylabel('Meridional curvature (1/µm)');axes[row,2].set_ylabel('Curvature gradient (10¹⁴/m²)')
    fig.suptitle('Forced event: profiles before masking; dashed lines = LEFT 3W/10W watch distances')
    fig.tight_layout();fig.savefig(D/'forced_curvature_profiles.png',dpi=150);fig.savefig(D/'forced_curvature_profiles.pdf');plt.close(fig)
    # A within-event screen is reported separately from the established
    # three-completed-event criterion; it cannot certify an avalanche by itself.
    flagged=[]
    for branch in rows[0]['watch']:
        history=[dict(avalanche_id=1,artifact_flag=0,**r['watch'][branch]) for r in rows];apply_progressive_flags(history)
        flagged.extend(dict(branch=branch,q_over_b=r['q_over_b'],reason='Within-event analogue: '+h['artifact_reason'].replace('three-event','three-snapshot')) for r,h in zip(rows,history) if h['artifact_flag'])
    report=dict(source_run=str(run),snapshots=rows,within_event_analog_screen=flagged,
        established_criterion='three completed events, not within-event snapshots',qualification_decision='PENDING physical profile review')
    (D/'forced_curvature_profiles.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(run/'curvature_profiles.npz',**payload)
    print(json.dumps(dict(q_values=list(frames),within_event_analog_screen=flagged)),flush=True)
if __name__=='__main__':main()
