"""Report completed or stopped forced mechanical qualification without relabeling it stochastic."""
from pathlib import Path
import sys,json,argparse,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from three_particle_forced_event import ContactEvent,MANIFEST
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint
from pf_sintering.three_particle_diagnostics import curvature_watch,radius_profile
from pf_sintering.three_particle_geometry import topology_status
D=Path('docs/three_particle/production_065');out=Path('runs/three_particle_production_065/forced_left_compiled')

def main():
    global out
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='forced_left_compiled');ap.add_argument('--prefix',default='forced');args=ap.parse_args()
    out=Path('runs/three_particle_production_065')/args.run
    result=json.loads((D/(args.run+'.json')).read_text())
    state,restart,_,_=load_event_checkpoint(out/'final.npz');base=restart['base_fields']
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g)
    before=event.metrics(base,0.);after=event.metrics(state,restart['cumulative_q_m']/MANIFEST['b_event_m'])
    frames=[(0.,base)]
    with np.load('runs/three_particle_production_065/forced_left/watch_q0.275000.npz') as d:frames.append((float(d['q']),tuple(d['fields'])))
    for checkpoint in sorted(out.glob('watch_q*.npz')):
        with np.load(checkpoint) as d:
            value=json.loads(str(d['restart_json']))['cumulative_q_m']/MANIFEST['b_event_m'] if 'restart_json' in d else float(d['q'])
            frames.append((value,tuple(d['fields'])))
    frames.append((after['q_over_b'],state));frames=sorted(dict(frames).items());audits=[]
    for q,fields in frames:
        row=event.metrics(fields,q);audits.append(dict(q=q,metrics=row,curvature=curvature_watch(fields[0],event.op,two_contact_center=True,gb_positions=[row['LEFT_z_TJ_m'],row['RIGHT_z_TJ_m']]),topology=topology_status(fields[0],{**g,'gb':np.array([row['LEFT_z_TJ_m'],row['RIGHT_z_TJ_m']])})))
    gates=dict(completed_one_b=bool(result['completed']),positive_chain_strain=after['chain_strain']>0,
        active_local_stress_drop=after['LEFT_sigma_local_Pa']<before['LEFT_sigma_local_Pa'],
        finite_positive_physical_clock=bool(np.isfinite(result['event_duration_s']) and result['event_duration_s']>0),
        conserved_material=abs(after['total_volume_m3']/before['total_volume_m3']-1)<1e-11,
        partition_closure=bool(np.max(abs(sum(state[1:])-state[0]))<5e-15),
        field_guard=bool(state[0].min()>=-1e-8 and state[0].max()<=1+1e-8),topology=not after['topology_stop'])
    report=dict(label='FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION',stochastic_result=False,
        completed=result['completed'],stop_reason=result['stop_reason'],stop_detail=result.get('stop_detail'),gates=gates,
        before=before,after=after,duration_s=result['event_duration_s'],
        center_volume_relative_event_change=after['V_center_m3']/before['V_center_m3']-1,
        energy_relative_event_change=after['G_phasefield_J']/before['G_phasefield_J']-1,
        material_relative_error=after['total_volume_m3']/before['total_volume_m3']-1,
        inactive_source_isolation='bitwise unit regression; total inactive eta may change through physical f evolution',
        sparse_morphology=audits,one_b_scalar_gates_pass=all(gates.values()),one_b_qualified=False,phase_b_enabled=False)
    review_path=D/'one_b_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        if review.get('final_checkpoint')==str(out/'final.npz') and review.get('final_checkpoint_sha256')==hashlib.sha256((out/'final.npz').read_bytes()).hexdigest():
            report['one_b_qualified']=bool(report['one_b_scalar_gates_pass'] and review.get('one_b_qualified'))
            report['qualification_review']=review_path.name
            report['qualification_scope']=review['qualification_scope']
    (D/(args.prefix+'_summary.json')).write_text(json.dumps(report,indent=2,default=float)+'\n')
    packets=result['packets'];q=np.r_[0.,[p['q_end_over_b'] for p in packets]]
    fig,axes=plt.subplots(2,3,figsize=(13,7));ax=axes.ravel()
    for name in ['LEFT','RIGHT']:ax[0].plot(q,np.r_[before[name+'_sigma_local_Pa'],[p[name+'_sigma_local_Pa'] for p in packets]]/1e6,label=name)
    ax[0].set_ylabel('Local production stress (MPa)');ax[0].legend()
    ax[1].plot(q,100*np.r_[0.,[p['chain_strain'] for p in packets]]);ax[1].set_ylabel('Chain strain (%)')
    ax[2].plot(q,np.r_[0.,[p['accumulated_event_time_model'] for p in packets]]*MANIFEST['seconds_per_model_time']);ax[2].set_ylabel('Physical event time (s)')
    ax[3].plot(q,np.r_[0.,[p['G_phasefield_J']/before['G_phasefield_J']-1 for p in packets]]);ax[3].set_ylabel('Relative free-energy change')
    ax[4].scatter([r['q'] for r in audits],[100*(r['metrics']['V_center_m3']/before['V_center_m3']-1) for r in audits]);ax[4].set_ylabel('Center volume change (%)\nsaved field samples only')
    for value,fields in frames:ax[5].plot(g['z']*1e9,radius_profile(fields[0],g)*1e9,label=f'q/b={value:.3f}')
    ax[5].set_xlim(-45,45);ax[5].set_ylim(136,141);ax[5].set_xlabel('z (nm)');ax[5].set_ylabel('Surface radius near contacts (nm)');ax[5].legend(fontsize=8)
    for a in ax[:5]:a.set_xlabel('q/b');a.grid(alpha=.2)
    fig.suptitle('FORCED mechanical qualification — '+('one-b completed' if result['completed'] else 'STOPPED before one b'))
    fig.tight_layout();fig.savefig(D/(args.prefix+'_event.pdf'));fig.savefig(D/(args.prefix+'_event.png'),dpi=130);plt.close(fig)
    print({k:v for k,v in report.items() if k not in ['before','after','sparse_morphology']},flush=True)
if __name__=='__main__':main()
