"""Descendant qualification after a forced completed root; not stochastic-root evidence."""
from pathlib import Path
import sys,json,time,os,argparse,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from three_particle_source_window import advance_source_window,DESCENDANT_CROSSING_TOLERANCE_S
from three_particle_buffered_event_probe import BufferedContactEvent
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint,save_event_checkpoint
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_diagnostics import diagnostics,curvature_watch
from pf_sintering.three_particle_renewal import RootClocks,locate_first_root
from pf_sintering.pr_avalanche import AvalancheController,AvalancheState,DescendantBarrier
from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
from monitor_current_state_transfer_curvature import apply_progressive_flags
D=Path('docs/three_particle/production_065');DESCENDANT_SEED=20260911


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root-checkpoint',type=Path,required=True);ap.add_argument('--out-name',default='forced_descendant_family');ap.add_argument('--resume-active',action='store_true');args=ap.parse_args()
    gate=json.loads((D/'qualification.json').read_text())
    if not gate.get('reload_qualified') or not gate.get('one_b_qualified'):raise RuntimeError('completed one-b mechanical qualification required')
    event_class=BufferedContactEvent if gate.get('event_engine')=='buffered_native' else ContactEvent
    state,restart,contact,label=load_event_checkpoint(args.root_checkpoint)
    if contact!='LEFT' or label!='FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION' or abs(restart['cumulative_q_m']/MANIFEST['b_event_m']-1)>1e-12:raise ValueError('requires a complete forced LEFT root')
    out=Path('runs/three_particle_production_065')/args.out_name;out.mkdir(exist_ok=args.resume_active)
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);reference=ContactEvent(g)
    reference.metrics(restart['base_fields'],0.);reference.bind(state);mass0=float(grain_volumes(state[0],g).sum())
    rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule'];p=MANIFEST['root_barrier_slice']
    controller=AvalancheController(barrier=DescendantBarrier(CompleteExpFloorParams(p['G0_eV'],p['Gfloor_eV'],p['a'],p['sigmahat_Pa'],p['n']),1.5),
        temperature_K=MANIFEST['temperature_K'],attempt_frequency_per_s=MANIFEST['clock_scale']/MANIFEST['seconds_per_model_time'],
        b_m=MANIFEST['b_event_m'],correlation_time_s=.009,rng=np.random.default_rng(DESCENDANT_SEED),facilitation_decay_alpha=.70)
    t=restart['event_time_model']*MANIFEST['seconds_per_model_time']
    records=[];completed_watches=[];status='RUNNING';wall=time.perf_counter();event_number=1;pending_restart=None
    if args.resume_active:
        with np.load(out/'family.npz') as data:
            saved=json.loads(str(data['metadata']));state=tuple(data['fields'].copy());g['gb']=data['gb'].copy()
        records=json.loads((out/'history.json').read_text())
        if records[-1]['phase']!='ACTIVE_ONE_B':raise ValueError('resume-active requires an accepted active-event checkpoint')
        event_number=saved['event_number'];t=saved['t_s']
        fields,pending_restart,active_contact,active_label=load_event_checkpoint(out/f'event_{event_number}.npz')
        if active_contact!='LEFT' or active_label!='DESCENDANT_AFTER_FORCED_ROOT':raise ValueError('wrong active checkpoint identity')
        if any(not np.array_equal(a,b) for a,b in zip(state,fields)):raise ValueError('family/event checkpoint mismatch')
        if records[-1]['time_since_forced_root_s']!=t:raise ValueError('history/checkpoint clock mismatch')
        if abs(records[-1]['q_over_b']-pending_restart['cumulative_q_m']/MANIFEST['b_event_m'])>1e-14:raise ValueError('history/checkpoint quota mismatch')
        expected=controller.manifest()
        for key in ['descendant_barrier','temperature_K','attempt_frequency_per_s','b_m','correlation_time_s']:
            if saved['controller'][key]!=expected[key]:raise ValueError('resume physical parameter mismatch: '+key)
        controller.state=AvalancheState(**saved['controller']['state']);controller.crossings=saved['controller']['crossings']
        controller.rng.bit_generator.state=saved['rng']
        if not controller.state.window_triggered:raise ValueError('active transit must retain its committed source crossing')
        completed_watches=[r['curvature'] for r in records if r['phase'] in ['FORCED_ROOT_COMPLETE','ONE_B_COMPLETE']]
        log=out/'resume_log.json';entries=json.loads(log.read_text()) if log.exists() else []
        entries.append(dict(event_number=event_number,q_over_b=records[-1]['q_over_b'],time_s=t,
            checkpoint_sha256=hashlib.sha256((out/f'event_{event_number}.npz').read_bytes()).hexdigest(),
            controller=saved['controller'],rng=saved['rng'],thresholds_redrawn=False,accepted_source_replayed=False))
        log.write_text(json.dumps(entries,indent=2)+'\n')
    else:
        controller.start(avalanche_id=1,root_cycle=1,start_time_s=0.);controller.complete_transit(t)
    def record(phase,q=0.):
        reference.bind(state);row=reference.metrics(state,q);cc=evaluate_contacts(state[0],reference.op,MANIFEST);extra,_=diagnostics(state[0],reference.op)
        if abs(row['total_volume_m3']/mass0-1)>1e-11:raise RuntimeError('family mass guard')
        if np.max(abs(sum(state[1:])-state[0]))>5e-15:raise RuntimeError('family ownership closure')
        if state[0].min() < -1e-8 or state[0].max()>1+1e-8:raise RuntimeError('unchanged family field guard')
        areas={k:np.pi*v['r_n_m']**2 for k,v in cc.items()}
        record=dict(time_since_forced_root_s=t,phase=phase,event_number=event_number,avalanche_id=1,q_over_b=q,
            metrics=row,contacts=cc,diagnostics=extra,descendant_hazard=controller.state.descendant_hazard,
            descendant_threshold=controller.state.descendant_threshold,source_amplitude=controller.state.source_amplitude,
            cluster_area_weighted_local_Pa=sum(areas[k]*cc[k]['sigma_local_Pa'] for k in cc)/sum(areas.values()),
            stochastic_root_result=False)
        if phase in ['FORCED_ROOT_COMPLETE','ONE_B_COMPLETE','ONE_B_FAILED']:
            record['curvature']=curvature_watch(state[0],reference.op,two_contact_center=True,gb_positions=[cc[k]['z_TJ_m'] for k in ['LEFT','RIGHT']])
            if phase!='ONE_B_FAILED':completed_watches.append(record['curvature'])
        records.append(record)
        if phase=='ONE_B_COMPLETE':
            for branch in record['curvature']:
                history=[dict(avalanche_id=1,artifact_flag=0,**w[branch]) for w in completed_watches];apply_progressive_flags(history)
                if history[-1]['artifact_flag']:raise RuntimeError('progressive curvature artifact: '+branch)
            if row['topology_stop']:raise RuntimeError('family topology terminal')
    def save():
        temp=out/'family.writing.npz';np.savez_compressed(temp,fields=np.array(state),gb=g['gb'],
            metadata=json.dumps(dict(status=status,t_s=t,controller=controller.manifest(),rng=controller.rng.bit_generator.state,event_number=event_number)))
        os.replace(temp,out/'family.npz');(out/'history.json').write_text(json.dumps(records,indent=2,default=float)+'\n')
    if not args.resume_active:
        (out/'launch.json').write_text(json.dumps(dict(label='DESCENDANT_QUALIFICATION_AFTER_FORCED_ROOT',seed=DESCENDANT_SEED,root_checkpoint=str(args.root_checkpoint),
            root_thresholds_drawn=False,qualification=gate,descendant_crossing_tolerance_s=DESCENDANT_CROSSING_TOLERANCE_S,descendant_parameters=controller.manifest(),physical_parameters_changed=False),indent=2)+'\n')
        record('FORCED_ROOT_COMPLETE',1.);save()
    try:
        while controller.state.avalanche_active:
            if pending_restart is None:
                deadline=controller.state.window_deadline_s
                while t<deadline-1e-14:
                    dt=min(MANIFEST['passive_hazard_quadrature_dt_model']*MANIFEST['seconds_per_model_time'],deadline-t)
                    probe=RootClocks.__new__(RootClocks);probe.active=None;probe.hazard={'LEFT':controller.state.descendant_hazard,'RIGHT':0.};probe.threshold={'LEFT':controller.state.descendant_threshold,'RIGHT':float('inf')}
                    def rates(fields):
                        reference.bind(fields);cc=evaluate_contacts(fields[0],reference.op,MANIFEST)['LEFT']
                        return {'LEFT':controller.rate(cc['sigma_local_Pa'],cc['r_n_m']),'RIGHT':0.}
                    evolved,elapsed,inc,child=locate_first_root(state,dt,lambda fields,seconds:advance_source_window(fields,seconds,g,(0,1),rule,reuse_small_step_preconditioner=True),rates,probe,
                        DESCENDANT_CROSSING_TOLERANCE_S)
                    state=tuple(evolved);t+=elapsed;reference.bind(state)
                    if child:controller.commit_crossing(crossing_time_s=t)
                    else:controller.state.descendant_hazard+=inc['LEFT'];controller.state.descendant_total_hazard+=inc['LEFT']
                    record('CHILD_CROSSING' if child else 'SOURCE_WINDOW');save()
                    if child:break
                if not controller.state.window_triggered:controller.expire_window(deadline);break
                if controller.state.S_completed>=26:raise RuntimeError('25-descendant safety cap; no false extinction')
                event_number+=1;controller.begin_transit();event=event_class(g,max_fast_blocks=gate.get('event_max_fast_blocks',240));event_start=t
                zero=event.run(state,maximum_accepted_states=0)
                save_event_checkpoint(out/f'event_{event_number}_q0.npz',state,zero[5]['event_restart'],contact='LEFT',label='DESCENDANT_AFTER_FORCED_ROOT')
            else:
                event=event_class(g,max_fast_blocks=gate.get('event_max_fast_blocks',240))
                event_start=t-pending_restart['event_time_model']*MANIFEST['seconds_per_model_time']
            def progress(packet,fields,event_restart):
                nonlocal state,t
                state=fields;t=event_start+event_restart['event_time_model']*MANIFEST['seconds_per_model_time']
                save_event_checkpoint(out/f'event_{event_number}.npz',state,event_restart,contact='LEFT',label='DESCENDANT_AFTER_FORCED_ROOT')
                record('ACTIVE_ONE_B',packet['q_end_over_b']);save()
                print('descendant',event_number-1,'q/b',packet['q_end_over_b'],'chain strain',records[-1]['metrics']['chain_strain'],flush=True)
            dq=gate['event_max_increment_over_b']
            result=event.run(state,restart=pending_restart,callback=progress,maximum_step_over_b=dq,initial_step_over_b=dq);pending_restart=None;state=result[:4];info=result[5];t=event_start+info['event_time_model']*MANIFEST['seconds_per_model_time']
            save_event_checkpoint(out/f'event_{event_number}_final.npz',state,info['event_restart'],contact='LEFT',label='DESCENDANT_AFTER_FORCED_ROOT')
            record('ONE_B_COMPLETE' if result[4] else 'ONE_B_FAILED',info['event_progress_over_b']);save()
            if not result[4]:raise RuntimeError(info['stop_reason'])
            controller.complete_transit(t)
            record('SOURCE_WINDOW_OPEN');save()
        cc=evaluate_contacts(state[0],reference.op,MANIFEST);g['gb']=np.array([cc[k]['z_TJ_m'] for k in ['LEFT','RIGHT']]);record('AVALANCHE_EXTINCT_REPINNED');status='FORCED_FAMILY_COMPLETE'
    except (RuntimeError,ValueError,FloatingPointError) as error:status='STOPPED: '+str(error)
    save();report=dict(status=status,descendants_completed=controller.state.S_completed-1,controller=controller.manifest(),wall_s=time.perf_counter()-wall,
        duration_s=t,stochastic_root_result=False,source=str(args.root_checkpoint),final=records[-1])
    (D/(args.out_name+'.json')).write_text(json.dumps(report,indent=2,default=float)+'\n');print(status,flush=True)
if __name__=='__main__':main()
