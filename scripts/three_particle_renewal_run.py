"""Gated current-field stochastic renewal for the selected 0.65 geometry.

Seed is fixed before any production draw; mechanical qualification never uses
these thresholds. This driver cannot bypass the recorded qualification gates.
"""
from pathlib import Path
import sys,json,time,os
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from three_particle_implicit_run import advance
from three_particle_source_window import advance_source_window,DESCENDANT_CROSSING_TOLERANCE_S
from three_particle_buffered_event_probe import BufferedContactEvent
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_event import update_ownership,save_event_checkpoint,ownership_pair_step
from pf_sintering.three_particle_renewal import RootClocks,locate_first_root
from pf_sintering.three_particle_geometry import topology_status,grain_volumes
from pf_sintering.three_particle_diagnostics import diagnostics,curvature_watch
from pf_sintering.pr_avalanche import AvalancheController,DescendantBarrier
from pf_sintering.exp_barrier_nucleation import CompleteExpFloorParams
D=Path('docs/three_particle/production_065')
PRODUCTION_SEED=20260910


def require_qualification():
    gate=json.loads((D/'qualification.json').read_text())
    if not all(gate.get(k) is True for k in ['reload_qualified','one_b_qualified','descendant_qualified','phase_b_enabled']):
        raise RuntimeError('Phase B DISABLED: numerical, one-b and descendant qualification required')
    return gate


def main():
    gate=require_qualification()
    event_class=BufferedContactEvent if gate.get('event_engine')=='buffered_native' else ContactEvent
    out=Path('runs/three_particle_production_065/stochastic_seed20260910');out.mkdir(exist_ok=False)
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9)
    source=Path('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999/post_transient.npz')
    with np.load(source) as data:f=data['f'].copy();g['ownership']=data['ownership'].copy();t=float(data['t_model'])*MANIFEST['seconds_per_model_time']
    start_t=t;state=(f,*(g['ownership']*f[None]));reference=ContactEvent(g);reference.metrics(state,0.)
    op=reference.op;it=HarmonicSurfaceDiffusion(op);rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
    clocks=RootClocks(np.random.default_rng(PRODUCTION_SEED));p=MANIFEST['root_barrier_slice']
    barrier=DescendantBarrier(CompleteExpFloorParams(p['G0_eV'],p['Gfloor_eV'],p['a'],p['sigmahat_Pa'],p['n']),1.5)
    avalanche=AvalancheController(barrier=barrier,temperature_K=MANIFEST['temperature_K'],
        attempt_frequency_per_s=MANIFEST['clock_scale']/MANIFEST['seconds_per_model_time'],b_m=MANIFEST['b_event_m'],
        correlation_time_s=.009,rng=clocks.rng,facilitation_decay_alpha=.70)
    records=[];event_number=0;status='RUNNING';wall=time.perf_counter();mass0=float(grain_volumes(f,g).sum())
    (out/'launch.json').write_text(json.dumps(dict(seed=PRODUCTION_SEED,source=str(source),qualification=gate,
        first_root_thresholds=clocks.threshold.copy(),root_hazards=clocks.hazard.copy(),stochastic_result=True),indent=2)+'\n')
    def field_advance(field,seconds):
        current=field.copy();elapsed=0.;h=min(seconds,.1)
        while elapsed<seconds-1e-14:
            h=min(h,seconds-elapsed)
            try:
                trial,error=advance(current,h/MANIFEST['seconds_per_model_time'],it,rule)
                if error>1:raise RuntimeError('embedded field error')
            except (FloatingPointError,RuntimeError):
                h*=.2
                if h<1e-10:raise RuntimeError('reload numerical timestep floor')
                continue
            current=trial;elapsed+=h;h*=min(2.,max(.5,.8/max(error,1e-12)**.5))
        return current
    def rates(field):
        return {k:v['root_rate_per_s'] for k,v in evaluate_contacts(field,op,MANIFEST).items()}
    def record(phase,q=0.):
        reference.bind(state)
        if abs(float(grain_volumes(state[0],g).sum())/mass0-1)>1e-11:raise RuntimeError('trajectory mass guard')
        if state[0].min() < -1e-8 or state[0].max()>1+1e-8:raise RuntimeError('unchanged field guard')
        if np.max(abs(sum(state[1:])-state[0]))>5e-15:raise RuntimeError('ownership closure guard')
        contacts=evaluate_contacts(state[0],op,MANIFEST);scalar,_=diagnostics(state[0],op)
        chain=reference.metrics(state,q)['chain_strain'];areas={k:np.pi*v['r_n_m']**2 for k,v in contacts.items()}
        # Existing contact-side stress fields are retained verbatim for center diagnostics.
        row=dict(time_s=t,phase=phase,event_number=event_number,avalanche_id=clocks.avalanche_id,
            contact=clocks.active,q_over_b=q,chain_strain=chain,contacts=contacts,
            center_volume_m3=float(grain_volumes(state[0],g)[1]),
            center_particle_mean_local_Pa=.5*(contacts['LEFT']['sigma_local_positive_Pa']+contacts['RIGHT']['sigma_local_negative_Pa']),
            cluster_area_weighted_local_Pa=sum(areas[k]*contacts[k]['sigma_local_Pa'] for k in areas)/sum(areas.values()),
            hazards=clocks.hazard.copy(),thresholds=clocks.threshold.copy(),
            H_over_Hstar={k:clocks.hazard[k]/clocks.threshold[k] for k in clocks.hazard},diagnostics=scalar)
        if phase in ['POST_TRANSIENT_NEW_TRAJECTORY','ONE_B_COMPLETE','ONE_B_FAILED']:
            row['curvature_watch']=curvature_watch(state[0],op,two_contact_center=True,gb_positions=[contacts[k]['z_TJ_m'] for k in ['LEFT','RIGHT']])
        records.append(row)
        if phase=='ONE_B_COMPLETE':
            from monitor_current_state_transfer_curvature import apply_progressive_flags
            completed=[r for r in records if r['phase']=='ONE_B_COMPLETE' and r['avalanche_id']==clocks.avalanche_id]
            for branch in row['curvature_watch']:
                history=[dict(avalanche_id=r['avalanche_id'],artifact_flag=0,**r['curvature_watch'][branch]) for r in completed]
                apply_progressive_flags(history)
                if history[-1]['artifact_flag']:raise RuntimeError('progressive curvature artifact: '+branch)
            if reference.metrics(state,q)['topology_stop']:raise RuntimeError('post-event topology terminal')
    def save():
        temp=out/'trajectory.writing.npz'
        np.savez_compressed(temp,fields=np.array(state),ownership=g['ownership'],gb=g['gb'],time_s=t,
            metadata=json.dumps(dict(status=status,root=clocks.snapshot(),avalanche=vars(avalanche.state),event_number=event_number)))
        os.replace(temp,out/'trajectory.npz')
        (out/'history.json').write_text(json.dumps(records,indent=2,default=float)+'\n')
    record('POST_TRANSIENT_NEW_TRAJECTORY');save()
    try:
        while t<start_t+20:
            update_ownership(op,g['ownership']);step=min(.2,start_t+20-t)
            fn,elapsed,increment,contact=locate_first_root(state[0],step,field_advance,rates,clocks,MANIFEST['passive_crossing_tolerance_model']*MANIFEST['seconds_per_model_time'])
            t+=elapsed;state=(fn,*(g['ownership']*fn[None]));clocks.commit(increment,contact)
            record('ROOT_CROSSING' if contact else 'RELOAD');save()
            if topology_status(fn,g)['stop']:status='PHYSICAL_TOPOLOGY_TERMINAL';break
            if contact is None:continue
            avalanche.start(avalanche_id=clocks.avalanche_id,root_cycle=clocks.avalanche_id,start_time_s=t)
            while avalanche.state.avalanche_active:
                event_number+=1;avalanche.begin_transit();pair=(0,1) if contact=='LEFT' else (1,2);event=event_class(g,pair,max_fast_blocks=gate.get('event_max_fast_blocks',60))
                event_start_t=t
                zero=event.run(state,maximum_accepted_states=0)
                save_event_checkpoint(out/f'event_{event_number}_q0.npz',state,zero[5]['event_restart'],contact=contact,label='GENUINE_STOCHASTIC_EVENT')
                def progress(packet,fields,restart):
                    nonlocal state,t
                    state=fields;t=event_start_t+restart['event_time_model']*MANIFEST['seconds_per_model_time']
                    record('ACTIVE_ONE_B',packet['q_end_over_b']);save()
                    save_event_checkpoint(out/f'event_{event_number}.npz',fields,restart,contact=contact,label='GENUINE_STOCHASTIC_EVENT')
                dq=gate['event_max_increment_over_b']
                result=event.run(state,callback=progress,maximum_step_over_b=dq,initial_step_over_b=dq);state=result[:4];info=result[5];t=event_start_t+info['event_time_model']*MANIFEST['seconds_per_model_time']
                reference.bind(state);record('ONE_B_COMPLETE' if result[4] else 'ONE_B_FAILED',info['event_progress_over_b']);save()
                if not result[4]:raise RuntimeError(info['stop_reason'])
                avalanche.complete_transit(t)
                # The source lives after a complete transit. Evolve current PF
                # through its 9 ms window, localizing any child in the full field.
                window_end=avalanche.state.window_deadline_s
                while t<window_end-1e-14:
                    step=min(MANIFEST['passive_hazard_quadrature_dt_model']*MANIFEST['seconds_per_model_time'],window_end-t)
                    def child_rates(fields):
                        reference.bind(fields)
                        cc=evaluate_contacts(fields[0],op,MANIFEST)[contact]
                        value=avalanche.rate(cc['sigma_local_Pa'],cc['r_n_m'])
                        return {'LEFT':value,'RIGHT':0.}
                    # A temporary clock performs field bisection without drawing RNG.
                    probe=RootClocks.__new__(RootClocks);probe.active=None;probe.hazard={'LEFT':avalanche.state.descendant_hazard,'RIGHT':0.};probe.threshold={'LEFT':avalanche.state.descendant_threshold,'RIGHT':float('inf')}
                    def window_advance(fields,seconds):
                        return advance_source_window(fields,seconds,g,pair,rule,reuse_small_step_preconditioner=True)
                    evolved,elapsed,increment,child=locate_first_root(state,step,window_advance,child_rates,probe,DESCENDANT_CROSSING_TOLERANCE_S)
                    t+=elapsed;state=tuple(evolved);reference.bind(state)
                    if child:avalanche.commit_crossing(crossing_time_s=t)
                    else:
                        avalanche.state.descendant_hazard+=increment['LEFT'];avalanche.state.descendant_total_hazard+=increment['LEFT']
                    record('CHILD_CROSSING' if child else 'FACILITATED_WINDOW');save()
                    if child:break
                if not avalanche.state.window_triggered:avalanche.expire_window(window_end)
                if avalanche.state.S_completed>25:raise RuntimeError('descendant safety cap; avalanche not declared extinct')
            # Freeze the newly evolved ownership; update tracker reference planes.
            cc=evaluate_contacts(state[0],op,MANIFEST);g['gb']=np.array([cc[k]['z_TJ_m'] for k in ['LEFT','RIGHT']])
            clocks.extinct();record('AVALANCHE_EXTINCT_REPINNED');save()
        if status=='RUNNING':status='TWENTY_SECONDS_COMPLETE'
    except (RuntimeError,ValueError,FloatingPointError) as error:
        status='STOPPED: '+str(error)
    save();print(status,t,'wall',time.perf_counter()-wall,flush=True)
if __name__=='__main__':main()
