"""FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION; never a sampled root."""
from pathlib import Path
import sys,json,time,argparse
from functools import partial
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_event import pair_transfer,ownership_pair_step,update_ownership,save_event_checkpoint,normalized_pair_ownership,load_event_checkpoint
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_diagnostics import radius_profile,curvature_watch
from pf_sintering.production_mass_transfer_event import current_state_mass_transfer_event
from pf_sintering.model_time_transport import ModelTimeGBTransport
from pf_sintering.quasistatic_event_continuation import fast_manifold_increment,fast_manifold_converged
from pr_closed_volume_passive_continuation import profile_metrics
D=Path('docs/three_particle/production_065')
MANIFEST=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text())
TOL=dict(affinity_MPa=.03,sigma_local_MPa=.005,sigma_integral_MPa=.005,A1_nm=.001,neck_nm=.001,energy_relative=1.5e-8)

class ContactEvent:
    def __init__(self,g,pair=(0,1),max_fast_blocks=60):
        self.g=g;self.op=PhaseAOperator(g);self.pair=pair;self.name='LEFT' if pair==(0,1) else 'RIGHT'
        self.dt=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4
        self.initial_span=None;self.max_fast_blocks=int(max_fast_blocks)
    def bind(self,state):
        phi=normalized_pair_ownership(state,self.g['ownership'],self.pair)
        update_ownership(self.op,phi)
    def metrics(self,state,q):
        self.bind(state);f=state[0];contacts=evaluate_contacts(f,self.op,MANIFEST);c=contacts[self.name]
        r=radius_profile(f,self.g);valid=np.flatnonzero(np.isfinite(r));z0=self.g['z'][valid[0]];z1=self.g['z'][valid[-1]]
        if self.name=='LEFT':z1=self.g['gb'][1]
        else:z0=self.g['gb'][0]
        geometric=profile_metrics(r,{**self.g,'W':self.op.W},dict(z_raw_min_m=-z0,lam=z1-z0,z1=c['z_TJ_m']))
        eta=np.array(state[1:]);weights=eta*self.g['r_c'][None,None,:]
        centers=np.sum(weights*self.g['z'][None,:,None],axis=(1,2))/np.sum(weights,axis=(1,2));span=centers[2]-centers[0]
        if self.initial_span is None:self.initial_span=span
        volumes=grain_volumes(f,self.g)
        positions=np.array([contacts[k]['z_TJ_m'] for k in ['LEFT','RIGHT']])
        topology=topology_status(f,{**self.g,'gb':positions})
        result=dict(**c,**geometric,r_TJ_m=c['r_n_m'],contact_area_m2=np.pi*c['r_n_m']**2,
            G_phasefield_J=self.op.energy(f),sigma_local_MPa=c['sigma_local_Pa']/1e6,
            sigma_integral_MPa=c['sigma_integral_continuous_Pa']/1e6,
            sigma_integral_continuous_MPa=c['sigma_integral_continuous_Pa']/1e6,
            transport_affinity_MPa=c['transport_affinity_Pa']/1e6,r_neck_nm=geometric['r_neck_smooth_m']*1e9,r_TJ_nm=c['r_n_m']*1e9,
            A1_nm=geometric['A1_magnitude_m']*1e9,chain_span_m=float(span),chain_strain=float(1-span/self.initial_span),
            LEFT_sigma_local_Pa=contacts['LEFT']['sigma_local_Pa'],RIGHT_sigma_local_Pa=contacts['RIGHT']['sigma_local_Pa'],
            LEFT_root_rate_per_s=contacts['LEFT']['root_rate_per_s'],RIGHT_root_rate_per_s=contacts['RIGHT']['root_rate_per_s'],
            q_over_b=q,total_volume_m3=float(volumes.sum()),
            LEFT_z_TJ_m=float(positions[0]),RIGHT_z_TJ_m=float(positions[1]),actual_center_span_m=float(np.diff(positions)[0]),topology_stop=topology['stop'],
            V_left_m3=float(volumes[0]),V_center_m3=float(volumes[1]),V_right_m3=float(volumes[2]),
            center_particle_mean_local_Pa=.5*(contacts['LEFT']['sigma_local_positive_Pa']+contacts['RIGHT']['sigma_local_negative_Pa']))
        for name,contact_record in contacts.items():
            result.update({name+'_'+key:value for key,value in contact_record.items()})
        # Match the live serializer's selected continuous integral stress.
        result['sigma_integral_Pa']=c['sigma_integral_continuous_Pa']
        return result
    def evaluator(self,*state):return self.metrics(state,0.)
    def relax(self,state,q):
        current=tuple(x.copy() for x in state);previous=self.metrics(current,q);consecutive=0
        for block in range(1,self.max_fast_blocks+1):
            for _ in range(10):
                self.bind(current);f=current[0];phi=self.g['ownership'];mu=self.op.potential(f).copy()
                fn=bounded_mobility_update(f,mu,self.op,self.dt)
                if fn.min() < -1e-8 or fn.max()>1+1e-8:raise RuntimeError('fast field bounds')
                pn=ownership_pair_step(phi,fn,self.op,self.pair,self.dt,1.0937500000000001e-25)
                current=(fn,*(pn*fn[None]))
            row=self.metrics(current,q)
            if row['topology_stop']:raise RuntimeError('three-grain topology guard')
            increment=fast_manifold_increment(previous,row)
            consecutive=consecutive+1 if block>=3 and fast_manifold_converged(increment,TOL) else 0
            if consecutive>=3:return current,row,dict(converged=True,iterations=block*10,blocks=block)
            previous=row
        return current,row,dict(converged=False,iterations=self.max_fast_blocks*10,blocks=self.max_fast_blocks)
    def run(self,state,target=1.,restart=None,callback=None,**options):
        first=self.metrics(state if restart is None else restart['base_fields'],0.);m=MANIFEST
        self.bind(state)
        transport=ModelTimeGBTransport(first['r_n_m']/2,1.380649e-23,m['temperature_K'],1e-29,m['b_event_m'],m['D_GB_m2_per_model_time'],0.)
        return current_state_mass_transfer_event(state,{**self.g,'W':self.op.W},{},self.evaluator,transport,target,
            fast_relax_fn=self.relax,state_metrics_fn=self.metrics,event_restart=restart,
            transfer_fn=partial(pair_transfer,pair=self.pair),accepted_progress_callback=callback,**options)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out-name',default='forced_left');ap.add_argument('--resume-event',type=Path);ap.add_argument('--max-fast-blocks',type=int,default=60);ap.add_argument('--max-increment-over-b',type=float,default=.02);args=ap.parse_args()
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9)
    with np.load(args.source) as d:f=d['f'].copy();phi=d['ownership'].copy();source_t=float(d['t_model'])
    if source_t*MANIFEST['seconds_per_model_time']<.39:raise ValueError('requires late post-transient source')
    g['ownership']=phi;state=(f,*(phi*f[None]));event=ContactEvent(g,max_fast_blocks=args.max_fast_blocks);before=event.metrics(state,0.);watch0=curvature_watch(f,event.op,two_contact_center=True)
    restart=None;prior_records=[]
    if args.resume_event:
        state,restart,contact,label=load_event_checkpoint(args.resume_event)
        if contact!='LEFT' or label!='FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION':raise ValueError('wrong forced restart identity')
        before=event.metrics(restart['base_fields'],0.)
        prior=json.loads((args.resume_event.parent/'progress.json').read_text());prior_records=prior['packets']
    out=Path('runs/three_particle_production_065')/args.out_name;out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise RuntimeError('refusing to overwrite forced-event output')
    start=time.perf_counter();records=list(prior_records)
    def callback(packet,fields,restart):
        records.append(packet);q=packet['q_end_over_b'];print('forced q/b',q,'strain',packet['chain_strain'],'sigma',packet['sigma_local_MPa'],'wall',time.perf_counter()-start,flush=True)
        save_event_checkpoint(out/'checkpoint.npz',fields,restart,contact='LEFT',label='FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION')
        (out/'progress.json').write_text(json.dumps(dict(before=before,packets=records,wall_s=time.perf_counter()-start),indent=2,default=float)+'\n')
    result=event.run(state,restart=restart,callback=callback,maximum_step_over_b=args.max_increment_over_b,initial_step_over_b=min(.02,args.max_increment_over_b));final=result[:4];complete=result[4];info=result[5];save_event_checkpoint(out/'final.npz',final,info['event_restart'],contact='LEFT',label='FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION');after=event.metrics(final,info['event_progress_over_b']);watch1=curvature_watch(final[0],event.op,two_contact_center=True)
    report=dict(label='FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION',completed=complete,source=str(args.source),source_time_s=source_t*MANIFEST['seconds_per_model_time'],
        before=before,after=after,event_duration_s=info['event_time_model']*MANIFEST['seconds_per_model_time'],
        stop_reason=info['stop_reason'],stop_detail=info.get('stop_detail'),packets=records,wall_s=time.perf_counter()-start,
        max_fast_blocks=args.max_fast_blocks,max_increment_over_b=args.max_increment_over_b,resume_event=str(args.resume_event) if args.resume_event else None,
        watch_before=watch0,watch_after=watch1,topology=topology_status(final[0],event.g),stochastic_result=False,
        material_relative_error=after['total_volume_m3']/before['total_volume_m3']-1,
        active_stress_change_Pa=after['LEFT_sigma_local_Pa']-before['LEFT_sigma_local_Pa'])
    (D/(args.out_name+'.json')).write_text(json.dumps(report,indent=2,default=float)+'\n');print('completed',complete,report['active_stress_change_Pa'],after['chain_strain'],flush=True)
if __name__=='__main__':main()
