"""Long-time paired current-field analysis with fixed null-transient criterion."""
from pathlib import Path
import sys,json,csv,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from three_particle_cmc_ripening import load_initial
from pf_sintering.three_particle_phase_a import PhaseAOperator
from three_particle_implicit_run import record

ROOT=Path('runs/three_particle_implicit');OUT=Path('docs/three_particle/implicit')
STAGES=['overlap_v1','extend_v1','late_v2','target_v2']
SCALE=.01557994316955921

def history(case,stages=STAGES):
    rows=[];reports=[]
    for stage in stages:
        path=ROOT/f'{stage}_{case}'
        with (path/'history.csv').open() as stream:
            for row in csv.DictReader(stream):
                r={k:float(v) for k,v in row.items() if v not in ['True','False']}
                if rows and r['t_model']<=rows[-1]['t_model']+1e-13:continue
                rows.append(r)
        reports.append(json.loads((path/'report.json').read_text()))
    return {k:np.array([r[k] for r in rows]) for k in rows[0]},reports


def at(data,key,t):return np.interp(t,data['t_model'],data[key])


def null_plateau(data):
    # Fixed protocol, applied on a fixed doubling-time lattice (model time).
    bounds=.1*2.**np.arange(20);bounds=bounds[bounds<=data['t_model'][-1]]
    tolerances={'CC_stress_Pa':50000.,'PF_geometric_stress_Pa':50000.,'psi_deg':.05,'kappa_per_m':50000.}
    intervals=[];consecutive=0;start=None
    for a,b in zip(bounds[:-1],bounds[1:]):
        changes={key:max(abs(at(data,side+'_'+key,b)-at(data,side+'_'+key,a)) for side in ['LEFT','RIGHT']) for key in tolerances}
        passed=all(changes[k]<v for k,v in tolerances.items());consecutive=consecutive+1 if passed else 0
        intervals.append(dict(start_model=float(a),end_model=float(b),changes=changes,passed=bool(passed)))
        if consecutive>=2 and start is None:start=float(b)
    return dict(t_start_model=start,t_start_s=None if start is None else start*SCALE,limits=tolerances,intervals=intervals)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data={};reports={}
    for case in ['unequal','null']:data[case],reports[case]=history(case)
    gate=null_plateau(data['null']);t0=gate['t_start_model'];end=min(d['t_model'][-1] for d in data.values())
    refined={}
    for case,stages in [('unequal',['refine_v2']),('null',['refine_v2','refine_target_v2'])]:
        if all((ROOT/f'{stage}_{case}'/'report.json').exists() for stage in stages):
            rr,dd=history(case,stages)
            if all(d['status']!='running' for d in dd):refined[case]=rr
    if len(refined)==2:end=min(end,*(d['t_model'][-1] for d in refined.values()))
    result=dict(null_transient=gate,common_end_model=float(end),common_end_s=float(end*SCALE),phase_b_authorized=False,full_phase_a_qualified=False)
    result['cases']={}
    if len(refined)==2 and t0 is not None:
        result['refinement']={'null_plateau':null_plateau(refined['null']),'cases':{}}
        for case,d in data.items():
            rd=refined[case]
            # Compare at actual coarse observations. Interpolating the sparse
            # coarse history onto a uniform grid adds sampling error that is
            # larger than the integration error for tiny volume changes.
            tt=d['t_model'][(d['t_model']>=t0)&(d['t_model']<=end)]
            keys=['LEFT_CC_stress_Pa','RIGHT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa','RIGHT_PF_geometric_stress_Pa','LEFT_kappa_per_m','RIGHT_kappa_per_m','LEFT_neck_r_m','RIGHT_neck_r_m','delta_mu_projected_Pa']
            error={k:float(np.max(abs(at(d,k,tt)-at(rd,k,tt)))) for k in keys}
            for grain in ['left','center','right']:
                k='V_'+grain+'_m3';error[k+'_relative']=float(np.max(abs(at(d,k,tt)-at(rd,k,tt)))/d[k][0])
            limits={k:(20000. if 'stress' in k or 'kappa' in k else 2e-12 if 'neck' in k else 5000.) for k in keys}
            limits.update({'V_'+grain+'_m3_relative':1e-7 for grain in ['left','center','right']})
            uniform_t=np.linspace(t0,end,401)
            uniform_volume_error=float(np.max(abs(at(d,'V_center_m3',uniform_t)-at(rd,'V_center_m3',uniform_t)))/d['V_center_m3'][0])
            result['refinement']['cases'][case]=dict(comparison_times_model=tt.tolist(),uniform_grid_linear_interpolation_center_volume_error_relative=uniform_volume_error,maximum_post_transient_metric_errors=error,limits=limits,pass_check=all(error[k]<=limits[k] for k in error))
    for case,d in data.items():
        r={}
        r['center_volume_change_relative']=float(at(d,'V_center_m3',end)/d['V_center_m3'][0]-1)
        r['grain_volume_changes_m3']=[float(at(d,'V_'+grain+'_m3',end)-d['V_'+grain+'_m3'][0]) for grain in ['left','center','right']]
        r['total_volume_max_relative_error']=float(np.max(abs(d['total_volume_m3']/d['total_volume_m3'][0]-1)))
        r['maximum_mirror_error']=float(max(d['mirror_error']))
        r['minimum_center_GB_separation_over_W']=float(min(d['center_GB_separation_over_W']))
        r['minimum_axis_core_f']=float(min(d['center_solid_core_f']))
        r['termination_status']=reports[case][-1]['status']
        r['last_accepted_model_time']=reports[case][-1]['t_model']
        r['coarse_total_wall_s']=sum(x['wall_s'] for x in reports[case])
        native=json.loads(Path(f'runs/three_particle_cmc/ripening_{case}/report.json').read_text())
        r['equivalent_native_steps']=reports[case][-1]['t_model']/native['initial_checkpoint']['dt_model']
        r['estimated_native_wall_s']=r['equivalent_native_steps']*native['wall_s']/native['steps']
        r['estimated_speedup']=r['estimated_native_wall_s']/r['coarse_total_wall_s']
        r['accepted_steps']=sum(x['accepted'] for x in reports[case]);r['rejected_steps']=sum(x['rejected'] for x in reports[case])
        if t0 is not None and t0<end:
            r['post_transient']={}
            for key in ['LEFT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa','LEFT_kappa_per_m','LEFT_neck_r_m','delta_mu_projected_Pa','V_center_m3']:
                tt=np.linspace(t0,end,201);yy=at(d,key,tt);slope=np.polyfit(tt*SCALE,yy,1)[0]
                edges=np.linspace(t0,end,4);subslopes=[]
                for a,b in zip(edges[:-1],edges[1:]):
                    xx=np.linspace(a,b,51);subslopes.append(float(np.polyfit(xx*SCALE,at(d,key,xx),1)[0]))
                r['post_transient'][key]=dict(start=float(yy[0]),end=float(yy[-1]),change=float(yy[-1]-yy[0]),slope_per_s=float(slope),three_window_slopes_per_s=subslopes)
        result['cases'][case]=r
    if t0 is not None and t0<end:
        result['paired_stress_increment_difference_Pa']={key:result['cases']['unequal']['post_transient'][key]['change']-result['cases']['null']['post_transient'][key]['change'] for key in ['LEFT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa']}
    result['status']='LONG_TIME_COMPARISON_COMPLETE' if t0 is not None and t0<end else 'NULL_TRANSIENT_NOT_YET_QUALIFIED'
    # Exact-time endpoint comparisons independently avoid temporal interpolation.
    result['exact_time_refinement_checks']={}
    for case,coarse_stage,refined_stage in [('unequal','target_v2','refine_v2'),('null','exact_refinement_check','refine_target_v2')]:
        cp=ROOT/f'{coarse_stage}_{case}'/'report.json';rp=ROOT/f'{refined_stage}_{case}'/'report.json'
        if not cp.exists() or not rp.exists():continue
        c=json.loads(cp.read_text());r=json.loads(rp.read_text())
        if abs(c['t_model']-r['t_model'])>1e-10:continue
        error={k:float(abs(c['final'][k]-r['final'][k])) for k in ['LEFT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa','LEFT_kappa_per_m','LEFT_neck_r_m','delta_mu_projected_Pa']}
        error['V_center_relative']=abs(c['final']['V_center_m3']-r['final']['V_center_m3'])/data[case]['V_center_m3'][0]
        result['exact_time_refinement_checks'][case]=dict(t_model=c['t_model'],errors=error)
    extra=ROOT/'target_refined_v2_unequal'/'report.json'
    if extra.exists():
        extra_report=json.loads(extra.read_text())
        result['unequal_refined_target']=dict(status=extra_report['status'],t_model=extra_report['t_model'],t_s=extra_report['t_s'],center_volume_change_relative=extra_report['grain_volume_relative_changes'][1],wall_s=extra_report['wall_s'])
    (OUT/'long_time_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    colors={'unequal':'#166a9c','null':'#b5651d'}
    figures=[]
    def page(title,fields,labels,scales=None,log=False,post=False):
        fig,axes=plt.subplots(len(fields),1,figsize=(9,3.0*len(fields)),squeeze=False)
        for ax,key,label,scale in zip(axes[:,0],fields,labels,scales or [1]*len(fields)):
            for case,d in data.items():
                t=d['t_model'];y=d[key]*scale
                if key.startswith('V_'):y=(d[key]/d[key][0]-1)*scale
                if post and t0 is not None:
                    tt=np.linspace(t0,end,201);y=(at(d,key,tt)-at(d,key,t0))*scale;t=tt
                ax.plot(t*SCALE,y,color=colors[case],label=case)
            if t0 is not None and not post:ax.axvline(t0*SCALE,color='gray',ls=':',label='analysis start')
            if log:ax.set_xscale('symlog',linthresh=2e-4)
            ax.set(xlabel='Physical time (s)',ylabel=label);ax.grid(alpha=.2);ax.legend()
        fig.suptitle(title);fig.tight_layout();figures.append(fig)
    page('Current-field ripening and its evolving control',['V_center_m3','delta_mu_projected_Pa'],['Center-volume change (%)','Projected μC − μouter (MPa)'],[100,1e-6],log=True)
    page('Both transient regimes must be excluded',['LEFT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa','LEFT_psi_deg'],['CC stress (MPa)','PF geometric stress (MPa)','Frozen-estimator angle (degrees)'],[1e-6,1e-6,1],log=True)
    if t0 is not None and t0<end:
        page('Stress increments after the null plateau criterion',['LEFT_CC_stress_Pa','LEFT_PF_geometric_stress_Pa'],['ΔCC stress (MPa)','ΔPF stress (MPa)'],[1e-6,1e-6],post=True)
    page('Current contact geometry and resolution',['LEFT_kappa_per_m','LEFT_neck_r_m','center_GB_separation_over_W'],['TJ K (1/nm)','Contact radius (nm)','Fixed GB separation / W'],[1e-9,1e9,1],log=True)
    with PdfPages(OUT/'long_time_plots.pdf') as pdf:
        for i,fig in enumerate(figures):
            pdf.savefig(fig);fig.savefig(OUT/f'long_time_{i+1}.png',dpi=150);plt.close(fig)
    # Commit compact numeric histories and metadata; full fields remain in runs.
    for case,d in data.items():
        with (OUT/f'history_{case}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream,lineterminator="\n");writer.writerow(d);writer.writerows(zip(*d.values()))
    for case,d in refined.items():
        with (OUT/f'history_refined_{case}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream,lineterminator="\n");writer.writerow(d);writer.writerows(zip(*d.values()))
    manifest=[]
    for case in data:
        for stage in STAGES+['refine_v2','refine_target_v2','exact_refinement_check','target_refined_v2']:
            root=ROOT/f'{stage}_{case}'
            if not root.exists():continue
            for name in ['checkpoint.npz','report.json','history.csv','frames.npz']:
                p=root/name;manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
