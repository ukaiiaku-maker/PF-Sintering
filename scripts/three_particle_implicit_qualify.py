"""Quantitative overlap gates, on all eleven native observation times."""
from pathlib import Path
import sys,json,csv
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_cmc_ripening import load_initial
from three_particle_implicit_run import projected_mu,record
from pf_sintering.three_particle_phase_a import PhaseAOperator

# Absolute floors keep an essentially zero null from creating meaningless
# relative errors. These tolerances precede the extended-time experiment.
LIMITS=dict(volume_change_fraction=1e-8,kappa_per_m=20000.,neck_r_m=2e-12,
            CC_stress_Pa=20000.,PF_geometric_stress_Pa=20000.,delta_mu_projected_Pa=5000.,
            field_max=2e-5,total_volume_relative=1e-11,mirror_error=1e-8)

def main():
    results={}
    for case in ['unequal','null']:
        g,m=load_initial(case);op=PhaseAOperator(g)
        native=Path(f'runs/three_particle_cmc/ripening_{case}')
        candidate=Path(f'runs/three_particle_implicit/overlap_v1_{case}')
        with np.load(native/'frames.npz') as d:nf=d['f']
        with np.load(candidate/'frames.npz') as d:af=d['f'];times=d['t_model']
        if len(af)!=len(nf):raise RuntimeError('missing overlap observations')
        metrics={k:[] for k in LIMITS};all_rows=[]
        for i,(a,n,t) in enumerate(zip(af,nf,times)):
            np.testing.assert_allclose(t,i*1000*m['dt_model'],rtol=1e-12,atol=1e-15)
            ra=record(a,op,m,t,i);rn=record(n,op,m,t,i)
            e={}
            e['volume_change_fraction']=max(abs(ra['V_'+grain+'_m3']-rn['V_'+grain+'_m3'])/rn['V_'+grain+'_m3'] for grain in ['left','center','right'])
            for key in ['kappa_per_m','neck_r_m','CC_stress_Pa','PF_geometric_stress_Pa']:
                e[key]=max(abs(ra[side+'_'+key]-rn[side+'_'+key]) for side in ['LEFT','RIGHT'])
            e['delta_mu_projected_Pa']=abs(ra['delta_mu_projected_Pa']-rn['delta_mu_projected_Pa'])
            e['field_max']=float(np.max(abs(a-n)))
            e['total_volume_relative']=abs(ra['total_volume_m3']/rn['total_volume_m3']-1)
            e['mirror_error']=ra['mirror_error']
            for k,v in e.items():metrics[k].append(v)
            all_rows.append(dict(t_model=float(t),errors=e))
        maximum={k:max(v) for k,v in metrics.items()}
        results[case]=dict(pass_overlap=all(maximum[k]<=LIMITS[k] for k in LIMITS),maximum_errors=maximum,observations=all_rows)
    report=dict(status='PASS' if all(v['pass_overlap'] for v in results.values()) else 'FAIL',limits=LIMITS,cases=results,phase_b_authorized=False)
    out=Path('docs/three_particle/implicit');out.mkdir(parents=True,exist_ok=True)
    (out/'overlap.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
