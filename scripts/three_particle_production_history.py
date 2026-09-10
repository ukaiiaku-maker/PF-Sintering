"""Re-evaluate preserved actual unequal PF frames with bicrystal metrology."""
from pathlib import Path
import sys,json,csv,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_cmc_ripening import load_initial
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes
DOC=Path('docs/three_particle/production_screen')

def main():
    manifest=json.loads((DOC/'bicrystal_launch_manifest.json').read_text())
    g,meta=load_initial('unequal');op=PhaseAOperator(g);records={};sources=[]
    entries=json.loads(Path('docs/three_particle/implicit/run_manifest.json').read_text())
    # One coherent coarse-to-refined continuation, exclude alternative timestep histories.
    names=['overlap_v1_unequal','extend_v1_unequal','refine_v2_unequal','target_refined_v2_unequal']
    for e in entries:
        p=Path(e['path'])
        if p.parent.name not in names or p.name not in ['frames.npz','checkpoint.npz']:continue
        assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
        sources.append(e)
        with np.load(p) as d:
            fields=d['f'];times=d['t_model']
            if fields.ndim==2:fields=fields[None];times=np.array([float(times)])
            for f,t in zip(fields,times):
                v=grain_volumes(f,g);record=dict(time_s=float(t*op.physics.seconds_per_model_time),center_volume_m3=float(v[1]))
                for name,contact in evaluate_contacts(f,op,manifest).items():
                    record.update({name+'_'+k:val for k,val in contact.items() if isinstance(val,(int,float))})
                records[float(t)]=record
    # Explicit initial checkpoint is part of the preserved trajectory.
    v=grain_volumes(g['f'],g);row=dict(time_s=0.,center_volume_m3=float(v[1]))
    for name,c in evaluate_contacts(g['f'],op,manifest).items():row.update({name+'_'+k:val for k,val in c.items() if isinstance(val,(int,float))})
    records[0.]=row;rows=[records[t] for t in sorted(records)]
    H={'LEFT':0.,'RIGHT':0.}
    for i,row in enumerate(rows):
        for name in H:
            if i:H[name]+=.5*(row[name+'_root_rate_per_s']+rows[i-1][name+'_root_rate_per_s'])*(row['time_s']-rows[i-1]['time_s'])
            row[name+'_hazard_sparse_quadrature']=H[name]
        areas={name:np.pi*row[name+'_r_n_m']**2 for name in H}
        row['cluster_area_weighted_local_stress_diagnostic_Pa']=sum(areas[name]*row[name+'_sigma_local_Pa'] for name in H)/sum(areas.values())
        row['center_two_contact_local_stress_diagnostic_Pa']=.5*(row['LEFT_sigma_local_positive_Pa']+row['RIGHT_sigma_local_negative_Pa'])
    with (DOC/'preserved_unequal_history.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary=dict(samples=len(rows),sources=sources,initial=rows[0],final=rows[-1],
        probability_any_root_sparse_estimate=1-np.exp(-sum(H.values())/manifest['root_threshold_multiplier']),
        late_loading_start=next(r for r in rows if r['time_s']>=.39884654514071577),
        exact_stochastic_crossing=False,trajectory_changed=False)
    (DOC/'preserved_unequal_history.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('samples',len(rows),'end',rows[-1]['time_s'],'probability',summary['probability_any_root_sparse_estimate'],flush=True)
if __name__=='__main__':main()
