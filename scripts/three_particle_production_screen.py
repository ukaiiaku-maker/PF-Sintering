"""Resolved CMC viability screen using unchanged live bicrystal root law."""
from pathlib import Path
import sys,json,time,hashlib,argparse
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf,describe
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.axisym_numba_kernel import flux_kernel,div_and_update_kernel
from three_particle_cmc_cleanup import metrics,check

DOC=Path('docs/three_particle/production_screen');OUT=Path('runs/three_particle_production_screen')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ratio',type=float);ap.add_argument('--report',default='screen.json');args=ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((DOC/'bicrystal_launch_manifest.json').read_text())
    rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
    results=[];start=time.perf_counter()
    # Geometric screen includes smaller ratios with a compensating absolute scale.
    for ratio in ([args.ratio] if args.ratio is not None else [.40,.50,.60,.65,.70,.74,.80]):
        unit,_=compatible_chain(ratio)
        minimum_nm=100*32e-9/(2*unit.half_length)
        scales=sorted(set([round(max(minimum_nm*1.025,50.),3),round(max(minimum_nm*1.3,100.),3)]))
        if ratio==.70:scales=sorted(set(scales+[100.]))
        for scale in scales:
            row=dict(ratio=ratio,outer_radius_nm=scale)
            try:
                c,o=compatible_chain(ratio,scale*1e-9);row['cmc']=describe(c,o,width=4e-9)
                g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);f=g['f'].copy()
                before=metrics(f,g,rule);dt=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4
                faces=[int(np.argmin(abs(g['z']+g['dz']/2-b))) for b in g['gb']]
                for _ in range(200):
                    flux_kernel(f,op.potential(f),g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
                    for j in faces:op.Jz[j]=0.
                    div_and_update_kernel(f,op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],dt,op.out)
                    f=op.out.copy()
                    if not np.isfinite(f).all() or f.min() < -1e-8 or f.max()>1+1e-8:raise FloatingPointError('cleanup field guard')
                changes,gates=check(metrics(f,g,rule),before,g)
                row.update(cleanup_steps=200,cleanup_changes=changes,cleanup_gates=gates,
                           topology=topology_status(f,g),initial=evaluate_contacts(f,op,manifest))
                v0=grain_volumes(f,g);initial=f.copy()
                for _ in range(1000):f=op.step(f,dt)
                elapsed=1000*dt*op.physics.seconds_per_model_time
                row.update(short_time_s=elapsed,final=evaluate_contacts(f,op,manifest),
                    center_relative_change=float(grain_volumes(f,g)[1]/v0[1]-1),
                    mass_relative_error=float(grain_volumes(f,g).sum()/v0.sum()-1),
                    mirror_error=float(np.max(abs(f-f[::-1]))),native_steps=1000)
                row['derivatives']={name:{key:(row['final'][name][key]-row['initial'][name][key])/elapsed
                    for key in ['sigma_local_Pa','sigma_integral_continuous_Pa','root_rate_per_s','transport_affinity_Pa']}
                    for name in ['LEFT','RIGHT']}
                path=OUT/f'ratio_{ratio:.2f}_Ro_{scale:.3f}nm.npz'
                np.savez_compressed(path,f_initial=initial,f_final=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],gb=g['gb'])
                row['checkpoint']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                row['status']='SCREENED' if all(gates.values()) and not row['topology']['stop'] else 'CLEANUP_OR_TOPOLOGY_REJECTED'
            except (ValueError,RuntimeError,FloatingPointError) as e:row.update(status='REJECTED',reason=str(e))
            results.append(row)
            (DOC/args.report).write_text(json.dumps(dict(candidates=results,events_enabled=False,stationary_null_is_gate=False,wall_s=time.perf_counter()-start),indent=2)+'\n')
            print(json.dumps(dict(ratio=ratio,Ro_nm=scale,status=row['status'],sigma_MPa=row.get('initial',{}).get('LEFT',{}).get('sigma_local_Pa',0)/1e6,rate=row.get('initial',{}).get('LEFT',{}).get('root_rate_per_s'),wall_s=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
