"""Constrained seed relaxation. Never silently promotes an unconverged state."""
from pathlib import Path
import sys,json,time,argparse
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pf_sintering.three_particle_geometry import ThreeParticleConfig,build_three_particle,grain_volumes
from pf_sintering.three_particle_phase_a import PhaseAOperator,initialize_step
from pf_sintering.three_particle_diagnostics import diagnostics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ratio',type=float,default=.7);ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--resume',action='store_true');ap.add_argument('--dtau',type=float,default=.04);args=ap.parse_args()
    out=Path(f'runs/three_particle_phase_a/ratio_{args.ratio:g}');out.mkdir(parents=True,exist_ok=True)
    g=build_three_particle(ThreeParticleConfig(center_ratio=args.ratio));op=PhaseAOperator(g);f=g['f'];v0=grain_volumes(f,g)
    if args.resume:
        with np.load(out/'initialization_candidate.npz',allow_pickle=False) as data:f=data['f'].copy()
    history=[];start=time.perf_counter()
    for i in range(args.steps):
        f,record=initialize_step(f,op,dtau=args.dtau)
        if i%1000==0 or i==args.steps-1:
            row,_=diagnostics(f,op);row.update(record,iteration=i+1,wall_s=time.perf_counter()-start)
            history.append(row);print(json.dumps({k:row[k] for k in ['iteration','wall_s','energy_J','projected_mu_residual','LEFT_psi_deg','mirror_error']}),flush=True)
    row,profiles=diagnostics(f,op)
    residual=record['projected_mu_residual'];volume_error=float(np.max(abs(grain_volumes(f,g)/v0-1)))
    angle_error=max(abs(row[k+'_psi_deg']-g['config'].psi_deg) for k in ['LEFT','RIGHT'])
    # These are necessary initialization gates, not a complete Phase-A pass.
    gates=dict(projected_residual=residual<1e-5,grain_volume=volume_error<1e-9,mirror=row['mirror_error']<1e-10,angle=angle_error<3,topology=not row['topology_stop'])
    eligible=all(gates.values())
    np.savez_compressed(out/'initialization_candidate.npz',f=f,z=g['z'],r_c=g['r_c'],config_json=json.dumps(asdict(g['config'])),canonical=False)
    report=dict(status='INITIALIZATION_NUMERICAL_GATES_PASS_REVIEW_CURVATURE' if eligible else 'INITIALIZATION_NOT_QUALIFIED',gates=gates,volume_error=volume_error,angle_error_deg=angle_error,residual=residual,wall_s=time.perf_counter()-start,history=history,final=row)
    (out/'initialization_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','final']}),flush=True)
if __name__=='__main__':main()
