"""Compile actual Phase-A evidence. Unrun scientific gates remain NOT RUN."""
from pathlib import Path
import sys,json,subprocess,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pf_sintering.three_particle_geometry import ThreeParticleConfig,build_three_particle,grain_volumes
from pf_sintering.three_particle_phase_a import PhaseAOperator,initialization_locality_gate
from pf_sintering.three_particle_diagnostics import diagnostics,curvature_watch

def main():
    reports={}
    for ratio in [.7,1.]:
        g=build_three_particle(ThreeParticleConfig(center_ratio=ratio));op=PhaseAOperator(g)
        root=Path(f'runs/three_particle_phase_a/ratio_{ratio:g}')
        with np.load(root/'initialization_candidate.npz',allow_pickle=False) as data:f=data['f'].copy()
        prior=json.loads((root/'initialization_report.json').read_text())
        row,_=diagnostics(f,op);locality=initialization_locality_gate(f,g['f'],g)
        reports[str(ratio)]={'status':'INITIALIZATION_REJECTED','numerical_gates':prior['gates'],'locality':locality,'final':row,'watch':curvature_watch(f,op),'projected_mu_residual':prior['residual'],'volume_error':prior['volume_error']}
        if ratio==.7:
            original=f.copy();v=grain_volumes(f,g);start=time.perf_counter()
            for _ in range(1000):f=op.step(f,4.8828125e-5,surface_enabled=False)
            no_transport=dict(status='PASS_OPERATOR_STATIONARITY_ON_UNQUALIFIED_CANDIDATE',steps=1000,wall_s=time.perf_counter()-start,field_bitwise_equal=bool(np.array_equal(f,original)),grain_volumes_bitwise_equal=bool(np.array_equal(grain_volumes(f,g),v)),canonical_initialization=False)
    report=dict(status='PHASE_A_NOT_QUALIFIED',phase_b_authorized=False,canonical_checkpoint=None,initializations=reports,no_surface_transport_control=no_transport,unequal_size_ripening='NOT_RUN_FROM_CANONICAL_STATE_INITIALIZATION_FAILED',equal_size_null='NOT_RUN_FROM_CANONICAL_STATE_INITIALIZATION_FAILED',ripening_plots_and_movie='NOT_GENERATED_NO_QUALIFIED_TRAJECTORY',benchmark=json.loads(Path('runs/three_particle_phase_a/benchmark/report.json').read_text()))
    Path('docs/three_particle/phase_a_gate_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['initializations','benchmark']},indent=2))
if __name__=='__main__':main()
