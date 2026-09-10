"""Short native-PF timing probe only; unqualified initial states cannot pass ripening."""
from pathlib import Path
import sys,time,json,resource
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pf_sintering.three_particle_geometry import build_three_particle,grain_volumes
from pf_sintering.three_particle_phase_a import PhaseAOperator

def main():
    out=Path('runs/three_particle_phase_a/benchmark');out.mkdir(parents=True,exist_ok=True)
    g=build_three_particle();op=PhaseAOperator(g);f=g['f'];dt=4.8828125e-5
    op.step(f,dt) # compile/warm cache, excluded from timing
    v0=grain_volumes(f,g);start=time.perf_counter();n=1000
    for _ in range(n):f=op.step(f,dt)
    wall=time.perf_counter()-start;v=grain_volumes(f,g)
    np.savez_compressed(out/'timing_probe.npz',f=f)
    report=dict(status='TIMING_ONLY_UNEQUILIBRATED_SEED',steps=n,dt_model=dt,wall_s=wall,model_increment=n*dt,seconds_per_model_increment=wall/(n*dt),physical_increment_s=n*dt*op.physics.seconds_per_model_time,center_volume_change_m3=float(v[1]-v0[1]),center_fraction_change_per_wall_hour=float((v[1]/v0[1]-1)*3600/wall),total_volume_error=float(v.sum()/v0.sum()-1),mirror_error=float(np.max(abs(f-f[::-1]))),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint_bytes=(out/'timing_probe.npz').stat().st_size,passive_M_s=op.physics.M_s,threads=1,qualification=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
