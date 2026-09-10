"""Independent fixed-step late-field convergence and sparse morphology audit."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_diagnostics import curvature_watch
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from three_particle_implicit_run import advance

def main():
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=HarmonicSurfaceDiffusion(op)
    m=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text())
    rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
    root=Path('runs/three_particle_production_screen/harmonic_continuation_0.65_119.999')
    files=sorted(root.glob('state_*.npz'),key=lambda p:float(p.stem.split('_')[1][:-1]));source=min(files,key=lambda p:abs(float(p.stem.split('_')[1][:-1])-10))
    with np.load(source) as d:f=d['f'].copy();t=float(d['t_model'])*m['seconds_per_model_time']
    fields=[];rows=[];start=time.perf_counter()
    for n in [4,8,16]:
        fn=f.copy();errors=[]
        for _ in range(n):fn,err=advance(fn,.04/n/m['seconds_per_model_time'],it,rule);errors.append(err)
        fields.append(fn);rows.append(dict(steps=n,dt_s=.04/n,max_embedded_error=max(errors),f_min=float(fn.min()),f_max=float(fn.max()),mass_relative_error=float(grain_volumes(fn,g).sum()/grain_volumes(f,g).sum()-1),energy_J=op.energy(fn),contacts=evaluate_contacts(fn,op,m)))
        print('late refinement',n,flush=True)
    for i in [0,1]:
        rows[i]['difference_to_finest_Linf']=float(np.max(abs(fields[i]-fields[-1])))
        rows[i]['LEFT_stress_difference_to_finest_Pa']=rows[i]['contacts']['LEFT']['sigma_local_Pa']-rows[-1]['contacts']['LEFT']['sigma_local_Pa']
    result=dict(source=str(source),source_time_s=t,duration_s=.04,runs=rows,wall_s=time.perf_counter()-start,topology=topology_status(fields[-1],g),curvature_watch=curvature_watch(fields[-1],op,two_contact_center=True))
    Path('docs/three_particle/production_065/late_timestep_refinement.json').write_text(json.dumps(result,indent=2,default=float)+'\n')
if __name__=='__main__':main()
