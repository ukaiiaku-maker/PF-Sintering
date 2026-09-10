"""Fixed-geometry spatial/native comparison of the face quadrature repair."""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes
D=Path('docs/three_particle/production_065')
def main():
    m=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text());c,o=compatible_chain(.65,119.999e-9);rows=[];start=time.perf_counter()
    duration=100*4.8828125e-5*(.5e-9/1.25e-9)**4
    for spacing in [.5e-9,.375e-9,.25e-9]:
        g=map_to_pf(c,o,4e-9,spacing);op=PhaseAOperator(g);old=g['f'].copy();new=old.copy();v0=grain_volumes(old,g);e0=op.energy(old)
        dt=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4;n=int(np.ceil(duration/dt));dt=duration/n
        for _ in range(n):
            old=op.step(old,dt);new=bounded_mobility_update(new,op.potential(new).copy(),op,dt)
        a=evaluate_contacts(old,op,m);b=evaluate_contacts(new,op,m)
        row=dict(spacing_nm=spacing*1e9,steps=n,dt_model=dt,duration_model=duration,
            original=a,harmonic=b,field_difference_linf=float(np.max(abs(old-new))),
            local_stress_difference_Pa=b['LEFT']['sigma_local_Pa']-a['LEFT']['sigma_local_Pa'],
            original_center_relative_change=float(grain_volumes(old,g)[1]/v0[1]-1),
            harmonic_center_relative_change=float(grain_volumes(new,g)[1]/v0[1]-1),
            harmonic_mass_error=float(grain_volumes(new,g).sum()/v0.sum()-1),
            harmonic_energy_change_J=op.energy(new)-e0,f_min=float(new.min()),f_max=float(new.max()))
        rows.append(row);(D/'mobility_spatial_refinement.json').write_text(json.dumps(dict(rows=rows,wall_s=time.perf_counter()-start),indent=2)+'\n')
        print(spacing,n,row['field_difference_linf'],row['local_stress_difference_Pa'],time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
