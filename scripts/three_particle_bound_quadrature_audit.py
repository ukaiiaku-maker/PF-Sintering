"""Diagnose bound leakage and test a continuum-consistent face quadrature."""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import bounded_mobility_update,HarmonicSurfaceDiffusion
from pf_sintering.three_particle_null import native_rhs
from pf_sintering.three_particle_geometry import grain_volumes

D=Path('docs/three_particle/production_065')
def main():
    c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);f=g['f'];mu=op.potential(f).copy()
    old=native_rhs(f,op);new=bounded_mobility_update(f,mu,op,1.)-f
    upper=f==1.;lower=f==0.
    result=dict(energy_has_hard_constraint=False,mobility_has_double_zeros=True,
        original_at_exact_upper_max=float(old[upper].max()),harmonic_at_exact_upper_max=float(new[upper].max()),
        original_at_exact_lower_min=float(old[lower].min()),harmonic_at_exact_lower_min=float(new[lower].min()),
        exact_upper_cells=int(upper.sum()),exact_lower_cells=int(lower.sum()))
    with np.load('runs/three_particle_production_screen/continuation_0.65_119.999/checkpoint.npz') as d:
        terminal=d['f'].copy()
    index=np.unravel_index(np.argmax(terminal),terminal.shape)
    # Off-trajectory boundary-state probe; never an accepted clipped field.
    probe=terminal.copy();probe[index]=1.
    old_probe=native_rhs(probe,op)
    new_probe=bounded_mobility_update(probe,op.potential(probe).copy(),op,1.)-probe
    result['boundary_probe']=dict(index=list(map(int,index)),original_rate=float(old_probe[index]),harmonic_rate=float(new_probe[index]),not_an_accepted_state=True)
    # Small-grid generator consistency and exact endpoint invariant test.
    result['initial_energy_J']=op.energy(f)
    result['finite_energy_outside_interval_J']=op.energy(f+.001)
    (D/'bound_quadrature_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
