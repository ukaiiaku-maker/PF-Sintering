"""Create an ordinary full-domain state from a paired symmetric half state."""
from pathlib import Path
import sys,argparse,hashlib,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_symmetry import reconstruct_full_state_from_paired_halves


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9)
    with np.load(args.source) as d:
        np.testing.assert_array_equal(d['z'],g['z']);np.testing.assert_array_equal(d['r_c'],g['r_c'])
        f0=d['f'].copy();phi0=d['ownership'].copy();t=float(d['t_model']);gb=d['gb'].copy()
    f,phi,state=reconstruct_full_state_from_paired_halves(f0,phi0)
    before_op=PhaseAOperator(dict(g,ownership=phi0));after_g=dict(g,ownership=phi);after_op=PhaseAOperator(after_g)
    manifest=json.loads(Path('docs/three_particle/production_screen/bicrystal_launch_manifest.json').read_text())
    source_mass=float(grain_volumes(f0,g).sum());new_mass=float(grain_volumes(f,after_g).sum())
    if abs(new_mass/source_mass-1)>1e-14:raise RuntimeError('one-time reconstruction changed material')
    if np.max(abs(f-f[::-1]))!=0 or np.max(abs(phi[0]-phi[2,::-1]))!=0 or np.max(abs(phi[1]-phi[1,::-1]))!=0:raise RuntimeError('reconstruction is not exact')
    if np.max(abs(sum(state[1:])-f))>5e-15:raise RuntimeError('grain fields do not close')
    checkpoint=args.out/'source.npz'
    np.savez_compressed(checkpoint,f=f,fields=np.array(state),ownership=phi,z=g['z'],r_c=g['r_c'],gb=gb,t_model=t,next_h=20.,events_enabled=False,symmetry_enforcement_enabled=False)
    report=dict(label='ONE_TIME_HALF_DOMAIN_TO_FULL_DOMAIN_RECONSTRUCTION',source=str(args.source),source_sha256=sha(args.source),checkpoint=str(checkpoint),
        method='pairwise mean of reflected half-domain samples; outer grain labels swapped; explicit full-domain concatenation',ordinary_full_domain=True,one_time_operation=True,
        symmetry_enforcement_after_handoff=False,reflection_guard_after_handoff=False,reflection_projection_after_handoff=False,selected_contact=None,events_enabled=False,thresholds_drawn=False,phase_b_enabled=False,
        source_time_s=t*after_op.physics.seconds_per_model_time,shape=list(f.shape),field_Linf_change=float(np.max(abs(f-f0))),source_mirror_error=float(np.max(abs(f0-f0[::-1]))),reconstructed_mirror_error=float(np.max(abs(f-f[::-1]))),
        ownership_center_mirror_error=float(np.max(abs(phi[1]-phi[1,::-1]))),ownership_outer_swap_error=float(np.max(abs(phi[0]-phi[2,::-1]))),grain_field_closure=float(np.max(abs(sum(state[1:])-f))),
        source_total_volume_m3=source_mass,reconstructed_total_volume_m3=new_mass,material_relative_change=new_mass/source_mass-1,
        source_energy_J=before_op.energy(f0),reconstructed_energy_J=after_op.energy(f),energy_relative_change=after_op.energy(f)/before_op.energy(f0)-1,
        source_volumes_m3=grain_volumes(f0,g).tolist(),reconstructed_volumes_m3=grain_volumes(f,after_g).tolist(),f_min=float(f.min()),f_max=float(f.max()),topology=topology_status(f,after_g),
        contacts=evaluate_contacts(f,after_op,manifest),no_clipping=True,no_fitted_correction=True)
    (args.out/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    report['checkpoint_sha256']=sha(checkpoint);(args.out/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['source_time_s','field_Linf_change','material_relative_change','energy_relative_change','reconstructed_mirror_error']},indent=2),flush=True)


if __name__=='__main__':main()
