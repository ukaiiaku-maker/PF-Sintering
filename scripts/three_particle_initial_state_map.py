#!/usr/bin/env python3
"""Map, minimally clean, and qualify the preselected earlier-stage seed."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,sys,time

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from pf_sintering.axisym_numba_kernel import flux_kernel,div_and_update_kernel
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_diagnostics import diagnostics
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_sharp_design import SharpDesign
from pf_sintering.three_particle_sharp_initial import map_sharp_to_pf
from three_particle_forced_event import MANIFEST

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/'docs/three_particle/initial_state_design/selection.json'
OUT=ROOT/'runs/three_particle_earlier_stage_initial_v4'
DOC=ROOT/'docs/three_particle/initial_state_design'
STEPS=5


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(f,g,op):
    contacts=evaluate_contacts(f,op,MANIFEST);vol=grain_volumes(f,g)
    scalar,_=diagnostics(f,op,gb_positions=[contacts[k]['z_TJ_m'] for k in ('LEFT','RIGHT')])
    return dict(volumes_m3=vol.tolist(),total_volume_m3=float(vol.sum()),energy_J=op.energy(f),
      f_min=float(f.min()),f_max=float(f.max()),contacts=contacts,topology=topology_status(f,g),
      diagnostics=scalar)


def main():
    if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('refusing to overwrite mapped seed')
    OUT.mkdir(parents=True,exist_ok=True)
    selection=json.loads(DESIGN.read_text());d=SharpDesign(**selection['chosen_design'])
    start=time.perf_counter();g=map_sharp_to_pf(d,.5e-9);op=PhaseAOperator(g);f=g['f'].copy()
    before=metrics(f,g,op);initial_vol=np.array(before['volumes_m3']);initial_energy=before['energy_J']
    dt=4.8828125e-5*(min(g['dr'],g['dz'])/1.25e-9)**4
    faces=[int(np.argmin(abs(g['z']+g['dz']/2-b))) for b in g['gb']]
    history=[]
    for step in range(1,STEPS+1):
        flux_kernel(f,op.potential(f),g['dr'],g['dz'],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
        for face in faces:op.Jz[face]=0.
        div_and_update_kernel(f,op.Jr,op.Jz,g['r_c'],g['r_f'],g['dr'],g['dz'],dt,op.out)
        f=op.out.copy()
        if not np.isfinite(f).all() or f.min() < -1e-8 or f.max()>1+1e-8:
            raise RuntimeError('cleanup field guard')
        if step in (1,5):history.append({'step':step,**metrics(f,g,op)})
    after=metrics(f,g,op);final_vol=np.array(after['volumes_m3'])
    changes=dict(max_grain_volume_relative=float(np.max(abs(final_vol/initial_vol-1))),
      total_volume_relative=after['total_volume_m3']/before['total_volume_m3']-1,
      energy_relative=after['energy_J']/initial_energy-1,
      field_Linf=float(np.max(abs(f-g['f']))),
      max_contact_radius_relative=max(abs(after['contacts'][k]['r_n_m']/before['contacts'][k]['r_n_m']-1) for k in ('LEFT','RIGHT')),
      max_angle_change_deg=max(abs(np.degrees(after['contacts'][k][f'theta_{s}_rad']-before['contacts'][k][f'theta_{s}_rad'])) for k in ('LEFT','RIGHT') for s in ('negative','positive')),
      max_stress_change_MPa=max(abs(after['contacts'][k]['sigma_local_Pa']-before['contacts'][k]['sigma_local_Pa']) for k in ('LEFT','RIGHT'))*1e-6)
    passed=(changes['max_grain_volume_relative']<2e-5 and changes['max_contact_radius_relative']<.005
      and changes['max_angle_change_deg']<1 and changes['max_stress_change_MPa']<.5
      and not after['topology']['stop'])
    metadata=dict(label='EARLIER_STAGE_THREE_PARTICLE_INITIAL_STATE',chosen_before_mapping=True,
      stochastic_seed_drawn=False,cleanup_steps=STEPS,cleanup_dt_model=dt,
      symmetry_enforcement_enabled=False,sharp_design=asdict(d),grid_shape=f.shape,
      width_m=d.W_m,spacing_m=g['dr'],dz_m=g['dz'],gb_m=g['gb'].tolist(),
      actual_initial_volumes_m3=final_vol.tolist(),physical_model_changed=False)
    source=OUT/'initial_state.npz'
    np.savez_compressed(source,fields=np.array((f,*(g['ownership']*f[None]))),f=f,
      ownership=g['ownership'],z=g['z'],r_c=g['r_c'],r_f=g['r_f'],gb=g['gb'],
      dr=g['dr'],dz=g['dz'],radii=g['radii'],centers=g['centers'],metadata=json.dumps(metadata),
      t_model=0.,symmetry_enforcement_enabled=False)
    report=dict(status='INITIAL_STATE_QUALIFIED' if passed else 'INITIAL_STATE_REJECTED',
      source=str(source),source_sha256=sha(source),selection_sha256=sha(DESIGN),before=before,after=after,
      cleanup_changes=changes,cleanup_passed=passed,history=history,wall_s=time.perf_counter()-start,
      no_pf_physical_time_advanced=True,no_stochastic_draw=True)
    (OUT/'mapping_report.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
    (DOC/'mapping_report.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
    print(report['status'],source,report['source_sha256'],changes,flush=True)

if __name__=='__main__':main()
