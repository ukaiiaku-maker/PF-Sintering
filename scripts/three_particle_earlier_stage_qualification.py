#!/usr/bin/env python3
"""Short events-off numerical qualification of the preselected mapped state."""
from pathlib import Path
import csv,json,sys,time
import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_sharp_initial import load_mapped_sharp_state
from three_particle_forced_event import MANIFEST
from three_particle_implicit_run import advance

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/three_particle_earlier_stage_initial_v4/initial_state.npz'
OUT=ROOT/'runs/three_particle_earlier_stage_qualification'
DOC=ROOT/'docs/three_particle/initial_state_design/qualification.json'

def main():
    if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('refusing to overwrite qualification')
    OUT.mkdir(parents=True,exist_ok=True);g,fields,meta=load_mapped_sharp_state(SOURCE)
    f=fields[0];op=PhaseAOperator(g);solver=FullJacobianSurfaceDiffusion(op,reuse_preconditioner=True)
    rule=json.loads((ROOT/'docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
    v0=grain_volumes(f,g);mass0=float(v0.sum());energy0=op.energy(f);rows=[];t=0.;h=.002
    start=time.perf_counter();rejected=0
    def record():
        contacts=evaluate_contacts(f,op,MANIFEST);v=grain_volumes(f,g)
        rows.append(dict(time_s=t,energy_J=op.energy(f),total_mass_relative=float(v.sum()/mass0-1),
          V_left_relative=float(v[0]/v0[0]-1),V_center_relative=float(v[1]/v0[1]-1),
          V_right_relative=float(v[2]/v0[2]-1),f_min=float(f.min()),f_max=float(f.max()),
          mirror_error=float(np.max(abs(f-f[::-1]))),topology_stop=topology_status(f,g)['stop'],
          LEFT_sigma_MPa=contacts['LEFT']['sigma_local_Pa']*1e-6,
          RIGHT_sigma_MPa=contacts['RIGHT']['sigma_local_Pa']*1e-6,
          LEFT_rate_per_s=contacts['LEFT']['root_rate_per_s'],RIGHT_rate_per_s=contacts['RIGHT']['root_rate_per_s'],
          LEFT_rTJ_nm=contacts['LEFT']['r_n_m']*1e9,RIGHT_rTJ_nm=contacts['RIGHT']['r_n_m']*1e9))
    record()
    while t<.01-1e-14:
        step=min(h,.01-t)
        try:trial,error=advance(f,step/MANIFEST['seconds_per_model_time'],solver,rule)
        except (FloatingPointError,RuntimeError):
            h*=.25;rejected+=1
            if h<1e-5:raise
            continue
        if error>1:
            h*=max(.25,.8/error**.5);rejected+=1;continue
        f=trial;t+=step;record();h*=min(1.5,max(.7,.9/max(error,1e-10)**.5))
    energies=np.array([r['energy_J'] for r in rows])
    passed=(np.all(np.diff(energies)<=abs(energy0)*1e-10)
      and max(abs(r['total_mass_relative']) for r in rows)<1e-11
      and max(r['f_max'] for r in rows)<=1+1e-8 and min(r['f_min'] for r in rows)>=-1e-8
      and not any(r['topology_stop'] for r in rows)
      and max(r['mirror_error'] for r in rows)<1e-10)
    with (OUT/'history.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'qualified_state.npz',fields=np.array((f,*(g['ownership']*f[None]))),
      f=f,ownership=g['ownership'],z=g['z'],r_c=g['r_c'],r_f=g['r_f'],gb=g['gb'],dr=g['dr'],dz=g['dz'],
      radii=g['radii'],centers=g['centers'],metadata=json.dumps(meta),t_model=t/MANIFEST['seconds_per_model_time'],
      symmetry_enforcement_enabled=False)
    report=dict(status='SHORT_NUMERICAL_QUALIFICATION_PASS' if passed else 'SHORT_NUMERICAL_QUALIFICATION_FAIL',
      events_enabled=False,stochastic_draws=False,duration_s=t,accepted_steps=len(rows)-1,rejected_steps=rejected,
      energy_relative=energies[-1]/energies[0]-1,maximum_mass_error=max(abs(r['total_mass_relative']) for r in rows),
      maximum_mirror_error=max(r['mirror_error'] for r in rows),initial=rows[0],final=rows[-1],wall_s=time.perf_counter()-start)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');DOC.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not passed:raise RuntimeError(report['status'])

if __name__=='__main__':main()
