"""Discarded late-field temporal stability probe; no symmetry projection."""
from pathlib import Path
import sys,json,hashlib,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from three_particle_implicit_run import advance
source=Path('runs/three_particle_volume_loading_065/checkpoint.npz');D=Path('docs/three_particle/volume_loading_065');rule=json.loads(Path('docs/three_particle/cmc/angle_calibration.json').read_text())['rule']
with np.load(source) as d:initial=d['f'].copy()
rows=[];fields=[]
for parts in [1,2,4]:
 c,o=compatible_chain(.65,119.999*1e-9);g=map_to_pf(c,o,4e-9,.5e-9);op=PhaseAOperator(g);it=HarmonicSurfaceDiffusion(op);f=initial.copy();start=time.perf_counter();errors=[];mirrors=[]
 for _ in range(parts):f,error=advance(f,20/parts,it,rule);errors.append(error);mirrors.append(float(np.max(abs(f-f[::-1]))))
 fields.append(f);rows.append(dict(parts=parts,error_estimates=errors,mirror_errors=mirrors,mirror_amplification=mirrors[-1]/float(np.max(abs(initial-initial[::-1]))),field_change=float(np.max(abs(f-initial))),energy_relative_change=op.energy(f)/op.energy(initial)-1,wall_s=time.perf_counter()-start));print(rows[-1],flush=True)
report=dict(label='DISCARDED_LATE_FIELD_SYMMETRY_STABILITY',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_already_exceeds_old_mirror_guard=True,input_mirror_error=float(np.max(abs(initial-initial[::-1]))),total_model_time=20.,rows=rows,no_accepted_fields_changed=True,no_symmetry_projection=True)
(D/'symmetry_step_audit.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(D/'symmetry_step_audit.npz',fields=np.array(fields))
