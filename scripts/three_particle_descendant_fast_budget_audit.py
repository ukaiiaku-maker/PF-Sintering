"""Retry one rejected descendant trial on an immutable discarded copy."""
from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_buffered_event_probe import BufferedContactEvent
from three_particle_forced_event import MANIFEST
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint,save_event_checkpoint
source=Path('runs/three_particle_production_065/forced_descendant_family/fast_budget_240_stop/event_2_final.npz')
state,restart,contact,label=load_event_checkpoint(source)
assert restart['last_rejection_reasons']==['fast_manifold']
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=BufferedContactEvent(g,max_fast_blocks=1024)
start=time.perf_counter();result=event.run(state,restart=restart,maximum_accepted_states=restart['accepted_steps_total']+1,initial_step_over_b=.0025,maximum_step_over_b=.0025)
info=result[5];accepted=info['event_restart']['accepted_steps_total']-restart['accepted_steps_total']
report=dict(label='OFF_TRAJECTORY_DESCENDANT_FAST_BUDGET_AUDIT',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 source_q_over_b=restart['cumulative_q_m']/MANIFEST['b_event_m'],accepted_increments=accepted,max_fast_blocks=1024,unchanged_convergence_tolerances=True,
 end_q_over_b=info['event_progress_over_b'],stop_reason=info['stop_reason'],stop_detail=info.get('stop_detail'),
 packet=info['packets'][-1] if info['packets'] else None,wall_s=time.perf_counter()-start,production_promoted=False,
 note='The accepted-state cap deliberately stops after one successful increment; this is not a completed one-b event.')
Path('docs/three_particle/production_065/descendant_fast_budget_audit.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
save_event_checkpoint('runs/three_particle_production_065/descendant_fast_budget_audit.npz',result[:4],info['event_restart'],contact=contact,label='OFF_TRAJECTORY_FAST_BUDGET_AUDIT')
print(dict(accepted_increments=accepted,stop_reason=info['stop_reason'],end_q=info['event_progress_over_b'],blocks=report['packet']['fast_relax_blocks'] if report['packet'] else None,wall_s=report['wall_s']),flush=True)
