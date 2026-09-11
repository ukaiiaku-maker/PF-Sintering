"""Review a saved complete conditional descendant without advancing its family."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from monitor_current_state_transfer_curvature import apply_progressive_flags
root=Path('runs/three_particle_production_065/forced_descendant_family')
docs=Path('docs/three_particle/production_065')
history=json.loads((root/'history.json').read_text())
before=next(r for r in history if r['phase']=='CHILD_CROSSING')
after=next(r for r in history if r['phase']=='ONE_B_COMPLETE')
with np.load(root/'event_2_final.npz') as data:
 fields=data['fields'].copy();restart=json.loads(str(data['restart_json']))
with np.load(root/'family.npz') as data:
 metadata=json.loads(str(data['metadata']));assert np.array_equal(fields,data['fields'])
assert after['q_over_b']==1 and restart['cumulative_q_m']==.25e-9
assert metadata['event_number']==2 and metadata['controller']['state']['S_completed']==2
assert history[-1]['phase']=='SOURCE_WINDOW_OPEN'
a,b=before['metrics'],after['metrics']
profiles=json.loads((root/'family_morphology.json').read_text())['rows'];flags={}
for branch in profiles[0]['watch']:
 rows=[dict(copy.deepcopy(row['watch'][branch]),avalanche_id=1,artifact_flag=0) for row in profiles]
 apply_progressive_flags(rows);flags[branch]=[r['artifact_flag'] for r in rows]
report=dict(status='NOT_QUALIFIED_MORPHOLOGY_REVIEW_REQUIRED',phase_b_enabled=False,
 label='CONDITIONAL_DESCENDANT_AFTER_FORCED_ROOT',genuine_stochastic_result=False,
 descendants_completed=1,completed_events_including_forced_root=2,avalanche_extinct=False,
 q_over_b=1.,accepted_steps=restart['accepted_steps_total'],duration_s=after['time_since_forced_root_s']-before['time_since_forced_root_s'],
 geometric_chain_strain_increment=b['chain_strain']-a['chain_strain'],geometric_chain_strain_total=b['chain_strain'],
 production_quota_strain_increment=json.loads((root/'volume_strain_analysis.json').read_text())['production_densification_strain_change'],
 center_relative_volume_change=b['V_center_m3']/a['V_center_m3']-1,
 relative_PF_energy_change=b['G_phasefield_J']/a['G_phasefield_J']-1,
 relative_material_error=b['total_volume_m3']/a['total_volume_m3']-1,
 f_min=float(fields[0].min()),f_max=float(fields[0].max()),ownership_closure=float(np.max(abs(fields[1:].sum(axis=0)-fields[0]))),
 LEFT_stress_before_Pa=a['LEFT_sigma_local_Pa'],LEFT_stress_after_Pa=b['LEFT_sigma_local_Pa'],RIGHT_stress_after_Pa=b['RIGHT_sigma_local_Pa'],
 transport_affinity_after_Pa=b['transport_affinity_Pa'],actual_center_span_m=b['actual_center_span_m'],topology_stop=b['topology_stop'],
 completed_event_monitor=dict(completed_profiles=2,three_completed_event_condition_evaluated=False,reason='There are only the forced root and one completed descendant; no three-completed-event failure is claimed.'),
 within_event_analogue=dict(quotas=[r['q_over_b'] for r in profiles],flags=flags),
 morphology_assessment='The partial-profile growth warning persists through q=1. The moving outside local peak reaches 3.7921W with amplitude 2.25899e15 /m2 and prominence 5.52454e14 /m2. Its narrow-window absence is not disappearance. This does not prove a numerical artifact, but available audits do not establish artifact-free repeated-event behavior. The local timestep overlap does not resolve full-event spatial morphology; the stricter fixed-quota tolerance audit did not converge. No criterion is relaxed to grant qualification.',
 source_window=metadata['controller']['state'],rng=metadata['rng'],
 checkpoint_sha256=hashlib.sha256((root/'event_2_final.npz').read_bytes()).hexdigest(),
 operational_status='Experimental worker paused at saved source-window opening for morphology review; raw checkpoint status RUNNING predates the operating-system pause. No second descendant quota has been accepted.')
(docs/'completed_descendant_review.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k not in ['source_window','rng','within_event_analogue','morphology_assessment']})
