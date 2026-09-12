"""Preserve coherent milestone/final snapshots and regenerate diagnostic reports.

Read-only toward the live simulation and protected production worktree. Never
launches events, changes physics, signals workers, or commits/pushes files.
"""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--poll-seconds',type=float,default=30.);ap.add_argument('--snapshot-root',type=Path,default=Path('docs/three_particle/volume_loading_065/automatic_snapshots'));args=ap.parse_args()
 os.chdir(ROOT);run=args.run;out=args.snapshot_root;out.mkdir(parents=True,exist_ok=True)
 while True:
  report_bytes=(run/'report.json').read_bytes();report=json.loads(report_bytes);last=report['history'][-1]
  terminal=report['status']!='RUNNING';targets=report['saved_milestones'];tag='final_'+report['status'].lower() if terminal else f'loss_{100*max(targets):g}pct' if targets else None
  if tag is not None and not (out/tag).exists():
   field_bytes=(run/'checkpoint.npz').read_bytes();launch=json.loads((run/'launch.json').read_text())
   with np.load(io.BytesIO(field_bytes)) as d:
    t=float(d['t_model'])*launch['physical_manifest']['seconds_per_model_time']
    coherent=abs(t-last['time_s'])<1e-10
    bounded=np.isfinite(d['f']).all() and d['f'].min()>=-1e-8 and d['f'].max()<=1+1e-8
   if coherent:
    reflection_guard=launch.get('reflection_guard_enabled',True)
    assert bounded and not report['events_enabled'] and not report['stochastic_thresholds_drawn'] and not report['phase_b_enabled']
    temp=out/(tag+'.tmp');temp.mkdir(exist_ok=True);(temp/'checkpoint.npz').write_bytes(field_bytes);(temp/'report.json').write_bytes(report_bytes);shutil.copyfile(run/'launch.json',temp/'launch.json')
    for target in targets:shutil.copyfile(run/f'loss_{100*target:g}pct.npz',temp/f'loss_{100*target:g}pct.npz')
    history=report['history'];validation=dict(status=report['status'],snapshot_time_s=t,center_loss_fraction=last['center_loss_fraction'],source_run=str(run),events_enabled=False,phase_b_enabled=False,no_clipping=True,no_fitted_correction=True,field_min=min(r['f_min'] for r in history),field_max=max(r['f_max'] for r in history),maximum_mass_error=max(abs(r['mass_relative_error']) for r in history),maximum_mirror_error=max(r['mirror_error'] for r in history),energy_monotone=all(b['energy_J']<=a['energy_J']*(1+1e-12) for a,b in zip(history,history[1:])),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in temp.iterdir() if p.is_file()})
    validation.update(reflection_guard_enabled=reflection_guard,reflection_is_diagnostic_only=not reflection_guard)
    assert validation['maximum_mass_error']<=1e-11 and (not reflection_guard or validation['maximum_mirror_error']<=1e-8) and validation['energy_monotone']
    (temp/'verification.json').write_text(json.dumps(validation,indent=2)+'\n');temp.rename(out/tag)
    subprocess.run([sys.executable,'scripts/three_particle_volume_loading_report.py','--run',str(out/tag)],check=True)
    for name in ['analysis.json','loading_vs_volume.pdf','loading_vs_volume.png','loading_diagnostics.pdf','loading_diagnostics.png','loading_decomposition.pdf','loading_decomposition.png']:shutil.copyfile(out.parent/name,out/tag/name)
    print('SNAPSHOT',tag,t,100*last['center_loss_fraction'],flush=True)
  if terminal and (out/tag).exists():print('FINAL_REPORT_READY',str(out/tag),flush=True);return
  time.sleep(max(1.,args.poll_seconds))

if __name__=='__main__':main()
