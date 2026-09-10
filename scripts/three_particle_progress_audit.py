"""Sparse read-only snapshots of the ongoing forced mechanical test."""
from pathlib import Path
import sys,json,argparse
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
from three_particle_forced_event import ContactEvent
from pf_sintering.three_particle_cmc import compatible_chain,map_to_pf
from pf_sintering.three_particle_diagnostics import curvature_watch
from pf_sintering.three_particle_geometry import topology_status
c,o=compatible_chain(.65,119.999e-9);g=map_to_pf(c,o,4e-9,.5e-9);event=ContactEvent(g)
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=Path('runs/three_particle_production_065/forced_left/checkpoint.npz'));args=ap.parse_args();source=args.checkpoint
with np.load(source) as d:
    if 'fields' in d:
        state=tuple(d['fields']);q=json.loads(str(d['restart_json']))['cumulative_q_m']/.25e-9 if 'restart_json' in d else float(d['q'])
        if 'base_fields' in d:event.metrics(tuple(d['base_fields']),0.)
    else:state=(d['f'].copy(),*d['eta'].copy());q=float(d['q'])
row=event.metrics(state,q);positions=[row['LEFT_z_TJ_m'],row['RIGHT_z_TJ_m']];watch=curvature_watch(state[0],event.op,two_contact_center=True,gb_positions=positions)
out=source.parent/f'watch_q{q:.6f}.npz'
if not out.exists():np.savez_compressed(out,fields=np.array(state),q=q)
Path(f'docs/three_particle/production_065/watch_q{q:.6f}.json').write_text(json.dumps(dict(q=q,metrics=row,watch=watch,topology=topology_status(state[0],{**g,'gb':np.array(positions)})),indent=2,default=float)+'\n')
print(q,row['LEFT_sigma_local_Pa'],flush=True)
