"""Compare compiled event with an immutable original-kernel accepted state."""
from pathlib import Path
import time,json
import numpy as np
reference=Path('runs/three_particle_production_065/forced_left/watch_q0.275000.npz')
with np.load(reference) as d:expected=d['fields'].copy();q=float(d['q'])
p=Path('runs/three_particle_production_065/forced_left_compiled/checkpoint.npz');start=time.monotonic()
while time.monotonic()-start<900:
    try:
        with np.load(p) as d:actual=d['fields'].copy();r=json.loads(str(d['restart_json']));current=r['cumulative_q_m']/.25e-9
    except (OSError,ValueError):time.sleep(1);continue
    if abs(current-q)<1e-12:
        result=dict(reference=str(reference),q=q,bitwise_identical=bool(np.array_equal(expected,actual)),field_Linf=float(np.max(abs(actual-expected))))
        Path('docs/three_particle/production_065/event_kernel_overlap.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True);break
    if current>q+1e-12:raise RuntimeError('missed reference q; no comparison fabricated')
    time.sleep(1)
else:raise RuntimeError('comparison waiting limit')
