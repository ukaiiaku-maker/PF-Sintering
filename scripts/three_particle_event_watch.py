"""Retain sparse accepted event checkpoints without altering the worker."""
from pathlib import Path
import sys,time,json,math,argparse,shutil
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();next_q=None;start=time.monotonic()
while time.monotonic()-start<10800:
    path=args.run/'checkpoint.npz'
    if not path.exists():time.sleep(2);continue
    with np.load(path) as d:
        metadata=json.loads(str(d['restart_json']));q=metadata['cumulative_q_m']/.25e-9
        if next_q is None:next_q=math.ceil((q+1e-12)*10)/10
        if q>=next_q-1e-12 or q>=1-1e-12:
            # Keep the same opened immutable inode; the writer replaces atomically.
            payload={k:d[k].copy() for k in d.files}
            np.savez_compressed(args.run/f'watch_q{q:.6f}.npz',**payload)
            print('saved event watch',q,flush=True);next_q=q+.1
        if q>=1-1e-12:break
    if (args.run/'final.npz').exists():break
    time.sleep(2)
