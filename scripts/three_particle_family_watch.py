"""Retain immutable quarter-b checkpoints from atomic conditional-family files."""
from pathlib import Path
import argparse,io,json,re,time
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--max-wall-s',type=float,default=86400);args=ap.parse_args()
    start=time.monotonic();seen={};saved={}
    for path in args.run.glob('event_*_q*.npz'):
        match=re.fullmatch(r'event_(\d+)_q([0-9.]+)\.npz',path.name)
        if match:saved[int(match[1])]=max(saved.get(int(match[1]),0.),float(match[2]))
    while time.monotonic()-start<args.max_wall_s:
        for path in args.run.glob('event_*.npz'):
            match=re.fullmatch(r'event_(\d+)\.npz',path.name)
            if not match:continue
            event=int(match[1]);mtime=path.stat().st_mtime_ns
            if seen.get(event)==mtime:continue
            data=path.read_bytes()
            with np.load(io.BytesIO(data)) as record:q=json.loads(str(record['restart_json']))['cumulative_q_m']/.25e-9
            seen[event]=mtime
            if q+1e-12>=saved.get(event,0.)+.25:
                dest=args.run/f'event_{event}_q{q:.6f}.npz'
                if not dest.exists():dest.write_bytes(data)
                saved[event]=q;print('retained',dest.name,flush=True)
        checkpoint=args.run/'family.npz'
        if checkpoint.exists():
            with np.load(io.BytesIO(checkpoint.read_bytes())) as record:status=json.loads(str(record['metadata']))['status']
            if status!='RUNNING':return
        time.sleep(5)
if __name__=='__main__':main()
