"""Discarded-copy event-increment overlap at a live saved state."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),
              str(Path(__file__).resolve().parent)]
import numpy as np

from three_particle_buffered_event_probe import BufferedContactEvent
from three_particle_forced_event import MANIFEST, TOL
from pf_sintering.three_particle_cmc import compatible_chain, map_to_pf
from pf_sintering.three_particle_event import load_event_checkpoint


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True); args=parser.parse_args()
    state,restart,contact,label=load_event_checkpoint(args.source)
    if label!="GENUINE_STOCHASTIC_EVENT": raise ValueError("requires genuine saved event")
    pair=(0,1) if contact=="LEFT" else (1,2); rows=[]; fields=[]; wall=time.perf_counter()
    for increment in (.005,.0025):
        c,o=compatible_chain(.65,119.999*1e-9); g=map_to_pf(c,o,4e-9,.5e-9)
        event=BufferedContactEvent(g,pair,max_fast_blocks=512)
        result=event.run(state,target=.01,initial_step_over_b=increment,
                         minimum_step_over_b=increment,maximum_step_over_b=increment)
        metric=event.metrics(result[:4],result[5]["event_progress_over_b"])
        rows.append(dict(increment_over_b=increment,completed=result[4],
                         stop_reason=result[5]["stop_reason"],metrics=metric,
                         clock_s=result[5]["event_time_model"]*MANIFEST["seconds_per_model_time"]))
        fields.append(np.array(result[:4])); print("DONE",increment,flush=True)
    candidate,reference=rows; candidate["field_linf_to_reference"]=float(np.max(np.abs(fields[0]-fields[1])))
    candidate["stress_difference_Pa"]=candidate["metrics"][contact+"_sigma_local_Pa"]-reference["metrics"][contact+"_sigma_local_Pa"]
    candidate["affinity_difference_Pa"]=(candidate["metrics"]["transport_affinity_MPa"]-reference["metrics"]["transport_affinity_MPa"])*1e6
    candidate["neck_difference_nm"]=candidate["metrics"]["r_neck_nm"]-reference["metrics"]["r_neck_nm"]
    candidate["relative_clock_difference"]=candidate["clock_s"]/reference["clock_s"]-1
    passed=bool(candidate["completed"] and reference["completed"] and
                candidate["field_linf_to_reference"]<1e-5 and
                abs(candidate["stress_difference_Pa"])<TOL["sigma_local_MPa"]*1e6 and
                abs(candidate["affinity_difference_Pa"])<TOL["affinity_MPa"]*1e6 and
                abs(candidate["neck_difference_nm"])<TOL["neck_nm"] and
                abs(candidate["relative_clock_difference"])<1e-4)
    report=dict(label="LIVE_DISCARDED_COPY_EVENT_INCREMENT_OVERLAP",
                source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                source_q_over_b=restart["cumulative_q_m"]/MANIFEST["b_event_m"],
                total_transfer_over_b=.01,contact=contact,rows=rows,passed=passed,
                production_state_modified=False,unchanged_physics_and_fast_tolerances=True,
                wall_s=time.perf_counter()-wall)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
