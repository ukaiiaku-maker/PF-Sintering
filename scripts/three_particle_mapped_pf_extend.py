#!/usr/bin/env python3
"""Extend the best mapped-PF candidate with events and stochastic draws off."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_sharp_initial import map_sharp_to_pf
from three_particle_implicit_run import advance
from three_particle_forced_event import MANIFEST
from three_particle_mapped_pf_screen import cleanup,design_from_row,measured

ROOT=Path(__file__).resolve().parents[1]


def write_history(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--maximum-seconds",type=float,default=2.)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/"docs/three_particle/initial_state_design/selected_initial_candidates.csv").open()))
    item=next(row for row in rows if row["design_id"]=="Ro150_q0.55_L45W_r0.75_th160_kc+10_ko+10")
    d=design_from_row(item);g=map_sharp_to_pf(d,.5e-9);f,_,_=cleanup(d,g);op=PhaseAOperator(g)
    solver=FullJacobianSurfaceDiffusion(op,reuse_preconditioner=True)
    ref=measured(f,g,op,0.);ref.update(mean_rTJ_nm=.5*(ref["LEFT_rTJ_nm"]+ref["RIGHT_rTJ_nm"]),
        mean_G_root_eV=.5*(ref["LEFT_G_root_eV"]+ref["RIGHT_G_root_eV"]))
    ref.update(LEFT_integrated_hazard=0.,RIGHT_integrated_hazard=0.,survival_probability=1.)
    history=[ref];H={"LEFT":0.,"RIGHT":0.};h=.002;t=0.;rejected=0
    rule=json.loads((ROOT/"docs/three_particle/cmc/angle_calibration.json").read_text())["rule"]
    milestones=[.01,.05,.1,.25,.5,1.,1.5,2.];reported=set();start=time.perf_counter()
    while t<args.maximum_seconds-1e-14:
        step=min(h,args.maximum_seconds-t)
        try:trial,error=advance(f,step/MANIFEST["seconds_per_model_time"],solver,rule)
        except (FloatingPointError,RuntimeError):
            h*=.25;rejected+=1
            if h<1e-5:raise
            continue
        if error>1:
            h*=max(.25,.8/error**.5);rejected+=1;continue
        prior=history[-1];f=trial;t+=step;current=measured(f,g,op,t,ref)
        for name in ("LEFT","RIGHT"):
            H[name]+=.5*step*(prior[f"{name}_root_rate_per_s"]+current[f"{name}_root_rate_per_s"])
            current[f"{name}_integrated_hazard"]=H[name]
        current["survival_probability"]=np.exp(-(H["LEFT"]+H["RIGHT"])/1.25)
        history.append(current);h*=min(1.5,max(.7,.9/max(error,1e-10)**.5))
        crossed=[m for m in milestones if m<=t and m not in reported]
        if crossed:
            reported.update(crossed);write_history(args.out/"extended_history.csv",history)
            np.savez_compressed(args.out/"latest_state.npz",f=f,time_s=t)
            print(json.dumps(dict(time_s=t,sigma_MPa=current["mean_sigma_MPa"],
              survival=current["survival_probability"],center_change=current["center_volume_m3"]/ref["center_volume_m3"]-1)),flush=True)
        peak=max(range(len(history)),key=lambda i:history[i]["mean_sigma_MPa"])
        if t>=.5 and t-history[peak]["time_s"]>=.25 and history[peak]["mean_sigma_MPa"]-current["mean_sigma_MPa"]>=.5:break
    write_history(args.out/"extended_history.csv",history)
    peak=max(range(len(history)),key=lambda i:history[i]["mean_sigma_MPa"]);p=history[peak]
    summary=dict(status="BEST_CANDIDATE_EVENTS_OFF_EXTENSION_COMPLETE",events_enabled=False,
      stochastic_draws=False,design_id=item["design_id"],sigma0_MPa=ref["mean_sigma_MPa"],
      sigma_max_MPa=p["mean_sigma_MPa"],delta_sigma_MPa=p["mean_sigma_MPa"]-ref["mean_sigma_MPa"],
      t_sigma_max_s=p["time_s"],end_time_s=t,maximum_is_endpoint_censored=peak==len(history)-1,
      survival_to_max=p["survival_probability"],delta_ln_Gamma_sites=p["delta_ln_Gamma_sites"],
      delta_ln_Gamma_barrier=p["delta_ln_Gamma_barrier"],delta_ln_Gamma=p["delta_ln_Gamma_total"],
      center_volume_change_at_max=p["center_volume_m3"]/ref["center_volume_m3"]-1,
      rTJ_change_at_max=.5*(p["LEFT_rTJ_nm"]+p["RIGHT_rTJ_nm"])/ref["mean_rTJ_nm"]-1,
      curvature_term_change_at_max_MPa=p["LEFT_curvature_term_MPa"]-ref["LEFT_curvature_term_MPa"],
      TJ_term_change_at_max_MPa=p["LEFT_TJ_term_MPa"]-ref["LEFT_TJ_term_MPa"],
      accepted_steps=len(history)-1,rejected_steps=rejected,wall_s=time.perf_counter()-start)
    (args.out/"extension_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
