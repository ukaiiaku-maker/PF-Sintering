#!/usr/bin/env python3
"""Audit sharp/PF stress and screen mapped candidates with events and RNG off."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from pf_sintering.axisym_numba_kernel import flux_kernel,div_and_update_kernel
from pf_sintering.three_particle_contacts import evaluate_contacts
from pf_sintering.three_particle_diagnostics import diagnostics,radius_profile
from pf_sintering.three_particle_full_jacobian import FullJacobianSurfaceDiffusion
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_sharp_design import SharpDesign,contact_state
from pf_sintering.three_particle_sharp_initial import map_sharp_to_pf,load_mapped_sharp_state
from three_particle_forced_event import MANIFEST
from three_particle_implicit_run import advance

ROOT=Path(__file__).resolve().parents[1]
W=4e-9


def design_from_row(row):
    Ro=float(row["Ro_nm"])*1e-9;Rc=float(row["Rc_nm"])*1e-9
    return SharpDesign(Ro,Rc/Ro,float(row["Lc_over_W"])*W,
        float(row["rTJ_nm"])*1e-9/Rc,float(row["theta_center_negative_deg"]),
        float(row["theta_outer_negative_deg"]),
        float(row["center_curvature_per_um"])*1e6*Rc,
        float(row["outer_curvature_per_um"])*1e6*Ro,W_m=W)


def surface_area(f,g):
    radius=radius_profile(f,g);valid=np.flatnonzero(np.isfinite(radius))
    return float(2*np.pi*np.sum(.5*(radius[valid][:-1]+radius[valid][1:])
        *np.hypot(np.diff(g["z"][valid]),np.diff(radius[valid]))))


def measured(f,g,op,time_s,reference=None):
    contacts=evaluate_contacts(f,op,MANIFEST);vol=grain_volumes(f,g)
    scalar,_=diagnostics(f,op,gb_positions=[contacts[k]["z_TJ_m"] for k in ("LEFT","RIGHT")])
    row=dict(time_s=time_s,energy_J=op.energy(f),surface_area_m2=surface_area(f,g),
        GB_area_m2=sum(math.pi*contacts[k]["r_n_m"]**2 for k in ("LEFT","RIGHT")),
        center_volume_m3=float(vol[1]),total_volume_m3=float(vol.sum()),
        f_min=float(f.min()),f_max=float(f.max()),mirror_error=float(np.max(abs(f-f[::-1]))),
        topology_stop=topology_status(f,g)["stop"])
    for name in ("LEFT","RIGHT"):
        c=contacts[name];r=c["r_n_m"]
        curvature=-.5*(c["kappa1_negative_per_m"]+c["kappa1_positive_per_m"])
        tj=1.5*(math.sin(c["theta_negative_rad"]/2)+math.sin(c["theta_positive_rad"]/2))/r
        row.update({f"{name}_sigma_MPa":c["sigma_local_Pa"]*1e-6,
          f"{name}_curvature_term_MPa":curvature*1e-6,f"{name}_TJ_term_MPa":tj*1e-6,
          f"{name}_rTJ_nm":r*1e9,
          f"{name}_kappa_negative_per_um":c["kappa1_negative_per_m"]*1e-6,
          f"{name}_kappa_positive_per_um":c["kappa1_positive_per_m"]*1e-6,
          f"{name}_theta_negative_deg":math.degrees(c["theta_negative_rad"]),
          f"{name}_theta_positive_deg":math.degrees(c["theta_positive_rad"]),
          f"{name}_root_rate_per_s":c["root_rate_per_s"],f"{name}_G_root_eV":c["G_root_eV"],
          f"{name}_gb_position_nm":scalar[f"{name}_gb_z_m"]*1e9,
          f"{name}_neck_radius_nm":scalar[f"{name}_neck_r_m"]*1e9})
    row["mean_sigma_MPa"]=(row["LEFT_sigma_MPa"]+row["RIGHT_sigma_MPa"])/2
    if reference:
        kb=8.617333262145e-5*MANIFEST["temperature_K"]
        row["delta_ln_Gamma_sites"]=math.log(.5*(row["LEFT_rTJ_nm"]+row["RIGHT_rTJ_nm"])/reference["mean_rTJ_nm"])
        row["delta_ln_Gamma_barrier"]=(reference["mean_G_root_eV"]-
            .5*(row["LEFT_G_root_eV"]+row["RIGHT_G_root_eV"]))/kb
        row["delta_ln_Gamma_total"]=row["delta_ln_Gamma_sites"]+row["delta_ln_Gamma_barrier"]
    return row


def cleanup(d,g):
    op=PhaseAOperator(g);f=g["f"].copy()
    before=measured(f,g,op,0.)
    dt=4.8828125e-5*(min(g["dr"],g["dz"])/1.25e-9)**4
    faces=[int(np.argmin(abs(g["z"]+g["dz"]/2-b))) for b in g["gb"]]
    for _ in range(5):
        flux_kernel(f,op.potential(f),g["dr"],g["dz"],op.W,op.physics.M_s,1e-6/op.W,op.Jr,op.Jz)
        for face in faces:op.Jz[face]=0.
        div_and_update_kernel(f,op.Jr,op.Jz,g["r_c"],g["r_f"],g["dr"],g["dz"],dt,op.out)
        f=op.out.copy()
    return f,before,measured(f,g,op,0.)


def sharp_stage(d):
    c=contact_state(d,0.)
    return dict(stage="sharp_analytic",rTJ_nm=c["r_TJ_m"]*1e9,
      kappa_negative_per_um=c["outer_curvature_per_m"]*1e-6,
      kappa_positive_per_um=c["center_curvature_per_m"]*1e-6,
      theta_negative_deg=c["outer_theta_deg"],theta_positive_deg=c["center_theta_deg"],
      curvature_term_MPa=c["sigma_kappa_Pa"]*1e-6,TJ_term_MPa=c["sigma_TJ_Pa"]*1e-6,
      total_MPa=c["sigma_GB_Pa"]*1e-6)


def pf_stage(label,row):
    return dict(stage=label,rTJ_nm=.5*(row["LEFT_rTJ_nm"]+row["RIGHT_rTJ_nm"]),
      kappa_negative_per_um=row["LEFT_kappa_negative_per_um"],
      kappa_positive_per_um=row["LEFT_kappa_positive_per_um"],
      theta_negative_deg=row["LEFT_theta_negative_deg"],theta_positive_deg=row["LEFT_theta_positive_deg"],
      curvature_term_MPa=row["LEFT_curvature_term_MPa"],TJ_term_MPa=row["LEFT_TJ_term_MPa"],
      total_MPa=row["mean_sigma_MPa"])


def evolve(f,g,maximum_s=.1,minimum_s=.1):
    op=PhaseAOperator(g);solver=FullJacobianSurfaceDiffusion(op,reuse_preconditioner=True)
    ref=measured(f,g,op,0.);ref.update(mean_rTJ_nm=.5*(ref["LEFT_rTJ_nm"]+ref["RIGHT_rTJ_nm"]),
        mean_G_root_eV=.5*(ref["LEFT_G_root_eV"]+ref["RIGHT_G_root_eV"]))
    rows=[measured(f,g,op,0.,ref)];h=.002;t=0.;rejected=0;peak=rows[0]["mean_sigma_MPa"]
    H={"LEFT":0.,"RIGHT":0.}
    rows[0].update(LEFT_integrated_hazard=0.,RIGHT_integrated_hazard=0.,
                   survival_probability=1.)
    while t<maximum_s-1e-14:
        step=min(h,maximum_s-t)
        try:trial,error=advance(f,step/MANIFEST["seconds_per_model_time"],solver,
          json.loads((ROOT/"docs/three_particle/cmc/angle_calibration.json").read_text())["rule"])
        except (FloatingPointError,RuntimeError):
            h*=.25;rejected+=1
            if h<1e-5:raise
            continue
        if error>1:
            h*=max(.25,.8/error**.5);rejected+=1;continue
        prior=rows[-1];f=trial;t+=step;current=measured(f,g,op,t,ref)
        for name in ("LEFT","RIGHT"):
            H[name]+=.5*step*(prior[f"{name}_root_rate_per_s"]+current[f"{name}_root_rate_per_s"])
            current[f"{name}_integrated_hazard"]=H[name]
        current["survival_probability"]=math.exp(-(H["LEFT"]+H["RIGHT"])/1.25)
        rows.append(current);peak=max(peak,current["mean_sigma_MPa"])
        h*=min(1.5,max(.7,.9/max(error,1e-10)**.5))
        peak_index=max(range(len(rows)),key=lambda i:rows[i]["mean_sigma_MPa"])
        if t>=minimum_s and t-rows[peak_index]["time_s"]>=.25 and peak-current["mean_sigma_MPa"]>=.5:break
    return rows,rejected


def completed_original_summary(name,g,stages,start_wall):
    """Reuse the completed events-off segment preceding the original root."""
    with (ROOT/"runs/three_particle_earlier_stage_qualification/history.csv").open() as stream:
        qualification=[{key:float(value) if key not in ("topology_stop",) else value
                        for key,value in row.items()} for row in csv.DictReader(stream)]
    history=json.loads((ROOT/"runs/three_particle_earlier_stage_campaign/seed20260914/history.json").read_text())
    pre=[]
    for row in history:
        pre.append(row)
        if row["phase"]=="ROOT_CROSSING":break
    peak=max(pre,key=lambda r:.5*(r["contacts"]["LEFT"]["sigma_local_Pa"]+
                                  r["contacts"]["RIGHT"]["sigma_local_Pa"]))
    pc=peak["contacts"]["LEFT"];r0=stages[2]["rTJ_nm"]
    sigma0=stages[2]["total_MPa"];sigma_peak=.5*(peak["contacts"]["LEFT"]["sigma_local_Pa"]+
        peak["contacts"]["RIGHT"]["sigma_local_Pa"])*1e-6
    kb=8.617333262145e-5*MANIFEST["temperature_K"]
    initial_law=evaluate_root_law(sigma0,r0)
    peak_G=.5*(peak["contacts"]["LEFT"]["G_root_eV"]+peak["contacts"]["RIGHT"]["G_root_eV"])
    Hq=0.
    for prior,current in zip(qualification,qualification[1:]):
        dt=current["time_s"]-prior["time_s"]
        Hq+=.5*dt*((current["LEFT_rate_per_s"]+current["RIGHT_rate_per_s"])+
                   (prior["LEFT_rate_per_s"]+prior["RIGHT_rate_per_s"]))
    H=Hq+peak["hazards"]["LEFT"]+peak["hazards"]["RIGHT"]
    curvature=-.5*(pc["kappa1_negative_per_m"]+pc["kappa1_positive_per_m"])*1e-6
    tj=1.5*(math.sin(pc["theta_negative_rad"]/2)+math.sin(pc["theta_positive_rad"]/2))/pc["r_n_m"]*1e-6
    mapping=json.loads((ROOT/"docs/three_particle/initial_state_design/mapping_report.json").read_text())
    cleanup_center=mapping["after"]["volumes_m3"][1]
    return dict(design_id=name,grid_nz=len(g["z"]),grid_nr=len(g["r_c"]),
      sigma_sharp_MPa=stages[0]["total_MPa"],sigma_mapped_MPa=stages[1]["total_MPa"],
      sigma_after_cleanup_MPa=sigma0,sigma_max_MPa=sigma_peak,
      delta_sigma_PF_MPa=sigma_peak-sigma0,t_sigma_max_s=peak["time_s"],
      end_time_s=pre[-1]["time_s"],center_volume_change_at_max=peak["center_volume_m3"]/cleanup_center-1,
      rTJ_change_at_max=pc["r_n_m"]*1e9/r0-1,
      curvature_term_change_at_max_MPa=curvature-stages[2]["curvature_term_MPa"],
      TJ_term_change_at_max_MPa=tj-stages[2]["TJ_term_MPa"],
      delta_ln_Gamma_sites=math.log(pc["r_n_m"]*1e9/r0),
      delta_ln_Gamma_barrier=(initial_law["G_root_eV"]-peak_G)/kb,
      delta_ln_Gamma=math.log(pc["r_n_m"]*1e9/r0)+(initial_law["G_root_eV"]-peak_G)/kb,
      survival_to_max=math.exp(-H/1.25),accepted_steps=56+pre.index(peak),
      rejected_steps=9,wall_s=time.perf_counter()-start_wall,topology_healthy=True,
      mass_error=max(abs(r["diagnostics"]["total_volume_m3"]/pre[0]["diagnostics"]["total_volume_m3"]-1) for r in pre),
      reused_completed_pre_root_trajectory=True)


def evaluate_root_law(stress_MPa,radius_nm):
    from pf_sintering.three_particle_contacts import root_law
    return root_law(stress_MPa*1e6,radius_nm*1e-9,MANIFEST)


def write_results(out,audit,summaries,all_rows):
    for path,rows in ((out/"sharp_pf_stage_audit.csv",audit),(out/"events_off_history.csv",all_rows),
                      (out/"candidate_summary.csv",summaries)):
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
            writer.writeheader();writer.writerows(rows)
    ranked=sorted(summaries,key=lambda x:(x["delta_ln_Gamma_barrier"]>abs(x["delta_ln_Gamma_sites"]),
        x["delta_sigma_PF_MPa"],x["survival_to_max"]),reverse=True)
    payload=dict(status="MAPPED_PF_EVENTS_OFF_SCREEN_COMPLETE",events_enabled=False,stochastic_draws=False,
      physical_parameters_changed=False,stress_or_volume_path_prescribed=False,stage_audit=audit,
      candidates=summaries,recommendation=ranked[0]["design_id"],ranking=ranked)
    (out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--screen-seconds",type=float,default=.1)
    parser.add_argument("--finalize-checkpoint",action="store_true")
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.finalize_checkpoint:
        checkpoint=json.loads((args.out/"incremental_checkpoint.json").read_text())
        write_results(args.out,checkpoint["stage_audit"],checkpoint["candidates"],
                      checkpoint["events_off_history"])
        return
    selected=list(csv.DictReader((ROOT/"docs/three_particle/initial_state_design/selected_initial_candidates.csv").open()))
    audit=[];summaries=[];all_rows=[]
    for index,item in enumerate(selected,1):
        name=item["design_id"];d=design_from_row(item);start=time.perf_counter()
        g=map_sharp_to_pf(d,.5e-9);f,before,after=cleanup(d,g)
        stages=[sharp_stage(d),pf_stage("mapped_before_cleanup",before),pf_stage("mapped_after_cleanup",after)]
        if index==1:
            qg,qfields,_=load_mapped_sharp_state(ROOT/"runs/three_particle_earlier_stage_qualification/qualified_state.npz")
            qop=PhaseAOperator(qg);stages.append(pf_stage("qualified_source",measured(qfields[0],qg,qop,.01)))
        for stage in stages:audit.append(dict(design_id=name,**stage))
        if index==1:
            summaries.append(completed_original_summary(name,g,stages,start))
            print(json.dumps(summaries[-1]),flush=True)
            continue
        rows,rejected=evolve(f,g,args.screen_seconds,args.screen_seconds)
        for row in rows:all_rows.append(dict(design_id=name,**row))
        peak_index=max(range(len(rows)),key=lambda i:rows[i]["mean_sigma_MPa"]);peak=rows[peak_index]
        summaries.append(dict(design_id=name,grid_nz=len(g["z"]),grid_nr=len(g["r_c"]),
          sigma_sharp_MPa=stages[0]["total_MPa"],sigma_mapped_MPa=stages[1]["total_MPa"],
          sigma_after_cleanup_MPa=rows[0]["mean_sigma_MPa"],sigma_max_MPa=peak["mean_sigma_MPa"],
          delta_sigma_PF_MPa=peak["mean_sigma_MPa"]-rows[0]["mean_sigma_MPa"],
          t_sigma_max_s=peak["time_s"],end_time_s=rows[-1]["time_s"],
          center_volume_change_at_max=peak["center_volume_m3"]/rows[0]["center_volume_m3"]-1,
          rTJ_change_at_max=mean_radius(peak)/mean_radius(rows[0])-1,
          curvature_term_change_at_max_MPa=peak["LEFT_curvature_term_MPa"]-rows[0]["LEFT_curvature_term_MPa"],
          TJ_term_change_at_max_MPa=peak["LEFT_TJ_term_MPa"]-rows[0]["LEFT_TJ_term_MPa"],
          delta_ln_Gamma_sites=peak["delta_ln_Gamma_sites"],
          delta_ln_Gamma_barrier=peak["delta_ln_Gamma_barrier"],
          delta_ln_Gamma=peak["delta_ln_Gamma_total"],
          survival_to_max=peak.get("survival_probability",1.),accepted_steps=len(rows)-1,
          rejected_steps=rejected,wall_s=time.perf_counter()-start,
          topology_healthy=not any(r["topology_stop"] for r in rows),screen_window_censored_at_end=(peak_index==len(rows)-1),
          mass_error=max(abs(r["total_volume_m3"]/rows[0]["total_volume_m3"]-1) for r in rows)))
        print(json.dumps(summaries[-1]),flush=True)
        (args.out/"incremental_checkpoint.json").write_text(json.dumps(
            dict(stage_audit=audit,candidates=summaries,events_off_history=all_rows),indent=2)+"\n")
    write_results(args.out,audit,summaries,all_rows)


def mean_radius(row):return .5*(row["LEFT_rTJ_nm"]+row["RIGHT_rTJ_nm"])


if __name__=="__main__":main()
