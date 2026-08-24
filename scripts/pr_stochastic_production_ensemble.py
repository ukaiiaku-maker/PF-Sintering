"""Ten-realization stochastic renewal ensemble within the declared PF envelope."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads
from PIL import Image

import pr_coarsening_stochastic_two_event as base
import pr_authoritative_stochastic_ten_event as authoritative
from pf_sintering.axisym_numba_kernel import NumbaScratch, axisym_gb_face_projected_step_fast
from pr_coarsening_driven_fourier_loading import integral,reservoir_remove_particle
from pr_experimental_long_sinkoff import build_case
from pr_full_deterministic_cycle import make_transport
from pr_tj_node_coupling_gate import make_evaluator


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/pr_current_head_regression/coarsening_driven_fourier/production_ensemble_10x5_courant_verified"
PRE_VERIFICATION_OUT=ROOT/"runs/pr_current_head_regression/coarsening_driven_fourier/production_ensemble_10x5"
N_REALIZATIONS=10; MAX_EVENTS=5; MAX_WORKERS=4
V_CUTOFF=.88; RN_CUTOFF_M=35e-9; SAMPLE_DT=.25
COMPETING_CROSSING_DT=.005
CLOCK_SCALE=15579943169.55921
PRODUCTION_C4=0.05


class LeanOutput:
    def __init__(self,out,geom,render=False):
        self.out=out;self.geom=geom;self.render=render;self.rows=[];self.contours=[];self.frames=[]
        (out/"frames").mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(parents=True,exist_ok=True)
    def add(self,state,row,branches):
        row=dict(row);row["sample_id"]=len(self.rows);self.rows.append(row)
        self.contours.append(dict(sample_id=row["sample_id"],**{
            f"{coord}_{side}":np.asarray(branches[side][f"{coord}_m"])
            for side in ("negative","positive") for coord in ("z","r")}))
        if self.render:
            path=self.out/"frames"/f"frame_{row['sample_id']:04d}.png"
            fig,ax=plt.subplots(figsize=(9,5),dpi=110)
            for side,color in (("negative","#b45309"),("positive","#1d4ed8")):
                z=np.asarray(branches[side]["z_m"])*1e9;r=np.asarray(branches[side]["r_m"])*1e9
                ax.plot(z,r,color=color);ax.plot(z,-r,color=color)
            ax.scatter([row["z_TJ_m"]*1e9]*2,[row["r_n_m"]*1e9,-row["r_n_m"]*1e9],c="crimson",s=18)
            ax.set_xlim(self.geom["z_min_m"]*1e9-5,self.geom["z_max_m"]*1e9+5);rad=self.geom["r_c"][-1]*1e9;ax.set_ylim(-rad,rad);ax.set_aspect("equal")
            ax.set_title(f"t={row['t_model']:.3f} cycle={row['cycle']} qcum/b={row['q_cumulative_over_b']:.2f} sigma={row['sigma_local_Pa']/1e6:.1f} MPa sink={'ON' if row['sink_state'] else 'OFF'}")
            ax.set_xlabel("z (nm)");ax.set_ylabel("r (nm)");ax.grid(alpha=.15);fig.savefig(path,bbox_inches="tight");plt.close(fig);self.frames.append(path)
    def flush(self):
        if self.rows:
            with (self.out/"history.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        arrays={}
        for item in self.contours:
            sid=item["sample_id"]
            for key,val in item.items():
                if key!="sample_id":arrays[f"sample_{sid:04d}_{key}"]=val
        np.savez_compressed(self.out/"contours.npz",**arrays)


def save_state(path,state,**meta):
    np.savez_compressed(path,f=state[0],particle=state[1],substrate=state[2],**{k:np.asarray(v) for k,v in meta.items()})


def wait_or_censor(state,cycle,completed,t_model,threshold,setup,geom,evaluator,output,out):
    vp0=integral(state[1],setup);H=0.;row,branches=base.measure(state,t_model=t_model,cycle=cycle,sink=0,q=0,qcum=completed,hazard=H,threshold=threshold,clock_scale=CLOCK_SCALE,setup=setup,geom=geom,evaluator=evaluator,vp_cycle0=vp0)
    output.add(state,row,branches);prev=row["Gamma_per_model_time"];current=row
    scratches=[NumbaScratch(*state[0].shape),NumbaScratch(*state[0].shape)];which=0;steps=int(round(SAMPLE_DT/setup["dt"]));elapsed=0.
    while True:
        state_before=tuple(x.copy() for x in state);elapsed_before=elapsed
        H_before=H;prev_before=prev;current_before=current
        for _ in range(steps):
            dest=0 if which!=0 else 1
            state=axisym_gb_face_projected_step_fast(*state,setup["p"],setup["Wc"],setup["dr"],setup["dz"],setup["r_c"],setup["r_f"],setup["dt"],setup["M_s"],setup["M_eta"],setup["W"],scratches[dest]);which=dest
            f,p,s,_=reservoir_remove_particle(state,setup);state=(f,p,s)
        block_dt=steps*setup["dt"];elapsed+=block_dt
        trial,branches=base.measure(state,t_model=t_model+elapsed,cycle=cycle,sink=0,q=0,qcum=completed,hazard=H,threshold=threshold,clock_scale=CLOCK_SCALE,setup=setup,geom=geom,evaluator=evaluator,vp_cycle0=vp0)
        H+=.5*(prev+trial["Gamma_per_model_time"])*block_dt;trial["H"]=H;trial["H_over_threshold"]=H/threshold
        output.add(state,trial,branches);output.flush();save_state(out/"checkpoints"/f"cycle{cycle}_waiting_latest.npz",state,t_model=t_model+elapsed,H=H,Hstar=threshold,Vratio=trial["Vp_over_Vp_cycle"],rn=trial["r_n_m"])
        print(f"R{out.name[-2:]} WAIT c{cycle} t={elapsed:.2f} V={trial['Vp_over_Vp_cycle']:.4f} rn={trial['r_n_m']*1e9:.2f} H/H*={H/threshold:.3f}",flush=True)
        invalid=[]
        if trial["Vp_over_Vp_cycle"]<V_CUTOFF:invalid.append("Vp_over_Vp_cycle_below_0.88")
        if trial["r_n_m"]<RN_CUTOFF_M:invalid.append("r_n_below_35_nm")
        if invalid and H>=threshold and H_before<threshold:
            alpha_h=(threshold-H_before)/max(H-H_before,1e-300)
            alpha_v=(current_before["Vp_over_Vp_cycle"]-V_CUTOFF)/max(current_before["Vp_over_Vp_cycle"]-trial["Vp_over_Vp_cycle"],1e-300)
            alpha_r=(current_before["r_n_m"]-RN_CUTOFF_M)/max(current_before["r_n_m"]-trial["r_n_m"],1e-300)
            if alpha_h<min(alpha_v,alpha_r):
                # Hazard crossed first inside the coarse block.  Discard only
                # that overshooting analysis row and replay from the preserved
                # valid state at fine cadence; PF equations and dt are unchanged.
                output.rows.pop();output.contours.pop();state=state_before
                elapsed=elapsed_before;H=H_before;prev=prev_before
                ref_steps=max(1,int(round(COMPETING_CROSSING_DT/setup["dt"])))
                scratches=[NumbaScratch(*state[0].shape),NumbaScratch(*state[0].shape)];which=0
                while True:
                    for _ in range(ref_steps):
                        dest=0 if which!=0 else 1
                        state=axisym_gb_face_projected_step_fast(*state,setup["p"],setup["Wc"],setup["dr"],setup["dz"],setup["r_c"],setup["r_f"],setup["dt"],setup["M_s"],setup["M_eta"],setup["W"],scratches[dest]);which=dest
                        f,p,s,_=reservoir_remove_particle(state,setup);state=(f,p,s)
                    ref_dt=ref_steps*setup["dt"];elapsed+=ref_dt
                    refined,branches=base.measure(state,t_model=t_model+elapsed,cycle=cycle,sink=0,q=0,qcum=completed,hazard=H,threshold=threshold,clock_scale=CLOCK_SCALE,setup=setup,geom=geom,evaluator=evaluator,vp_cycle0=vp0)
                    H+=.5*(prev+refined["Gamma_per_model_time"])*ref_dt;refined["H"]=H;refined["H_over_threshold"]=H/threshold
                    output.add(state,refined,branches);output.flush();save_state(out/"checkpoints"/f"cycle{cycle}_waiting_latest.npz",state,t_model=t_model+elapsed,H=H,Hstar=threshold,Vratio=refined["Vp_over_Vp_cycle"],rn=refined["r_n_m"])
                    ref_invalid=[]
                    if refined["Vp_over_Vp_cycle"]<V_CUTOFF:ref_invalid.append("Vp_over_Vp_cycle_below_0.88")
                    if refined["r_n_m"]<RN_CUTOFF_M:ref_invalid.append("r_n_below_35_nm")
                    print(f"R{out.name[-2:]} REFINE c{cycle} t={elapsed:.4f} V={refined['Vp_over_Vp_cycle']:.4f} H/H*={H/threshold:.4f}",flush=True)
                    if H>=threshold and not ref_invalid:
                        save_state(out/"checkpoints"/f"cycle{cycle}_before_nucleation.npz",state,t_model=t_model+elapsed,H=H,Hstar=threshold)
                        return "nucleated",state,t_model+elapsed,refined,None
                    if ref_invalid:
                        save_state(out/"checkpoints"/f"cycle{cycle}_censored.npz",state,t_model=t_model+elapsed,H=H,Hstar=threshold,Vratio=refined["Vp_over_Vp_cycle"],rn=refined["r_n_m"])
                        return "censored",state,t_model+elapsed,refined,ref_invalid
                    prev=refined["Gamma_per_model_time"]
        if invalid:
            save_state(out/"checkpoints"/f"cycle{cycle}_censored.npz",state,t_model=t_model+elapsed,H=H,Hstar=threshold,Vratio=trial["Vp_over_Vp_cycle"],rn=trial["r_n_m"])
            return "censored",state,t_model+elapsed,trial,invalid
        if H>=threshold:
            save_state(out/"checkpoints"/f"cycle{cycle}_before_nucleation.npz",state,t_model=t_model+elapsed,H=H,Hstar=threshold)
            return "nucleated",state,t_model+elapsed,trial,None
        prev=trial["Gamma_per_model_time"];current=trial


def configure_worker(out):
    authoritative.configure_barrier();base.OUT=out;base.MIN_SOLVABLE_VOLUME_RATIO=-math.inf
    assert PRODUCTION_C4 == 0.05


def run_realization(index,seed):
    out=OUT/f"realization_{index:02d}";out.mkdir(parents=True,exist_ok=True);configure_worker(out);set_num_threads(2)
    record=json.loads((out/"seed_record.json").read_text());rng=np.random.default_rng(seed);thresholds=rng.exponential(size=MAX_EVENTS);record.update(thresholds_generated=True,thresholds=thresholds.tolist(),worker_received_event_integrator_courant=PRODUCTION_C4,worker_courant_assertion_passed=(PRODUCTION_C4==0.05));(out/"seed_record.json").write_text(json.dumps(record,indent=2)+"\n")
    geom,setup=build_case();evaluator=make_evaluator(setup,geom);transport=make_transport(geom);state=(geom["f"].copy(),geom["e1"].copy(),geom["e2"].copy());output=LeanOutput(out,geom,render=False)
    worker_manifest=dict(
        realization=index,seed=seed,
        worker_received_event_integrator_courant=PRODUCTION_C4,
        worker_courant_assertion_passed=(PRODUCTION_C4==0.05),
        grid_shape=list(state[0].shape),dr_m=setup["dr"],dz_m=setup["dz"],
        W_m=setup["W"],R_cyl_m=geom["R_cyl"],lambda_m=geom["lam"],
        b_event_m=transport.b_m,
        barrier=dict(G0_eV=base.BARRIER.G0_eV,
                     G_floor_eV=base.BARRIER.G_floor_eV,
                     a=base.BARRIER.a,
                     sigma_hat_Pa=base.BARRIER.sigma_hat_pa,
                     n=base.BARRIER.n))
    (out/"worker_manifest.json").write_text(
        json.dumps(worker_manifest,indent=2)+"\n")
    events=[];censor=None;blocker=None;t_model=0.;started=time.monotonic()
    for j,threshold in enumerate(thresholds):
        cycle=j+1;wait_start=t_model;status,state,t_model,nuc,reasons=wait_or_censor(state,cycle,j,t_model,float(threshold),setup,geom,evaluator,output,out)
        if status=="censored":
            censor=dict(realization=index,cycle=cycle,reasons=reasons,
                        t_model=t_model,t_wait_model=t_model-wait_start,
                        t_wait_s=(t_model-wait_start)*.01557994316955921,
                        H=nuc["H"],H_star=float(threshold),
                        H_over_Hstar=nuc["H_over_threshold"],
                        sigma_Pa=nuc["sigma_local_Pa"],
                        Vp_over_Vp_cycle=nuc["Vp_over_Vp_cycle"],
                        r_n_m=nuc["r_n_m"]);break
        pre_local=nuc["sigma_local_Pa"];pre_integral=nuc["sigma_integral_Pa"]
        try:
            state,t_model,restart=base.run_event(
                state,cycle=cycle,completed=j,t_model=t_model,nucleation=nuc,
                threshold=float(threshold),clock_scale=CLOCK_SCALE,
                setup=setup,geom=geom,evaluator=evaluator,
                transport=transport,output=output,
                explicit_max_fourth_order_courant=PRODUCTION_C4)
        except RuntimeError as exc:
            blocker=dict(realization=index,cycle=cycle,
                         reason="event_failed_inside_validity_envelope",
                         message=str(exc),
                         Vp_over_Vp_cycle=nuc["Vp_over_Vp_cycle"],
                         r_n_m=nuc["r_n_m"])
            break
        final=output.rows[-1]
        if restart["cumulative_q_m"]<transport.b_m*(1-1e-12):
            blocker=dict(realization=index,cycle=cycle,reason="event_failed_inside_validity_envelope",q_over_b=restart["cumulative_q_m"]/transport.b_m);break
        events.append(dict(realization=index,event_number=cycle,
                           t_wait_model=nuc["t_model"]-wait_start,
                           t_wait_s=(nuc["t_model"]-wait_start)*.01557994316955921,
                           H_star=float(threshold),sigma_nuc_Pa=pre_local,
                           sigma_nuc_integral_Pa=pre_integral,
                           G_nuc_star_eV=nuc["G_star_eV"],
                           r_n_nuc_m=nuc["r_n_m"],
                           Vp_over_Vp_cycle=nuc["Vp_over_Vp_cycle"],
                           N_sites=nuc["N_sites"],
                           delta_sigma_local_Pa=pre_local-final["sigma_local_Pa"],
                           delta_sigma_integral_Pa=pre_integral-final["sigma_integral_Pa"],
                           t_event_model=restart["event_time_model"],
                           t_event_s=restart["event_time_model"]*.01557994316955921,
                           sigma_post_local_Pa=final["sigma_local_Pa"],
                           sigma_post_integral_Pa=final["sigma_integral_Pa"],
                           event_integrator_courant=restart["explicit_max_fourth_order_courant"]));
        with (out/"event_summary.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(events[0]));w.writeheader();w.writerows(events)
    output.flush()
    if output.frames:
        imgs=[Image.open(p).convert("P",palette=Image.ADAPTIVE) for p in output.frames];imgs[0].save(out/"morphology.gif",save_all=True,append_images=imgs[1:],duration=300,loop=0)
    result=dict(realization=index,seed=seed,events_completed=len(events),censored=censor is not None,censor=censor,blocker=blocker,thresholds=thresholds.tolist(),worker_received_event_integrator_courant=PRODUCTION_C4,worker_courant_assertion_passed=(PRODUCTION_C4==0.05 and all(e["event_integrator_courant"]==0.05 for e in events)),wall_seconds=time.monotonic()-started,events=events)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print("REALIZATION DONE",json.dumps({k:result[k] for k in ("realization","events_completed","censored","blocker")}),flush=True);return result


def render_representative(results):
    candidates=[r for r in results if r["events_completed"]>0]
    if not candidates:return None
    chosen=max(candidates,key=lambda r:(r["events_completed"],-r["realization"]))
    source=OUT/f"realization_{chosen['realization']:02d}"
    with (source/"history.csv").open(newline="") as h:rows=list(csv.DictReader(h))
    contour=np.load(source/"contours.npz")
    frame_dir=OUT/"representative_frames";frame_dir.mkdir(exist_ok=True)
    all_z=np.concatenate([contour[k] for k in contour.files if k.endswith(("z_negative","z_positive"))])*1e9
    all_r=np.concatenate([contour[k] for k in contour.files if k.endswith(("r_negative","r_positive"))])*1e9
    xlim=(float(all_z.min()-5),float(all_z.max()+5));rmax=float(all_r.max()+5);frames=[]
    for i,row in enumerate(rows):
        sid=int(row["sample_id"]);fig,ax=plt.subplots(figsize=(9,5),dpi=110)
        for side,color in (("negative","#b45309"),("positive","#1d4ed8")):
            z=contour[f"sample_{sid:04d}_z_{side}"]*1e9;r=contour[f"sample_{sid:04d}_r_{side}"]*1e9
            ax.plot(z,r,color=color);ax.plot(z,-r,color=color)
        ztj=float(row["z_TJ_m"])*1e9;rtj=float(row["r_n_m"])*1e9
        ax.scatter([ztj,ztj],[rtj,-rtj],c="crimson",s=18)
        ax.set_xlim(*xlim);ax.set_ylim(-rmax,rmax);ax.set_aspect("equal")
        ax.set_title(f"R{chosen['realization']} t={float(row['t_model']):.3f} cycle={row['cycle']} qcum/b={float(row['q_cumulative_over_b']):.2f} sigma={float(row['sigma_local_Pa'])/1e6:.1f} MPa sink={'ON' if float(row['sink_state']) else 'OFF'}")
        ax.set(xlabel="z (nm)",ylabel="r (nm)");ax.grid(alpha=.15)
        path=frame_dir/f"frame_{i:04d}.png";fig.savefig(path,bbox_inches="tight");plt.close(fig);frames.append(path)
    images=[Image.open(path).convert("P",palette=Image.ADAPTIVE) for path in frames]
    images[0].save(OUT/"representative_morphology.gif",save_all=True,
                   append_images=images[1:],duration=300,loop=0)
    contour.close()
    return chosen["realization"]


def aggregate(results):
    events=[e for r in results for e in r["events"]];censored=[r["censor"] for r in results if r["censor"]];blockers=[r["blocker"] for r in results if r["blocker"]]
    if events:
        with (OUT/"ensemble_events.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(events[0]));w.writeheader();w.writerows(events)
    if censored:
        fields=sorted({k for x in censored for k in x});
        with (OUT/"ensemble_censoring.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(censored)
    fig,axes=plt.subplots(4,2,figsize=(14,16),constrained_layout=True);axes=axes.ravel()
    for r in results:
        p=OUT/f"realization_{r['realization']:02d}/history.csv";
        if p.exists():
            rr=list(csv.DictReader(p.open()));axes[0].plot([float(x["t_model"]) for x in rr],[float(x["sigma_local_Pa"])/1e6 for x in rr],alpha=.7,label=f"R{r['realization']}")
    axes[0].set_title("stress-time trajectories")
    if events:
        axes[1].hist([e["sigma_nuc_Pa"]/1e6 for e in events],bins="auto");axes[1].set_title("nucleation stress")
        axes[2].hist([e["t_wait_model"] for e in events],bins="auto");axes[2].set_title("waiting time")
        axes[3].hist([e["delta_sigma_local_Pa"]/1e6 for e in events],bins="auto");axes[3].set_title("local stress drops")
        axes[4].scatter([e["H_star"] for e in events],[e["t_wait_model"] for e in events]);axes[4].set(xlabel="H*",ylabel="t_wait")
        axes[5].scatter([e["Vp_over_Vp_cycle"] for e in events],[e["sigma_nuc_Pa"]/1e6 for e in events]);axes[5].set(xlabel="V/Vcycle",ylabel="sigma_nuc MPa")
        axes[6].scatter([e["sigma_nuc_Pa"]/1e6 for e in events],[e["sigma_nuc_integral_Pa"]/1e6 for e in events]);axes[6].set(xlabel="local nuc MPa",ylabel="integral nuc MPa")
        axes[7].scatter([e["event_number"] for e in events],[e["delta_sigma_local_Pa"]/1e6 for e in events]);axes[7].set(xlabel="event number",ylabel="stress drop MPa")
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/"ensemble_summary.png",dpi=180);plt.close(fig)
    # Empirical survival including right-censoring at observed cycle wait.
    obs=[(e["t_wait_model"],1) for e in events]+[(c["t_wait_model"],0) for c in censored]
    obs.sort();at_risk=len(obs);surv=1.;curve=[(0.,1.)]
    for t,event in obs:
        if event:surv*=1.-1./at_risk
        at_risk-=1;curve.append((t,surv))
    if obs:
        fig,ax=plt.subplots(figsize=(8,5));ax.step([x[0] for x in curve],[x[1] for x in curve],where="post");ax.set(xlabel="waiting time (model)",ylabel="Kaplan-Meier survival");ax.grid(alpha=.2);fig.savefig(OUT/"waiting_time_kaplan_meier.png",dpi=180);plt.close(fig)
    representative=render_representative(results)
    summary=dict(N_completed=len(events),N_censored=len(censored),N_blockers=len(blockers),representative_morphology_realization=representative,realizations=results,censoring_reasons=[c["reasons"] for c in censored])
    (OUT/"ensemble_result.json").write_text(json.dumps(summary,indent=2)+"\n");print("ENSEMBLE COMPLETE",json.dumps({k:summary[k] for k in ("N_completed","N_censored","N_blockers")}),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest_path=OUT/"campaign_manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());seeds=manifest["seeds"]
        if len(seeds)!=N_REALIZATIONS or manifest.get("thresholds_generated"):
            raise RuntimeError("existing campaign is not an unstarted ten-seed launch")
        print("REUSING PRE-THRESHOLD RECORDED SEEDS",flush=True)
    else:
        if (PRE_VERIFICATION_OUT/"campaign_manifest.json").exists():
            old_manifest=json.loads((PRE_VERIFICATION_OUT/"campaign_manifest.json").read_text())
            seeds=old_manifest["seeds"]
            seed_source=str(PRE_VERIFICATION_OUT/"campaign_manifest.json")
        else:
            seeds=[int.from_bytes(os.urandom(16),"big") for _ in range(N_REALIZATIONS)]
            seed_source="128 bits os.urandom"
        export_bytes=authoritative.EXPORT.read_bytes()
        exported=json.loads(export_bytes)
        barrier_slice=next(x for x in exported["barrier_model"]["temperature_slices"]
                           if math.isclose(x["T_K"],1830.15,abs_tol=1e-10))
        pip_freeze=subprocess.run(
            [sys.executable,"-m","pip","freeze","--all"],
            check=True,capture_output=True,text=True).stdout.splitlines()
        manifest=dict(created_unix_time=time.time(),seeds=seeds,
                      thresholds_generated=False,
                      seed_source=seed_source,
                      production_baseline_commit=os.environ.get("PRODUCTION_BASELINE_COMMIT"),
                      python_environment=dict(
                          executable=sys.executable,version=sys.version,
                          platform=platform.platform(),pip_freeze=pip_freeze),
                      barrier_export=dict(
                          path=str(authoritative.EXPORT),
                          sha256=hashlib.sha256(export_bytes).hexdigest(),
                          creep_fit_b_m=exported["constants"]["b_m"],
                          exact_temperature_slice=barrier_slice),
                      settings=dict(T_K=1830.15,b_m=.25e-9,
                      Nsites="2*pi*rTJ/b",tau_ratio=20,
                      seconds_per_model_time=.01557994316955921,C4=PRODUCTION_C4,
                      V_cutoff=V_CUTOFF,rn_cutoff_m=RN_CUTOFF_M,
                      matched_outer_node=False,
                      geometry=dict(R_cyl_m=100e-9,W_m=10e-9,
                                    dx_m=1.25e-9,
                                    lambda_over_R_cyl=math.sqrt(2)*math.pi,
                                    epsilon1=0.4,epsilon2=0.0)))
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    for i,seed in enumerate(seeds,1):
        d=OUT/f"realization_{i:02d}";d.mkdir(exist_ok=True);seed_path=d/"seed_record.json"
        if not seed_path.exists():
            seed_path.write_text(json.dumps(dict(realization=i,seed=seed,source=manifest.get("seed_source","128 bits os.urandom"),recorded_before_thresholds=True,thresholds_generated=False),indent=2)+"\n")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS,mp_context=mp.get_context("spawn")) as ex:
        futures=[ex.submit(run_realization,i,s) for i,s in enumerate(seeds,1)];results=[f.result() for f in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda x:x["realization"]);manifest["thresholds_generated"]=True;manifest_path.write_text(json.dumps(manifest,indent=2)+"\n");aggregate(results)


if __name__=="__main__":main()
