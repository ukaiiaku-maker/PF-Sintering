#!/usr/bin/env python3
"""Closed-system, padded-domain, no-event conservation qualification to t=34."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_avalanche_final_balanced as balanced  # noqa: E402
from pf_sintering.axisym import axisym_volume  # noqa: E402
from pf_sintering.axisym_numba_kernel import (  # noqa: E402
    NumbaScratch, axisym_gb_face_projected_step_fast,
)
from pf_sintering.conservative_bounded_phase import ConservativeBoundedPhaseProjector  # noqa: E402
from pf_sintering.continuous_field_tj import ContinuousFieldTJTracker  # noqa: E402
from pf_sintering.experimental_pr_metrology import measure_experimental_pr_state  # noqa: E402
from pf_sintering.fourier_max_pr_geometry import build_fourier_max_pr_geometry  # noqa: E402
from pr_geometry_conservation_tj_event_audit import outer_radius, contour_volume  # noqa: E402
from pr_tj_node_coupling_gate import make_evaluator  # noqa: E402


OUT = Path(os.environ.get(
    "PR_CLOSED_OUT",
    str(ROOT / "runs/pr_closed_volume_no_event_t34_eps0p25_12W")))
T_FINAL = float(os.environ.get("PR_CLOSED_T_FINAL", "34.0"))
ANALYSIS_DT = float(os.environ.get("PR_CLOSED_ANALYSIS_DT", "0.25"))
CHECKPOINT_DT = float(os.environ.get("PR_CLOSED_CHECKPOINT_DT", "0.50"))
INTERNAL_AUDIT_STRIDE = 256
VOLUME_TOLERANCE = 1.0e-8


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def build_case():
    geom = build_fourier_max_pr_geometry(
        R_cyl=100e-9, W=10e-9, spacing=1.25e-9, eps1=0.25,
        tail_margin_W=12.0, radial_margin_W=12.0,
        close_left_substrate=True)
    _, qualified = balanced.make_setup(10.0, 1.25)
    setup = {**qualified, "dr": geom["dr"], "dz": geom["dz"],
             "r_c": geom["r_c"], "r_f": geom["r_f"], "z": geom["z"],
             "lam": geom["lam"]}
    setup["M_s"] *= balanced.SURFACE_RATE_SCALE
    setup["local_metrology_window_m"] = 3.0*setup["W"]
    setup["tau_surface_model"] = balanced.TAU_SURFACE_NEW_MODEL
    setup["tau_coarsen_model"] = balanced.TAU_COARSEN_NEW_MODEL
    return geom, setup


def make_tracker_evaluator(setup, geom, previous=None):
    tracker = ContinuousFieldTJTracker(
        setup["z"], setup["r_c"], geom["z1"],
        6.0*max(setup["dr"], setup["dz"]))
    if previous is not None:
        tracker.previous_z_m = float(previous[0])
        tracker.previous_r_m = float(previous[1])
    return make_evaluator(setup, geom, tj_tracker=tracker)


class FieldArchive:
    def __init__(self, path: Path, shape, *, resume=False):
        self.handle = h5py.File(path, "a" if resume else "w")
        if not resume:
            self.handle.attrs.update(
                system_mass_boundary="closed", external_reservoir_enabled=False,
                no_nucleation=True, no_event=True, epsilon1=0.25,
                tail_margin_W=12.0, radial_margin_W=12.0)
            for name in ("f", "e1", "e2"):
                self.handle.create_dataset(
                    name, shape=(0, *shape), maxshape=(None, *shape),
                    chunks=(1, *shape), dtype="f4", compression="lzf")
            for name in ("radius_m",):
                self.handle.create_dataset(
                    name, shape=(0, shape[0]), maxshape=(None, shape[0]),
                    chunks=(1, shape[0]), dtype="f4", compression="lzf")
            for name in ("t_model", "step"):
                self.handle.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")

    def append(self, state, radius, t_model, step):
        n = len(self.handle["t_model"])
        for name, data in zip(("f", "e1", "e2"), state):
            ds=self.handle[name]; ds.resize(n+1,axis=0); ds[n]=np.asarray(data,dtype=np.float32)
        ds=self.handle["radius_m"]; ds.resize(n+1,axis=0); ds[n]=np.asarray(radius,dtype=np.float32)
        for name,value in (("t_model",t_model),("step",step)):
            ds=self.handle[name]; ds.resize(n+1,axis=0); ds[n]=value
        self.handle.flush()

    def close(self):
        self.handle.close()


def render_outputs(setup, rows):
    t=np.asarray([r["t_model"] for r in rows])
    fig,ax=plt.subplots(4,1,figsize=(10,12),sharex=True)
    ax[0].plot(t,[r["Vf_relative"] for r in rows],label="Vf")
    ax[0].plot(t,[r["Veta_relative"] for r in rows],label="V1+V2")
    ax[0].plot(t,[r["Vcontour_relative"] for r in rows],label="contour")
    ax[0].axhline(0,color="k",lw=.8);ax[0].set_ylabel("relative change");ax[0].legend()
    ax[1].plot(t,np.asarray([r["V1_m3"] for r in rows])/rows[0]["V1_m3"],label="V1/V1(0)")
    ax[1].plot(t,np.asarray([r["V2_m3"] for r in rows])/rows[0]["V2_m3"],label="V2/V2(0)")
    ax[1].set_ylabel("grain volume ratio");ax[1].legend()
    ax[2].semilogy(t,np.maximum([abs(r["partition_volume_error_m3"]) for r in rows],1e-40),label="|V1+V2-Vf|")
    ax[2].semilogy(t,np.maximum([r["closure_max"] for r in rows],1e-20),label="max point closure")
    ax[2].set_ylabel("closure");ax[2].legend()
    ax[3].plot(t,np.asarray([r["radial_margin_m"] for r in rows])*1e9,label="radial")
    ax[3].plot(t,np.asarray([r["left_z_margin_m"] for r in rows])*1e9,label="left z")
    ax[3].plot(t,np.asarray([r["right_z_margin_m"] for r in rows])*1e9,label="right z")
    ax[3].axhline(120,color="k",ls="--",lw=.8,label="12W")
    ax[3].set_ylabel("f=0.5 margin (nm)");ax[3].set_xlabel("model time");ax[3].legend(ncol=4)
    fig.tight_layout();fig.savefig(OUT/"closed_volume_conservation.png",dpi=180);fig.savefig(OUT/"closed_volume_conservation.pdf");plt.close(fig)

    frames_field=[];frames_contour=[]
    with h5py.File(OUT/"closed_volume_fields.h5") as h:
        initial=np.asarray(h["radius_m"][0],dtype=float)*1e9
        z=setup["z"]*1e9;r=setup["r_c"]*1e9
        extent=[z[0],z[-1],r[0],r[-1]]
        for i,row in enumerate(rows):
            f=np.asarray(h["f"][i]);e1=np.asarray(h["e1"][i]);e2=np.asarray(h["e2"][i]);radius=np.asarray(h["radius_m"][i],dtype=float)*1e9
            fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
            for a,data,title,cmap,lim in (
                    (axes[0,0],f,"conserved f","viridis",(0,1)),
                    (axes[0,1],e1,"grain e1","magma",(0,1)),
                    (axes[1,0],e2,"grain e2","magma",(0,1)),
                    (axes[1,1],e1-e2,"e1-e2","coolwarm",(-1,1))):
                a.imshow(data.T,origin="lower",extent=extent,aspect="equal",cmap=cmap,vmin=lim[0],vmax=lim[1])
                a.plot(z,initial,"w--",lw=.7,alpha=.45)
                a.plot(z,radius,"w-",lw=.8)
                a.plot(row["z_TJ_m"]*1e9,row["r_TJ_m"]*1e9,"co",ms=4)
                a.set_xlim(0,setup["z"][-1]*1e9);a.set_ylim(0,setup["r_f"][-1]*1e9)
                a.set_xlabel("z nm");a.set_ylabel("r nm");a.set_title(title)
            fig.suptitle(f"closed no-event t={row['t_model']:.2f}  dV/V={row['Vf_relative']:+.2e}")
            fig.canvas.draw();frames_field.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy());plt.close(fig)

            fig,a=plt.subplots(figsize=(11,4),constrained_layout=True)
            a.plot(z,initial,"k--",lw=1,alpha=.35,label="initial f=0.5")
            a.plot(z,radius,"tab:blue",lw=1.5,label="current f=0.5")
            a.fill_between(z,0,radius,color="tab:blue",alpha=.12)
            a.plot(row["z_TJ_m"]*1e9,row["r_TJ_m"]*1e9,"ro",ms=5,label="field TJ")
            a.set_xlim(0,setup["z"][-1]*1e9);a.set_ylim(0,setup["r_f"][-1]*1e9);a.set_aspect("equal")
            a.set_xlabel("z nm");a.set_ylabel("r nm");a.set_title(f"t={row['t_model']:.2f}, Vcontour/V0-1={row['Vcontour_relative']:+.2e}");a.legend()
            fig.canvas.draw();frames_contour.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy());plt.close(fig)
    imageio.mimsave(OUT/"closed_volume_full_field.gif",frames_field,duration=.10,loop=0)
    imageio.mimsave(OUT/"closed_volume_contour.gif",frames_contour,duration=.10,loop=0)


def main():
    set_num_threads(8)
    OUT.mkdir(parents=True,exist_ok=True)
    geom,setup=build_case()
    manifest=dict(
        purpose="closed-volume no-event conservation gate",
        system_mass_boundary="closed", external_reservoir_enabled=False,
        explicit_coarsening_source=False,
        evolution="conservative no-flux PF surface diffusion plus ownership evolution",
        epsilon1=0.25, grid_shape=list(geom["f"].shape),
        dr_m=setup["dr"],dz_m=setup["dz"],W_m=setup["W"],
        tail_margin_W=12.0,radial_margin_W=12.0,
        left_substrate_cap_closed=True,t_final_model=T_FINAL,
        volume_gate_relative=VOLUME_TOLERANCE,no_nucleation=True,no_event=True)
    atomic_json(OUT/"launch_manifest.json",manifest)

    checkpoint=OUT/"checkpoint_latest.npz"
    history_path=OUT/"closed_volume_history.csv"
    archive_path=OUT/"closed_volume_fields.h5"
    if checkpoint.exists() and history_path.exists() and archive_path.exists():
        with np.load(checkpoint) as saved:
            state=tuple(np.asarray(saved[k]).copy() for k in ("f","e1","e2"))
            step=int(saved["step"]);V0=float(saved["V0_m3"])
        with history_path.open(newline="") as stream:
            rows=[{k:(v if k=="phase" else float(v)) for k,v in row.items()}
                  for row in csv.DictReader(stream)]
        previous=(rows[-1]["z_TJ_m"],rows[-1]["r_TJ_m"])
        archive=FieldArchive(archive_path,state[0].shape,resume=True)
    else:
        state=(geom["f"].copy(),geom["e1"].copy(),geom["e2"].copy())
        step=0;rows=[];V0=axisym_volume(state[0],setup["r_c"],setup["dr"],setup["dz"])
        previous=None;archive=FieldArchive(archive_path,state[0].shape,resume=False)
    evaluator=make_tracker_evaluator(setup,geom,previous)
    projector=ConservativeBoundedPhaseProjector()
    scratches=[NumbaScratch(*state[0].shape),NumbaScratch(*state[0].shape)]
    current=step%2
    analysis_steps=max(1,int(round(ANALYSIS_DT/setup["dt"])))
    checkpoint_steps=max(1,int(round(CHECKPOINT_DT/setup["dt"])))
    final_steps=int(round(T_FINAL/setup["dt"]))
    started=time.monotonic()

    def observe():
        f,e1,e2=state
        scalar,_,_=measure_experimental_pr_state(state,setup,evaluator)
        radius=outer_radius(f,setup["r_c"]);finite=np.flatnonzero(np.isfinite(radius))
        Vf=axisym_volume(f,setup["r_c"],setup["dr"],setup["dz"])
        V1=axisym_volume(e1,setup["r_c"],setup["dr"],setup["dz"])
        V2=axisym_volume(e2,setup["r_c"],setup["dr"],setup["dz"])
        Vc=contour_volume(setup["z"],radius)
        row=dict(phase="passive_closed",t_model=step*setup["dt"],step=float(step),
                 Vf_m3=Vf,V1_m3=V1,V2_m3=V2,Veta_m3=V1+V2,
                 partition_volume_error_m3=V1+V2-Vf,
                 closure_max=float(np.max(np.abs(e1+e2-f))),
                 closure_rms=float(np.sqrt(np.mean((e1+e2-f)**2))),
                 Vcontour_m3=Vc,Vf_relative=Vf/V0-1,
                 Veta_relative=(V1+V2)/V0-1,
                 Vcontour_relative=(Vc/(rows[0]["Vcontour_m3"] if rows else Vc)-1),
                 radial_margin_m=float(setup["r_f"][-1]-np.nanmax(radius)),
                 left_z_margin_m=float(setup["z"][finite[0]]-0.5*setup["dz"]),
                 right_z_margin_m=float(setup["z"][-1]+0.5*setup["dz"]-setup["z"][finite[-1]]),
                 boundary_f_max=float(max(np.max(f[0]),np.max(f[-1]),np.max(f[:,-1]))),
                 z_TJ_m=float(scalar["z_TJ_m"]),r_TJ_m=float(scalar["r_TJ_m"]),
                 sigma_local_Pa=float(scalar["sigma_local_Pa"]),
                 sigma_integral_Pa=float(scalar["sigma_integral_Pa"]))
        if abs(row["Vf_relative"]) >= VOLUME_TOLERANCE:
            atomic_npz(checkpoint,f=f,e1=e1,e2=e2,step=step,V0_m3=V0)
            raise RuntimeError(f"closed volume invariant failed: {row['Vf_relative']:+.6e}")
        rows.append(row);write_csv(history_path,rows);archive.append(state,radius,row["t_model"],step)
        atomic_json(OUT/"run_state.json",dict(status="RUNNING",latest=row,
                    wall_seconds=time.monotonic()-started,projector=projector.manifest()))
        print("CLOSED",f"t={row['t_model']:.3f}",f"dV/V={row['Vf_relative']:+.3e}",
              f"dVc/Vc={row['Vcontour_relative']:+.3e}",
              f"margins_nm={row['left_z_margin_m']*1e9:.1f}/"
              f"{row['right_z_margin_m']*1e9:.1f}/"
              f"{row['radial_margin_m']*1e9:.1f}",flush=True)

    if not rows:
        observe()
    while step < final_steps:
        target=0 if current!=0 else 1
        state=axisym_gb_face_projected_step_fast(
            *state,setup["p"],setup["Wc"],setup["dr"],setup["dz"],
            setup["r_c"],setup["r_f"],setup["dt"],setup["M_s"],
            setup["M_eta"],setup["W"],scratches[target])
        current=target;step+=1
        state,_=projector(state,setup)
        if step%INTERNAL_AUDIT_STRIDE==0:
            volume=axisym_volume(state[0],setup["r_c"],setup["dr"],setup["dz"])
            if abs(volume/V0-1)>=VOLUME_TOLERANCE:
                atomic_npz(checkpoint,f=state[0],e1=state[1],e2=state[2],step=step,V0_m3=V0)
                archive.close();raise RuntimeError("internal closed-volume invariant failed")
        if step%analysis_steps==0 or step==final_steps:
            observe()
        if step%checkpoint_steps==0 or step==final_steps:
            atomic_npz(checkpoint,f=state[0],e1=state[1],e2=state[2],step=step,V0_m3=V0)
    archive.close()
    max_field=max(abs(row["Vf_relative"]) for row in rows)
    max_contour=max(abs(row["Vcontour_relative"]) for row in rows)
    passed=max_field<VOLUME_TOLERANCE
    result=dict(outcome=("CLOSED_VOLUME_NO_EVENT_GATE_PASS" if passed else "FAIL"),
                system_mass_boundary="closed",external_reservoir_enabled=False,
                maximum_abs_field_volume_relative_drift=max_field,
                maximum_abs_contour_volume_relative_drift=max_contour,
                maximum_partition_closure=max(row["closure_max"] for row in rows),
                minimum_left_z_margin_m=min(row["left_z_margin_m"] for row in rows),
                minimum_right_z_margin_m=min(row["right_z_margin_m"] for row in rows),
                minimum_radial_margin_m=min(row["radial_margin_m"] for row in rows),
                samples=len(rows),final_t_model=rows[-1]["t_model"],
                projector=projector.manifest(),wall_seconds=time.monotonic()-started)
    atomic_json(OUT/"closed_volume_result.json",result)
    render_outputs(setup,rows)
    atomic_json(OUT/"run_state.json",dict(status="COMPLETE",result=result))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
