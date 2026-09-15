#!/usr/bin/env python3
"""Decompose the completed earlier-stage trajectory before its first root."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def contact_terms(contact):
    k_minus=contact["kappa1_negative_per_m"]
    k_plus=contact["kappa1_positive_per_m"]
    theta_minus=contact["theta_negative_rad"]
    theta_plus=contact["theta_positive_rad"]
    radius=contact["r_n_m"]
    curvature=-0.5*(k_minus+k_plus)
    tj=1.5*(math.sin(theta_minus/2)+math.sin(theta_plus/2))/radius
    return dict(curvature_term_MPa=curvature*1e-6,TJ_term_MPa=tj*1e-6,
        total_MPa=(curvature+tj)*1e-6,rTJ_nm=radius*1e9,
        kappa_negative_per_um=k_minus*1e-6,kappa_positive_per_um=k_plus*1e-6,
        theta_negative_deg=math.degrees(theta_minus),
        theta_positive_deg=math.degrees(theta_plus))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("campaign",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    history=json.loads((args.campaign/"history.json").read_text())
    pre=[]
    for record in history:
        pre.append(record)
        if record["phase"] == "ROOT_CROSSING":break
    rows=[]
    for record in pre:
        for name in ("LEFT","RIGHT"):
            d=record["diagnostics"];c=contact_terms(record["contacts"][name])
            rows.append(dict(time_s=record["time_s"],contact=name,**c,
                neck_radius_nm=d[f"{name}_neck_r_m"]*1e9,
                center_volume_change=record["center_volume_m3"]/pre[0]["center_volume_m3"]-1,
                gb_position_nm=d[f"{name}_gb_z_m"]*1e9,
                surface_area_relative=record["surface_area_m2"]/pre[0]["surface_area_m2"]-1,
                GB_area_relative=record["GB_area_m2"]/pre[0]["GB_area_m2"]-1,
                interfacial_energy_relative=record["total_interfacial_energy_J"]/pre[0]["total_interfacial_energy_J"]-1))
    with (args.out/"pre_root_decomposition.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    left=[r for r in rows if r["contact"]=="LEFT"]
    peak=max(range(len(left)),key=lambda i:left[i]["total_MPa"])
    anchors={"start":left[0],"maximum":left[peak],"root":left[-1]}
    intervals={
        "initial_rise":{key:anchors["maximum"][key]-anchors["start"][key]
            for key in ("curvature_term_MPa","TJ_term_MPa","total_MPa","rTJ_nm",
                        "kappa_negative_per_um","kappa_positive_per_um",
                        "theta_negative_deg","theta_positive_deg")},
        "post_peak_relaxation":{key:anchors["root"][key]-anchors["maximum"][key]
            for key in ("curvature_term_MPa","TJ_term_MPa","total_MPa","rTJ_nm",
                        "kappa_negative_per_um","kappa_positive_per_um",
                        "theta_negative_deg","theta_positive_deg")}}
    payload=dict(status="PRE_ROOT_DECOMPOSITION_COMPLETE",anchors=anchors,interval_changes=intervals,
        interpretation=("The initial rise is produced by a more negative fitted mean meridional "
          "curvature; its gain exceeds the simultaneous TJ-term loss. The fitted curvature term "
          "reverses at the stress maximum, after which both curvature and TJ terms decrease."))
    (args.out/"pre_root_decomposition.json").write_text(json.dumps(payload,indent=2)+"\n")
    t=[r["time_s"] for r in left]
    fig,ax=plt.subplots(3,2,figsize=(12,11),sharex=True)
    ax[0,0].plot(t,[r["curvature_term_MPa"] for r in left],label=r"$-\gamma_s\bar{k}_m$")
    ax[0,0].plot(t,[r["TJ_term_MPa"] for r in left],label="TJ term")
    ax[0,0].plot(t,[r["total_MPa"] for r in left],label="sum",lw=2)
    ax[0,0].set_ylabel("stress contribution (MPa)");ax[0,0].legend()
    ax[0,1].plot(t,[r["kappa_negative_per_um"] for r in left],label="negative side")
    ax[0,1].plot(t,[r["kappa_positive_per_um"] for r in left],label="positive side")
    ax[0,1].set_ylabel(r"$k_m$ ($\mu$m$^{-1}$)");ax[0,1].legend()
    ax[1,0].plot(t,[r["theta_negative_deg"] for r in left],label="negative side")
    ax[1,0].plot(t,[r["theta_positive_deg"] for r in left],label="positive side")
    ax[1,0].set_ylabel("one-sided angle (deg)");ax[1,0].legend()
    ax[1,1].plot(t,[r["rTJ_nm"] for r in left],label="TJ radius")
    ax[1,1].plot(t,[r["neck_radius_nm"] for r in left],"--",label="neck radius")
    ax[1,1].set_ylabel("radius (nm)");ax[1,1].legend()
    ax[2,0].plot(t,[100*r["center_volume_change"] for r in left],label="center volume")
    ax[2,0].plot(t,[100*r["surface_area_relative"] for r in left],label="surface area")
    ax[2,0].plot(t,[100*r["GB_area_relative"] for r in left],label="GB area")
    ax[2,0].plot(t,[100*r["interfacial_energy_relative"] for r in left],label="energy")
    ax[2,0].set_ylabel("change (%)");ax[2,0].set_xlabel("time (s)");ax[2,0].legend(fontsize=8)
    ax[2,1].plot(t,[r["gb_position_nm"] for r in left],label="LEFT GB")
    right=[r for r in rows if r["contact"]=="RIGHT"]
    ax[2,1].plot(t,[r["gb_position_nm"] for r in right],label="RIGHT GB")
    ax[2,1].set_ylabel("GB position (nm)");ax[2,1].set_xlabel("time (s)");ax[2,1].legend()
    for a in ax.flat:a.grid(alpha=.2);a.axvline(left[peak]["time_s"],color="tab:red",alpha=.25)
    fig.tight_layout();fig.savefig(args.out/"pre_root_decomposition.png",dpi=180);plt.close(fig)
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
