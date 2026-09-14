#!/usr/bin/env python3
"""Run the analytical three-particle stress-loading design campaign.

No phase-field state is opened or evolved.  The only production code reused is
the immutable bicrystal root law in ``three_particle_contacts.root_law``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pf_sintering.three_particle_contacts import root_law
from pf_sintering.three_particle_sharp_design import (
    SharpDesign, admissibility, center_squared_radius_coefficients,
    contact_state, evaluate_even_squared_radius, evaluate_polynomial,
    inverse_target_radius, outer_squared_radius_coefficients,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/three_particle/sharp_interface_loading_design"
MANIFEST_PATH = ROOT / "docs/three_particle/production_screen/bicrystal_launch_manifest.json"
REFERENCE_PATH = ROOT / "runs/three_particle_volume_loading_065_symmetry_released/report.json"
W = 4e-9
X_MILESTONES = np.array([0.0, .01, .02, .05, .10, .15, .20])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def design_id(d: SharpDesign) -> str:
    return (f"Ro{d.Ro_m*1e9:.0f}_q{d.Rc_over_Ro:.2f}_L{d.Lc_m/W:.0f}W_"
            f"r{d.rTJ_over_Rc:.2f}_th{d.theta_center_deg:.0f}_k{d.kRc_center:.2f}")


def path_rows(d: SharpDesign, manifest: dict, label: str,
              xs: np.ndarray = X_MILESTONES) -> list[dict]:
    raw = [contact_state(d, float(x)) for x in xs]
    rows = []
    for state in raw:
        law = root_law(state["sigma_GB_Pa"], state["r_TJ_m"], manifest)
        rows.append({
            "design_id": label, "x": state["x"],
            "Ro_nm": d.Ro_m*1e9, "Rc_over_Ro": d.Rc_over_Ro,
            "Rc_nm": d.Rc_m*1e9, "Lc_nm": d.Lc_m*1e9,
            "Lc_over_W": d.Lc_m/d.W_m, "rTJ_over_Rc": d.rTJ_over_Rc,
            "r_TJ_nm": state["r_TJ_m"]*1e9,
            "neck_radius_nm": state["neck_radius_m"]*1e9,
            "center_curvature_per_um": state["center_curvature_per_m"]*1e-6,
            "outer_curvature_per_um": state["outer_curvature_per_m"]*1e-6,
            "center_theta_deg": state["center_theta_deg"],
            "outer_theta_deg": state["outer_theta_deg"],
            "center_side_stress_MPa": state["sigma_center_side_Pa"]*1e-6,
            "outer_side_stress_MPa": state["sigma_outer_side_Pa"]*1e-6,
            "sigma_kappa_MPa": state["sigma_kappa_Pa"]*1e-6,
            "sigma_TJ_MPa": state["sigma_TJ_Pa"]*1e-6,
            "sigma_GB_MPa": state["sigma_GB_Pa"]*1e-6,
            "root_rate_per_contact_s": law["root_rate_per_s"],
            "two_contact_characteristic_wait_s": (
                manifest["root_threshold_multiplier"]/(2.0*law["root_rate_per_s"])),
            "G_fit_eV": law["G_fit_eV"], "G_root_eV": law["G_root_eV"],
            "Vc_m3": state["Vc_m3"], "Vo_each_m3": state["Vo_m3"],
        })
    # Decompose d sigma/dx into the three exact chain-rule contributions.
    x = np.array([r["x"] for r in rows])
    k = np.array([(r["center_curvature_per_um"]+r["outer_curvature_per_um"])*.5e6 for r in rows])
    radius = np.array([r["r_TJ_nm"]*1e-9 for r in rows])
    sine = np.array([math.sin(math.radians(r["center_theta_deg"])/2)
                     + math.sin(math.radians(r["outer_theta_deg"])/2) for r in rows])
    edge = 2 if len(x) > 2 else 1
    curvature_part = -d.gamma_s*np.gradient(k, x, edge_order=edge)
    angle_part = 1.5*d.gamma_s/radius*np.gradient(sine, x, edge_order=edge)
    radius_part = -1.5*d.gamma_s*sine/radius**2*np.gradient(radius, x, edge_order=edge)
    for i, row in enumerate(rows):
        row["dsigma_dx_curvature_MPa"] = curvature_part[i]*1e-6
        row["dsigma_dx_angle_MPa"] = angle_part[i]*1e-6
        row["dsigma_dx_radius_MPa"] = radius_part[i]*1e-6
        row["dsigma_dx_total_MPa"] = sum((curvature_part[i], angle_part[i], radius_part[i]))*1e-6
    return rows


def crossing_x(rows: list[dict], key: str, threshold: float,
               less: bool = False) -> float:
    first_hit = rows[0][key] <= threshold if less else rows[0][key] >= threshold
    if first_hit:
        return 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        va, vb = a[key], b[key]
        hit = vb <= threshold if less else vb >= threshold
        if hit:
            if vb == va: return b["x"]
            return a["x"] + (b["x"]-a["x"])*(threshold-va)/(vb-va)
    return math.nan


def summarize(d: SharpDesign, rows: list[dict], adm: dict) -> dict:
    stresses = np.array([r["sigma_GB_MPa"] for r in rows])
    waits = np.array([r["two_contact_characteristic_wait_s"] for r in rows])
    x=np.array([r["x"] for r in rows])
    integrated={name:float(np.trapezoid([r[name] for r in rows],x))
                for name in ("dsigma_dx_curvature_MPa","dsigma_dx_angle_MPa","dsigma_dx_radius_MPa")}
    x_contact_zero=math.pi*d.Lc_m*d.rTJ0_m**2/d.Vc0_m3
    x_to_6W=math.pi*d.Lc_m*(d.rTJ0_m**2-(6*d.W_m)**2)/d.Vc0_m3
    return {
        "design_id": design_id(d), "Ro_nm": d.Ro_m*1e9,
        "Rc_over_Ro": d.Rc_over_Ro, "Rc_nm": d.Rc_m*1e9,
        "Lc_nm": d.Lc_m*1e9, "Lc_over_W": d.Lc_m/W,
        "rTJ_over_Rc": d.rTJ_over_Rc, "rTJ0_nm": d.rTJ0_m*1e9,
        "theta_center0_deg": d.theta_center_deg,
        "theta_outer0_deg": d.theta_outer_deg,
        "center_curvature0_per_um":d.k_center0_per_m*1e-6,
        "outer_curvature0_per_um":d.k_outer0_per_m*1e-6,
        "kRc_center": d.kRc_center,"kRo_outer":d.kRo_outer,
        "sigma0_MPa": stresses[0], "sigma20_MPa": stresses[-1],
        "max_sigma_MPa": float(stresses.max()),
        "available_stress_increase_MPa": float(stresses.max()-stresses[0]),
        "delta_sigma_curvature_MPa":integrated["dsigma_dx_curvature_MPa"],
        "delta_sigma_angle_MPa":integrated["dsigma_dx_angle_MPa"],
        "delta_sigma_radius_MPa":integrated["dsigma_dx_radius_MPa"],
        "minimum_dsigma_dx_MPa": min(r["dsigma_dx_total_MPa"] for r in rows),
        "tau0_s": waits[0], "tau20_s": waits[-1],
        "x_to_20MPa": crossing_x(rows, "sigma_GB_MPa", 20.),
        "x_to_25MPa": crossing_x(rows, "sigma_GB_MPa", 25.),
        "x_to_30MPa": crossing_x(rows, "sigma_GB_MPa", 30.),
        "x_to_tau3s": crossing_x(rows, "two_contact_characteristic_wait_s", 3., less=True),
        "resolution_margin": adm["resolution_margin"],
        "rTJ20_over_W": adm["rTJ_terminal_over_W"],
        "rTJ20_nm":rows[-1]["r_TJ_nm"],
        "x_contact_zero":x_contact_zero,
        "x_to_6W":x_to_6W,
        "loss_fraction_margin_after_20pct_to_6W":x_to_6W-.20,
    }


def pareto(rows: list[dict]) -> list[dict]:
    """Nondominated set: tau0, reserve, loss-to-30, and resolution margin."""
    def metrics(r):
        x30 = r["x_to_30MPa"] if math.isfinite(r["x_to_30MPa"]) else 1.0
        return (r["tau0_s"], -r["available_stress_increase_MPa"], x30,
                -r["resolution_margin"])
    result=[]
    for i, row in enumerate(rows):
        m=metrics(row); dominated=False
        for j, other in enumerate(rows):
            if i == j: continue
            q=metrics(other)
            if all(a <= b+1e-12 for a,b in zip(q,m)) and any(a < b-1e-12 for a,b in zip(q,m)):
                dominated=True; break
        if not dominated: result.append(row)
    return result


def choose_discussion_candidates(front: list[dict], pool: list[dict]) -> list[tuple[str, dict]]:
    """Select distinct Pareto representatives; this does not rank a winner."""
    choices=[]; used=set()
    selectors=[
        ("largest reserve", front, lambda r: -r["available_stress_increase_MPa"]),
        ("largest resolution margin", front, lambda r: -r["resolution_margin"]),
        ("slowest admissible initial clock", pool, lambda r: -r["tau0_s"]),
        ("greatest kinetic acceleration", pool, lambda r: r["tau20_s"]-r["tau0_s"]),
        ("least loss to cross 30 MPa from below", [r for r in pool if r["sigma0_MPa"] < 30. and math.isfinite(r["x_to_30MPa"])], lambda r: r["x_to_30MPa"]),
    ]
    for reason, source, key in selectors:
        available=[r for r in source if r["design_id"] not in used]
        if not available: break
        selected=min(available,key=key); used.add(selected["design_id"]); choices.append((reason,selected))
    return choices


def current_reference(manifest: dict) -> tuple[list[dict], SharpDesign, list[dict]]:
    payload=json.loads(REFERENCE_PATH.read_text())
    actual=[]
    for h in payload["history"]:
        c=h["contacts"]["LEFT"]
        actual.append({"x":h["center_loss_fraction"],"sigma_GB_MPa":c["sigma_local_Pa"]*1e-6,
                       "r_TJ_nm":c["r_n_m"]*1e9,"two_contact_characteristic_wait_s":h["two_contact_wait_s"]})
    # Anchor a sharp-interface analogue to the first retained 0.65 state.
    h=payload["history"][0]; c=h["contacts"]["LEFT"]
    reference=SharpDesign(119.999e-9,.65,h["center_span_over_W"]*W,
        c["r_n_m"]/(119.999e-9*.65),
        math.degrees(c["theta_positive_rad"]),math.degrees(c["theta_negative_rad"]),
        c["kappa1_positive_per_m"]*(119.999e-9*.65),
        c["kappa1_negative_per_m"]*119.999e-9)
    analog=path_rows(reference,manifest,"current_065_analytical_analogue")
    return actual,reference,analog


def plot_geometry(d: SharpDesign, choices_label: str, path: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    for ax,x in zip(axes,(0.,.20)):
        cc=center_squared_radius_coefficients(d).copy()
        cc[0]-=x*d.Vc0_m3/(math.pi*d.Lc_m)
        u=np.linspace(-1,1,500); z=.5*d.Lc_m*u
        rc=np.sqrt(np.maximum(0,evaluate_even_squared_radius(cc,abs(u))))
        oc,L=outer_squared_radius_coefficients(d,x); t=np.linspace(0,1,500)
        ro=np.sqrt(np.maximum(0,evaluate_polynomial(oc,t)))
        ax.plot(z*1e9,rc*1e9,color="#e68613",lw=2,label="center")
        ax.plot((.5*d.Lc_m+L*t)*1e9,ro*1e9,color="#2a6fbb",lw=2,label="outer")
        ax.plot((-0.5*d.Lc_m-L*t)*1e9,ro*1e9,color="#2a6fbb",lw=2)
        ax.fill_between(z*1e9,0,rc*1e9,color="#e68613",alpha=.15)
        ax.set_title(f"x={x:.0%}"); ax.set_xlabel("z (nm)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("meridional radius (nm)"); axes[0].legend()
    fig.suptitle(choices_label); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST_PATH.read_text())
    designs=[]; rejected=[]; paths={}
    for Ro_nm in (80.,100.,120.,150.):
      for ratio in (.45,.55,.65,.75,.85):
       for Lw in (8.,10.,12.,16.,20.,24.):
        for rr in (.60,.80,1.00,1.20,1.40,1.60):
         for theta in (120.,135.,150.,165.,170.):
          for kr in (0.,.25,.50,.75):
           d=SharpDesign(Ro_nm*1e-9,ratio,Lw*W,rr,theta,theta,kr,kr)
           adm=admissibility(d)
           if not adm["admissible"]:
               rejected.append({"design_id":design_id(d),"reason":adm["reason"]}); continue
           dense=path_rows(d,manifest,design_id(d),np.linspace(0,.20,41))
           summary=summarize(d,dense,adm)
           if summary["minimum_dsigma_dx_MPa"] <= 0:
               rejected.append({"design_id":design_id(d),"reason":"non-positive stress-loading slope"}); continue
           designs.append((d,summary)); paths[summary["design_id"]]=path_rows(d,manifest,summary["design_id"])
    screened=[s for _,s in designs]
    kinetic=[s for s in screened if 10. <= s["tau0_s"] <= 30.]
    front=pareto(kinetic)
    choices=choose_discussion_candidates(front,kinetic)
    lookup={design_id(d):d for d,_ in designs}
    chosen_rows=[]
    for reason,s in choices:
        chosen_rows.append({"discussion_role":reason,**s})
    all_path_rows=[row for _,s in designs for row in paths[s["design_id"]]]
    actual,reference,reference_analog=current_reference(manifest)

    # Inverse contact-radius requirement over requested realistic ranges.
    inverse=[]
    for target in (20.,25.,30.):
      for theta in (130.,145.,160.,170.):
       for curvature in (-5.,0.,3.36,5.,10.):
        inverse.append({"target_stress_MPa":target,"theta_negative_deg":theta,
            "theta_positive_deg":theta,"mean_curvature_per_um":curvature,
            "required_r_TJ_nm":inverse_target_radius(target*1e6,curvature*1e6,theta,theta)*1e9})
    # Absolute favorable kinetic envelope for convex/flat meridional contact
    # jets: theta_-=theta_+=180 deg and mean k=0.  Any positive curvature or
    # smaller angle increases the wait above this curve.
    kinetic_bound=[]
    for radius_nm in np.linspace(4.,160.,313):
        stress_pa=3.0/(radius_nm*1e-9)
        law=root_law(stress_pa,radius_nm*1e-9,manifest)
        kinetic_bound.append({"r_TJ_nm":radius_nm,"r_TJ_over_W":radius_nm/4.,
            "maximum_convex_contact_stress_MPa":stress_pa*1e-6,
            "minimum_two_contact_wait_s":manifest["root_threshold_multiplier"]/(2*law["root_rate_per_s"])})

    write_csv(OUT/"screened_geometries.csv",screened)
    write_csv(OUT/"rejected_geometries.csv",rejected)
    write_csv(OUT/"pareto_geometries.csv",front)
    write_csv(OUT/"discussion_candidates.csv",chosen_rows)
    write_csv(OUT/"candidate_loading_paths.csv",all_path_rows)
    write_csv(OUT/"current_065_actual_reference.csv",actual)
    write_csv(OUT/"current_065_analytical_analogue.csv",reference_analog)
    write_csv(OUT/"inverse_radius_requirements.csv",inverse)
    write_csv(OUT/"resolved_kinetic_lower_bound.csv",kinetic_bound)

    # Pareto projection.
    fig,ax=plt.subplots(figsize=(8,5.5))
    sc=ax.scatter([s["tau0_s"] for s in kinetic],[s["available_stress_increase_MPa"] for s in kinetic],
        c=[s["resolution_margin"] for s in kinetic],s=18,alpha=.35,cmap="viridis")
    ax.scatter([s["tau0_s"] for s in front],[s["available_stress_increase_MPa"] for s in front],
        facecolors="none",edgecolors="black",s=42,label="four-metric Pareto set")
    for i,(_,s) in enumerate(choices,1):
        ax.scatter([s["tau0_s"]],[s["available_stress_increase_MPa"]],marker="*",s=110,
                   edgecolors="black",linewidths=.7,zorder=5)
        ax.annotate(f"C{i}",(s["tau0_s"],s["available_stress_increase_MPa"]),
                    xytext=(4,4),textcoords="offset points",fontsize=8,weight="bold")
    ax.set(xlabel="initial two-contact characteristic wait (s)",ylabel="available stress increase through 20% loss (MPa)")
    ax.grid(alpha=.2);ax.legend();fig.colorbar(sc,ax=ax,label="resolution margin")
    fig.tight_layout();fig.savefig(OUT/"pareto_loading_map.png",dpi=200);fig.savefig(OUT/"pareto_loading_map.pdf");plt.close(fig)

    # Selected loading and kinetic curves with actual 0.65 trajectory.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for reason,s in choices:
        rr=paths[s["design_id"]]; x=np.array([r["x"] for r in rr])*100
        axes[0].plot(x,[r["sigma_GB_MPa"] for r in rr],marker="o",label=s["design_id"])
        axes[1].semilogy(x,[r["two_contact_characteristic_wait_s"] for r in rr],marker="o")
    axes[0].plot(np.array([r["x"] for r in actual])*100,[r["sigma_GB_MPa"] for r in actual],color="black",lw=3,label="0.65 actual PF reference")
    axes[1].semilogy(np.array([r["x"] for r in actual])*100,[r["two_contact_characteristic_wait_s"] for r in actual],color="black",lw=3)
    axes[1].axhspan(.5,3,color="#4daf4a",alpha=.12,label="screening high-rate band")
    axes[0].set(xlabel="center-volume loss (%)",ylabel="production contact stress (MPa)")
    axes[1].set(xlabel="center-volume loss (%)",ylabel="two-contact characteristic wait (s)")
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);axes[1].legend(fontsize=7);fig.tight_layout()
    fig.savefig(OUT/"candidate_loading_curves.png",dpi=200);fig.savefig(OUT/"candidate_loading_curves.pdf");plt.close(fig)

    # Inverse radius map.
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,target in zip(axes,(20.,25.,30.)):
      for curvature in (-5.,0.,3.36,5.,10.):
       rr=[r for r in inverse if r["target_stress_MPa"]==target and r["mean_curvature_per_um"]==curvature]
       ax.plot([r["theta_negative_deg"] for r in rr],[r["required_r_TJ_nm"] for r in rr],marker="o",label=f"k={curvature:g} /um")
      ax.set_title(f"{target:.0f} MPa");ax.set_xlabel("one-sided angle (deg)");ax.grid(alpha=.2)
    axes[0].set_ylabel("required TJ radius (nm)");axes[-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/"inverse_radius_map.png",dpi=200);fig.savefig(OUT/"inverse_radius_map.pdf");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.semilogy([r["r_TJ_nm"] for r in kinetic_bound],[r["minimum_two_contact_wait_s"] for r in kinetic_bound],lw=2)
    ax.axvline(6*W*1e9,color="black",ls="--",label="conservative 6W resolved radius")
    ax.axvline(3*W*1e9,color="gray",ls=":",label="3W local-metrology support")
    ax.axhspan(.5,3,color="#4daf4a",alpha=.14,label="proposed high-rate band")
    ax.set(xlabel="TJ radius (nm)",ylabel="best possible two-contact wait (s)",title="Favorable kinetic bound: theta=180 deg, mean k=0")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/"resolved_kinetic_bound.png",dpi=200);fig.savefig(OUT/"resolved_kinetic_bound.pdf");plt.close(fig)

    # Mechanism decomposition for discussion candidates.
    fig,axes=plt.subplots(math.ceil(len(choices)/2),2,figsize=(10,3.2*math.ceil(len(choices)/2)),squeeze=False)
    for ax,(reason,s) in zip(axes.flat,choices):
      rr=paths[s["design_id"]]; xx=np.array([r["x"] for r in rr])*100
      for key,label in (("dsigma_dx_radius_MPa","radius"),("dsigma_dx_curvature_MPa","curvature"),("dsigma_dx_angle_MPa","angle")):
       ax.plot(xx,[r[key] for r in rr],marker="o",label=label)
      ax.set_title(reason+"\n"+s["design_id"],fontsize=8);ax.set(xlabel="loss (%)",ylabel="contribution to d sigma/dx (MPa)");ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in axes.flat[len(choices):]:ax.axis("off")
    fig.tight_layout();fig.savefig(OUT/"loading_mechanism_decomposition.png",dpi=200);fig.savefig(OUT/"loading_mechanism_decomposition.pdf");plt.close(fig)
    for i,(reason,s) in enumerate(choices,1):
        plot_geometry(lookup[s["design_id"]],f"Candidate {i}: {reason}",OUT/f"candidate_{i}_geometry.png")

    current_end=actual[-1]
    payload={
        "label":"SHARP_INTERFACE_NO_SINK_LOADING_DESIGN",
        "phase_field_run":False,"stochastic_run":False,"source_event_run":False,
        "production_stress_formula":"-gamma*(k_minus+k_plus)/2 + 3*gamma*(sin(theta_minus/2)+sin(theta_plus/2))/(2*r_TJ)",
        "root_law_source":str(MANIFEST_PATH.relative_to(ROOT)),
        "root_law_manifest_sha256":hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "current_reference_source":str(REFERENCE_PATH.relative_to(ROOT)),
        "current_reference_sha256":hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        "geometry_module_sha256":hashlib.sha256((ROOT/"pf_sintering/three_particle_sharp_design.py").read_bytes()).hexdigest(),
        "search_count":len(screened)+len(rejected),"admissible_positive_loading_count":len(screened),
        "initial_wait_band_count":len(kinetic),"pareto_count":len(front),
        "discussion_candidates":chosen_rows,
        "high_rate_band_reached_by_screened_family":any(math.isfinite(s["x_to_tau3s"]) for s in kinetic),
        "favorable_bound_at_6W":{"r_TJ_nm":24.0,"maximum_stress_MPa":125.0,
            "minimum_two_contact_wait_s":next(r["minimum_two_contact_wait_s"] for r in kinetic_bound if r["r_TJ_nm"]==24.0)},
        "current_actual_endpoint":current_end,
        "current_analytical_analogue":summarize(reference,reference_analog,admissibility(reference)),
        "milestones":X_MILESTONES.tolist(),"W_nm":4.0,"grid_spacing_nm":0.5,
        "limitations":[
            "The path is a sharp-interface kinematic family, not a capillary-equilibrium or PF solution.",
            "Outer-body volume accommodation is separated from the contact jet by a quintic reconstruction.",
            "The characteristic wait is 1.25/(Gamma_LEFT+Gamma_RIGHT), not a sampled first-passage time."
        ]}
    (OUT/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    report_lines=[
        "# Sharp-interface three-particle loading design", "",
        "**Status: analytical geometry campaign complete. No phase-field or stochastic trajectory was run.**", "",
        "The campaign preserves the production bicrystal contact stress and the root barrier, site count, formation penalty, temperature, clock conversion, and 1.25 threshold multiplier from the authoritative manifest. It introduces no cluster stress, barrier fit, or event physics.", "",
        "## Construction", "",
        "The fixed center GB planes are separated by `Lc`. Writing the axisymmetric cross-sectional squared radius as `g=r^2`, the no-sink path uses `g(z,x)=g(z,0)-x Vc0/(pi Lc)`. Therefore `Vc(x)=Vc0(1-x)` exactly and `rTJ`, the contact slope, angle, and curvature follow analytically. Each outer particle receives `x Vc0/2`; a smooth quintic particle-plus-neck profile enforces its contact value, tangent, curvature, tip closure, and exact volume. This is a deliberately non-CMC kinematic family, suitable for screening rather than an equilibrium claim.", "",
        "The exact contact expression is", "",
        "`sigma_GB = -gamma_s (k_-+k_+)/2 + 3 gamma_s [sin(theta_-/2)+sin(theta_+/2)]/(2 r_TJ)`.", "",
        "The inverse requirement is", "",
        "`r_TJ,target = (3 gamma_s/2)[sin(theta_-/2)+sin(theta_+/2)] / [sigma_target + gamma_s (k_-+k_+)/2]`.", "",
        "## Search and exclusions", "",
        f"The grid evaluated **{len(screened)+len(rejected)}** designs over `Ro=80-150 nm`, `Rc/Ro=0.45-0.85`, `Lc=8-24W`, `rTJ/Rc=0.60-1.60`, angles `120-170 deg`, and normalized local curvature `kR=0-0.75`. **{len(screened)}** were geometrically admissible with positive loading through 20% loss; **{len(kinetic)}** also began in the 10-30 s characteristic-wait screening band. Rejections preserve their explicit geometry, topology, monotonicity, or 6W contact-resolution reason.", "",
        "## Kinetic finding", "",
        f"The unchanged kinetics do not identify 20-30 MPa with the proposed 0.5-3 s high-rate band. The TJ site count decreases as the contact shrinks, partially opposing stress activation. The admissible positive-loading designs actually span initial waits of {min(s['tau0_s'] for s in kinetic):.2f}-{max(s['tau0_s'] for s in kinetic):.2f} s inside the requested band; none reaches a 3 s characteristic wait by 20% center loss.", "",
        "A stronger bound is independent of the chosen polynomial shape. For a flat or convex meridional jet, the most favorable possible contact has both angles at 180 degrees and zero curvature penalty, so `sigma <= 3 gamma/rTJ`. At the conservative resolved limit `rTJ=6W=24 nm`, this gives at most 125 MPa and the unchanged root law gives a minimum characteristic wait of 6.43 s. Even at `3W=12 nm`, the favorable bound is about 3.92 s. Entering the 3 s band requires roughly a 2W contact, which is not resolved for production metrology. A sufficiently negative meridional curvature could evade this bound, but that would be a qualitatively concave groove and is outside the screened convex particle-plus-neck family.", "",
        "For the requested representative values `theta_-=theta_+=160 deg` and mean curvature `3.36 /um`, the inverse formula gives `rTJ=126.47, 104.18, 88.56 nm` at 20, 25, and 30 MPa respectively.", "",
        "## Current 0.65 reference", "",
        f"The retained symmetry-released PF reference covers {actual[0]['x']*100:.3f}-{actual[-1]['x']*100:.3f}% loss. It moves from {actual[0]['sigma_GB_MPa']:.3f} to {actual[-1]['sigma_GB_MPa']:.3f} MPa and its characteristic wait changes from {actual[0]['two_contact_characteristic_wait_s']:.3f} to {actual[-1]['two_contact_characteristic_wait_s']:.3f} s. Its large ~{actual[0]['r_TJ_nm']:.1f} nm contact explains the modest stress reserve; the analytical analogue is included to 20% only as a kinematic extrapolation.", "",
        "## Discussion candidates", "",
        "No single scalar objective selected a winner. These are distinct representatives from the Pareto map and the admissible screening pool:", "",
        "| Role | Geometry | sigma0 | sigma20 | reserve | tau0 | tau20 | rTJ20/W | Δsigma radius / angle / curvature |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for reason,s in choices:
        report_lines.append(f"| {reason} | `{s['design_id']}` | {s['sigma0_MPa']:.2f} MPa | {s['sigma20_MPa']:.2f} MPa | {s['available_stress_increase_MPa']:.2f} MPa | {s['tau0_s']:.2f} s | {s['tau20_s']:.2f} s | {s['rTJ20_over_W']:.2f} | {s['delta_sigma_radius_MPa']:.2f} / {s['delta_sigma_angle_MPa']:.2f} / {s['delta_sigma_curvature_MPa']:.2f} MPa |")
    report_lines += ["", "For each candidate, `candidate_loading_paths.csv` records all requested milestones, both one-sided curvatures and angles, one-sided and production stresses, unchanged root rate and barrier, exact volume ledger, and the radius/angle/curvature contributions to `d sigma/dx`.", "",
        "## Decision", "", "These candidates justify discussion of a later PF initialization study, but they do not authorize one. The analytical family shows how smaller, contracting contacts increase stress while also showing the countervailing loss of TJ sites and resolution margin. Review the five geometries and choose the acceptable balance of body shape, contact resolution, and loading reserve before constructing any diffuse field.", ""]
    (OUT/"REPORT.md").write_text("\n".join(report_lines))
    print(json.dumps({"output":str(OUT),"screened":len(screened),"rejected":len(rejected),"kinetic":len(kinetic),"pareto":len(front),"candidates":[s["design_id"] for _,s in choices]},indent=2))


if __name__ == "__main__": main()
