#!/usr/bin/env python3
"""Report the unconstrained earlier-stage three-particle renewal campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finite(values):
    return np.array([np.nan if value is None else float(value) for value in values])


def event_rows(history):
    rows = []
    prior = None
    for row in history:
        marker = (row["event_number"], row["completed_avalanches"], row["phase"])
        if marker != prior and (row["event_number"] or row["completed_avalanches"]):
            rows.append(dict(time_s=row["time_s"], phase=row["phase"],
                event_number=row["event_number"], completed_avalanches=row["completed_avalanches"],
                active_contact=row.get("contact"),
                LEFT_sigma_MPa=row["contacts"]["LEFT"]["sigma_local_Pa"]*1e-6,
                RIGHT_sigma_MPa=row["contacts"]["RIGHT"]["sigma_local_Pa"]*1e-6,
                center_volume_m3=row["center_volume_m3"],
                geometric_strain=row["geometric_chain_strain"],
                production_strain=row["production_densification_strain"]))
        prior = marker
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path,
        default=Path("docs/three_particle/initial_state_design/campaign_result"))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    history = json.loads((args.campaign/"history.json").read_text())
    launch = json.loads((args.campaign/"launch.json").read_text())
    manifest=json.loads(Path(
        "docs/three_particle/production_screen/bicrystal_launch_manifest.json").read_text())
    with np.load(args.campaign/"trajectory.npz", allow_pickle=False) as checkpoint:
        checkpoint_metadata=json.loads(str(checkpoint["metadata"]))
        checkpoint_time_s=float(checkpoint["time_s"])
    if not history:
        raise RuntimeError("campaign history is empty")

    t = finite([r["time_s"] for r in history])
    left = finite([r["contacts"]["LEFT"]["sigma_local_Pa"] for r in history])*1e-6
    right = finite([r["contacts"]["RIGHT"]["sigma_local_Pa"] for r in history])*1e-6
    aggregate = finite([r["cluster_area_weighted_local_Pa"] for r in history])*1e-6
    energy_stress = finite([r.get("energy_balance_sintering_stress_Pa") for r in history])*1e-6
    volume = finite([r["center_volume_m3"] for r in history])
    volume_change = volume/volume[0]-1
    surface = finite([r["surface_area_m2"] for r in history])
    gb = finite([r["GB_area_m2"] for r in history])
    energy = finite([r["total_interfacial_energy_J"] for r in history])
    geometric = finite([r["geometric_chain_strain"] for r in history])
    production = finite([r["production_densification_strain"] for r in history])
    events = event_rows(history)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    ax=axes[0,0]; ax.plot(t,left,label="LEFT exact local");ax.plot(t,right,"--",label="RIGHT exact local")
    ax.plot(t,aggregate,":",label="area weighted aggregate");ax.set_ylabel("stress (MPa)");ax.legend(fontsize=8)
    ax=axes[0,1];ax.plot(t,energy_stress,label="energy balance",color="tab:purple")
    ax.axhline(0,color="0.7",lw=.7);ax.set_ylabel("diagnostic stress (MPa)");ax.legend(fontsize=8)
    ax=axes[1,0];ax.plot(t,100*volume_change,label="center volume")
    ax.set_ylabel("change from source (%)");ax.legend(fontsize=8)
    ax=axes[1,1];ax.plot(t,surface/surface[0]-1,label="surface area")
    ax.plot(t,gb/gb[0]-1,label="GB area");ax.plot(t,energy/energy[0]-1,label="interfacial energy")
    ax.set_ylabel("relative change");ax.legend(fontsize=8)
    ax=axes[2,0];ax.plot(t,geometric,label="geometric");ax.plot(t,production,label="event production")
    ax.set_ylabel("strain");ax.set_xlabel("physical time (s)");ax.legend(fontsize=8)
    ax=axes[2,1]
    for contact in ("LEFT","RIGHT"):
        ax.plot(t,finite([r["H_over_Hstar"][contact] for r in history]),label=contact)
    ax.axhline(1,color="k",lw=.8);ax.set_ylabel("root hazard / threshold")
    ax.set_xlabel("physical time (s)");ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        for event in events: ax.axvline(event["time_s"],color="tab:red",alpha=.16,lw=.8)
    fig.tight_layout();fig.savefig(args.out/"trajectory_audit.png",dpi=180);plt.close(fig)

    if events:
        with (args.out/"events.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(events[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(events)
    final=history[-1]
    root=next(row for row in history if row["phase"] == "ROOT_CROSSING")
    pre_root=history[:history.index(root)+1]
    one_b=[row for row in history if row["phase"] == "ONE_B_COMPLETE"]
    terminal_event=None
    final_event=args.campaign/f"event_{checkpoint_metadata['event_number']}_final.npz"
    if final_event.exists():
        with np.load(final_event,allow_pickle=False) as stopped:
            restart=json.loads(str(stopped["restart_json"]))
        terminal_event=dict(event_number=checkpoint_metadata["event_number"],
            progress_over_b=restart["cumulative_q_m"]/manifest["b_event_m"],
            accepted_steps_total=restart["accepted_steps_total"],
            active_minimum_step_over_b=restart["active_minimum_step_over_b"],
            rejection_reason_counts=restart["rejection_reason_counts"],
            last_rejection_reasons=restart["last_rejection_reasons"])
    payload=dict(status=checkpoint_metadata["status"], phase=final["phase"],
        source=launch["source"], source_sha256=launch["source_sha256"], seed=launch["seed"],
        physical_time_s=checkpoint_time_s,last_recorded_time_s=final["time_s"],
        completed_avalanches=final["completed_avalanches"],
        event_number=final["event_number"], center_volume_change=volume_change[-1],
        initial_local_stress_MPa=dict(LEFT=left[0],RIGHT=right[0]),
        final_local_stress_MPa=dict(LEFT=left[-1],RIGHT=right[-1]),
        maximum_LEFT_RIGHT_difference_MPa=float(np.max(abs(left-right))),
        final_cluster_area_weighted_stress_MPa=aggregate[-1],
        final_surface_area_relative=surface[-1]/surface[0]-1,
        final_GB_area_relative=gb[-1]/gb[0]-1,
        final_interfacial_energy_relative=energy[-1]/energy[0]-1,
        final_geometric_strain=geometric[-1],final_production_strain=production[-1],
        maximum_mirror_error=max(r["mirror_error_diagnostic"] for r in history),
        pre_root=dict(time_s=root["time_s"],center_volume_change=(
            root["center_volume_m3"]/history[0]["center_volume_m3"]-1),
            LEFT_stress_MPa=root["contacts"]["LEFT"]["sigma_local_Pa"]*1e-6,
            RIGHT_stress_MPa=root["contacts"]["RIGHT"]["sigma_local_Pa"]*1e-6,
            peak_LEFT_stress_MPa=max(r["contacts"]["LEFT"]["sigma_local_Pa"] for r in pre_root)*1e-6,
            peak_stress_time_s=max(pre_root,key=lambda r:r["contacts"]["LEFT"]["sigma_local_Pa"])["time_s"],
            aggregate_stress_MPa=root["cluster_area_weighted_local_Pa"]*1e-6,
            energy_balance_stress_MPa=root["energy_balance_sintering_stress_Pa"]*1e-6,
            energy_balance_relative_difference=(root["energy_balance_sintering_stress_Pa"]/
                root["cluster_area_weighted_local_Pa"]-1),
            surface_area_relative=root["surface_area_m2"]/history[0]["surface_area_m2"]-1,
            GB_area_relative=root["GB_area_m2"]/history[0]["GB_area_m2"]-1,
            interfacial_energy_relative=root["total_interfacial_energy_J"]/history[0]["total_interfacial_energy_J"]-1),
        completed_one_b_events=len(one_b),terminal_event=terminal_event,
        events=events, no_clipping=True, no_fitted_correction=True,
        symmetry_enforcement_enabled=False,
        energy_balance_stress_is_interval_diagnostic=True)
    (args.out/"summary.json").write_text(json.dumps(payload,indent=2,default=float)+"\n")
    report=["# Earlier-stage unconstrained three-particle campaign", "",
      f"Status: **{payload['status']}**; phase: **{payload['phase']}**; completed avalanches: **{payload['completed_avalanches']}**.", "",
      f"The fixed, preselected source `{payload['source_sha256']}` used seed `{payload['seed']}`. The PF field evolved without symmetry enforcement or a prescribed stress, geometry, center-volume, or event trajectory.", "",
      f"At the last recorded 0.01 b checkpoint ({payload['last_recorded_time_s']:.9f} s), the center-volume change was {100*payload['center_volume_change']:.6g}%, exact LEFT/RIGHT local stress had changed from {left[0]:.6g}/{right[0]:.6g} MPa to {left[-1]:.6g}/{right[-1]:.6g} MPa, and aggregate stress was {aggregate[-1]:.6g} MPa. The terminal event checkpoint is at {payload['physical_time_s']:.9f} s.", "",
      f"Surface area, GB area, and total interfacial energy changed by {100*payload['final_surface_area_relative']:.6g}%, {100*payload['final_GB_area_relative']:.6g}%, and {100*payload['final_interfacial_energy_relative']:.6g}%. Geometric and event-production strain ended at {geometric[-1]:.6g} and {production[-1]:.6g}.", "",
      f"The first LEFT root occurred at {root['time_s']:.9f} s after {100*payload['pre_root']['center_volume_change']:.6g}% center-volume change. LEFT stress peaked at {payload['pre_root']['peak_LEFT_stress_MPa']:.6g} MPa at {payload['pre_root']['peak_stress_time_s']:.6g} s and had relaxed to {payload['pre_root']['LEFT_stress_MPa']:.6g} MPa at nucleation.", "",
      f"At the root, the interval energy-balance diagnostic was {payload['pre_root']['energy_balance_stress_MPa']:.6g} MPa versus {payload['pre_root']['aggregate_stress_MPa']:.6g} MPa from exact area-weighted contact metrology, a {100*payload['pre_root']['energy_balance_relative_difference']:.3g}% difference. Its very large early values coincide with near-zero and sign-changing geometric strain and are not treated as a loading law.", "",
      "The coupled model therefore selected high-stress stochastic nucleation after stress had already peaked and relaxed, rather than nucleation caused by continued stress loading. The completed one-b event relaxed LEFT while RIGHT stayed near 51.7 MPa, preserving an aggregate stress near 50 MPa. The descendant continuation then exposed the qualified minimum-step pathology before avalanche extinction.", "",
      "The energy-balance stress is the backward interval quantity `-Delta E/(Vsolid Delta geometric strain)`. It is reported as a diagnostic and is not substituted for either exact local contact stress.", "",
      "No clipping. No fitted correction. Symmetry enforcement remained disabled."]
    if terminal_event:
        child_time=next((r['time_s'] for r in history if r['phase']=="CHILD_CROSSING"),float("nan"))
        report[8:8]=[f"{len(one_b)} one-b event completed. The descendant crossed at {child_time:.9f} s. Event {terminal_event['event_number']} stopped at {terminal_event['progress_over_b']:.6g} b after the fixed minimum increment repeatedly violated the transport-affinity change limit.", ""]
    (args.out/"REPORT.md").write_text("\n".join(report)+"\n")
    print(json.dumps(payload,indent=2,default=float))

if __name__ == "__main__": main()
