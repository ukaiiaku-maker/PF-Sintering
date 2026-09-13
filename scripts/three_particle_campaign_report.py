"""Create final plots, avalanche table, movie, and interpretation."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


PHASE_COLORS = {
    "RELOAD": "#31688e", "ROOT_CROSSING": "#35b779",
    "ACTIVE_ONE_B": "#d1495b", "EVENT_TRANSPORT_PAUSED": "#b2182b",
    "EVENT_TRANSPORT_RECOVERY": "#ef8a62", "ONE_B_COMPLETE": "#f28e2b",
    "SOURCE_WINDOW_OPEN": "#9c6ade", "FACILITATED_WINDOW": "#8064a2",
    "CHILD_CROSSING": "#e15759", "AVALANCHE_EXTINCT_REPINNED": "#59a14f",
    "POST_TRANSIENT_NEW_TRAJECTORY": "#777777"}


def contact_value(row, contact, key):
    return float(row["contacts"][contact][key])


def avalanche_table(history):
    roots = [row for row in history if row["phase"] == "ROOT_CROSSING"]
    extinctions = {int(row["avalanche_id"]): row for row in history
                   if row["phase"] == "AVALANCHE_EXTINCT_REPINNED"}
    table = []
    previous_end = float(history[0]["time_s"])
    for root in roots:
        aid = int(root["avalanche_id"])
        if aid not in extinctions:
            continue
        end = extinctions[aid]
        contact = root["contact"]
        opposite = "RIGHT" if contact == "LEFT" else "LEFT"
        segment = [row for row in history
                   if int(row["avalanche_id"]) == aid and
                   root["time_s"] <= row["time_s"] <= end["time_s"]]
        completed = [row for row in segment if row["phase"] == "ONE_B_COMPLETE"]
        next_roots = [row for row in roots if row["time_s"] > end["time_s"]]
        next_root = next_roots[0] if next_roots else None
        table.append(dict(
            avalanche_id=aid, root_contact=contact,
            pre_root_center_loss_fraction=(
                1-root["center_volume_m3"]/history[0]["center_volume_m3"]),
            pre_root_local_stress_MPa=contact_value(root, contact, "sigma_local_Pa")/1e6,
            root_waiting_time_s=float(root["time_s"])-previous_end,
            avalanche_size=len(completed),
            avalanche_duration_s=float(end["time_s"])-float(root["time_s"]),
            local_stress_drop_MPa=(contact_value(root, contact, "sigma_local_Pa")-
                                   contact_value(end, contact, "sigma_local_Pa"))/1e6,
            opposite_contact_stress_change_MPa=(
                contact_value(end, opposite, "sigma_local_Pa")-
                contact_value(root, opposite, "sigma_local_Pa"))/1e6,
            production_strain_increment=(end["production_densification_strain"]-
                                         root["production_densification_strain"]),
            geometric_strain_increment=(end["geometric_chain_strain"]-
                                        root["geometric_chain_strain"]),
            center_volume_change_during_avalanche_m3=(
                end["center_volume_m3"]-root["center_volume_m3"]),
            center_volume_change_during_following_reload_m3=(
                next_root["center_volume_m3"]-end["center_volume_m3"]
                if next_root else None)))
        previous_end = float(end["time_s"])
    return table


def phase_volume_changes(history):
    """Accumulate observed center-volume changes by the ending phase."""
    groups = {
        "passive_reload": {"RELOAD", "ROOT_CROSSING"},
        "active_event_transit": {
            "ACTIVE_ONE_B", "EVENT_TRANSPORT_PAUSED",
            "EVENT_TRANSPORT_RECOVERY", "ONE_B_COMPLETE"},
        "facilitated_source_window": {
            "SOURCE_WINDOW_OPEN", "FACILITATED_WINDOW", "CHILD_CROSSING",
            "AVALANCHE_EXTINCT_REPINNED"},
    }
    changes = {name: 0.0 for name in groups}
    for previous, current in zip(history, history[1:]):
        delta = float(current["center_volume_m3"])-float(previous["center_volume_m3"])
        for name, phases in groups.items():
            if current["phase"] in phases:
                changes[name] += delta
                break
    return changes


def make_plots(history, out):
    t0 = float(history[0]["time_s"])
    t = np.array([row["time_s"]-t0 for row in history])
    left = np.array([contact_value(row, "LEFT", "sigma_local_Pa")/1e6
                     for row in history])
    right = np.array([contact_value(row, "RIGHT", "sigma_local_Pa")/1e6
                      for row in history])
    center = np.array([row["center_particle_mean_local_Pa"]/1e6 for row in history])
    cluster = np.array([row["cluster_area_weighted_local_Pa"]/1e6 for row in history])
    volume0 = float(history[0]["center_volume_m3"])
    loss = np.array([1-row["center_volume_m3"]/volume0 for row in history])
    qstrain = np.array([row["production_densification_strain"] for row in history])
    gstrain = np.array([row["geometric_chain_strain"] for row in history])
    hleft = np.array([row["H_over_Hstar"]["LEFT"] for row in history])
    hright = np.array([row["H_over_Hstar"]["RIGHT"] for row in history])
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    ax = axes[0, 0]
    for values, label, style in [(left,"LEFT","-"),(right,"RIGHT","-"),
                                  (center,"center mean","--"),(cluster,"cluster weighted",":")]:
        ax.plot(t, values, style, label=label, lw=1.5)
    ax.set(xlabel="Time since production source (s)", ylabel="Stress (MPa)",
           title="Contact-resolved and aggregate stress")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax = axes[0, 1]; ax.plot(t, loss*100, color="#4e79a7")
    ax.set(xlabel="Time since production source (s)", ylabel="Center-volume loss (%)",
           title="Center volume"); ax.grid(alpha=.2)
    descendant = np.array([
        (np.nan if row.get("descendant_H_over_Hstar") is None
         else float(row["descendant_H_over_Hstar"])) for row in history])
    ax = axes[1, 0]; ax.plot(t,hleft,label="root LEFT"); ax.plot(t,hright,label="root RIGHT")
    ax.plot(t, descendant, "--", color="#d1495b", label="active descendant")
    ax.axhline(1,color="k",lw=.7,ls="--"); ax.set_ylim(bottom=0)
    ax.set(xlabel="Time since production source (s)", ylabel="Hazard H/H*",
           title="Independent root and descendant clocks"); ax.legend(); ax.grid(alpha=.2)
    ax = axes[1, 1]; ax.plot(t,qstrain,label="production quota strain")
    ax.plot(t,gstrain,label="geometric strain")
    ax.set(xlabel="Time since production source (s)", ylabel="Strain",
           title="Production and geometric strain"); ax.legend(); ax.grid(alpha=.2)
    ax = axes[2, 0]
    phases = sorted(set(row["phase"] for row in history))
    for phase in phases:
        keep=np.array([row["phase"]==phase for row in history])
        ax.scatter(loss[keep]*100,left[keep],s=9,label=phase,
                   color=PHASE_COLORS.get(phase,"#999999"),alpha=.8)
    ax.set(xlabel="Center-volume loss from production source (%)",
           ylabel="LEFT stress (MPa)",title="Phase-resolved loading/event path")
    ax.legend(fontsize=6,ncol=2); ax.grid(alpha=.2)
    ax=axes[2,1]; ax.plot(t,left,label="LEFT"); ax.plot(t,right,label="RIGHT")
    for row in history:
        if row["phase"] in ("ROOT_CROSSING","AVALANCHE_EXTINCT_REPINNED"):
            ax.axvline(row["time_s"]-t0,color=PHASE_COLORS[row["phase"]],lw=.8,alpha=.7)
    ax.set(xlabel="Time since production source (s)",ylabel="Stress (MPa)",
           title="Root and avalanche-extinction markers"); ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out/"campaign_diagnostics.png",dpi=180)
    fig.savefig(out/"campaign_diagnostics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True,
                             constrained_layout=True)
    for ax, values, contact in zip(axes, (left, right), ("LEFT", "RIGHT")):
        for phase in phases:
            keep = np.array([row["phase"] == phase for row in history])
            ax.scatter(loss[keep]*100, values[keep], s=10, label=phase,
                       color=PHASE_COLORS.get(phase, "#999999"), alpha=.8)
        ax.set(xlabel="Center-volume loss from production source (%)",
               ylabel=f"{contact} local stress (MPa)",
               title=f"{contact} phase-resolved path")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(out/"contact_stress_vs_center_volume_loss.png", dpi=180)
    fig.savefig(out/"contact_stress_vs_center_volume_loss.pdf")
    plt.close(fig)


def make_movie(run, out):
    index_path=run/"frames/index.json"
    if not index_path.exists(): return None
    index=json.loads(index_path.read_text())
    initial=Path("runs/three_particle_production_065/production_initial_2pct.npz")
    with np.load(initial) as data:
        z_axis=data["z"].copy(); r_axis=data["r_c"].copy()
    sources=[initial]
    labels=[dict(time_s=305.76900020092603,phase="SELECTED_SOURCE",q_over_b=0.)]
    for item in index:
        sources.append(Path(item["path"])); labels.append(item)
    fig,ax=plt.subplots(figsize=(9,4.5))
    def draw(i):
        ax.clear()
        with np.load(sources[i]) as data:
            f=data["fields"][0] if "fields" in data else data["f"]
            z=data["z"] if "z" in data else z_axis
            r=data["r_c"] if "r_c" in data else r_axis
        stride=max(1,f.shape[0]//500)
        ax.imshow(f[::stride].T,origin="lower",aspect="auto",vmin=0,vmax=1,
                  extent=[z[0]*1e9,z[-1]*1e9,r[0]*1e9,r[-1]*1e9],cmap="viridis")
        row=labels[i]
        ax.set(title=f"{row['phase']}  t={row['time_s']:.4g} s  q/b={row.get('q_over_b',0):.3f}",
               xlabel="z (nm)",ylabel="r (nm)")
    from matplotlib.animation import FuncAnimation
    animation=FuncAnimation(fig,draw,frames=len(sources),interval=450,repeat=True)
    target=out/"morphology_cycle.gif"
    animation.save(target,writer=PillowWriter(fps=3))
    plt.close(fig)
    return str(target)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    history=json.loads((args.run/"history.json").read_text())
    table=avalanche_table(history)
    make_plots(history,args.out); movie=make_movie(args.run,args.out)
    if table:
        with (args.out/"avalanches.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    completed=max(row["completed_avalanches"] for row in history)
    volume_changes = phase_volume_changes(history)
    payload=dict(label="THREE_PARTICLE_STOCHASTIC_RENEWAL_CAMPAIGN",
                 run=str(args.run),records=len(history),completed_avalanches=completed,
                 avalanches=table,movie=movie,
                 center_volume_change_by_phase_m3=volume_changes,
                 symmetry_enforcement_enabled=False,physics_parameters_unchanged=True)
    (args.out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    cycle_text = (f"The trajectory contains {completed} complete avalanche cycle"
                  f"{'s' if completed != 1 else ''}. "
                  if completed else
                  "The trajectory has not yet completed an avalanche cycle, so cycle-level conclusions remain provisional. ")
    interpretation=(
        cycle_text+
        "The full-domain three-particle trajectory resolves independent LEFT and RIGHT contact loading. "
        "It uses the bicrystal local-stress definition, root barrier, stochastic threshold law, one-b transfer, "
        "descendant facilitation, and 9 ms source lifetime. A selected-contact event can relax its local stress "
        "while the opposite contact and area-weighted cluster stress remain elevated. In the bicrystal this "
        "nucleation-limited mechanism produces a strongly oscillatory response; in the cluster, asynchronous "
        "contact loading can retain high average stress while strain accumulates as the center particle evolves. "
        "The phase-resolved center-volume changes in summary.json separate passive reload, active transit, and "
        "facilitated windows. They must be assessed over complete renewal cycles alongside quota and geometric "
        "strain; quota strain alone does not show when physical densification occurs.")
    (args.out/"MANUSCRIPT_INTERPRETATION.md").write_text(interpretation+"\n")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
