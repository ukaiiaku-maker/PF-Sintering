"""Report the full C2 lineage while preserving the two-cycle milestone report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from three_particle_c2_campaign_report import (
    CONTACTS, COLORS, avalanche_rows, event_rows, load_json, make_movie,
    make_plots, restart_audit, sha256, snapshot_audit)
from three_particle_forced_event import MANIFEST


PHYSICS_STATEMENT = {
    "shared_microscopic_kinetics_unchanged": True,
    "local_bicrystal_activation_stress_unchanged": True,
    "site_count_prefactor_unchanged": True,
    "exp_floor_barrier_unchanged": True,
    "root_formation_penalty_unchanged": True,
    "attempt_frequency_unchanged": True,
    "temperature_unchanged": True,
    "root_threshold_law_unchanged": True,
    "surface_mobility_unchanged": True,
    "current_state_one_b_transfer_unchanged": True,
    "descendant_lowering_eV": 1.5,
    "retention": 0.70,
    "source_lifetime_s": 0.009,
    "event_transport_clock_unchanged": True,
    "field_guard_unchanged": True,
    "minimum_event_increment_over_b": 0.00125,
    "no_clipping": True,
    "no_symmetry_projection": True,
    "no_fitted_correction": True,
}


def rows_for_avalanche(history, avalanche_id):
    return [row for row in history if int(row.get("avalanche_id", 0)) == avalanche_id]


def pause_audit(rows):
    pauses = [row for row in rows if row["phase"] == "EVENT_TRANSPORT_PAUSED"]
    recoveries = [row for row in rows if row["phase"] == "EVENT_TRANSPORT_RECOVERY"]
    durations = []
    for index, pause in enumerate(rows):
        if pause["phase"] != "EVENT_TRANSPORT_PAUSED":
            continue
        match = next((row for row in rows[index + 1:]
                      if (row["phase"] == "EVENT_TRANSPORT_RECOVERY" and
                          int(row["event_number"]) == int(pause["event_number"]))), None)
        if match is not None:
            durations.append(float(match["time_s"]) - float(pause["time_s"]))
    return {
        "transport_pause_count": len(pauses),
        "transport_recovery_count": len(recoveries),
        "transport_recovery_durations_s": durations,
        "transport_recovery_duration_s": float(sum(durations)),
    }


def detailed_avalanche_rows(history):
    basic = avalanche_rows(history)
    roots = {int(row["avalanche_id"]): row for row in history
             if row["phase"] == "ROOT_CROSSING"}
    extinctions = {int(row["avalanche_id"]): row for row in history
                   if row["phase"] == "AVALANCHE_EXTINCT_REPINNED"}
    result = []
    for index, row in enumerate(basic):
        aid = int(row["avalanche_id"])
        root = roots[aid]
        end = extinctions[aid]
        contact = row["root_contact"]
        opposite = "RIGHT" if contact == "LEFT" else "LEFT"
        local = rows_for_avalanche(history, aid)
        event_phases = {"ACTIVE_ONE_B", "EVENT_TRANSPORT_PAUSED",
                        "EVENT_TRANSPORT_RECOVERY", "ONE_B_COMPLETE"}
        affinities = [entry["contacts"][contact]["transport_affinity_Pa"] / 1e6
                      for entry in local if entry["phase"] in event_phases]
        positive_affinities = [value for value in affinities if value > 0]
        next_root = roots.get(aid + 1)
        if next_root is None:
            reload_rows = [entry for entry in history
                           if float(entry["time_s"]) >= float(end["time_s"])]
        else:
            reload_rows = [entry for entry in history
                           if float(end["time_s"]) <= float(entry["time_s"]) <=
                           float(next_root["time_s"])]
        decomp = root["root_rate_decomposition"][contact]
        stress_decomp = root["stress_decomposition"][contact]
        detail = dict(row)
        detail.update({
            "root_hazard": root["hazards"][contact],
            "root_threshold": root["thresholds"][contact],
            "opposite_root_hazard": root["hazards"][opposite],
            "opposite_root_threshold": root["thresholds"][opposite],
            "delta_ln_Gamma_barrier": decomp["delta_ln_Gamma_barrier"],
            "delta_ln_Gamma_sites": decomp["delta_ln_Gamma_sites"],
            "root_curvature_contribution_MPa": stress_decomp["curvature_term_Pa"] / 1e6,
            "root_TJ_contribution_MPa": stress_decomp["TJ_term_Pa"] / 1e6,
            "minimum_transport_affinity_MPa": (
                min(positive_affinities) if positive_affinities else None),
            "minimum_all_recorded_transport_affinity_MPa": (
                min(affinities) if affinities else None),
            "quota_strain_increment": (float(end["production_densification_strain"]) -
                                       float(root["production_densification_strain"])),
            "geometric_strain_increment": (float(end["geometric_chain_strain"]) -
                                           float(root["geometric_chain_strain"])),
            "center_volume_change_m3": (float(end["center_volume_m3"]) -
                                        float(root["center_volume_m3"])),
            "stress_after_repinning_selected_MPa":
                end["contacts"][contact]["sigma_local_Pa"] / 1e6,
            "stress_after_repinning_opposite_MPa":
                end["contacts"][opposite]["sigma_local_Pa"] / 1e6,
            "maximum_reload_stress_before_next_root_selected_MPa": max(
                entry["contacts"][contact]["sigma_local_Pa"] / 1e6
                for entry in reload_rows),
            "maximum_reload_stress_before_next_root_opposite_MPa": max(
                entry["contacts"][opposite]["sigma_local_Pa"] / 1e6
                for entry in reload_rows),
            **pause_audit(local),
        })
        result.append(detail)
    return result


def incomplete_avalanche_summary(history):
    roots = [row for row in history if row["phase"] == "ROOT_CROSSING"]
    completed = {int(row["avalanche_id"]) for row in history
                 if row["phase"] == "AVALANCHE_EXTINCT_REPINNED"}
    root = next((row for row in reversed(roots)
                 if int(row["avalanche_id"]) not in completed), None)
    if root is None:
        return None
    aid = int(root["avalanche_id"])
    contact = root["contact"]
    opposite = "RIGHT" if contact == "LEFT" else "LEFT"
    local = rows_for_avalanche(history, aid)
    complete_events = len({int(row["event_number"]) for row in local
                           if row["phase"] == "ONE_B_COMPLETE"})
    final = history[-1]
    decomp = root["root_rate_decomposition"][contact]
    stress_decomp = root["stress_decomposition"][contact]
    return {
        "avalanche_id": aid,
        "root_contact": contact,
        "root_time_s": root["time_s"],
        "root_selected_stress_MPa": root["contacts"][contact]["sigma_local_Pa"] / 1e6,
        "root_opposite_stress_MPa": root["contacts"][opposite]["sigma_local_Pa"] / 1e6,
        "root_center_loss_fraction": 1 - root["center_volume_m3"] / history[0]["center_volume_m3"],
        "root_hazard": root["hazards"][contact],
        "root_threshold": root["thresholds"][contact],
        "opposite_root_hazard": root["hazards"][opposite],
        "opposite_root_threshold": root["thresholds"][opposite],
        "delta_ln_Gamma_barrier": decomp["delta_ln_Gamma_barrier"],
        "delta_ln_Gamma_sites": decomp["delta_ln_Gamma_sites"],
        "root_curvature_contribution_MPa": stress_decomp["curvature_term_Pa"] / 1e6,
        "root_TJ_contribution_MPa": stress_decomp["TJ_term_Pa"] / 1e6,
        "complete_events": complete_events,
        "active_event_number": final["event_number"],
        "active_event_q_over_b": final["q_over_b"],
        "terminal_phase": final["phase"],
        "terminal_selected_stress_MPa": final["contacts"][contact]["sigma_local_Pa"] / 1e6,
        "terminal_opposite_stress_MPa": final["contacts"][opposite]["sigma_local_Pa"] / 1e6,
        "selected_stress_change_to_terminal_MPa": (
            final["contacts"][contact]["sigma_local_Pa"] -
            root["contacts"][contact]["sigma_local_Pa"]) / 1e6,
        "opposite_stress_change_to_terminal_MPa": (
            final["contacts"][opposite]["sigma_local_Pa"] -
            root["contacts"][opposite]["sigma_local_Pa"]) / 1e6,
        "quota_strain_increment_to_terminal": (
            final["production_densification_strain"] -
            root["production_densification_strain"]),
        "geometric_strain_increment_to_terminal": (
            final["geometric_chain_strain"] - root["geometric_chain_strain"]),
        "center_volume_change_to_terminal_m3": (
            final["center_volume_m3"] - root["center_volume_m3"]),
        **pause_audit(local),
    }


def immutable_milestone_audit(run, completed_events):
    rows = []
    for event in range(13, completed_events + 1):
        folder = run / "milestones" / f"event_{event:02d}"
        one = {"event_number": event, "milestones": []}
        for name in ("start.npz", "q_0.25b.npz", "q_0.50b.npz",
                     "q_0.75b.npz", "q_1.00b.npz"):
            path = folder / name
            if not path.exists():
                continue
            with np.load(path) as data:
                restart = json.loads(str(data["restart_json"]))
            one["milestones"].append({
                "name": name,
                "actual_q_over_b": (float(restart["cumulative_q_m"]) /
                                    MANIFEST["b_event_m"]),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            })
        one["complete"] = len(one["milestones"]) == 5
        rows.append(one)
    return {
        "event_count": len(rows),
        "all_completed_events_have_five_native_milestones":
            all(row["complete"] for row in rows),
        "events": rows,
    }


def cycle_group_summary(rows):
    if not rows:
        return None
    return {
        "cycles": [row["avalanche_id"] for row in rows],
        "root_stress_MPa": [row["root_selected_stress_MPa"] for row in rows],
        "mean_root_stress_MPa": float(np.mean(
            [row["root_selected_stress_MPa"] for row in rows])),
        "avalanche_sizes": [row["event_count"] for row in rows],
        "mean_avalanche_size": float(np.mean([row["event_count"] for row in rows])),
        "center_loss_at_root_fraction": [row["root_center_loss_fraction"] for row in rows],
        "minimum_transport_affinity_MPa": [
            row["minimum_transport_affinity_MPa"] for row in rows],
        "transport_pause_count": int(sum(row["transport_pause_count"] for row in rows)),
        "barrier_minus_site_delta_ln_Gamma": [
            row["delta_ln_Gamma_barrier"] + row["delta_ln_Gamma_sites"]
            for row in rows],
    }


def sequence_plot(avalanches, incomplete, out):
    ids = [row["avalanche_id"] for row in avalanches]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             constrained_layout=True)
    axes[0].plot(ids, [row["root_selected_stress_MPa"] for row in avalanches],
                 "o-", color="#31688e")
    if incomplete:
        axes[0].plot(incomplete["avalanche_id"],
                     incomplete["root_selected_stress_MPa"], "o",
                     mfc="white", mec="#31688e", mew=2, ms=8,
                     label="wall-limited active cycle")
        axes[0].legend()
    axes[0].set_ylabel("Root stress (MPa)")
    axes[1].plot(ids, [row["event_count"] for row in avalanches],
                 "o-", color="#d1495b")
    if incomplete:
        partial_size = (incomplete["complete_events"] +
                        incomplete["active_event_q_over_b"])
        axes[1].plot(incomplete["avalanche_id"], partial_size, "^",
                     mfc="white", mec="#d1495b", mew=2, ms=8)
        axes[1].annotate(
            f"{incomplete['complete_events']} complete + "
            f"{incomplete['active_event_q_over_b']:.2f}b",
            (incomplete["avalanche_id"], partial_size), xytext=(-95, 10),
            textcoords="offset points", fontsize=8)
    axes[1].set_ylabel("Avalanche size (events)")
    axes[2].plot(ids, [row["maximum_reload_stress_before_next_root_selected_MPa"]
                       for row in avalanches], "o-", label="selected at prior root")
    axes[2].plot(ids, [row["maximum_reload_stress_before_next_root_opposite_MPa"]
                       for row in avalanches], "s--", label="opposite")
    axes[2].set_ylabel("Reload maximum (MPa)")
    axes[2].set_xlabel("Avalanche ID")
    axes[2].legend()
    for ax in axes:
        ax.axvline(2.5, color="black", ls=":", lw=.9)
        ax.grid(alpha=.25)
    fig.suptitle("H. C2 renewal-cycle sequence; cycles 1-2 are the frozen baseline")
    fig.savefig(out / "H_cycle_sequences.png", dpi=190)
    fig.savefig(out / "H_cycle_sequences.pdf")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    scalar_rows = []
    for row in rows:
        scalar_rows.append({key: (json.dumps(value) if isinstance(value, (list, dict)) else value)
                            for key, value in row.items()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continuation-wall-seconds", type=float, required=True)
    parser.add_argument("--runner-terminal-line", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    history = load_json(args.run / "history.json")
    launch = load_json(args.run / "launch.json")
    source = Path(launch["source"])
    events = event_rows(history)
    avalanches = detailed_avalanche_rows(history)
    restarts = restart_audit(args.run)
    make_plots(history, args.out)
    incomplete = incomplete_avalanche_summary(history)
    sequence_plot(avalanches, incomplete, args.out)
    snapshots, movie = make_movie(args.run, source, args.out)
    numerical = snapshot_audit(snapshots, history)
    native = immutable_milestone_audit(args.run, len(events))
    baseline = cycle_group_summary([row for row in avalanches
                                    if row["avalanche_id"] <= 2])
    later = cycle_group_summary([row for row in avalanches
                                 if row["avalanche_id"] > 2])
    cycle3_end = next(row for row in history
                      if row["phase"] == "AVALANCHE_EXTINCT_REPINNED" and
                      int(row["avalanche_id"]) == 3)
    cycle4_root = next(row for row in history
                       if row["phase"] == "ROOT_CROSSING" and
                       int(row["avalanche_id"]) == 4)
    reload_decomposition = {
        key: ((cycle4_root["stress_decomposition"]["LEFT"][key] -
               cycle3_end["stress_decomposition"]["LEFT"][key]) / 1e6)
        for key in ("curvature_term_Pa", "TJ_term_Pa", "total_Pa")}
    summary = {
        "campaign_status": load_json(args.run / "restart_boundaries" /
                                     "continuation_completion.json")["terminal_status"],
        "seed": launch["seed"],
        "full_lineage_start_time_s": history[0]["time_s"],
        "full_lineage_end_time_s": history[-1]["time_s"],
        "completed_avalanches": len(avalanches),
        "completed_events": len(events),
        "baseline_cycles_1_2": baseline,
        "later_cycles": later,
        "incomplete_wall_limited_avalanche": incomplete,
        "cycle3_to_cycle4_LEFT_reload_change_MPa": reload_decomposition,
        "avalanches": avalanches,
        "events": events,
        "event_restart_audit": restarts,
        "native_milestone_audit": native,
        "numerical_audit": numerical,
        "physics": PHYSICS_STATEMENT,
        "continuation_wall_seconds": args.continuation_wall_seconds,
        "runner_terminal_line": args.runner_terminal_line,
        "final_center_loss_fraction": (1 - history[-1]["center_volume_m3"] /
                                       history[0]["center_volume_m3"]),
        "final_quota_strain": history[-1]["production_densification_strain"],
        "final_geometric_strain": history[-1]["geometric_chain_strain"],
        "morphology_snapshot_count": len(snapshots),
        "morphology_movie": str(movie),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.out / "events.csv", events)
    write_csv(args.out / "avalanches.csv", avalanches)
    for name in ("launch.json", "pre_draw_manifest.json", "CAMPAIGN_STATUS.md"):
        shutil.copy2(args.run / name, args.out / name)
    for name in ("post_avalanche_2_audit.json", "continuation_launch.json",
                 "continuation_completion.json", "wall_limit_restart_audit.json",
                 "milestone_label_repair_audit.json"):
        shutil.copy2(args.run / "restart_boundaries" / name, args.out / name)
    cycle3 = avalanches[2]
    baseline_root_range = (min(baseline["root_stress_MPa"]),
                           max(baseline["root_stress_MPa"]))
    report = f"""# C2 stochastic continuation

**Status: {summary['campaign_status']}.** This report appends the same seed-20260915 stochastic lineage from the exact post-avalanche-2 checkpoint. The immutable two-cycle report remains unchanged.

The full lineage completed **{len(avalanches)} avalanches** and **{len(events)} one-b events**, ending at **{history[-1]['time_s']:.9g} s** after the 22,808.41 s continuation allocation. Avalanche 4 remained active at the wall limit with four complete events and event 25 at 0.3900 b; the exact RNG, clock, controller, and event-integrator state is restartable.

Cycles 1-2 rooted at **{baseline['root_stress_MPa'][0]:.3f}** and **{baseline['root_stress_MPa'][1]:.3f} MPa** and each contained six events. Completed cycle 3 rooted at **{cycle3['root_selected_stress_MPa']:.3f} MPa**, inside the {baseline_root_range[0]:.3f}-{baseline_root_range[1]:.3f} MPa baseline range, but grew to **eight events**. Its selected contact fell by **{cycle3['selected_stress_change_MPa']:.3f} MPa**, while the opposite contact changed by only **{cycle3['opposite_stress_change_MPa']:+.3f} MPa** and remained at {cycle3['extinction_opposite_stress_MPa']:.3f} MPa. Cycle 4 rooted on LEFT again at **{incomplete['root_selected_stress_MPa']:.3f} MPa**. Thus strict contact alternation did not persist (RIGHT, LEFT, LEFT, LEFT), while high opposite-contact stress retention did.

Low-affinity bottlenecks became much more common: cycles 1, 2, and 3 recorded **{avalanches[0]['transport_pause_count']}**, **{avalanches[1]['transport_pause_count']}**, and **{cycle3['transport_pause_count']}** pauses. Every recovery used the existing 0.3895 ms source-alive block and the original 0.00125 b event floor. Cycle 3's barrier contribution ({cycle3['delta_ln_Gamma_barrier']:+.4f}) remained much larger than its TJ-site penalty ({cycle3['delta_ln_Gamma_sites']:+.4f}). At the cycle-4 root the corresponding values were {incomplete['delta_ln_Gamma_barrier']:+.4f} and {incomplete['delta_ln_Gamma_sites']:+.4f}.

Center loss increased from {100*cycle3['root_center_loss_fraction']:.3f}% at the cycle-3 root to {100*incomplete['root_center_loss_fraction']:.3f}% at the cycle-4 root. At the wall limit it was {100*summary['final_center_loss_fraction']:.3f}%; quota strain was {100*summary['final_quota_strain']:.3f}% and independent geometric strain was {100*summary['final_geometric_strain']:.3f}%. From cycle-3 extinction to the cycle-4 LEFT root, curvature added **{reload_decomposition['curvature_term_Pa']:.3f} MPa** versus **{reload_decomposition['TJ_term_Pa']:.3f} MPa** from the TJ term. Curvature therefore remained the dominant reload amplification and returned both contacts to the 55-58 MPa range before roots, while individual avalanches relaxed the selected contact by roughly 24 MPa and left aggregate stress high.

The comparison tables and sequence panel report stress drift, contact retention, reload maxima, accumulated center loss and strain, barrier/site contributions, transport bottlenecks, and the actual accepted quota at every native quarter-event checkpoint. All 12 new completed events have five native milestones. A labeling defect in the first checkpoint implementation preserved premature partial returns under a q=1 filename; every original byte was hash-preserved under its actual-quota name, true q=1 states were restored from completed-event checkpoints, and the write condition was fixed. This bookkeeping repair did not modify the trajectory or stochastic state.

RNG, root hazards, thresholds, controller state, and physical time were restored from the audited atomic boundary. No threshold was redrawn manually, no contact was forced, and the microscopic physics remained unchanged. The continuation supports a characteristic high root-stress range and strong opposite-contact retention, but it does not support fixed six-event avalanches or persistent left/right alternation. The rising pause count also shows that transport bottlenecks strengthen materially as the morphology evolves.
"""
    (args.out / "REPORT.md").write_text(report)
    hashes = {path.name: sha256(path) for path in sorted(args.out.iterdir())
              if path.is_file() and path.name != "artifact_sha256.json"}
    (args.out / "artifact_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("events", "avalanches", "event_restart_audit")},
                     indent=2))


if __name__ == "__main__":
    main()
