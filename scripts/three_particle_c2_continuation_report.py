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
    for pause in pauses:
        match = next((row for row in recoveries
                      if int(row["event_number"]) == int(pause["event_number"]) and
                      float(row["q_over_b"]) == float(pause["q_over_b"]) and
                      float(row["time_s"]) >= float(pause["time_s"])), None)
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
        next_root = roots.get(aid + 1)
        if next_root is None:
            reload_rows = [entry for entry in history
                           if float(entry["time_s"]) >= float(end["time_s"])]
        else:
            reload_rows = [entry for entry in history
                           if float(end["time_s"]) <= float(entry["time_s"]) <=
                           float(next_root["time_s"])]
        decomp = root["root_rate_decomposition"][contact]
        detail = dict(row)
        detail.update({
            "root_hazard": root["hazards"][contact],
            "root_threshold": root["thresholds"][contact],
            "opposite_root_hazard": root["hazards"][opposite],
            "opposite_root_threshold": root["thresholds"][opposite],
            "delta_ln_Gamma_barrier": decomp["delta_ln_Gamma_barrier"],
            "delta_ln_Gamma_sites": decomp["delta_ln_Gamma_sites"],
            "minimum_transport_affinity_MPa": min(affinities) if affinities else None,
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


def sequence_plot(avalanches, out):
    ids = [row["avalanche_id"] for row in avalanches]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             constrained_layout=True)
    axes[0].plot(ids, [row["root_selected_stress_MPa"] for row in avalanches],
                 "o-", color="#31688e")
    axes[0].set_ylabel("Root stress (MPa)")
    axes[1].plot(ids, [row["event_count"] for row in avalanches],
                 "o-", color="#d1495b")
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
    sequence_plot(avalanches, args.out)
    snapshots, movie = make_movie(args.run, source, args.out)
    numerical = snapshot_audit(snapshots, history)
    native = immutable_milestone_audit(args.run, len(events))
    baseline = cycle_group_summary([row for row in avalanches
                                    if row["avalanche_id"] <= 2])
    later = cycle_group_summary([row for row in avalanches
                                 if row["avalanche_id"] > 2])
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
                 "continuation_completion.json"):
        shutil.copy2(args.run / "restart_boundaries" / name, args.out / name)
    report = f"""# C2 stochastic continuation

**Status: {summary['campaign_status']}.** This report appends the same seed-20260915 stochastic lineage from the exact post-avalanche-2 checkpoint. The immutable two-cycle report remains unchanged.

The full lineage completed **{len(avalanches)} avalanches** and **{len(events)} one-b events**, ending at **{history[-1]['time_s']:.9g} s**. Cycles 1-2 had root stresses {baseline['root_stress_MPa']} MPa and avalanche sizes {baseline['avalanche_sizes']}. Later cycles had root stresses {later['root_stress_MPa'] if later else []} MPa and avalanche sizes {later['avalanche_sizes'] if later else []}.

The comparison tables and sequence panel report stress drift, alternating-contact retention, reload maxima, accumulated center loss and strain, barrier/site contributions, transport bottlenecks, and the actual accepted quota at every native quarter-event checkpoint. RNG, root hazards, thresholds, controller state, and physical time were restored from the audited atomic boundary. No threshold was redrawn manually, no contact was forced, and the microscopic physics remained unchanged.
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
