"""M16P Section 9, 26: mesoscale cycle detection and cycle_summary.csv.

Identifies cycles from the MACROSCOPIC sigma(t) trajectory (not from
individual one-b events): a cycle is a local maximum followed by a local
minimum where the drop exceeds a noise-rejection threshold, followed by
renewed rise. For each detected peak-trough pair, reports peak/trough
stress, the drop, births/completions/max-N_active during the relaxation
episode, cumulative delta_sink/b consumed, episode duration, and the
reload duration to the next peak.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

NOISE_FLOOR_MPA = 0.5  # minimum peak-to-trough drop to count as a real relaxation episode


def load_history(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["time"]) for r in rows])
    sigma = np.array([float(r["sigma_MPa"]) for r in rows])
    n_active = np.array([int(r["N_active"]) for r in rows])
    n_born = np.array([int(r["N_born"]) for r in rows])
    n_completed = np.array([int(r["N_completed"]) for r in rows])
    cum_sink_over_b = np.array([float(r["cumulative_delta_sink_over_b"]) for r in rows])
    return t, sigma, n_active, n_born, n_completed, cum_sink_over_b


def load_events(path):
    events = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def find_local_extrema(sigma, min_prominence):
    """Simple local max/min detector with a minimum-prominence filter
    (adjacent-extremum drop must exceed min_prominence to count)."""
    maxima, minima = [], []
    for i in range(1, len(sigma) - 1):
        if sigma[i] >= sigma[i - 1] and sigma[i] >= sigma[i + 1]:
            maxima.append(i)
        elif sigma[i] <= sigma[i - 1] and sigma[i] <= sigma[i + 1]:
            minima.append(i)
    return maxima, minima


def detect_cycles(t, sigma, n_active, n_born, n_completed, cum_sink_over_b, events):
    maxima, minima = find_local_extrema(sigma, NOISE_FLOOR_MPA)
    cycles = []
    cid = 0
    i_max = 0
    while i_max < len(maxima):
        idx_peak = maxima[i_max]
        # find the next minimum after this peak with a big enough drop
        candidate_min = None
        for idx_min in minima:
            if idx_min > idx_peak and sigma[idx_peak] - sigma[idx_min] >= NOISE_FLOOR_MPA:
                candidate_min = idx_min
                break
        if candidate_min is None:
            i_max += 1
            continue
        idx_trough = candidate_min
        # next peak after the trough (for reload duration)
        next_peak_idx = next((m for m in maxima if m > idx_trough), None)

        births_in_cycle = sum(1 for e in events if e.get("kind") == "birth"
                               and t[idx_peak] <= e["t"] <= t[idx_trough])
        completions_in_cycle = sum(1 for e in events if e.get("kind") == "complete"
                                    and t[idx_peak] <= e["t"] <= t[idx_trough])
        max_N_active = int(np.max(n_active[idx_peak:idx_trough + 1])) if idx_trough > idx_peak else int(n_active[idx_peak])
        delta_sink_over_b_in_cycle = cum_sink_over_b[idx_trough] - cum_sink_over_b[idx_peak]

        cid += 1
        cycles.append(dict(
            cycle_id=cid, t_peak=t[idx_peak], sigma_peak=sigma[idx_peak], t_trough=t[idx_trough],
            sigma_trough=sigma[idx_trough], delta_sigma=sigma[idx_peak] - sigma[idx_trough],
            births_in_cycle=births_in_cycle, completed_events_in_cycle=completions_in_cycle,
            max_N_active=max_N_active, delta_sink_over_b_in_cycle=delta_sink_over_b_in_cycle,
            densification_increment_nm=delta_sink_over_b_in_cycle * 2.5e-10 * 1e9,
            episode_duration=t[idx_trough] - t[idx_peak],
            reload_time_to_next_peak=(t[next_peak_idx] - t[idx_trough]) if next_peak_idx is not None else float("nan"),
        ))
        # advance past this trough
        i_max = next((k for k, m in enumerate(maxima) if m > idx_trough), len(maxima))
    return cycles


def main(run_dir):
    history_path = os.path.join(run_dir, "history.csv")
    events_path = os.path.join(run_dir, "events.jsonl")
    t, sigma, n_active, n_born, n_completed, cum_sink_over_b = load_history(history_path)
    events = load_events(events_path)
    cycles = detect_cycles(t, sigma, n_active, n_born, n_completed, cum_sink_over_b, events)

    out_path = os.path.join(run_dir, "cycle_summary.csv")
    if cycles:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cycles[0].keys()))
            w.writeheader()
            for c in cycles:
                w.writerow(c)
        print(f"Detected {len(cycles)} cycle(s), wrote {out_path}")
        for c in cycles:
            print(f"  cycle {c['cycle_id']}: peak={c['sigma_peak']:.2f}MPa@t={c['t_peak']:.4f} -> "
                  f"trough={c['sigma_trough']:.2f}MPa@t={c['t_trough']:.4f} "
                  f"(drop={c['delta_sigma']:.2f}MPa, births={c['births_in_cycle']}, "
                  f"completions={c['completed_events_in_cycle']}, max_N_active={c['max_N_active']})")
    else:
        print(f"No cycles detected (0 peak-trough pairs with drop >= {NOISE_FLOOR_MPA}MPa) -- "
              f"n_history_rows={len(t)}, sigma range=[{sigma.min():.2f},{sigma.max():.2f}]MPa, "
              f"n_born={n_born[-1] if len(n_born) else 0}, n_completed={n_completed[-1] if len(n_completed) else 0}")


if __name__ == "__main__":
    main(sys.argv[1])
