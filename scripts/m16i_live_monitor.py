"""Milestone 16I Addendum Section A17: standalone, read-only CLI monitor.

    python scripts/m16i_live_monitor.py --run runs/<campaign>/finite --refresh 60

Periodically regenerates the same dashboard products the Jupyter notebook
shows (via the shared pf_sintering.live_dashboard module), writing a
refreshed `<run>/live/progress.png` and printing a one-line status
summary plus the most recent events table to the terminal. Never writes
to checkpoints, PF fields, or sink/hazard state -- reads only.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")
from pf_sintering.live_dashboard import read_status, recent_events_table, render_dashboard_figure  # noqa: E402


def print_status_line(run_dir):
    status = read_status(run_dir)
    if not status:
        print(f"[{run_dir}] no live/status.json yet -- has the run started?")
        return
    print(f"[{run_dir}] {status.get('barrier_regime', '?').upper()}  "
          f"t={status.get('simulation_time', float('nan')):.3f}  step={status.get('step', '?')}  "
          f"events={status.get('event_count', '?')}  "
          f"sink={'ACTIVE' if status.get('sink_active') else 'inactive'}  "
          f"hazard={status.get('hazard', float('nan')):.4e}  "
          f"sigma_s={status.get('sigma_s_1p5W_MPa', float('nan')):.4f} MPa  "
          f"X_neck={status.get('X_neck_nm', float('nan')):.3f} nm  "
          f"strain={status.get('sintering_strain', float('nan')):.4e}  "
          f"mass_drift={status.get('mass_drift', float('nan')):.2e}")


def print_events(run_dir, n_recent=5):
    events = recent_events_table(run_dir, n_recent=n_recent)
    if not events:
        return
    print("recent events:")
    header = list(events[0].keys())
    print("  " + "  ".join(f"{h:>16s}" for h in header))
    for row in events:
        print("  " + "  ".join(f"{row[h]:>16s}" for h in header))


def run_once(run_dir):
    print_status_line(run_dir)
    print_events(run_dir)
    try:
        fig = render_dashboard_figure(run_dir)
        import matplotlib.pyplot as plt
        out_path = run_dir.rstrip("/") + "/live/progress.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"refreshed {out_path}")
    except Exception as exc:
        print(f"dashboard render skipped: {exc}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="regime directory, e.g. runs/<campaign>/finite")
    ap.add_argument("--refresh", type=float, default=60.0, help="seconds between refreshes; <=0 runs once and exits")
    args = ap.parse_args()

    if args.refresh <= 0:
        run_once(args.run)
    else:
        print(f"monitoring {args.run} every {args.refresh}s (Ctrl-C to stop; the simulation is unaffected)")
        while True:
            run_once(args.run)
            time.sleep(args.refresh)
