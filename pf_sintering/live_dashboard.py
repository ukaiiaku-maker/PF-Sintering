"""Milestone 16I Addendum: shared read-only monitoring/plotting code, used
by (1) the campaign driver's own live-output writer, (2) the Jupyter
notebook dashboard (`notebooks/m16i_live_monitor.ipynb`), and (3) the
standalone CLI monitor (`scripts/m16i_live_monitor.py`) -- a single
implementation so the three consumers never duplicate plotting logic or
drift out of sync with each other.

Every function here is READ-ONLY with respect to the simulation: it only
ever reads `status.json` / `history.csv` / `events.jsonl` / PNG files
already written by the production driver, and writes only its OWN output
artifacts (a regenerated dashboard figure, or nothing at all if called
from a script that just wants the parsed data). Nothing in this module
ever opens a PF checkpoint (.npz) or touches sink/hazard state.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .plotting_style import apply_style, mark_events, style_axes  # noqa: E402


def read_status(run_dir):
    """Reads finite/live/status.json (or the equivalent for other
    regimes) if present. Returns {} if not yet written."""
    path = os.path.join(run_dir, "live", "status.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}  # a partially-written file should never be possible
        # (atomic tmp+rename, see live_status_write below) but tolerate
        # a race gracefully rather than crashing the read-only monitor


def read_history_csv(run_dir, max_rows=None):
    path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", None):
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    if max_rows is not None:
        rows = rows[-max_rows:]
    return rows


def read_events(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def read_event_summary(run_dir, n_recent=5):
    path = os.path.join(run_dir, "event_summary.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-n_recent:]


def atomic_write_json(path, obj):
    """tmp-file + os.replace so a concurrent reader (Jupyter, the CLI
    monitor) never observes a partially-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def build_status_dict(row, sink, regime, last_event_time, checkpoint_step, wall_timestamp):
    return dict(
        wall_timestamp=wall_timestamp, step=row["step"], simulation_time=row["time"],
        barrier_regime=regime, sink_active=bool(sink.active), event_count=sink.n_events,
        hazard=sink.hazard,
        sigma_s_1p5W_MPa=row.get("sigma_s_1p5W_MPa"), sigma_s_2p5W_MPa=row.get("sigma_s_2p5W_MPa"),
        a_contact_nm=row.get("a_contact_nm"), a_over_a0=row.get("a_over_a0"),
        X_neck_nm=row.get("X_neck_nm"), sintering_strain=row.get("sintering_strain"),
        RBM_cumulative_displacement_nm=row.get("RBM_cumulative_displacement_nm"),
        free_surface_area_per_volume=row.get("free_surface_area_per_volume_1_per_nm"),
        mass_drift=row.get("mass_drift"), last_event_time=last_event_time, checkpoint_step=checkpoint_step,
    )


def render_dashboard_figure(run_dir, n_recent_history=4000):
    """The dashboard (Addendum Section A14, extended by the barrier-
    ladder correction's Section 13): sigma_s(t), X_neck(t), hazard(t),
    sintering_strain(t), S/V vs strain, latest morphology, r_neck(t),
    and the sigma_curvature/sigma_width decomposition with Q(t) overlaid
    -- with event markers on every time-series panel. Gracefully shows
    nothing (not an error) for regimes/older runs whose history.csv
    predates the sigma_curvature_1p5W_MPa/sigma_width_1p5W_MPa/Q_1p5W
    columns (Section 14: existing runs are never touched, so their
    history simply lacks these fields -- read_history_csv already
    tolerates missing keys via .get()). Returns the matplotlib Figure
    (caller decides whether to save it, show it inline in Jupyter, or
    both)."""
    apply_style()
    status = read_status(run_dir)
    rows = read_history_csv(run_dir, max_rows=n_recent_history)
    events = read_events(run_dir)
    event_times = [e["t"] for e in events]

    fig, axs = plt.subplots(4, 2, figsize=(12, 13))
    t = [r.get("time") for r in rows]

    axs[0, 0].plot(t, [r.get("sigma_s_1p5W_MPa") for r in rows], color="#B4530A", linewidth=1.3)
    mark_events(axs[0, 0], event_times)
    style_axes(axs[0, 0], None, "sigma_s (MPa)")

    axs[0, 1].plot(t, [r.get("X_neck_nm") for r in rows], color="#2E8B57", linewidth=1.3)
    mark_events(axs[0, 1], event_times)
    style_axes(axs[0, 1], None, "X_neck (nm)")

    axs[1, 0].plot(t, [r.get("hazard") for r in rows], color="#8A2BE2", linewidth=1.3)
    mark_events(axs[1, 0], event_times)
    style_axes(axs[1, 0], None, "hazard (dimensionless)")

    axs[1, 1].plot(t, [r.get("sintering_strain") for r in rows], color="#4C4C9D", linewidth=1.3)
    mark_events(axs[1, 1], event_times)
    style_axes(axs[1, 1], None, "sintering strain")

    strain = [r.get("sintering_strain") for r in rows]
    sv = [r.get("free_surface_area_per_volume_1_per_nm") for r in rows]
    axs[2, 0].plot(strain, sv, color="0.2", linewidth=1.1)
    style_axes(axs[2, 0], "sintering strain", "A_free/V (1/nm)")

    morph_path = os.path.join(run_dir, "live", "latest_morphology.png")
    if os.path.exists(morph_path):
        img = plt.imread(morph_path)
        axs[2, 1].imshow(img)
        axs[2, 1].axis("off")
    else:
        axs[2, 1].text(0.5, 0.5, "no morphology snapshot yet", ha="center", va="center", transform=axs[2, 1].transAxes)
        axs[2, 1].axis("off")

    # Barrier-ladder correction Section 13: r_neck(t) and the
    # sigma_curvature/sigma_width decomposition with Q(t) overlaid --
    # the primary live diagnostic for the high-stress finite case's
    # central question (is r_neck falling faster than X_neck?).
    axs[3, 0].plot(t, [r.get("r_neck_1p5W_nm") for r in rows], color="#B22222", linewidth=1.3)
    mark_events(axs[3, 0], event_times)
    style_axes(axs[3, 0], "time", "r_neck (nm, 1.5W window)")

    ax_q = axs[3, 1]
    ax_q.plot(t, [r.get("sigma_curvature_1p5W_MPa") for r in rows], color="#B4530A", linewidth=1.1, label="sigma_curvature")
    ax_q.plot(t, [r.get("sigma_width_1p5W_MPa") for r in rows], color="#2E8B57", linewidth=1.1, label="sigma_width")
    mark_events(ax_q, event_times)
    style_axes(ax_q, "time", "MPa")
    ax_q.legend(frameon=False, fontsize=8, loc="upper left")
    ax_q2 = ax_q.twinx()
    q_vals = [r.get("Q_1p5W") for r in rows]
    ax_q2.plot(t, q_vals, color="0.3", linewidth=1.0, linestyle=":", label="Q")
    # M16J Section 30 audit: Q = X_neck/(C_GB*r_neck) is positive by construction
    # (X_neck>0, r_neck>0, C_GB=sqrt(1-(gamma_gb/2/gamma_s)^2)>0) -- confirmed against
    # production history data. A shared-subplot twinx() with no explicit floor let the
    # dotted Q line visually sit near the negative sigma_width curve (a DIFFERENT axis,
    # different units) and read as if Q itself were negative. This was a display-clarity
    # issue only; the underlying data was never wrong. Fix: give the twin axis an
    # explicit floor at 0 and mark the sigma_s=0 boundary (Q=1) explicitly.
    q_finite = [v for v in q_vals if v is not None and np.isfinite(v)]
    q_top = max(1.5, 1.1 * max(q_finite)) if q_finite else 1.5
    ax_q2.set_ylim(0, q_top)
    ax_q2.axhline(1.0, color="0.3", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_q2.set_ylabel("Q = X_neck/(C_GB*r_neck)  [dashed line: Q=1, sigma_s=0]")
    ax_q2.grid(False)

    header = (f"{status.get('barrier_regime', '?').upper()}  t={status.get('simulation_time', float('nan')):.2f}  "
              f"step={status.get('step', '?')}  events={status.get('event_count', '?')}  "
              f"sink={'ACTIVE' if status.get('sink_active') else 'inactive'}  "
              f"sigma_s={status.get('sigma_s_1p5W_MPa', float('nan')):.4f} MPa  "
              f"X_neck={status.get('X_neck_nm', float('nan')):.3f} nm  "
              f"strain={status.get('sintering_strain', float('nan')):.4e}  "
              f"mass_drift={status.get('mass_drift', float('nan')):.2e}")
    fig.suptitle(header, fontsize=10.5, color="0.25", y=1.0)
    fig.tight_layout()
    return fig


def recent_events_table(run_dir, n_recent=5):
    return read_event_summary(run_dir, n_recent=n_recent)
