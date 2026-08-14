"""Milestone 16I Section 19: shared plotting configuration, kept separate
from the physics layer (no colors/styling hard-coded into the campaign
driver's simulation loop, only into figure-generation functions that
import from here). Publication-oriented defaults: no chart titles unless
needed for internal diagnostics, no grid, readable labeled axes with
units, larger fonts, consistent regime styling across every figure.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REGIME_COLOR = {"off": "#4C4C9D", "zero": "#2E8B57", "finite": "#B4530A"}
REGIME_LABEL = {"off": "OFF (infinite barrier)", "zero": "ZERO (continuous)", "finite": "FINITE barrier"}
REGIME_ORDER = ("off", "zero", "finite")

EVENT_MARKER_COLOR = "#B4530A"
EVENT_MARKER_STYLE = dict(color=EVENT_MARKER_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)

RC_DEFAULTS = {
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def apply_style():
    plt.rcParams.update(RC_DEFAULTS)


def new_fig(figsize=(7, 5)):
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11, color="0.35")
    ax.grid(False)


def regime_plot_kwargs(regime):
    return dict(color=REGIME_COLOR.get(regime, "0.3"), label=REGIME_LABEL.get(regime, regime), linewidth=1.8)


def mark_events(ax, event_times, y_range=None):
    for t in event_times:
        ax.axvline(t, **EVENT_MARKER_STYLE)
