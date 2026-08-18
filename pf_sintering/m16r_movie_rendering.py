"""Milestone 16R: mirrored, equal-aspect, dual-panel movie rendering.

Corrects the M16Q movie's visualization (not physics): the M16Q production
kinetics already used the branch-resolved TJ (`m16q_branch_curvature.
measure_branch_resolved_sigma`) to drive the hazard/RBM, but its frames
(`m16p_barrier_run.save_frame`) plotted only the r>=0 half-plane with an
unconstrained (non-equal) aspect ratio and a single crop -- visually
distorting curvature and showing only half the axisymmetric geometry.

This module draws the SAME physical state using:
  - the FULL mirrored axisymmetric cross-section (r>=0 data reflected to
    r<0 -- the physical solid of revolution has both), never re-deriving
    or independently tracking a second TJ: (+r_tj, z_tj) and (-r_tj, z_tj)
    are exact mirrors of the ONE physical TJ coordinate;
  - `ax.set_aspect("equal", adjustable="box")` on every geometry panel, so
    1nm in r occupies the same screen distance as 1nm in z;
  - FIXED axis limits passed in by the caller (never autoscaled per
    frame);
  - a dedicated neck/TJ zoom panel (also equal-aspect) alongside the full
    cross-section, so the reader can see both overall morphology and the
    local feature driving the stress;
  - a compact annotation strip (not covering morphology) and a
    sigma(t) inset with birth/completion markers.
"""
from __future__ import annotations

import numpy as np


def draw_mirrored_geometry(ax, f, particle, substrate, z, r_c, r_tj, z_tj,
                            r_max_plot_nm, z_range_plot_nm, title=None):
    """Draws the FULL axisymmetric cross-section (mirrored about r=0) into
    a pre-existing axes, with equal aspect and fixed limits. `r_tj`/`z_tj`
    are in meters (the ONE physical TJ coordinate); markers are placed at
    BOTH (+r_tj,z_tj) and (-r_tj,z_tj) -- exact mirrors, never
    independently computed."""
    ax.clear()
    Z_nm = z[:, None] * 1e9 * np.ones((1, len(r_c)))
    R_nm = r_c[None, :] * 1e9 * np.ones((len(z), 1))
    R_neg_nm = -R_nm

    for R_side in (R_nm, R_neg_nm):
        ax.contourf(R_side, Z_nm, particle, levels=[0.5, 1.5], colors=["#4C72B0"], alpha=0.6)
        ax.contourf(R_side, Z_nm, substrate, levels=[0.5, 1.5], colors=["#DD8452"], alpha=0.6)
        ax.contour(R_side, Z_nm, f, levels=[0.5], colors="black", linewidths=0.8)

    if np.isfinite(r_tj) and np.isfinite(z_tj):
        r_tj_nm, z_tj_nm = r_tj * 1e9, z_tj * 1e9
        ax.plot([r_tj_nm, -r_tj_nm], [z_tj_nm, z_tj_nm], marker="x", color="red",
                ms=9, mew=2, linestyle="None")

    ax.set_xlim(-r_max_plot_nm, r_max_plot_nm)
    ax.set_ylim(*z_range_plot_nm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("z (nm)")
    if title:
        ax.set_title(title, fontsize=10)


def draw_neck_zoom(ax, f, particle, substrate, z, r_c, r_tj, z_tj,
                    half_width_r_nm, half_width_z_nm, title="neck / TJ zoom (r>=0 side)"):
    """Equal-aspect zoom centered on the ACTUAL physical TJ location
    (r_tj, z_tj) -- NOT mirrored/centered at r=0. The TJ sits at
    r_tj~O(100nm), far from the axis, so a window centered at r=0 (as an
    earlier version of this function did) shows empty space near the
    axis instead of the neck -- fixed here to center on r_tj directly.
    Only the r>=0 physical side is shown (a mirrored zoom would need a
    window ~2*r_tj wide to include both sides, defeating the purpose of
    a zoom); the full cross-section panel already shows both sides
    mirrored. Fixed half-widths (caller-chosen, constant across the whole
    movie -- never autoscaled)."""
    ax.clear()
    Z_nm = z[:, None] * 1e9 * np.ones((1, len(r_c)))
    R_nm = r_c[None, :] * 1e9 * np.ones((len(z), 1))

    ax.contourf(R_nm, Z_nm, particle, levels=[0.5, 1.5], colors=["#4C72B0"], alpha=0.6)
    ax.contourf(R_nm, Z_nm, substrate, levels=[0.5, 1.5], colors=["#DD8452"], alpha=0.6)
    ax.contour(R_nm, Z_nm, f, levels=[0.5], colors="black", linewidths=1.0)

    z_tj_nm = z_tj * 1e9 if np.isfinite(z_tj) else 0.0
    r_tj_nm = r_tj * 1e9 if np.isfinite(r_tj) else 0.0
    if np.isfinite(r_tj) and np.isfinite(z_tj):
        ax.plot([r_tj_nm], [z_tj_nm], marker="x", color="red", ms=11, mew=2.5, linestyle="None")

    ax.set_xlim(r_tj_nm - half_width_r_nm, r_tj_nm + half_width_r_nm)
    ax.set_ylim(z_tj_nm - half_width_z_nm, z_tj_nm + half_width_z_nm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(title, fontsize=10)


def draw_stress_inset(ax, t_hist, sigma_particle_hist, sigma_neighbor_hist, sigma_avg_hist,
                       current_t, birth_times, completion_times, t_window=None):
    """sigma(t) trajectory up to current_t, all three variants, with a
    current-time marker and birth/completion event markers. `t_window`
    (half-width in t) restricts the x-range to a scrolling recent window
    if given, else shows the full history so far."""
    ax.clear()
    mask = t_hist <= current_t
    if t_window is not None:
        mask &= t_hist >= current_t - t_window
    ax.plot(t_hist[mask], sigma_particle_hist[mask], color="#4C72B0", linewidth=0.9, label="sigma_particle")
    ax.plot(t_hist[mask], sigma_neighbor_hist[mask], color="#DD8452", linewidth=0.9, label="sigma_neighbor")
    ax.plot(t_hist[mask], sigma_avg_hist[mask], color="black", linewidth=1.2, linestyle="--", label="sigma_avg")
    ax.axvline(current_t, color="gray", linewidth=0.8)
    for bt in birth_times:
        if (t_window is None or bt >= current_t - t_window) and bt <= current_t:
            ax.axvline(bt, color="green", linewidth=0.8, alpha=0.6)
    for ct in completion_times:
        if (t_window is None or ct >= current_t - t_window) and ct <= current_t:
            ax.axvline(ct, color="purple", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("sigma (MPa)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("sigma(t) -- green=birth, purple=completion", fontsize=9)


def annotate_frame(fig, t, sigma_avg, sigma_particle, sigma_neighbor, n_active, n_completed,
                    cum_delta_sink_over_b, regime_label):
    """Compact TWO-LINE annotation strip at the top of the figure -- does
    not overlay the morphology panels. Split across two lines (rather
    than one long suptitle) because a single-line string at a legible
    fontsize overflowed the figure width and was clipped at the canvas
    edge -- verified directly against a rendered frame."""
    line1 = f"t={t:.4f}   sigma_avg={sigma_avg:.2f}MPa   sigma_particle={sigma_particle:.2f}MPa   sigma_neighbor={sigma_neighbor:.2f}MPa"
    line2 = f"N_active={n_active}   N_completed={n_completed}   cum_dsink/b={cum_delta_sink_over_b:.3f}   [{regime_label}]"
    fig.suptitle(line1 + "\n" + line2, fontsize=9, y=0.995)
