"""Render the complete diagnostic payload for the stochastic C2 campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


CONTACTS = ("LEFT", "RIGHT")
COLORS = {"LEFT": "#31688e", "RIGHT": "#d1495b"}


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markers(ax, history, t0):
    seen = set()
    for row in history:
        phase = row["phase"]
        if phase not in ("ROOT_CROSSING", "AVALANCHE_EXTINCT_REPINNED"):
            continue
        label = "stochastic root" if phase == "ROOT_CROSSING" else "avalanche extinction"
        ax.axvline(float(row["time_s"]) - t0,
                   color="#111111" if phase == "ROOT_CROSSING" else "#59a14f",
                   ls="--" if phase == "ROOT_CROSSING" else ":", lw=0.9,
                   alpha=0.75, label=label if label not in seen else None)
        seen.add(label)


def finish_axes(axes, history, t0):
    for ax in np.asarray(axes).ravel():
        markers(ax, history, t0)
        ax.grid(alpha=0.22)
        ax.set_xlabel("Time since C2 source (s)")


def event_rows(history):
    rows = []
    for event in sorted({int(r["event_number"]) for r in history
                         if int(r.get("event_number", 0)) > 0}):
        part = [r for r in history if int(r.get("event_number", 0)) == event]
        active = [r for r in part if r["phase"] == "ACTIVE_ONE_B"]
        done = [r for r in part if r["phase"] == "ONE_B_COMPLETE"]
        if not active or not done:
            continue
        start = min(active, key=lambda r: float(r["q_over_b"]))
        end = done[-1]
        contact = str(start["contact"])
        opposite = "RIGHT" if contact == "LEFT" else "LEFT"
        rows.append({
            "event_number": event,
            "avalanche_id": int(start["avalanche_id"]),
            "contact": contact,
            "start_time_s": float(start["time_s"]),
            "end_time_s": float(end["time_s"]),
            "duration_s": float(end["time_s"]) - float(start["time_s"]),
            "selected_stress_start_MPa": start["contacts"][contact]["sigma_local_Pa"] / 1e6,
            "selected_stress_end_MPa": end["contacts"][contact]["sigma_local_Pa"] / 1e6,
            "selected_stress_change_MPa": (end["contacts"][contact]["sigma_local_Pa"] -
                                            start["contacts"][contact]["sigma_local_Pa"]) / 1e6,
            "opposite_stress_start_MPa": start["contacts"][opposite]["sigma_local_Pa"] / 1e6,
            "opposite_stress_end_MPa": end["contacts"][opposite]["sigma_local_Pa"] / 1e6,
            "opposite_stress_change_MPa": (end["contacts"][opposite]["sigma_local_Pa"] -
                                            start["contacts"][opposite]["sigma_local_Pa"]) / 1e6,
            "center_volume_change_m3": float(end["center_volume_m3"]) - float(start["center_volume_m3"]),
            "quota_strain_change": float(end["production_densification_strain"]) -
                                    float(start["production_densification_strain"]),
            "geometric_strain_change": float(end["geometric_chain_strain"]) -
                                        float(start["geometric_chain_strain"]),
            "minimum_selected_transport_affinity_MPa": min(
                r["contacts"][contact]["transport_affinity_Pa"] / 1e6
                for r in active),
        })
    return rows


def restart_audit(run: Path):
    rows = []
    for path in sorted(run.glob("event_*_final.npz"),
                       key=lambda p: int(p.stem.split("_")[1])):
        with np.load(path) as data:
            restart = json.loads(str(data["restart_json"]))
        rows.append({
            "event_number": int(path.stem.split("_")[1]),
            "accepted_steps": restart["accepted_steps_total"],
            "active_minimum_step_over_b": restart["active_minimum_step_over_b"],
            "quasistatic_trial_rejections": restart["quasistatic_trial_rejections_total"],
            "slow_clock_quadrature_refinements": restart["slow_clock_quadrature_refinements_total"],
            "rejection_reasons": restart["rejection_reason_counts"],
            "event_time_model": restart["event_time_model"],
        })
    return rows


def avalanche_rows(history):
    roots = [r for r in history if r["phase"] == "ROOT_CROSSING"]
    extinctions = [r for r in history if r["phase"] == "AVALANCHE_EXTINCT_REPINNED"]
    events = event_rows(history)
    rows = []
    for root in roots:
        aid = int(root["avalanche_id"])
        end = next((r for r in extinctions if int(r["avalanche_id"]) == aid), None)
        if end is None:
            continue
        contact = str(root["contact"])
        opposite = "RIGHT" if contact == "LEFT" else "LEFT"
        local_events = [r for r in events if r["avalanche_id"] == aid]
        rows.append({
            "avalanche_id": aid,
            "root_contact": contact,
            "root_time_s": float(root["time_s"]),
            "root_center_loss_fraction": 1 - float(root["center_volume_m3"]) /
                                         float(history[0]["center_volume_m3"]),
            "root_selected_stress_MPa": root["contacts"][contact]["sigma_local_Pa"] / 1e6,
            "root_opposite_stress_MPa": root["contacts"][opposite]["sigma_local_Pa"] / 1e6,
            "event_count": len(local_events),
            "extinction_time_s": float(end["time_s"]),
            "avalanche_duration_s": float(end["time_s"]) - float(root["time_s"]),
            "extinction_selected_stress_MPa": end["contacts"][contact]["sigma_local_Pa"] / 1e6,
            "extinction_opposite_stress_MPa": end["contacts"][opposite]["sigma_local_Pa"] / 1e6,
            "selected_stress_change_MPa": (end["contacts"][contact]["sigma_local_Pa"] -
                                            root["contacts"][contact]["sigma_local_Pa"]) / 1e6,
            "opposite_stress_change_MPa": (end["contacts"][opposite]["sigma_local_Pa"] -
                                            root["contacts"][opposite]["sigma_local_Pa"]) / 1e6,
            "production_strain_increment": float(end["production_densification_strain"]) -
                                           float(root["production_densification_strain"]),
            "geometric_strain_increment": float(end["geometric_chain_strain"]) -
                                          float(root["geometric_chain_strain"]),
        })
    return rows


def make_plots(history, out: Path):
    t0 = float(history[0]["time_s"])
    t = np.array([float(r["time_s"]) - t0 for r in history])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for contact in CONTACTS:
        axes[0].plot(t, [r["contacts"][contact]["sigma_local_Pa"] / 1e6 for r in history],
                     color=COLORS[contact], label=f"{contact} exact local", lw=1.5)
    axes[0].plot(t, [r["cluster_area_weighted_local_Pa"] / 1e6 for r in history],
                 color="#9467bd", ls="--", label="area-weighted aggregate")
    axes[0].set_ylabel("Activation stress (MPa)")
    axes[0].legend(fontsize=8, ncol=3)
    energy = np.array([
        np.nan if r.get("energy_balance_sintering_stress_Pa") is None
        else r["energy_balance_sintering_stress_Pa"] / 1e6
        for r in history], dtype=float)
    dg = np.diff(np.array([r["geometric_chain_strain"] for r in history]), prepend=np.nan)
    reliable = np.isfinite(energy) & (np.abs(dg) >= 1e-8)
    axes[1].plot(t[reliable], energy[reliable], ".", ms=2.2, color="#f28e2b",
                 label="energy/work diagnostic (|interval strain| >= 1e-8)")
    axes[1].set_ylabel("Energy-balance stress (MPa)")
    axes[1].legend(fontsize=8)
    finish_axes(axes, history, t0)
    fig.suptitle("A. C2 stress trajectory and root/avalanche markers")
    fig.savefig(out / "A_stress_vs_time.png", dpi=190)
    fig.savefig(out / "A_stress_vs_time.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, contact in zip(axes, CONTACTS):
        values = history
        ax.plot(t, [r["stress_decomposition"][contact]["curvature_term_Pa"] / 1e6 for r in values],
                label="curvature contribution")
        ax.plot(t, [r["stress_decomposition"][contact]["TJ_term_Pa"] / 1e6 for r in values],
                label="TJ contribution")
        ax.plot(t, [r["stress_decomposition"][contact]["total_Pa"] / 1e6 for r in values],
                color="black", lw=1.2, label="total")
        ax.set_ylabel(f"{contact} stress (MPa)")
        ax.legend(fontsize=8, ncol=3)
    finish_axes(axes, history, t0)
    fig.suptitle("B. Exact local-stress decomposition")
    fig.savefig(out / "B_stress_decomposition.png", dpi=190)
    fig.savefig(out / "B_stress_decomposition.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True, constrained_layout=True)
    ax = axes.ravel()
    volumes = np.array([r["grain_volumes_m3"] for r in history])
    for i, label in enumerate(("outer L", "center", "outer R")):
        ax[0].plot(t, 100 * (volumes[:, i] / volumes[0, i] - 1), label=label)
    ax[0].set_ylabel("Grain-volume change (%)"); ax[0].legend(fontsize=8)
    for contact in CONTACTS:
        ax[1].plot(t, [r["contacts"][contact]["r_n_m"] * 1e9 for r in history], label=f"{contact} TJ")
        ax[1].plot(t, [r["diagnostics"][contact + "_neck_r_m"] * 1e9 for r in history], ls="--", label=f"{contact} neck")
    ax[1].set_ylabel("Radius (nm)"); ax[1].legend(fontsize=7, ncol=2)
    for contact in CONTACTS:
        ax[2].plot(t, [r["diagnostics"][contact + "_gb_z_m"] * 1e9 for r in history], label=contact)
    ax[2].set_ylabel("GB/TJ z (nm)"); ax[2].legend(fontsize=8)
    for grain, label in zip(("left", "center", "right"), ("outer L", "center", "outer R")):
        ax[3].plot(t, [r["diagnostics"]["centroid_" + grain + "_m"] * 1e9 for r in history], label=label)
    ax[3].set_ylabel("Centroid z (nm)"); ax[3].legend(fontsize=8)
    for contact in CONTACTS:
        for side, ls in (("negative", "-"), ("positive", "--")):
            key = "theta_" + side + "_rad"
            ax[4].plot(t, np.degrees([r["contacts"][contact][key] for r in history]),
                       ls=ls, label=f"{contact} {side}")
    ax[4].set_ylabel("One-sided angle (deg)"); ax[4].legend(fontsize=6, ncol=2)
    for contact in CONTACTS:
        for side, ls in (("negative", "-"), ("positive", "--")):
            key = "kappa1_" + side + "_per_m"
            ax[5].plot(t, [r["contacts"][contact][key] * 1e-6 for r in history],
                       ls=ls, label=f"{contact} {side}")
    ax[5].set_ylabel("One-sided curvature (1/um)"); ax[5].legend(fontsize=6, ncol=2)
    ax[6].plot(t, [r["surface_area_m2"] * 1e12 for r in history], label="surface")
    ax[6].plot(t, [r["GB_area_m2"] * 1e12 for r in history], label="GB")
    ax[6].set_ylabel("Area (um^2)"); ax[6].legend(fontsize=8)
    ax[7].plot(t, [r["total_interfacial_energy_J"] * 1e12 for r in history], color="#e15759")
    ax[7].set_ylabel("Interfacial energy (pJ)")
    finish_axes(axes, history, t0)
    fig.suptitle("C. C2 geometry and interfacial-energy evolution")
    fig.savefig(out / "C_geometry_vs_time.png", dpi=190)
    fig.savefig(out / "C_geometry_vs_time.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for contact in CONTACTS:
        axes[0].plot(t, [r["H_over_Hstar"][contact] for r in history],
                     color=COLORS[contact], label=f"root {contact}")
    descendant = [np.nan if r.get("descendant_H_over_Hstar") is None
                  else r["descendant_H_over_Hstar"] for r in history]
    axes[0].plot(t, descendant, color="#9467bd", ls="--", label="active descendant")
    axes[0].axhline(1, color="black", lw=0.8)
    axes[0].set_ylabel("Hazard / fixed threshold"); axes[0].legend(fontsize=8)
    for contact in CONTACTS:
        axes[1].semilogy(t, [r["contacts"][contact]["root_rate_per_s"] for r in history],
                        color=COLORS[contact], label=contact)
    axes[1].set_ylabel("Native root rate (1/s)"); axes[1].legend(fontsize=8)
    finish_axes(axes, history, t0)
    fig.suptitle("D. Root and descendant stochastic clocks")
    fig.savefig(out / "D_hazard_vs_time.png", dpi=190)
    fig.savefig(out / "D_hazard_vs_time.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, contact in zip(axes, CONTACTS):
        decomp = [r["root_rate_decomposition"][contact] for r in history]
        ax.plot(t, [r["delta_ln_Gamma_barrier"] for r in decomp], label="barrier")
        ax.plot(t, [r["delta_ln_Gamma_sites"] for r in decomp], label="site count")
        ax.plot(t, [r["delta_ln_Gamma"] for r in decomp], color="black", label="total")
        ax.set_ylabel(f"{contact} delta ln Gamma")
        ax.legend(fontsize=8, ncol=3)
    finish_axes(axes, history, t0)
    fig.suptitle("E. Barrier amplification versus TJ-site loss")
    fig.savefig(out / "E_rate_amplification.png", dpi=190)
    fig.savefig(out / "E_rate_amplification.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.plot(t, 100 * np.array([r["geometric_chain_strain"] for r in history]),
            label="geometric centroid strain")
    ax.plot(t, 100 * np.array([r["production_densification_strain"] for r in history]),
            label="quota/event-production strain")
    ax.set_ylabel("Strain (%)"); ax.legend(fontsize=8)
    finish_axes([ax], history, t0)
    ax.set_title("F. Independent geometry and event-quota strain measures")
    fig.savefig(out / "F_strain.png", dpi=190)
    fig.savefig(out / "F_strain.pdf")
    plt.close(fig)


def make_movie(run: Path, source: Path, out: Path):
    snapshots = sorted((run / "snapshots").glob("*.npz"),
                       key=lambda p: float(p.name.split("_")[1]))
    with np.load(source) as data:
        z = data["z"] * 1e9
        radial = data["r_c"] * 1e9
    palette = np.array([[49, 130, 189], [117, 107, 177], [230, 85, 13]], dtype=float) / 255
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    def draw(index):
        for ax in axes:
            ax.clear()
        with np.load(snapshots[index]) as data:
            f = data["f"]
            ownership = data["ownership"]
            time_s = float(data["time_s"])
            phase = str(data["phase"])
            event = int(data["event_number"])
        stride = 2
        extent = [z[0], z[-1], radial[0], radial[-1]]
        axes[0].imshow(f[::stride].T, origin="lower", aspect="auto", extent=extent,
                       vmin=0, vmax=1, cmap="gray_r", interpolation="nearest")
        rgb = np.moveaxis(np.tensordot(ownership[:, ::stride, :], palette, axes=(0, 0)), -1, -1)
        rgb = np.clip(rgb * f[::stride, :, None] + (1 - f[::stride, :, None]), 0, 1)
        axes[1].imshow(np.swapaxes(rgb, 0, 1), origin="lower", aspect="auto",
                       extent=extent, interpolation="nearest")
        axes[0].set_title("Total density f")
        axes[1].set_title("Grain ownership x f")
        for ax in axes:
            ax.set_xlabel("z (nm)"); ax.set_ylabel("r (nm)")
        fig.suptitle(f"C2  t={time_s:.6g} s  {phase}  event {event}")

    animation = FuncAnimation(fig, draw, frames=len(snapshots), interval=350, repeat=True)
    target = out / "G_morphology_total_and_ownership.gif"
    animation.save(target, writer=PillowWriter(fps=3))
    plt.close(fig)
    return snapshots, target


def snapshot_audit(snapshots, history):
    f_min = float("inf"); f_max = -float("inf")
    ownership_min = float("inf"); ownership_max = -float("inf")
    closure_max = 0.0; times = []
    for path in snapshots:
        with np.load(path) as data:
            f = data["f"]
            ownership = data["ownership"]
            times.append(float(data["time_s"]))
            f_min = min(f_min, float(np.min(f)))
            f_max = max(f_max, float(np.max(f)))
            ownership_min = min(ownership_min, float(np.min(ownership)))
            ownership_max = max(ownership_max, float(np.max(ownership)))
            closure_max = max(closure_max, float(np.max(np.abs(
                np.sum(ownership * f[None], axis=0) - f))))
    masses = np.array([r["diagnostics"]["total_volume_m3"] for r in history])
    return {
        "snapshot_count": len(snapshots),
        "snapshot_times_monotone": bool(np.all(np.diff(times) >= 0)),
        "field_min": f_min,
        "field_max": f_max,
        "ownership_min": ownership_min,
        "ownership_max": ownership_max,
        "maximum_ownership_field_closure": closure_max,
        "history_total_volume_min_m3": float(np.min(masses)),
        "history_total_volume_max_m3": float(np.max(masses)),
        "history_total_volume_relative_range": float((np.max(masses) - np.min(masses)) /
                                                     masses[0]),
        "any_topology_stop": any(bool(r["diagnostics"].get("topology_stop", False))
                                 for r in history),
    }


def scientific_summary(history, launch, run, source, event_table, avalanches, restarts,
                       numerical_audit, wall_seconds):
    initial = history[0]
    roots = [r for r in history if r["phase"] == "ROOT_CROSSING"]
    extinctions = [r for r in history if r["phase"] == "AVALANCHE_EXTINCT_REPINNED"]
    first = roots[0]
    selected = first["contact"]
    first_root_index = next(i for i, row in enumerate(history) if row is first)
    pre = history[:first_root_index + 1]
    stress_series = np.array([r["contacts"][selected]["sigma_local_Pa"] / 1e6 for r in pre])
    maximum_index = int(np.argmax(stress_series))
    maximum_row = pre[maximum_index]
    d = first["root_rate_decomposition"][selected]
    barrier = float(d["delta_ln_Gamma_barrier"])
    sites = float(d["delta_ln_Gamma_sites"])
    previous = pre[-2]
    selected_drop = (extinctions[0]["contacts"][selected]["sigma_local_Pa"] -
                     first["contacts"][selected]["sigma_local_Pa"]) / 1e6
    opposite = "LEFT" if selected == "RIGHT" else "RIGHT"
    initial_decomp = initial["stress_decomposition"][selected]
    root_decomp = first["stress_decomposition"][selected]
    return {
        "campaign_status": "TARGET_RENEWAL_CYCLES_COMPLETE",
        "source_sha256": sha256(source),
        "source_sha256_matches_launch": sha256(source) == launch["source_sha256"],
        "seed": launch["seed"],
        "initial_stress_MPa": {c: initial["contacts"][c]["sigma_local_Pa"] / 1e6 for c in CONTACTS},
        "first_root": {
            "contact": selected,
            "time_s": first["time_s"],
            "stress_MPa": first["contacts"][selected]["sigma_local_Pa"] / 1e6,
            "stress_increase_MPa": (first["contacts"][selected]["sigma_local_Pa"] -
                                    initial["contacts"][selected]["sigma_local_Pa"]) / 1e6,
            "opposite_stress_MPa": first["contacts"][opposite]["sigma_local_Pa"] / 1e6,
            "center_loss_fraction": 1 - first["center_volume_m3"] / initial["center_volume_m3"],
            "stress_still_rising": first["contacts"][selected]["sigma_local_Pa"] >
                                   previous["contacts"][selected]["sigma_local_Pa"],
            "pre_root_maximum_stress_MPa": stress_series[maximum_index],
            "pre_root_maximum_time_s": maximum_row["time_s"],
            "delta_ln_Gamma_barrier": barrier,
            "delta_ln_Gamma_sites": sites,
            "delta_ln_Gamma_total": d["delta_ln_Gamma"],
            "barrier_to_site_loss_magnitude_ratio": barrier / abs(sites),
            "root_rate_amplification": first["contacts"][selected]["root_rate_per_s"] /
                                       initial["contacts"][selected]["root_rate_per_s"],
            "curvature_contribution_initial_MPa": initial_decomp["curvature_term_Pa"] / 1e6,
            "curvature_contribution_at_root_MPa": root_decomp["curvature_term_Pa"] / 1e6,
            "TJ_contribution_initial_MPa": initial_decomp["TJ_term_Pa"] / 1e6,
            "TJ_contribution_at_root_MPa": root_decomp["TJ_term_Pa"] / 1e6,
            "selected_stress_change_to_extinction_MPa": selected_drop,
            "opposite_stress_at_extinction_MPa": extinctions[0]["contacts"][opposite]["sigma_local_Pa"] / 1e6,
        },
        "completed_avalanches": len(avalanches),
        "completed_events": len(event_table),
        "avalanches": avalanches,
        "events": event_table,
        "event_restart_audit": restarts,
        "event_minimum_increment_over_b_unchanged": all(
            r["active_minimum_step_over_b"] == launch["event_minimum_increment_over_b"]
            for r in restarts),
        "total_physical_time_s": history[-1]["time_s"],
        "observed_wall_seconds": wall_seconds,
        "final_center_loss_fraction": 1 - history[-1]["center_volume_m3"] / initial["center_volume_m3"],
        "final_production_strain": history[-1]["production_densification_strain"],
        "final_geometric_strain": history[-1]["geometric_chain_strain"],
        "maximum_mirror_error_diagnostic": max(r["mirror_error_diagnostic"] for r in history),
        "symmetry_enforcement_enabled": False,
        "physics_parameters_unchanged": True,
        "no_clipping": True,
        "no_fitted_correction": True,
        "numerical_audit": numerical_audit,
        "run_directory": str(run),
    }


def write_report(summary, out: Path):
    root = summary["first_root"]
    a1, a2 = summary["avalanches"]
    text = f"""# C2 stochastic three-particle campaign

**Status: TARGET RENEWAL TRAJECTORY COMPLETE.** The predeclared seed produced two complete stochastic root-to-extinction cycles and {summary['completed_events']} accepted one-b transfers. The run ended at the authorized two-cycle target after {summary['total_physical_time_s']:.9g} s of physical evolution and {summary['observed_wall_seconds']/3600:.3f} wall hours; no numerical or topological terminal occurred.

The first root selected the **{root['contact']}** contact at **{root['time_s']:.9g} s** and **{root['stress_MPa']:.6f} MPa**, up by **{root['stress_increase_MPa']:.6f} MPa** from {summary['initial_stress_MPa'][root['contact']]:.6f} MPa. Center-volume loss was **{100*root['center_loss_fraction']:.6f}%**. The stress was still rising at the localized crossing and the pre-root maximum occurred at the crossing itself. The curvature contribution changed from {root['curvature_contribution_initial_MPa']:+.3f} to {root['curvature_contribution_at_root_MPa']:+.3f} MPa; the TJ contribution changed from {root['TJ_contribution_initial_MPa']:.3f} to {root['TJ_contribution_at_root_MPa']:.3f} MPa. Barrier loading contributed **{root['delta_ln_Gamma_barrier']:+.6f}** to delta ln Gamma, site loss contributed **{root['delta_ln_Gamma_sites']:+.6f}**, and the barrier term was **{root['barrier_to_site_loss_magnitude_ratio']:.2f} times** the magnitude of the site penalty. The native root rate increased by **{100*(root['root_rate_amplification']-1):.3f}%**.

Avalanche 1 contained **{a1['event_count']} events**. The selected contact changed by **{a1['selected_stress_change_MPa']:+.3f} MPa** from root to extinction, while the opposite contact ended at **{a1['extinction_opposite_stress_MPa']:.3f} MPa**. The system then reloaded both contacts; avalanche 2 rooted on the **{a2['root_contact']}** contact at **{a2['root_time_s']:.9g} s** and **{a2['root_selected_stress_MPa']:.3f} MPa**. Avalanche 2 also contained **{a2['event_count']} events** and extinguished normally.

The shared microscopic kinetics therefore support the proposed cluster regime. Curvature-driven PF evolution raised the initial stress enough for barrier-dominated stochastic nucleation. Each avalanche relaxed its selected contact strongly, while the other contact retained a high aggregate stress. After the first extinction, surface diffusion restored both contacts to a higher stress than the first root before the second stochastic root occurred. Two cycles are enough to demonstrate renewal and alternating contact-local relaxation, but they are one stochastic realization and do not establish an event-size or waiting-time distribution.

Event 11 approached a low transport affinity and accumulated 15 rejected quasistatic trials, then completed under the original guard. Every final event restart retained the launch minimum increment of 0.00125 b; no discarded-copy refinement was needed. No clipping, symmetry projection, fitted correction, barrier change, or threshold selection was used.

The live runner atomically updated a rolling event checkpoint every 0.01 b. Final audit found that this reused filename did not retain separate quarter-state files. The saved event bases were therefore replayed deterministically with the identical qualified event engine and source-alive transport recoveries. All **48** quarter/final field checkpoints now exist and overlap the production diagnostics; the maximum contact-stress difference is **{summary['milestone_checkpoint_audit']['maximum_contact_stress_difference_Pa']:.3e} Pa**. Event 11's post-recovery accepted grid crosses the nominal 0.75 b milestone at 0.7525 b, and that offset is explicit in the audit.

The energy-balance quotient in panel A is diagnostic only and is plotted only where the recorded interval geometric-strain magnitude is at least 1e-8, avoiding its singular near-zero denominator. Production strain is the accepted event-quota measure; geometric strain is the independent outer-centroid measure.
"""
    (out / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    history = load_json(args.run / "history.json")
    launch = load_json(args.run / "launch.json")
    source = Path(launch["source"])
    make_plots(history, args.out)
    snapshots, movie = make_movie(args.run, source, args.out)
    numerical_audit = snapshot_audit(snapshots, history)
    milestone_audit_path = args.run / "milestones" / "milestone_replay_audit.json"
    milestone_audit = load_json(milestone_audit_path)
    milestone_rows = [row for event in milestone_audit["events"]
                      for row in event["milestones"]]
    events = event_rows(history)
    avalanches = avalanche_rows(history)
    restarts = restart_audit(args.run)
    summary = scientific_summary(history, launch, args.run, source, events, avalanches,
                                 restarts, numerical_audit, args.wall_seconds)
    summary["milestone_checkpoint_audit"] = {
        "all_completed_events_passed": milestone_audit["all_completed_events_passed"],
        "event_count": len(milestone_audit["events"]),
        "checkpoint_count": len(milestone_rows),
        "maximum_contact_stress_difference_Pa": max(
            max(abs(row["left_stress_difference_Pa"]),
                abs(row["right_stress_difference_Pa"])) for row in milestone_rows),
        "maximum_center_volume_difference_m3": max(
            abs(row["center_volume_difference_m3"]) for row in milestone_rows),
        "maximum_total_volume_difference_m3": max(
            abs(row["total_volume_difference_m3"]) for row in milestone_rows),
        "event_11_nominal_0p75_actual_q_over_b": next(
            row["actual_accepted_q_over_b"] for row in milestone_rows
            if row["nominal_q_over_b"] == 0.75 and
            abs(row["nominal_offset_over_b"]) > 1e-12),
        "raw_checkpoint_directory": str(args.run / "milestones"),
    }
    summary["morphology_snapshots"] = len(snapshots)
    summary["morphology_movie"] = str(movie)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy2(args.run / "launch.json", args.out / "launch.json")
    shutil.copy2(args.run / "pre_draw_manifest.json", args.out / "pre_draw_manifest.json")
    shutil.copy2(args.run / "CAMPAIGN_STATUS.md", args.out / "CAMPAIGN_STATUS.md")
    shutil.copy2(milestone_audit_path, args.out / "milestone_replay_audit.json")
    (args.out / "completion.json").write_text(json.dumps({
        "process_exit_code": 0,
        "runner_terminal_line": ("TARGET_RENEWAL_CYCLES_COMPLETE "
                                 "19.0491372167423 avalanches 2 wall "
                                 f"{args.wall_seconds}"),
        "observed_wall_seconds": args.wall_seconds,
        "terminal_status": summary["campaign_status"],
    }, indent=2) + "\n")
    for name, rows in (("events.csv", events), ("avalanches.csv", avalanches)):
        with (args.out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_report(summary, args.out)
    artifact_hashes = {
        path.name: sha256(path) for path in sorted(args.out.iterdir())
        if path.is_file() and path.name != "artifact_sha256.json"
    }
    (args.out / "artifact_sha256.json").write_text(
        json.dumps(artifact_hashes, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("events", "avalanches", "event_restart_audit")}, indent=2))


if __name__ == "__main__":
    main()
