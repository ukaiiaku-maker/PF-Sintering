"""Long fixed-horizon sink-OFF evolution of the experimental surrogate.

No hazard, rigid-body event, activation-force probe, or stress-based early
stop is present.  The first decision is made only at t=100; an unqualified
but still evolving/ambiguous trajectory continues automatically to t=300.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pf_sintering.axisym_numba_kernel import (
    NumbaScratch, axisym_gb_face_projected_step_fast,
)
from pf_sintering.experimental_particle_geometry import (
    build_phase_field_surrogate, load_initial_experimental_geometry,
)
from pf_sintering.experimental_pr_metrology import measure_experimental_pr_state
from pf_sintering.pr_experimental_geometry import GeometryHistoryWriter
from pr_full_deterministic_cycle import make_transport, source_volume_derivative_m2
from pr_tj_activation_gate import make_setup
from pr_tj_node_coupling_gate import make_evaluator, node_at_state


ROOT = Path("runs/pr_experiment_derived_completion")
CASE_KIND = os.environ.get("PR_RENEWAL_CASE", "experimental").strip().lower()
if CASE_KIND not in {"experimental", "fourier"}:
    raise ValueError("PR_RENEWAL_CASE must be experimental or fourier")
OUT = (ROOT/"experimental_sinkoff" if CASE_KIND == "experimental"
       else ROOT/"fourier_fallback"/"sinkoff")
SOURCE = Path("sinter_results_v83.mat")
INITIAL = (ROOT/"checkpoints"/"experimental_surrogate_initial.npz"
           if CASE_KIND == "experimental"
           else ROOT/"fourier_fallback"/"checkpoints"/"fourier_initial.npz")
PRE_PHASE = f"{CASE_KIND}_pre_event_sinkoff"
ANALYSIS_DT = 5.0
IMAGE_DT = 10.0
CHECKPOINT_DT = 25.0
FIRST_HORIZON = 100.0
FINAL_HORIZON = 300.0
THREADS = 8


GEOMETRY_FIELDS = (
    "d_center_m", "r_n_m", "A_GB_m2", "L_TJ_m", "z_TJ_m", "r_TJ_m",
    "theta_negative_rad", "theta_positive_rad", "theta_average_rad",
    "theta_negative_deg", "theta_positive_deg", "theta_average_deg",
    "kappa1_negative_per_m", "kappa1_positive_per_m",
    "kappa1_average_per_m", "kappa2_negative_per_m",
    "kappa2_positive_per_m", "kappa2_neck_per_m", "local_fit_window_m",
    "local_fit_negative_z_min_m", "local_fit_negative_z_max_m",
    "local_fit_positive_z_min_m", "local_fit_positive_z_max_m",
    "local_fit_negative_points", "local_fit_positive_points",
    "L_f_negative_m", "L_f_positive_m", "L_f_combined_m",
    "delta_phi_f_negative_rad", "delta_phi_f_positive_rad",
    "delta_phi_f_combined_rad", "P_h_m", "w_bar_2D_m", "w_N_m",
    "s_line_m", "gamma_J_per_m2", "V1_m3", "V2_m3", "V_solid_m3",
    "A_free_m2", "A_free_negative_m2", "A_free_positive_m2",
    "z_c1_m", "z_c2_m", "equivalent_radius_1_m", "equivalent_radius_2_m",
    "equivalent_particle_radius_m", "equivalent_volume_radius_m",
    "maximum_radial_extent_m", "G_gamma_J", "G_phasefield_J",
)
EXPERIMENTAL_FIELDS = (
    "P_h_particle_m", "w_bar_2D_particle_m", "w_N_particle_m",
    "particle_axial_length_m", "particle_max_radius_m",
    "particle_z_at_max_radius_m", "sigma_local_Pa", "sigma_integral_Pa",
    "sigma_MW_Pa", "sigma_N_Pa", "sigma_local_negative_Pa",
    "sigma_local_positive_Pa", "sigma_integral_negative_Pa",
    "sigma_integral_positive_Pa", "sigma_MW_negative_Pa",
    "sigma_MW_positive_Pa", "sigma_N_negative_Pa", "sigma_N_positive_Pa",
    "contact_negative_per_m", "contact_positive_per_m",
    "line_3D_negative_per_m", "line_3D_positive_per_m",
    "k_MW_per_m", "k_N_per_m", "mu_neck_geom_Pa",
    "mu_particle_geom_Pa", "delta_mu_PR_geom_Pa", "mu_neck_PF_Pa",
    "mu_particle_PF_Pa", "delta_mu_PR_PF_Pa",
    "kappa_m_neck_support_per_m", "kappa_theta_neck_support_per_m",
    "kappa_m_particle_support_per_m", "kappa_theta_particle_support_per_m",
    "z_particle_support_m", "neck_support_start_W", "neck_support_end_W",
    "particle_support_half_width_W", "contour_smoothing_window_points",
    "neck_support_points", "particle_support_points",
)
FIELDS = (
    "sample_id", "phase", "t_model", "stage_elapsed_model", "cycle_number",
    "sink_state", "q_m", "q_over_b", "q_cumulative_over_b", "pf_step",
    "event_accepted_step", "mass_relative_error", "Fq_path_N",
    "Fq_path_available", "dVsource_dq_m2", "sigma_path_diagnostic_Pa",
    "delta_mu_transport_Pa", "mu_GB_Pa", "mu_TJ_Pa", "N_sites",
    "G_star_J", "Gamma_per_model_time", "H", "H_threshold",
    "H_over_threshold",
) + GEOMETRY_FIELDS + EXPERIMENTAL_FIELDS


def atomic_json(path, value):
    temporary = Path(str(path)+".writing")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = Path(str(path)+".writing")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def build_case():
    if CASE_KIND == "experimental":
        record = load_initial_experimental_geometry(SOURCE)
        geom = build_phase_field_surrogate(
            record, W=10e-9, spacing=1.25e-9)
    else:
        from pf_sintering.fourier_max_pr_geometry import (
            build_fourier_max_pr_geometry,
        )
        geom = build_fourier_max_pr_geometry(
            R_cyl=100e-9, W=10e-9, spacing=1.25e-9)
    _, qualified = make_setup(10.0, 1.25)
    setup = {
        **qualified, "dr": geom["dr"], "dz": geom["dz"],
        "r_c": geom["r_c"], "r_f": geom["r_f"], "z": geom["z"],
        "lam": geom["lam"]}
    return geom, setup


class AlternatingStepper:
    def __init__(self, state, setup):
        self.state = tuple(np.asarray(x).copy() for x in state)
        self.setup = setup
        self.scratch = [NumbaScratch(*state[0].shape),
                        NumbaScratch(*state[0].shape)]
        self.current = None

    def step(self):
        output = 0 if self.current != 0 else 1
        s = self.setup
        self.state = axisym_gb_face_projected_step_fast(
            *self.state, s["p"], s["Wc"], s["dr"], s["dz"], s["r_c"],
            s["r_f"], s["dt"], s["M_s"], s["M_eta"], s["W"],
            self.scratch[output])
        self.current = output


def save_state(path, state, *, t_model, step, status):
    atomic_npz(
        path, current_f=state[0], current_particle=state[1],
        current_neighbor=state[2], t_model=np.array(t_model),
        pf_step=np.array(step), q_over_b=np.array(0.0),
        sink_state=np.array(0), status=np.array(status))


def load_state(path):
    with np.load(path, allow_pickle=False) as saved:
        state = tuple(saved[key].copy() for key in (
            "current_f", "current_particle", "current_neighbor"))
        t = float(saved["t_model"]) if "t_model" in saved else 0.0
        step = int(saved["pf_step"]) if "pf_step" in saved else 0
    return state, t, step


def render(path, branches, row, geom):
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=145)
    for side, color in (("negative", "#b45309"),
                        ("positive", "#1d4ed8")):
        z = np.asarray(branches[side]["z_m"])*1e9
        r = np.asarray(branches[side]["r_m"])*1e9
        ax.plot(z, r, color=color, lw=1.6)
        ax.plot(z, -r, color=color, lw=1.6)
    ax.scatter([row["z_TJ_m"]*1e9]*2,
               [row["r_TJ_m"]*1e9, -row["r_TJ_m"]*1e9],
               color="#dc2626", edgecolor="white", linewidth=0.7,
               s=32, zorder=5)
    ax.set_xlim(geom["z_min_m"]*1e9-5, geom["z_max_m"]*1e9+5)
    radial = float(geom["r_c"][-1]*1e9)
    ax.set_ylim(-radial, radial)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("z (nm)")
    ax.set_ylabel("r (nm)")
    ax.set_title(
        f"{CASE_KIND} surrogate, sink OFF, t={row['t_model']:.1f}\n"
        f"r_n={row['r_n_m']*1e9:.3f} nm, "
        f"sigma_local={row['sigma_local_Pa']/1e6:.2f} MPa, "
        f"Delta mu_PR(PF)={row['delta_mu_PR_PF_Pa']/1e6:.2f} MPa")
    ax.grid(alpha=0.16)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def measurement(state, t_model, step, setup, geom, evaluator, transport,
                mass0, sample_id):
    measured, branches, _ = measure_experimental_pr_state(
        state, setup, evaluator)
    _, transport_node = node_at_state(
        state, evaluator, setup, transport, active=False)
    dVdq = source_volume_derivative_m2(state, setup, geom)
    row = dict(
        sample_id=int(sample_id), phase=PRE_PHASE,
        t_model=float(t_model), stage_elapsed_model=float(t_model),
        cycle_number=0.0, sink_state=0.0, q_m=0.0, q_over_b=0.0,
        q_cumulative_over_b=0.0, pf_step=int(step), event_accepted_step=0.0,
        mass_relative_error=float(measured["V_solid_m3"]/mass0-1.0),
        Fq_path_N=0.0, Fq_path_available=0.0,
        dVsource_dq_m2=float(dVdq), sigma_path_diagnostic_Pa=0.0,
        delta_mu_transport_Pa=float(transport_node["transport_affinity_Pa"]),
        mu_GB_Pa=float(transport_node["mu_GB_Pa"]),
        mu_TJ_Pa=float(transport_node["mu_TJ_Pa"]),
        N_sites=0.0, G_star_J=0.0, Gamma_per_model_time=0.0,
        H=0.0, H_threshold=0.0, H_over_threshold=0.0)
    row.update({name: float(measured[name])
                for name in GEOMETRY_FIELDS+EXPERIMENTAL_FIELDS})
    return row, branches


def relative_change(final, initial):
    return float(final/initial-1.0)


def gate(rows, horizon):
    selected = [row for row in rows
                if row["phase"] == PRE_PHASE
                and row["t_model"] <= horizon+1e-9]
    if len(selected) < 6:
        raise ValueError("insufficient long-run samples for gate")
    first, last = selected[0], selected[-1]
    late_start = max(20.0, horizon-50.0 if horizon <= 100.0 else horizon-100.0)
    late = [row for row in selected if row["t_model"] >= late_start]
    t = np.asarray([row["t_model"] for row in late])
    def slope(name):
        return float(np.polyfit(t, [row[name] for row in late], 1)[0])
    last3 = selected[-3:]
    relaxed = [row for row in selected if row["t_model"] >= 10.0]
    local_min = min(relaxed, key=lambda row: row["sigma_local_Pa"])
    local_rise = last["sigma_local_Pa"]/local_min["sigma_local_Pa"]-1.0
    integral_rise = (
        last["sigma_integral_Pa"]/local_min["sigma_integral_Pa"]-1.0)
    rn_change = relative_change(last["r_n_m"], first["r_n_m"])
    rn_loading_change = relative_change(
        last["r_n_m"], local_min["r_n_m"])
    result = dict(
        horizon_model=float(horizon), sample_count=len(selected),
        initial=first, final=last,
        changes_fraction=dict(
            r_n=rn_change,
            r_n_over_resolved_stress_loading_interval=rn_loading_change,
            particle_max_radius=relative_change(
                last["particle_max_radius_m"], first["particle_max_radius_m"]),
            particle_axial_length=relative_change(
                last["particle_axial_length_m"], first["particle_axial_length_m"]),
            sigma_local_from_initial=relative_change(
                last["sigma_local_Pa"], first["sigma_local_Pa"]),
            sigma_integral_from_initial=relative_change(
                last["sigma_integral_Pa"], first["sigma_integral_Pa"]),
            sigma_local_from_post_initial_minimum=local_rise,
            sigma_integral_from_post_initial_minimum=integral_rise),
        late_slopes_per_model_time=dict(
            r_n_m=slope("r_n_m"),
            sigma_local_Pa=slope("sigma_local_Pa"),
            sigma_integral_Pa=slope("sigma_integral_Pa"),
            delta_mu_PR_geom_Pa=slope("delta_mu_PR_geom_Pa"),
            delta_mu_PR_PF_Pa=slope("delta_mu_PR_PF_Pa")),
        post_initial_local_minimum=dict(
            t_model=local_min["t_model"], value_Pa=local_min["sigma_local_Pa"]),
        integral_baseline_at_local_stress_minimum=dict(
            t_model=local_min["t_model"],
            value_Pa=local_min["sigma_integral_Pa"]))
    signs = dict(
        final_delta_mu_geom_positive=all(
            row["delta_mu_PR_geom_Pa"] > 0.0 for row in last3),
        final_delta_mu_PF_positive=all(
            row["delta_mu_PR_PF_Pa"] > 0.0 for row in last3),
        resolved_neck_narrowing=rn_loading_change <= -5e-3,
        local_stress_resolved_rise=local_rise >= 1e-2,
        integral_stress_resolved_rise=integral_rise >= 5e-3,
        late_neck_slope_negative=slope("r_n_m") < 0.0,
        late_local_slope_positive=slope("sigma_local_Pa") > 0.0,
        late_integral_slope_positive=slope("sigma_integral_Pa") > 0.0)
    result["signs"] = signs
    result["qualified_PR_loading"] = bool(all(signs.values()))
    result["morphology_still_evolving"] = bool(max(
        abs(result["changes_fraction"]["r_n"]),
        abs(result["changes_fraction"]["particle_max_radius"]),
        abs(result["changes_fraction"]["particle_axial_length"])) >= 5e-4)
    return result


def summary_plot(rows):
    selected = [row for row in rows
                if row["phase"] == PRE_PHASE]
    t = np.asarray([row["t_model"] for row in selected])
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 12.0), dpi=155,
                             sharex=True, constrained_layout=True)
    axes[0].plot(t, np.asarray([row["r_n_m"] for row in selected])*1e9,
                 "o-", ms=3, color="#1d4ed8", label="neck radius")
    axes[0].plot(t, np.asarray([row["particle_max_radius_m"]
                               for row in selected])*1e9,
                 "s-", ms=3, color="#64748b", label="particle max radius")
    axes[0].set_ylabel("radius (nm)")
    axes[0].legend()
    for name, label, color in (
            ("sigma_local_Pa", "local", "#1d4ed8"),
            ("sigma_integral_Pa", "integral", "#ea580c"),
            ("sigma_MW_Pa", "mean width", "#16a34a"),
            ("sigma_N_Pa", "normal support", "#7e22ce")):
        axes[1].plot(t, np.asarray([row[name] for row in selected])/1e6,
                     "o-", ms=2.5, label=label, color=color)
    axes[1].set_ylabel("geometric stress (MPa)")
    axes[1].legend(ncol=2)
    axes[2].plot(t, np.asarray([row["delta_mu_PR_geom_Pa"]
                               for row in selected])/1e6,
                 "o-", ms=3, label="smoothed contour", color="#0f766e")
    axes[2].plot(t, np.asarray([row["delta_mu_PR_PF_Pa"]
                               for row in selected])/1e6,
                 "s--", ms=3, label="PF variational mu", color="#be123c")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_ylabel("Delta mu_PR (MPa)")
    axes[2].legend()
    axes[3].plot(t, np.asarray([row["V1_m3"] for row in selected])
                 /selected[0]["V1_m3"]-1.0, label="particle volume")
    axes[3].plot(t, np.asarray([row["A_free_positive_m2"] for row in selected])
                 /selected[0]["A_free_positive_m2"]-1.0,
                 label="particle free area")
    axes[3].plot(t, [row["mass_relative_error"] for row in selected],
                 label="total mass error", ls="--")
    axes[3].set_ylabel("fractional change")
    axes[3].set_xlabel("model time")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("Experimental-derived geometry: long sink-OFF PR diagnostic")
    fig.savefig(OUT/"experimental_sinkoff_diagnostics.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    set_num_threads(THREADS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"images").mkdir(exist_ok=True)
    (OUT/"checkpoints").mkdir(exist_ok=True)
    geom, setup = build_case()
    if CASE_KIND == "fourier":
        atomic_json(OUT.parent/"fourier_geometry_manifest.json", dict(
            construction="single corrected Fourier neck mode plus C3 crest cap",
            equation="R=R0+eps1*cos(2*pi*z/lambda)+eps2*cos(pi*z/lambda)",
            R_cyl_m=geom["R_cyl"], R0_over_R_cyl=math.sqrt(1.0-0.5*0.4**2),
            eps1=geom["EPS1"], eps2=geom["EPS2"],
            lambda_over_R_cyl=geom["lambda_over_R_cyl"], kR=geom["kR"],
            near_maximum_PR_growth=bool(abs(geom["kR"]-1/math.sqrt(2)) < 1e-14),
            unequal_lobe_coordinate_present=False,
            W_m=setup["W"], dr_m=setup["dr"], dz_m=setup["dz"],
            Nz=geom["Nz"], Nr=geom["Nr"],
            total_cells=geom["Nz"]*geom["Nr"],
            r_neck_sharp_m=geom["r_neck_sharp_m"],
            r_crest_sharp_m=geom["r_crest_sharp_m"],
            particle_diffuse_volume_m3=geom["particle_diffuse_volume_m3"],
            boundary_condition_z="noflux",
            unchanged_operator="axisym_gb_face_projected_step_fast"))
    evaluator = make_evaluator(setup, geom)
    transport = make_transport(geom)
    history_csv = OUT/"experimental_sinkoff_history.csv"
    contours_npz = OUT/"experimental_sinkoff_contours.npz"
    latest = OUT/"checkpoints"/"sinkoff_latest.npz"
    if latest.exists():
        state, t_model, step = load_state(latest)
        writer = GeometryHistoryWriter.reopen(
            str(history_csv), str(contours_npz))
        print("resuming experimental sink-OFF", t_model, step, flush=True)
    else:
        if INITIAL.exists():
            state, _, _ = load_state(INITIAL)
        else:
            INITIAL.parent.mkdir(parents=True, exist_ok=True)
            state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
            save_state(INITIAL, state, t_model=0.0, step=0,
                       status=f"{CASE_KIND}_initial")
        t_model, step = 0.0, 0
        writer = GeometryHistoryWriter(
            str(history_csv), str(contours_npz), FIELDS)
    mass0 = float(np.sum(
        state[0]*setup["r_c"][None, :])*2.0*math.pi*setup["dr"]*setup["dz"])
    if writer.rows:
        # Recover the true campaign mass reference, not the restart mass.
        mass0 = writer.rows[0]["V_solid_m3"]
    else:
        row, branches = measurement(
            state, t_model, step, setup, geom, evaluator, transport,
            mass0, len(writer.rows))
        writer.append(row, branches)
        writer.flush()
        save_state(latest, state, t_model=t_model, step=step,
                   status="running")
        render(OUT/"images"/"sinkoff_t000p0.png", branches, row, geom)

    analysis_steps = max(1, int(round(ANALYSIS_DT/setup["dt"])))
    checkpoint_steps = max(1, int(round(CHECKPOINT_DT/setup["dt"])))
    image_steps = max(1, int(round(IMAGE_DT/setup["dt"])))
    first_steps = int(math.ceil(FIRST_HORIZON/setup["dt"]))
    final_steps = int(math.ceil(FINAL_HORIZON/setup["dt"]))
    stepper = AlternatingStepper(state, setup)
    wall_started = time.monotonic()
    decision = None
    try:
        while step < final_steps:
            stepper.step()
            step += 1
            if step % analysis_steps != 0 and step not in (first_steps, final_steps):
                continue
            t_model = step*setup["dt"]
            row, branches = measurement(
                stepper.state, t_model, step, setup, geom, evaluator,
                transport, mass0, len(writer.rows))
            writer.append(row, branches)
            writer.flush()
            save_state(latest, stepper.state, t_model=t_model, step=step,
                       status="running")
            print(
                "EXPERIMENT SINKOFF", f"t={t_model:.3f}",
                f"rn={row['r_n_m']*1e9:.4f}nm",
                f"sigmaL={row['sigma_local_Pa']/1e6:.3f}MPa",
                f"sigmaI={row['sigma_integral_Pa']/1e6:.3f}MPa",
                f"dmuG={row['delta_mu_PR_geom_Pa']/1e6:.3f}MPa",
                f"dmuPF={row['delta_mu_PR_PF_Pa']/1e6:.3f}MPa",
                f"wall={time.monotonic()-wall_started:.1f}s", flush=True)
            if step % image_steps == 0 or step in (first_steps, final_steps):
                tag = f"{t_model:07.1f}".replace(".", "p")
                render(OUT/"images"/f"sinkoff_t{tag}.png",
                       branches, row, geom)
            if step % checkpoint_steps == 0 or step in (first_steps, final_steps):
                tag = f"{t_model:07.1f}".replace(".", "p")
                save_state(
                    OUT/"checkpoints"/f"sinkoff_t{tag}.npz",
                    stepper.state, t_model=t_model, step=step,
                    status="running")
            if step >= first_steps and decision is None:
                decision = gate(writer.rows, FIRST_HORIZON)
                atomic_json(OUT/"gate_t100.json", decision)
                if decision["qualified_PR_loading"]:
                    print(f"{CASE_KIND.upper()} PR LOADING QUALIFIED AT T=100",
                          flush=True)
                    break
                print(
                    "T=100 NOT QUALIFIED; CONTINUING FIXED LONG HORIZON TO T=300",
                    flush=True)
            if step >= final_steps:
                decision = gate(writer.rows, FINAL_HORIZON)
                atomic_json(OUT/"gate_t300.json", decision)
                break
    except BaseException:
        t_model = step*setup["dt"]
        save_state(latest, stepper.state, t_model=t_model, step=step,
                   status="interrupted")
        writer.flush()
        raise

    t_model = step*setup["dt"]
    outcome = (f"{CASE_KIND.upper()}_PR_LOADING_QUALIFIED"
               if decision["qualified_PR_loading"]
               else f"{CASE_KIND.upper()}_GEOMETRY_FAILED_LONG_PR_GATE")
    terminal = OUT/"checkpoints"/"experimental_sinkoff_terminal.npz"
    save_state(terminal, stepper.state, t_model=t_model, step=step,
               status=outcome)
    save_state(latest, stepper.state, t_model=t_model, step=step,
               status=outcome)
    summary_plot(writer.rows)
    result = dict(
        outcome=outcome, decision=decision,
        terminal_restart=str(terminal), terminal_t_model=t_model,
        terminal_step=step, wall_seconds=time.monotonic()-wall_started,
        no_hazard=True, no_RBM_event=True, no_stress_early_stop=True,
        first_negative_decision_time_model=FIRST_HORIZON,
        extended_negative_horizon_model=FINAL_HORIZON)
    atomic_json(OUT/"experimental_sinkoff_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
