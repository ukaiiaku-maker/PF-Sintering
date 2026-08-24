"""Three-way early-time kinetic decomposition of the fixed Fourier candidate."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numba import set_num_threads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from pf_sintering.axisym import axisym_free_energy_gb, axisym_volume  # noqa: E402
from pf_sintering.axisym_numba_kernel import (  # noqa: E402
    NumbaScratch, axisym_gb_face_projected_step_fast, div_and_update_kernel,
    eta_update_kernel, flux_kernel, mu_kernel,
)
from pf_sintering.experimental_pr_metrology import (  # noqa: E402
    measure_experimental_pr_state,
)
from pf_sintering.sharp_interface_stability import (  # noqa: E402
    energy as sharp_energy, gradient_fd, total_volume,
)
from pr_experimental_long_sinkoff import build_case  # noqa: E402
from pr_tj_node_coupling_gate import make_evaluator  # noqa: E402

OUT = ROOT / "runs" / "pr_current_head_regression" / "fourier_kinetic_decomposition"
SAMPLE_TIMES = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
THREADS = 8


def contour_radius(f, r_c):
    return np.asarray(measure_R_of_z(f, r_c), dtype=float)


def neck_amplitude(radius, z, lam):
    selected = np.isfinite(radius) & (z >= 0.0) & (z <= lam)
    design = np.column_stack([
        np.ones(np.count_nonzero(selected)),
        np.cos(2.0*math.pi*z[selected]/lam),
    ])
    constant, amplitude = np.linalg.lstsq(
        design, radius[selected], rcond=None)[0]
    return float(amplitude), float(constant)


def ownership_fractions(f, e1, e2):
    denom = np.where(f > 1e-12, f, 1.0)
    c1 = np.where(f > 1e-12, e1/denom, 0.5)
    c1 = np.clip(c1, 0.0, 1.0)
    return c1, 1.0-c1


class VariantStepper:
    def __init__(self, kind, state, setup):
        self.kind = kind
        self.state = tuple(np.asarray(value).copy() for value in state)
        self.setup = setup
        self.scratch = [NumbaScratch(*state[0].shape),
                        NumbaScratch(*state[0].shape)]
        self.which = 0
        self.c1, self.c2 = ownership_fractions(*state)
        self.f_fixed = self.state[0].copy()

    def step(self):
        s = self.setup
        old = self.state
        work = self.scratch[self.which]
        self.which = 1-self.which
        if self.kind == "FULL":
            self.state = axisym_gb_face_projected_step_fast(
                *old, s["p"], s["Wc"], s["dr"], s["dz"], s["r_c"],
                s["r_f"], s["dt"], s["M_s"], s["M_eta"], s["W"], work)
        elif self.kind == "SURF":
            mu_kernel(old[0], old[1], old[2], s["p"].W_f, s["p"].k_f,
                      s["Wc"], s["dr"], s["dz"], s["r_c"], s["r_f"],
                      work.mu)
            flux_kernel(old[0], work.mu, s["dr"], s["dz"], s["W"],
                        s["M_s"], 1e-6/s["W"], work.Jr, work.Jz)
            div_and_update_kernel(old[0], work.Jr, work.Jz, s["r_c"],
                                  s["r_f"], s["dr"], s["dz"], s["dt"],
                                  work.f_new)
            np.multiply(self.c1, work.f_new, out=work.e1_new)
            np.multiply(self.c2, work.f_new, out=work.e2_new)
            self.state = work.f_new, work.e1_new, work.e2_new
        elif self.kind == "GB":
            eta_update_kernel(
                old[1], old[2], self.f_fixed, s["Wc"], s["p"].k_eta,
                s["dr"], s["dz"], s["r_c"], s["r_f"], s["dt"],
                s["M_eta"], 1e-4, work.e1_new, work.e2_new)
            np.copyto(work.f_new, self.f_fixed)
            self.state = work.f_new, work.e1_new, work.e2_new
        else:
            raise ValueError(self.kind)


def sharp_diagnostics():
    archive = np.load(
        ROOT / "runs" / "pr_current_head_regression" /
        "gate6_fourier_candidate_sharp_profile_and_modes.npz")
    z = archive["z_over_Rmax"]
    radius = archive["R_over_Rmax"]
    modes = archive["modes"].copy()
    igb = int(archive["gb_index"])
    gate = json.loads((
        ROOT / "runs" / "pr_current_head_regression" /
        "gate6_candidate_hessians.json").read_text())
    candidate = next(item for item in gate["candidates"]
                     if item["candidate"] == "fourier_candidate")
    eigenvalues = np.asarray(
        candidate["resolution_levels"][-1]["lowest_eigenvalues"])
    dz = float(z[1]-z[0])
    free = np.arange(len(radius)-1)
    base = radius.copy()

    def expand(values):
        result = base.copy(); result[free] = values; return result
    def F(values):
        return sharp_energy(expand(values), dz, 1.0,
                            candidate["gamma_gb"], [igb], "capped")
    def V(values):
        return total_volume(expand(values), dz, "capped")
    gf = gradient_fd(F, base[free])
    gv = gradient_fd(V, base[free])
    multiplier = float(np.dot(gf, gv)/np.dot(gv, gv))
    descent = np.zeros_like(base)
    descent[free] = -(gf-multiplier*gv)
    descent /= np.linalg.norm(descent)
    lam_over_rmax = 2.0*math.sqrt(2.0)*math.pi/(math.sqrt(0.92)+0.4)
    template = np.cos(2.0*math.pi*z/lam_over_rmax)
    for j in range(modes.shape[1]):
        if np.dot(modes[:, j], template) < 0.0:
            modes[:, j] *= -1.0
    return z, radius, modes, eigenvalues, descent


def observe(kind, state, t, setup, geom, evaluator, mass0):
    measured, _, _ = measure_experimental_pr_state(state, setup, evaluator)
    radius = contour_radius(state[0], setup["r_c"])
    amplitude, constant = neck_amplitude(radius, setup["z"], setup["lam"])
    energy = axisym_free_energy_gb(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], bc_z="noflux")
    volume = axisym_volume(state[0], setup["r_c"], setup["dr"], setup["dz"])
    return dict(
        variant=kind, t_model=float(t), r_n_m=float(measured["r_n_m"]),
        A_neck_m=amplitude, fitted_C_m=constant, G_J=float(energy),
        sigma_local_Pa=float(measured["sigma_local_Pa"]),
        sigma_integral_Pa=float(measured["sigma_integral_Pa"]),
        volume_relative_error=float(volume/mass0-1.0)), radius


def velocity_projections(radius0, radius1, dt, setup, sharp):
    zsharp, _, modes, eigenvalues, descent = sharp
    rscale = (math.sqrt(0.92)+0.4)*100e-9
    physical_z = zsharp*rscale
    good0 = np.isfinite(radius0); good1 = np.isfinite(radius1)
    v0 = np.interp(physical_z, setup["z"][good0], radius0[good0])
    v1 = np.interp(physical_z, setup["z"][good1], radius1[good1])
    velocity = (v1-v0)/dt
    velocity[-1] = 0.0
    norm = np.linalg.norm(velocity)
    rows = []
    for j, value in enumerate(eigenvalues):
        rows.append(dict(
            mode=j, eigenvalue=float(value), negative=bool(value < 0.0),
            coefficient_m_per_model_time=float(np.dot(velocity, modes[:, j])),
            normalized_overlap=float(np.dot(velocity, modes[:, j])/(norm+1e-300))))
    return rows, float(np.dot(velocity, descent)/(norm+1e-300))


def main():
    set_num_threads(THREADS)
    OUT.mkdir(parents=True, exist_ok=True)
    geom, setup = build_case()
    initial = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    np.savez_compressed(OUT/"identical_initial_state.npz", f=initial[0],
                        e1=initial[1], e2=initial[2], z=setup["z"],
                        r_c=setup["r_c"], dr=setup["dr"], dz=setup["dz"])
    evaluator = make_evaluator(setup, geom)
    mass0 = axisym_volume(initial[0], setup["r_c"], setup["dr"], setup["dz"])
    sharp = sharp_diagnostics()
    rows = []
    projections = {}
    started = time.monotonic()
    sample_steps = [int(round(value/setup["dt"])) for value in SAMPLE_TIMES]
    for kind in ("SURF", "GB", "FULL"):
        stepper = VariantStepper(kind, initial, setup)
        radii = {}
        for step in range(sample_steps[-1]+1):
            if step in sample_steps:
                t = step*setup["dt"]
                row, radius = observe(kind, stepper.state, t, setup, geom,
                                      evaluator, mass0)
                rows.append(row); radii[step] = radius
                print(kind, f"t={t:.4f}", f"rn={row['r_n_m']*1e9:.5f}",
                      f"A={row['A_neck_m']*1e9:.5f}",
                      f"G={row['G_J']:.8e}", flush=True)
            if step < sample_steps[-1]:
                stepper.step()
        for label, end_step in (("very_early", sample_steps[1]),
                                ("early_resolved", sample_steps[4])):
            mode_rows, gradient_overlap = velocity_projections(
                radii[0], radii[end_step], end_step*setup["dt"], setup, sharp)
            projections[f"{kind}_{label}"] = dict(
                interval_model_time=end_step*setup["dt"], modes=mode_rows,
                constrained_first_gradient_descent_normalized_overlap=(
                    gradient_overlap))
    fields = list(rows[0])
    with (OUT/"kinetic_decomposition_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    by_kind = {kind: [row for row in rows if row["variant"] == kind]
               for kind in ("SURF", "GB", "FULL")}
    summary = {}
    for kind, selected in by_kind.items():
        first, last = selected[0], selected[-1]
        summary[kind] = {
            "delta_r_n_nm": (last["r_n_m"]-first["r_n_m"])*1e9,
            "delta_A_neck_nm": (last["A_neck_m"]-first["A_neck_m"])*1e9,
            "delta_G_J": last["G_J"]-first["G_J"],
            "delta_sigma_local_MPa": (
                last["sigma_local_Pa"]-first["sigma_local_Pa"])/1e6,
            "delta_sigma_integral_MPa": (
                last["sigma_integral_Pa"]-first["sigma_integral_Pa"])/1e6,
            "neck_direction": (
                "unchanged" if abs(last["r_n_m"]-first["r_n_m"]) < 1e-15
                else ("narrows" if last["r_n_m"] < first["r_n_m"]
                      else "broadens")),
        }
    result = dict(
        configuration=dict(
            geometry="unchanged corrected Fourier candidate", dt=setup["dt"],
            dr=setup["dr"], dz=setup["dz"], M_s=setup["M_s"],
            M_eta=setup["M_eta"], W=setup["W"], samples=SAMPLE_TIMES),
        summary=summary, projections=projections,
        wall_seconds=time.monotonic()-started,
        no_parameter_tuning=True, no_stochastic_run=True)
    (OUT/"kinetic_decomposition_result.json").write_text(
        json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
