"""Slow geometry-neutral particle loss plus fast conservative PF transport."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from numba import set_num_threads

os.environ["PR_RENEWAL_CASE"] = "fourier"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))

from m16a_gb_benchmark import measure_R_of_z  # noqa: E402
from pf_sintering.axisym import axisym_free_energy_gb, axisym_volume  # noqa: E402
from pf_sintering.axisym_numba_kernel import (  # noqa: E402
    NumbaScratch, axisym_gb_face_projected_step_fast,
)
from pf_sintering.experimental_pr_metrology import measure_experimental_pr_state  # noqa: E402
from pf_sintering.hussein_eq4_reference import lambda_c_over_R_eq4  # noqa: E402
from pr_current_head_gate6 import fixed_tip_constrained_modes  # noqa: E402
from pr_experimental_long_sinkoff import build_case  # noqa: E402
from pr_fourier_kinetic_decomposition import neck_amplitude  # noqa: E402
from pr_tj_node_coupling_gate import make_evaluator  # noqa: E402

OUT = ROOT/"runs"/"pr_current_head_regression"/"coarsening_driven_fourier"
TAU_SURF_MODEL = 1.0
RATE_RATIO = 20.0
TAU_COARSEN_MODEL = RATE_RATIO*TAU_SURF_MODEL
SAMPLE_DT = 0.5
FIRST_HORIZON = 10.0
FINAL_HORIZON = 20.0


def integral(field, setup):
    return float(2.0*math.pi*np.sum(
        setup["r_c"][None, :]*field)*setup["dr"]*setup["dz"])


def closed_surface_diffusion_only(state, setup):
    """Production-closed coarsening callback: no explicit solid source.

    Exterior morphology evolves only through the conservative no-flux PF
    surface-diffusion operator that precedes this callback.  Grain ownership
    continues to evolve through the existing eta/GB operator.  The fourth
    return value retains callback compatibility and is identically zero.
    """
    f, particle, substrate = (np.asarray(field) for field in state)
    closure = float(np.max(np.abs(particle+substrate-f)))
    if closure > 5e-15:
        raise RuntimeError(
            f"closed surface-diffusion callback received broken partition: {closure}")
    return f, particle, substrate, 0.0


def reservoir_remove_particle(state, setup):
    """Prescribe only net particle loss; no neck mask or preferred region."""
    f, particle, substrate = state
    vp = integral(particle, setup)
    tau_coarsen_model = float(setup.get(
        "tau_coarsen_model", TAU_COARSEN_MODEL))
    if not math.isfinite(tau_coarsen_model) or tau_coarsen_model <= 0.0:
        raise ValueError("tau_coarsen_model must be positive and finite")
    target = vp*(1.0-math.exp(-setup["dt"]/tau_coarsen_model))
    fb = np.clip(f, 0.0, 1.0)
    surface = 16.0*fb*fb*(1.0-fb)*(1.0-fb)
    ownership = particle/np.maximum(particle+substrate, 1e-30)
    candidate = surface*np.clip(ownership, 0.0, 1.0)
    norm = integral(candidate, setup)
    if norm <= 0.0:
        raise RuntimeError("particle free-surface reservoir support vanished")
    remove = candidate*(target/norm)
    cap = 0.25*np.minimum(np.maximum(particle, 0.0), np.maximum(f, 0.0))
    remove = np.minimum(remove, cap)
    achieved = integral(remove, setup)
    # The timestep-scale request is tiny; capping should never be active.
    if abs(achieved/target-1.0) > 1e-8:
        raise RuntimeError(f"reservoir removal cap active: {achieved/target}")
    return f-remove, particle-remove, substrate, achieved


def reservoir_remove_continuous_surface(state, setup):
    """Apply the external-reservoir recession continuously across the TJ.

    The particle depletion quota remains ``Vp*(1-exp(-dt/tau))``.  A uniform
    normal-recession source is placed on the *connected* free surface and is
    normalized so its particle-owned share equals that quota.  Removal is
    split between grain fields in their existing local ownership ratio, so
    ``particle/f`` and ``substrate/f`` are unchanged algebraically and
    ``particle+substrate=f`` remains exact.  The returned amount is the total
    local solid transferred to the external-reservoir ledger.

    This is the continuous-source correction for the branchwise velocity jump
    diagnosed by ``pr_reservoir_tj_continuity_audit.py``.
    """
    f, particle, substrate = state
    vp = integral(particle, setup)
    tau_coarsen_model = float(setup.get(
        "tau_coarsen_model", TAU_COARSEN_MODEL))
    if not math.isfinite(tau_coarsen_model) or tau_coarsen_model <= 0.0:
        raise ValueError("tau_coarsen_model must be positive and finite")
    target_particle = vp*(1.0-math.exp(-setup["dt"]/tau_coarsen_model))
    fb = np.clip(f, 0.0, 1.0)
    surface = 16.0*fb*fb*(1.0-fb)*(1.0-fb)
    ownership = np.clip(
        particle/np.maximum(particle+substrate, 1e-30), 0.0, 1.0)
    particle_norm = integral(surface*ownership, setup)
    if particle_norm <= 0.0:
        raise RuntimeError("continuous particle-owned surface support vanished")
    remove_f = surface*(target_particle/particle_norm)
    cap = 0.25*np.maximum(f, 0.0)
    if np.any(remove_f > cap):
        raise RuntimeError("continuous-surface reservoir removal cap active")
    remove_particle = ownership*remove_f
    remove_substrate = (1.0-ownership)*remove_f
    achieved_particle = integral(remove_particle, setup)
    if abs(achieved_particle/target_particle-1.0) > 1e-8:
        raise RuntimeError(
            "continuous-surface reservoir particle quota mismatch: "
            f"{achieved_particle/target_particle}")
    total_external = integral(remove_f, setup)
    return (f-remove_f, particle-remove_particle,
            substrate-remove_substrate, total_external)


def reservoir_transfer_particle_to_substrate_closed(state, setup):
    """Closed-domain Ostwald transfer from particle to substrate surface.

    This is the conservative replacement for the historical external-reservoir
    recession.  It preserves the same particle depletion clock, but the exact
    removed particle volume is deposited on the substrate-owned diffuse free
    surface.  Both supports vary continuously with diffuse ownership, phase
    closure is algebraic, and the total axisymmetric solid volume is unchanged.
    """
    f, particle, substrate = (np.asarray(field, dtype=float) for field in state)
    vp = integral(particle, setup)
    tau_coarsen_model = float(setup.get(
        "tau_coarsen_model", TAU_COARSEN_MODEL))
    if not math.isfinite(tau_coarsen_model) or tau_coarsen_model <= 0.0:
        raise ValueError("tau_coarsen_model must be positive and finite")
    target = vp * (1.0 - math.exp(-setup["dt"] / tau_coarsen_model))
    fb = np.clip(f, 0.0, 1.0)
    surface = 16.0 * fb * fb * (1.0 - fb) * (1.0 - fb)
    ownership = np.clip(
        particle / np.maximum(particle + substrate, 1e-30), 0.0, 1.0)
    remove_support = surface * ownership
    deposit_support = surface * (1.0 - ownership)
    remove_norm = integral(remove_support, setup)
    deposit_norm = integral(deposit_support, setup)
    if min(remove_norm, deposit_norm) <= 0.0:
        raise RuntimeError("closed reservoir has no donor or receiver support")
    remove = remove_support * (target / remove_norm)
    deposit = deposit_support * (target / deposit_norm)
    if np.any(remove > 0.25 * np.minimum(np.maximum(particle, 0.0), fb)):
        raise RuntimeError("closed reservoir donor cap active")
    if np.any(deposit > 0.25 * np.minimum(1.0 - fb, 1.0 - substrate)):
        raise RuntimeError("closed reservoir receiver cap active")
    f_new = f - remove + deposit
    particle_new = particle - remove
    substrate_new = substrate + deposit
    removed = integral(remove, setup)
    deposited = integral(deposit, setup)
    total_before = integral(f, setup)
    total_after = integral(f_new, setup)
    relative = (total_after - total_before) / max(abs(total_before), 1e-300)
    closure = float(np.max(np.abs(particle_new + substrate_new - f_new)))
    if abs(removed / target - 1.0) > 1e-10:
        raise RuntimeError("closed reservoir particle quota mismatch")
    if abs(deposited / target - 1.0) > 1e-10 or abs(relative) > 5e-13:
        raise RuntimeError("closed reservoir total-volume closure failure")
    if closure > 5e-15:
        raise RuntimeError("closed reservoir phase-partition closure failure")
    return f_new, particle_new, substrate_new, dict(
        particle_removed_m3=removed, substrate_deposited_m3=deposited,
        total_volume_relative_error=relative, partition_closure=closure,
        external_reservoir_m3=0.0,
        definition="closed particle-to-substrate diffuse free-surface transfer")


def instantaneous_hessian(radius, z, zgb):
    finite = np.isfinite(radius)
    indices = np.where(finite)[0]
    first, last = int(indices[0]), int(indices[-1])
    zr = z[first:last+1]
    rr = np.interp(np.arange(first, last+1), indices,
                   radius[finite])
    # Enforce the physical axis closure explicitly one axial cell past the
    # final diffuse crossing, then hold that DOF fixed in the Hessian.
    zr = np.r_[zr, zr[-1]+(z[1]-z[0])]
    rr = np.r_[rr, 0.0]
    rscale = float(np.max(rr))
    zg = np.linspace(float(zr[0]), float(zr[-1]), 65)/rscale
    rg = np.interp(zg*rscale, zr, rr)/rscale
    dzg = float(zg[1]-zg[0])
    igb = int(np.argmin(np.abs(zg-zgb/rscale)))
    values, _, _, _, residual = fixed_tip_constrained_modes(
        rg, dzg, igb, n_modes=4)
    return float(values[0]), [float(value) for value in values], float(residual)


def eq4_state(radius, z, lam):
    selected = np.isfinite(radius) & (z >= 0.0) & (z <= lam)
    matrix = np.column_stack([
        np.ones(np.count_nonzero(selected)),
        np.cos(2.0*math.pi*z[selected]/lam)])
    constant, amplitude = np.linalg.lstsq(
        matrix, radius[selected], rcond=None)[0]
    rcyl = math.sqrt(constant*constant+0.5*amplitude*amplitude)
    epsilon = abs(amplitude)/rcyl
    lamc = lambda_c_over_R_eq4(epsilon, 0.0, 160.0)
    return float(rcyl), float(epsilon), float((lam/rcyl)/lamc), float(lamc)


def observe(state, t, step, setup, geom, evaluator, vp0, removed):
    measured, _, node = measure_experimental_pr_state(state, setup, evaluator)
    radius = np.asarray(measure_R_of_z(state[0], setup["r_c"]))
    amplitude, constant = neck_amplitude(radius, setup["z"], setup["lam"])
    eigenvalue, eigs, residual = instantaneous_hessian(
        radius, setup["z"], node["z_TJ_m"])
    rcyl, epsilon, ratio, lamc = eq4_state(radius, setup["z"], setup["lam"])
    vp = integral(state[1], setup)
    energy = axisym_free_energy_gb(
        *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], bc_z="noflux")
    return dict(
        t_model=float(t), step=int(step), Vp_m3=vp, Vp_over_Vp0=vp/vp0,
        cumulative_external_reservoir_m3=float(removed),
        r_n_m=float(measured["r_n_m"]), A_neck_m=amplitude,
        fitted_C_m=constant, sigma_local_Pa=float(measured["sigma_local_Pa"]),
        sigma_integral_Pa=float(measured["sigma_integral_Pa"]),
        lambda_min_H=eigenvalue, hessian_eigenvalues=json.dumps(eigs),
        hessian_tangent_residual=residual, Eq4_Rcyl_m=rcyl,
        Eq4_epsilon1=epsilon, Eq4_lambda_c_over_R=lamc,
        Eq4_lambda_over_lambda_c=ratio,
        d_center_m=float(measured["d_center_m"]),
        z_TJ_m=float(measured["z_TJ_m"]), G_J=float(energy))


def save_checkpoint(path, state, row):
    np.savez_compressed(path, f=state[0], particle=state[1],
                        substrate=state[2], **{key: np.asarray(value)
                        for key, value in row.items()
                        if key != "hessian_eigenvalues"})


def main():
    set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUT/"checkpoints"; checkpoints.mkdir(exist_ok=True)
    geom, setup = build_case()
    if geom.get("handoff") != "single_mode_maximum_PR_fourier_fallback":
        raise RuntimeError("case selector did not produce Fourier geometry")
    state = (geom["f"].copy(), geom["e1"].copy(), geom["e2"].copy())
    evaluator = make_evaluator(setup, geom)
    vp0 = integral(state[1], setup)
    scratch = [NumbaScratch(*state[0].shape), NumbaScratch(*state[0].shape)]
    current = 0
    sample_steps = max(1, int(round(SAMPLE_DT/setup["dt"])))
    first_steps = int(round(FIRST_HORIZON/setup["dt"]))
    final_steps = int(round(FINAL_HORIZON/setup["dt"]))
    rows = []
    removed = 0.0
    started = time.monotonic()
    qualified = False
    for step in range(final_steps+1):
        if step % sample_steps == 0 or step in (first_steps, final_steps):
            t = step*setup["dt"]
            row = observe(state, t, step, setup, geom, evaluator, vp0, removed)
            rows.append(row)
            save_checkpoint(checkpoints/f"state_t{t:07.3f}.npz", state, row)
            print("COARSEN", f"t={t:.3f}", f"Vp={row['Vp_over_Vp0']:.5f}",
                  f"rn={row['r_n_m']*1e9:.5f}nm",
                  f"A={row['A_neck_m']*1e9:.5f}nm",
                  f"eig={row['lambda_min_H']:+.5f}",
                  f"Eq4={row['Eq4_lambda_over_lambda_c']:.4f}",
                  f"sigL={row['sigma_local_Pa']/1e6:.3f}MPa",
                  f"sigI={row['sigma_integral_Pa']/1e6:.3f}MPa", flush=True)
            neck_max = max(item["r_n_m"] for item in rows)
            at_max = next(item for item in rows if item["r_n_m"] == neck_max)
            narrowing = row["r_n_m"]/neck_max-1.0 <= -5e-3
            stresses = (row["sigma_local_Pa"] > at_max["sigma_local_Pa"] and
                        row["sigma_integral_Pa"] > at_max["sigma_integral_Pa"])
            qualified = bool(narrowing and stresses)
            if qualified and step >= first_steps:
                break
        if step == final_steps:
            break
        out = 0 if current != 0 else 1
        state = axisym_gb_face_projected_step_fast(
            *state, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
            setup["r_c"], setup["r_f"], setup["dt"], setup["M_s"],
            setup["M_eta"], setup["W"], scratch[out])
        current = out
        new_f, new_particle, new_substrate, loss = reservoir_remove_particle(
            state, setup)
        state = new_f, new_particle, new_substrate
        removed += loss
    fields = list(rows[0])
    with (OUT/"coarsening_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    result = dict(
        outcome=("COARSENING_PR_STRESS_LOADING_QUALIFIED" if qualified
                 else "COARSENING_HORIZON_REACHED_WITHOUT_DUAL_STRESS_LOADING"),
        tau_surface_model=TAU_SURF_MODEL, tau_coarsen_model=TAU_COARSEN_MODEL,
        rate_ratio=RATE_RATIO, neck_unprotected=True,
        geometry_neutral_surface_weight=True, fixed_separation_no_RBM=True,
        no_stochastic_run=True, final=rows[-1],
        wall_seconds=time.monotonic()-started)
    (OUT/"coarsening_result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
