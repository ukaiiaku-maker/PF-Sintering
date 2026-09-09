#!/usr/bin/env python3
"""Final fresh-seed avalanche trajectory with balanced effective transport.

This is a narrow production configuration of ``pr_avalanche_renewal_five``:
the root/descendant clocks and finite one-b mechanics remain unchanged while
the initial Fourier amplitude, surface-channel timescale, and TJ metrology are
set exactly as declared in the final handoff.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pr_avalanche_renewal_five as renewal  # noqa: E402
from pf_sintering.continuous_field_tj import (  # noqa: E402
    ContinuousFieldTJTracker,
)
from pf_sintering.conservative_bounded_phase import (  # noqa: E402
    ConservativeBoundedPhaseProjector,
)
from pf_sintering.fourier_max_pr_geometry import (  # noqa: E402
    LAMBDA_OVER_RCYL,
    build_fourier_max_pr_geometry,
)
from pf_sintering.model_time_transport import (  # noqa: E402
    mullins_pf_coefficient_m4_per_model_time,
)
from pr_tj_activation_gate import make_setup  # noqa: E402
from pr_tj_node_coupling_gate import (  # noqa: E402
    B_PF_MODEL,
    D_GB_MODEL,
    SURFACE_MOBILITY_MODEL,
    make_evaluator,
)


OUT = ROOT / (
    "runs/pr_current_head_regression/coarsening_driven_fourier/"
    "avalanche_renewal_three_eps0p25_Dgb_over_Ds0p5_root_penalty_continuous_tj")
EPS1_NEW = 0.25
BASELINE_DGB_OVER_DS = 0.3
TARGET_DGB_OVER_DS = 0.5
SURFACE_RATE_SCALE = BASELINE_DGB_OVER_DS / TARGET_DGB_OVER_DS
COARSEN_OVER_SURFACE = 20.0
TAU_SURFACE_OLD_MODEL = 1.0
TAU_SURFACE_NEW_MODEL = TAU_SURFACE_OLD_MODEL / SURFACE_RATE_SCALE
TAU_COARSEN_NEW_MODEL = COARSEN_OVER_SURFACE * TAU_SURFACE_NEW_MODEL
MAX_TJ_JUMP_CELLS = 6.0
EXTRA_RUN_METADATA = {}
LOADING_DT_SCALE = 1.0
LOCAL_METROLOGY_WINDOW_OVER_W = 3.0
AVALANCHES_REQUESTED = 3


def build_balanced_case():
    geom = build_fourier_max_pr_geometry(
        R_cyl=100e-9, W=10e-9, spacing=1.25e-9, eps1=EPS1_NEW,
        tail_margin_W=12.0, radial_margin_W=12.0,
        close_left_substrate=True)
    _, qualified = make_setup(10.0, 1.25)
    setup = {
        **qualified,
        "dr": geom["dr"], "dz": geom["dz"],
        "r_c": geom["r_c"], "r_f": geom["r_f"], "z": geom["z"],
        "lam": geom["lam"],
    }
    setup["M_s_unscaled"] = float(setup["M_s"])
    setup["M_s"] *= SURFACE_RATE_SCALE
    setup["dt_unscaled"] = float(setup["dt"])
    if not (0.0 < LOADING_DT_SCALE <= 1.0):
        raise ValueError("LOADING_DT_SCALE must be in (0, 1]")
    setup["dt"] *= LOADING_DT_SCALE
    setup["surface_rate_scale"] = SURFACE_RATE_SCALE
    setup["loading_dt_scale"] = LOADING_DT_SCALE
    setup["local_metrology_window_m"] = (
        LOCAL_METROLOGY_WINDOW_OVER_W * setup["W"])
    setup["local_metrology_window_over_W"] = LOCAL_METROLOGY_WINDOW_OVER_W
    setup["tau_surface_model"] = TAU_SURFACE_NEW_MODEL
    setup["tau_coarsen_model"] = TAU_COARSEN_NEW_MODEL
    # M_eta and dt intentionally retain their qualified values.  Reducing M_s
    # cannot invalidate the existing explicit stability bound.
    return geom, setup


def build_continuous_evaluator(setup, geom):
    jump_limit = MAX_TJ_JUMP_CELLS * max(setup["dr"], setup["dz"])
    tracker = ContinuousFieldTJTracker(
        z=setup["z"], r_c=setup["r_c"], initial_z_m=geom["z1"],
        max_jump_m=jump_limit)
    return make_evaluator(setup, geom, tj_tracker=tracker)


def main():
    seconds = renewal.SECONDS_PER_MODEL_TIME
    current_ds_model = D_GB_MODEL / BASELINE_DGB_OVER_DS
    target_ds_model = D_GB_MODEL / TARGET_DGB_OVER_DS
    event_mobility = SURFACE_MOBILITY_MODEL * SURFACE_RATE_SCALE
    event_bpf = mullins_pf_coefficient_m4_per_model_time(event_mobility, 1.0)
    if not math.isclose(event_bpf, B_PF_MODEL * SURFACE_RATE_SCALE,
                        rel_tol=1e-14):
        raise AssertionError("surface mobility/B_PF mapping is inconsistent")
    if not math.isclose(D_GB_MODEL / target_ds_model,
                        TARGET_DGB_OVER_DS, rel_tol=0.0, abs_tol=0.0):
        raise AssertionError("target effective diffusivity ratio is not unity")

    renewal.OUT = OUT
    renewal.AVALANCHES_REQUESTED = AVALANCHES_REQUESTED
    renewal.CASE_BUILDER = build_balanced_case
    renewal.EVALUATOR_BUILDER = build_continuous_evaluator
    renewal.RESERVOIR_STEP = renewal.closed_surface_diffusion_only
    renewal.PHASE_PROJECTOR = ConservativeBoundedPhaseProjector()
    renewal.SYSTEM_MASS_BOUNDARY = "closed"
    renewal.EVENT_SURFACE_MOBILITY_MODEL = event_mobility
    renewal.EVENT_B_PF_MODEL = event_bpf
    renewal.RUN_METADATA = dict(
        final_configuration="eps0p25_balanced_effective_transport_continuous_field_TJ",
        epsilon1=EPS1_NEW,
        epsilon2=0.0,
        tail_margin_W=12.0,
        radial_margin_W=12.0,
        left_substrate_cap_closed=True,
        lambda_over_R_cyl=LAMBDA_OVER_RCYL,
        volume_match_R0_over_R_cyl=math.sqrt(1.0 - 0.5 * EPS1_NEW ** 2),
        diffusivity_mapping=(
            "qualified effective model-time ratio: "
            "D_GB/D_s=tau_surface/tau_GB; mobility linear in D_s"),
        physical_atomistic_Ds_available=False,
        D_GB_current_m2_per_model_time=D_GB_MODEL,
        D_s_current_effective_m2_per_model_time=current_ds_model,
        D_GB_over_D_s_current=BASELINE_DGB_OVER_DS,
        D_GB_current_effective_m2_per_s=D_GB_MODEL / seconds,
        D_s_current_effective_m2_per_s=current_ds_model / seconds,
        D_s_new_effective_m2_per_model_time=target_ds_model,
        D_s_new_effective_m2_per_s=target_ds_model / seconds,
        D_GB_over_D_s_new=TARGET_DGB_OVER_DS,
        surface_rate_scale=SURFACE_RATE_SCALE,
        loading_dt_scale=LOADING_DT_SCALE,
        local_metrology_window_over_W=LOCAL_METROLOGY_WINDOW_OVER_W,
        local_metrology_window_m=(
            LOCAL_METROLOGY_WINDOW_OVER_W * 10.0e-9),
        local_metrology_resolution_basis=(
            "branch-local quadratic curvature fit must span at least 3W; "
            "the prior 1.5W/15 nm fit was under-resolved"),
        loading_M_s_old_m6_per_J_model_time=1.0e-33,
        loading_M_s_new_m6_per_J_model_time=1.0e-33 * SURFACE_RATE_SCALE,
        event_M_s_old_m6_per_J_model_time=SURFACE_MOBILITY_MODEL,
        event_M_s_new_m6_per_J_model_time=event_mobility,
        event_B_PF_old_m4_per_model_time=B_PF_MODEL,
        event_B_PF_new_m4_per_model_time=event_bpf,
        tau_surface_old_model=TAU_SURFACE_OLD_MODEL,
        tau_surface_new_model=TAU_SURFACE_NEW_MODEL,
        tau_coarsen_new_model=TAU_COARSEN_NEW_MODEL,
        tau_coarsen_over_tau_surface=COARSEN_OVER_SURFACE,
        M_eta_unchanged=True,
        D_GB_unchanged=True,
        tj_definition="intersection(f=0.5, particle-substrate=0)",
        tj_connectivity="nearest previous physical intersection",
        tj_max_jump_cells=MAX_TJ_JUMP_CELLS,
        tj_failure_policy="stop rather than switch branch",
        same_tj_for_activation_Nsites_and_movie=True,
        previous_strong_PR_case_classification=(
            "preserved_unmodified_unsuccessful_strong_PR_case"),
        **EXTRA_RUN_METADATA)

    print(
        "EFFECTIVE TRANSPORT "
        f"DGB={D_GB_MODEL:.12e} m2/model-time "
        f"Ds_old={current_ds_model:.12e} m2/model-time "
        f"DGB/Ds_old={BASELINE_DGB_OVER_DS:.6f} "
        f"Ds_new={target_ds_model:.12e} m2/model-time "
        f"DGB/Ds_new={TARGET_DGB_OVER_DS:.6f} "
        f"surface_rate_scale={SURFACE_RATE_SCALE:.6f} "
        f"tau_coarsen/tau_surface={COARSEN_OVER_SURFACE:.1f}",
        flush=True)
    renewal.main()


if __name__ == "__main__":
    main()
