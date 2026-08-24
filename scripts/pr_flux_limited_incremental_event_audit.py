"""Audit a 0.01 b PR event driven by boundary-diffusion mass flux."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(__file__))
from m16a_gb_benchmark import measure_R_of_z
from m16g_pr_derived_particle_asperity import find_gb_trough
from m16h_three_regime_sink_barrier import build_case
from pf_sintering.axisym import axisym_volume
from pf_sintering.axisym_sink_rbm import AxisymSink, HazardParams
from pf_sintering.pr_stress_metrology import branch_patch_3d_stress, pf_contour_estimators
from pf_sintering.rigid_rbm_deposition import (
    apply_fixed_tj_source,
    local_chemical_potential_transport_drive,
    make_fixed_tj_source_support,
    representation_corrected_flux_limited_event_step,
    representation_corrected_union_transfer,
)
from pr_event_metrology_resolution_audit import B_M, SECONDS_PER_MODEL_TIME, advance, four_estimators

SOURCE = "/private/tmp/pr_stress_metrology_sinkoff/states_and_profiles.npz"
OUT_DIR = "/private/tmp/pr_flux_limited_incremental_event_audit"
DOSE_B = 0.01
MAX_INCREMENT_B = 0.0025
PATCH_LENGTHS_NM = (15.0, 20.0, 25.0)


def local_coordinates(state, setup, z_hint):
    f, e1, e2 = state
    R = measure_R_of_z(f, setup["r_c"])
    z_gb, _ = find_gb_trough(R, setup["z"], z_hint, lam=setup["lam"])
    met = pf_contour_estimators(R, setup["z"], z_gb, gamma_s=1.0, psi_reference_deg=160.0)
    rn = met["r_neck"]
    tj_support, _ = make_fixed_tj_source_support(
        f, e1, e2, setup["r_c"], setup["z"], z_gb, rn, setup["W"])
    transport = local_chemical_potential_transport_drive(
        f, e1, e2, setup["p"], setup["Wc"], setup["dr"], setup["dz"],
        setup["r_c"], setup["r_f"], setup["z"], z_gb, rn, setup["W"],
        tj_source_support=tj_support)
    patches = {
        str(length): branch_patch_3d_stress(
            R, setup["z"], z_gb, patch_length_m=length * 1e-9,
            tangent_span_m=5e-9, psi_reference_deg=160.0, gamma_s=1.0)
        for length in PATCH_LENGTHS_NM
    }
    return dict(z_gb=z_gb, r_neck=rn, contact_area=met["contact_area"],
                **transport,
                transport_affinity_MPa=transport["transport_affinity_Pa"] / 1e6,
                branch_patch=patches)


def fixed_event_support(state, setup, coordinates):
    return make_fixed_tj_source_support(
        *state, setup["r_c"], setup["z"], coordinates["z_gb"],
        coordinates["r_neck"], setup["W"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    saved = np.load(SOURCE)
    geom, p, Wc, dr, dz, Ms, Meta, dt, _, _ = build_case()
    setup = dict(p=p, Wc=Wc, dr=dr, dz=dz, r_c=geom["r_c"], r_f=geom["r_f"],
                 z=geom["z"], lam=geom["lam"], M_s=Ms, M_eta=Meta, dt=dt, W=p.W)
    pre = (saved["f_20480"], saved["particle_20480"], saved["neighbor_20480"])
    pre_coord = local_coordinates(pre, setup, geom["z1"])
    z_hint = pre_coord["z_gb"]
    tj_support, tj_particle_fraction = fixed_event_support(pre, setup, pre_coord)
    transport_affinity = pre_coord["transport_affinity_Pa"]
    Dgb = 1e-3 * math.exp(-1.5e5 / (8.314 * 1200.0))
    hp = HazardParams(T=1200.0, D_gb=Dgb, GS=2.0 * geom["R_z1"], b=B_M)
    sink = AxisymSink(active=True)
    f_event, p_event, n_event, completed, kinetic = (
        representation_corrected_flux_limited_event_step(
            *pre, sink, hp, transport_affinity, dt, dz, setup["r_c"],
            setup["z"], z_hint, pre_coord["contact_area"], tj_support,
            tj_particle_fraction, seconds_per_model_time=SECONDS_PER_MODEL_TIME,
            max_increment_fraction_b=MAX_INCREMENT_B,
            event_quota_m=DOSE_B * B_M))
    if not completed:
        print("INCOMPLETE", kinetic, flush=True)
        raise RuntimeError("0.01 b diagnostic did not complete inside one PF interval")

    incremental = (f_event, p_event, n_event)
    controls = advance(pre, setup)
    event_steps = advance(incremental, setup)
    stage_coordinates = dict(pre=pre_coord,
                             deposition=local_coordinates(incremental, setup, z_hint))
    deltas = {}
    for n in (1, 3):
        ec = local_coordinates(event_steps[n], setup, z_hint)
        cc = local_coordinates(controls[n], setup, z_hint)
        stage_coordinates[f"event_{n}PF"] = ec
        stage_coordinates[f"control_{n}PF"] = cc
        e4 = four_estimators(event_steps[n][0], setup, z_hint)
        c4 = four_estimators(controls[n][0], setup, z_hint)
        deltas[str(n)] = dict(
            four_estimators={key: e4[key] - c4[key]
                             for key in ("local", "integral", "mean_width", "normal_support",
                                         "peak_curvature", "curvature_integral")},
            transport_affinity_MPa=(ec["transport_affinity_MPa"]
                                    - cc["transport_affinity_MPa"]),
            branch_patch_MPa={length: (ec["branch_patch"][length]["sigma_3D_local_MPa"]
                                       - cc["branch_patch"][length]["sigma_3D_local_MPa"])
                              for length in ec["branch_patch"]})

    # Direct one-packet body-union comparison using the identical fixed TJ
    # reservoir.  There is no Dgb-prescribed free-surface spreading here.
    mono = representation_corrected_union_transfer(
        *pre, DOSE_B * B_M, dz, setup["r_c"], setup["z"], z_hint)
    monolithic = apply_fixed_tj_source(
        *mono[:3], setup["r_c"], tj_support, tj_particle_fraction,
        mono[3]["V_source_weighted"])
    factor = 2.0 * math.pi * dr * dz
    result = dict(
        dose_b=DOSE_B, max_increment_fraction_b=MAX_INCREMENT_B,
        D_gb_m2_per_s=Dgb, transport_affinity=pre_coord, completed=completed,
        transport_drive_source="current-state local chemical potential; no nucleation stress",
        surface_channel="fixed W-scale TJ arrival, then ordinary M_s PF relaxation",
        kinetic=kinetic, coordinates=stage_coordinates, matched_deltas=deltas,
        mass_relative_error=(axisym_volume(incremental[0], setup["r_c"], dr, dz)
                             - axisym_volume(pre[0], setup["r_c"], dr, dz))
                            / axisym_volume(pre[0], setup["r_c"], dr, dz),
        monolithic_comparison=dict(
            source_volume_m3=mono[3]["V_source_weighted"] * factor,
            max_field_difference=float(np.max(np.abs(monolithic[0] - incremental[0]))),
            note="same fixed TJ source; comparison isolates body-union packetization"))
    with open(os.path.join(OUT_DIR, "flux_limited_0p01b.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
