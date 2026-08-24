"""Diagnostic-only resolution audit for the provisional PR 0.01 b sign."""
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
from pf_sintering.axisym import axisym_free_energy_gb
from pf_sintering.axisym_numba_kernel import NumbaScratch, axisym_gb_face_projected_step_fast
from pf_sintering.curvature_compatible_deposition import make_redistribution_fn
from pf_sintering.pr_stress_metrology import local_3d_contact_stress, pf_contour_estimators
from pf_sintering.rigid_rbm_deposition import representation_corrected_union_transfer

SOURCE = "/private/tmp/pr_deterministic_event_gate/pr_event_0p01b_states.npz"
OUT_DIR = "/private/tmp/pr_event_metrology_resolution_audit"
B_M = 0.25e-9
WINDOWS_NM = (7.5, 10.0, 12.5, 15.0, 20.0, 25.0)
METHODS = {"quadratic": 2, "cubic_local_polynomial": 3}
DOSES_B = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)
VIRTUAL_DOSES_B = (0.001, 0.0025, 0.005, 0.01, 0.02)
SECONDS_PER_MODEL_TIME = 1e4


def contour(f, setup, z_hint):
    R = measure_R_of_z(f, setup["r_c"])
    z_gb, _ = find_gb_trough(R, setup["z"], z_hint, lam=setup["lam"])
    return R, z_gb


def endpoint_fit(R, z, z_gb, window_m, degree):
    j = int(np.nanargmin(np.where(np.abs(z - z_gb) <= window_m, R, np.nan)))
    z0, rn = float(z[j]), float(R[j])
    values = []
    for sign in (-1, +1):
        mask = np.isfinite(R) & (sign * (z - z0) >= 0) & (np.abs(z - z0) <= window_m)
        idx = np.where(mask)[0]
        if len(idx) < degree + 2:
            values.append((float("nan"), len(idx))); continue
        c = np.polyfit(z[idx] - z0, R[idx], degree)
        rp = float(np.polyval(np.polyder(c, 1), 0.0))
        rpp = float(np.polyval(np.polyder(c, 2), 0.0))
        values.append((-rpp / (1 + rp * rp) ** 1.5, len(idx)))
    stress = local_3d_contact_stress(values[0][0], values[1][0], rn, math.radians(160), 1.0)
    return dict(r_neck=rn, kappa1=values[0][0], kappa2=values[1][0], n1=values[0][1], n2=values[1][1],
                kappa2_neck=stress["kappa2_neck"], s_line_3D=stress["s_line_3D"],
                sigma_local_3D_MPa=stress["sigma_3D_local_MPa"])


def four_estimators(f, setup, z_hint):
    R, z_gb = contour(f, setup, z_hint)
    m = pf_contour_estimators(R, setup["z"], z_gb, gamma_s=1.0, psi_reference_deg=160.0)
    return dict(z_gb=z_gb, local=m["local_reference"]["sigma_3D_local_MPa"],
                integral=m["sigma_3D_integral_MPa"], mean_width=m["sigma_3D_mean_width_MPa"],
                normal_support=m["sigma_3D_normal_support_MPa"],
                peak_curvature=float(m["curvature_per_m"][np.argmax(np.abs(m["curvature_per_m"]))]),
                curvature_integral=float(np.trapezoid(m["curvature_per_m"], m["curvature_s_m"])))


def advance(state, setup, steps=3):
    f, e1, e2 = [x.copy() for x in state]
    scratch = NumbaScratch(*f.shape); out = {}
    for n in range(1, steps + 1):
        f, e1, e2 = axisym_gb_face_projected_step_fast(
            f, e1, e2, setup["p"], setup["Wc"], setup["dr"], setup["dz"], setup["r_c"],
            setup["r_f"], setup["dt"], setup["M_s"], setup["M_eta"], setup["W"], scratch)
        f, e1, e2 = f.copy(), e1.copy(), e2.copy()
        if n in (1, 3): out[n] = (f.copy(), e1.copy(), e2.copy())
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    x = np.load(SOURCE)
    geom, p, Wc, dr, dz, Ms, Meta, dt, _, _ = build_case()
    setup = dict(p=p, Wc=Wc, dr=dr, dz=dz, r_c=geom["r_c"], r_f=geom["r_f"], z=geom["z"],
                 lam=geom["lam"], M_s=Ms, M_eta=Meta, dt=dt, W=p.W)
    pre = (x["f_pre"], x["particle_pre"], x["neighbor_pre"])
    stages = {
        "pre": pre,
        "union": (x["f_union"], x["particle_union"], x["neighbor_union"]),
        "deposition": (x["f_deposited"], x["particle_deposited"], x["neighbor_deposited"]),
        "event_1PF": (x["f_event_pf1"], x["particle_event_pf1"], x["neighbor_event_pf1"]),
        "event_3PF": (x["f_event_pf3"], x["particle_event_pf3"], x["neighbor_event_pf3"]),
        "control_1PF": (x["f_control_pf1"], x["particle_control_pf1"], x["neighbor_control_pf1"]),
        "control_3PF": (x["f_control_pf3"], x["particle_control_pf3"], x["neighbor_control_pf3"]),
    }
    z_hint = four_estimators(pre[0], setup, geom["z1"])["z_gb"]

    matrix = []
    for pf_n in (1, 3):
        ev, ctrl = stages[f"event_{pf_n}PF"], stages[f"control_{pf_n}PF"]
        Re, ze = contour(ev[0], setup, z_hint); Rc, zc = contour(ctrl[0], setup, z_hint)
        for window_nm in WINDOWS_NM:
            for method, degree in METHODS.items():
                me = endpoint_fit(Re, setup["z"], ze, window_nm * 1e-9, degree)
                mc = endpoint_fit(Rc, setup["z"], zc, window_nm * 1e-9, degree)
                matrix.append(dict(pf_steps=pf_n, window_nm=window_nm, method=method,
                                   delta_kappa1=me["kappa1"] - mc["kappa1"],
                                   delta_kappa2=me["kappa2"] - mc["kappa2"],
                                   delta_kappa2_neck=me["kappa2_neck"] - mc["kappa2_neck"],
                                   delta_s_line_3D=me["s_line_3D"] - mc["s_line_3D"],
                                   delta_sigma_local_3D_MPa=me["sigma_local_3D_MPa"] - mc["sigma_local_3D_MPa"],
                                   event=me, control=mc))

    all_four = {}
    baseline_for = {"union": "pre", "deposition": "pre", "event_1PF": "control_1PF", "event_3PF": "control_3PF"}
    measured = {name: four_estimators(state[0], setup, z_hint) for name, state in stages.items()}
    for name, base in baseline_for.items():
        all_four[name] = dict(event=measured[name], control=measured[base],
                              delta={k: measured[name][k] - measured[base][k]
                                     for k in ("local", "integral", "mean_width", "normal_support",
                                               "peak_curvature", "curvature_integral")})

    Dgb = 1e-3 * math.exp(-1.5e5 / (8.314 * 1200.0))
    redist = make_redistribution_fn(p.W, Ms, D_m2s=Dgb, method="curvature_bump")
    controls = advance(pre, setup)
    dose_scan = []
    for dose in DOSES_B:
        if dose == 0:
            event_states = controls
        else:
            fe, pe, ne, _ = representation_corrected_union_transfer(
                *pre, dose * B_M, dz, setup["r_c"], setup["z"], z_hint,
                redistribution_fn=redist, dt_seconds=dt * SECONDS_PER_MODEL_TIME)
            event_states = advance((fe, pe, ne), setup)
        row = dict(dose_b=dose)
        for n in (1, 3):
            em = four_estimators(event_states[n][0], setup, z_hint)
            cm = four_estimators(controls[n][0], setup, z_hint)
            row[f"delta_local_{n}PF_MPa"] = em["local"] - cm["local"]
            row[f"delta_integral_{n}PF_MPa"] = em["integral"] - cm["integral"]
            row[f"delta_mean_width_{n}PF_MPa"] = em["mean_width"] - cm["mean_width"]
            row[f"delta_normal_support_{n}PF_MPa"] = em["normal_support"] - cm["normal_support"]
        dose_scan.append(row)
    slopes = {}
    xdose = np.array([r["dose_b"] for r in dose_scan])
    for n in (1, 3):
        for estimator in ("local", "integral", "mean_width", "normal_support"):
            y = np.array([r[f"delta_{estimator}_{n}PF_MPa"] for r in dose_scan])
            coef = np.polyfit(xdose, y, 1)
            slopes[f"{estimator}_{n}PF"] = dict(
                slope_MPa_per_b=float(coef[0]), intercept_MPa=float(coef[1]))

    G0 = axisym_free_energy_gb(*pre, p, Wc, dr, dz, setup["r_c"], setup["r_f"], bc_z="noflux")
    event_path = []
    Rpre, zpre = contour(pre[0], setup, z_hint)
    rn = float(np.nanmin(Rpre[np.abs(setup["z"] - zpre) <= 15e-9]))
    area = math.pi * rn * rn
    for dose in VIRTUAL_DOSES_B:
        fv, pv, nv, _ = representation_corrected_union_transfer(
            *pre, dose * B_M, dz, setup["r_c"], setup["z"], z_hint,
            redistribution_fn=redist, dt_seconds=dt * SECONDS_PER_MODEL_TIME)
        Gv = axisym_free_energy_gb(fv, pv, nv, p, Wc, dr, dz, setup["r_c"], setup["r_f"], bc_z="noflux")
        dq = dose * B_M; force = -(Gv - G0) / dq
        event_path.append(dict(dose_b=dose, dq_m=dq, G=Gv, delta_G=Gv-G0,
                               F_event_path_inst_plus_N=force,
                               Sigma_event_path_inst_plus_MPa=force / area / 1e6))

    reaction_fit = np.polyfit(
        [row["dose_b"] for row in event_path[:3]],
        [row["Sigma_event_path_inst_plus_MPa"] for row in event_path[:3]], 1)

    result = dict(window_matrix=matrix, all_four_stages=all_four, small_dose_scan=dose_scan,
                  small_dose_slopes=slopes,
                  one_sided_event_path_energy_derivative=event_path,
                  event_path_zero_extrapolation=dict(
                      fit_doses_b=list(VIRTUAL_DOSES_B[:3]),
                      Sigma_event_path_inst_plus_zero_MPa=float(reaction_fit[1]),
                      slope_MPa_per_b=float(reaction_fit[0]), contact_area_m2=area))
    with open(os.path.join(OUT_DIR, "metrology_resolution_audit.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
