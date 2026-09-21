#!/usr/bin/env python3
"""Unilateral C2 RIGHT-root force decomposition without event evolution."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

n = types.ModuleType("numba")
n.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else lambda f: f)
n.prange = range
n.get_num_threads = lambda: 1
n.set_num_threads = lambda _: None
sys.modules.setdefault("numba", n)
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
sk = types.ModuleType("skimage")
me = types.ModuleType("skimage.measure")
me.find_contours = lambda *a, **k: None
sk.measure = me
sys.modules.setdefault("skimage", sk)
sys.modules.setdefault("skimage.measure", me)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]

from pf_sintering.bicrystal_configurational_force import (
    dynamic_axial_configurational_balance,
    multigrain_variational_derivatives,
)
from pf_sintering.constrained_densification_relaxation import multigrain_energy
from pf_sintering.three_particle_diagnostics import radius_profile
from pf_sintering.three_particle_sharp_initial import load_mapped_sharp_state
from pf_sintering.work_conjugate_densification import (
    positive_one_contact_direction_components,
    triple_junction_support,
)

B = 0.25e-9
SOURCE = ROOT/"runs/three_particle_c2_source/post_cleanup_state.npz"
RUN = ROOT/"runs/three_particle_c2_campaign/stochastic_seed20260915"
SNAP = RUN/"snapshots/t_0002.772296143_ROOT_CROSSING.npz"
DOC = ROOT/"docs/three_particle/c2_one_sided_force_decomposition"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_support(raw, r_c):
    positive = np.maximum(np.asarray(raw, dtype=float), 0.0)
    norm = float(np.sum(positive*np.asarray(r_c)[None, :]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("reservoir support has zero weight")
    return positive/norm


def reservoir_supports(f, mu_f, z, r_c, gb_z, tj_r, width):
    supports = {
        f"TJ_Gaussian_{multiple}W": triple_junction_support(
            f, z, r_c, z_tj=gb_z, r_tj=tj_r,
            width=multiple*width)
        for multiple in (1, 2, 3)
    }
    bounded = np.minimum(1.0, np.maximum(0.0, f))
    free_surface = 16.0*bounded**2*(1.0-bounded)**2
    nearby = free_surface*np.exp(
        -0.5*((np.asarray(z)[:, None]-gb_z)/(6.0*width))**2)
    supports["distributed_nearby_free_surface"] = normalized_support(nearby, r_c)
    selected = nearby > 1e-8*float(np.max(nearby))
    chemical_weight = np.zeros_like(f)
    chemical_weight[selected] = float(np.max(mu_f[selected]))-mu_f[selected]
    supports["chemical_low_mu_nearby_free_surface"] = normalized_support(
        nearby*chemical_weight, r_c)
    return supports


def curvature_rows(radius, z, contacts, width):
    rows = []
    labels = [
        ("left", 0, -1), ("center_from_left", 0, 1),
        ("center_from_right", 1, -1), ("right", 1, 1)]
    for label, contact, side in labels:
        z0 = float(contacts[contact])
        ids = np.where(np.isfinite(radius) & (side*(z-z0) >= 0))[0]
        zz, rr = z[ids], radius[ids]
        slope = np.gradient(rr, zz, edge_order=2)
        second = np.gradient(slope, zz, edge_order=2)
        total = -second/(1.0+slope*slope)**1.5 + 1.0/(rr*np.sqrt(1.0+slope*slope))
        distance = np.abs(zz-z0)/width
        for lo, hi in ((3, 5), (5, 8), (8, 12)):
            use = ((distance >= lo) & (distance <= hi) & (rr > 2.0*width))
            values = total[use]
            rows.append(dict(
                particle_window=label, contact=contact, side=side,
                lo_widths=lo, hi_widths=hi, cells=int(len(values)),
                mean_total_curvature_per_m=float(np.mean(values)),
                std_total_curvature_per_m=float(np.std(values)),
                relative_std=float(np.std(values)/max(abs(np.mean(values)), 1e-300))))
    return rows


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    g, _, _ = load_mapped_sharp_state(SOURCE)
    with np.load(SNAP) as data:
        f0 = data["f"].copy()
        phi0 = data["ownership"].copy()
        gb = data["gb"].copy()
        root_time = float(data["time_s"])
    radius = radius_profile(f0, g)
    valid = np.isfinite(radius)
    tj_r = float(np.interp(gb[1], g["z"][valid], radius[valid]))
    mu_f, g_phi = multigrain_variational_derivatives(f0, phi0, g)
    supports = reservoir_supports(
        f0, mu_f, g["z"], g["r_c"], gb[1], tj_r, g["config"].width)
    factor = 2.0*math.pi*g["dr"]*g["dz"]
    radial = np.asarray(g["r_c"])[None, :]
    energy0 = multigrain_energy(f0, phi0, g)
    force_rows = []
    steps = (.01, .005, .0025, .00125, .000625, .0003125, .00015625)
    for support_name, support in supports.items():
        for delta_over_b in steps:
            du = delta_over_b*B
            direction = positive_one_contact_direction_components(
                f0, phi0, g["z"], g["r_c"], dr=g["dr"], dz=g["dz"],
                delta_u_m=du, gb_z_m=gb[1], tj_r_m=tj_r,
                width_m=g["config"].width, moving_grain=2,
                neighbor_grain=1, source_support=support)
            integral = lambda value: factor*float(np.sum(radial*value))
            f_swept = -integral(mu_f*direction.f_swept_u)
            f_deposit = -integral(mu_f*direction.f_deposit_u)
            f_force = f_swept+f_deposit
            phi_force = -factor*float(np.sum(
                radial[None, :, :]*g_phi*direction.ownership_u))
            exact_force = f_force+phi_force
            finite_difference = -(
                multigrain_energy(direction.state.f, direction.state.ownership, g)
                - energy0)/du
            volume_rate = direction.swept_volume_rate_m2
            mu_source = integral(mu_f*direction.f_deposit_u)/volume_rate
            mu_swept = integral(-mu_f*direction.f_swept_u)/volume_rate
            force_rows.append(dict(
                support=support_name, delta_u_over_b=delta_over_b,
                delta_u_m=du, finite_difference_force_N=finite_difference,
                exact_directional_force_N=exact_force,
                finite_difference_relative_error=(finite_difference-exact_force)
                    /max(abs(exact_force), 1e-300),
                F_f_N=f_force, F_phi_N=phi_force,
                F_swept_N=f_swept, F_deposit_N=f_deposit,
                swept_volume_rate_m2=volume_rate,
                mu_source_Pa=mu_source, mu_swept_contact_Pa=mu_swept,
                delta_mu_swept_minus_source_Pa=mu_swept-mu_source,
                F_deposit_identity_error_N=f_deposit+volume_rate*mu_source,
                F_f_identity_error_N=f_force-volume_rate*(mu_swept-mu_source),
                mass_relative_error=direction.state.mass_relative_error,
                partition_residual=direction.state.partition_residual))
    write_csv(DOC/"one_sided_force_refinement.csv", force_rows)

    finest = [row for row in force_rows if row["delta_u_over_b"] == steps[-1]]
    write_csv(DOC/"reservoir_sensitivity.csv", finest)
    forces = np.asarray([row["exact_directional_force_N"] for row in finest])
    reservoir_relative_range = float(
        np.ptp(forces)/max(abs(float(np.mean(forces))), 1e-300))

    balance = dynamic_axial_configurational_balance(f0, phi0, g)
    control_rows = []
    for span in (2, 3, 5, 8, 12):
        i1 = int(np.argmin(np.abs(balance["z_m"]-(gb[1]-span*g["config"].width))))
        i2 = int(np.argmin(np.abs(balance["z_m"]-(gb[1]+span*g["config"].width))))
        surface_difference = float(balance["resultant_N"][i2]-balance["resultant_N"][i1])
        bulk = float(np.trapezoid(
            balance["bulk_force_per_length_N_per_m"][i1:i2+1],
            balance["z_m"][i1:i2+1]))
        control_rows.append(dict(
            span_widths=span, z_left_m=balance["z_m"][i1],
            z_right_m=balance["z_m"][i2],
            left_resultant_N=balance["resultant_N"][i1],
            right_resultant_N=balance["resultant_N"][i2],
            surface_difference_N=surface_difference,
            bulk_residual_integral_N=bulk,
            balance_error_N=surface_difference-bulk,
            relative_balance_error=(surface_difference-bulk)
                /max(abs(surface_difference), abs(bulk), 1e-300)))
    write_csv(DOC/"configurational_control_surfaces.csv", control_rows)
    profile_rows = []
    for index, z_value in enumerate(balance["z_m"]):
        profile_rows.append(dict(
            z_m=z_value, distance_from_right_GB_over_W=(z_value-gb[1])/g["config"].width,
            resultant_N=balance["resultant_N"][index],
            bulk_force_per_length_N_per_m=balance["bulk_force_per_length_N_per_m"][index]))
    write_csv(DOC/"configurational_profile.csv", profile_rows)

    curvatures = curvature_rows(radius, g["z"], gb, g["config"].width)
    write_csv(DOC/"curvature_windows.csv", curvatures)
    branch_statistics = []
    for label in sorted({row["particle_window"] for row in curvatures}):
        subset = [row for row in curvatures if row["particle_window"] == label]
        means = np.asarray([row["mean_total_curvature_per_m"] for row in subset])
        branch_statistics.append(dict(
            particle_window=label,
            maximum_within_window_relative_std=max(row["relative_std"] for row in subset),
            window_mean_relative_range=float(np.ptp(means)/max(abs(float(np.mean(means))), 1e-300)),
            adequate_CMC_plateau=bool(
                max(row["relative_std"] for row in subset) <= 0.05
                and np.ptp(means)/max(abs(float(np.mean(means))), 1e-300) <= 0.05)))
    write_csv(DOC/"curvature_plateau_assessment.csv", branch_statistics)
    cmc_adequate = all(row["adequate_CMC_plateau"] for row in branch_statistics)
    maximum_config_balance_error = max(abs(row["relative_balance_error"])
                                       for row in control_rows)
    if reservoir_relative_range > 0.05:
        decision = "C2_EVENT_FORCE_DEPENDS_ON_TRANSPORT_RESERVOIR"
    elif maximum_config_balance_error > 0.2:
        decision = "C2_PF_CONFIGURATIONAL_BALANCE_INCONSISTENT"
    elif not cmc_adequate:
        decision = "C2_DYNAMIC_ROOT_HAS_NO_CMC_SHARP_INTERFACE_FORCE"
    else:
        decision = "C2_ONE_SIDED_FORCE_RESERVOIR_INDEPENDENT"
    baseline = next(row for row in finest if row["support"] == "TJ_Gaussian_1W")
    summary = dict(
        classification=decision,
        retained_artifact="literal signed rigid-frame operator; positive side is the event direction",
        superseded_simplex_tangent_force_reported=False,
        coordinate_definition="U_L=U_C=0; U_R=-u; q_L=0; q_R=u",
        derivative_definition="literal positive one-sided event operator, u->0+",
        negative_contact_opening_used=False,
        simplex_tangent_used=False,
        finite_event_launched=False,
        root_stationarized=False,
        clipping_used=False,
        root_time_s=root_time,
        finest_delta_u_over_b=steps[-1],
        baseline_support="TJ_Gaussian_1W",
        baseline_exact_directional_force_N=baseline["exact_directional_force_N"],
        baseline_finite_difference_force_N=baseline["finite_difference_force_N"],
        baseline_F_f_N=baseline["F_f_N"],
        baseline_F_phi_N=baseline["F_phi_N"],
        baseline_F_swept_N=baseline["F_swept_N"],
        baseline_F_deposit_N=baseline["F_deposit_N"],
        baseline_mu_source_Pa=baseline["mu_source_Pa"],
        baseline_mu_swept_contact_Pa=baseline["mu_swept_contact_Pa"],
        baseline_delta_mu_Pa=baseline["delta_mu_swept_minus_source_Pa"],
        baseline_swept_volume_rate_m2=baseline["swept_volume_rate_m2"],
        reservoir_force_min_N=float(np.min(forces)),
        reservoir_force_max_N=float(np.max(forces)),
        reservoir_force_mean_N=float(np.mean(forces)),
        reservoir_relative_range=reservoir_relative_range,
        maximum_configurational_relative_balance_error=maximum_config_balance_error,
        all_particle_windows_have_CMC_plateau=cmc_adequate,
        CMC_acceptance_rule="within-window relative std <=5% and cross-window mean range <=5%",
        history_sha256=sha256(RUN/"history.json"),
        trajectory_sha256=sha256(RUN/"trajectory.npz"))
    (DOC/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
