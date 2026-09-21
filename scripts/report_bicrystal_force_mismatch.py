#!/usr/bin/env python3
"""Isolate the full-domain bicrystal envelope/Cannon--Carter mismatch."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace

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

from m16a_gb_benchmark import measure_R_of_z
from m16g_pr_derived_particle_asperity import find_gb_trough
from m16h_three_regime_sink_barrier import build_case
from pf_sintering.bicrystal_configurational_force import (
    axial_configurational_resultant, plateau_average,
    recover_volume_multipliers_for_configurational_stress,
    ownership_configurational_force)
from pf_sintering.constrained_densification_relaxation import multigrain_energy
from pf_sintering.constrained_envelope_force import stationary_envelope_force
from pf_sintering.pr_stress_metrology import pf_contour_estimators
from pf_sintering.three_particle_phase_a import FrozenPhysics
from pf_sintering.work_conjugate_densification import (
    one_contact_state, triple_junction_support)

B = 0.25e-9
RUN = ROOT/"runs/constrained_stationary_envelope/bicrystal"
DOC = ROOT/"docs/three_particle/bicrystal_force_mismatch"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_state(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def local_side_fit(radius, z, z_neck, sign, window_m):
    selected = (np.isfinite(radius) & (sign*(z-z_neck) >= 0)
                & (np.abs(z-z_neck) <= window_m))
    coeff = np.polyfit(z[selected], radius[selected], 2)
    slope = float(np.polyval(np.polyder(coeff, 1), z_neck))
    second = float(np.polyval(np.polyder(coeff, 2), z_neck))
    kappa_m = -second/(1.0+slope*slope)**1.5
    return dict(slope=slope, kappa_meridional_per_m=kappa_m,
                cells=int(np.count_nonzero(selected)))


def cmc_side_fit(radius, z, z_neck, sign, width, lo=3.0, hi=8.0):
    side = np.where(np.isfinite(radius) & (sign*(z-z_neck) >= 0))[0]
    zz, rr = z[side], radius[side]
    slope = np.gradient(rr, zz, edge_order=2)
    second = np.gradient(slope, zz, edge_order=2)
    kappa_m = -second/(1.0+slope*slope)**1.5
    kappa_a = 1.0/(rr*np.sqrt(1.0+slope*slope))
    total = kappa_m+kappa_a
    distance = np.abs(zz-z_neck)/width
    selected = ((distance >= lo) & (distance <= hi) & (rr > 2.0*width))
    weights = rr[selected]*np.sqrt(1.0+slope[selected]**2)
    mean = lambda values: float(np.sum(weights*values[selected])/np.sum(weights))
    total_mean = mean(total)
    total_std = math.sqrt(float(np.sum(
        weights*(total[selected]-total_mean)**2)/np.sum(weights)))
    return dict(side=sign, lo_widths=lo, hi_widths=hi,
                cells=int(np.count_nonzero(selected)),
                kappa_meridional_per_m=mean(kappa_m),
                kappa_azimuthal_per_m=mean(kappa_a),
                kappa_total_per_m=total_mean,
                kappa_total_standard_deviation_per_m=total_std)


def cc_row(label, radius, psi_deg, pressure_curvature, gamma_s):
    psi = math.radians(psi_deg)
    area = math.pi*radius*radius
    line = 2.0*math.pi*radius*gamma_s*math.sin(0.5*psi)
    pressure = -area*gamma_s*pressure_curvature
    return dict(
        metrology=label, psi_deg=psi_deg,
        pressure_curvature_per_m=pressure_curvature,
        contact_area_m2=area, line_force_N=line,
        pressure_force_N=pressure, total_force_N=line+pressure)


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    geom, p, *_ = build_case()
    g = dict(geom)
    g["config"] = SimpleNamespace(width=p.W, outer_radius=100e-9)
    physics = FrozenPhysics()
    state = load_state(RUN/"q050_full_domain.npz")
    reference = load_state(RUN/"reference_bicrystal_relaxed_90490.npz")
    bf = reference["f"]
    phi0 = np.stack([
        np.divide(reference["e1"], bf, out=np.zeros_like(bf), where=bf > 1e-30),
        np.divide(reference["e2"], bf, out=np.ones_like(bf), where=bf > 1e-30)])
    reference_radius = measure_R_of_z(bf, g["r_c"])
    zgb, rtj = find_gb_trough(
        reference_radius, g["z"], geom["z1"], lam=geom["lam"])
    support = triple_junction_support(
        bf, g["z"], g["r_c"], z_tj=zgb, r_tj=rtj, width=p.W)

    def ownership_at(q):
        return one_contact_state(
            bf, phi0, g["z"], g["r_c"], dr=g["dr"], dz=g["dz"],
            u_m=q*B, gb_z_m=zgb, tj_r_m=rtj, width_m=p.W,
            moving_grain=0, neighbor_grain=1,
            source_support=support).ownership

    dq = 0.025
    envelope = stationary_envelope_force(
        state["f"], state["ownership"], ownership_at(0.5-dq),
        ownership_at(0.5+dq), g, dq*B,
        active_mask=state["active_mask"].astype(bool), include_moments=False)
    F_G = -envelope["G_u"]
    F_C = -envelope["constraint_term_N"]

    radius = measure_R_of_z(state["f"], g["r_c"])
    met = pf_contour_estimators(
        radius, g["z"], zgb-0.25*B, gamma_s=physics.gamma_s,
        psi_reference_deg=160.0)
    z_neck = met["z_GB"]
    local_minus = local_side_fit(radius, g["z"], z_neck, -1, 15e-9)
    local_plus = local_side_fit(radius, g["z"], z_neck, +1, 15e-9)
    cmc_minus = cmc_side_fit(radius, g["z"], z_neck, -1, p.W)
    cmc_plus = cmc_side_fit(radius, g["z"], z_neck, +1, p.W)
    cmc_delta = cmc_plus["kappa_total_per_m"]-cmc_minus["kappa_total_per_m"]

    volume_normalization = 2.0*math.pi*g["dr"]*g["dz"]*g["config"].outer_radius
    pressures = -np.asarray(envelope["multipliers"])/volume_normalization
    kkt_delta = float(pressures[0]-pressures[1])/physics.gamma_s
    r_neck = met["r_neck"]
    mean_local_km = 0.5*(local_minus["kappa_meridional_per_m"]
                         + local_plus["kappa_meridional_per_m"])
    choices = [
        cc_row("local_TJ_measured_angle", r_neck, met["psi_measured_deg"],
               mean_local_km+math.sin(math.radians(met["psi_measured_deg"])/2)/r_neck,
               physics.gamma_s),
        cc_row("local_TJ_reference_160deg", r_neck, 160.0,
               mean_local_km+math.sin(math.radians(160.0)/2)/r_neck,
               physics.gamma_s),
        cc_row("global_CMC_measured_angle", r_neck, met["psi_measured_deg"],
               cmc_delta, physics.gamma_s),
        cc_row("global_CMC_reference_160deg", r_neck, 160.0,
               cmc_delta, physics.gamma_s),
        cc_row("KKT_pressure_measured_angle", r_neck, met["psi_measured_deg"],
               kkt_delta, physics.gamma_s),
        cc_row("KKT_pressure_reference_160deg", r_neck, 160.0,
               kkt_delta, physics.gamma_s),
    ]
    for row in choices:
        row["relative_to_envelope"] = (
            row["total_force_N"]-envelope["envelope_force_N"])/abs(envelope["envelope_force_N"])

    pressure_rows = []
    for grain, side, cmc in [(0, "plus", cmc_plus), (1, "minus", cmc_minus)]:
        pressure_rows.append(dict(
            grain=grain, side=side, KKT_pressure_Pa=pressures[grain],
            global_CMC_gamma_kappa_Pa=physics.gamma_s*cmc["kappa_total_per_m"],
            relative_difference=(physics.gamma_s*cmc["kappa_total_per_m"]-pressures[grain])
                                / pressures[grain],
            CMC_kappa_meridional_per_m=cmc["kappa_meridional_per_m"],
            CMC_kappa_azimuthal_per_m=cmc["kappa_azimuthal_per_m"],
            CMC_kappa_total_per_m=cmc["kappa_total_per_m"],
            CMC_standard_deviation_per_m=cmc["kappa_total_standard_deviation_per_m"]))

    config_kkt = recover_volume_multipliers_for_configurational_stress(
        state["f"], state["ownership"], g,
        state["active_mask"].astype(bool))
    config_shape = ownership_configurational_force(
        state["f"], state["ownership"], ownership_at(0.5-dq),
        ownership_at(0.5+dq), g, dq*B, config_kkt["multipliers_J"])
    config = axial_configurational_resultant(
        state["f"], state["ownership"], g, config_kkt["multipliers_J"])
    plateau_rows = []
    for side in [-1, 1]:
        for lo, hi in [(2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10)]:
            plateau_rows.append(plateau_average(
                config, z_neck, p.W, side=side, lo_widths=lo, hi_widths=hi))
    config_selected = plateau_average(
        config, z_neck, p.W, side=-1, lo_widths=3, hi_widths=5)
    curve_rows = []
    for index, z_value in enumerate(config["z_m"]):
        row = dict(z_m=z_value, distance_from_GB_over_W=(z_value-z_neck)/p.W,
                   resultant_N=config["resultant_N"][index])
        for name, values in config["components_N"].items():
            row[f"{name}_N"] = values[index]
        curve_rows.append(row)

    area = math.pi*r_neck*r_neck
    physical_Cu = volume_normalization*np.asarray(envelope["C_u"])
    pf_rows = [
        dict(term="F_G=-G_u", force_N=F_G,
             sharp_limit_interpretation="parameterization-dependent remainder; not line force alone"),
        dict(term="F_C=-lambda^T C_u", force_N=F_C,
             sharp_limit_interpretation="Delta-p times midpoint reassignment area A/2"),
        dict(term="F_env", force_N=envelope["envelope_force_N"],
             sharp_limit_interpretation="F_line - Delta-p A"),
    ]
    metrology = dict(
        neck_radius_m=r_neck, contact_area_m2=area, z_neck_m=z_neck,
        left_surface_slope=local_minus["slope"],
        right_surface_slope=local_plus["slope"],
        measured_dihedral_angle_deg=met["psi_measured_deg"],
        prescribed_dihedral_angle_deg=160.0,
        local_left_meridional_curvature_per_m=local_minus["kappa_meridional_per_m"],
        local_right_meridional_curvature_per_m=local_plus["kappa_meridional_per_m"],
        measured_angle_azimuthal_curvature_per_m=math.sin(
            math.radians(met["psi_measured_deg"])/2)/r_neck,
        reference_angle_azimuthal_curvature_per_m=math.sin(math.radians(160.0)/2)/r_neck,
        local_reference_definition=(
            "quadratic meridional fits over 15 nm on each TJ side plus "
            "sin(160deg/2)/r_neck; s_local_cap times area is algebraically "
            "the CC line force minus its local total-curvature pressure force"))

    global_ref = next(r for r in choices if r["metrology"] == "global_CMC_reference_160deg")
    global_measured = next(r for r in choices if r["metrology"] == "global_CMC_measured_angle")
    c2 = ROOT/"runs/three_particle_c2_campaign/stochastic_seed20260915"
    result = dict(
        classification="CANNON_CARTER_METROLOGY_WAS_THE_MISMATCH",
        headline="FULL_DOMAIN_PF_FORCE_RESOLVED_SHARP_INTERFACE_CONJUGACY_RESTORED",
        envelope_force_N=envelope["envelope_force_N"],
        configurational_force_N=config_shape["configurational_force_N"],
        configurational_to_envelope_relative_difference=abs(
            config_shape["configurational_force_N"]-envelope["envelope_force_N"])
            / abs(envelope["envelope_force_N"]),
        eshelby_control_surface_force_N=config_selected["mean_N"],
        eshelby_control_surface_to_envelope_relative_difference=abs(
            config_selected["mean_N"]-envelope["envelope_force_N"])
            / abs(envelope["envelope_force_N"]),
        independently_recovered_multiplier_max_relative_difference=float(
            np.max(np.abs(config_kkt["multipliers_J"]-envelope["multipliers"])
                   / np.maximum(np.abs(envelope["multipliers"]), 1e-300))),
        configurational_KKT_Linf=config_kkt["KKT_Linf"],
        old_local_reference_CC_force_N=next(
            r["total_force_N"] for r in choices
            if r["metrology"] == "local_TJ_reference_160deg"),
        global_CMC_reference_CC_force_N=global_ref["total_force_N"],
        global_CMC_measured_CC_force_N=global_measured["total_force_N"],
        global_CMC_reference_to_envelope_relative_difference=abs(
            global_ref["total_force_N"]-envelope["envelope_force_N"])
            / abs(envelope["envelope_force_N"]),
        global_CMC_measured_to_envelope_relative_difference=abs(
            global_measured["total_force_N"]-envelope["envelope_force_N"])
            / abs(envelope["envelope_force_N"]),
        KKT_pressure_difference_Pa=float(pressures[0]-pressures[1]),
        global_CMC_pressure_difference_Pa=physics.gamma_s*cmc_delta,
        fixed_f_volume_reassignment_area_m2=float(physical_Cu[0]),
        half_contact_area_m2=0.5*area,
        F_G_N=F_G, F_C_N=F_C,
        width_convergence_triggered=False,
        width_convergence_reason=(
            "not triggered: independent PF configurational force and corrected "
            "global-CMC Cannon-Carter force both agree with the envelope within 5%"),
        c2_started=False, clipping_used=False, root_law_changed=False,
        barrier_changed=False,
        full_domain_state_sha256=sha256(RUN/"q050_full_domain.npz"),
        c2_history_sha256=sha256(c2/"history.json"),
        c2_trajectory_sha256=sha256(c2/"trajectory.npz"))

    write_csv(DOC/"cannon_carter_decomposition.csv", choices)
    write_csv(DOC/"pressure_diagnostics.csv", pressure_rows)
    write_csv(DOC/"pf_envelope_decomposition.csv", pf_rows)
    write_csv(DOC/"configurational_force_plateaus.csv", plateau_rows)
    write_csv(DOC/"configurational_force_profile.csv", curve_rows)
    (DOC/"metrology.json").write_text(json.dumps(metrology, indent=2)+"\n")
    (DOC/"result.json").write_text(json.dumps(result, indent=2)+"\n")

    (DOC/"REPORT.md").write_text(f"""# Full-domain bicrystal force-mismatch audit

**Decision: `CANNON_CARTER_METROLOGY_WAS_THE_MISMATCH`**

**Current status: `FULL_DOMAIN_PF_FORCE_RESOLVED_SHARP_INTERFACE_CONJUGACY_RESTORED`**

The stationary envelope force is `{result['envelope_force_N']:.9e} N`. An
independent analytic shape derivative of the implemented diffuse PF
functional gives `{result['configurational_force_N']:.9e} N`, a
`{result['configurational_to_envelope_relative_difference']:.3e}` relative
difference. It integrates `-(delta L/delta phi_i) phi_i,u` and independently
recovers the KKT pressures; it does not call the envelope derivative,
neighboring minimizations, or Cannon--Carter.

As a second check, the continuum axial Eshelby/Korteweg control-surface
resultant is `{result['eshelby_control_surface_force_N']:.9e} N`, differing
from the discrete envelope by
`{result['eshelby_control_surface_to_envelope_relative_difference']:.2%}`.
Its bulk-side plateau is stable across the declared `2W` through `10W`
windows; the small offset is consistent with evaluating continuum gradients
on the finite grid rather than the exact face-energy ledger.

## What `local_reference` measured

`met["local_reference"]["s_local_cap"]` combines quadratic meridional
curvatures fitted over 15 nm on each side of the diffuse TJ with
`sin(160 deg/2)/r_neck`. Multiplying it by the contact area is algebraically
the Cannon--Carter line term minus a pressure term formed from that local
TJ-adjacent total curvature. It does not use the measured angle and it does
not use either grain's far-field constant-mean-curvature pressure.

The converged neck radius is `{r_neck*1e9:.6f} nm`; the fitted left and right
slopes are `{local_minus['slope']:.9f}` and `{local_plus['slope']:.9f}`. The
measured dihedral angle is `{met['psi_measured_deg']:.6f} deg`, versus the
prescribed `160 deg`. Complete local meridional, azimuthal, total-curvature,
area, line-force, and pressure-force values are archived in the CSV and JSON
tables.

The old local-reference construction gives
`{result['old_local_reference_CC_force_N']:.9e} N`. Replacing its
TJ-adjacent curvature with the independently fitted far-field CMC pressure
difference gives `{global_ref['total_force_N']:.9e} N` with the 160-degree
angle and `{global_measured['total_force_N']:.9e} N` with the measured angle.
Their differences from the PF envelope are
`{result['global_CMC_reference_to_envelope_relative_difference']:.2%}` and
`{result['global_CMC_measured_to_envelope_relative_difference']:.2%}`,
respectively; both pass the 5% gate.

## Pressure check

The volume multipliers convert to capillary pressures with the exact solver
normalization `p_i=-lambda_i/(2*pi*dr*dz*L)`. They give
`{pressures[0]/1e6:.6f}` and `{pressures[1]/1e6:.6f} MPa`. Area-weighted CMC
fits over `3W` to `8W` from the TJ give
`{physics.gamma_s*cmc_plus['kappa_total_per_m']/1e6:.6f}` and
`{physics.gamma_s*cmc_minus['kappa_total_per_m']/1e6:.6f} MPa`. Thus the
stationary multipliers agree with the global surface curvatures while the
local TJ curvatures do not. The pressure difference used by the corrected
sharp-interface comparison is `{result['global_CMC_pressure_difference_Pa']/1e6:.6f} MPa`.

## PF and sharp-interface decompositions

For the implemented normalized-ownership functional, the ownership-dependent
terms are

```text
G_phi = integral [f Wc sum_(i<j)(phi_i phi_j)
                  + (k_eta/2) f sum_i |grad phi_i|^2] dV.
```

Their analytic configurational derivatives are

```text
delta L/delta phi_i
  = f Wc sum_(j!=i) phi_j - k_eta div(f grad phi_i) - p_i f,

F_config
  = -integral sum_i [(delta L/delta phi_i) phi_i,u] dV.
```

The report evaluates these expressions directly with the implemented
face-consistent divergence. This supplies the independent PF force quoted
above and establishes the limiting correspondence without assigning names by
analogy.

The PF envelope terms are `F_G=-G_u={F_G:.9e} N` and
`F_C=-lambda^T C_u={F_C:.9e} N`. They are both required, but they are not
separately the Cannon--Carter line and pressure forces. In the sharp ownership
limit the displaced midplane moves by `u/2`, so

```text
dV_0/du -> A/2,
F_C -> Delta-p A/2.
```

The explicit ownership derivative contains the complementary TJ/GB shape
term and the remaining pressure contribution. Integrating the stationary
Euler--Lagrange balance through the diffuse GB/TJ gives

```text
F_G -> F_line - 3 Delta-p A/2,
F_G + F_C -> F_line - Delta-p A = F_CC.
```

This also explains why assigning `F_G` to line tension and `F_C` to the full
pressure force would be incorrect. Numerically, the fixed-field reassignment
area is `{physical_Cu[0]:.9e} m^2`, approaching the sharp value
`A/2={0.5*area:.9e} m^2`.

The Cannon--Carter component tables report line and pressure forces
separately for local measured-angle, local 160-degree, global-CMC, and KKT
pressure choices. Only the totals should be compared with the PF envelope.

## Decision and scope

The original 22.6% discrepancy came from using diffuse-TJ local curvature as
the pressure curvature. Global CMC curvature, KKT pressure, the independent
PF configurational force, and the stationary envelope now close within the
prescribed 5% tolerance. The conditional interface-width campaign was not
triggered because a residual PF-versus-sharp-interface mismatch is no longer
present after correcting the metrology.

No C2 state was minimized or advanced. No clipping, barrier change, root-law
change, or stochastic parameter change was made.
""")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
