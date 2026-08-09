"""Milestone 14F Section 17C Question A: thermodynamic TJ qualification
AFTER the gamma_GB normalization fix (constrained_eta.gamma_gb_target_
to_declared, Milestone 14E Section 4/5).

Deliberately idealized setup (NOT the rate-competition question, Section
17C Question B): surface mobility is boosted well above the physical
baseline so free-surface relaxation is fast/non-rate-limiting, isolating
the THERMODYNAMIC question -- given adequate time and adequate free-
surface relaxation, does the unprescribed TJ (both f and eta relaxing
together, psi never imposed) approach F_TJ->0 and
psi->2*acos(gamma_gb/(2*gamma_s)) (Young-Herring)? Reuses the qualified
sinusoidal-substrate/particle-contact geometry and the qualified
face_projected transport + constrained tangent-cone eta integrator
unmodified -- only the gamma_gb CALIBRATION and surface_mobility_scale
are set for this specific idealized test.

IMPORTANT decoupling (found empirically in this milestone): `gamma_gb`
serves two DIFFERENT roles in the existing code, both currently read
from the SAME `p.gamma_gb_ref` -- (a) the F-field's OWN groove-coupling
strength `Wc=36*gamma_gb_ref/W` inside `ch_exact_energy.mu0_bulk` (the
mechanism Milestones 14/14B/14C established and calibrated using the
DECLARED value directly, always with eta FROZEN), and (b) the eta
profile's own static/kinetic energy inside `constrained_eta.py`'s
`structural_thermodynamic_force` (which Milestone 14E Section 4 showed
needs the `GB_ENERGY_CALIBRATION_FACTOR` correction). Applying the
correction globally (via `gamma_gb_override`) fixes (b) but silently
WEAKENS (a) by the same 19x, breaking the established Wc-driven
mechanism. This script keeps `p.gamma_gb_ref` at the TARGET (uncorrected)
value for the f-transport step, and passes a SEPARATE, corrected `Params`
copy (`dataclasses.replace`, only `gamma_gb_ref` changed) to the eta
update -- the two roles are decoupled rather than sharing one value.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from m14_mechanism_screen import BC_X, BC_Y, build_config  # noqa: E402

from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.constrained_eta import (
    constrained_tangent_cone_eta_update,
    f_weighted_ownership_volumes,
    gamma_gb_target_to_declared,
)
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step
from pf_sintering.tj_force import compute_neck_tj_forces


def build_state(args, gamma_gb_target):
    args2 = argparse.Namespace(**vars(args))
    args2.gamma_gb_override = gamma_gb_target
    p = build_params(build_config(args2))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def run(args, target_times):
    gamma_gb_target = args.gamma_gb_target
    gamma_gb_declared_for_eta = gamma_gb_target_to_declared(gamma_gb_target)
    p, f, e1, e2, e3 = build_state(args, gamma_gb_target)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    p.M_eta = args.M_eta_base * args.M_eta_scale
    p_eta = dataclasses.replace(p, gamma_gb_ref=gamma_gb_declared_for_eta)
    psi_Y_nominal = math.degrees(2 * math.acos(min(1.0, gamma_gb_target / (2 * p.gamma_s))))
    print(f"=== m14f TJ thermo {args.label}: gamma_gb_target={gamma_gb_target:.4f} "
          f"(p.gamma_gb_ref, f-coupling)  gamma_gb_declared_for_eta={gamma_gb_declared_for_eta:.6f} "
          f"psi_Y_nominal={psi_Y_nominal:.2f}deg "
          f"M_eta={p.M_eta:.4e} M_s={M_s:.4e} surface_mobility_scale={args.surface_mobility_scale} "
          f"dx={args.dx_nm}nm Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s ===")

    mass0 = float(f.sum()) * p.dx * p.dx
    V10, V20, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)

    def sample(step, t):
        # p.gamma_gb_ref == gamma_gb_target here, so tj_force.gb_vector's
        # effective_gamma(s,p) (and hence F_TJ/xi_gb) is ALREADY evaluated
        # against the target GB energy -- no separate correction needed
        # (contrast the single-p version tried earlier in this milestone).
        rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
        F = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
        mass = float(f.sum()) * p.dx * p.dx
        V1, V2, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
        row = dict(step=step, time_s=t, mass=mass, mass_drift=(mass - mass0) / mass0, F=F,
                   V1=V1, V2=V2, V_drift=((V1 + V2) - (V10 + V20)) / (V10 + V20))
        for label, tj in (("top", rep.top), ("bottom", rep.bottom)):
            if tj is not None and tj.resolved:
                row[f"psi_{label}_deg"] = tj.psi_deg
                row[f"F_TJ_mag_{label}"] = tj.F_TJ_mag
            else:
                row[f"psi_{label}_deg"] = math.nan
                row[f"F_TJ_mag_{label}"] = math.nan
        return row

    rows = [sample(0, 0.0)]
    step = 0
    for t_target in target_times:
        if t_target <= 0.0:
            continue
        n_target = round(t_target / p.dt)
        while step < n_target:
            mu = mu_isotropic(f, e1, e2, e3, s, p)
            f, fdiag = variational_surface_diffusion_step(f, mu, p.dx, p.dt, p.interface_width, M_s,
                                                            bc_x=BC_X, bc_y=BC_Y)
            e1, e2, e3, ediag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p_eta, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        row = sample(step, step * p.dt)
        rows.append(row)
        print(f"  t={step*p.dt:.4e}s psi_top={row['psi_top_deg']:.2f}deg F_TJ_top={row['F_TJ_mag_top']:.4e} "
              f"F={row['F']:.6e} mass_drift={row['mass_drift']:.2e} V2={row['V2']:.6e}")

    return dict(label=args.label, gamma_gb_target=gamma_gb_target, gamma_gb_declared_for_eta=gamma_gb_declared_for_eta,
                gamma_s=p.gamma_s, psi_Y_nominal=psi_Y_nominal, M_eta=p.M_eta, M_s=M_s,
                surface_mobility_scale=args.surface_mobility_scale, dx_nm=args.dx_nm, p_dt=p.dt, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", type=str, default="baseline")
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=3.0)
    ap.add_argument("--gamma-gb-target", type=float, default=1.0)
    ap.add_argument("--dt-override", type=float, default=None)
    ap.add_argument("--M-eta-base", type=float, default=4.266666666666666e-09)
    ap.add_argument("--M-eta-scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--times", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    target_times = [float(x) for x in args.times.split(",")]
    result = run(args, target_times)

    with open(args.out, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
