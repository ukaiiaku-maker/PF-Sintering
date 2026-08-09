"""Milestone 14E Sections 6-8: static Young-Herring TJ relaxation and
dynamic effective TJ mobility audit.

Reuses the qualified sinusoidal-substrate/particle-contact geometry
(m14_mechanism_screen.build_config) but, unlike every Milestone 12-14D
run, turns eta kinetics ON (constrained_eta.constrained_tangent_cone_eta_
update) alongside the qualified face_projected surface transport
(variational_surface_diffusion_step). f and eta relax together from the
unmodified analytic initial condition -- psi is never prescribed. Tracks
psi(t), F_TJ(t), and TJ position for several gamma_gb/gamma_s ratios, and
reports the TJ's own (unprescribed) approach toward, or departure from,
the nominal Young angle 2*gamma_s*cos(psi_eq/2)=gamma_gb.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m14_mechanism_screen import BC_X, BC_Y, build_config  # noqa: E402

from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update, f_weighted_ownership_volumes
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step
from pf_sintering.tj_force import compute_neck_tj_forces


def build_state(args):
    p = build_params(build_config(args))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def run(args, target_times, M_eta_scale):
    p, f, e1, e2, e3 = build_state(args)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    p.M_eta = args.M_eta_base * M_eta_scale
    psi_Y_nominal = math.degrees(2 * math.acos(min(1.0, p.gamma_gb / (2 * p.gamma_s))))
    print(f"=== m14e TJ balance {args.label}: gamma_gb={p.gamma_gb:.4f} psi_Y_nominal={psi_Y_nominal:.2f}deg "
          f"M_eta={p.M_eta:.4e} dx={args.dx_nm}nm Nx={p.Nx} Ny={p.Ny} dt={p.dt:.4e}s ===")

    mass0 = float(f.sum()) * p.dx * p.dx
    V10, V20, _ = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)

    def sample(step, t):
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
                row[f"F_TJ_x_{label}"] = tj.F_TJ_x
                row[f"tj_xy_{label}"] = tuple(float(v) for v in tj.tj_xy)
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
            e1, e2, e3, ediag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, bc_x=BC_X, bc_y=BC_Y)
            step += 1
        row = sample(step, step * p.dt)
        rows.append(row)
        print(f"  t={step*p.dt:.4e}s psi_top={row['psi_top_deg']:.2f}deg F_TJ_top={row['F_TJ_mag_top']:.4e} "
              f"F={row['F']:.6e} mass_drift={row['mass_drift']:.2e} V_drift={row['V_drift']:.2e}")

    return dict(label=args.label, gamma_gb=p.gamma_gb, gamma_s=p.gamma_s, psi_Y_nominal=psi_Y_nominal,
                M_eta=p.M_eta, dx_nm=args.dx_nm, p_dt=p.dt, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", type=str, default="baseline")
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=24.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.3)
    ap.add_argument("--gamma-gb-override", type=float, default=None)
    ap.add_argument("--dt-override", type=float, default=None)
    ap.add_argument("--M-eta-base", type=float, default=4.266666666666666e-09)
    ap.add_argument("--M-eta-scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--times", type=str, default="0,0.01,0.03,0.06,0.10,0.15,0.20")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    target_times = [float(x) for x in args.times.split(",")]
    result = run(args, target_times, args.M_eta_scale)

    with open(args.out, "w") as fh:
        json.dump(result, fh, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
