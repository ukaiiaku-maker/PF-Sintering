"""Milestone 14B Section 8: save a canonical full-field checkpoint state for
the combined high-amplitude / high-gamma_gb condition at a specific physical
time, for later local-stress-calculation use. Re-runs the SAME deterministic
construction/stepping as m14_mechanism_screen.py from t=0 up to the
requested time, then dumps f, eta fields, mu, the authoritative face flux,
TJ geometry, the neck control-volume mask, F, D_h, and physical/reduced time
to a compressed .npz. Does not activate the sink.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from m14_mechanism_screen import BC_X, BC_Y, build_state, step_f  # noqa: E402

from pf_sintering.ch_crossover_diagnostics import neck_region_mask
from pf_sintering.ch_exact_energy import exact_free_energy_isotropic, mu_isotropic
from pf_sintering.model import Sink
from pf_sintering.surface_transport import (
    exact_dissipation_face_projected,
    m_s_ref,
    surface_flux_face_projected,
)
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--r2-nm", type=float, default=80.0)
    ap.add_argument("--aspect-ratio", type=float, default=2.0)
    ap.add_argument("--overlap-nm", type=float, default=20.0)
    ap.add_argument("--wavelength-nm", type=float, default=480.0)
    ap.add_argument("--amplitude-nm", type=float, default=72.0)
    ap.add_argument("--w-nm", type=float, default=20.0)
    ap.add_argument("--surface-mobility-scale", type=float, default=0.9)
    ap.add_argument("--gamma-gb-override", type=float, default=1.4142136)
    ap.add_argument("--dt-override", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dx-nm", type=float, default=2.5)
    ap.add_argument("--mobility-multiple-for-tau", type=float, default=3.0)
    ap.add_argument("--save-time", type=float, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    p, f, e1, e2, e3 = build_state(args)
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)

    n_target = round(args.save_time / p.dt)
    step = 0
    while step < n_target:
        f, diag = step_f(f, e1, e2, e3, s, p, M_s)
        step += 1
    t = step * p.dt
    tau = args.mobility_multiple_for_tau * t

    rep = compute_neck_tj_forces(f, e1, e2, e3, s, p)
    sub = compute_subgrid_contact(f, e1, e2, p)
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    fp = surface_flux_face_projected(f, mu, p.dx, p.interface_width, M_s, BC_X, BC_Y)
    F = exact_free_energy_isotropic(f, e1, e2, e3, s, p)
    Dx, Dy = exact_dissipation_face_projected(fp)
    D_h = (p.dx * p.dx) * (float(np.sum(Dx)) + float(np.sum(Dy)))

    neck_mask = None
    M_neck = math.nan
    if sub.resolved:
        neck_mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
        M_neck = float((f * neck_mask).sum()) * p.dx * p.dx

    np.savez_compressed(
        args.out,
        f=f, e1=e1, e2=e2, e3=e3, mu=mu,
        Jx_face=fp["Jx_face"], Jy_face=fp["Jy_face"],
        tj_top=np.array(rep.top.tj_xy if rep.top and rep.top.resolved else [math.nan, math.nan]),
        tj_bottom=np.array(rep.bottom.tj_xy if rep.bottom and rep.bottom.resolved else [math.nan, math.nan]),
        psi_top_deg=rep.top.psi_deg if rep.top and rep.top.resolved else math.nan,
        psi_bottom_deg=rep.bottom.psi_deg if rep.bottom and rep.bottom.resolved else math.nan,
        neck_mask=neck_mask if neck_mask is not None else np.zeros_like(f, dtype=bool),
        L_contact_TJ_sub=sub.L_contact_TJ_sub if sub.resolved else math.nan,
        M_neck_f=M_neck, F=F, D_h=D_h, t=t, tau=tau, dx=p.dx, Nx=p.Nx, Ny=p.Ny,
        gamma_gb=p.gamma_gb, amplitude_nm=args.amplitude_nm, surface_mobility_scale=args.surface_mobility_scale,
        label=args.label,
    )
    print(f"saved {args.out}: label={args.label} t={t:.4e}s tau={tau:.4e}s M_neck_f={M_neck:.6e} "
          f"L_contact={sub.L_contact_TJ_sub*1e9 if sub.resolved else float('nan'):.3f}nm F={F:.6e} D_h={D_h:.4e}")


if __name__ == "__main__":
    main()
