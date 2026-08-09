"""Milestone 14D Sections 1-7: substrate/particle width metrics and the
initial-geometry (t=0) comparison figure for the old (A=72nm/lambda=480nm)
vs. new (A=100nm/lambda=320nm) sinusoidal-substrate candidates, both at
gamma_gb=1.4142136. Run from the repository root; writes
runs/m14c_figs/m14d_fig0_initial_geometry.png and prints the width-metric
table (W_sub_50, W_sub_contact, W_particle_full, W_particle_at_TJ and
their ratios -- see the module docstring derivation in the Milestone 14D
report, Sections 2-4)."""

import math
import sys

sys.path.insert(0, "scripts")
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m14_mechanism_screen import build_config
from pf_sintering.model import Sink, build_params, initialize_fields, reproject
from pf_sintering.tj_force import compute_tj_force
from pf_sintering.tj_subgrid import compute_subgrid_contact


def build(A_nm, lam_nm, gamma_gb, dx_nm=2.5, mobility=0.9):
    args = argparse.Namespace(dx_nm=dx_nm, r2_nm=80.0, aspect_ratio=2.0, overlap_nm=20.0,
                               wavelength_nm=lam_nm, amplitude_nm=A_nm, w_nm=20.0,
                               surface_mobility_scale=mobility, gamma_gb_override=gamma_gb,
                               dt_override=None, seed=42)
    p = build_params(build_config(args))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3, args

def main():
    geoms = [(72.0, 480.0, "OLD A72/lam480"), (100.0, 320.0, "NEW A100/lam320")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    rows = []
    for ax, (A, lam, label) in zip(axes, geoms):
        p, f, e1, e2, e3, args = build(A, lam, 1.4142136)
        s = Sink(threshold=math.inf)
        sub = compute_subgrid_contact(f, e1, e2, p)
        top = compute_tj_force(f, e1, e2, (sub.top.x_sub, sub.top.y_sub), s, p) if sub.resolved else None
        bot = compute_tj_force(f, e1, e2, (sub.bottom.x_sub, sub.bottom.y_sub), s, p) if sub.resolved else None
        psi_top = top.psi_deg if top and top.resolved else float("nan")
        psi_bot = bot.psi_deg if bot and bot.resolved else float("nan")

        extent = [0, p.Nx * p.dx * 1e9, 0, p.Ny * p.dx * 1e9]
        ax.imshow(f, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1, alpha=0.7)
        if sub.resolved:
            ax.plot(sub.top.x_sub * 1e9, sub.top.y_sub * 1e9, "g^", ms=10)
            ax.plot(sub.bottom.x_sub * 1e9, sub.bottom.y_sub * 1e9, "gv", ms=10)
        ax.set_title(f"{label}\nL_contact0={sub.L_contact_TJ_sub*1e9:.2f}nm psi_top={psi_top:.1f}deg "
                     f"psi_bot={psi_bot:.1f}deg", fontsize=10)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")

        x_tj = 0.5 * (sub.top.x_sub + sub.bottom.x_sub)
        wall_mean = p.substrate_wall_frac * p.Nx * p.dx
        Rx, Ry = p.Rx, p.Ry
        Am, lamm = A * 1e-9, lam * 1e-9
        val = (x_tj - wall_mean) / Am
        W_sub_contact = (lamm * math.acos(val) / math.pi) if abs(val) <= 1 else float("nan")
        cx = wall_mean + Am + Rx - p.initial_overlap
        val2 = 1 - ((x_tj - cx) / Rx) ** 2
        W_particle_at_tj = 2 * Ry * math.sqrt(val2) if val2 >= 0 else float("nan")
        max_slope_deg = math.degrees(math.atan(Am * 2 * math.pi / lamm))
        rows.append(dict(label=label, A=A, lam=lam, W_sub_50=(lamm / 3) * 1e9, W_particle_full=2 * Ry * 1e9,
                          W_sub_contact=W_sub_contact * 1e9, W_particle_at_tj=W_particle_at_tj * 1e9,
                          ratio_50=(lamm / 3) / (2 * Ry), ratio_contact=W_sub_contact / W_particle_at_tj,
                          max_slope_deg=max_slope_deg, L_contact0=sub.L_contact_TJ_sub * 1e9,
                          psi_top=psi_top, psi_bot=psi_bot, x_tj=x_tj * 1e9))

    fig.suptitle("Milestone 14D: initial geometry, old (A72/480) vs new (A100/320) candidate")
    fig.tight_layout()
    fig.savefig("runs/m14c_figs/m14d_fig0_initial_geometry.png", dpi=140)
    print("wrote initial geometry figure")

    print(f"{'label':20s} {'A':>5s} {'lam':>5s} {'W_sub_50':>9s} {'W_part_full':>11s} {'ratio_50':>8s} "
          f"{'W_sub_ctc':>10s} {'W_part_tj':>10s} {'ratio_ctc':>9s} {'slope_deg':>9s} {'L0(nm)':>7s} {'psi0':>6s}")
    for r in rows:
        print(f"{r['label']:20s} {r['A']:5.0f} {r['lam']:5.0f} {r['W_sub_50']:9.2f} {r['W_particle_full']:11.2f} "
              f"{r['ratio_50']:8.3f} {r['W_sub_contact']:10.2f} {r['W_particle_at_tj']:10.2f} "
              f"{r['ratio_contact']:9.3f} {r['max_slope_deg']:9.1f} {r['L_contact0']:7.2f} {r['psi_top']:6.2f}")


if __name__ == "__main__":
    main()
