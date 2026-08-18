"""RECOVERY geometry audit: quantify the analytic TJ/fillet/body
construction's continuity (position, tangent, curvature) using
arclength-based derivatives (robust to near-vertical tangents, unlike
R(z) derivatives), for the exact production parameters
(chi=1.5, ratio=0.185, psi=160deg, R_p=1000nm).

Reports, per branch (particle z>=0, substrate/neighbor z<=0):
  - fillet-to-body join location
  - curvature just inside the fillet vs just inside the body (the C1-only
    discontinuity)
  - the measured dihedral angle (outward tangent rays at the TJ)
  - theta(s), kappa_m(s), kappa_a(s) vs arclength from the TJ
"""
from __future__ import annotations

import math

import numpy as np

from pf_sintering.m16j_geometry import (
    particle_R_of_z,
    solve_body_c2_transition,
    substrate_R_of_z_sphere,
    young_herring_slope,
)

solve_body_fillet = solve_body_c2_transition  # audit the CORRECTED construction

PSI_DEG = 160.0
R_P_NM = 1000.0
CHI = 1.5
RATIO = 0.185
W_NM = 6.0


def arclength_derivatives(r, z):
    """r,z: 1-D arrays ordered along the branch (from the TJ outward).
    Returns s (arclength from the first point), theta(s) (tangent angle
    from +r axis, radians), kappa_m(s) (signed meridional curvature),
    kappa_a(s) (azimuthal curvature z'/r)."""
    ds = np.hypot(np.diff(r), np.diff(z))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    # resample onto a uniform-in-s grid for stable finite differences
    s_uniform = np.linspace(s[0], s[-1], len(s))
    r_u = np.interp(s_uniform, s, r)
    z_u = np.interp(s_uniform, s, z)
    dr_ds = np.gradient(r_u, s_uniform)
    dz_ds = np.gradient(z_u, s_uniform)
    norm = np.hypot(dr_ds, dz_ds)
    dr_ds, dz_ds = dr_ds / norm, dz_ds / norm
    theta = np.arctan2(dz_ds, dr_ds)
    d2r_ds2 = np.gradient(dr_ds, s_uniform)
    d2z_ds2 = np.gradient(dz_ds, s_uniform)
    kappa_m = dr_ds * d2z_ds2 - dz_ds * d2r_ds2
    kappa_a = np.where(r_u > 1e-12, dz_ds / np.maximum(r_u, 1e-12), np.nan)
    return s_uniform, theta, kappa_m, kappa_a, r_u, z_u


def audit_particle_branch(a, m, R_p, W_nm, n_probe=20000, s_max=700.0):
    fil = solve_body_fillet(a, m, R_p, R_p, +1.0)
    z_T = fil["z_tangent"]
    z_hi = z_T + 4.0 * (R_p * 0.001 + 50.0) + 100.0  # generous, well past the tangent point
    z_probe = np.linspace(0.0, min(z_hi, 900.0), n_probe)
    R_of_z, info = particle_R_of_z(z_probe, a, m, R_p, 8.0, W_nm)
    finite = np.isfinite(R_of_z)
    r, z = R_of_z[finite], z_probe[finite]
    s, theta, kappa_m, kappa_a, r_u, z_u = arclength_derivatives(r, z)
    mask = s <= s_max
    return dict(s=s[mask], theta=theta[mask], kappa_m=kappa_m[mask], kappa_a=kappa_a[mask],
                r=r_u[mask], z=z_u[mask], fillet=fil, z_tangent=z_T, R_body=R_p,
                s_tangent=float(np.interp(z_T, z_u, s)) if z_T <= z_u[-1] else float("nan"))


def audit_substrate_branch(a, m, R_s, W_nm, n_probe=20000, s_max=700.0):
    fil = solve_body_fillet(a, m, R_s, -R_s, -1.0)
    z_T = fil["z_tangent"]
    z_lo = z_T - 4.0 * (R_s * 0.001 + 50.0) - 100.0
    z_probe = np.linspace(max(z_lo, -900.0), 0.0, n_probe)
    R_of_z, info = substrate_R_of_z_sphere(z_probe, a, m, R_s, 8.0, W_nm)
    finite = np.isfinite(R_of_z)
    r, z = R_of_z[finite], z_probe[finite]
    order = np.argsort(-z)  # from TJ (z=0) outward to more-negative z, matching particle's "outward" convention
    r, z = r[order], z[order]
    s, theta, kappa_m, kappa_a, r_u, z_u = arclength_derivatives(r, z)
    mask = s <= s_max
    return dict(s=s[mask], theta=theta[mask], kappa_m=kappa_m[mask], kappa_a=kappa_a[mask],
                r=r_u[mask], z=z_u[mask], fillet=fil, z_tangent=z_T, R_body=R_s,
                s_tangent=float(np.interp(-z_T, -z_u, s)) if z_T >= z_u[-1] else float("nan"))


def main():
    m = young_herring_slope(PSI_DEG)
    a = RATIO * R_P_NM
    R_p = R_P_NM
    R_s = CHI * R_P_NM
    print(f"a={a}nm m={m:.6f} (psi={PSI_DEG}deg)")

    part = audit_particle_branch(a, m, R_p, W_NM)
    sub = audit_substrate_branch(a, m, R_s, W_NM)

    print("\n=== PARTICLE branch ===")
    print(f"  transition length z1={part['fillet']['z1']:.3f}nm  overshoot={part['fillet'].get('overshoot','n/a')}")
    print(f"  body radius R_p={R_p}nm  curvature 1/R_p={1/R_p:.6f} 1/nm")
    print(f"  fillet-to-body join point at s={part['s_tangent']:.3f}nm from TJ "
          f"(r,z)=({part['fillet']['r_tangent']:.3f},{part['fillet']['z_tangent']:.3f})nm")

    print("\n=== SUBSTRATE/NEIGHBOR branch ===")
    print(f"  transition length z1={sub['fillet']['z1']:.3f}nm  overshoot={sub['fillet'].get('overshoot','n/a')}")
    print(f"  body radius R_s={R_s}nm  curvature 1/R_s={1/R_s:.6f} 1/nm")
    print(f"  fillet-to-body join point at s={sub['s_tangent']:.3f}nm from TJ "
          f"(r,z)=({sub['fillet']['r_tangent']:.3f},{sub['fillet']['z_tangent']:.3f})nm")

    # dihedral angle: outward tangent direction at s->0+ for each branch
    theta_p0 = part["theta"][2]  # skip the very first sample (numerical noise at s=0)
    theta_s0 = sub["theta"][2]
    print(f"\n=== Dihedral angle check ===")
    print(f"  particle outward tangent angle (from +r axis) = {math.degrees(theta_p0):.3f}deg")
    print(f"  substrate outward tangent angle (from +r axis) = {math.degrees(theta_s0):.3f}deg")
    included = math.degrees(abs(theta_p0 - theta_s0))
    print(f"  included angle between outward tangent rays = {included:.3f}deg")
    print(f"  supplementary (180-included) = {180-included:.3f}deg")
    print(f"  target psi_deg = {PSI_DEG}deg -- {'MATCHES included angle' if abs(included-PSI_DEG)<1.0 else ('MATCHES supplementary' if abs(180-included-PSI_DEG)<1.0 else 'NO MATCH within 1deg')}")

    # curvature/tangent-angle jump AT the fillet-body join (finite-difference straddle)
    for label, br in (("particle", part), ("substrate", sub)):
        s_t = br["s_tangent"]
        if not np.isfinite(s_t):
            continue
        i = int(np.searchsorted(br["s"], s_t))
        i = max(2, min(i, len(br["s"]) - 3))
        kappa_before = br["kappa_m"][i - 2]
        kappa_after = br["kappa_m"][i + 2]
        theta_before = br["theta"][i - 2]
        theta_after = br["theta"][i + 2]
        print(f"\n  [{label}] at fillet/body join (s~{s_t:.2f}nm): "
              f"theta jump={math.degrees(theta_after-theta_before):.4f}deg  "
              f"kappa_m jump={kappa_after-kappa_before:.6e} 1/nm "
              f"(kappa_m before={kappa_before:.6e}, after={kappa_after:.6e})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label, br, color in (("particle", part, "C0"), ("substrate/neighbor", sub, "C1")):
        axes[0, 0].plot(br["s"], np.degrees(br["theta"]), color=color, label=label)
        axes[0, 1].plot(br["s"], br["kappa_m"], color=color, label=label)
        axes[1, 0].plot(br["s"], br["kappa_a"], color=color, label=label)
        axes[1, 1].plot(br["r"], br["z"], color=color, label=label)
        if np.isfinite(br["s_tangent"]):
            for ax, key in ((axes[0, 0], "theta"), (axes[0, 1], "kappa_m"), (axes[1, 0], "kappa_a")):
                ax.axvline(br["s_tangent"], color=color, ls="--", alpha=0.4)
    axes[0, 0].set_xlabel("s (nm from TJ)"); axes[0, 0].set_ylabel("theta (deg)"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xlabel("s (nm from TJ)"); axes[0, 1].set_ylabel("kappa_m (1/nm)"); axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_xlabel("s (nm from TJ)"); axes[1, 0].set_ylabel("kappa_a (1/nm)"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_xlabel("r (nm)"); axes[1, 1].set_ylabel("z (nm)"); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_aspect("equal", adjustable="box")
    fig.suptitle("CURRENT (circular-fillet, C1-only) analytic geometry audit")
    fig.tight_layout()
    fig.savefig("recovery_geometry_audit_corrected.png", dpi=120)
    plt.close(fig)
    print("\nWrote recovery_geometry_audit_corrected.png")


if __name__ == "__main__":
    main()
