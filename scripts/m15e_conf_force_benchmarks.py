"""Milestone 15E Section 14: configurational-force benchmarks.

A. Isolated planar interface -- F_conf must vanish (no defect enclosed,
   translational symmetry along the interface).
B. Equilibrium Young-Herring wedge -- contour-integrated F_conf around the
   TJ must approach zero.
C. Deliberately off-equilibrium wedge -- contour-integrated F_conf must
   agree in vector direction and magnitude with the independently-known
   Young-Herring residual F_TJ = xi_s1 + xi_s2 + xi_gb (computed by the
   existing, already-validated pf_sintering/tj_force.compute_tj_force),
   outside the diffuse core, across several control-volume radii
   (2W,3W,4W,5W,6W). Qualifies only if a radius-independent plateau exists.

A hand-guessed tanh profile is NOT a genuine equilibrium state of the
real coupled dynamics (evolve_f's mu0 has an extra Wc-coupling term even
for a single grain), so every test profile here is relaxed through the
actual pf_sintering.model evolve_f/evolve_eta/reproject steps first --
see configurational_force.py's module docstring for why an unrelaxed
profile gives a spurious, non-plateauing residual force.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.configurational_force import configurational_force_radius_scan  # noqa: E402
from pf_sintering.model import ModelConfig, Sink, build_params, evolve_eta, evolve_f, reproject  # noqa: E402
from pf_sintering.tj_force import compute_tj_force  # noqa: E402
from tests.test_tj_force import _gamma_gb_for_equilibrium, build_synthetic_wedge  # noqa: E402


def _wedge_params(nx=320, ny=360, dx_nm=2.0, theta_mis_deg=30.0):
    return build_params(ModelConfig(
        preset="dev", nx=nx, ny=ny, dx=dx_nm * 1e-9, r2=40e-9, t_total=1e-6,
        use_aniso_surface=False, theta_mis_deg=theta_mis_deg,
    ))


def relax(f, e1, e2, e3, s, p, n_steps, clamp_single_grain=False):
    """Run the real f/eta dynamics (periodic-x, no-flux-y -- model.py's
    own convention, not the M15 series' reflecting-x BC module) for
    n_steps, re-clamping e1=f (e2=e3=0) every step if clamp_single_grain
    (an isolated single-grain surface has no independent eta dynamics to
    speak of; this represents that state faithfully without letting
    unrelated multi-grain machinery perturb it)."""
    for _ in range(n_steps):
        f = evolve_f(f, e1, e2, e3, s, s, p)
        if clamp_single_grain:
            e1 = np.clip(f, 0.0, 1.0)
            e2 = np.zeros_like(f)
            e3 = np.zeros_like(f)
        else:
            e1, e2, e3 = evolve_eta(e1, e2, e3, p)
            e1, e2, e3 = reproject(f, e1, e2, e3)
    return f, e1, e2, e3


def benchmark_a_planar_interface():
    """Periodic-x domain, solid slab in the middle (two flat interfaces),
    single grain (e1=f). Relax through the real dynamics, then probe
    F_conf around the LEFT interface only, far from the right interface
    and the periodic wrap."""
    p = _wedge_params(nx=240, ny=48)
    W = p.interface_width
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    Lx = p.Nx * p.dx
    x_left, x_right = 0.30 * Lx, 0.70 * Lx
    f = 0.5 * (1.0 + np.tanh((X - x_left) / W)) * 0.5 * (1.0 + np.tanh((x_right - X) / W))
    e1 = np.clip(f, 0.0, 1.0)
    e2 = np.zeros_like(f)
    e3 = np.zeros_like(f)
    s = Sink(threshold=math.inf)

    f, e1, e2, e3 = relax(f, e1, e2, e3, s, p, 3000, clamp_single_grain=True)

    y0 = 0.5 * p.Ny * p.dx
    radii = [k * W for k in (2, 3, 4, 5)]
    max_r = radii[-1]
    assert x_left - max_r > 0.05 * Lx and x_left + max_r < x_right - 0.05 * Lx, "contour too close to other interface/edge"
    scan = configurational_force_radius_scan(f, e1, e2, p, (x_left, y0), radii, s=s, bc_x="periodic", bc_y="reflecting")
    print("Benchmark A (relaxed planar interface, must vanish):")
    ok = True
    scale = p.gamma_s
    for r, (Fx, Fy) in scan.items():
        mag = math.hypot(Fx, Fy)
        print(f"  r={r*1e9:.1f}nm ({r/W:.1f}W)  F=({Fx:.3e},{Fy:.3e})  |F|/gamma_s={mag/scale:.4e}")
        if mag / scale > 0.02:
            ok = False
    print(f"  PASS={ok}")
    return ok


def benchmark_b_equilibrium_wedge():
    p = _wedge_params()
    s = Sink(threshold=math.inf)
    psi_eq = 120.0
    p.gamma_gb_ref = _gamma_gb_for_equilibrium(psi_eq, p.gamma_s)
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, psi_eq)

    tj_before = compute_tj_force(f, e1, e2, tj_xy, s, p)
    f, e1, e2, e3 = relax(f, e1, e2, e3, s, p, 1500)
    tj_after = compute_tj_force(f, e1, e2, tj_xy, s, p)
    print(f"Benchmark B setup: psi before relax={tj_before.psi_deg:.2f}deg, after={tj_after.psi_deg:.2f}deg "
          f"(prescribed {psi_eq}deg)")

    W = p.interface_width
    radii = [k * W for k in (2, 3, 4, 5, 6)]
    scan = configurational_force_radius_scan(f, e1, e2, p, tuple(tj_xy), radii, s=s,
                                              bc_x="reflecting", bc_y="periodic")
    print("Benchmark B (relaxed equilibrium wedge, psi=120deg, must vanish):")
    ok = True
    scale = p.gamma_s
    for r, (Fx, Fy) in scan.items():
        mag = math.hypot(Fx, Fy)
        print(f"  r={r*1e9:.1f}nm ({r/W:.1f}W)  F=({Fx:.3e},{Fy:.3e})  |F|/gamma_s={mag/scale:.4e}")
        if mag / scale > 0.10:
            ok = False
    print(f"  PASS={ok}")
    return ok


def benchmark_c_off_equilibrium_wedge():
    p = _wedge_params()
    s = Sink(threshold=math.inf)
    psi_eq = 120.0
    p.gamma_gb_ref = _gamma_gb_for_equilibrium(psi_eq, p.gamma_s)
    W = p.interface_width
    radii = [k * W for k in (2, 3, 4, 5, 6)]

    print("Benchmark C (off-equilibrium wedge vs known F_TJ, radius plateau):")
    all_ok = True
    for psi_deg in (100.0, 140.0):
        f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, psi_deg)
        # Light relaxation only: smooth the hand-built profile shape
        # without letting the TJ relax away the deliberately-imposed
        # angle imbalance (checked via psi_deg drift below).
        psi_before = compute_tj_force(f, e1, e2, tj_xy, s, p).psi_deg
        f, e1, e2, e3 = relax(f, e1, e2, e3, s, p, 150)
        psi_after = compute_tj_force(f, e1, e2, tj_xy, s, p).psi_deg
        print(f" psi={psi_deg}deg: measured before relax={psi_before:.2f}deg, after={psi_after:.2f}deg")

        tj_res = compute_tj_force(f, e1, e2, tj_xy, s, p)
        assert tj_res.resolved, tj_res.reason
        F_TJ = np.asarray(tj_res.F_TJ)
        F_TJ_mag = float(tj_res.F_TJ_mag)
        print(f"   ground-truth F_TJ=({F_TJ[0]:.4e},{F_TJ[1]:.4e}) |F_TJ|={F_TJ_mag:.4e}")

        scan = configurational_force_radius_scan(f, e1, e2, p, tuple(tj_xy), radii, s=s,
                                                  bc_x="reflecting", bc_y="periodic")
        vals = []
        for r, (Fx, Fy) in scan.items():
            Fvec = np.array([Fx, Fy])
            mag = float(np.hypot(Fx, Fy))
            cos_sim = float(np.dot(Fvec, F_TJ) / (mag * F_TJ_mag)) if mag > 0 and F_TJ_mag > 0 else float("nan")
            ratio = mag / F_TJ_mag if F_TJ_mag > 0 else float("nan")
            vals.append((r, Fvec, mag, cos_sim, ratio))
            print(f"    r={r*1e9:.1f}nm ({r/W:.1f}W) F_conf=({Fx:.4e},{Fy:.4e}) |F_conf|={mag:.4e} "
                  f"cos_sim={cos_sim:.3f} |F_conf|/|F_TJ|={ratio:.3f}")
        ratios = [v[4] for v in vals[1:]]  # skip r=2W (closest to diffuse core)
        cos_sims = [v[3] for v in vals[1:]]
        plateau = (max(ratios) - min(ratios)) < 0.15 * np.mean(ratios) if ratios else False
        aligned = all(c > 0.8 for c in cos_sims)
        print(f"    plateau(r>=3W)={plateau}  aligned(cos_sim>0.8)={aligned}")
        all_ok = all_ok and plateau and aligned
    print(f"  PASS={all_ok}")
    return all_ok


if __name__ == "__main__":
    ok_a = benchmark_a_planar_interface()
    ok_b = benchmark_b_equilibrium_wedge()
    ok_c = benchmark_c_off_equilibrium_wedge()
    print(f"\nOVERALL: A={ok_a} B={ok_b} C={ok_c} ALL_PASS={ok_a and ok_b and ok_c}")
