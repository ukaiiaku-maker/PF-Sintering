"""Milestone 15F Section 21: re-run the M15E local configurational-force
benchmarks with a BC-CONSISTENT relaxation.

FINDING (before any code change to configurational_force.py itself):
Milestone 15E's `m15e_conf_force_benchmarks.py::relax()` used the OLDER
`model.evolve_f`/`model.evolve_eta` (model.py's hardcoded periodic-X/
one-sided-Y convention) to pre-relax synthetic wedge profiles, but then
evaluated the Eshelby tensor / contour force with `bc_x="reflecting",
bc_y="periodic"` (the M15 series' own, physically-correct convention for
this geometry). Numerically confirmed (this milestone): after 1500 such
relaxation steps, deep single-grain "bulk" points near the x=0 boundary
show e1 stuck at ~0.997 (not 1.0) and |div(Eshelby tensor)| ~1e7-1e8x
the natural scale gamma_s/W right at that boundary -- a genuine BC
mismatch artifact, not a physical defect -- that could plausibly
propagate inward over 1500 diffusive steps and contaminate the whole
relaxed field, independent of any "grand potential" question. This
script re-relaxes with the ACTUAL M15-series production dynamics
(mu_isotropic + variational_surface_diffusion_step +
constrained_tangent_cone_eta_update, bc_x=reflecting/bc_y=periodic
throughout, matching the Eshelby-tensor evaluation exactly) and repeats
Milestone 15E's benchmarks B (equilibrium wedge) and C (off-equilibrium
wedge vs tj_force ground truth) to see whether the sign-tracking failure
found there was actually this BC-mismatch artifact rather than a defect
in the Eshelby-tensor derivation itself.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.ch_exact_energy import mu_isotropic  # noqa: E402
from pf_sintering.configurational_force import configurational_force_radius_scan  # noqa: E402
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update  # noqa: E402
from pf_sintering.model import ModelConfig, Sink, build_params  # noqa: E402
from pf_sintering.surface_transport import m_s_ref, variational_surface_diffusion_step  # noqa: E402
from pf_sintering.tj_force import compute_tj_force  # noqa: E402
from tests.test_tj_force import _gamma_gb_for_equilibrium, build_synthetic_wedge  # noqa: E402

BC_X, BC_Y = "reflecting", "periodic"


def _wedge_params(nx=320, ny=360, dx_nm=2.0):
    return build_params(ModelConfig(
        preset="dev", nx=nx, ny=ny, dx=dx_nm * 1e-9, r2=40e-9, t_total=1e-6, use_aniso_surface=False,
    ))


def relax_bc_consistent(f, e1, e2, e3, s, p, n_steps):
    """Same physical dynamics the M15-series production trajectories use
    (mu_isotropic + variational_surface_diffusion_step +
    constrained_tangent_cone_eta_update), all under bc_x=reflecting,
    bc_y=periodic -- the SAME convention configurational_force.py's
    Eshelby-tensor evaluation uses, unlike M15E's relax()."""
    M_s = m_s_ref(p.M_f, p.interface_width)
    dt = p.dt
    for _ in range(n_steps):
        mu = mu_isotropic(f, e1, e2, e3, s, p)
        f, _ = variational_surface_diffusion_step(f, mu, p.dx, dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
        e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p, dt=dt, use_eta3=False,
                                                              bc_x=BC_X, bc_y=BC_Y)
    return f, e1, e2, e3


def benchmark_b_equilibrium_wedge():
    p = _wedge_params()
    s = Sink(threshold=math.inf)
    psi_eq = 120.0
    p.gamma_gb_ref = _gamma_gb_for_equilibrium(psi_eq, p.gamma_s)
    f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, psi_eq)

    tj_before = compute_tj_force(f, e1, e2, tj_xy, s, p)
    f, e1, e2, e3 = relax_bc_consistent(f, e1, e2, e3, s, p, 1500)
    tj_after = compute_tj_force(f, e1, e2, tj_xy, s, p)
    print(f"Benchmark B setup: psi before relax={tj_before.psi_deg:.2f}deg, after={tj_after.psi_deg:.2f}deg "
          f"(prescribed {psi_eq}deg); ground-truth |F_TJ|/gamma_s after relax={tj_after.F_TJ_mag/p.gamma_s:.4f}")

    W = p.interface_width
    radii = [k * W for k in (2, 3, 4, 5, 6)]
    scan = configurational_force_radius_scan(f, e1, e2, p, tuple(tj_xy), radii, s=s, bc_x=BC_X, bc_y=BC_Y)
    print("Benchmark B (BC-consistent relaxed equilibrium wedge, psi=120deg, must vanish):")
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

    print("Benchmark C (off-equilibrium wedge vs known F_TJ, BC-consistent relaxation):")
    all_ok = True
    for psi_deg in (100.0, 140.0):
        f, e1, e2, e3, tj_xy, *_ = build_synthetic_wedge(p, psi_deg)
        psi_before = compute_tj_force(f, e1, e2, tj_xy, s, p).psi_deg
        f, e1, e2, e3 = relax_bc_consistent(f, e1, e2, e3, s, p, 150)
        psi_after = compute_tj_force(f, e1, e2, tj_xy, s, p).psi_deg
        print(f" psi={psi_deg}deg: measured before relax={psi_before:.2f}deg, after={psi_after:.2f}deg")

        tj_res = compute_tj_force(f, e1, e2, tj_xy, s, p)
        assert tj_res.resolved, tj_res.reason
        F_TJ = np.asarray(tj_res.F_TJ)
        F_TJ_mag = float(tj_res.F_TJ_mag)
        print(f"   ground-truth F_TJ=({F_TJ[0]:.4e},{F_TJ[1]:.4e}) |F_TJ|={F_TJ_mag:.4e}")

        scan = configurational_force_radius_scan(f, e1, e2, p, tuple(tj_xy), radii, s=s, bc_x=BC_X, bc_y=BC_Y)
        vals = []
        for r, (Fx, Fy) in scan.items():
            Fvec = np.array([Fx, Fy])
            mag = float(np.hypot(Fx, Fy))
            cos_sim = float(np.dot(Fvec, F_TJ) / (mag * F_TJ_mag)) if mag > 0 and F_TJ_mag > 0 else float("nan")
            ratio = mag / F_TJ_mag if F_TJ_mag > 0 else float("nan")
            vals.append((r, Fvec, mag, cos_sim, ratio))
            print(f"    r={r*1e9:.1f}nm ({r/W:.1f}W) F_conf=({Fx:.4e},{Fy:.4e}) |F_conf|={mag:.4e} "
                  f"cos_sim={cos_sim:.3f} |F_conf|/|F_TJ|={ratio:.3f}")
        ratios = [v[4] for v in vals[1:]]
        cos_sims = [v[3] for v in vals[1:]]
        plateau = (max(ratios) - min(ratios)) < 0.15 * np.mean(ratios) if ratios else False
        aligned = all(c > 0.8 for c in cos_sims)
        print(f"    plateau(r>=3W)={plateau}  aligned(cos_sim>0.8)={aligned}")
        all_ok = all_ok and plateau and aligned
    print(f"  PASS={all_ok}")
    return all_ok


if __name__ == "__main__":
    ok_b = benchmark_b_equilibrium_wedge()
    ok_c = benchmark_c_off_equilibrium_wedge()
    print(f"\nOVERALL (BC-consistent relaxation): B={ok_b} C={ok_c} ALL_PASS={ok_b and ok_c}")
