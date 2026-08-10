"""Milestone 15F Section 2: audit of the width-refinement physical
contract. Holds gamma_s, gamma_GB, physical M_GB [m^4/(J*s)], and
surface_mobility_scale fixed while varying W (interface_width) and dx,
and prints/verifies that the PHYSICAL invariants specified in Section 2
are actually held constant by the EXISTING build_params machinery:

    k_f    = 3*gamma_s*W          (varies with W by design)
    W_f    = 12*gamma_s/W         (varies with W by design)
    M_f    = M_f_base*scale, M_f_base=(20nm)^4/(tau_target*k_f)
             => M_f*k_f = (20nm)^4/tau_target*scale, W-INDEPENDENT
    M_s_int = m_s_ref(M_f,W) = M_f*W*SECH8_INTEGRAL
             = (20nm)^4*scale*SECH8_INTEGRAL/(tau_target*3*gamma_s), W-INDEPENDENT
             (surface_transport.m_s_ref -- the "existing integrated
             surface-mobility relation" Section 2 refers to)

    k_eta  = 4*gamma_GB*W/pi^2    (varies with W by design)
    Wc     = 4*gamma_GB/W         (varies with W by design)
    M_eta  = pi^2*M_GB/(4*W)      (varies with W by design, so that the
             PHYSICAL M_GB is what's actually held fixed)

No code changes are needed for this contract -- `build_config`'s existing
`interface_width_override=W_nm*1e-9` + `gb_mobility_m4_J_s=M_GB` +
`surface_mobility_scale` already implement it (confirmed empirically
below). This script is a verification/audit, not a new mechanism.
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, ".")
from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients, m_gb_from_m_eta  # noqa: E402
from pf_sintering.model import ModelConfig, build_params  # noqa: E402
from pf_sintering.surface_transport import m_s_ref  # noqa: E402

GAMMA_S = 1.0
GAMMA_GB = 1.4
M_GB_PHYSICAL = 3.4584297349917956e-15  # M15E's M_GB_scale=100 * M_GB_REF, reused for continuity
SURFACE_MOBILITY_SCALE = 0.3


def audit(W_nm, dx_nm):
    p = build_params(ModelConfig(
        preset="dev", nx=64, ny=64, dx=dx_nm * 1e-9, r2=40e-9, t_total=1e-6,
        interface_width_override=W_nm * 1e-9, gamma_gb_override=GAMMA_GB,
        gb_mobility_m4_J_s=M_GB_PHYSICAL, surface_mobility_scale=SURFACE_MOBILITY_SCALE,
        use_aniso_surface=False,
    ))
    assert p.gamma_s == GAMMA_S
    Ms_int = m_s_ref(p.M_f, p.interface_width)
    Mf_kf = p.M_f * p.k_f
    M_GB_recovered = m_gb_from_m_eta(p.M_eta, p.interface_width)
    print(f"W={W_nm:5.2f}nm dx={dx_nm:5.3f}nm  k_f={p.k_f:.4e} W_f={p.W_f:.4e} M_f={p.M_f:.4e}  "
          f"M_f*k_f={Mf_kf:.6e}  M_s_int={Ms_int:.6e}  |  "
          f"k_eta={p.k_eta:.4e} Wc={p.W_cpl_f:.4e} M_eta={p.M_eta:.4e}  "
          f"M_GB_recovered={M_GB_recovered:.6e}")
    return dict(Mf_kf=Mf_kf, Ms_int=Ms_int, M_GB_recovered=M_GB_recovered)


if __name__ == "__main__":
    ladder = [(20.0, 2.5), (10.0, 1.25), (7.5, 1.25), (5.0, 0.625)]
    results = [audit(W, dx) for W, dx in ladder]

    Mf_kf_vals = [r["Mf_kf"] for r in results]
    Ms_int_vals = [r["Ms_int"] for r in results]
    MGB_vals = [r["M_GB_recovered"] for r in results]

    def relvar(vals):
        return (max(vals) - min(vals)) / abs(vals[0])

    print()
    print(f"M_f*k_f relative spread across W ladder: {relvar(Mf_kf_vals):.2e}  (expect ~0, i.e. W-invariant)")
    print(f"M_s_int  relative spread across W ladder: {relvar(Ms_int_vals):.2e}  (expect ~0)")
    print(f"M_GB_recovered relative spread across W ladder: {relvar(MGB_vals):.2e}  "
          f"(expect ~0, input was {M_GB_PHYSICAL:.6e})")
    assert relvar(Mf_kf_vals) < 1e-9, "M_f*k_f (surface physical rate) is NOT W-invariant"
    assert relvar(Ms_int_vals) < 1e-9, "M_s_int (integrated surface mobility) is NOT W-invariant"
    assert relvar(MGB_vals) < 1e-9, "M_GB (physical GB mobility) is NOT W-invariant"
    print()
    print("PASS: all physical invariants (M_f*k_f / M_s_int / M_GB) confirmed W-independent "
          "using the EXISTING build_config(interface_width_override=..., gb_mobility_m4_J_s=..., "
          "surface_mobility_scale=...) contract -- no new code needed for width invariance itself.")
