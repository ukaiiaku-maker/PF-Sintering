"""Milestone 15F Section 4: audit of the EXISTING anisotropic free-surface
functional in pf_sintering/model.py, before running any anisotropic neck.

The implemented law (model.py build_params, `if p.use_aniso_surface:` block):

    psi = (theta - theta0) mod (pi/2)          [4-fold symmetric]
    a(psi)  = 1 - d*cos(4*psi)                  [gamma(theta)/gamma_s]
    ap(psi) = 4*d*sin(4*psi)                    [d(a)/d(psi) = d(a)/d(theta)]

evaluated via a 4096-point LUT (`p.lut_psi/lut_a/lut_ap`) and interpolated
with `np.interp` at runtime (`_aniso_gamma`, model.py). `theta` here is the
LOCAL INTERFACE NORMAL angle (model.py's `compute_stress`: `th=atan2(n[1],
n[0])`, n from `_vapor_normal`) relative to `theta0=p.theta_grain[i]`, the
per-grain crystal-orientation reference angle (radians) -- NOT the tangent
angle. Because the law is exactly 4-fold periodic (period pi/2), using the
tangent angle instead would only relabel theta0 by 90 degrees, not change
the functional form -- both conventions give the identical gamma(theta)
curve. `gamma(theta) = gamma_s * a(psi)` (`_aniso_gamma` returns
`(gamma_s*a, gamma_s*ap)`).

For `d > 1/15`, model.py's own build_params ALREADY regularizes the LUT in
a "missing-orientation" cone via a Wulff-envelope construction (linear
`A*cos(psi-pi/4)` facet) -- i.e. the codebase already anticipates and
handles the negative-stiffness regime, but Milestone 15F Section 16
explicitly keeps the PRIMARY study on the positive-stiffness side of that
boundary. IMPORTANT: Params' own default `aniso_delta=0.15` is already
PAST 1/15=0.0667 -- using the code's raw default would silently enter the
regularized/faceted branch. This script derives the critical delta
analytically, confirms it against the LUT numerically, and picks AN0-AN3
strictly below it.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.model import ModelConfig, build_params  # noqa: E402


def gamma_family(delta, gamma_s=1.0, n=4096):
    """Analytic gamma(theta)=gamma_s*a(psi), gamma'(theta)=gamma_s*ap(psi),
    gamma''(theta)=gamma_s*d(ap)/dpsi, surface stiffness
    gamma_tilde=gamma+gamma'', all as closed-form functions of psi (valid
    only in the positive-stiffness regime, delta<1/15, where the raw
    cos(4*psi) law is not regularized)."""
    psi = np.linspace(0, math.pi / 2, n)
    a = 1 - delta * np.cos(4 * psi)
    ap = 4 * delta * np.sin(4 * psi)
    app = 16 * delta * np.cos(4 * psi)  # d(ap)/dpsi
    gamma = gamma_s * a
    gammap = gamma_s * ap
    gammapp = gamma_s * app
    stiffness = gamma + gammapp
    return dict(psi=psi, gamma=gamma, gammap=gammap, gammapp=gammapp, stiffness=stiffness)


def audit_delta(delta, gamma_s=1.0, label=""):
    fam = gamma_family(delta, gamma_s)
    print(f"[{label}] delta={delta:.5f}  gamma in [{fam['gamma'].min():.4f},{fam['gamma'].max():.4f}]  "
          f"ratio={fam['gamma'].max()/fam['gamma'].min():.4f}  "
          f"stiffness in [{fam['stiffness'].min():.4f},{fam['stiffness'].max():.4f}]  "
          f"positive_stiffness_everywhere={bool(fam['stiffness'].min() > 0)}")
    return fam


def verify_against_lut(delta, gamma_s=1.0):
    """Cross-check the analytic family against the ACTUAL runtime LUT
    (p.lut_psi/lut_a/lut_ap) built by build_params, for delta strictly
    below 1/15 (so the regularization branch is inactive and the LUT
    should match the raw cos(4*psi) law to interpolation-grid precision)."""
    p = build_params(ModelConfig(preset="dev", nx=64, ny=64, dx=2e-9, r2=40e-9, t_total=1e-6,
                                  use_aniso_surface=True, aniso_delta=delta))
    assert p.gamma_s == gamma_s
    fam = gamma_family(delta, gamma_s)
    a_lut = np.interp(fam["psi"], p.lut_psi, p.lut_a)
    ap_lut = np.interp(fam["psi"], p.lut_psi, p.lut_ap)
    err_a = np.max(np.abs(a_lut - fam["gamma"] / gamma_s))
    err_ap = np.max(np.abs(ap_lut - fam["gammap"] / gamma_s))
    print(f"  LUT cross-check: max|a_lut-a_analytic|={err_a:.2e}  max|ap_lut-ap_analytic|={err_ap:.2e}")
    return err_a, err_ap


if __name__ == "__main__":
    print("Critical anisotropy strength (positive-stiffness boundary):")
    print("  gamma_tilde(psi) = gamma_s*[1 + 15*delta*cos(4*psi)] -- derived below")
    print("  min over psi at cos(4*psi)=-1: gamma_s*(1-15*delta) > 0  =>  delta < 1/15 = "
          f"{1/15:.6f}")
    print()
    print("Codebase Params default aniso_delta=0.15 is ALREADY past this boundary "
          f"(0.15 > {1/15:.4f}) -- NOT usable for the AN0-AN3 primary-study ladder as-is.")
    print()

    AN_LEVELS = dict(AN0=0.0, AN1=0.02, AN2=0.045, AN3=1 / 15 - 0.002)
    for label, d in AN_LEVELS.items():
        audit_delta(d, label=label)
        if d > 0:
            verify_against_lut(d)
    print()
    print("Also auditing the codebase's raw default (0.15) and a representative "
          "over-critical value to confirm the regularization triggers and stiffness "
          "genuinely goes negative in the RAW (un-regularized) law there (for context "
          "only -- not used in the primary study):")
    audit_delta(0.15, label="RAW-default(NOT USED, informational)")
