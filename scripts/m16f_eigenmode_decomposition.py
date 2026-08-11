"""Milestone 16F Section 6: full Hessian eigenmode decomposition of the
exact e1bar=e2bar=0.4 two-mode/two-GB state, for psi=160 and psi=80.

For each of the first 8 eigenmodes of the constrained sharp-interface
Hessian (pf_sintering.sharp_interface_stability, applied directly to
the exact, un-relaxed initial profile -- this milestone's Section 6
wants the eigenstructure of the ACTUAL published starting point, not a
separately relaxed proxy), quantifies:

  - GB1/GB2 position displacement (via the local quadratic
    implicit-function-theorem estimate d(z_min) = -mode'(z_gb)/R''(z_gb))
  - GB1/GB2 trough RADIUS change (mode's own value at the GB index)
  - delta(V_grain1), delta(V_grain2) (2*pi*integral(R*mode)dz over each
    grain's z-range)
  - overlap (inner product) with cos(2*pi*z/lambda) and cos(pi*z/lambda)
  - symmetric vs antisymmetric GB motion (sign/magnitude comparison of
    the two GB displacements)

and classifies each mode as GRAIN-COARSENING/GB-MIGRATION (dominated by
opposite-sign volume changes between the two grains, i.e. mass moving
from one grain to the other with comparatively little net free-surface
area change), P-R/FREE-SURFACE (dominated by projection onto the
long-wavelength free-surface modes with comparatively little
grain-volume asymmetry), or MIXED.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.hussein_eq4_reference import R0_over_Rcyl, R_over_Rcyl, psi_to_gamma_ratio, z1_over_lambda  # noqa: E402
from pf_sintering.sharp_interface_stability import volume_constrained_eigenmodes  # noqa: E402

EPS1 = EPS2 = 0.4


def build_grid(Nz=240, lam_over_Rcyl=1.14 * math.pi):
    lam = lam_over_Rcyl
    L = 2 * lam
    dz = L / Nz
    z = (np.arange(Nz) + 0.5) * dz
    R = R_over_Rcyl(z / lam, EPS1, EPS2)  # Rcyl=1
    z1 = z1_over_lambda(EPS1, EPS2) * lam
    z2 = 2 * lam - z1_over_lambda(EPS1, EPS2) * lam
    i1 = int(np.argmin(np.abs(z - z1)))
    i2 = int(np.argmin(np.abs(z - z2)))
    return z, R, dz, lam, L, i1, i2


def analyze_modes(psi_deg, n_modes=8):
    z, R, dz, lam, L, i1, i2 = build_grid()
    Nz = len(z)
    gamma_ratio = psi_to_gamma_ratio(psi_deg)
    evals, evecs, lam_lagrange, gnorm = volume_constrained_eigenmodes(
        R, dz, 1.0, gamma_ratio, [i1, i2], "periodic", n_modes=n_modes)

    cos1 = np.cos(2 * math.pi * z / lam)
    cos2 = np.cos(math.pi * z / lam)
    cos1_hat = cos1 / np.linalg.norm(cos1)
    cos2_hat = cos2 / np.linalg.norm(cos2)

    # grain masks: grain1 (inner) is i1<index<i2 (matching z1<z<z2)
    idx = np.arange(Nz)
    inner_mask = (idx > i1) & (idx < i2)
    outer_mask = ~inner_mask

    def local_curvature(field, i):
        # second derivative via central difference (periodic-safe)
        ip = (i + 1) % Nz
        im = (i - 1) % Nz
        return (field[ip] - 2 * field[i] + field[im]) / dz ** 2

    def local_deriv(field, i):
        ip = (i + 1) % Nz
        im = (i - 1) % Nz
        return (field[ip] - field[im]) / (2 * dz)

    Rpp1 = local_curvature(R, i1)
    Rpp2 = local_curvature(R, i2)

    results = []
    for k in range(len(evals)):
        mode = evecs[:, k]
        mode = mode / np.linalg.norm(mode)  # unit-normalize for comparability
        gb1_radius_change = mode[i1]
        gb2_radius_change = mode[i2]
        modep1 = local_deriv(mode, i1)
        modep2 = local_deriv(mode, i2)
        gb1_shift = -modep1 / Rpp1 if abs(Rpp1) > 1e-12 else float("nan")
        gb2_shift = -modep2 / Rpp2 if abs(Rpp2) > 1e-12 else float("nan")

        dV1 = 2 * math.pi * float(np.sum(R[inner_mask] * mode[inner_mask])) * dz
        dV2 = 2 * math.pi * float(np.sum(R[outer_mask] * mode[outer_mask])) * dz

        overlap1 = float(np.dot(mode, cos1_hat))
        overlap2 = float(np.dot(mode, cos2_hat))

        same_sign = (gb1_shift > 0) == (gb2_shift > 0)
        motion_type = "SYMMETRIC (same-direction)" if same_sign else "ANTISYMMETRIC (seesaw)"

        # Classification heuristic, corrected: dV1~=-dV2 for EVERY mode
        # (a trivial consequence of the volume constraint applied to a
        # 2-way partition of the whole domain, not a signature specific
        # to genuine grain-transfer modes -- confirmed directly, an
        # earlier version of this classifier used |dV1-dV2|/(|dV1|+|dV2|)
        # and found it was identically ~1.000 for literally every mode,
        # uninformative by construction). The real discriminant is
        # whether the mode's SHAPE is spread over the whole domain
        # (projects strongly onto the smooth cos(2*pi*z/lambda) /
        # cos(pi*z/lambda) basis -- a genuine free-surface/P-R-type
        # mode) or LOCALIZED near the two GB troughs (small projection
        # onto both smooth cosines, large GB1/GB2 shift relative to the
        # mode's overall norm -- a genuine GB-migration/groove mode).
        surface_mode_strength = math.sqrt(overlap1 ** 2 + overlap2 ** 2)
        if surface_mode_strength > 0.5:
            cls = "P-R/FREE-SURFACE"
        elif surface_mode_strength < 0.15:
            cls = "GRAIN-COARSENING/GB-MIGRATION"
        else:
            cls = "MIXED"
        transfer_signal = surface_mode_strength

        results.append(dict(mode=k, eig=float(evals[k]), gb1_shift=float(gb1_shift), gb2_shift=float(gb2_shift),
                             gb1_radius_change=float(gb1_radius_change), gb2_radius_change=float(gb2_radius_change),
                             dV1=dV1, dV2=dV2, overlap_lambda_mode=overlap1, overlap_2lambda_mode=overlap2,
                             motion_type=motion_type, transfer_signal=float(transfer_signal), classification=cls))
    return results


if __name__ == "__main__":
    for psi_deg in (160.0, 80.0):
        print(f"\n=== psi={psi_deg} ===")
        results = analyze_modes(psi_deg)
        for r in results:
            print(f"mode {r['mode']}: eig={r['eig']:+.4e} class={r['classification']:28s} "
                  f"motion={r['motion_type']:24s} surf_strength={r['transfer_signal']:.3f} "
                  f"dV1={r['dV1']:+.4f} dV2={r['dV2']:+.4f} "
                  f"GB1_shift={r['gb1_shift']:+.4f} GB2_shift={r['gb2_shift']:+.4f} "
                  f"overlap(lam)={r['overlap_lambda_mode']:+.3f} overlap(2lam)={r['overlap_2lambda_mode']:+.3f}")
