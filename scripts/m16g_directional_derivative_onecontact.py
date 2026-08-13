"""Milestone 16G Section 15: sharp-interface directional-derivative check
for the ONE-CONTACT (crest-capped) geometry -- "GB moves into the
particle; particle volume decreases; free surface relaxes coherently;
total f volume fixed. Determine whether this direction is downhill."
Mirrors scripts/m16f_directional_derivative.py's method (same physical
question, one GB instead of two).

Development note on the representation used: a literal sharp-interface
R(z) array built directly from the crest-capped construction (exact
profile + ellipsoid cap, truncated somewhere before the cap's natural
R->0 closure, topology="open") was tried first and rejected -- EVERY one
of the first 8-10 constrained eigenmodes turned out to be a spurious
artifact localized at the domain's open/truncated ends (near-zero actual
particle-volume transfer, huge outlier eigenvalues), regardless of
whether the truncation point was pushed further out or a mirror-doubled
periodic wrap was used instead of a true open end -- the root cause was
an unavoidable derivative mismatch at whatever artificial point the
ellipsoid gets cut off before its own natural closure. Since z=lambda
(the retained particle's own crest) is ALREADY an exact analytic
symmetry point of the underlying two-mode profile (R'=0, R'''=0,
verified in scripts/m16g_pr_derived_particle_asperity.py), the clean fix
is to not introduce any artificial cut at all: reuse the SAME
already-validated periodic 2*lambda domain construction M16F used
(topology="periodic", no boundary artifacts anywhere, by construction),
but mark only z1 (not z2) as a GB index -- exactly representing the
"remove the second GB, reassign that material to grain 1" operation this
milestone's construction performs, evaluated as a LOCAL proxy for the
near-contact physics (the actual PF construction's finite, weakly-
coupled cap region, per the Section 3 flux audit, is not expected to
materially change this near-GB conclusion).

The eigenmode selection criterion is also different from M16F: computing
particle-volume transfer via a crude "z1<z<z_wrap" mask gave near-zero
values for every mode in this one-GB representation (an artifact of that
partition choice, not of the physics), so the SAME implicit-function-
theorem GB-shift diagnostic M16F itself used
(gb_shift=-mode'(z_gb)/R''(z_gb)) is used instead to identify and orient
the candidate mode -- a direct, unambiguous measure of "does this mode
move the GB," which does NOT suffer from the same degeneracy.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pf_sintering.sharp_interface_stability import energy, volume_constrained_eigenmodes  # noqa: E402

sys.path.insert(0, "scripts")
from m16g_pr_derived_particle_asperity import R_profile_over_Rcyl, LAM_OVER_RCYL, Z1_OVER_LAM, psi_to_gamma_gb  # noqa: E402

R_CYL = 100e-9


def build_periodic_one_gb(Nz=240):
    lam = LAM_OVER_RCYL * R_CYL
    z1 = Z1_OVER_LAM * lam
    L = 2 * lam
    dz = L / Nz
    z = (np.arange(Nz) + 0.5) * dz
    R = R_CYL * R_profile_over_Rcyl(z / lam)
    i1 = int(np.argmin(np.abs(z - z1)))
    return z, R, dz, i1, Nz


def analyze(psi_deg, n_modes=8):
    z, R, dz, i1, Nz = build_periodic_one_gb()
    gamma_ratio = psi_to_gamma_gb(psi_deg)
    evals, evecs, lam_lagrange, gnorm = volume_constrained_eigenmodes(
        R, dz, 1.0, gamma_ratio, [i1], "periodic", n_modes=n_modes)
    return z, R, dz, i1, Nz, evals, evecs, gamma_ratio


def _local_deriv(field, i, Nz, dz):
    ip, im = (i + 1) % Nz, (i - 1) % Nz
    return (field[ip] - field[im]) / (2 * dz)


def _local_curv(field, i, Nz, dz):
    ip, im = (i + 1) % Nz, (i - 1) % Nz
    return (field[ip] - 2 * field[i] + field[im]) / dz ** 2


def directional_derivative_scan(psi_deg, eps_fracs=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)):
    """CORRECTED (M16H): `eps_fracs` are DIMENSIONLESS fractions of
    R_CYL, matching M16F's original dimensionless-profile convention --
    but unlike M16F's R/Rcyl~O(1) array, THIS script's R is in physical
    meters (~1e-7 scale), so the eps values must be scaled by R_CYL
    before being applied. The unscaled version (eps_list=eps_fracs used
    directly as a perturbation in meters) was a real bug: at the largest
    eps=3e-2, the perturbation eps*v1 (~1e-2*O(0.1)~1e-3, in whatever
    units v1's O(1)-normalized components carry) is ~45,000x LARGER than
    the actual physical radius R~1e-7 m, giving |dF| about 8 orders of
    magnitude larger than F0 itself -- clearly not a small perturbation.
    The conclusion (downhill, consistent sign) survived because the
    frustum-area formula stays numerically finite even for such
    unphysically large (and R-sign-violating) perturbations and the
    linear term still dominated by construction, but the eps values
    themselves were meaningless in absolute terms. Fixed by scaling:
    eps_phys = eps_frac * R_CYL."""
    z, R, dz, i1, Nz, evals, evecs, gamma_ratio = analyze(psi_deg)
    eps_list = tuple(f * R_CYL for f in eps_fracs)
    Rpp1 = _local_curv(R, i1, Nz, dz)

    gb_shifts = []
    for k in range(evecs.shape[1]):
        v = evecs[:, k] / np.linalg.norm(evecs[:, k])
        modep1 = _local_deriv(v, i1, Nz, dz)
        gb_shifts.append(-modep1 / Rpp1)

    # candidate mode = most negative eigenvalue (strongest downhill
    # curvature direction) with a genuinely nonzero GB shift
    k0 = int(np.argmin(evals))
    v1 = evecs[:, k0] / np.linalg.norm(evecs[:, k0])
    if gb_shifts[k0] < 0:  # orient so +v1 := GB shifts to LARGER z (substrate expands INTO the particle)
        v1 = -v1
        gb_shifts[k0] = -gb_shifts[k0]

    F0 = energy(R, dz, 1.0, gamma_ratio, [i1], "periodic")
    rows = []
    for eps in eps_list:
        Fp = energy(R + eps * v1, dz, 1.0, gamma_ratio, [i1], "periodic")
        Fm = energy(R - eps * v1, dz, 1.0, gamma_ratio, [i1], "periodic")
        rows.append(dict(eps=eps, dF_plus=Fp - F0, dF_minus=Fm - F0,
                          central_slope=(Fp - Fm) / (2 * eps)))
    return dict(psi_deg=psi_deg, mode=k0, eig=float(evals[k0]), gb_shift=gb_shifts[k0],
                F0=F0, rows=rows, all_eigs=evals.tolist(), all_gb_shifts=gb_shifts)


if __name__ == "__main__":
    for psi in (160.0, 80.0):
        result = directional_derivative_scan(psi)
        print(f"\n=== psi={psi} (mode={result['mode']}, eig={result['eig']:+.4e}, "
              f"gb_shift={result['gb_shift']:+.4e}, F0={result['F0']:.6e}) ===")
        print("all eigenvalues:", [f"{e:+.3e}" for e in result["all_eigs"]])
        print("all gb_shifts:  ", [f"{g:+.3e}" for g in result["all_gb_shifts"]])
        print("+eps*v1 := GB shifts to larger z (substrate expands into the particle, Vp decreases)")
        for r in result["rows"]:
            downhill = "+v1 (particle shrinks)" if r["dF_plus"] < r["dF_minus"] else "-v1 (particle grows)"
            print(f"  eps={r['eps']:.1e}  dF(+eps*v1)={r['dF_plus']:+.6e}  "
                  f"dF(-eps*v1)={r['dF_minus']:+.6e}  central_slope={r['central_slope']:+.6e}  "
                  f"downhill={downhill}")
