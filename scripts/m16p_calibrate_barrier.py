"""M16P Section 6: barrier calibration -- solve A0 ONCE from the actual
deterministic sigma(t) trajectory so the integrated hazard reaches ln(2)
at the moment sigma first crosses the chosen target stress (the median
first-passage condition, same methodology M16K/M16L used).

    Lambda(sigma,T) = r0*(b/GS)^3*exp(-max(0,A0-sigma*V0)/(kB*T))
    H(t) = integral_0^t Lambda(sigma(t'),T) dt'

Solve for A0 such that H(t_target) = ln(2), where t_target is the first
time sigma(t) reaches the target stress in the deterministic trajectory.
"""
from __future__ import annotations

import csv
import math
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, ".")

B = 2.5e-10
V0_OVER_B3 = 12.5
V0 = V0_OVER_B3 * B ** 3
GS = 201.74e-9
R0 = 1e12
T = 1000.0
KB = 1.380649e-23


def integrated_hazard(t, sigma_MPa, A0_J):
    """Trapezoidal integral of Lambda(sigma(t),T) dt over the trajectory."""
    sigma_Pa = sigma_MPa * 1e6
    lam = np.array([R0 * (B / GS) ** 3 * math.exp(-max(0.0, A0_J - s * V0) / (KB * T)) for s in sigma_Pa])
    return float(np.trapezoid(lam, t))


def calibrate(t, sigma_MPa, target_MPa):
    idx = np.searchsorted(sigma_MPa, target_MPa)
    if idx >= len(t) or sigma_MPa[idx] < target_MPa:
        raise ValueError(f"target {target_MPa}MPa never reached in trajectory "
                          f"(max sigma={np.max(sigma_MPa):.2f}MPa) -- choose a lower target")
    t_target = t[idx]
    t_sub = t[: idx + 1]
    sigma_sub = sigma_MPa[: idx + 1]

    def resid(A0_eV):
        A0_J = A0_eV * 1.602176634e-19
        return integrated_hazard(t_sub, sigma_sub, A0_J) - math.log(2.0)

    # bracket: A0 too low -> H >> ln2; A0 too high -> H << ln2
    lo, hi = 0.01, 5.0
    if resid(lo) < 0:
        raise ValueError("even A0=0.01eV gives H<ln(2) at target -- target likely unreachable this fast")
    if resid(hi) > 0:
        raise ValueError("even A0=5.0eV gives H>ln(2) at target -- widen the search bracket")
    A0_eV = brentq(resid, lo, hi, xtol=1e-6)
    return A0_eV, t_target


def main(csv_path, target_MPa):
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["time"]) for r in rows])
    sigma = np.array([float(r["sigma_MPa"]) for r in rows])
    A0_eV, t_target = calibrate(t, sigma, target_MPa)
    print(f"target={target_MPa}MPa reached at t={t_target:.4f}")
    print(f"calibrated A0={A0_eV:.6f}eV (V0={V0_OVER_B3}*b^3, GS={GS*1e9:.2f}nm, r0={R0:.2e}, T={T}K)")
    A0_J = A0_eV * 1.602176634e-19
    H_check = integrated_hazard(t[: np.searchsorted(sigma, target_MPa) + 1],
                                 sigma[: np.searchsorted(sigma, target_MPa) + 1], A0_J)
    print(f"verification: H(t_target)={H_check:.6f} (target ln(2)={math.log(2):.6f})")
    return A0_eV, t_target


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-mpa", type=float, required=True)
    args = ap.parse_args()
    main(args.csv, args.target_mpa)
