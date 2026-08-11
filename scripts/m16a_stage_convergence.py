"""Milestone 16A Section 10: dx/W convergence of the PR threshold.
Repeats a representative subset of wavelengths (near/below/above 2*pi)
at a finer dx and at a smaller W, checking that the SIGN of omega near
the threshold does not depend on mobility tuning and that the apparent
neutral-wavelength shift (observed near lambda/R0=2*pi at the baseline
dx=2.5nm/W=20nm) narrows as the interface sharpens."""
from __future__ import annotations

import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m16a_pr_benchmark import run_case  # noqa: E402

RATIOS = {"5.5": 5.5, "2pi": 2 * math.pi, "7": 7.0}

if __name__ == "__main__":
    # finer dx, same W -- dx=1.25nm needs ~16x smaller dt (4th-order
    # stability) and 4x more cells, so this is bounded to a shorter
    # step budget than the dx=2.5nm main campaign; sufficient to confirm
    # the SIGN/ordering of omega, not to re-fit the full dispersion curve.
    for label, ratio in RATIOS.items():
        run_case(f"conv_dx1.25_R0-40_lamR0-{label}", 40.0, ratio, dx_nm=1.25, W_nm=20.0,
                  n_steps_total=60000, n_sample=30)
    # baseline dx, smaller W (same interface_cells/W ratio requirement:
    # dx=1.25nm/W=10nm keeps W/dx=8, matching the baseline's W/dx=8)
    for label, ratio in RATIOS.items():
        run_case(f"conv_W10_R0-40_lamR0-{label}", 40.0, ratio, dx_nm=1.25, W_nm=10.0,
                  n_steps_total=60000, n_sample=30)
