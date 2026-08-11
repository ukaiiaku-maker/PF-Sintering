"""Milestone 16A Sections 7-8: main single-crystal PR growth-rate
campaign. lambda/R0 in {4, 5.5, 2*pi, 7, 8}, R0=40nm, W=20nm, dx=2.5nm,
long horizon for a clean exponential fit.
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m16a_pr_benchmark import run_case  # noqa: E402

RATIOS = {"4": 4.0, "5.5": 5.5, "2pi": 2 * math.pi, "7": 7.0, "8": 8.0}

if __name__ == "__main__":
    for label, ratio in RATIOS.items():
        run_case(f"pr_main_R0-40_lamR0-{label}", 40.0, ratio, dx_nm=2.5, W_nm=20.0,
                  n_steps_total=200000, n_sample=60)
