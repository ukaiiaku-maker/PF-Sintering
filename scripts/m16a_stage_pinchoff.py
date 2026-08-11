"""Milestone 16A Section 9: nonlinear pinch-off. Starts with a larger
(already-nonlinear-scale) perturbation at an unstable wavelength to
reach substantial neck thinning within a bounded step budget (Sections
7-8 already established the LINEAR-regime prediction; this section is
specifically about the nonlinear endpoint)."""
from __future__ import annotations

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m16a_pr_benchmark import run_case  # noqa: E402

if __name__ == "__main__":
    run_case("pinchoff_R0-40_lamR0-8_eps0.3", 40.0, 8.0, dx_nm=2.5, W_nm=20.0,
              eps0_frac=0.3, n_steps_total=400000, n_sample=80)
