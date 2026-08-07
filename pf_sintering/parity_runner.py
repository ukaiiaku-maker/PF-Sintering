"""Wire MATLAB-parity kernels into the existing v64 Python runner.

This module keeps the higher-level runner/checkpoint/diagnostic logic unchanged
while replacing low-level kernels where deterministic MATLAB translation
mismatches were identified.
"""

from . import model as _model
from . import runner as _runner
from .parity_kernels import evolve_f, ostwald_substrate

# Keep direct imports from pf_sintering.model and the runner's module globals
# on the same corrected implementation.
_model.evolve_f = evolve_f
_model.ostwald_substrate = ostwald_substrate
_runner.evolve_f = evolve_f
_runner.ostwald_substrate = ostwald_substrate

SinteringModel = _runner.SinteringModel

__all__ = ["SinteringModel"]
