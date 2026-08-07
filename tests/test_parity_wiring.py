from pf_sintering.model import evolve_f, ostwald_substrate
from pf_sintering.parity_kernels import (
    evolve_f as parity_evolve_f,
    ostwald_substrate as parity_ostwald_substrate,
)


def test_package_uses_matlab_parity_kernels():
    assert evolve_f is parity_evolve_f
    assert ostwald_substrate is parity_ostwald_substrate
