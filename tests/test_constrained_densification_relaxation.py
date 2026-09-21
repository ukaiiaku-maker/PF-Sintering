from types import SimpleNamespace

import numpy as np

from pf_sintering.axisym import r_centers_faces
from pf_sintering.constrained_densification_relaxation import (
    constraint_values,
    multigrain_energy,
    relax_constrained,
    restore_constraints,
)


def _case():
    nz, nr = 28, 16
    dr = dz = 0.5e-9
    z = (np.arange(nz)-(nz-1)/2)*dz
    rc, rf = r_centers_faces(nr, dr)
    radius = 3.0e-9
    f = 0.5*(1+np.tanh((radius-rc[None, :])/1.0e-9))
    f = np.broadcast_to(f, (nz, nr)).copy()
    right = 0.5*(1+np.tanh(z[:, None]/1.0e-9))
    right = np.broadcast_to(right, f.shape).copy()
    ownership = np.stack([1-right, right])
    g = dict(z=z, r_c=rc, r_f=rf, dr=dr, dz=dz,
             config=SimpleNamespace(width=2.0e-9, outer_radius=5.0e-9))
    return f, ownership, g


def test_constraint_restoration_and_relaxation_preserve_manifold():
    f, ownership, g = _case()
    target = constraint_values(
        f, ownership, g["z"], g["r_c"], g["config"].outer_radius)
    perturbed = f.copy()
    perturbation = 1e-4*np.exp(-((g["z"][:, None]/2e-9)**2
                                  +(g["r_c"][None, :]/2e-9)**2))
    perturbed += perturbation
    restored, info = restore_constraints(perturbed, ownership, g, target)
    assert info["absolute_constraint_residual"] <= 1e-10
    before = multigrain_energy(restored, ownership, g)
    result = relax_constrained(
        restored, ownership, g, target, max_iterations=5, record_every=1)
    after = multigrain_energy(result.f, ownership, g)
    actual = constraint_values(
        result.f, ownership, g["z"], g["r_c"], g["config"].outer_radius)
    assert after <= before
    np.testing.assert_allclose(actual, target, rtol=2e-12, atol=1e-10)
    assert result.f.min() >= -2e-14
    assert result.f.max() <= 1+2e-14
