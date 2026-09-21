from types import SimpleNamespace

import numpy as np

from pf_sintering.axisym import r_centers_faces
from pf_sintering.bicrystal_configurational_force import (
    axial_configurational_resultant, multigrain_variational_derivatives,
    ownership_configurational_force, plateau_average)
from pf_sintering.constrained_densification_relaxation import multigrain_energy


def test_uniform_zero_multiplier_state_has_zero_configurational_resultant():
    nz, nr = 12, 10
    dr = dz = 1e-9
    rc, rf = r_centers_faces(nr, dr)
    f = np.ones((nz, nr))
    ownership = np.stack([np.ones_like(f), np.zeros_like(f)])
    g = dict(z=np.arange(nz)*dz, r_c=rc, r_f=rf, dr=dr, dz=dz,
             config=SimpleNamespace(width=4e-9, outer_radius=10e-9))
    result = axial_configurational_resultant(f, ownership, g, np.zeros(2))
    np.testing.assert_allclose(result["resultant_N"], 0.0, atol=1e-24)
    row = plateau_average(result, 5.5e-9, 1e-9, side=-1,
                          lo_widths=2, hi_widths=5)
    assert abs(row["mean_N"]) <= 1e-24


def test_analytic_ownership_shape_force_matches_fixed_field_energy_derivative():
    nz, nr = 18, 12
    dr = dz = 0.5e-9
    z = (np.arange(nz)-(nz-1)/2)*dz
    rc, rf = r_centers_faces(nr, dr)
    f = np.broadcast_to(
        0.5*(1+np.tanh((3e-9-rc[None, :])/1e-9)), (nz, nr)).copy()
    g = dict(z=z, r_c=rc, r_f=rf, dr=dr, dz=dz,
             config=SimpleNamespace(width=2e-9, outer_radius=5e-9))

    def ownership(u):
        right = np.broadcast_to(
            0.5*(1+np.tanh((z[:, None]+u)/1e-9)), f.shape).copy()
        return np.stack([1-right, right])

    u0, du = 0.2e-9, 0.01e-9
    minus, center, plus = ownership(u0-du), ownership(u0), ownership(u0+du)
    force = ownership_configurational_force(
        f, center, minus, plus, g, du, np.zeros(2))
    finite_difference = -(multigrain_energy(f, plus, g)
                          - multigrain_energy(f, minus, g))/(2*du)
    np.testing.assert_allclose(
        force["configurational_force_N"], finite_difference,
        rtol=1e-4, atol=1e-18)


def test_multigrain_variational_derivatives_match_admissible_direction():
    nz, nr = 20, 14
    dr = dz = 0.5e-9
    z = (np.arange(nz)-(nz-1)/2)*dz
    rc, rf = r_centers_faces(nr, dr)
    radial = 0.5*(1+np.tanh((4e-9-rc[None, :])/1e-9))
    axial = 0.5*(1+np.tanh((4e-9-np.abs(z[:, None]))/1e-9))
    f = radial*axial
    t = np.broadcast_to(0.5*(1+np.tanh(z[:, None]/1.5e-9)), f.shape)
    phi = np.stack([0.2+0.1*(1-t), 0.5+0.05*t, 0.3-0.1*(1-t)-0.05*t])
    g = dict(z=z, r_c=rc, r_f=rf, dr=dr, dz=dz,
             config=SimpleNamespace(width=2e-9, outer_radius=6e-9))
    shape = np.sin(np.pi*(z-z.min())/(z.max()-z.min()))[:, None]
    df = 0.01*np.broadcast_to(shape, f.shape)
    dphi = np.stack([
        0.01*np.broadcast_to(shape, f.shape),
        -0.006*np.broadcast_to(shape, f.shape),
        -0.004*np.broadcast_to(shape, f.shape)])
    mu, gphi = multigrain_variational_derivatives(f, phi, g)
    factor = 2.0*np.pi*dr*dz
    analytic = factor*np.sum(rc[None, :]*(mu*df+np.sum(gphi*dphi, axis=0)))
    epsilon = 1e-5
    finite = (multigrain_energy(f+epsilon*df, phi+epsilon*dphi, g)
              - multigrain_energy(f-epsilon*df, phi-epsilon*dphi, g))/(2*epsilon)
    np.testing.assert_allclose(analytic, finite, rtol=2e-7, atol=1e-20)
