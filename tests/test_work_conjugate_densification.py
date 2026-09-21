import numpy as np

from pf_sintering.work_conjugate_densification import (
    conservative_shift_toward_minus_z,
    one_contact_state,
    triple_junction_support,
)


def _state():
    nz, nr = 48, 24
    dz = dr = 0.5e-9
    z = (np.arange(nz)-(nz-1)/2)*dz
    r = (np.arange(nr)+0.5)*dr
    radius = 4.0e-9
    f = 0.5*(1.0+np.tanh((radius-r[None, :])/(1.0e-9)))
    f = np.broadcast_to(f, (nz, nr)).copy()
    chi = 0.5*(1.0+np.tanh(z[:, None]/1.0e-9))
    chi = np.broadcast_to(chi, f.shape).copy()
    ownership = np.stack([1.0-chi, chi])
    return f, ownership, z, r, dr, dz


def test_conservative_subcell_shift_preserves_column_mass_and_bounds():
    field = np.linspace(0.0, 1.0, 32)[:, None]*np.ones((1, 7))
    shifted = conservative_shift_toward_minus_z(field, 0.2, 1.0)
    assert np.allclose(shifted.sum(axis=0), field.sum(axis=0), rtol=0, atol=1e-14)
    assert shifted.min() >= field.min()
    assert shifted.max() <= field.max()


def test_physical_coordinate_is_identity_at_zero_and_conserves_volume():
    f, ownership, z, r, dr, dz = _state()
    support = triple_junction_support(
        f, z, r, z_tj=0.0, r_tj=4.0e-9, width=1.0e-9)
    zero = one_contact_state(
        f, ownership, z, r, dr=dr, dz=dz, u_m=0.0,
        gb_z_m=0.0, tj_r_m=4.0e-9, width_m=1.0e-9,
        moving_grain=1, neighbor_grain=0, source_support=support)
    assert np.array_equal(zero.f, f)
    assert np.allclose(zero.ownership, ownership, rtol=0, atol=1e-15)

    moved = one_contact_state(
        f, ownership, z, r, dr=dr, dz=dz, u_m=0.25e-9,
        gb_z_m=0.0, tj_r_m=4.0e-9, width_m=1.0e-9,
        moving_grain=1, neighbor_grain=0, source_support=support)
    assert moved.swept_volume_m3 > 0.0
    assert moved.work_area_m2 > 0.0
    assert moved.mass_relative_error < 3e-15
    assert moved.partition_residual < 3e-15
    assert moved.f_bounds[0] >= -2e-12
    assert moved.f_bounds[1] <= 1.0+2e-12
    assert moved.moving_body_displacement_m == 0.25e-9
    assert moved.ownership_midplane_displacement_m == 0.125e-9
