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


def test_signed_frame_coordinate_keeps_fixed_frames_and_moves_midplane():
    from pf_sintering.work_conjugate_densification import (
        bounded_signed_shift_toward_minus_z,
        conservative_signed_shift_toward_minus_z,
        signed_one_contact_frame_state,
    )
    field = np.linspace(0.0, 1.0, 32)[:, None]*np.ones((1, 7))
    for displacement in (-0.2, 0.2):
        shifted = conservative_signed_shift_toward_minus_z(field, displacement, 1.0)
        assert np.allclose(shifted.sum(axis=0), field.sum(axis=0), rtol=0, atol=1e-14)
        bounded = bounded_signed_shift_toward_minus_z(field, displacement, 1.0)
        assert bounded.min() >= field.min()
        assert bounded.max() <= field.max()

    f, ownership, z, r, dr, dz = _state()
    axial = 0.5*(1.0+np.tanh((9.0e-9-np.abs(z[:, None]))/1.0e-9))
    f = f*axial
    support = triple_junction_support(
        f, z, r, z_tj=0.0, r_tj=4.0e-9, width=1.0e-9)
    for u in (-0.1e-9, 0.1e-9):
        state = signed_one_contact_frame_state(
            f, ownership, z, r, dr=dr, dz=dz, approach_m=u,
            gb_z_m=0.0, tj_r_m=4.0e-9, width_m=1.0e-9,
            moving_grain=1, neighbor_grain=0, source_support=support)
        assert state.moving_frame_translation_z_m == -u
        assert state.ownership_plane_translation_z_m == -0.5*u
        assert state.mass_relative_error < 3e-15
        assert state.partition_residual < 3e-15
        assert state.f_bounds[0] >= -2e-12
        assert state.f_bounds[1] <= 1.0+2e-12


def test_positive_direction_splits_exactly_into_sweep_and_deposit():
    from pf_sintering.work_conjugate_densification import (
        positive_one_contact_direction_components,
    )
    f, ownership, z, r, dr, dz = _state()
    axial = 0.5*(1.0+np.tanh((9.0e-9-np.abs(z[:, None]))/1.0e-9))
    f = f*axial
    support = triple_junction_support(
        f, z, r, z_tj=0.0, r_tj=4.0e-9, width=1.0e-9)
    du = 0.01e-9
    direction = positive_one_contact_direction_components(
        f, ownership, z, r, dr=dr, dz=dz, delta_u_m=du,
        gb_z_m=0.0, tj_r_m=4.0e-9, width_m=1.0e-9,
        moving_grain=1, neighbor_grain=0, source_support=support)
    np.testing.assert_allclose(
        direction.f_swept_u+direction.f_deposit_u,
        (direction.state.f-f)/du, rtol=2e-13, atol=1e-6)
    volume_factor = 2.0*np.pi*dr*dz
    swept_rate = -volume_factor*np.sum(r[None, :]*direction.f_swept_u)
    deposit_rate = volume_factor*np.sum(r[None, :]*direction.f_deposit_u)
    np.testing.assert_allclose(swept_rate, deposit_rate, rtol=2e-13, atol=1e-28)
    np.testing.assert_allclose(
        swept_rate, direction.swept_volume_rate_m2, rtol=2e-13, atol=1e-28)
