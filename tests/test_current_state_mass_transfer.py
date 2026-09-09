import numpy as np

from pf_sintering.current_state_mass_transfer import (
    TransferMaskParameters,
    axisymmetric_integral,
    axisymmetric_weights,
    bounded_conservative_transfer,
    build_current_state_masks,
    transport_clock_increment,
)


def _state():
    z = np.linspace(-8.0, 8.0, 65)
    r = np.linspace(0.125, 8.125, 33)
    Z, R = np.meshgrid(z, r, indexing="ij")
    distance = np.hypot(Z, R-3.0)
    f = 0.5 * (1.0-np.tanh((distance-3.0)/0.5))
    phi = 0.5 * (1.0+np.tanh(Z/0.5))
    return (f, f*phi, f*(1.0-phi)), z, r


def test_bounded_transfer_is_axisymmetrically_conservative():
    state, z, r = _state()
    receiver, donor, metadata = build_current_state_masks(
        state[0], z, r, z_TJ_m=0.0, r_TJ_m=3.0, W_m=0.5,
        parameters=TransferMaskParameters())
    weights = axisymmetric_weights(r, r[1]-r[0], z[1]-z[0])
    capacity = min(
        axisymmetric_integral(receiver*(1.0-state[0]), weights),
        axisymmetric_integral(donor*state[0], weights))
    updated, diagnostic = bounded_conservative_transfer(
        state, receiver, donor, transfer_volume_m3=0.01*capacity,
        r_c=r, dr=r[1]-r[0], dz=z[1]-z[0])
    assert metadata["supports_disjoint"]
    assert diagnostic["immutable_parent_geometry_used"] is False
    assert diagnostic["live_state_clipped"] is False
    assert diagnostic["relative_mass_closure"] < 2e-13
    assert diagnostic["total_volume_relative_error"] < 2e-14
    assert np.min(updated[0]) >= 0.0
    assert np.max(updated[0]) <= 1.0
    assert np.max(np.abs(updated[1]+updated[2]-updated[0])) < 5e-15


def test_masks_follow_current_tj_not_parent_geometry():
    state, z, r = _state()
    receiver0, donor0, _ = build_current_state_masks(
        state[0], z, r, z_TJ_m=0.0, r_TJ_m=3.0, W_m=0.5)
    receiver1, donor1, _ = build_current_state_masks(
        state[0], z, r, z_TJ_m=0.5, r_TJ_m=3.0, W_m=0.5)
    assert not np.array_equal(receiver0, receiver1)
    assert not np.array_equal(donor0, donor1)


def test_clock_uses_both_endpoint_volume_rates():
    assert transport_clock_increment(2.0, 4.0, 2.0) == 0.75
