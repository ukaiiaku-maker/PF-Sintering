import numpy as np
import pytest

from pf_sintering.continuous_field_tj import (
    ContinuousFieldTJTracker,
    TJTrackingError,
    field_intersection_candidates,
)


def manufactured_fields(z_interface=2.25):
    z = np.arange(6.0)
    r = np.arange(0.5, 5.0, 1.0)
    radius = 3.2 + 0.1 * z
    f = 0.5 * (1.0 - np.tanh((r[None, :] - radius[:, None]) / 0.5))
    ownership = np.tanh((z - z_interface) / 0.5)[:, None]
    particle = 0.5 * f * (1.0 + ownership)
    substrate = f - particle
    return z, r, f, particle, substrate


def test_field_intersection_is_subcell_and_uses_surface_ownership():
    z, r, f, particle, substrate = manufactured_fields()
    candidates = field_intersection_candidates(f, particle, substrate, z, r)
    assert len(candidates) == 1
    z_tj, r_tj = candidates[0]
    assert 2.0 < z_tj < 3.0
    assert 3.3 < r_tj < 3.6


def test_tracker_rejects_disconnected_many_cell_jump():
    z, r, f, particle, substrate = manufactured_fields(2.25)
    tracker = ContinuousFieldTJTracker(
        z=z, r_c=r, initial_z_m=2.25, max_jump_m=1.5)
    first = tracker.locate(f, particle, substrate)
    assert first["jump_m"] == 0.0
    _, _, f2, particle2, substrate2 = manufactured_fields(4.25)
    with pytest.raises(TJTrackingError, match="refusing to switch branches"):
        tracker.locate(f2, particle2, substrate2)

