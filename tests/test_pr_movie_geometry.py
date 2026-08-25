import h5py
import numpy as np

from pf_sintering.pr_movie_geometry import (
    MovieGeometryArchive, resample_branch_by_arclength,
)


def branches():
    return {
        "negative": {"z_m": np.array([0.0, -1.0, -2.0]),
                     "r_m": np.array([1.0, 1.5, 0.0])},
        "positive": {"z_m": np.array([2.0, 1.0, 0.0]),
                     "r_m": np.array([0.0, 1.5, 1.0])},
    }


def frame(t):
    return dict(
        t_model=t, cycle=1, avalanche_id=1, event_number=1,
        sink_state=1, avalanche_active=1, q_event_over_b=0.5,
        Q_avalanche_over_b=0.5, Q_cumulative_over_b=0.5,
        sigma_local_Pa=80e6, sigma_integral_Pa=75e6,
        r_neck_m=1.0, Vp_over_Vp_cycle=0.95,
        z_TJ_m=0.0, r_TJ_m=1.0, S_completed=0,
        pending_children=0)


def test_resample_orients_both_branches_from_tj():
    for side in branches().values():
        z, r = resample_branch_by_arclength(
            side["z_m"], side["r_m"], npoint=16, z_TJ_m=0.0, r_TJ_m=1.0)
        assert z.shape == (16,)
        assert r.shape == (16,)
        assert np.hypot(z[0], r[0] - 1.0) < 1e-12


def test_archive_is_fixed_size_append_only_and_strict_time(tmp_path):
    path = tmp_path / "movie.h5"
    with MovieGeometryArchive(
            path, nbranch=16, seconds_per_model_time=0.5) as archive:
        archive.append(branches(), frame(0.0), frame_type="root_nucleation")
        archive.append(branches(), frame(0.1), frame_type="active_1b_transit",
                       flush=True)
        try:
            archive.append(branches(), frame(0.1), frame_type="child_completion")
        except ValueError as error:
            assert "strictly" in str(error)
        else:
            raise AssertionError("duplicate frame time was accepted")
    with h5py.File(path, "r") as file:
        assert file["geometry/z_negative_m"].shape == (2, 16)
        assert np.allclose(file["time/t_s"][:], [0.0, 0.05])
        assert file["state/frame_id"][:].tolist() == [0, 1]
        assert file["state/r_GB_m"][0] == 1.0


def test_archive_can_reopen_and_append_for_exact_checkpoint_continuation(tmp_path):
    path = tmp_path / "movie_resume.h5"
    with MovieGeometryArchive(
            path, nbranch=16, seconds_per_model_time=0.5) as archive:
        archive.append(branches(), frame(0.0), frame_type="root_nucleation")
    with MovieGeometryArchive(
            path, nbranch=16, seconds_per_model_time=0.5,
            mode="a") as archive:
        assert archive.nframe == 1
        assert archive.last_t_model == 0.0
        archive.append(
            branches(), frame(0.1), frame_type="active_1b_transit",
            flush=True)
    with h5py.File(path, "r") as file:
        assert file["state/frame_id"][:].tolist() == [0, 1]
        assert np.allclose(file["time/t_model"][:], [0.0, 0.1])
