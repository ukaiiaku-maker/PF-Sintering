import math

import numpy as np

from pf_sintering.experimental_pr_metrology import _particle_silhouette
from pf_sintering.pr_experimental_geometry import contour_reconstruction


def test_particle_normal_support_is_radial_not_axial():
    branch = {
        "z_m": np.array([0.0, 1.0, 2.0, 4.0]),
        "r_m": np.array([1.0, 2.0, 3.0, 0.0]),
    }
    result = _particle_silhouette(branch)
    assert result["particle_axial_length_m"] == 4.0
    assert result["w_N_particle_m"] == 6.0
    assert result["w_N_m"] == 6.0
    assert result["P_h_m"] == result["P_h_particle_m"]
    assert result["w_bar_2D_m"] == result["w_bar_2D_particle_m"]


def test_contour_support_and_hull_exclude_cropped_substrate():
    particle = {
        "z_m": np.array([0.0, 1.0, 2.0, 4.0]),
        "r_m": np.array([1.0, 2.0, 3.0, 0.0]),
        "grain_label": np.array([0, 1, 1, 1], dtype=np.int8),
    }
    substrate = {
        "z_m": np.array([0.0, -1.0, -2.0]),
        "r_m": np.array([1.0, 50.0, 100.0]),
        "grain_label": np.array([0, 2, 2], dtype=np.int8),
    }
    reconstructed = contour_reconstruction(
        {"negative": substrate, "positive": particle})
    assert reconstructed["w_N_m"] == 6.0
    particle_only = _particle_silhouette(particle)
    assert math.isclose(
        reconstructed["P_h_m"], particle_only["P_h_particle_m"],
        rel_tol=1e-14)
