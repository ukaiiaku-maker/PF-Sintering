"""Exact radial vacuum padding for axisymmetric phase-field states."""
from __future__ import annotations

import math

import numpy as np


def pad_case_radially_with_vacuum(geom: dict, setup: dict, *,
                                  target_outer_face_m: float):
    """Return copied case dictionaries with zero-valued outer radial cells.

    Existing cells and their physical coordinates are copied bit-for-bit. No
    interpolation, scaling, translation, or deformation is performed.
    """
    dr = float(setup["dr"])
    old_face = float(np.asarray(setup["r_f"])[-1])
    if target_outer_face_m <= old_face:
        raise ValueError("target radial face must exceed the current boundary")
    cells = int(math.ceil((float(target_outer_face_m)-old_face)/dr))
    new_face = old_face + cells*dr
    padded_geom = dict(geom)
    for key in ("f", "e1", "e2"):
        field = np.asarray(geom[key])
        padded_geom[key] = np.pad(
            field, ((0, 0), (0, cells)), mode="constant",
            constant_values=0.0)
        if not np.array_equal(padded_geom[key][:, :field.shape[1]], field):
            raise RuntimeError(f"radial padding changed existing {key} cells")
    old_rc = np.asarray(setup["r_c"], dtype=float)
    old_rf = np.asarray(setup["r_f"], dtype=float)
    added_rc = old_rc[-1] + dr*np.arange(1, cells+1, dtype=float)
    added_rf = old_rf[-1] + dr*np.arange(1, cells+1, dtype=float)
    new_rc = np.concatenate([old_rc, added_rc])
    new_rf = np.concatenate([old_rf, added_rf])
    if not np.array_equal(new_rc[:old_rc.size], old_rc):
        raise RuntimeError("radial padding changed existing cell coordinates")
    if not np.array_equal(new_rf[:old_rf.size], old_rf):
        raise RuntimeError("radial padding changed existing face coordinates")
    padded_setup = dict(setup, r_c=new_rc, r_f=new_rf)
    padded_geom.update(
        r_c=new_rc, r_f=new_rf, Nr=int(new_rc.size),
        radial_margin_W=(float(geom.get("radial_margin_W", math.nan))
                         + cells*dr/float(setup["W"])))
    diagnostics = dict(
        radial_padding_cells=cells,
        old_radial_face_m=old_face,
        new_radial_face_m=new_face,
        requested_radial_face_m=float(target_outer_face_m),
        dr_m=dr,
        existing_field_values_preserved_exactly=True,
        existing_radial_coordinates_preserved_exactly=True,
        added_cells_initialized_as_vacuum=True,
        physical_state_rescaled=False,
        physical_state_interpolated=False,
        physical_state_deformed=False,
    )
    return padded_geom, padded_setup, diagnostics
