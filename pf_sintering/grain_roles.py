"""Milestone 16M Section 2: explicit particle/substrate role mapping.

M16L's central finding was that `pf_sintering.m16j_geometry.build_candidate_geometry`
assigns e1=SUBSTRATE, e2=PARTICLE -- the OPPOSITE of what
`pf_sintering.axisym_sink_rbm.py`'s functions (and the OLDER M16G geometry
they were originally validated against) assume (e1=particle, e2=substrate).
Every M16K driver script that called those functions directly with the
M16J geometry's raw e1/e2 was silently advecting/measuring the substrate.

This module exists so that ambiguity is never again implicit: geometry
construction should be followed IMMEDIATELY by a call to
`roles_from_m16j_geometry(e1, e2)`, and every downstream RBM/transport
call should receive explicitly named `particle_field`/`substrate_field`
arguments (or a `GrainRoles` object), never raw `e1`/`e2`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GrainRoles:
    """Explicit, self-documenting particle/substrate field handles. Build
    via `roles_from_m16j_geometry`; never construct directly from raw
    e1/e2 without going through that function (or an equivalent, audited
    mapping for a DIFFERENT geometry module -- the mapping is
    geometry-construction-specific, not universal)."""
    particle: np.ndarray
    substrate: np.ndarray


def roles_from_m16j_geometry(e1: np.ndarray, e2: np.ndarray) -> GrainRoles:
    """M16J flat-substrate (and finite-chi) geometry convention:
    e1=substrate, e2=particle (`ind_inner~1` for z<0/substrate,
    `e1=f*ind_inner`). Returns the SWAPPED, explicitly-named mapping."""
    return GrainRoles(particle=e2, substrate=e1)


def assert_particle_above_substrate(particle, substrate, z, r_c, dz=None, dr=None) -> None:
    """Runtime guard against accidental role reversal: for this project's
    axisymmetric flat-substrate convention, the particle's mass-weighted
    COM must sit at a LARGER z than the substrate's (particle sits above
    the flat substrate). Raises AssertionError if this is violated (e.g.
    if particle/substrate were passed in swapped)."""
    from pf_sintering.axisym_sink_rbm import particle_com_z
    dz_eff = dz if dz is not None else float(z[1] - z[0])
    dr_eff = dr if dr is not None else dz_eff
    com_particle = particle_com_z(particle, z, r_c, dr_eff, dz_eff)
    com_substrate = particle_com_z(substrate, z, r_c, dr_eff, dz_eff)
    assert np.isfinite(com_particle) and np.isfinite(com_substrate), (
        f"non-finite COM: particle={com_particle}, substrate={com_substrate}")
    assert com_particle > com_substrate, (
        f"GRAIN-ROLE REVERSAL SUSPECTED: particle COM z={com_particle*1e9:.4f}nm "
        f"is not above substrate COM z={com_substrate*1e9:.4f}nm -- check that "
        f"particle_field/substrate_field were not passed in swapped.")
