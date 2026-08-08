"""EXPERIMENTAL / DIAGNOSTIC ONLY -- not wired into production physics.

Milestone 12B Sections 15-17: particle and neck control-volume flux
closures for the unified transport law, and a flux-divergence-vs-shape-
change cross-check.

Section 15 -- particle (grain 2) control volume: `dV2/dt` measured directly
from `f_weighted_ownership_volumes` is split EXACTLY into two pieces via
    V2_new - V2_old = sum((f_new-f_old)*w2_new)*dx^2       [[transport]]
                     + sum(f_old*(w2_new-w2_old))*dx^2     [[eta migration]]
(w_i = eta_i/sum_j eta_j, the ownership fraction). This is an exact
identity (not an approximation -- add the two lines and every f_new*w2_new
/ f_old*w2_old term telescopes back to V2_new - V2_old), so "transport"
(conserved-f surface diffusion, weighted by the POST-step ownership) and
"eta ownership migration" (structural relabeling at fixed PRE-step f) are
cleanly separated and sum to exactly the measured total -- satisfying the
handoff's "do NOT call an eta-ownership change mass transport unless f
moves consistently" requirement.

The transport piece is independently cross-checked against the continuum
identity d/dt integral(f*w2)dV |_transport-only = integral(J . grad(w2)) dV
(integrating -integral(w2 * div(J)) dV by parts, assuming zero net flux
across the domain's own periodic/no-flux boundary, which
surface_transport.py's construction guarantees) -- computed directly from
the actual Jx, Jy field `surface_divergence_update` already returns, not
re-derived.

Section 16 -- neck control volume: `bc_ops.flux_divergence` gives an EXACT
per-cell -div(J) whose sum over any subset of cells equals (by the same
discrete divergence theorem `flux_divergence`'s docstring already proves)
the net inflow across that subset's own boundary, regardless of the
subset's shape. Splitting `ch_crossover_diagnostics.neck_region_mask` into
a particle-facing half and a substrate-facing half (via the same X-
deviation-from-substrate-baseline classifier `m12b_grid_convergence.py`
uses for TJ branches) and summing -div(J)*dx^2 separately over each half
gives an exact, disjoint decomposition of dM_neck/dt -- whether the
particle-facing portion of the neck is gaining or losing mass relative to
the substrate-facing portion, without needing face-by-face boundary
bookkeeping (Section 16's literal "J_particle_to_neck" terms are not
separately identified here; this reports the physically equivalent
question of which half of the neck control volume is accumulating mass).

**Milestone 13 Section 8 fix**: the cell-based half-split above
(`neck_control_volume_flux_balance`) returned an exactly-zero substrate-
half contribution on the representative Milestone 12B state -- a
degenerate result (see that milestone's report), not a validated finding.
`neck_boundary_face_flux_balance` replaces it with the DIRECT boundary-
face accounting Section 8 explicitly asks for: every FACE where a masked
cell borders an unmasked cell is identified, its actual face-centered flux
(the same `face_average`-interpolated Jx/Jy production's own conservative
update uses, not a cell-centered proxy) is signed as into/out-of the
control volume, and each such face is classified particle-side/substrate-
side by the face-averaged substrate-baseline deviation at that face (not
by which cell happens to be "inside"). Summing ALL boundary faces (both
sides together) reproduces the volume-integrated `-div(J)` exactly (an
algebraic identity following from `flux_divergence`'s own telescoping
argument -- every INTERIOR face of the mask cancels between its two
mask-side contributions, leaving only the boundary faces), verified
directly in `tests/test_flux_closure.py` rather than assumed.
`neck_control_volume_flux_balance` (cell-based) is kept for backward
compatibility/comparison but should no longer be treated as the
particle/substrate split of record.

Section 17 -- cross-check: -div(J) (Section 16's exact per-cell rate) must
have the correct sign relationship to the measured f=0.5 contour's local
motion (interface retreating into vapor where -div(J)>0 locally near the
interface, advancing where <0) -- checked pointwise near the interface,
not just integrated.
"""

from __future__ import annotations

import math

import numpy as np

from .bc_ops import _shift, flux_divergence, grad_bc
from .surface_transport import face_average


def particle_volume_rate_decomposition(f_old, f_new, e1_old, e2_old, e3_old,
                                        e1_new, e2_new, e3_new, dt, dx, eps=1e-30):
    """Exact split of dV2/dt into f-transport and eta-ownership-migration
    contributions (module docstring). Returns a dict with V2_old, V2_new,
    dV2_dt_total, dV2_dt_transport, dV2_dt_eta_migration (the latter two
    sum exactly to the total, up to float roundoff)."""
    denom_old = e1_old + e2_old + e3_old + eps
    denom_new = e1_new + e2_new + e3_new + eps
    w2_old = e2_old / denom_old
    w2_new = e2_new / denom_new

    V2_old = float(np.sum(f_old * w2_old)) * dx * dx
    V2_new = float(np.sum(f_new * w2_new)) * dx * dx
    transport_term = float(np.sum((f_new - f_old) * w2_new)) * dx * dx
    eta_term = float(np.sum(f_old * (w2_new - w2_old))) * dx * dx

    return dict(
        V2_old=V2_old, V2_new=V2_new,
        dV2_dt_total=(V2_new - V2_old) / dt,
        dV2_dt_transport=transport_term / dt,
        dV2_dt_eta_migration=eta_term / dt,
        closure_residual=(V2_new - V2_old) - (transport_term + eta_term),
    )


def transport_rate_flux_check(Jx, Jy, e2_new, denom_new, dx, bc_x, bc_y):
    """Independent check on the transport piece above: integral(J . grad(w2))
    dV, from the ACTUAL cell-centered flux field (not re-derived), compared
    against particle_volume_rate_decomposition's dV2_dt_transport (module
    docstring derivation). Uses w2 evaluated at the post-step ownership
    (matching the convention used for the transport term above)."""
    w2 = e2_new / denom_new
    gx, gy = grad_bc(w2, dx, bc_x=bc_x, bc_y=bc_y)
    return float(np.sum(Jx * gx + Jy * gy)) * dx * dx


def _classify_mask_half(p, mask, wall_mean):
    """Split `mask`'s cells into particle-facing / substrate-facing halves
    via the same X-deviation-from-substrate-baseline heuristic used for TJ
    branch classification (module docstring, Section 16)."""
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    x_s = wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)
    dev = X - x_s
    threshold = 0.5 * p.interface_width  # midpoint between "at the substrate" and "well into the particle"
    particle_half = mask & (dev > threshold)
    substrate_half = mask & ~particle_half
    return particle_half, substrate_half


def neck_control_volume_flux_balance(Jx, Jy, mask, p, wall_mean, bc_x, bc_y):
    """Exact per-cell -div(J) (bc_ops.flux_divergence) summed over the
    particle-facing and substrate-facing halves of `mask` (Section 16).
    Returns dM/dt for each half and their exact sum (== net dM_neck/dt from
    transport alone, at fixed eta -- matches the direct before/after mass
    difference to O(dt) time-discretization error)."""
    Jx_face = face_average(Jx, axis=1, bc=bc_x)
    Jy_face = face_average(Jy, axis=0, bc=bc_y)
    neg_div = -flux_divergence(Jx_face, Jy_face, p.dx, bc_x=bc_x, bc_y=bc_y)

    particle_half, substrate_half = _classify_mask_half(p, mask, wall_mean)
    dM_particle_half = float(np.sum(neg_div[particle_half])) * p.dx * p.dx
    dM_substrate_half = float(np.sum(neg_div[substrate_half])) * p.dx * p.dx
    dM_total = float(np.sum(neg_div[mask])) * p.dx * p.dx
    return dict(
        dM_neck_dt_particle_half=dM_particle_half,
        dM_neck_dt_substrate_half=dM_substrate_half,
        dM_neck_dt_total=dM_total,
        closure_residual=dM_total - (dM_particle_half + dM_substrate_half),
        n_particle_half=int(particle_half.sum()), n_substrate_half=int(substrate_half.sum()),
    )


def _substrate_deviation_field(p, wall_mean):
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    x_s = wall_mean + p.sinusoid_amplitude * np.cos(2 * math.pi * Y / p.sinusoid_wavelength + p.sinusoid_phase)
    return X - x_s


def neck_boundary_face_flux_balance(Jx, Jy, mask, p, wall_mean, bc_x, bc_y):
    """Milestone 13 Section 8: direct boundary-face-flux accounting around
    `mask` (see module docstring). For each of the two face families
    (x-faces between (r,c)/(r,c+1), y-faces between (r,c)/(r+1,c) in the
    `_shift`-consistent "+axis" convention `face_average` already uses),
    a face is a MASK BOUNDARY face iff exactly one of its two neighboring
    cells is inside `mask`. Its signed contribution to d(mass in mask)/dt
    is `-J_face` if the inside cell is on the "-axis" side (flux in the
    "+axis" direction carries mass OUT of the mask) or `+J_face` if the
    inside cell is on the "+axis" side (flux in the "+axis" direction
    carries mass INTO the mask). Classified particle-side/substrate-side by
    the face-averaged substrate-baseline deviation (X - x_s(Y)) AT that
    face, not by either neighboring cell alone.

    Returns dM/dt for the particle-side boundary, the substrate-side
    boundary, and their sum (which must equal the volume-integrated
    -div(J) over mask to roundoff, verified in tests/test_flux_closure.py)."""
    Jx_face = face_average(Jx, axis=1, bc=bc_x)
    Jy_face = face_average(Jy, axis=0, bc=bc_y)
    dev = _substrate_deviation_field(p, wall_mean)
    threshold = 0.5 * p.interface_width

    m = mask
    m_right = _shift(m.astype(float), -1, axis=1, bc=bc_x) > 0.5
    m_down = _shift(m.astype(float), -1, axis=0, bc=bc_y) > 0.5
    dev_face_x = face_average(dev, axis=1, bc=bc_x)
    dev_face_y = face_average(dev, axis=0, bc=bc_y)

    boundary_x = m != m_right
    in_x = np.where(m & ~m_right, -Jx_face, np.where(~m & m_right, Jx_face, 0.0))
    particle_x = boundary_x & (dev_face_x > threshold)
    substrate_x = boundary_x & ~(dev_face_x > threshold)

    boundary_y = m != m_down
    in_y = np.where(m & ~m_down, -Jy_face, np.where(~m & m_down, Jy_face, 0.0))
    particle_y = boundary_y & (dev_face_y > threshold)
    substrate_y = boundary_y & ~(dev_face_y > threshold)

    dM_particle = float(np.sum(in_x[particle_x]) + np.sum(in_y[particle_y])) * p.dx
    dM_substrate = float(np.sum(in_x[substrate_x]) + np.sum(in_y[substrate_y])) * p.dx
    dM_total_faces = dM_particle + dM_substrate

    neg_div = -flux_divergence(Jx_face, Jy_face, p.dx, bc_x=bc_x, bc_y=bc_y)
    dM_total_volume = float(np.sum(neg_div[mask])) * p.dx * p.dx

    return dict(
        dM_neck_dt_particle_side=dM_particle,
        dM_neck_dt_substrate_side=dM_substrate,
        dM_neck_dt_total_from_faces=dM_total_faces,
        dM_neck_dt_total_from_volume=dM_total_volume,
        closure_residual=dM_total_faces - dM_total_volume,
        n_particle_faces=int(particle_x.sum() + particle_y.sum()),
        n_substrate_faces=int(substrate_x.sum() + substrate_y.sum()),
    )


def flux_divergence_shape_change_cross_check(f_old, f_new, Jx, Jy, dt, p, bc_x, bc_y, band=0.49):
    """Section 17: -div(J), reconstructed here independently from the same
    cell-centered Jx, Jy surface_divergence_update returns, must predict
    the ACTUAL local f change (df/dt = -div(J) is the transport law itself)
    at every point near the current interface -- i.e. sign(-div(J)) must
    match sign(f_new-f_old) (locally more solid where -div(J)>0: the
    f=0.5 contour is advancing into the vapor there; locally less solid
    where -div(J)<0: the contour is retreating), restricted to the diffuse
    interface band |f_old-0.5|<band where "local shape change" is
    meaningful (far from any interface both quantities are ~0 and the sign
    comparison is not informative). Exact agreement here is expected (both
    sides use the identical flux_divergence construction production's own
    surface_divergence_update applies) -- this validates that
    flux_closure.py's independent reconstruction of -div(J) is consistent
    with the actual conservative update, and confirms the physical
    sign convention (advancing/retreating) pointwise, not just in
    aggregate."""
    Jx_face = face_average(Jx, axis=1, bc=bc_x)
    Jy_face = face_average(Jy, axis=0, bc=bc_y)
    neg_div_pred = -flux_divergence(Jx_face, Jy_face, p.dx, bc_x=bc_x, bc_y=bc_y)
    df_actual = (f_new - f_old) / dt

    band_mask = np.abs(f_old - 0.5) < band
    if not band_mask.any():
        return dict(band_fraction=0.0, sign_agreement_fraction=math.nan, max_abs_diff=math.nan)
    agree = np.sign(neg_div_pred[band_mask]) == np.sign(df_actual[band_mask])
    max_abs_diff = float(np.max(np.abs(neg_div_pred[band_mask] - df_actual[band_mask])))
    return dict(band_fraction=float(band_mask.mean()),
                sign_agreement_fraction=float(agree.mean()),
                max_abs_diff=max_abs_diff)
