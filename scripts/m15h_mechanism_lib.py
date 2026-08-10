"""Milestone 15H: extends m15g_mechanism_lib.py with mobility-scaled
single-operator trials (Section 5) and a conservative control-volume
surface mass balance (Section 9). Reuses m15g's trial_evolve/
operator_audit/flux_budget/tj_velocity_kinematic unchanged.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m15_gb_surface_rate_competition import BC_X, BC_Y  # noqa: E402
from m15g_mechanism_lib import _mu_field  # noqa: E402
from pf_sintering.bc_ops import flux_divergence  # noqa: E402
from pf_sintering.constrained_eta import constrained_tangent_cone_eta_update  # noqa: E402
from pf_sintering.surface_transport import surface_flux_face_projected, variational_surface_diffusion_step  # noqa: E402


def trial_evolve_scaled(f, e1, e2, e3, s, p, mode, n_steps, M_f_scale=1.0, M_eta_scale=1.0,
                         bc_x=BC_X, bc_y=BC_Y):
    """Section 5: like m15g_mechanism_lib.trial_evolve, but scales p.M_f
    (surface mobility) and/or p.M_eta (GB mobility) by the given factors
    for the duration of the trial only -- p itself is never mutated
    (dataclasses.replace makes a shallow copy with new field values).
    M_s_int = m_s_ref(M_f,W) is exactly proportional to M_f (Section 2's
    audit), so M_f_scale directly IS the M_s-integrated-mobility scale;
    M_eta_scale directly IS the physical M_GB scale (M_GB=pi^2*M_eta*W/4,
    linear in M_eta)."""
    assert mode in ("SURF", "GB", "FULL")
    p_trial = dataclasses.replace(p, M_f=p.M_f * M_f_scale, M_eta=p.M_eta * M_eta_scale)
    f = f.copy(); e1 = e1.copy(); e2 = e2.copy(); e3 = e3.copy()
    dt = p.dt
    from pf_sintering.surface_transport import m_s_ref
    M_s = m_s_ref(p_trial.M_f, p_trial.interface_width)
    for _ in range(n_steps):
        if mode in ("SURF", "FULL"):
            mu = _mu_field(f, e1, e2, e3, s, p_trial, bc_x, bc_y)
            f, _ = variational_surface_diffusion_step(f, mu, p_trial.dx, dt, p_trial.interface_width, M_s,
                                                        bc_x=bc_x, bc_y=bc_y)
        if mode in ("GB", "FULL"):
            e1, e2, e3, _ = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p_trial, dt=dt, use_eta3=False,
                                                                  bc_x=bc_x, bc_y=bc_y)
    return f, e1, e2, e3


def control_volume_mass_balance(f_before, f_after, mu_before, p, tj_xy, n_steps, dt,
                                 shells_W=((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)),
                                 bc_x=BC_X, bc_y=BC_Y):
    """Section 9: for concentric EUCLIDEAN-distance shells around tj_xy
    (distance in units of W -- a reasonable proxy for arclength this
    close to the TJ, since W << R2), report:
      - Delta(sum f) over the shell (direct mass count, physical units
        via *dx^2), a genuine "which region gains/loses mass" answer;
      - the flux-divergence-predicted Delta (-dt*sum(div J)*dx^2) as an
        exact-conservation cross-check (should match Delta(sum f) to
        floating point for a SURF-only trial, since f's own update IS
        -dt*div(J) cell-by-cell -- this is the "conserved-field
        divergence as authoritative cell-level counterpart" Section 10
        asks for, reused here for the mass-balance check too).
    M_s must be passed via p (uses p.M_f/p.interface_width directly,
    matching whatever trial produced f_before/mu_before)."""
    from pf_sintering.surface_transport import m_s_ref
    Nx, Ny, dx = p.Nx, p.Ny, p.dx
    x = (np.arange(1, Nx + 1)) * dx
    y = (np.arange(1, Ny + 1)) * dx
    X, Y = np.meshgrid(x, y)
    dist = np.hypot(X - tj_xy[0], Y - tj_xy[1])
    W = p.interface_width

    M_s = m_s_ref(p.M_f, p.interface_width)
    fp = surface_flux_face_projected(f_before, mu_before, dx, W, M_s, bc_x, bc_y)
    # EXACT conservative divergence primitive (bc_ops.flux_divergence) of
    # the SAME face fluxes variational_surface_diffusion_step applies --
    # not a hand-rolled face-difference, which risks a BC-handling
    # mismatch at the domain edges.
    div_total = flux_divergence(fp["Jx_face"], fp["Jy_face"], dx, bc_x=bc_x, bc_y=bc_y)

    out = {}
    for lo_W, hi_W in shells_W:
        mask = (dist >= lo_W * W) & (dist < hi_W * W)
        n_cells = int(mask.sum())
        if n_cells == 0:
            out[f"{lo_W:g}-{hi_W:g}W"] = None
            continue
        mass_before = float(f_before[mask].sum()) * dx * dx
        mass_after = float(f_after[mask].sum()) * dx * dx
        delta_mass = mass_after - mass_before
        predicted_delta = -n_steps * dt * float(div_total[mask].sum()) * dx * dx
        out[f"{lo_W:g}-{hi_W:g}W"] = dict(n_cells=n_cells, delta_mass=delta_mass,
                                           predicted_delta_from_div=predicted_delta,
                                           rel_check=(abs(delta_mass - predicted_delta) /
                                                      max(abs(delta_mass), 1e-30)))
    return out
