import math

import numpy as np

from pf_sintering.ch_exact_energy import mu_isotropic
from pf_sintering.ch_crossover_diagnostics import neck_ch_mass_balance, neck_region_mask
from pf_sintering.constrained_eta import constrained_variational_eta_update
from pf_sintering.flux_closure import (
    flux_divergence_shape_change_cross_check,
    neck_boundary_face_flux_balance,
    neck_control_volume_flux_balance,
    particle_volume_rate_decomposition,
    transport_rate_flux_check,
)
from pf_sintering.model import ModelConfig, Sink, build_params, initialize_fields, reproject
from pf_sintering.surface_transport import m_s_ref, surface_divergence_update
from pf_sintering.tj_force import compute_neck_tj_forces
from pf_sintering.tj_subgrid import compute_subgrid_contact

BC_X, BC_Y = "reflecting", "periodic"


def _sinusoidal_state():
    p = build_params(ModelConfig(
        preset="dev", geometry="sinusoidal_substrate", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6, seed=42,
        sinusoid_wavelength=480e-9, sinusoid_amplitude=24e-9,
        interface_width_override=20e-9, eta_diffusivity_fixed_physical=True,
        use_aniso_surface=False, surface_mobility_scale=0.3,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    return p, f, e1, e2, e3


def _one_step(f, e1, e2, e3, s, p, M_s):
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)
    e1_new, e2_new, e3_new, ediag = constrained_variational_eta_update(e1, e2, e3, f_new, s, p, bc_x=BC_X, bc_y=BC_Y)
    return f_new, e1_new, e2_new, e3_new, diag


def test_particle_volume_rate_decomposition_closes_exactly():
    p, f, e1, e2, e3 = _sinusoidal_state()
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    f_new, e1n, e2n, e3n, diag = _one_step(f, e1, e2, e3, s, p, M_s)

    r = particle_volume_rate_decomposition(f, f_new, e1, e2, e3, e1n, e2n, e3n, p.dt, p.dx)
    assert abs(r["closure_residual"]) < 1e-6 * max(abs(r["V2_new"]), 1e-300)
    assert math.isclose(r["dV2_dt_transport"] + r["dV2_dt_eta_migration"], r["dV2_dt_total"], rel_tol=1e-6, abs_tol=1e-30)


def test_transport_rate_flux_check_matches_decomposition():
    p, f, e1, e2, e3 = _sinusoidal_state()
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    f_new, e1n, e2n, e3n, diag = _one_step(f, e1, e2, e3, s, p, M_s)

    r = particle_volume_rate_decomposition(f, f_new, e1, e2, e3, e1n, e2n, e3n, p.dt, p.dx)
    denom_new = e1n + e2n + e3n + 1e-30
    flux_based = transport_rate_flux_check(diag["Jx"], diag["Jy"], e2n, denom_new, p.dx, BC_X, BC_Y)
    # both derived from the same underlying flux field via an integration-
    # by-parts identity that is exact only up to O(dx) boundary/interpolation
    # effects (Jx is cell-centered here, not the exact face flux used in the
    # conservative update) -- a loose but meaningful relative tolerance.
    scale = max(abs(r["dV2_dt_transport"]), abs(flux_based), 1e-300)
    assert abs(r["dV2_dt_transport"] - flux_based) / scale < 0.5


def test_neck_control_volume_flux_balance_closes_and_matches_direct_mass_change():
    p, f, e1, e2, e3 = _sinusoidal_state()
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
    mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
    # Milestone 13 Section 8: wall_frac*Nx*dx, NOT (wall_frac-0.5)*Nx*dx --
    # the latter is model.initialize_fields' CENTERED-coordinate formula;
    # every array here (mask, Jx, Jy) uses the UNCENTERED convention
    # (x=(arange(1,Nx+1))*dx), where the correct wall position has no -0.5
    # shift (see scripts/m12b_grid_convergence.py's run_campaign).
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx

    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)

    r = neck_control_volume_flux_balance(diag["Jx"], diag["Jy"], mask, p, wall_mean, BC_X, BC_Y)
    assert abs(r["closure_residual"]) < 1e-20
    assert r["n_particle_half"] + r["n_substrate_half"] == int(mask.sum())

    direct = neck_ch_mass_balance(f, f_new, mask, p) / p.dt
    scale = max(abs(direct), abs(r["dM_neck_dt_total"]), 1e-300)
    assert abs(direct - r["dM_neck_dt_total"]) / scale < 1e-6


def test_neck_boundary_face_flux_balance_closes_and_is_not_degenerate():
    # Section 8: the direct boundary-face accounting must reproduce the
    # volume-integrated -div(J) exactly, AND (with the wall_mean fix above)
    # must NOT collapse to an all-particle-side/zero-substrate-side split
    # the way the older cell-based half-split did on this same state.
    p, f, e1, e2, e3 = _sinusoidal_state()
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    sub = compute_subgrid_contact(f, e1, e2, p)
    assert sub.resolved
    mask = neck_region_mask(p, (sub.top.x_sub, sub.top.y_sub), (sub.bottom.x_sub, sub.bottom.y_sub))
    wall_mean = p.substrate_wall_frac * p.Nx * p.dx

    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)

    r = neck_boundary_face_flux_balance(diag["Jx"], diag["Jy"], mask, p, wall_mean, BC_X, BC_Y)
    assert abs(r["closure_residual"]) < 1e-20
    assert math.isclose(r["dM_neck_dt_total_from_faces"], r["dM_neck_dt_total_from_volume"], rel_tol=1e-9, abs_tol=1e-30)
    assert r["n_particle_faces"] > 0
    assert r["n_substrate_faces"] > 0


def test_flux_divergence_shape_change_cross_check_agrees_at_interface():
    p, f, e1, e2, e3 = _sinusoidal_state()
    s = Sink(threshold=math.inf)
    M_s = m_s_ref(p.M_f, p.interface_width)
    mu = mu_isotropic(f, e1, e2, e3, s, p)
    f_new, diag = surface_divergence_update(f, mu, p.dx, p.dt, p.interface_width, M_s, bc_x=BC_X, bc_y=BC_Y)

    r = flux_divergence_shape_change_cross_check(f, f_new, diag["Jx"], diag["Jy"], p.dt, p, BC_X, BC_Y)
    assert r["band_fraction"] > 0.0
    assert r["sign_agreement_fraction"] > 0.999
