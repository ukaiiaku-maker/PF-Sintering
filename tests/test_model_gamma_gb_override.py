import math

from pf_sintering.gb_obstacle_energy import gb_obstacle_coefficients, m_eta_from_m_gb
from pf_sintering.model import ModelConfig, _gb_energy, build_params


def test_gamma_gb_override_none_preserves_existing_behavior():
    p_default = build_params(ModelConfig(preset="dev"))
    p_explicit_none = build_params(ModelConfig(preset="dev", gamma_gb_override=None))
    expected = _gb_energy(30.0)  # default theta_mis_deg
    assert math.isclose(p_default.gamma_gb, expected, rel_tol=1e-12)
    assert math.isclose(p_default.gamma_gb, p_explicit_none.gamma_gb, rel_tol=1e-12)
    assert math.isclose(p_default.gamma_gb, p_default.gamma_gb_ref, rel_tol=1e-12)


def test_gamma_gb_override_sets_gamma_gb_and_rederives_dependents():
    # Milestone 14 Section 8 (coefficient formula updated by Milestone 14G's
    # obstacle-equilibrium calibration): a single config path re-derives
    # EVERY gamma_gb-dependent coefficient consistently
    # (k_eta/W_cpl_f=Wc from gb_obstacle_energy.gb_obstacle_coefficients,
    # delta_GB=interface_width convention), not just a diagnostic-only
    # gamma_gb_ref.
    gamma_s = 1.0
    for psi_eq_deg in (90.0, 120.0, 150.0):
        psi_eq = math.radians(psi_eq_deg)
        gamma_gb = 2.0 * gamma_s * math.cos(psi_eq / 2.0)
        p = build_params(ModelConfig(preset="dev", gamma_gb_override=gamma_gb))
        assert math.isclose(p.gamma_gb, gamma_gb, rel_tol=1e-12)
        assert math.isclose(p.gamma_gb_ref, gamma_gb, rel_tol=1e-12)
        expected = gb_obstacle_coefficients(gamma_gb, p.interface_width)
        assert math.isclose(p.k_eta, expected["k_eta"], rel_tol=1e-12)
        assert math.isclose(p.W_cpl_f, expected["Wc"], rel_tol=1e-12)
        # not tied to theta_mis_deg's empirical Read-Shockley-like curve
        assert not math.isclose(p.gamma_gb, _gb_energy(30.0), rel_tol=1e-6)


def test_gamma_gb_override_changes_active_groove_coupling_in_mu():
    # The override must actually reach the DYNAMICALLY ACTIVE coefficient
    # (Wc=36*gamma_gb_ref/W in ch_exact_energy.mu0_bulk / model.evolve_f's
    # own mu0), not merely a reported/diagnostic value.
    import numpy as np
    from pf_sintering.ch_exact_energy import mu_isotropic
    from pf_sintering.model import Sink, initialize_fields, reproject

    def build(gamma_gb_override):
        p = build_params(ModelConfig(
            preset="dev", geometry="sinusoidal_substrate", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
            contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
            sinusoid_wavelength=480e-9, sinusoid_amplitude=24e-9, interface_width_override=20e-9,
            use_aniso_surface=False, surface_mobility_scale=0.3, gamma_gb_override=gamma_gb_override,
        ))
        f, e1, e2, e3 = initialize_fields(p)
        e1, e2, e3 = reproject(f, e1, e2, e3)
        return p, f, e1, e2, e3

    s = Sink(threshold=math.inf)
    p_lo, f, e1, e2, e3 = build(0.517638)
    mu_lo = mu_isotropic(f, e1, e2, e3, s, p_lo)
    p_hi, _, _, _, _ = build(1.414214)
    mu_hi = mu_isotropic(f, e1, e2, e3, s, p_hi)  # identical f/e-fields, only gamma_gb differs
    assert not np.allclose(mu_lo, mu_hi)


def test_gb_mobility_m4_J_s_none_preserves_existing_behavior():
    p_default = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                          contact_orientation="short_plane", initial_overlap=20e-9,
                                          t_total=1e-6, eta_mobility_scale=1.0))
    p_explicit_none = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                                contact_orientation="short_plane", initial_overlap=20e-9,
                                                t_total=1e-6, eta_mobility_scale=1.0, gb_mobility_m4_J_s=None))
    assert math.isclose(p_default.M_eta, p_explicit_none.M_eta, rel_tol=1e-12)


def test_gb_mobility_m4_J_s_sets_m_eta_via_analytic_mapping():
    # Milestone 14G Section 13: gated on Section 12's three-way M_gb_eff
    # consistency check (see MILESTONE_14G report) -- maps the physical
    # M_GB [m^4/(J*s)] input through M_eta = pi^2*M_GB/(4*W_GB).
    M_gb = 3.5e-17
    p = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                  contact_orientation="short_plane", initial_overlap=20e-9,
                                  t_total=1e-6, gb_mobility_m4_J_s=M_gb))
    expected = m_eta_from_m_gb(M_gb, p.interface_width)
    assert math.isclose(p.M_eta, expected, rel_tol=1e-12)
    # overrides eta_mobility_scale entirely when set
    p_scaled = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                         contact_orientation="short_plane", initial_overlap=20e-9,
                                         t_total=1e-6, gb_mobility_m4_J_s=M_gb, eta_mobility_scale=99.0))
    assert math.isclose(p_scaled.M_eta, expected, rel_tol=1e-12)
