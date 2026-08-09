import math

from pf_sintering.model import ModelConfig, _gb_energy, build_params


def test_gamma_gb_override_none_preserves_existing_behavior():
    p_default = build_params(ModelConfig(preset="dev"))
    p_explicit_none = build_params(ModelConfig(preset="dev", gamma_gb_override=None))
    expected = _gb_energy(30.0)  # default theta_mis_deg
    assert math.isclose(p_default.gamma_gb, expected, rel_tol=1e-12)
    assert math.isclose(p_default.gamma_gb, p_explicit_none.gamma_gb, rel_tol=1e-12)
    assert math.isclose(p_default.gamma_gb, p_default.gamma_gb_ref, rel_tol=1e-12)


def test_gamma_gb_override_sets_gamma_gb_and_rederives_dependents():
    # Milestone 14 Section 8: a single config path re-derives EVERY
    # gamma_gb-dependent coefficient consistently (k_eta=3*gamma_gb*W,
    # W_cpl_f=36*gamma_gb/W), not just a diagnostic-only gamma_gb_ref.
    gamma_s = 1.0
    for psi_eq_deg in (90.0, 120.0, 150.0):
        psi_eq = math.radians(psi_eq_deg)
        gamma_gb = 2.0 * gamma_s * math.cos(psi_eq / 2.0)
        p = build_params(ModelConfig(preset="dev", gamma_gb_override=gamma_gb))
        assert math.isclose(p.gamma_gb, gamma_gb, rel_tol=1e-12)
        assert math.isclose(p.gamma_gb_ref, gamma_gb, rel_tol=1e-12)
        assert math.isclose(p.k_eta, 3.0 * gamma_gb * p.interface_width, rel_tol=1e-12)
        assert math.isclose(p.W_cpl_f, 36.0 * gamma_gb / p.interface_width, rel_tol=1e-12)
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
