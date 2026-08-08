import math

from pf_sintering.model import ModelConfig, build_params


def _cfg(dx_nm, **overrides):
    return ModelConfig(
        preset="dev", dx=dx_nm * 1e-9, r2=80e-9, aspect_ratio=2.0,
        contact_orientation="short_plane", initial_overlap=20e-9, t_total=1e-6,
        coarsening_rate_scale=3.0, surface_mobility_scale=0.3, eta_mobility_scale=1.0,
        **overrides,
    )


def test_interface_width_override_default_is_none_and_preserves_production_w():
    p = build_params(_cfg(5.0))
    assert p.interface_width == p.dx * 4.0  # interface_cells default


def test_interface_width_override_sets_fixed_physical_w_independent_of_dx():
    for dx_nm in (5.0, 2.5, 1.25):
        p = build_params(_cfg(dx_nm, interface_width_override=20e-9))
        assert p.interface_width == 20e-9


def test_eta_diffusivity_fixed_physical_default_false_preserves_production_m_eta():
    p_default = build_params(_cfg(5.0))
    p_explicit_false = build_params(_cfg(5.0, eta_diffusivity_fixed_physical=False))
    assert p_default.M_eta == p_explicit_false.M_eta


def test_default_config_m_eta_times_k_eta_scales_as_inverse_dx_squared():
    # Documents the audited production behavior (Milestone 8 Section 4):
    # M_eta*k_eta (the physical structural-relaxation diffusivity) is NOT
    # dx-independent in the default relative-interface-width configuration --
    # it grows by exactly 4x each time dx halves.
    p5 = build_params(_cfg(5.0))
    p25 = build_params(_cfg(2.5))
    ratio = (p25.M_eta * p25.k_eta) / (p5.M_eta * p5.k_eta)
    assert math.isclose(ratio, 4.0, rel_tol=1e-9)


def test_eta_diffusivity_fixed_physical_makes_m_eta_k_eta_dx_independent():
    # With BOTH a fixed physical interface width (so k_eta doesn't drift with
    # dx either) and eta_diffusivity_fixed_physical=True, M_eta*k_eta must be
    # the same fixed physical diffusivity at every resolution (Milestone 8
    # Sections 3-4).
    vals = []
    for dx_nm in (5.0, 2.5, 1.25):
        p = build_params(_cfg(
            dx_nm, interface_width_override=20e-9, eta_diffusivity_fixed_physical=True,
        ))
        vals.append(p.M_eta * p.k_eta)
    assert math.isclose(vals[0], vals[1], rel_tol=1e-9)
    assert math.isclose(vals[1], vals[2], rel_tol=1e-9)


def test_fixed_w_and_fixed_eta_diffusivity_also_leave_m_f_k_f_dx_independent():
    # Sanity check: M_f*k_f was already dx-independent in the default config
    # (Milestone 7 Section 9); confirm the new overrides don't disturb that.
    p5 = build_params(_cfg(5.0, interface_width_override=20e-9, eta_diffusivity_fixed_physical=True))
    p25 = build_params(_cfg(2.5, interface_width_override=20e-9, eta_diffusivity_fixed_physical=True))
    assert math.isclose(p5.M_f * p5.k_f, p25.M_f * p25.k_f, rel_tol=1e-12)
