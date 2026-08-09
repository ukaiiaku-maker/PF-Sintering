import math

import numpy as np

from pf_sintering.constrained_eta import (
    GB_ENERGY_CALIBRATION_FACTOR,
    constrained_variational_eta_update,
    f_weighted_ownership_volumes,
    gamma_gb_target_to_declared,
    local_wc,
    structural_thermodynamic_force,
)
from pf_sintering.model import ModelConfig, Sink, build_params, effective_gamma, lap9


def _state(seed=0):
    p = build_params(ModelConfig(
        preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=20e-9, t_total=1e-6, eta_mobility_scale=1.0,
    ))
    s = Sink(threshold=math.inf)
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter
    f = np.clip(0.5 + 0.3 * gaussian_filter(rng.normal(size=(p.Ny, p.Nx)), sigma=3), 0.0, 1.0)
    w1 = np.clip(0.5 + 0.3 * gaussian_filter(rng.normal(size=(p.Ny, p.Nx)), sigma=3), 0.0, 1.0)
    e1 = f * w1
    e2 = f * (1 - w1)
    e3 = np.zeros_like(f)
    return p, s, f, e1, e2, e3


def test_zero_sum_constraint_preserved_by_variational_update_before_projection():
    # sum_i d(eta_i)/dt = 0 is an exact algebraic identity -- check the
    # variational deltas alone (before reproject) sum to (near) zero.
    p, s, f, e1, e2, e3 = _state()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, s, p)
    d1, d2 = diag["d1"], diag["d2"]
    assert np.max(np.abs(d1 + d2)) < 1e-6 * max(np.max(np.abs(d1)), np.max(np.abs(d2)))


def test_total_f_unaffected_by_eta_update():
    p, s, f, e1, e2, e3 = _state()
    f_before = f.copy()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, s, p)
    assert np.array_equal(f, f_before)  # eta update never touches f


def test_sum_eta_stays_close_to_f_after_projection():
    p, s, f, e1, e2, e3 = _state()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, s, p)
    residual = np.abs((e1n + e2n) - f)
    assert np.max(residual) < 0.05  # local projection keeps this small


def test_projection_correction_small_relative_to_variational_change():
    p, s, f, e1, e2, e3 = _state()
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, s, p)
    # the projection should be a minor correction, not comparable to the
    # variational update itself (Section 15's explicit requirement)
    assert diag["projection_fraction"] < 0.5


def test_ownership_transfer_allowed_between_grains():
    # Construct a case where e1's own thermodynamic force clearly differs
    # from e2's (e1 rough/high-curvature, e2 smooth) -- ownership should
    # shift without changing total f.
    p = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                  contact_orientation="short_plane", initial_overlap=20e-9,
                                  t_total=1e-6, eta_mobility_scale=1.0))
    s = Sink(threshold=math.inf)
    x = (np.arange(1, p.Nx + 1)) * p.dx
    y = (np.arange(1, p.Ny + 1)) * p.dx
    X, Y = np.meshgrid(x, y)
    f = np.ones((p.Ny, p.Nx)) * 0.9
    e1 = 0.45 + 0.05 * np.sin(2 * math.pi * Y / (y.max() / 4))  # rough (high curvature)
    e2 = f - e1  # smooth complement
    e3 = np.zeros_like(f)
    e1n, e2n, e3n, diag = constrained_variational_eta_update(e1, e2, e3, f, s, p)
    assert np.max(np.abs(e1n - e1)) > 1e-9  # ownership actually moves
    assert diag["variational_change"] > 0
    assert math.isclose(float((e1n + e2n).sum()), float(f.sum()), rel_tol=1e-6)


def test_f_weighted_ownership_volumes_sum_to_v2_f_total():
    p, s, f, e1, e2, e3 = _state()
    V1, V2, V3 = f_weighted_ownership_volumes(f, e1, e2, e3, p.dx)
    V_f_total = float(f.sum()) * p.dx * p.dx
    assert math.isclose(V1 + V2 + V3, V_f_total, rel_tol=1e-6)


def test_local_wc_matches_evolve_f_construction():
    # local_wc must reproduce model.evolve_f's own Wc exactly (same
    # gamma_gb_ref / effective_gamma(s, p) branch, same 36/W scaling).
    p, s, f, e1, e2, e3 = _state()
    Wc = local_wc(f, e1, e2, e3, s, p)
    fb = np.clip(f, 0.0, 1.0)
    es = [np.clip(e, 0.0, fb) for e in (e1, e2, e3)]
    pair = np.maximum(0.0, es[0] * es[1])
    gl_expected = np.full_like(f, p.gamma_gb_ref)
    mask = pair > 1e-20
    if np.any(mask):
        gl_expected[mask] = effective_gamma(s, p)
    Wc_expected = 36.0 * gl_expected / p.interface_width
    assert np.allclose(Wc, Wc_expected)


def test_structural_force_gradient_only_where_coupling_vanishes():
    # Where f == 0, the coupling term 2*Wc*eta_i*(f^2/2-f) vanishes and g_i
    # reduces to the pure gradient term -- a necessary, but not sufficient,
    # special case of the complete formula (see
    # test_structural_force_matches_full_free_energy_derivative below for
    # the general machine-precision check).
    p, s, f, e1, e2, e3 = _state()
    f_zero = np.zeros_like(f)
    Wc = local_wc(f_zero, e1, e2, e3, s, p)
    g1 = structural_thermodynamic_force(e1, f_zero, Wc, p.dx, p.k_eta, bc_x="periodic", bc_y="reflecting")
    expected = -p.k_eta * lap9(e1, p.dx)
    assert np.allclose(g1, expected)


def test_structural_force_matches_full_free_energy_derivative():
    # Milestone 12B Section 5: g_i = delta F/delta eta_i for the COMPLETE
    # free energy F = integral[(W_f/2)f^2(1-f)^2 + Wc*eta2*(f^2/2-f)
    # + (k_eta/2)*sum_i|grad(eta_i)|^2]dV, verified via a finite-difference
    # directional-derivative test (same methodology as
    # tests/test_ch_exact_energy.py). Uses the lap9-self-adjoint
    # construction -0.5*k_eta*eta_i*lap9(eta_i) for the gradient-energy
    # term (a naive |grad|^2 central-difference sum is NOT the discrete
    # adjoint of lap9 and fails this test by ~47%; see MILESTONE_12B report).
    p, s, f, e1, e2, e3 = _state()

    def free_energy(f_, e1_, e2_, e3_):
        Wc = local_wc(f_, e1_, e2_, e3_, s, p)
        eta2 = e1_ * e1_ + e2_ * e2_ + e3_ * e3_
        e_bulk = 0.5 * p.W_f * f_ * f_ * (1 - f_) ** 2 + Wc * eta2 * (0.5 * f_ * f_ - f_)
        e_grad_eta = -0.5 * p.k_eta * (e1_ * lap9(e1_, p.dx) + e2_ * lap9(e2_, p.dx))
        return float(np.sum(e_bulk + e_grad_eta)) * p.dx * p.dx

    Wc = local_wc(f, e1, e2, e3, s, p)
    g1 = structural_thermodynamic_force(e1, f, Wc, p.dx, p.k_eta, bc_x="periodic", bc_y="reflecting")

    rng = np.random.default_rng(1)
    q = rng.normal(size=e1.shape)
    rhs = (p.dx * p.dx) * float(np.sum(g1 * q))

    rel_errs = []
    for eps in (1e-2, 1e-3, 1e-4):
        f_plus = free_energy(f, e1 + eps * q, e2, e3)
        f_minus = free_energy(f, e1 - eps * q, e2, e3)
        lhs = (f_plus - f_minus) / (2 * eps)
        rel_errs.append(abs(lhs - rhs) / abs(rhs))

    assert min(rel_errs) < 1e-8


def test_gb_energy_calibration_factor_matches_analytic_isolated_gb():
    # Milestone 14E Section 4 / 14F Section 17: declared gamma_gb and the
    # eta profile's actual implemented excess free energy differ by an
    # exact, analytically-derived factor (18 from the bulk term + 1 from
    # the gradient term, at f=1, isolated planar GB, natural tanh
    # profile) -- verified here via direct numerical integration of the
    # analytic profile, independent of the simulation machinery.
    W = 20e-9
    gamma_gb_declared = 1.0
    k_eta = 3 * gamma_gb_declared * W
    Wc = 36 * gamma_gb_declared / W

    x = np.linspace(-500e-9, 500e-9, 200001)
    eta2 = 0.5 * (1 + np.tanh(x / W))
    eta1 = 1 - eta2
    bulk = Wc * (eta1 ** 2 + eta2 ** 2) * (0.5 - 1.0)
    ge1 = np.gradient(eta1, x)
    ge2 = np.gradient(eta2, x)
    grad_e = 0.5 * k_eta * (ge1 ** 2 + ge2 ** 2)
    background = -0.5 * Wc
    gamma_gb_implemented = np.trapezoid((bulk + grad_e) - background, x)

    assert math.isclose(gamma_gb_implemented, GB_ENERGY_CALIBRATION_FACTOR * gamma_gb_declared, rel_tol=1e-6)


def test_gamma_gb_target_to_declared_round_trips():
    for target in (0.5, 1.0, 1.6, 3.0):
        declared = gamma_gb_target_to_declared(target)
        assert math.isclose(declared * GB_ENERGY_CALIBRATION_FACTOR, target, rel_tol=1e-12)
