import itertools
import math

import numpy as np

from pf_sintering.constrained_eta import (
    constrained_tangent_cone_eta_update,
    local_wc,
    tangent_cone_projected_velocity,
)
from pf_sintering.flux_closure import particle_volume_rate_decomposition
from pf_sintering.model import ModelConfig, Sink, build_params, initialize_fields, reproject


def _kkt_reference_velocity(eta_vals, v0_vals, active_tol=1e-12):
    """Brute-force reference: enumerate every subset of the active-boundary
    index set as the "blocked" (v_i=0) set, keep only feasible candidates
    (every active-boundary index not blocked must end up with v_i>=0), and
    return the one minimizing ||v-v0||^2 -- the exact KKT solution for this
    small (N<=3) convex QP, independent of tangent_cone_projected_velocity's
    own iterative construction."""
    N = len(eta_vals)
    active_idx = [i for i in range(N) if eta_vals[i] <= active_tol]
    best_v, best_obj = None, math.inf
    for r in range(len(active_idx) + 1):
        for B in itertools.combinations(active_idx, r):
            B = set(B)
            free = [i for i in range(N) if i not in B]
            if not free:
                continue
            lam = sum(v0_vals[i] for i in free) / len(free)
            v = [0.0] * N
            for i in free:
                v[i] = v0_vals[i] - lam
            feasible = all(v[i] >= -1e-9 for i in active_idx)
            if feasible:
                obj = sum((v[i] - v0_vals[i]) ** 2 for i in range(N))
                if obj < best_obj:
                    best_obj, best_v = obj, v
    return best_v


def _check_against_reference(eta_vals, v0_vals, active_tol=1e-12):
    eta_arrays = [np.array([[e]]) for e in eta_vals]
    v0_arrays = [np.array([[v]]) for v in v0_vals]
    v_fast = tangent_cone_projected_velocity(eta_arrays, v0_arrays, active_tol=active_tol)
    v_fast_scalar = [float(v[0, 0]) for v in v_fast]
    v_ref = _kkt_reference_velocity(eta_vals, v0_vals, active_tol=active_tol)
    assert v_ref is not None, "reference solver found no feasible point"
    for a, b in zip(v_fast_scalar, v_ref):
        assert math.isclose(a, b, abs_tol=1e-9), (v_fast_scalar, v_ref)
    return v_fast_scalar


def test_interior_point_two_grains():
    # No active constraints: pure sum-zero projection, v_i = v0_i - mean(v0).
    v = _check_against_reference([0.3, 0.4], [1.0, -0.2])
    assert math.isclose(sum(v), 0.0, abs_tol=1e-9)


def test_one_active_boundary_two_grains():
    # eta_1=0 with an unconstrained velocity that would push it further
    # negative -- must be blocked at v_1=0, forcing v_2=0 too (N=2 has
    # nowhere else for the "blocked" flux to go).
    v = _check_against_reference([0.0, 0.5], [-1.0, 1.0])
    assert math.isclose(v[0], 0.0, abs_tol=1e-9)
    assert math.isclose(v[1], 0.0, abs_tol=1e-9)


def test_one_active_boundary_three_grains_redistributes_among_free():
    # eta_1=0, blocked; grains 2 and 3 remain free and continue to
    # redistribute between themselves.
    v = _check_against_reference([0.0, 0.3, 0.3], [-1.0, 1.0, -0.5])
    assert math.isclose(v[0], 0.0, abs_tol=1e-9)
    assert math.isclose(v[1] + v[2], 0.0, abs_tol=1e-9)
    assert abs(v[1]) > 1e-6  # grains 2/3 still actively exchanging


def test_two_active_boundaries_three_grains():
    # eta_1=eta_2=0, grain 3 dominant; both active-boundary velocities are
    # driven further negative by v0 -- both must block, freezing everything.
    v = _check_against_reference([0.0, 0.0, 1.0], [-1.0, -0.8, 1.8])
    assert all(math.isclose(vi, 0.0, abs_tol=1e-9) for vi in v)


def test_two_active_boundaries_one_recovers():
    # eta_1=eta_2=0; grain 1's unconstrained velocity is actually positive
    # (growing from zero, allowed) while grain 2's is negative (blocked).
    v = _check_against_reference([0.0, 0.0, 1.0], [0.5, -1.0, 0.5])
    assert v[1] == 0.0 or math.isclose(v[1], 0.0, abs_tol=1e-9)


def test_equal_thermodynamic_forces_gives_zero_velocity():
    # Equal v0 for every grain -> mean-subtraction gives exactly zero
    # velocity for all (no ownership transfer when forces are balanced).
    v = _check_against_reference([0.4, 0.3, 0.2], [0.7, 0.7, 0.7])
    assert all(math.isclose(vi, 0.0, abs_tol=1e-12) for vi in v)


def test_strongly_unequal_forces_interior():
    v = _check_against_reference([0.5, 0.3, 0.1], [10.0, -3.0, -100.0])
    assert math.isclose(sum(v), 0.0, abs_tol=1e-9)


def _sinusoidal_state(seed=0):
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


def test_tangent_cone_update_conserves_f_exactly():
    p, s, f, e1, e2, e3 = _sinusoidal_state()
    e1n, e2n, e3n, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p)
    residual = np.abs((e1n + e2n) - np.clip(f, 0.0, 1.0))
    assert np.max(residual) < 1e-10


def test_tangent_cone_update_stays_nonnegative():
    p, s, f, e1, e2, e3 = _sinusoidal_state()
    e1n, e2n, e3n, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p)
    assert np.min(e1n) >= -1e-12
    assert np.min(e2n) >= -1e-12


def test_tangent_cone_safety_correction_is_negligible():
    # Section 11: the SAFETY-NET clip (distinct from the expected
    # f-tracking rescale, see module docstring) must be roundoff-scale
    # relative to the variational step actually taken, at the production
    # timestep -- this is the direct replacement for Milestone 12B's
    # ~0.84 projection/variational ratio.
    p, s, f, e1, e2, e3 = _sinusoidal_state()
    e1n, e2n, e3n, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p)
    assert diag["safety_fraction"] < 1e-6


def test_tangent_cone_ownership_transfer_allowed_between_grains():
    p = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                  contact_orientation="short_plane", initial_overlap=20e-9,
                                  t_total=1e-6, eta_mobility_scale=1.0))
    s = Sink(threshold=math.inf)
    y = (np.arange(1, p.Ny + 1)) * p.dx
    Y = np.tile(y[:, None], (1, p.Nx))
    f = np.ones((p.Ny, p.Nx)) * 0.9
    e1 = 0.45 + 0.05 * np.sin(2 * math.pi * Y / (y.max() / 4))
    e2 = f - e1
    e3 = np.zeros_like(f)
    e1n, e2n, e3n, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p)
    assert np.max(np.abs(e1n - e1)) > 1e-9
    assert diag["variational_change"] > 0
    assert math.isclose(float((e1n + e2n).sum()), float(f.sum()), rel_tol=1e-6)


def test_tangent_cone_zero_eta_grain_never_goes_negative_under_strong_forcing():
    # Construct a point already at the eta_1=0 boundary with a strong
    # thermodynamic force that would (unconstrained) drive it further
    # negative; the tangent-cone step must keep it pinned at exactly 0,
    # not overshoot and require a large clip.
    p = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                  contact_orientation="short_plane", initial_overlap=20e-9,
                                  t_total=1e-6, eta_mobility_scale=1.0))
    s = Sink(threshold=math.inf)
    f = np.ones((p.Ny, p.Nx)) * 0.9
    e1 = np.zeros_like(f)
    e2 = f.copy()
    e3 = np.zeros_like(f)
    e1n, e2n, e3n, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f, s, p)
    assert np.allclose(e1n, 0.0, atol=1e-10)
    assert np.allclose(e2n, f, atol=1e-10)


def test_tangent_cone_void_region_rescale_matches_ownership_convention_at_zero_mobility():
    # Regression test (Milestone 13 Section 13): with M_eta=0 no eta motion
    # is physically possible, so particle_volume_rate_decomposition's
    # eta-migration bucket must be exactly zero (to roundoff) on a REAL
    # sharp-interface state -- not just on the smooth synthetic fields the
    # other tests here use. An earlier version of the f-tracking rescale's
    # "no prior ownership" fallback assigned an even f/N split at every
    # such point, including deep in the diffuse tail where model.reproject
    # itself deliberately zeroes eta (fb<=0.005, its "void" convention);
    # since f_weighted_ownership_volumes reads ~0 ownership there too, that
    # mismatch produced a spurious eta-migration signal (measured at ~50x
    # the genuine transport signal on this exact state before the fix).
    import dataclasses
    p0 = build_params(ModelConfig(preset="dev", dx=5e-9, r2=80e-9, aspect_ratio=2.0,
                                   contact_orientation="short_plane", initial_overlap=20e-9,
                                   t_total=1e-6))
    p = dataclasses.replace(p0, M_eta=0.0)
    f, e1, e2, e3 = initialize_fields(p)
    e1, e2, e3 = reproject(f, e1, e2, e3)
    s = Sink(threshold=math.inf)

    f_old, e1_old, e2_old, e3_old = f.copy(), e1.copy(), e2.copy(), e3.copy()
    # a small synthetic f perturbation stands in for one transport step,
    # keeping this test independent of surface_transport's own machinery
    rng = np.random.default_rng(0)
    f_new = np.clip(f + 1e-4 * rng.normal(size=f.shape), 0.0, 1.0)
    e1n, e2n, e3n, diag = constrained_tangent_cone_eta_update(e1, e2, e3, f_new, s, p)

    r = particle_volume_rate_decomposition(f_old, f_new, e1_old, e2_old, e3_old, e1n, e2n, e3n, p.dt, p.dx)
    assert abs(r["dV2_dt_eta_migration"]) < 1e-6 * max(abs(r["dV2_dt_transport"]), 1e-300)
