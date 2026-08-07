import math

import numpy as np

from pf_sintering.constrained_relaxation import relax_at_fixed_V2_L, summarize_constrained_state
from pf_sintering.diagnostics import wall_x0
from pf_sintering.model import ModelConfig, Sink, build_params, center
from pf_sintering.separation_constraint import enforce_separation, enforce_V2_and_L
from pf_sintering.static_neck_geometry import build_neck_state
from pf_sintering.structural_projection import eta_masses


def _reference():
    p = build_params(ModelConfig(
        preset="dev", nx=96, ny=128, dx=5e-9, r2=80e-9,
        aspect_ratio=2.0, contact_orientation="short_plane", t_total=1e-6,
    ))
    s = Sink(threshold=1.0)
    v20 = math.pi * (80e-9) ** 2
    L0 = 100e-9
    return p, s, v20, L0


def test_enforce_separation_restores_target_without_touching_a_converged_state():
    p, s, v20, L0 = _reference()
    st = build_neck_state(p, 30e-9, v20, L0)
    w0 = wall_x0(p)
    f, e1, e2, e3, corr = enforce_separation(st.f.copy(), st.e1.copy(), st.e2.copy(), st.e3.copy(), p, w0, L0, tol=1e-13)
    # Already within tolerance (build_neck_state hits L to ~1e-12 relative): should be a no-op.
    assert not corr.applied
    assert np.allclose(f, st.f) and np.allclose(e2, st.e2)


def test_enforce_separation_corrects_an_induced_offset():
    p, s, v20, L0 = _reference()
    st = build_neck_state(p, 30e-9, v20, L0)
    w0 = wall_x0(p)
    # Induce a deliberate 0.3nm offset, then correct back to L0.
    f, e1, e2, e3, _ = enforce_separation(st.f.copy(), st.e1.copy(), st.e2.copy(), st.e3.copy(), p, w0, L0 + 0.3e-9, tol=1e-13)
    L_shifted = center(e2, p) - w0
    assert abs(L_shifted - (L0 + 0.3e-9)) < 1e-12

    f, e1, e2, e3, corr = enforce_separation(f, e1, e2, e3, p, w0, L0, tol=1e-13)
    assert corr.applied
    L_final = center(e2, p) - w0
    assert abs(L_final - L0) < 1e-12


def test_enforce_V2_and_L_holds_both_constraints_over_repeated_calls():
    # Matches how relax_at_fixed_V2_L actually uses this function: v_targets
    # is captured once and the same target is passed to every call across
    # many steps. Simulate a few steps' worth of small drift (each much
    # smaller than the single large offset used in the induced-offset test
    # above, matching the ~1e-4 nm/step scale observed in real relaxation
    # runs) and confirm both constraints stay tightly held across the
    # sequence, not just within a single call.
    p, s, v20, L0 = _reference()
    st = build_neck_state(p, 30e-9, v20, L0)
    w0 = wall_x0(p)
    f, e1, e2, e3 = st.f.copy(), st.e1.copy(), st.e2.copy(), st.e3.copy()
    v_targets = eta_masses(e1, e2, e3, p.use_eta3)
    tol_L = 1e-4 * p.dx

    for i in range(10):
        drift = 2e-13 * (1 if i % 2 == 0 else -1)
        f, e1, e2, e3, _ = enforce_separation(f, e1, e2, e3, p, w0, L0 + drift, tol=1e-14)
        f, e1, e2, e3, rep = enforce_V2_and_L(f, e1, e2, e3, p, w0, L0, v_targets, tol_L)

    L_now = center(e2, p) - w0
    v2_now = float(e2.sum() * p.dx * p.dx)
    assert abs(L_now - L0) <= tol_L
    assert abs(v2_now - v20) / v20 < 1e-5


def test_short_constrained_relaxation_holds_V2_and_L_and_produces_finite_diagnostics():
    p, s, v20, L0 = _reference()
    res = relax_at_fixed_V2_L(p, s, v20, L0, x_neck_seed=30e-9, n_steps=200, check_every=20, converge_window=5, converge_rtol=1e-12)
    assert res.steps_run == 200  # should not spuriously "converge" this early
    assert abs(res.dL_final) < 1e-4 * p.dx
    assert abs(res.dV2_final) < 1e-8

    summ = summarize_constrained_state(res.f, res.e1, res.e2, res.e3, p, s)
    assert summ["tj_resolved"]
    assert summ["n_tj_resolved"] == 2
    for k in ("x_neck_m", "A_GB_m2", "E_surf_J", "E_gb_J", "G_interface_J",
              "psi_top_deg", "psi_bottom_deg", "F_TJ_top_mag", "F_TJ_bottom_mag"):
        assert math.isfinite(summ[k]), f"{k} not finite: {summ[k]}"
    assert abs(summ["V2"] - v20) / v20 < 1e-8
