import numpy as np

from pf_sintering.model import ModelConfig, build_params, center, contact_width, initialize_fields
from pf_sintering.diagnostics import wall_x0
from pf_sintering.static_neck_geometry import build_neck_state


def _reference():
    p = build_params(ModelConfig(
        preset="dev", nx=96, ny=128, dx=5e-9, r2=80e-9,
        aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=8e-9, t_total=1e-6,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    x0, _ = contact_width(e1, e2, p)
    v20 = float(e2.sum() * p.dx * p.dx)
    L0 = float(center(e2, p) - wall_x0(p))
    return p, x0, v20, L0


def test_family_hits_V2_and_L_targets_tightly_while_varying_x_neck():
    # Fractions kept within the reliably-convergent region for this reference
    # geometry. Very wide necks (roughly frac > 0.8 here -- Milestone 15B's
    # exact eta1+eta2=f initializer, gb_signed_distance.py, fixed a spurious
    # far-field grain-2 ownership bug that had inflated contact_width's x0
    # estimate ~5.5x; the true, much narrower x0 shrinks the range of
    # x_neck/x0 fractions the (cx, Rfar) family below can still jointly
    # satisfy alongside this reference's V2/L) can make (V2, L, x_neck)
    # jointly infeasible for the flat-collar/circular-cap construction --
    # see static_neck_geometry.py's module docstring and
    # MILESTONE_1_2_REPORT.md-successor findings: a sufficiently wide flat
    # collar at fixed x_neck can already exceed v2_target on its own before
    # Rfar is even searched, and build_neck_state then falls back to the
    # closest feasible sample rather than hitting L exactly.
    p, x0, v20, L0 = _reference()
    achieved_x = []
    for frac in (0.3, 0.5, 0.65, 0.8):
        st = build_neck_state(p, x_neck_target=frac * x0, v2_target=v20, L_target=L0)
        assert abs(st.V2_achieved - v20) / v20 < 1e-6
        assert abs(st.L_achieved - L0) / L0 < 1e-6
        assert np.all(np.isfinite(st.f)) and np.all(np.isfinite(st.e2))
        assert st.f.min() >= -1e-9 and st.f.max() <= 1 + 1e-9
        achieved_x.append(st.x_neck_achieved)

    # x_neck should track the target monotonically even if not hit exactly.
    assert all(achieved_x[i] < achieved_x[i + 1] for i in range(len(achieved_x) - 1))
    for target_frac, ach in zip((0.3, 0.5, 0.65, 0.8), achieved_x):
        assert abs(ach - target_frac * x0) / x0 < 0.1


def test_infeasible_family_member_raises_or_falls_back_without_crashing():
    # A very wide neck target for this reference geometry is expected to be
    # at or past the edge of feasibility; the solver must not silently
    # return a badly-off state -- it should either raise or clearly land far
    # from the target (caller-checkable via *_achieved vs *_target).
    p, x0, v20, L0 = _reference()
    st = build_neck_state(p, x_neck_target=1.5 * x0, v2_target=v20, L_target=L0)
    assert np.all(np.isfinite(st.f))
    # Document (not silently hide) that this regime is poorly converged.
    L_relerr = abs(st.L_achieved - L0) / L0
    assert L_relerr > 1e-3, "expected this regime to be at/past the feasibility edge"
