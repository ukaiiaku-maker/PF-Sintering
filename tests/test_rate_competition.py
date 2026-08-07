import math

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.rate_competition import rich_sample, run_sinkoff_trajectory
from pf_sintering.model import Sink, compute_stress


def _geometry(**overrides):
    p = build_params(ModelConfig(
        preset="dev", nx=96, ny=128, dx=5e-9, r2=80e-9,
        aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=5e-9, t_total=1e-6, **overrides,
    ))
    f, e1, e2, e3 = initialize_fields(p)
    return p, f, e1, e2, e3


def test_rich_sample_extends_base_sample_with_tj_and_curvature_fields():
    p, f, e1, e2, e3 = _geometry()
    s = Sink(threshold=1.0)
    v20 = float(e2.sum() * p.dx * p.dx)
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    assert not stop, reason
    row = rich_sample(f, e1, e2, e3, s, st, p, step=0, time_s=0.0, v20=v20)

    # base diagnostics.sample() fields still present
    for k in ("V2", "x_neck_m", "sigma_Pa", "E_surf_J", "G_interface_J"):
        assert k in row and math.isfinite(row[k])

    # new fields present and finite when resolved
    assert row["tj_resolved"]
    assert row["n_tj_resolved"] == 2
    for name in ("top", "bottom"):
        assert math.isfinite(row[f"psi_deg_{name}"])
        assert math.isfinite(row[f"F_TJ_mag_{name}"])
        assert row[f"xi_s1_{name}"] is not None
        assert row[f"xi_gb_{name}"] is not None
    assert math.isfinite(row["kappa_top_1pm"])
    assert math.isfinite(row["F_drive"])


def test_sinkoff_trajectory_never_activates_sink_and_holds_mass():
    p, f0, e1_0, e2_0, e3_0 = _geometry()
    samples = run_sinkoff_trajectory(
        p, f0, e1_0, e2_0, e3_0, target_dv2_frac=5e-5, max_steps=2000, sample_every=20,
    )
    assert len(samples) >= 2
    for row in samples:
        assert row["sink_active"] == 0
        assert row["strain"] == 0.0
    last = samples[-1]
    assert abs(last["V2_ratio"] - 1.0) > 0  # some coarsening actually occurred
    assert abs(last["V2_ratio"] - 1.0) <= 1e-3  # but not wildly beyond the target


def test_rate_controls_change_the_coarsening_trajectory():
    p_slow, f0, e1_0, e2_0, e3_0 = _geometry(coarsening_rate_scale=0.3)
    p_fast, *_ = _geometry(coarsening_rate_scale=3.0)
    n = 200
    s_slow = run_sinkoff_trajectory(p_slow, f0, e1_0, e2_0, e3_0, target_dv2_frac=1.0, max_steps=n, sample_every=n)
    s_fast = run_sinkoff_trajectory(p_fast, f0, e1_0, e2_0, e3_0, target_dv2_frac=1.0, max_steps=n, sample_every=n)
    dv2_slow = abs(s_slow[-1]["V2_ratio"] - 1.0)
    dv2_fast = abs(s_fast[-1]["V2_ratio"] - 1.0)
    assert dv2_fast > dv2_slow
