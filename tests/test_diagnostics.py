import math

import numpy as np

from pf_sintering.diagnostics import contact_area, sample, summarize, wall_x0
from pf_sintering.model import ModelConfig, Sink, build_params, compute_stress, initialize_fields


def _state(**overrides):
    p = build_params(ModelConfig(preset="dev", nx=96, ny=128, dx=5e-9, r2=80e-9, t_total=1e-6, **overrides))
    f, e1, e2, e3 = initialize_fields(p)
    s = Sink(threshold=1.0)
    st, stop, reason = compute_stress(f, e1, e2, e3, s, p)
    assert not stop, reason
    return p, f, e1, e2, e3, s, st


def test_wall_x0_is_fixed_geometric_constant():
    p, *_ = _state()
    x0 = wall_x0(p)
    assert math.isfinite(x0)
    assert np.isclose(x0, (p.substrate_wall_frac - 0.5) * p.Nx * p.dx)


def test_sample_reports_finite_required_fields():
    p, f, e1, e2, e3, s, st = _state()
    v20 = float(e2.sum() * p.dx * p.dx)
    d = sample(f, e1, e2, e3, s, st, p, step=1, time_s=p.dt, v20=v20)
    for k in ("V2", "V2_ratio", "separation_m", "x_neck_m", "A_GB_m2", "sigma_Pa",
              "sigma_lt_Pa", "sigma_curv_Pa", "E_surf_J", "E_gb_J", "G_interface_J",
              "sink_active", "hazard", "hazard_threshold", "quota_progress"):
        assert math.isfinite(d[k]), f"{k} not finite: {d[k]}"
    assert d["V2_ratio"] == 1.0
    assert d["sink_active"] == 0
    assert np.isclose(d["G_interface_J"], d["E_surf_J"] + d["E_gb_J"])
    assert np.isclose(d["A_GB_m2"], contact_area(e1, e2, p))


def test_summarize_reports_signed_deltas_and_slopes():
    p, f, e1, e2, e3, s, st = _state()
    v20 = float(e2.sum() * p.dx * p.dx)
    s0 = sample(f, e1, e2, e3, s, st, p, step=1, time_s=p.dt, v20=v20)
    # Synthesize a second sample representing coarsening + neck narrowing.
    s1 = dict(s0)
    s1["step"] = 2
    s1["V2"] = s0["V2"] * 0.999
    s1["x_neck_m"] = s0["x_neck_m"] * 0.98
    s1["sigma_Pa"] = s0["sigma_Pa"] * 1.05
    s1["G_interface_J"] = s0["G_interface_J"] * 0.999
    summary = summarize([s0, s1])
    assert summary["d_V2"] < 0
    assert summary["d_x_neck_m"] < 0
    assert summary["d_sigma_Pa"] > 0
    assert summary["d_G_interface_J"] < 0
    # Both V2 and x_neck decrease together, so the sign of dx_neck/dV2 is positive.
    assert summary["dx_neck_dV2"] > 0
    assert summary["dsigma_dV2"] < 0
