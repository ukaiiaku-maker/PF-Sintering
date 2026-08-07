import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contact_geometry_audit import dequantized_totals, jump_events  # noqa: E402

from pf_sintering.model import ModelConfig, build_params, initialize_fields
from pf_sintering.operator_ledger import OPERATORS, run_ledger_trajectory_v2


def test_jump_events_and_dequantized_totals_are_consistent():
    p = build_params(ModelConfig(
        preset="dev", nx=96, ny=128, dx=5e-9, r2=80e-9,
        aspect_ratio=2.0, contact_orientation="short_plane",
        initial_overlap=20e-9, t_total=1e-6,
        coarsening_rate_scale=3.0, surface_mobility_scale=0.3,
    ))
    f0, e1_0, e2_0, e3_0 = initialize_fields(p)
    steps = run_ledger_trajectory_v2(p, f0, e1_0, e2_0, e3_0, target_dv2_frac=3e-4, max_steps=500)

    jumps = jump_events(steps, p.dx)
    assert len(jumps) >= 1
    for j in jumps:
        assert abs(j["value"]) >= 0.5 * p.dx
        assert j["operator"] in OPERATORS

    dequant = dequantized_totals(steps, p.dx)
    # For any operator/key with a detected jump, the dequantized total must
    # be strictly smaller in magnitude than the raw total (the jump removed).
    for j in jumps:
        op, key = j["operator"], j["key"]
        raw = dequant["raw"][op][key]
        dq = dequant["dequantized"][op][key]
        assert abs(dq) < abs(raw)
        assert dequant["n_excluded"][op][key] >= 1

    # Operators with no jumps must have identical raw and dequantized totals.
    jumped_ops_keys = {(j["operator"], j["key"]) for j in jumps}
    for op in OPERATORS:
        for key in ("L_contact_TJ", "L_GB_geom"):
            if (op, key) not in jumped_ops_keys:
                assert dequant["raw"][op][key] == dequant["dequantized"][op][key]
