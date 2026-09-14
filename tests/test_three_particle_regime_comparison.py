import csv
import math
from pathlib import Path


OUT = Path("docs/three_particle/regime_comparison")


def rows(name):
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_cluster_quantiles_are_complete_ordered_and_obey_survival_identity():
    data = rows("cluster_quantile_competition.csv")
    assert len(data) == 30
    by_design = {}
    for row in data:
        by_design.setdefault(row["design_id"], []).append(row)
        assert math.isclose(float(row["Lambda"]), -math.log(float(row["survival"])), rel_tol=1e-12)
    assert len(by_design) == 10
    for path in by_design.values():
        path.sort(key=lambda row: float(row["event_probability"]))
        assert [float(row["event_probability"]) for row in path] == [0.1, 0.5, 0.9]
        assert all(float(a["x"]) < float(b["x"]) for a, b in zip(path, path[1:]))


def test_mobility_and_root_rate_scalings_reach_median_at_each_target():
    data = rows("mobility_and_root_sensitivity.csv")
    assert len(data) == 40
    for row in data:
        lam = float(row["baseline_Lambda"])
        multiplier = float(row["required_surface_mobility_multiplier"])
        assert math.isclose(lam / multiplier, math.log(2), rel_tol=1e-12)
        assert math.isclose(float(row["equivalent_root_rate_reduction_factor"]), multiplier)
        assert math.isclose(float(row["remaining_site_count_fraction"]), 1.0 / multiplier)


def test_bicrystal_completed_cycles_and_full_search_bounds_are_explicit():
    bic = rows("bicrystal_cycles.csv")
    assert [int(row["cycle"]) for row in bic] == [1, 2]
    assert all(float(row["loading_duration_s"]) > 0 for row in bic)
    assert all(0 < float(row["fraction_hazard_final_high_stress_portion"]) < 1 for row in bic)
    bounds = rows("maximum_thermodynamic_loading.csv")
    assert [float(row["target_center_loss"]) for row in bounds] == [0.01, 0.02, 0.05, 0.1]
    assert all(
        float(row["maximum_Delta_sigma_MPa"])
        >= float(row["fully_20pct_thermodynamic_maximum_Delta_sigma_MPa"])
        for row in bounds
    )
