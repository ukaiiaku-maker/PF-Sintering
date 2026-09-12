import numpy as np

from scripts.three_particle_production_source import build


def test_production_source_preserves_full_field_and_partition(tmp_path):
    source = tmp_path / "source.npz"
    out = tmp_path / "production.npz"
    report = tmp_path / "production.json"
    # Exercise the real selected source through an isolated copy so geometry
    # identity, bounds, topology, and four-field closure are all checked.
    selected = (
        "runs/three_particle_volume_loading_065_symmetry_released/"
        "loss_2pct.npz")
    with np.load(selected) as data:
        np.savez_compressed(source, **{key: data[key] for key in data.files})
        expected = data["f"].copy()
    payload = build(source, out, report)
    with np.load(out) as data:
        np.testing.assert_array_equal(data["f"], expected)
        np.testing.assert_array_equal(data["fields"][0], expected)
        np.testing.assert_allclose(data["fields"][1:].sum(axis=0), expected,
                                   rtol=0, atol=5e-15)
        assert not bool(data["symmetry_enforcement_enabled"])
        assert not bool(data["reflection_guard_enabled"])
    assert payload["selected_before_random_draw"]
    assert not payload["stochastic_thresholds_drawn"]
