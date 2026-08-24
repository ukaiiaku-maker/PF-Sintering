"""Production C4 must be an explicit worker-to-integrator contract."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_event_call_threads_explicit_courant():
    full = (ROOT / "scripts" / "pr_full_deterministic_cycle.py").read_text()
    assert "explicit_max_fourth_order_courant=LONG_EXPLICIT_COURANT" in full
    assert "explicit_max_fourth_order_courant=(\n            explicit_max_fourth_order_courant)" in full


def test_production_courant_is_hard_frozen_without_globals_patch():
    source = (ROOT / "scripts" / "pr_stochastic_production_ensemble.py").read_text()
    assert "PRODUCTION_C4=0.05" in source
    assert 'event_call.__globals__' not in source
    assert "explicit_max_fourth_order_courant=PRODUCTION_C4" in source
    assert "assert PRODUCTION_C4 == 0.05" in source


def test_worker_verifies_integrator_reported_courant():
    source = (ROOT / "scripts" / "pr_coarsening_stochastic_two_event.py").read_text()
    assert 'actual_courant = result[4].get(' in source
    assert 'assert actual_courant == explicit_max_fourth_order_courant' in source
