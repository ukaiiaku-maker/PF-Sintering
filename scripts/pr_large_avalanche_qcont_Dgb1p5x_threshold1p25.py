#!/usr/bin/env python3
"""Large-avalanche production: 1.5x D_GB and 1.25x root thresholds."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_full_corrected_production_campaign as campaign  # noqa: E402
import pr_full_corrected_production_slowgb_large_avalanche as production  # noqa: E402
from pf_sintering.event_integrator_selection import (  # noqa: E402
    estimate_and_select_event_integrator,
)
from pf_sintering.model_time_pr_event import _node_state  # noqa: E402


OUT = ROOT / "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904"
QUALIFIED_D_GB = 9.193882380929775e-15
D_GB = 1.37908e-14
QUALIFIED_EVENT_DURATION_S = 0.041935548244046716
ROOT_THRESHOLD_MULTIPLIER = 1.25
DELTA_G_DESC_INITIAL_EV = 1.50
DELTA_G_DESC_ESCALATED_EV = 1.75
FACILITATION_DECAY = 0.60
VOLUME_ERROR_CEILING = 1.0e-7
AVALANCHES_REQUESTED = 10
EVENT_SURFACE_ACTIVE_LENGTH_M = 15.0e-9
PRESERVED_PRELAUNCH_OS_SEED = 299814244860662850753714406905245801947
QUALIFIED_Q0 = ROOT / (
    "runs/pr_direct_adaptive_q_event_regression_v5/fields/q_0p000000.npz")

SOURCE_FILES = (
    Path(__file__),
    ROOT / "scripts/pr_full_corrected_production_slowgb_large_avalanche.py",
    ROOT / "scripts/pr_avalanche_renewal_five.py",
    ROOT / "pf_sintering/event_integrator_selection.py",
    ROOT / "pf_sintering/production_q_event.py",
    ROOT / "pf_sintering/quasistatic_event_continuation.py",
    ROOT / "pf_sintering/model_time_transport.py",
    ROOT / "pf_sintering/pr_avalanche.py",
)


def source_provenance() -> dict:
    entries = {}
    combined = hashlib.sha256()
    for path in SOURCE_FILES:
        data = path.read_bytes()
        relative = str(path.relative_to(ROOT))
        digest = hashlib.sha256(data).hexdigest()
        entries[relative] = digest
        combined.update(relative.encode())
        combined.update(b"\0")
        combined.update(data)
    return dict(
        git_HEAD=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        production_source_snapshot_sha256=combined.hexdigest(),
        production_source_files_sha256=entries)


def cheap_transport_preflight() -> dict:
    """Check D_GB clock scaling and q-path independence on one saved state."""
    if not QUALIFIED_Q0.exists():
        raise FileNotFoundError(QUALIFIED_Q0)
    geom, setup = campaign.padded_case_builder()
    evaluator = production.corrected.evaluator_builder(setup, geom)
    with np.load(QUALIFIED_Q0, allow_pickle=False) as saved:
        state = tuple(np.asarray(saved[key]).copy()
                      for key in ("f", "particle", "substrate"))
    evaluated = evaluator(*state)
    reference = replace(
        production.make_transport(geom),
        D_gb_m2_per_model_time=QUALIFIED_D_GB)
    candidate = replace(reference, D_gb_m2_per_model_time=D_GB)

    nodes = {}
    for name, transport in (("qualified", reference), ("candidate", candidate)):
        _branches, node, _coordinates = _node_state(
            evaluated, state[0], setup["r_c"], setup["z"], transport,
            production.corrected.EVENT_SURFACE_MOBILITY, active=True)
        nodes[name] = node
    rate_ratio = (
        nodes["candidate"]["Vdot_GB_m3_per_model_time"]
        / nodes["qualified"]["Vdot_GB_m3_per_model_time"])
    D_ratio = D_GB / QUALIFIED_D_GB
    fraction_change = max(abs(
        nodes["candidate"]["branch_rate_over_net_GB"][side]
        - nodes["qualified"]["branch_rate_over_net_GB"][side])
        for side in ("negative", "positive"))
    affinity_change = abs(
        nodes["candidate"]["transport_affinity_Pa"]
        / nodes["qualified"]["transport_affinity_Pa"] - 1.0)
    predicted_duration = QUALIFIED_EVENT_DURATION_S / D_ratio
    selection = estimate_and_select_event_integrator(
        "auto", transport=candidate,
        affinity_Pa=nodes["candidate"]["transport_affinity_Pa"],
        surface_length_m=EVENT_SURFACE_ACTIVE_LENGTH_M,
        surface_B_m4_per_model_time=production.corrected.EVENT_B_PF,
        seconds_per_model_time=production.renewal.SECONDS_PER_MODEL_TIME)
    passed = (
        math.isclose(rate_ratio, D_ratio, rel_tol=2.0e-6)
        and fraction_change < 1.0e-6
        and affinity_change < 1.0e-6
        and selection["event_integrator_selected"] == "quasistatic")
    result = dict(
        status="PASS" if passed else "FAIL",
        check_kind="static saved-state algebra/code-path; no PF evolution",
        qualified_D_GB_m2_per_model_time=QUALIFIED_D_GB,
        candidate_D_GB_m2_per_model_time=D_GB,
        D_GB_ratio=D_ratio,
        GB_volume_rate_ratio=rate_ratio,
        maximum_branch_partition_absolute_change=fraction_change,
        transport_affinity_relative_change=affinity_change,
        qualified_event_duration_s=QUALIFIED_EVENT_DURATION_S,
        predicted_candidate_event_duration_s=predicted_duration,
        fast_manifold_state_change_from_clock_scaling=False,
        selector=selection)
    if not passed:
        raise RuntimeError("large-avalanche transport preflight failed: "
                           + json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    preflight = cheap_transport_preflight()
    provenance = source_provenance()

    production.OUT = OUT
    production.D_GB_REFERENCE = QUALIFIED_D_GB
    production.D_GB_TARGET = D_GB
    production.REFERENCE_MEDIAN_EVENT_S = QUALIFIED_EVENT_DURATION_S
    production.TARGET_EVENTS_PER_AVALANCHE = 15
    production.TARGET_AVALANCHE_RELAXATION_S = 0.42
    production.TARGET_MEDIAN_EVENT_S = preflight[
        "predicted_candidate_event_duration_s"]
    production.DESCENDANT_DELTA_G0_EV = DELTA_G_DESC_INITIAL_EV
    production.ROOT_THRESHOLD_MULTIPLIER = ROOT_THRESHOLD_MULTIPLIER
    production.DESCENDANT_ESCALATION_AFTER_AVALANCHES = 2
    production.DESCENDANT_ESCALATION_IF_ALL_S_BELOW = 10
    production.DESCENDANT_ESCALATION_NEW_DELTA_G_EV = (
        DELTA_G_DESC_ESCALATED_EV)
    production.SOLID_VOLUME_RELATIVE_TOLERANCE = VOLUME_ERROR_CEILING
    production.AVALANCHES_REQUESTED = AVALANCHES_REQUESTED
    production.MAX_DESCENDANTS_SAFETY = 25
    production.REUSED_SEED = PRESERVED_PRELAUNCH_OS_SEED
    production.CONTINUATION = None
    production.EXTRA_RUN_METADATA = dict(
        campaign="large-avalanche q-continuation D_GB1.5x threshold1.25",
        campaign_timestamp_label="20260904",
        seed_provenance=(
            "fresh OS seed recorded by the first prelaunch attempt and reused "
            "after a manifest-only TypeError; no physical step had occurred"),
        scientific_targets=dict(
            typical_avalanche_size_b_events="10-20",
            avalanche_relaxation_duration_s="0.2-0.6",
            root_loading_threshold_change="1.25x stochastic threshold"),
        root_attempt_physics_changed=False,
        threshold_old_definition="unscaled Exp(1) root draw",
        threshold_new_definition="1.25 * same-distribution Exp(1) root draw",
        threshold_multiplier=ROOT_THRESHOLD_MULTIPLIER,
        initial_descendant_lowering_eV=DELTA_G_DESC_INITIAL_EV,
        descendant_decay=FACILITATION_DECAY,
        one_time_escalation_rule=(
            "after two complete avalanches, if both S<10, use 1.75 eV "
            "starting with avalanche 3; never change within an avalanche"),
        volume_error_ceiling=VOLUME_ERROR_CEILING,
        affinity_definition=(
            "instantaneous corrected zero-storage TJ-node transport affinity"),
        q_continuation_tolerances=(
            "pf_sintering.production_q_event.DEFAULT_STEP_LIMITS"),
        prelaunch_transport_check=preflight,
        **provenance)
    production.main()


if __name__ == "__main__":
    main()
