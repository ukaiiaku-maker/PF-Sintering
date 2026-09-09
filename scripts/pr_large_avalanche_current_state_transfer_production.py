#!/usr/bin/env python3
"""Eight-avalanche production with current-state conservative transfer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import pr_corrected_normalized_ownership_production as corrected  # noqa: E402
import pr_full_corrected_production_campaign as campaign  # noqa: E402
import pr_full_corrected_production_slowgb_large_avalanche as production  # noqa: E402
import pr_large_avalanche_qcont_Dgb1p5x_threshold1p25 as rejected_parent  # noqa: E402
from pf_sintering.axisym import axisym_volume  # noqa: E402
from pf_sintering.production_mass_transfer_event import (  # noqa: E402
    EVENT_INTEGRATOR_NAME,
    current_state_mass_transfer_event,
)
from pr_quasistatic_q_continuation import relax_fast_manifold, scalar_state  # noqa: E402


OUT = ROOT / "runs/pr_large_avalanche_current_state_transfer_8avalanche_v1"
QUALIFICATION = ROOT / "runs/pr_current_state_mass_transfer_qualification_v1/decision.json"
FAILURE_RECORD = ROOT / (
    "runs/pr_large_avalanche_qcont_Dgb1p5x_threshold1p25_20260904/"
    "QUASISTATIC_GEOMETRIC_Q_CONTINUATION_NOT_PRODUCTION_VIABLE.json")
SOURCE_FILES = (
    Path(__file__),
    ROOT / "pf_sintering/current_state_mass_transfer.py",
    ROOT / "pf_sintering/production_mass_transfer_event.py",
    ROOT / "pf_sintering/pr_experimental_geometry.py",
    ROOT / "pf_sintering/experimental_pr_metrology.py",
    ROOT / "scripts/pr_avalanche_renewal_five.py",
    ROOT / "scripts/pr_full_corrected_production_slowgb_large_avalanche.py",
)


def source_provenance():
    combined = hashlib.sha256()
    entries = {}
    for path in SOURCE_FILES:
        data = path.read_bytes()
        relative = str(path.relative_to(ROOT))
        entries[relative] = hashlib.sha256(data).hexdigest()
        combined.update(relative.encode())
        combined.update(b"\0")
        combined.update(data)
    return dict(
        git_HEAD=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_snapshot_sha256=combined.hexdigest(),
        source_files_sha256=entries)


def current_state_event_call(
        state, setup, geom, evaluator, transport, quota_fraction_b,
        *, event_restart=None, state_callback=None, progress_callback=None,
        **kwargs):
    """Production adapter retaining the common event-call contract."""
    reference = (state if event_restart is None
                 else event_restart["base_fields"])
    volume0 = axisym_volume(
        reference[0], setup["r_c"], setup["dr"], setup["dz"])

    def metrics(fields, q_over_b):
        row = scalar_state(fields, q_over_b, setup, geom, evaluator, volume0)
        record = evaluator(*fields)
        row["contact_area_m2"] = float(record["contact_area_m2"])
        row["sigma_integral_continuous_MPa"] = float(
            row["sigma_integral_continuous_Pa"])*1.0e-6
        return row

    def fast_relax(fields, q_over_b):
        relaxed, row, _history, _projector = relax_fast_manifold(
            fields, q_over_b, setup, geom, volume0)
        record = evaluator(*relaxed)
        row["contact_area_m2"] = float(record["contact_area_m2"])
        row["sigma_integral_continuous_MPa"] = float(
            row["sigma_integral_continuous_Pa"])*1.0e-6
        return relaxed, row, dict(
            converged=bool(row.get("fast_manifold_converged", 0)),
            iterations=int(row.get("fast_relax_blocks", 0))*10,
            blocks=int(row.get("fast_relax_blocks", 0)),
            model_time=float(row.get("fast_relax_model_time", 0.0)))

    return current_state_mass_transfer_event(
        state, setup, geom, evaluator, transport, quota_fraction_b,
        fast_relax_fn=fast_relax, state_metrics_fn=metrics,
        event_restart=event_restart,
        accepted_state_callback=state_callback,
        accepted_progress_callback=progress_callback,
        initial_step_over_b=0.02,
        minimum_step_over_b=0.0025,
        maximum_step_over_b=0.02,
        maximum_accepted_states=400,
        **kwargs)


current_state_event_call.event_integrator_name = EVENT_INTEGRATOR_NAME


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    gate = json.loads(QUALIFICATION.read_text())
    if (gate.get("status") != "PASS_CURRENT_STATE_TRANSFER_TWO_EVENT_GATE"
            or not gate.get("production_launch_authorized")):
        raise RuntimeError("current-state two-event qualification did not pass")
    if not FAILURE_RECORD.exists():
        raise RuntimeError("rejected geometric-q method has no failure record")

    preflight = rejected_parent.cheap_transport_preflight()
    production.OUT = OUT
    production.EVENT_QUALIFICATION = QUALIFICATION
    production.quasistatic_event_call = current_state_event_call
    production.D_GB_REFERENCE = rejected_parent.QUALIFIED_D_GB
    production.D_GB_TARGET = rejected_parent.D_GB
    production.REFERENCE_MEDIAN_EVENT_S = (
        rejected_parent.QUALIFIED_EVENT_DURATION_S)
    production.TARGET_EVENTS_PER_AVALANCHE = 15
    production.TARGET_AVALANCHE_RELAXATION_S = 0.42
    production.TARGET_MEDIAN_EVENT_S = preflight[
        "predicted_candidate_event_duration_s"]
    production.DESCENDANT_DELTA_G0_EV = (
        rejected_parent.DELTA_G_DESC_INITIAL_EV)
    production.ROOT_THRESHOLD_MULTIPLIER = (
        rejected_parent.ROOT_THRESHOLD_MULTIPLIER)
    production.DESCENDANT_ESCALATION_AFTER_AVALANCHES = 2
    production.DESCENDANT_ESCALATION_IF_ALL_S_BELOW = 10
    production.DESCENDANT_ESCALATION_NEW_DELTA_G_EV = (
        rejected_parent.DELTA_G_DESC_ESCALATED_EV)
    production.SOLID_VOLUME_RELATIVE_TOLERANCE = (
        rejected_parent.VOLUME_ERROR_CEILING)
    production.AVALANCHES_REQUESTED = 8
    production.MAX_DESCENDANTS_SAFETY = 25
    # Preserve the stochastic identity selected before the rejected campaign;
    # no threshold is inspected or redrawn for convenience.
    production.REUSED_SEED = rejected_parent.PRESERVED_PRELAUNCH_OS_SEED
    production.CONTINUATION = None
    production.EXTRA_RUN_METADATA = dict(
        campaign="large-avalanche current-state conservative transfer",
        rejected_parent_method=(
            "QUASISTATIC_GEOMETRIC_Q_CONTINUATION_NOT_PRODUCTION_VIABLE"),
        rejected_parent_failure_record=str(FAILURE_RECORD.resolve()),
        deterministic_qualification=str(QUALIFICATION.resolve()),
        event_propagator=EVENT_INTEGRATOR_NAME,
        q_definition="accumulated transferred material only",
        transfer_definition="DeltaV=A_GB(current)*Deltaq",
        mask_definition=(
            "current f interface; TJ receiver 0-3W; outer branch donor 3-10W; "
            "rebuilt every accepted increment"),
        immutable_parent_event_geometry_used=False,
        live_state_clipping_used=False,
        stochastic_seed_reused=True,
        stochastic_barriers_changed=False,
        stochastic_thresholds_changed=False,
        D_GB_changed=False,
        facilitation_changed=False,
        hazard_logic_changed=False,
        current_state_source_provenance=source_provenance())
    production.main()


if __name__ == "__main__":
    main()
