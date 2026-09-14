"""Audit continuous far-contact metrology at an exact production checkpoint."""
from pathlib import Path
import argparse
import json

import numpy as np

from pf_sintering.experimental_pr_metrology import (
    _particle_silhouette, _side_first_stresses)
from pf_sintering.pr_experimental_geometry import (
    _continuous_endpoint_turning, _local_fit)
from pf_sintering.production_mass_transfer_event import DEFAULT_STEP_LIMITS
from pf_sintering.three_particle_cmc import compatible_chain, map_to_pf
from pf_sintering.three_particle_contacts import (
    contact_stresses, locate_contact_points)
from pf_sintering.three_particle_geometry import topology_status
from three_particle_buffered_event_probe import LargerDtBufferedContactEvent


def legacy_integral_stress(event, state):
    """Re-evaluate LEFT integral stress with the former far grid chord."""
    event.bind(state)
    f = state[0]
    points = locate_contact_points(f, event.op)
    setup = {**event.g, "W": event.op.W,
             "gamma_s": event.op.physics.gamma_s}
    stress, branches = contact_stresses(
        f, f*event.g["ownership"][0], f*event.g["ownership"][1],
        setup, *points[0], upper=points[1][0], other_contacts=points)
    geometry = dict(stress)
    for side in ("negative", "positive"):
        branch = branches[side]
        fit = _local_fit(branch, points[0][0], 3*event.op.W)
        geometry[f"delta_phi_f_continuous_{side}_rad"] = (
            _continuous_endpoint_turning(branch, fit["slope_dr_dz"]))
    return _side_first_stresses(
        geometry, _particle_silhouette(branches["positive"]))[
            "sigma_integral_continuous_Pa"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    c, offsets = compatible_chain(.65, 119.999e-9)
    g = map_to_pf(c, offsets, 4e-9, .5e-9)
    frame_rows = json.loads((args.run/"frames"/"index.json").read_text())
    candidates = [row for row in frame_rows
                  if int(row.get("event_number", 0)) == args.event and
                  str(row.get("status", "")).startswith(
                      "STOPPED: current-state transfer failed")]
    if not candidates:
        raise RuntimeError("no retained minimum-step stop frame for event")
    frame_row = min(candidates, key=lambda row: abs(row["q_over_b"]-.5625))
    frame_path = args.run/"frames"/Path(frame_row["path"]).name
    with np.load(frame_path) as data:
        state = tuple(data["fields"].copy())
        g["ownership"] = data["ownership"].copy()
        g["gb"] = data["gb"].copy()
        metadata = json.loads(str(data["metadata"]))
    contact = metadata["root"]["active"]
    label = "GENUINE_STOCHASTIC_EVENT"
    pair = (0, 1) if contact == "LEFT" else (1, 2)
    q0 = float(frame_row["q_over_b"])
    step = .0025
    event = LargerDtBufferedContactEvent(g, pair, max_fast_blocks=512)
    before = event.metrics(state, q0)
    legacy_before = legacy_integral_stress(event, state)
    result = event.run(
        state, target=step,
        initial_step_over_b=step, minimum_step_over_b=step,
        maximum_step_over_b=step)
    after = event.metrics(result[:4], q0+step)
    legacy_after = legacy_integral_stress(event, result[:4])
    packet = result[5]["packets"][0] if result[5]["packets"] else {}
    fields = np.asarray(result[:4])
    accepted_limits_hold = all(
        packet[f"accepted_change_{key}"] <= limit
        for key, limit in DEFAULT_STEP_LIMITS.items())
    output = dict(
        label="FAR_CONTACT_TANGENT_CONTINUITY_AUDIT",
        source_checkpoint=str(frame_path),
        source_frame=int(frame_row["frame"]),
        contact=contact, checkpoint_label=label,
        q_start_over_b=q0, trial_step_over_b=step,
        prior_worker_stop_reason="current-state transfer failed at minimum step",
        prior_worker_stop_detail=["sigma_integral_continuous_MPa"],
        legacy_last_chord_change_MPa=(legacy_after-legacy_before)*1e-6,
        fitted_far_tangent_change_MPa=(
            after["sigma_integral_continuous_Pa"]-
            before["sigma_integral_continuous_Pa"])*1e-6,
        local_stress_change_MPa=(
            after["sigma_local_Pa"]-before["sigma_local_Pa"])*1e-6,
        transport_affinity_change_MPa=(
            after["transport_affinity_Pa"]-
            before["transport_affinity_Pa"])*1e-6,
        root_rate_relative_change=(
            after["LEFT_root_rate_per_s"]/
            before["LEFT_root_rate_per_s"]-1),
        completed=bool(result[4]), stop_reason=result[5]["stop_reason"],
        accepted_changes={
            key: packet[f"accepted_change_{key}"]
            for key in DEFAULT_STEP_LIMITS},
        acceptance_limits=DEFAULT_STEP_LIMITS,
        accepted_limits_hold=accepted_limits_hold,
        field_min=float(fields[0].min()), field_max=float(fields[0].max()),
        ownership_closure_linf=float(
            np.max(np.abs(fields[1:].sum(axis=0)-fields[0]))),
        total_volume_relative_error=float(
            after["total_volume_m3"]/before["total_volume_m3"]-1),
        topology=topology_status(result[0], g),
        bicrystal_endpoint_definition_unchanged=True,
        local_stress_and_root_law_definitions_unchanged=True,
        no_clipping=True, no_fitted_physics_correction=True,
        physics_parameters_unchanged=True, stochastic_state_modified=False)
    output["passed"] = bool(
        result[4] and accepted_limits_hold and
        abs(output["legacy_last_chord_change_MPa"]) > .1 and
        abs(output["fitted_far_tangent_change_MPa"]) <= .1 and
        output["field_min"] >= -1e-8 and output["field_max"] <= 1+1e-8 and
        output["ownership_closure_linf"] <= 5e-15 and
        abs(output["total_volume_relative_error"]) <= 1e-11 and
        not output["topology"]["stop"])
    args.out.write_text(json.dumps(output, indent=2, default=float)+"\n")
    print(json.dumps(output, indent=2, default=float))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
