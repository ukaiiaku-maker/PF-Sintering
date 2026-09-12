"""Freeze a deterministic events-off checkpoint as the production source.

This is a representation handoff only. It preserves the full asymmetric PF
field and reconstructs the three explicit grain fields from the saved ownership
partition without reflection, averaging, clipping, or further relaxation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parent)]

import numpy as np

from pf_sintering.three_particle_cmc import compatible_chain, map_to_pf
from pf_sintering.three_particle_geometry import grain_volumes, topology_status
from three_particle_forced_event import ContactEvent, MANIFEST


DEFAULT_SOURCE = Path(
    "runs/three_particle_volume_loading_065_symmetry_released/loss_2pct.npz")
DEFAULT_OUT = Path("runs/three_particle_production_065/production_initial_2pct.npz")
DEFAULT_REPORT = Path(
    "docs/three_particle/production_065/production_initial_2pct.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, out, report):
    # Match the loading driver's expression exactly; the two mathematically
    # equivalent products differ by one floating-point ulp in the grid origin.
    c, offsets = compatible_chain(.65, 119.999*1e-9)
    g = map_to_pf(c, offsets, 4e-9, .5e-9)
    with np.load(source) as data:
        if bool(data["events_enabled"]):
            raise ValueError("production source must come from events-off loading")
        np.testing.assert_array_equal(g["z"], data["z"])
        np.testing.assert_array_equal(g["gb"], data["gb"])
        f = data["f"].copy()
        ownership = data["ownership"].copy()
        t_model = float(data["t_model"])
    if np.max(np.abs(ownership.sum(axis=0) - 1.0)) > 5e-15:
        raise ValueError("ownership partition is not closed")
    fields = np.array((f, *(ownership * f[None])))
    if np.max(np.abs(fields[1:].sum(axis=0) - fields[0])) > 5e-15:
        raise ValueError("explicit grain fields do not close to f")
    g["ownership"] = ownership
    event = ContactEvent(g)
    event.metrics(tuple(fields), 0.0)
    contacts = event.metrics(tuple(fields), 0.0)
    volumes = grain_volumes(f, g)
    health = dict(
        f_min=float(f.min()), f_max=float(f.max()),
        material_volume_m3=float(volumes.sum()),
        ownership_closure=float(np.max(np.abs(fields[1:].sum(axis=0)-f))),
        mirror_error_diagnostic=float(np.max(np.abs(f-f[::-1]))),
        topology=topology_status(f, {
            **g, "gb": np.array([contacts["LEFT_z_TJ_m"],
                                    contacts["RIGHT_z_TJ_m"]])}))
    if health["f_min"] < -1e-8 or health["f_max"] > 1+1e-8:
        raise ValueError("unchanged field bound failed")
    if health["topology"]["stop"]:
        raise ValueError("selected source is topology-terminal")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".writing.npz")
    np.savez_compressed(
        temp, fields=fields, f=f, ownership=ownership, z=g["z"],
        r_c=g["r_c"], gb=g["gb"], t_model=t_model,
        events_enabled=False, symmetry_enforcement_enabled=False,
        reflection_guard_enabled=False, no_symmetry_projection=True,
        source_sha256=sha256(source))
    os.replace(temp, out)
    payload = dict(
        label="SELECTED_FULL_DOMAIN_PRODUCTION_INITIAL_MICROSTRUCTURE",
        source=str(source), source_sha256=sha256(source), output=str(out),
        output_sha256=sha256(out), selected_before_random_draw=True,
        stochastic_thresholds_drawn=False,
        physical_time_s=t_model*MANIFEST["seconds_per_model_time"],
        fields=list(("f", "eta_left", "eta_center", "eta_right")),
        reconstruction="eta_i = saved ownership_i * unchanged saved f",
        symmetry_enforcement_enabled=False, reflection_guard_enabled=False,
        no_reflection_averaging=True, no_symmetry_projection=True,
        no_clipping=True, no_fitted_correction=True, health=health)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".writing.json")
    temporary.write_text(json.dumps(payload, indent=2)+"\n")
    os.replace(temporary, report)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out, args.report), indent=2))


if __name__ == "__main__":
    main()
