#!/usr/bin/env python3
"""Freeze the approved C2 post-cleanup PF field before stochastic sampling."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from pf_sintering.three_particle_geometry import grain_volumes,topology_status
from pf_sintering.three_particle_phase_a import PhaseAOperator
from pf_sintering.three_particle_sharp_initial import map_sharp_to_pf
from three_particle_mapped_pf_screen import cleanup,design_from_row,measured

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/three_particle_c2_source"
DOC=ROOT/"docs/three_particle/mapped_pf_initial_screen/c2_source_report.json"
DESIGN_ID="Ro150_q0.55_L45W_r0.75_th160_kc+10_ko+10"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()


def main():
    if OUT.exists() and any(OUT.iterdir()):raise RuntimeError("refusing to overwrite frozen C2 source")
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/"docs/three_particle/initial_state_design/selected_initial_candidates.csv").open() as stream:
        row=next(r for r in csv.DictReader(stream) if r["design_id"]==DESIGN_ID)
    design=design_from_row(row);g=map_sharp_to_pf(design,.5e-9);f,before,after=cleanup(design,g)
    volumes=grain_volumes(f,g);topology=topology_status(f,g)
    if topology["stop"] or f.min() < -1e-8 or f.max() > 1+1e-8:
        raise RuntimeError("C2 post-cleanup field failed source guards")
    fields=np.array((f,*(g["ownership"]*f[None])))
    metadata=dict(label="C2_MINIMALLY_CLEANED_POST_MAPPING_SOURCE",design_id=DESIGN_ID,
      sharp_design=asdict(design),cleanup_steps=5,events_enabled=False,stochastic_draws=False,
      symmetry_enforcement_enabled=False,physical_parameters_changed=False,
      grid_shape=list(f.shape),width_m=design.W_m,spacing_m=g["dr"],dz_m=g["dz"],
      source_selected_before_stochastic_draw=True)
    source=OUT/"post_cleanup_state.npz"
    np.savez_compressed(source,fields=fields,f=f,ownership=g["ownership"],z=g["z"],
      r_c=g["r_c"],r_f=g["r_f"],gb=g["gb"],dr=g["dr"],dz=g["dz"],
      radii=g["radii"],centers=g["centers"],metadata=json.dumps(metadata),
      t_model=0.,symmetry_enforcement_enabled=False)
    report=dict(status="C2_SOURCE_FROZEN_BEFORE_STOCHASTIC_DRAW",source=str(source.relative_to(ROOT)),
      source_sha256=sha(source),design_id=DESIGN_ID,branch=git("branch","--show-current"),
      branch_head=git("rev-parse","HEAD"),working_tree_porcelain=git("status","--porcelain=v1"),
      physical_parameters_changed=False,events_enabled=False,stochastic_draws=False,
      topology=topology,field_bounds=[float(f.min()),float(f.max())],
      ownership_closure_Linf=float(np.max(abs(fields[1:].sum(axis=0)-f))),
      grain_volumes_m3=volumes.tolist(),total_volume_m3=float(volumes.sum()),
      mapped_before_cleanup=before,post_cleanup=after)
    DOC.write_text(json.dumps(report,indent=2,default=float)+"\n")
    print(json.dumps(report,indent=2,default=float),flush=True)


if __name__=="__main__":main()
