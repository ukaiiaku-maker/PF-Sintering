#!/usr/bin/env python3
"""Tabulate the established energy-conjugate stress for selected sharp states."""
from pathlib import Path
import csv
import math

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/"docs/three_particle/initial_state_design"


def main():
    selected=list(csv.DictReader((DOC/"selected_initial_candidates.csv").open()))
    derivatives=list(csv.DictReader((DOC/"local_perturbations.csv").open()))
    spacing={row["design_id"]:row for row in derivatives
             if row["coordinate"] == "gb_spacing"}
    rows=[]
    for candidate in selected:
        derivative=spacing[candidate["design_id"]]
        length=float(candidate["Lc_over_W"])*4e-9
        contact_area=math.pi*(float(candidate["rTJ_nm"])*1e-9)**2
        force=-float(derivative["dE_dq_J"])/length
        rows.append(dict(design_id=candidate["design_id"],
            dE_dlnLc_J=float(derivative["dE_dq_J"]),
            energy_conjugate_force_N=force,
            one_contact_area_m2=contact_area,
            energy_conjugate_stress_MPa=force/contact_area*1e-6,
            exact_local_activation_stress_MPa=float(candidate["sigma0_MPa"])))
    with (DOC/"initial_energy_conjugate_stress.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    lines=["# Initial energy-conjugate stress", "",
      "For the independent `gb_spacing` perturbation, the established virtual-work diagnostic is evaluated as `F_sint = -(dE/d ln Lc)/Lc` and `sigma_energy = F_sint/(pi r_TJ^2)`. Moving both contacts by half the spacing increment gives total work `F_sint dLc`. This diagnostic does not drive the PF evolution.", "",
      "|design|energy-conjugate stress (MPa)|exact local activation stress (MPa)|", "|---|---:|---:|"]
    lines += [f"|`{r['design_id']}`|{r['energy_conjugate_stress_MPa']:.6g}|{r['exact_local_activation_stress_MPa']:.6g}|" for r in rows]
    (DOC/"ENERGY_STRESS.md").write_text("\n".join(lines)+"\n")

if __name__ == "__main__": main()
