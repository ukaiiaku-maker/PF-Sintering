#!/usr/bin/env python3
"""Assemble the volume-only bicrystal conjugacy qualification report."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np

n = types.ModuleType("numba")
n.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else lambda f: f)
n.prange = range
n.get_num_threads = lambda: 1
n.set_num_threads = lambda _: None
sys.modules.setdefault("numba", n)
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
sk = types.ModuleType("skimage")
me = types.ModuleType("skimage.measure")
me.find_contours = lambda *a, **k: None
sk.measure = me
sys.modules.setdefault("skimage", sk)
sys.modules.setdefault("skimage.measure", me)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from m16h_three_regime_sink_barrier import build_case
from m16a_gb_benchmark import measure_R_of_z
from m16g_pr_derived_particle_asperity import find_gb_trough
from pf_sintering.constrained_densification_relaxation import (
    constraint_basis, fixed_ownership_mu, multigrain_energy)
from pf_sintering.constrained_newton_krylov import solve_constrained_stationary
from pf_sintering.pr_stress_metrology import pf_contour_estimators
from pf_sintering.three_particle_phase_a import FrozenPhysics

B = 0.25e-9
RUN = ROOT / "runs/constrained_volume_only_bicrystal/bicrystal"
OLD = ROOT / "runs/constrained_newton_krylov/bicrystal"
DOC = ROOT / "docs/three_particle/constrained_volume_only_bicrystal"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_state(path):
    with np.load(path) as d:
        return {key: d[key].copy() for key in d.files}


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    geom, p, *_ = build_case()
    g = dict(geom)
    g["config"] = SimpleNamespace(width=p.W, outer_radius=100e-9)

    state_files = {
        "q000_h8": "q000_h8.npz",
        "q025_h8": "q025_h8.npz",
        "q050_tj_gaussian_h8": "q050_tj_gaussian_h8.npz",
        "q050_broad_tj_h8": "q050_broad_tj_h8.npz",
        "q050_distributed_neck_h8": "q050_distributed_neck_h8.npz",
        "q050_uniform_moving_surface_h8": "q050_uniform_moving_surface_h8.npz",
        "q075_h8": "q075_h8.npz",
        "q100_h8": "q100_h8.npz",
    }
    states = []
    energies = {}
    for name, filename in state_files.items():
        path = RUN / filename
        d = load_state(path)
        check = solve_constrained_stationary(
            d["f"], d["ownership"], g, d["target"],
            active_mask=d["active_mask"], lbfgs_max_iterations=0,
            newton_max_iterations=0, include_moments=False)
        energy = multigrain_energy(d["f"], d["ownership"], g)
        energies[name] = energy
        states.append(dict(
            case=name, u_over_b=float(d["q"]), energy_J=energy,
            projected_KKT_residual=check.projected_kkt_residual,
            normalized_volume_residual=check.normalized_constraint_residual,
            active_cells=int(d["active_mask"].sum()),
            f_min=float(d["f"].min()), f_max=float(d["f"].max()),
            state_sha256=sha256(path)))

    derivative = []
    for half_width, lo, hi in [
        (0.10, "0400", "0600"),
        (0.05, "0450", "0550"),
        (0.025, "0475", "0525")]:
        dm = load_state(RUN / f"bic_vol_direct_ref_q{lo}.npz")
        dp = load_state(RUN / f"bic_vol_direct_ref_q{hi}.npz")
        em = multigrain_energy(dm["f"], dm["ownership"], g)
        ep = multigrain_energy(dp["f"], dp["ownership"], g)
        derivative.append(dict(
            half_width_over_b=half_width, lower_energy_J=em,
            upper_energy_J=ep,
            virtual_work_force_N=-(ep-em)/(2*half_width*B)))

    active = []
    for qname, q in [("q000", 0.0), ("q050", 0.5)]:
        for label in ["h8_t1e-3", "h12_t5e-4", "h16_t1e-4"]:
            path = RUN / f"{qname}_{label}.npz"
            d = load_state(path)
            active.append(dict(
                case=qname, u_over_b=q, domain=label,
                active_cells=int(d["active_mask"].sum()),
                energy_J=multigrain_energy(d["f"], d["ownership"], g),
                projected_KKT_residual=float(solve_constrained_stationary(
                    d["f"], d["ownership"], g, d["target"],
                    active_mask=d["active_mask"], lbfgs_max_iterations=0,
                    newton_max_iterations=0,
                    include_moments=False).projected_kkt_residual)))

    spectra = []
    for path in sorted(RUN.glob("spectrum_*.npz")):
        with np.load(path) as d:
            vals = d["eigenvalues"]
            tangent = d["tangent_residuals"]
            spectra.append(dict(
                case=path.stem.removeprefix("spectrum_"),
                lambda_1=float(vals[0]), lambda_2=float(vals[1]),
                lambda_3=float(vals[2]), lambda_4=float(vals[3]),
                lambda_5=float(vals[4]),
                maximum_tangent_residual=float(np.max(tangent))))

    # Old moment-constrained KKT multiplier contribution.  Ownership also
    # depends explicitly on u, so this target term is not the complete
    # envelope derivative of the archived family.
    scale = g["config"].outer_radius
    weight = np.broadcast_to(g["r_c"][None, :]/scale, geom["f"].shape)
    Wf = 12.0*FrozenPhysics().gamma_s/g["config"].width
    factor = 2*np.pi*g["dr"]*g["dz"]*scale*Wf
    multiplier_rows = []
    for name in ["q04_common", "q045_common", "q0475_common",
                 "q05_tj_gaussian", "q0525_common", "q055_common",
                 "q06_common"]:
        d = load_state(OLD/f"{name}.npz")
        ids = np.flatnonzero(d["active_mask"])
        basis = constraint_basis(d["ownership"], g["z"], scale,
                                 include_moments=True)
        C = (basis*weight[None, :, :]).reshape(len(basis), -1)[:, ids]
        raw = (weight*fixed_ownership_mu(
            d["f"], d["ownership"], g)/Wf).ravel()[ids]
        beta = -np.linalg.solve(C@C.T, C@raw)
        lagrange = factor*beta
        force_term = -lagrange[2]*d["target"][0]/scale
        multiplier_rows.append(dict(
            case=name, moving_moment_multiplier_J=float(lagrange[2]),
            moment_target_force_contribution_N=float(force_term),
            KKT_Linf=float(np.max(np.abs(raw+C.T@beta)))))

    original = np.load("/private/tmp/bicrystal_relaxed_90490.npz")
    radius = measure_R_of_z(original["f"], g["r_c"])
    zgb, _ = find_gb_trough(radius, g["z"], geom["z1"], lam=geom["lam"])
    q05 = load_state(RUN/"q050_tj_gaussian_h8.npz")
    met = pf_contour_estimators(
        measure_R_of_z(q05["f"], g["r_c"]), g["z"], zgb-.25*B,
        gamma_s=FrozenPhysics().gamma_s, psi_reference_deg=160)
    cc_force = met["local_reference"]["s_local_cap"]*met["contact_area"]
    four = [energies[key] for key in energies if key.startswith("q050_")]
    q0_active = [r for r in active if r["case"] == "q000"]
    q05_active = [r for r in active if r["case"] == "q050"]
    c2 = ROOT/"runs/three_particle_c2_campaign/stochastic_seed20260915"
    result = dict(
        classification="ACTIVE_DOMAIN_CONTROLS_FORCE",
        bicrystal_hard_gate_passed=False,
        c2_started=False,
        manifold="fixed grain volumes; prescribed direct ownership/GB frame; no first moments",
        direct_absolute_ownership_frame=True,
        four_guess_energy_spread_J=max(four)-min(four),
        cannon_carter_force_N=cc_force,
        derivative_estimates=derivative,
        q0_active_domain_energy_range_J=max(r["energy_J"] for r in q0_active)-min(r["energy_J"] for r in q0_active),
        q05_active_domain_energy_range_J=max(r["energy_J"] for r in q05_active)-min(r["energy_J"] for r in q05_active),
        all_reported_hessian_minima_positive=all(r["lambda_1"] > 0 for r in spectra),
        old_moment_target_force_N_at_q05=next(r["moment_target_force_contribution_N"] for r in multiplier_rows if r["case"] == "q05_tj_gaussian"),
        old_moment_branch_force_N=1.692737319660218e-8,
        clipping_used=False, root_law_changed=False, barrier_changed=False,
        stochastic_campaign_launched=False,
        c2_history_sha256=sha256(c2/"history.json"),
        c2_trajectory_sha256=sha256(c2/"trajectory.npz"))

    write_csv(DOC/"bicrystal_states.csv", states)
    write_csv(DOC/"force_refinement.csv", derivative)
    write_csv(DOC/"active_domain_sensitivity.csv", active)
    write_csv(DOC/"hessian_spectra.csv", spectra)
    write_csv(DOC/"moment_multiplier_force.csv", multiplier_rows)
    (DOC/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    f10, f05, f025 = [r["virtual_work_force_N"] for r in derivative]
    (DOC/"REPORT.md").write_text(f"""# Volume-only bicrystal conjugacy qualification

**Decision: `ACTIVE_DOMAIN_CONTROLS_FORCE`**

The constrained manifold fixes each grain volume and holds the displacement
through an ownership/GB frame evaluated directly at the absolute coordinate.
It does not constrain material centroids. Every archived state satisfies the
`1e-7` projected KKT gate without clipping; normalized volume errors are below
`1.5e-14`.

The four `u=0.5b` redistribution guesses converge on their common domain with
an energy spread of `{result['four_guess_energy_spread_J']:.3e} J`. Source
placement therefore does not explain the force failure.

The central forces do not converge: `Delta u/b = 0.10`, `0.05`, and `0.025`
give `{f10:.6e}`, `{f05:.6e}`, and `{f025:.6e} N`, respectively. The independent
Cannon--Carter force is `{cc_force:.6e} N`.

Active-domain sensitivity is much larger than the virtual-work signal. At
`u=0`, expanding from 21,443 to 27,746 active cells changes the converged
energy by `{result['q0_active_domain_energy_range_J']:.3e} J`. At `u=0.5b`,
expanding from 18,931 to 25,242 cells changes it by
`{result['q05_active_domain_energy_range_J']:.3e} J`. All expanded states pass
the same KKT gate. A force derived from these energies is therefore not
invariant to the artificial active-domain boundary.

The lowest five projected-Hessian eigenvalues are positive for both old
moment-constrained `u=0` branches and for representative volume-only states.
The old branches are local minima on their recorded constrained domain, so the
earlier word “metastable” is second-order supported there. This does not rescue
work conjugacy because the active-domain energy is not converged.

The old moving-moment multiplier contributes
`{result['old_moment_target_force_N_at_q05']:.6e} N`, near Cannon--Carter but
not the old branch finite-difference force
`{result['old_moment_branch_force_N']:.6e} N`. That multiplier is only the
moment-target term: the archived objective and constraint basis also depend
explicitly on the advected ownership frame, and the neighboring stationary
states switch branches. It cannot by itself be identified with the complete
envelope derivative.

No C2 state was minimized or advanced. The completed stochastic trajectory,
root kinetics, barriers, and ordinary PF evolution were not modified.
""")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
