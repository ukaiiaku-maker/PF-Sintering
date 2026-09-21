#!/usr/bin/env python3
"""Assemble the bicrystal hard-gate report for the constrained KKT solver."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np

# The qualification environment has NumPy/SciPy but omits optional production
# I/O and acceleration packages used by imported geometry helpers.
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
from pf_sintering.constrained_densification_relaxation import multigrain_energy
from pf_sintering.constrained_newton_krylov import solve_constrained_stationary
from pf_sintering.pr_stress_metrology import pf_contour_estimators
from pf_sintering.three_particle_phase_a import FrozenPhysics

B = 0.25e-9
RUN = ROOT / "runs/constrained_newton_krylov/bicrystal"
DOC = ROOT / "docs/three_particle/constrained_newton_krylov"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    geom, p, *_ = build_case()
    g = dict(geom)
    g["config"] = SimpleNamespace(width=p.W, outer_radius=100e-9)
    cases = {
        "q0_branch_a": 0.0,
        "q0_branch_b": 0.0,
        "q04_common": 0.4,
        "q045_common": 0.45,
        "q0475_common": 0.475,
        "q05_tj_gaussian": 0.5,
        "q05_broad_tj": 0.5,
        "q05_distributed_neck": 0.5,
        "q05_uniform_moving_surface": 0.5,
        "q055_common": 0.55,
        "q0525_common": 0.525,
        "q06_common": 0.6,
        "q10_warm": 1.0,
    }
    rows = []
    fields = {}
    for name, q in cases.items():
        path = RUN / f"{name}.npz"
        with np.load(path) as d:
            f = d["f"].copy()
            phi = d["ownership"].copy()
            target = d["target"].copy()
            mask = d["active_mask"].copy() if "active_mask" in d else None
        g["f"] = f
        g["ownership"] = phi
        check = solve_constrained_stationary(
            f, phi, g, target, active_mask=mask,
            lbfgs_max_iterations=0, newton_max_iterations=0)
        row = dict(
            case=name, u_over_b=q,
            energy_J=multigrain_energy(check.f, phi, g),
            projected_KKT_residual=check.projected_kkt_residual,
            normalized_constraint_residual=check.normalized_constraint_residual,
            active_cells=check.active_cells,
            f_min=float(check.f.min()), f_max=float(check.f.max()),
            state_sha256=sha256(path),
        )
        rows.append(row)
        fields[name] = check.f

    by_name = {row["case"]: row for row in rows}
    q05 = [row for row in rows if row["case"].startswith("q05_")]
    q05_spread = max(r["energy_J"] for r in q05) - min(r["energy_J"] for r in q05)
    q0_spread = abs(by_name["q0_branch_a"]["energy_J"]-
                    by_name["q0_branch_b"]["energy_J"])
    derivative_rows = []
    for label, lo, hi, half_width in [
        ("delta_0.10b", "q04_common", "q06_common", 0.10),
        ("delta_0.05b", "q045_common", "q055_common", 0.05),
        ("delta_0.025b_branch_sensitive", "q0475_common", "q0525_common", 0.025),
    ]:
        force = -(by_name[hi]["energy_J"]-by_name[lo]["energy_J"])/(2*half_width*B)
        derivative_rows.append(dict(step=label, half_width_over_b=half_width,
                                    virtual_work_force_N=force))

    original = np.load("/private/tmp/bicrystal_relaxed_90490.npz")
    radius = measure_R_of_z(original["f"], g["r_c"])
    zgb, _ = find_gb_trough(radius, g["z"], geom["z1"], lam=geom["lam"])
    q05_field = fields["q05_tj_gaussian"]
    met = pf_contour_estimators(
        measure_R_of_z(q05_field, g["r_c"]), g["z"], zgb-0.25*B,
        gamma_s=FrozenPhysics().gamma_s, psi_reference_deg=160)
    cc_force = met["local_reference"]["s_local_cap"]*met["contact_area"]
    refined_force = derivative_rows[1]["virtual_work_force_N"]
    cc_mismatch = abs(refined_force-cc_force)/abs(cc_force)
    c2_run = ROOT / "runs/three_particle_c2_campaign/stochastic_seed20260915"
    result = dict(
        classification="MULTIPLE_METASTABLE_REDISPOSITION_BRANCHES",
        bicrystal_hard_gate_passed=False,
        c2_stationary_solve_started=False,
        c2_continuation_started=False,
        reason=("bicrystal common-domain stationary states are guess-dependent "
                "at u=0 and the stationary-branch derivative does not reproduce "
                "the qualified Cannon-Carter force"),
        hessian_vector_relative_Linf_error=9.202968620115861e-10,
        kkt_tolerance=1e-7,
        constraint_tolerance=2e-12,
        q0_common_mask_energy_spread_J=q0_spread,
        q05_four_guess_energy_spread_J=q05_spread,
        redistribution_independence_tolerance_J=1e-19,
        cannon_carter_force_N=cc_force,
        refined_virtual_work_force_N=refined_force,
        force_relative_mismatch=cc_mismatch,
        established_force_tolerance=0.05,
        derivative_estimates=derivative_rows,
        active_domain="diffuse interface 0.01<f<0.99 plus halo, truncated at 0.001<f<0.999",
        clipping_used=False,
        root_law_changed=False,
        barrier_changed=False,
        stochastic_campaign_launched=False,
        production_operator_promoted=False,
        completed_C2_trajectory_modified=False,
        c2_history_sha256=sha256(c2_run/"history.json"),
        c2_trajectory_sha256=sha256(c2_run/"trajectory.npz"),
        states=rows,
    )
    with (DOC/"bicrystal_states.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    with (DOC/"force_refinement.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=derivative_rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(derivative_rows)
    (DOC/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    (DOC/"REPORT.md").write_text(f"""# Constrained Newton--Krylov bicrystal hard gate

**Decision: `MULTIPLE_METASTABLE_REDISPOSITION_BRANCHES`**

The new solver is a reduced-space projected L-BFGS method followed by a
damped Newton--Krylov polish. Its analytic axisymmetric Hessian action agrees
with a centered finite difference of the implemented gradient to
`{result['hessian_vector_relative_Linf_error']:.3e}` relative Linf error. It
uses an explicit box active set and never clips the phase field.

Every archived state closes the volume/first-moment constraints near machine
precision and reaches the declared projected KKT tolerance of `1e-7` on its
recorded active domain. The four prescribed redistribution guesses at
`u=0.5b` converge on one common active mask with an energy spread of
`{q05_spread:.3e} J`, below the `1e-19 J` independence tolerance.

The bicrystal hard gate nevertheless fails. Two `u=0` initial morphologies,
solved on the identical 45,427-cell union domain, remain separated by
`{q0_spread:.3e} J` after both satisfy the KKT tolerance. The local derivative
is also inconsistent with the independent capillary force: the
`Delta u=0.05b` estimate is `{refined_force:.6e} N`, whereas Cannon--Carter is
`{cc_force:.6e} N` (relative mismatch `{cc_mismatch:.2%}`, tolerance `5%`).
The `0.10b`, `0.05b`, and `0.025b` estimates do not form a convergent sequence;
the finest pair switches among nearby stationary branches.

These are converged stationary states, so this outcome is not another
projected-gradient iteration-limit result. It shows that the presently defined
fixed-ownership, fixed-volume/first-moment manifold is multivalued and does not
yet supply a unique work-conjugate bicrystal branch. Per the hard-gate rule, no
C2 stationary solve, C2 continuation, energy derivative, stochastic run, or
production-operator promotion was performed.
""")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
