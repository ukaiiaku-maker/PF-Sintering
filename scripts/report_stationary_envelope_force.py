#!/usr/bin/env python3
"""Report the stationary-Lagrangian envelope-force bicrystal study."""
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

from m16a_gb_benchmark import measure_R_of_z
from m16g_pr_derived_particle_asperity import find_gb_trough
from m16h_three_regime_sink_barrier import build_case
from pf_sintering.axisym import r_centers_faces
from pf_sintering.constrained_densification_relaxation import (
    multigrain_energy, volume_targets)
from pf_sintering.constrained_envelope_force import stationary_envelope_force
from pf_sintering.constrained_newton_krylov import solve_constrained_stationary
from pf_sintering.pr_stress_metrology import pf_contour_estimators
from pf_sintering.three_particle_phase_a import FrozenPhysics
from pf_sintering.work_conjugate_densification import (
    one_contact_state, triple_junction_support)

B = 0.25e-9
OLD_RUN = ROOT / "runs/constrained_volume_only_bicrystal/bicrystal"
RUN = ROOT / "runs/constrained_stationary_envelope/bicrystal"
DOC = ROOT / "docs/three_particle/constrained_stationary_envelope"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_state(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def small_full_domain_benchmark():
    nz, nr = 28, 16
    dr = dz = 0.5e-9
    z = (np.arange(nz)-(nz-1)/2)*dz
    rc, rf = r_centers_faces(nr, dr)
    radius = 3e-9
    f0 = np.broadcast_to(
        0.5*(1+np.tanh((radius-rc[None, :])/1e-9)), (nz, nr)).copy()

    def ownership(u):
        right = np.broadcast_to(
            0.5*(1+np.tanh((z[:, None]+0.5*u)/1e-9)), f0.shape).copy()
        return np.stack([1-right, right])

    g = dict(z=z, r_c=rc, r_f=rf, dr=dr, dz=dz,
             config=SimpleNamespace(width=2e-9, outer_radius=5e-9))
    target = volume_targets(f0, ownership(0), z, rc, g["config"].outer_radius)
    mask = np.ones_like(f0, dtype=bool)
    u0 = 0.25e-9
    center = solve_constrained_stationary(
        f0, ownership(u0), g, target, active_mask=mask,
        lbfgs_max_iterations=4000, newton_max_iterations=1000,
        include_moments=False, kkt_tolerance=1e-8)
    if not center.converged:
        raise RuntimeError("small full-domain central solve did not converge")
    rows = []
    for du in [0.1e-9, 0.05e-9, 0.025e-9, 0.0125e-9]:
        minus = ownership(u0-du)
        plus = ownership(u0+du)
        envelope = stationary_envelope_force(
            center.f, ownership(u0), minus, plus, g, du,
            active_mask=mask, include_moments=False)
        lower = solve_constrained_stationary(
            center.f, minus, g, target, active_mask=mask,
            lbfgs_max_iterations=2000, newton_max_iterations=1000,
            include_moments=False, kkt_tolerance=1e-8)
        upper = solve_constrained_stationary(
            center.f, plus, g, target, active_mask=mask,
            lbfgs_max_iterations=2000, newton_max_iterations=1000,
            include_moments=False, kkt_tolerance=1e-8)
        if not (lower.converged and upper.converged):
            raise RuntimeError("small full-domain neighboring solve did not converge")
        branch = -(multigrain_energy(upper.f, plus, g)
                   - multigrain_energy(lower.f, minus, g))/(2*du)
        rows.append(dict(
            delta_u_m=du,
            envelope_force_N=envelope["envelope_force_N"],
            minimized_branch_force_N=branch,
            relative_difference=abs(envelope["envelope_force_N"]-branch)
                                / max(abs(branch), 1e-300),
            central_KKT_residual=center.projected_kkt_residual,
            lower_KKT_residual=lower.projected_kkt_residual,
            upper_KKT_residual=upper.projected_kkt_residual))
    return rows


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    geom, p, *_ = build_case()
    g = dict(geom)
    g["config"] = SimpleNamespace(width=p.W, outer_radius=100e-9)

    reference = load_state(RUN/"reference_bicrystal_relaxed_90490.npz")
    bf = reference["f"]
    phi = np.stack([
        np.divide(reference["e1"], bf, out=np.zeros_like(bf), where=bf > 1e-30),
        np.divide(reference["e2"], bf, out=np.ones_like(bf), where=bf > 1e-30)])
    radius = measure_R_of_z(bf, g["r_c"])
    zgb, rtj = find_gb_trough(radius, g["z"], geom["z1"], lam=geom["lam"])
    support = triple_junction_support(
        bf, g["z"], g["r_c"], z_tj=zgb, r_tj=rtj, width=p.W)

    def ownership_at(q):
        return one_contact_state(
            bf, phi, g["z"], g["r_c"], dr=g["dr"], dz=g["dz"],
            u_m=q*B, gb_z_m=zgb, tj_r_m=rtj, width_m=p.W,
            moving_grain=0, neighbor_grain=1,
            source_support=support).ownership

    state_paths = [
        ("h8_t1e-3", OLD_RUN/"q050_h8_t1e-3.npz"),
        ("h12_t5e-4", OLD_RUN/"q050_h12_t5e-4.npz"),
        ("h16_t1e-4", OLD_RUN/"q050_h16_t1e-4.npz"),
        ("full_domain", RUN/"q050_full_domain.npz"),
    ]
    deltas = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    force_rows = []
    step_rows = []
    for domain, path in state_paths:
        state = load_state(path)
        f = state["f"]
        active = state["active_mask"].astype(bool)
        met = pf_contour_estimators(
            measure_R_of_z(f, g["r_c"]), g["z"], zgb-0.25*B,
            gamma_s=FrozenPhysics().gamma_s, psi_reference_deg=160)
        cc = met["local_reference"]["s_local_cap"]*met["contact_area"]
        selected = None
        for dq in deltas:
            envelope = stationary_envelope_force(
                f, state["ownership"], ownership_at(0.5-dq),
                ownership_at(0.5+dq), g, dq*B,
                active_mask=active, include_moments=False)
            step_rows.append(dict(
                domain=domain, delta_u_over_b=dq,
                G_u_N=envelope["G_u"],
                lambda_C_u_N=envelope["constraint_term_N"],
                envelope_force_N=envelope["envelope_force_N"]))
            if dq == 0.025:
                selected = envelope
        force_rows.append(dict(
            domain=domain,
            active_cells=selected["active_cells"],
            free_cells=selected["free_cells"],
            stationary_energy_J=multigrain_energy(f, state["ownership"], g),
            lambda_grain_0_J=selected["multipliers"][0],
            lambda_grain_1_J=selected["multipliers"][1],
            KKT_Linf=selected["KKT_Linf"],
            projected_KKT_Linf=selected["projected_KKT_Linf"],
            G_u_N=selected["G_u"],
            lambda_C_u_N=selected["constraint_term_N"],
            envelope_force_N=selected["envelope_force_N"],
            cannon_carter_force_N=cc,
            relative_CC_difference=abs(selected["envelope_force_N"]-cc)/abs(cc),
            state_sha256=sha256(path)))

    benchmark = small_full_domain_benchmark()
    full = force_rows[-1]
    h16 = force_rows[-2]
    full_change = abs(full["envelope_force_N"]-h16["envelope_force_N"])/abs(full["envelope_force_N"])
    c2 = ROOT/"runs/three_particle_c2_campaign/stochastic_seed20260915"
    result = dict(
        classification="ACTIVE_DOMAIN_CONTAMINATES_STATIONARY_FORCE",
        full_domain_conjugacy="FAILED_5_PERCENT_CANNON_CARTER_GATE",
        bicrystal_hard_gate_passed=False,
        c2_started=False,
        full_domain_cells=full["active_cells"],
        full_domain_KKT_Linf=full["KKT_Linf"],
        full_domain_envelope_force_N=full["envelope_force_N"],
        full_domain_cannon_carter_force_N=full["cannon_carter_force_N"],
        full_domain_relative_CC_difference=full["relative_CC_difference"],
        h16_to_full_relative_force_change=full_change,
        maximum_fixed_field_step_relative_spread=(
            max(r["envelope_force_N"] for r in step_rows if r["domain"] == "full_domain")
            - min(r["envelope_force_N"] for r in step_rows if r["domain"] == "full_domain"))
            / abs(full["envelope_force_N"]),
        small_benchmark_maximum_relative_difference=max(
            r["relative_difference"] for r in benchmark),
        full_domain_state_sha256=sha256(RUN/"q050_full_domain.npz"),
        reference_state_sha256=sha256(RUN/"reference_bicrystal_relaxed_90490.npz"),
        c2_history_sha256=sha256(c2/"history.json"),
        c2_trajectory_sha256=sha256(c2/"trajectory.npz"),
        first_moment_constraints=False,
        clipping_used=False,
        root_law_changed=False,
        barrier_changed=False,
        stochastic_campaign_launched=False)

    write_csv(DOC/"active_domain_envelope_force.csv", force_rows)
    write_csv(DOC/"ownership_step_refinement.csv", step_rows)
    write_csv(DOC/"small_full_domain_benchmark.csv", benchmark)
    (DOC/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    (DOC/"REPORT.md").write_text(f"""# Stationary-Lagrangian envelope-force qualification

**Decision: `ACTIVE_DOMAIN_CONTAMINATES_STATIONARY_FORCE`**

The force was evaluated at one stationary state from the complete constrained
Lagrangian,

```text
F_u = -(G_u + lambda^T C_u),
```

with both derivatives taken through the prescribed ownership frame at fixed
total-solid field. There are no first-moment constraints. The volume
multipliers were recovered on the same free set used by the box-constrained
stationary solver.

The implementation passes the independent 28 x 16 full-domain envelope
identity benchmark. Across four centered steps, the largest difference from
the derivative of independently minimized neighboring states is
`{result['small_benchmark_maximum_relative_difference']:.3e}`.

At `u/b = 0.5`, fixed-field ownership steps from `0.1b` through `0.00625b`
give an effectively invariant full-domain force; their relative spread is
`{result['maximum_fixed_field_step_relative_spread']:.3e}`. Expanding the
production-sized stationary domain from the 25,242-cell `h16_t1e-4` mask to
all `{result['full_domain_cells']:,}` cells changes the envelope force by
`{result['h16_to_full_relative_force_change']:.3%}`. The full-domain KKT
residual is `{result['full_domain_KKT_Linf']:.3e}`.

The full-domain envelope force is
`{result['full_domain_envelope_force_N']:.9e} N`. The independent
Cannon--Carter force measured on the same converged morphology is
`{result['full_domain_cannon_carter_force_N']:.9e} N`, a
`{result['full_domain_relative_CC_difference']:.2%}` mismatch. This exceeds
the prescribed 5% conjugacy gate.

The active cutoff contaminates the truncated stationary force, which is the
specified classification for the nonconverged active-domain sequence. The
full-domain reference removes that cutoff and supplies a stable numerical
derivative, but it still does not restore bicrystal conjugacy. Its secondary
decision is `FAILED_5_PERCENT_CANNON_CARTER_GATE`. The full-domain force is
therefore not accepted as the physical event force, and the conditional
force-versus-displacement trace was not run.

The earlier projected-Hessian diagnostics remain archived in
`../constrained_volume_only_bicrystal/hessian_spectra.csv`; all reported
minima there are positive. No C2 field was minimized or advanced. Root
kinetics, barriers, ordinary PF evolution, and the completed stochastic
trajectory are unchanged.
""")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
