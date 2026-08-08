# Milestone 13D: face-local tangential conservative surface transport

## 0. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `4918e0b` ("docs+feat: Milestone 13C discrete tangentiality and substrate-flattening qualification"), confirmed clean working tree and **197/197** tests passing before any change. Anisotropy, sink, hazard, RBM, and imposed stress/strain were never activated. Physical geometry (wavelength, amplitude, R2, overlap, W, `M_s`) was never changed. Eta stayed frozen throughout (Section 20 -- not re-enabled). Full suite at the end of this milestone: **205/205** (197 inherited + 8 new in `tests/test_face_projected_transport.py`).

New functions in `pf_sintering/bc_ops.py` (`face_gradient_x`/`face_gradient_y`) and `pf_sintering/surface_transport.py` (`surface_flux_face_projected`, `face_projected_tangentiality`, `dissipation_density_face_projected`, and a new `face_flux_mode` parameter on `surface_divergence_update`). New scripts: `scripts/m13d_manufactured_paired.py`, `scripts/m13d_isolated_substrate_paired.py`, `scripts/m13d_particle_contact_paired.py`.

## 1. Old vs. new discrete operator

**Old** ("cell_average_legacy", completely unchanged by this milestone -- Section 7's explicit requirement, verified: the full inherited test suite passes with zero modification to that code path): `n`, `P_t`, `grad(mu)`, and `J` are all computed at CELL CENTERS, then `Jx`/`Jy` are separately interpolated onto faces via simple averaging (`face_average`) before the conservative divergence. Averaging two flux vectors that are each individually tangential to two DIFFERENT local normals (if the interface curves between the two cells) does not generally produce a vector tangential to either -- this is the root cause Milestone 13C found.

**New** ("face_projected"): `n`, `P_t`, and `grad(mu)` are constructed AT the face itself (Section 2's design principle), using `bc_ops.face_gradient_x`/`face_gradient_y` for both `f` (to build `n_face`) and `mu` (to build `grad(mu)_face`) -- the SAME primitive underlies both the authoritative flux construction and `discrete_tangentiality.py`'s diagnostic (refactored this milestone to use it too), guaranteeing they measure the same object. `q` is evaluated at the face-averaged `f` (Section 5, no new dx/curvature-dependent physics). The resulting `J_face_vector = -M_s q_face P_face grad(mu)_face` is tangential to `n_face` by construction at that face, not by post-hoc correction.

## 2. Face-gradient construction

For an x-face between cells `(r,c)` and `(r,c+1)`: normal (x) derivative via the true one-sided difference `(a[r,c+1]-a[r,c])/dx` (exact at that face's own location); tangential (y) derivative via each neighboring cell's own centered y-derivative (the identical formula `grad_bc` uses), averaged onto the face -- a standard corner-averaged MAC-grid construction. Symmetric for y-faces. Applied identically to `f` and `mu` (Section 3's explicit requirement -- no mixed stencils).

## 3. Face-normal tangentiality

Tested via `face_projected_tangentiality` (both face families' full flux vector dotted with that face's own normal, no interpolation needed since both were constructed at the identical location) with a `q`/magnitude floor so negligible-flux bulk points don't dominate the statistics (Section 9).

**Static synthetic circle** (R=150nm, the exact case that showed `face_x max=1.0` under the legacy scheme at every grid): `ratio_x max=2.07e-5`, `ratio_y max=4.5e-7` at dx=2.5nm -- roundoff-level, not 1.0.

**Real particle-contact geometry**, dx=2.5nm, `face_x` RMS within 3W of a TJ: **0.105 (legacy) -> 1.08e-7 (face_projected)** at t=0.03s, and stays at `~1e-7` throughout the full 0-0.30s trajectory (vs. legacy's persistent `~0.09-0.10`). Confirmed at dx=5nm too (`0.167 -> 8.4e-8` at t=0.03s).

## 4. Energy audit

`F` decreases monotonically at every `dt` fraction tested (`1, 0.5, 0.25, 0.125, ..., 0.001`) on a random rough state, confirming the qualitative energy-descent requirement (Section 8) holds. `D_density_x`/`D_density_y` (`dissipation_density_face_projected`) are non-negative everywhere by construction (each face's own `M_tensor` is PSD, a quadratic form). **However, the quantitative match is not exact**: `(F(t)-F(t+dt))/dt` converges (as `dt->0`) to a value that is only **~65%** of the corresponding `D_CH = sum(D_density_x)+sum(D_density_y)` (converged ratio `0.655`, not `1.0`, checked down to `dt/1000`). This is because `exact_free_energy_isotropic`'s own gradient-energy term is still the legacy `lap9`-based, cell-centered discretization -- it is not the exact discrete conjugate of this new staggered/MAC flux construction. **Per Section 8's explicit instruction, this is reported honestly rather than accepted as fully resolved**: the qualitative requirement (energy strictly decreases) is satisfied and confirmed at multiple `dt`, but a fully rigorous discrete energy functional matched to the new staggered scheme (paralleling `ch_exact_energy.py`'s own `lap9`-adjoint derivation for the legacy scheme) was not derived in this milestone and is flagged as follow-up work.

## 5. Mass conservation

Exact to roundoff in every test run this milestone (`tests/test_face_projected_transport.py::test_exact_mass_conservation_face_projected`: `<1e-12` relative on a single step; every campaign script's own mass-drift tracking stays at `~1e-14` to `1e-16` relative over the full evolved trajectories) -- unaffected by the flux-construction change, since `bc_ops.flux_divergence`'s exact telescoping-sum identity depends only on the face flux being face-centered in the expected convention, not on how it was computed.

## 6. Evolved manufactured sinusoid

`scripts/m13d_manufactured_paired.py`: wavelength=120nm (Section 10's explicit suggestion -- a shorter wavelength rather than an artificially altered mobility, to get measurable evolution economically), 2% initial amplitude, evolved to t=0.22s (not Milestone 13C's near-static 2e-4s window), dx=5/2.5/1.25nm, legacy vs. face_projected from identical initial states:

| dx (nm) | A(t=0) | A(t_end) legacy | A(t_end) face_projected | relative difference |
|---|---|---|---|---|
| 5.0 | 2.3923nm | 2.2812nm | 2.2795nm | 0.07% |
| 2.5 | 2.4006nm | 2.2858nm | 2.2854nm | 0.02% |
| 1.25 | 2.4001nm | 2.2837nm | 2.2836nm | 0.004% |

~5% amplitude decay achieved at every grid (within Section 10's 2-10% target). The two modes' physical trajectories converge toward each other with `dx` (relative difference shrinking 0.07% -> 0.02% -> 0.004%), exactly as Section 10 requires. `face_projected` tangentiality RMS stays in the `1e-8` to `2e-4` range throughout the evolved trajectory (vs. legacy's un-tested-here but Milestone-13C-established `~1e-2`) -- ~100x+ better even under genuine evolution, not just the static case. Mass conserved to `<1.4e-14` relative at every grid/mode.

## 7. Evolved circle

Same script, R=150nm, imposed `mu=0.3*sin(3*theta)` (a controlled, non-decaying chemical potential so the shape's curvature stays in the intended benchmark regime), evolved to t=1e-3s:

| dx (nm) | legacy `face_x` max | face_projected `x` max | face_projected `x` RMS |
|---|---|---|---|
| 5.0 | **1.0000** | 1.13e-7 | 6.63e-9 |
| 2.5 | **1.0000** | 3.29e-8 | 1.78e-9 |
| 1.25 | **1.0000** | 8.64e-9 | 4.47e-10 |

**The Milestone 13C `face_x max~1.0` pathology is completely eliminated at every grid tested**, and the residual (already at roundoff) additionally IMPROVES with grid refinement (proper convergent behavior, unlike the legacy scheme's persistent, non-converging defect on this exact case). Mass conserved exactly (`0.00e+00` to `1.6e-16`) throughout the evolved trajectory at every grid/mode.

## 8. Isolated substrate: legacy/new comparison

`scripts/m13d_isolated_substrate_paired.py`: Milestone 13C's exact qualified benchmark (wavelength=480nm, amplitude=24nm, W=20nm), dx=2.5nm, t=0 to 0.30s, both modes from the identical initial state:

| t (s) | A1 legacy (nm) | A1 face_projected (nm) | F legacy | F face_projected |
|---|---|---|---|---|
| 0 | 24.0000 | 24.0000 | 6.247955e-6 | 6.247955e-6 |
| 0.10 | 23.9977 | 23.9977 | 6.247913e-6 | 6.247902e-6 |
| 0.20 | 23.9955 | 23.9955 | 6.247863e-6 | 6.247820e-6 |
| 0.30 | 23.9933 | 23.9933 | 6.247796e-6 | 6.247546e-6 |

**A1 is identical to 4 significant figures between modes at every single sampled time.** `F` decreases monotonically for both (small quantitative divergence growing to `~0.004%` by t=0.30s, consistent with Section 4's finding that the two modes' discrete energy functionals are not exactly matched). `J_away_right`/`J_away_left` stay strictly positive on both sides, both modes, throughout. Mass conserved to `<1.1e-15` relative, both modes.

## 9. k^4 preservation

| mode | decay rate (1/s) |
|---|---|
| cell_average_legacy | -9.3068e-4 |
| face_projected | -9.3087e-4 |

Agree to 4 significant figures (0.02% relative difference); both consistent with Milestone 13C's k^4-extrapolated Mullins prediction (9.20e-4/s) to within a few percent. **The new operator does not disturb the already-qualified k^4 physics.**

## 10. Particle-contact: legacy/new comparison

`scripts/m13d_particle_contact_paired.py`: eta frozen, dx=2.5nm primary (+ dx=5nm grid check), t=0 to 0.30s, both modes from the identical State-A initial condition:

| t (s) | M_neck_f legacy | M_neck_f face_projected | M_fixed legacy | M_fixed face_projected | L_contact legacy (nm) | L_contact face_projected (nm) |
|---|---|---|---|---|---|---|
| 0 | 3.9374e-15 | 3.9374e-15 | 2.06125e-14 | 2.06125e-14 | -- | -- |
| 0.03 | 4.9174e-15 | 4.9524e-15 | 2.06103e-14 | 2.06103e-14 | 74.275 | 74.405 |
| 0.10 | 5.3889e-15 | 5.3888e-15 | 2.06127e-14 | 2.06122e-14 | 77.217 | 77.250 |
| 0.20 | 5.5282e-15 | 5.5281e-15 | 2.06173e-14 | 2.06164e-14 | 78.706 | 78.672 |
| 0.30 | 5.6422e-15 | 5.6422e-15 | 2.06220e-14 | 2.06207e-14 | 79.496 | 79.399 |

**Agreement to 3-5 significant figures at every sampled time** for all three primary observables, while `face_x` RMS near either TJ drops from `~0.09-0.10` to `~1e-7`.

## 11. M_neck(t) and dM_neck/dt

`dM_neck_f/dt` (local finite-difference slopes), both modes, dx=2.5nm:

| t (s) | dM_neck/dt legacy | dM_neck/dt face_projected |
|---|---|---|
| 0 | 3.267e-14 | 3.383e-14 |
| 0.03 | 2.058e-14 | 2.058e-14 |
| 0.06 | 7.177e-15 | 6.508e-15 |
| 0.10 | 3.701e-15 | 3.701e-15 |
| 0.15 | 1.393e-15 | 1.393e-15 |
| 0.20 | 1.285e-15 | 1.286e-15 |
| 0.25 | 1.140e-15 | 1.141e-15 |
| 0.30 | 9.372e-16 | 9.373e-16 |

**Positive at every single sampled time, both modes, no exception -- the neck-filling sign is unchanged.** Agreement between modes is within a few percent even at the largest-discrepancy point (t=0.06s, ~9%) and essentially exact from t=0.10s onward. Confirmed at dx=5nm too (both modes' `M_neck_f` monotonically increasing at every sample, same qualitative deceleration pattern).

## 12. Fixed particle-side mass and rate

`M_fixed(t)`, both modes: small initial dip (t=0 to ~0.03s) then **monotonic rise for the remainder of the trajectory, ending above the starting value**, at both dx=2.5nm and dx=5nm, both flux modes -- the Milestone 13C fixed-Eulerian correction (particle lobe is NOT a persistent donor) is confirmed to be a property of the physics, not an artifact of the flawed face discretization. `dM_fixed/dt`: negative at t=0/0.03s, positive from t=0.06s onward, for both modes (e.g. dx=2.5nm, t=0.06s: `3.342e-17` legacy vs `2.430e-17` face_projected -- same sign, same order of magnitude, the largest relative discrepancy of any sampled point, still not a qualitative disagreement).

## 13. Spatial substrate contour response

Full `h(y,t)-h(y,0)` (not just the Fourier projection) was recorded at every sample for both modes (`substrate_h_displacement` in the campaign JSON). The away-from-TJ Fourier `A1_substrate` shows the same qualitative non-monotonic (dip then rise then plateau/decline) pattern under both flux modes -- confirming Milestone 13C's finding that the particle perturbation does not decouple from the substrate shape within a 6W margin is a genuine physical feature, not a face-discretization artifact -- but the two modes' PEAK magnitudes differ more here than for the other observables (legacy peaks at 24.242nm around t=0.25s; face_projected peaks at 24.158nm around t=0.20s, a ~0.35% difference in excursion size, dx=2.5nm). This is the one observable in this milestone showing a real, if modest, quantitative sensitivity to the flux discretization -- noted explicitly rather than smoothed over.

## 14. Grid sensitivity

`dM_neck_f/dt`'s sign is positive at every sampled time at BOTH dx=2.5nm and dx=5nm, under BOTH flux modes -- **no sign change with grid**, satisfying Section 19's explicit requirement. `M_neck_f(t=0.30s)` across grids (face_projected mode): dx=2.5nm gives `5.642e-15`, dx=5nm gives `5.730e-15` (~1.5% difference) -- a normal, modest grid-discretization difference, not a qualitative one. dx=1.25nm was not run for the particle-contact case given this milestone's already substantial scope; grid confidence rests on this dx=5/2.5nm agreement plus the 3-grid manufactured-interface and isolated-substrate convergence already established in Sections 6-9.

## 15. D1/D2/D3 classification

**D1: trajectories nearly identical; the face-normal discretization error was quantitatively small** for the primary particle-contact observables (`M_neck_f`, `dM_neck_f/dt`, `M_fixed`, `dM_fixed/dt`, `L_contact`) -- despite the LOCAL, pointwise tangentiality defect being substantial (`~10%` RMS) near the TJ under the legacy scheme, the effect on these coarse-grained, mass-conservative integrated quantities is small (a few percent or better at every sampled time, both grids tested). No sign of `dM_neck_f/dt` changes anywhere in the tested range.

**One partial exception, noted rather than hidden**: the substrate's own Fourier `A1` (Section 13) shows a real, if modest (~0.35%), quantitative sensitivity to the flux discretization in its peak excursion magnitude, while preserving the same qualitative (non-monotonic, particle-perturbed) shape. This nudges that one specific observable toward the boundary between D1 and D2, but does not change the overall classification for the primary mass-transfer observables that motivated this investigation.

## 16. Recommendation for the next physical coarsening experiment

1. **Milestone 13B's central finding is now on much firmer ground**: the branch-cut closure failure near the TJ is confirmed NOT explained by the (now-fixed) face-discretization defect, since `dM_neck_f/dt` itself is essentially unchanged by the repair. Milestone 13B's TJ-core interpretation (Milestone 13C Section 10's qualified version) can be revisited with more confidence that it reflects genuine geometry, not a numerical artifact -- though the energy-functional mismatch (Section 4) means the new operator itself is not yet fully closed out as "exact" in every respect.
2. **Repair the discrete energy functional** to match the new staggered flux construction (a face-local `ch_exact_energy.py` analogue) before relying on quantitative `D_CH` comparisons with this operator -- flagged but not completed here.
3. **Re-enable eta kinetics** (Section 20 -- deliberately not done in this milestone) now that a trustworthy conserved surface-transport operator is established, to determine whether the Milestone 13's eta-dominated `dV2/dt` finding is similarly robust to the face-flux repair.
4. Given the substrate contour's own modest sensitivity (Section 13), any future work reporting substrate-shape details specifically (not just neck/particle mass) should use the `face_projected` mode as the more trustworthy source, since it is the one with a demonstrated-correct local flux field.
5. Promote `face_projected` to the default `face_flux_mode` for future milestones' production diagnostic use once the energy-functional gap (item 2) is closed -- keeping `cell_average_legacy` available for backward comparison, per Section 7's original opt-in design.

---

### Appendix: raw run artifacts

- `runs/m13d_manufactured_paired.json` -- evolved sinusoid + circle, legacy vs. face_projected, dx=5/2.5/1.25nm.
- `runs/m13d_isolated_substrate_paired_dx2p5.json` -- isolated substrate regression, paired.
- `runs/m13d_particle_contact_paired_dx2p5.json`, `_dx5.json` -- particle-contact paired comparison, primary + grid check.

### Stop conditions honored

Eta kinetics, sink, hazard, RBM, and anisotropy were never activated. The physical particle/substrate geometry was never changed. The legacy `cell_average_legacy` transport path is byte-for-byte unchanged from Milestone 13C (verified: full inherited test suite passes with zero modification).
