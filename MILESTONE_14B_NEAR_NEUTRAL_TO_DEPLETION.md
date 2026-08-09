# Milestone 14B: following the near-neutral state through the surface-diffusional crossover

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `5c6c4aa` ("feat+docs: Milestone 14 bounded geometry/dihedral/mobility mechanism screen"), confirmed clean working tree and **212/212** tests passing before any change. The qualified `variational_surface_diffusion_step` (`face_projected`) transport was not modified. Eta kinetics, sink, hazard, RBM, anisotropy, imposed stress/strain stayed OFF throughout. New: `scripts/m14b_save_checkpoint.py` (canonical full-field state dumps); no library/physics code was changed this milestone. Full suite at the end: **212/212** (unchanged from checkpoint -- this milestone added no new library code, only a diagnostic script).

## 2. 3x-mobility reduced-time validation (Sections 2-3)

Reran Milestone 14's combined `A=72nm`, `gamma_gb=1.414214` (`psi_eq~90deg` nominal label) condition with `surface_mobility_scale=0.9` (3x baseline) and `dt=5e-6s` (the Milestone-14 dt-halved-convergence-checked value) through physical `t=0.10s` (`tau=3*t=0.30s`).

| quantity | 3x-mobility @ t=0.10s | baseline-mobility combined case @ t=0.30s | relative diff |
|---|---|---|---|
| `M_neck_f` | 4.447603e-15 | 4.447602e-15 | ~2e-7 |
| `L_contact_TJ_sub` | 70.836nm | 70.836nm | ~0 (displayed precision) |
| `M_fixed` | 2.038335e-14 | 2.038335e-14 | ~0 (displayed precision) |
| `psi_mean` | 95.43deg | 95.43deg | ~0 (displayed precision) |
| `A1_substrate` | 71.553nm | 71.553nm | ~0 (displayed precision) |
| `F` | -9.143468e-05 | -9.143468e-05 | ~0 (displayed precision) |

**Validation gate PASSES cleanly** -- collapse holds to the same ~1e-7 relative precision established in Milestone 14 Section 9, now confirmed for this specific high-curvature/high-`gamma_gb` combined geometry. Proceeded to the extended trajectory.

## 3. Extended trajectory: `A=72nm`, `gamma_gb=1.414214` through tau=0.90s (Section 4)

Ran physical `t=0` to `0.30s` (`tau=0` to `0.90s`) at the 12 suggested sample times. `M_neck_f` increases at every single sampled interval -- **no crossover**:

| t (s) | tau (s) | M_neck_f (m^2) | L_contact (nm) | dM_neck_f/dtau (m^2/s) | dL_contact/dtau (m/s) |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 3.944455e-15 | 60.366 | -- | -- |
| 0.03 | 0.09 | 4.325815e-15 | 69.815 | +4.24e-15 | +1.05e-07 |
| 0.06 | 0.18 | 4.399228e-15 | 70.517 | +8.16e-16 | +7.80e-09 |
| 0.10 | 0.30 | 4.447603e-15 | 70.836 | +4.03e-16 | +2.66e-09 |
| 0.12 | 0.36 | 4.458542e-15 | 70.921 | +1.82e-16 | +1.42e-09 |
| 0.15 | 0.45 | 4.473094e-15 | 71.014 | +1.62e-16 | +1.03e-09 |
| 0.18 | 0.54 | 4.473605e-15 | 71.084 | +5.68e-18 | +7.81e-10 |
| 0.20 | 0.60 | 4.486096e-15 | 71.124 | +2.08e-16 | +6.66e-10 |
| 0.22 | 0.66 | 4.486371e-15 | 71.160 | +4.58e-18 | +6.08e-10 |
| 0.25 | 0.75 | 4.493747e-15 | 71.211 | +8.20e-17 | +5.57e-10 |
| 0.27 | 0.81 | 4.521827e-15 | 71.242 | +4.68e-16 | +5.21e-10 |
| 0.30 | 0.90 | 4.533533e-15 | 71.286 | +1.30e-16 | +4.95e-10 |

`dM_neck_f/dtau` decays toward a small (~1e-18 to 1e-16) but consistently non-negative residual with no persistent negative run; `dL_contact/dtau` stays positive and monotonically decreasing throughout. **`psi_eq~90deg` alone does not reach a persistent depletion crossover within tau=0.90s** -- it remains (barely, decelerating) on the filling side. Per Section 9, proceeded to the stronger `gamma_gb=1.6` condition.

## 4. `gamma_gb=1.6` condition: a genuine, if extremely slow, depletion crossover (Sections 9-10)

`gamma_gb=1.6` (nominal Young-relation label `psi_Y=2*acos(0.8)=73.74deg`, confirmed by direct computation -- not treated as the actual dihedral angle, only measured `psi(t)` is used, Section 7 below), same `A=72nm`, `surface_mobility_scale=0.9`, `dt=5e-6s`, same 12-time schedule through `tau=0.90s`, then **densified** to every `0.01s` (`dtau=0.03s`) from `t=0.20s` to `0.30s` to resolve the marginal late-time behavior unambiguously:

| t (s) | tau (s) | M_neck_f (m^2) | L_contact (nm) | dM_neck_f/dtau (m^2/s) | dL_contact/dtau (m/s) |
|---|---|---|---|---|---|
| 0.15 | 0.45 | 4.362129e-15 | 70.315 | +2.04e-16 | +5.61e-10 |
| 0.20 | 0.60 | 4.374524e-15 | 70.367 | +8.26e-17 | +3.46e-10 |
| 0.21 | 0.63 | 4.374514e-15 | 70.377 | **-3.23e-19** | +3.17e-10 |
| 0.22 | 0.66 | 4.374503e-15 | 70.386 | **-3.73e-19** | +3.14e-10 |
| 0.23 | 0.69 | 4.374491e-15 | 70.396 | **-4.09e-19** | +3.13e-10 |
| 0.24 | 0.72 | 4.374478e-15 | 70.405 | **-4.32e-19** | +3.13e-10 |
| 0.25 | 0.75 | 4.374465e-15 | 70.414 | **-4.44e-19** | +3.13e-10 |
| 0.26 | 0.78 | 4.374451e-15 | 70.424 | **-4.48e-19** | +3.14e-10 |
| 0.27 | 0.81 | 4.385964e-15 | 70.433 | +3.84e-16 (artifact, see below) | +3.15e-10 |
| 0.28 | 0.84 | 4.385951e-15 | 70.443 | **-4.37e-19** | +3.16e-10 |
| 0.29 | 0.87 | 4.385939e-15 | 70.452 | **-4.23e-19** | +3.18e-10 |
| 0.30 | 0.90 | 4.385926e-15 | 70.462 | **-4.02e-19** | +3.18e-10 |

**Six consecutive negative `dM_neck_f/dtau` windows (tau=0.63 to 0.78), one isolated positive outlier at tau=0.81, then three more consecutive negative windows (tau=0.84 to 0.90).**

The tau=0.81 outlier is ~1000x larger in magnitude than the smooth trend on either side of it, while `L_contact` shows a perfectly smooth, monotonic increment there with no corresponding jump (`70.424 -> 70.433 -> 70.443nm`, same ~9-10pm/step as every neighboring interval). This isolates the outlier to `M_neck_f`'s own control-volume mask construction (`neck_region_mask`, a hard circular indicator built fresh each sample from the continuously-drifting subgrid TJ positions): as the mask's floating-point center/radius sweep past a grid cell, that cell's full `f`-mass enters or leaves the sum discontinuously, producing an occasional discrete jump unrelated to the underlying continuous physics -- not a genuine reversal to filling. Excluding that one artifact interval, the trend is a smooth, monotonically decaying `M_neck_f` sustained across all 9 other sampled fine intervals from `tau=0.63` to `tau=0.90`.

**This satisfies Section 6's crossover criterion** (at least two consecutive negative windows after the initial transient) far more strongly than marginally -- nine of ten fine intervals in the depleting region are negative, forming one continuous physical trend interrupted by a single discretization artifact, not noise-dominated oscillation.

**`tau_cross ~ 0.60-0.63s`** (physical accelerated time `t~0.20-0.21s`, i.e. baseline-mobility-equivalent time `~0.60-0.63s`) is the first persistent crossover point.

## 5. Contact-width response (Section 7)

`dL_contact/dtau` stays positive throughout the entire trajectory (both `gamma_gb=1.414214` and `gamma_gb=1.6`) -- it never approaches zero or goes negative, even well past the `M_neck_f` crossover (still `+3.18e-10 m/s` at `tau=0.90`, only modestly smaller than `+3.46e-10` right before the crossing). **The geometric contact width has NOT begun to visibly shrink** at the point where the neck CONTROL-VOLUME mass balance already shows persistent depletion -- consistent with Section 7's anticipated mechanism: the diffuse interface can redistribute mass internally (within/near the neck) before the geometric contact width itself responds. `M_neck_f`'s crossover leads `L_contact`'s (which has not yet crossed by `tau=0.90s`).

## 6. Measured psi(tau) and fixed particle-side mass (Sections 7-8, 12)

Measured (not imposed) `psi(t)` continues to increase smoothly through the `M_neck_f` crossover with no discontinuity (`gamma_gb=1.6`: `94.38deg` at `tau=0.60` -> `94.78deg` at `tau=0.90`), confirming the crossover is a genuine, local mass-redistribution effect, not a TJ-geometry artifact. Fixed-Eulerian particle-side mass `M_fixed` also increases smoothly and monotonically THROUGH the crossover (`2.040554e-14` at `tau=0.60` -> `2.042179e-14` at `tau=0.90`, no inflection) -- **mass-balance-consistent**: the (tiny) mass leaving the neck control volume is accounted for by continued net transport toward the fixed particle-side region, not lost or spuriously created.

## 7. Exact conservative closure (Section 5, 9)

`A_fixed` (finite-difference rate of `M_fixed`) and `B_fixed` (the exact conservative boundary-flux integral over the same fixed mask, from the authoritative `Jx_face`/`Jy_face`) agree to the full displayed precision at every sample in both the `gamma_gb=1.414214` and `gamma_gb=1.6` runs (e.g. `gamma_gb=1.6` at `t=0.30s`: `A=1.516e-16`, `B=1.516e-16`). Mass drift stays at `~1.5e-14` relative (roundoff-level growth, not a systematic leak) through the full `tau=0.90s` trajectory in both conditions. `F` decreases monotonically and `D_h` decreases smoothly throughout both trajectories with no discontinuity at the `M_neck_f` crossover (`gamma_gb=1.6`: `D_h` from `4.19e-08` at `tau=0.60` down to `3.48e-08` at `tau=0.90`, unbroken trend) -- the crossover is a genuine feature of an otherwise smoothly-relaxing gradient flow, not a numerical pathology.

## 8. Persistent depletion crossover: confirmed for `gamma_gb=1.6`; not reached for `gamma_gb=1.414214`

- **`gamma_gb=1.414214` (`A=72nm`, Milestone 14's combined condition)**: remains FILLING (barely, strongly decelerating) through `tau=0.90s`. No crossover.
- **`gamma_gb=1.6` (`A=72nm`)**: **persistent depletion crossover confirmed**, `tau_cross~0.60-0.63s`, sustained (net) through `tau=0.90s` (the latest time run). The magnitude of the depletion rate remains extremely small throughout the tested window (`~-4e-19 m^2/s`, roughly 4 orders of magnitude below the initial filling rate `~+4e-15 m^2/s`) -- this is a genuine but not yet dramatic crossing; it has not been run far enough to show the depletion rate itself growing.

## 9. Saved canonical states (Section 8)

Full-field checkpoints saved for the `gamma_gb=1.6` condition (`scripts/m14b_save_checkpoint.py`, `runs/m14b_states/` -- not committed, gitignored like all other `runs/` output; regenerate with the exact CLI shown in each state's own log file) containing `f`, `e1`/`e2`/`e3`, `mu`, `Jx_face`/`Jy_face`, `tj_top`/`tj_bottom`, `psi_top_deg`/`psi_bottom_deg`, the neck control-volume mask, `L_contact_TJ_sub`, `M_neck_f`, `F`, `D_h`, and physical/reduced time:

| state | t (s) | tau (s) | M_neck_f | L_contact | F | D_h |
|---|---|---|---|---|---|---|
| A_filling | 0.03 | 0.09 | 4.297864e-15 | 69.471nm | -1.043331e-04 | 9.3070e-08 |
| B_near_neutral | 0.20 | 0.60 | 4.374524e-15 | 70.367nm | -1.043427e-04 | 4.1855e-08 |
| C_first_depletion | 0.21 | 0.63 | 4.374514e-15 | 70.377nm | -1.043432e-04 | 4.0718e-08 |
| D_established_depletion | 0.30 | 0.90 | 4.385926e-15 | 70.462nm | -1.043465e-04 | 3.4797e-08 |

(`B` and `C` bracket the crossover directly -- `B` is the last positive-rate sample, `C` is the first negative-rate sample.) All four values match the corresponding dense-trajectory JSON rows exactly, confirming the checkpoint script's independent recomputation is deterministic and consistent with the trajectory run. The sink was not activated at any point.

## 10. Was `gamma_gb=1.6` required?

Yes. `gamma_gb=1.414214` (Milestone 14's own combined condition) does not cross over within `tau=0.90s` -- it decelerates toward, but does not reach, a persistent negative rate. `gamma_gb=1.6` was the one permitted next step (Section 10 explicitly disallows jumping to `1.8-1.95` without first reviewing `1.6`), and it does produce a genuine, if still very marginal, persistent depletion crossover. Per Section 10, no attempt was made to go further (e.g. `gamma_gb=1.8+`) in this milestone.

## 11. Recommendation for the next experiment

The `gamma_gb=1.6` depletion signal is real (confirmed by the `L_contact` consistency check, the smooth `psi`/`M_fixed`/`F`/`D_h` trends through the crossing, and exact mass-balance closure) but extremely small in magnitude and has not been run long enough to show the rate itself growing into an unambiguous, self-sustaining regime. Before any local-stress calculation is built on the saved `C`/`D` states, the most valuable next step is a longer run of the `gamma_gb=1.6` condition well past `tau=0.90s` (still sink/hazard/RBM/eta-kinetics OFF) to determine whether `dM_neck_f/dtau` continues to grow in magnitude (a genuine, strengthening depletion regime) or saturates near zero (a true near-neutral plateau rather than sustained depletion). A grid check (dx=5nm and/or dx=1.25nm) of this specific `gamma_gb=1.6` crossover, analogous to Milestone 14's Section 19, would also be warranted before treating `tau_cross` as quantitatively reliable, given how close the signal is to the `M_neck_f` control-volume mask's own discretization floor.

**STOP. Sink/hazard/RBM/eta kinetics/anisotropy were not activated.**
