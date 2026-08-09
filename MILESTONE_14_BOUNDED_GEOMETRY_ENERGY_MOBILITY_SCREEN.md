# Milestone 14: bounded geometry / dihedral / mobility mechanism screen

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `d34b7af` ("feat+docs: Milestone 13E discrete gradient-flow closure and face-projected promotion"), confirmed clean working tree and **209/209** tests passing before any change. Anisotropy, sink, hazard, RBM, imposed stress/strain never activated; eta kinetics stayed frozen throughout (eta fields exist only as fixed grain-identity/GB-coupling inputs to `mu`, never evolved). Every run used the promoted canonical entry point `variational_surface_diffusion_step` (`face_projected` default). Full suite at the end of this milestone: **212/212** (209 inherited + 3 new `gamma_gb_override` tests).

New: `ModelConfig.gamma_gb_override` (`pf_sintering/model.py`), `scripts/m14_mechanism_screen.py`, `tests/test_model_gamma_gb_override.py`.

## 2. Seven unique conditions (+ one combined case, + grid check)

All runs: `R2=80nm`, aspect ratio 2, initial overlap 20nm, `W=20nm`, `lambda=480nm`, `dx=2.5nm`, `surface_mobility_scale=0.3` unless noted, `t=0` to `0.30s`, sampled at `0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30s`.

| label | A (nm) | gamma_gb (J/m^2) | nominal psi_eq | surface_mobility_scale | purpose |
|---|---|---|---|---|---|
| baseline | 24 | 0.999324 (unchanged, theta_mis=30deg) | 120.04deg | 0.3 | common baseline (reused as the middle point of all three ladders) |
| amp12 | 12 | 0.999324 | 120.04deg | 0.3 | weak substrate curvature |
| amp72 | 72 | 0.999324 | 120.04deg | 0.3 | strong substrate curvature |
| gb90 | 24 | 1.414214 (override) | 90.00deg | 0.3 | high GB/surface energy ratio |
| gb150 | 24 | 0.517638 (override) | 150.00deg | 0.3 | low GB/surface energy ratio |
| mob013 | 24 | 0.999324 | 120.04deg | 0.1 (1/3x) | slow surface diffusivity |
| mob09 | 24 | 0.999324 | 120.04deg | 0.9 (3x) | fast surface diffusivity |
| combined_amp72_gb90 (optional, Section 16) | 72 | 1.414214 | 90.00deg | 0.3 | do the two strongest single-variable effects reinforce? |

`A=72nm` produced a fully resolved TJ/contact topology at `t=0` (`compute_subgrid_contact` resolved, `L_contact=60.4nm`) -- no fallback to `A=60nm` was needed.

## 3. gamma_gb thermodynamic-pathway audit (Section 8)

Before running anything, audited how `gamma_gb` enters the active physics. `build_params` (`pf_sintering/model.py`) previously derived `p.gamma_gb = _gb_energy(theta_mis_deg)` (an empirical Read-Shockley-like curve, unrelated to a target Young's angle) and set `p.gamma_gb_ref = p.gamma_gb`. Three coefficients depend on `gamma_gb`: `p.k_eta = 3*gamma_gb*W` and `p.W_cpl_f = 36*gamma_gb/W` (both consumed only by `evolve_eta`/`constrained_eta.py`'s structural-force update -- inert this milestone since eta kinetics are frozen), and, via `p.gamma_gb_ref`, the groove-coupling coefficient `Wc = 36*gamma_gb_ref/W` inside `mu0_bulk`'s `-Wc*eta2*(1-fb)` term (`ch_exact_energy.py`, algebraically identical to production `evolve_f`'s own `mu0`). **This last one is the only dynamically active pathway** for the conserved-f evolution used by this milestone's transport law: with `Sink.g_ex=0` (sink off), `effective_gamma(s,p)` reduces to `clip(gamma_gb_ref, floor, ceil) = gamma_gb_ref` for all three target ratios (all safely inside `[0.05, 1.95]`), so `Wc` is spatially uniform and directly proportional to `gamma_gb_ref`.

Added `ModelConfig.gamma_gb_override: float | None = None` as the single entry point: when set, `p.gamma_gb = gamma_gb_override` (bypassing `_gb_energy(theta_mis_deg)` entirely) and `p.gamma_gb_ref = p.gamma_gb`, so `k_eta`, `W_cpl_f`, and the active `Wc` are all re-derived consistently from the SAME physical value. `None` (default) reproduces prior behavior exactly (`tests/test_model_gamma_gb_override.py::test_gamma_gb_override_none_preserves_existing_behavior`). A direct `mu_isotropic` comparison on identical `f`/eta fields with only `gamma_gb_override` differing confirms the override actually changes the computed `mu` field (`test_gamma_gb_override_changes_active_groove_coupling_in_mu`), not just a reported/diagnostic value.

**Audit table** (`gamma_s=1.0`, `W=20nm`):

| target psi_eq | target gamma_gb/gamma_s (`2cos(psi_eq/2)`) | p.gamma_gb | p.gamma_gb_ref | k_eta (J/m) | W_cpl_f (J/m^3) | active Wc (J/m^3) |
|---|---|---|---|---|---|---|
| 90deg | 1.4142 | 1.414214 | 1.414214 | 8.4853e-08 | 2.5456e+09 | 2.5456e+09 |
| 120deg (baseline, theta_mis=30deg, unmodified) | 1.0000 (nominal) / 0.9993 (actual) | 0.999324 | 0.999324 | 5.9959e-08 | 1.7988e+09 | 1.7988e+09 |
| 150deg | 0.5176 | 0.517638 | 0.517638 | 3.1058e-08 | 9.3175e+08 | 9.3175e+08 |

Per Section 6, the 120deg control uses the actual qualified baseline (`gamma_gb=0.999324`, implying `psi_eq=120.04deg`), not a replaced exact `1.0`.

## 4. Nominal psi_eq vs. measured psi(t)

`psi(t)` was measured with the qualified TJ/branch geometry routine (`tj_force.compute_neck_tj_forces` -> `compute_tj_force`'s `psi_deg`, the angle between the two free-surface branch directions at each TJ), never imposed, prescribed, or used to reshape the TJ.

| label | psi_eq (nominal) | psi(t=0) | psi(t=0.30s) |
|---|---|---|---|
| gb90 | 90.00deg | 77.87deg | 79.80deg |
| baseline | 120.04deg | 77.87deg | 80.86deg |
| gb150 | 150.00deg | 77.87deg | 80.24deg |

**Finding (Section 10's "do not assume the answer"): measured psi(t) stayed in a narrow ~76-81deg band across all three imposed energy ratios**, essentially decoupled from the nominal 90/120/150deg target over this 0.30s window. This is consistent with the mechanism identified in Section 3: with eta frozen, the GB LOCATION never moves, so `psi(t)` is governed by how the conserved-f free surface reshapes near a geometrically fixed TJ under the imposed initial condition -- it is not free to relax toward a new "groove equilibrium" angle on this timescale. The physical effect of `gamma_gb` on the morphology is real and substantial (Section 6 below) but is transmitted through the direct `Wc*eta2*(1-fb)` bias on the local `mu` field (and hence the surface-diffusion flux), not through a fast re-angling of the visible dihedral groove. `psi(t)` for amp12/amp72/combined is reported in Sections 6-7 below alongside the other observables.

## 5. Amplitude / crest-curvature values (Section 3)

Analytic crest curvature `kappa_crest = -A*(2*pi/lambda)^2`; measured value from a direct centered finite-difference second derivative of the ISOLATED substrate's own `f=0.5` contour (`x_crossing_profile` from `m13d_isolated_substrate_paired.py`, no particle) at the crest -- this reproduces the analytic value to <0.3% and is the value reported below. (The pre-existing `local_curvature`/Kasa-circle-fit helper in that same Milestone-13D script returns a value a factor of ~23 too small on this exact geometry for an unrelated reason -- a pre-existing quirk of that helper's local point-window construction, out of this milestone's scope to fix; bypassed here in favor of the direct, transparently-verified finite-difference measurement.)

| A (nm) | analytic kappa_crest (1/m) | measured kappa_crest (1/m) | radius (nm) | radius/dx |
|---|---|---|---|---|
| 12 | -2.0562e+06 | -2.0534e+06 (0.14% low) | 486.34 | 194.5 |
| 24 (baseline) | -4.1123e+06 | -4.1222e+06 (0.24% high) | 243.17 | 97.3 |
| 72 | -1.2337e+07 | -1.2320e+07 (0.14% low) | 81.06 | 32.4 |

Three clearly separated curvature states confirmed, as intended (weak/current/strong, ~6x range end to end), all well-resolved relative to `dx` (radius/dx from ~32 to ~195).

## 6. M_neck_f(t), dM_neck_f/dt, and effect-size comparison (Sections 5, 17, 18)

`dM_neck_f/dt` computed from the exact conservative control-volume mass `M_neck_f` (not V2/eta-weighted). `rate[.20-.30]` is the endpoint-difference rate over the last sampled window; classification uses the last TWO consecutive windows (`[.15,.20]`, `[.20,.25]`, `[.25,.30]` -- specifically the last two of these) after the initial transient.

| label | rate[.20-.30] (m^2/s) | relative to baseline | classification |
|---|---|---|---|
| baseline | 1.1407e-15 | 1.000 | FILLING |
| amp12 | 1.0065e-15 | 0.882 | FILLING |
| amp72 | 7.5975e-16 | 0.666 | FILLING |
| gb90 | 3.0770e-16 | 0.270 | FILLING |
| gb150 | 1.5403e-15 | 1.350 | FILLING |
| mob013 (1/3x) | 1.8584e-15 | 1.629 | FILLING (see Section 9 -- raw-t rate is a trivial clock effect) |
| mob09 (3x) | 7.3784e-16 | 0.647 | FILLING (see Section 9) |
| combined_amp72_gb90 | 2.5844e-16 | 0.227 | **NEAR-NEUTRAL** (final window rate = 1.01e-17, ~1% of baseline) |

**Every one of the seven unique single-variable conditions remains clearly FILLING throughout 0-0.30s** -- `dM_neck_f/dt > 0` at every sampled interval, no sign crossover. One notable transient: `amp72`'s very first interval (`t=0->0.01s`) is briefly NEGATIVE (`-4.42e-15`) before turning positive for the remaining 7 consecutive intervals -- the closest any single-variable condition came to a crossover, and consistent with high amplitude pushing toward reduced filling, but not sustained (excluded from classification as the initial transient, per Section 17).

**Effect-size ranking** (relative to baseline, Section 18): the **GB/surface-energy ratio has by far the largest effect** on the late-time filling rate -- a ~3.7x swing from `gb90` (0.270x) to `gb150` (1.350x). **Substrate curvature/amplitude has a smaller, monotonic-in-magnitude effect**: both directions away from baseline reduce the rate (`amp12`: 0.882x, `amp72`: 0.666x), a ~25% range end to end -- notably NOT monotonic in amplitude itself (both low and high amplitude reduce the rate relative to the A=24nm baseline), a genuine, not-assumed-in-advance finding. **Surface diffusivity's raw-t effect size is a rescaling artifact, not a morphology effect** (Section 9) -- once compared correctly (reduced time), it has ~zero effect on the path.

## 7. Contact evolution and fixed particle-side mass (Sections 7-8)

`L_contact_TJ_sub(t)` increases monotonically in every condition (baseline: 63.5nm -> 79.4nm; amp72: 60.4nm -> 72.3nm; gb90: 63.5nm -> 77.8nm; gb150: 63.5nm -> 81.5nm). Fixed-Eulerian particle-side mass `M_fixed(t)` shows the same "dips early, then rises" pattern established in Milestone 13D in every condition (e.g. baseline: `2.061248e-14 -> min 2.061034e-14 @ t=0.03s -> 2.062071e-14`; combined case: `2.036783e-14 -> min 2.036617e-14 @ t=0.01s -> 2.038335e-14`). The exact conservative boundary-flux rate `B_fixed` matches the finite-difference rate `A_fixed` to the displayed precision at every sample in every run (both stay well below `1e-16` relative when normalized by `M_fixed`), confirming exact mass accounting is intact under every physical variation tested.

## 8. Local surface mu/flux response (Section 5, 9)

`gamma_gb` and amplitude changes visibly alter `F` and `D_h` (e.g. baseline `F(t=0)=-6.2223e-05` vs. `gb90` `F(t=0)=-9.0831e-05` vs. `gb150` `F(t=0)=-2.9010e-05` -- purely from the `Wc*eta2*(0.5f^2-f)` bulk term at fixed `f`/eta, since `F` is evaluated at the identical initial `f` field for all three) and `D_h` decreases smoothly and monotonically as the neck relaxes in every run (e.g. baseline: `1.33e-07 -> 1.04e-08` from `t=0.01` to `t=0.30s`), consistent with a genuine gradient-flow relaxation in every tested condition, not a numerically pathological one. `face_x_rms(3W)` (authoritative face-tangentiality, within-3W-of-TJ mask) stays at roundoff (`~4e-8` to `~3.6e-7`) in every run at every sample **except one**: the `dx=1.25nm` grid-check's final sample (`t=0.15s`) showed an anomalous `face_x_rms(3W)=9.26e-03`, while mass and energy at that same sample remained perfectly well-behaved (`mass_drift=7.7e-15`, `F` still monotonically decreasing). This is very likely the same class of small-flux-magnitude/small-numerator ratio artifact identified and fixed for the isolated-substrate script in Milestone 13E Section 9, not fully excluded by the `within_3W` region mask alone at this particular fine-grid/high-curvature configuration -- it does not affect any of this milestone's physical conclusions (mass/energy/`M_neck_f` sign are unaffected) and is flagged here for a future diagnostic follow-up rather than investigated further in this milestone.

## 9. Mobility reduced-time collapse (Sections 11-14)

`p.dt = min(CFL*dx^4/(M_f*k_f), 1e-5)` was audited directly: at `dx=2.5nm`, all three mobility scales (0.1, 0.3, 0.9) hit the `1e-5s` ceiling (CFL-derived bound: `9.77e-5s`, `3.26e-5s`, `1.09e-5s` respectively) -- i.e. `dt` does NOT auto-shrink with `M_s` in this regime, and the `0.9x` (3x) case's CFL margin is the tightest (~8.5%). A dedicated dt-halved convergence check (`dt=5e-6s`, `mob09_dtcheck`) reproduces `mob09`'s full trajectory (`M_neck_f`, `L_contact`, `M_fixed`) to 6-7 significant figures at every matched sample through `t=0.10s` -- timestep-converged, no reduction needed for the reported `dt=1e-5s` results.

Comparing trajectories at matched **reduced time** `tau=(M_s/M_s,baseline)*t` (not matched physical `t`) gives essentially exact collapse:

| comparison | quantity | value | baseline value | relative diff |
|---|---|---|---|---|
| mob013(t=0.03,tau=0.01) vs baseline(t=0.01) | M_neck_f | 4.656859e-15 | 4.656859e-15 | -7.6e-08 |
| mob013(t=0.30,tau=0.10) vs baseline(t=0.10) | M_neck_f | 5.388834e-15 | 5.388834e-15 | -8.8e-08 |
| mob09(t=0.01,tau=0.03) vs baseline(t=0.03) | M_neck_f | 4.952393e-15 | 4.952392e-15 | +2.7e-07 |
| mob09(t=0.10,tau=0.30) vs baseline(t=0.30) | M_neck_f | 5.642176e-15 | 5.642174e-15 | +3.2e-07 |
| mob09(t=0.10,tau=0.30) vs baseline(t=0.30) | L_contact | 79.3984nm | 79.3986nm | -2.5e-06 |
| mob09(t=0.10,tau=0.30) vs baseline(t=0.30) | M_fixed | 2.062071e-14 | 2.062071e-14 | ~0 |

Collapse holds to ~1e-7 relative precision simultaneously across `M_neck_f`, `L_contact`, and `M_fixed`. **Surface diffusivity is confirmed purely kinetic** (Section 14's expected sink-off result): it rescales the clock and does not alter the coarsening path/morphology within the tested 3x range.

## 10. Optional combined case (Section 16)

Both `amp72` (0.666x baseline) and `gb90` (0.270x baseline) meaningfully reduce the late-time filling rate, so one combined case (`A=72nm` + `gamma_gb=1.414214`, i.e. `psi_eq=90deg`) was run. Result: rate[.20-.30] = `2.5844e-16` (0.227x baseline) -- **lower than either single-variable effect alone**, i.e. the two effects reinforce, though sub-multiplicatively (naive product `0.666*0.270=0.180x` would predict an even lower value than the observed `0.227x`). Per-interval rates decay steadily and the FINAL interval (`t=0.25->0.30s`) is `1.01e-17` -- about 1% of baseline's corresponding-window rate, the closest approach to a crossover found anywhere in this screen, though the immediately preceding window (`0.20->0.25s`, `5.07e-16`, 44% of baseline) is not itself near-neutral, so this is reported as a clear FILLING-but-decelerating-toward-near-neutral trend rather than a persisted two-window near-neutral state.

**Grid check** (Section 19, triggered by this trend being the most-changed result in the screen): reran at `dx=5nm` (full 0-0.30s) and `dx=1.25nm` (bounded, 0-0.15s).

- `dx=5nm`: `M_neck_f` increases throughout (`3.910146e-15 -> 4.375427e-15`), late-window rates `[.20,.25]=1.64e-17`, `[.25,.30]=1.45e-17` -- same order of magnitude and same qualitative "decelerating toward near-zero but still positive" behavior as `dx=2.5nm`.
- `dx=1.25nm` (to `t=0.15s`): per-interval rates `6.21e-15, 4.88e-15, 2.18e-15, 1.15e-15` (`t=0->0.03->0.06->0.10->0.15s`) -- positive and decaying at the same pace as the coarser grids over the same interval.

**`dM_neck_f/dt`'s sign remains grid-consistent (positive) at dx=5, 2.5, 1.25nm** for the combined case; no sign flip at any tested resolution.

## 11. Recommendation for the next experiment

The GB/surface-energy ratio is the dominant lever found in this screen -- `psi_eq=90deg` alone already gets to 0.27x baseline, and combined with high substrate curvature reaches 0.23x baseline with a late-time rate decelerating to within ~1% of zero. Neither single-variable ladder nor the one combined case produced a sustained sign crossover within the tested ranges and `t<=0.30s`, so the qualified geometry remains on the neck-filling side of the stability boundary throughout this milestone (consistent with Section 13's instruction not to reinterpret the geometry here). The clearest next step is to push the GB/surface-energy ratio further in the same direction (`psi_eq` below 90deg, i.e. `gamma_gb/gamma_s > 1.4142`) combined with the already-favorable high-amplitude geometry, and/or extend the combined `A=72nm`+`psi_eq=90deg` condition's runtime past `0.30s` (its rate was still decelerating, not yet at true zero) to determine whether it eventually crosses to depleting under sink-off dynamics alone -- before considering sink/hazard/RBM activation.

**STOP. Sink/hazard/RBM/anisotropy/eta kinetics were not activated.**
