# Milestone 13: separating true surface-diffusional coarsening from grain-ownership / GB migration

## 0. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `5c4ec10`, confirmed clean working tree and **176/176** tests passing before any change. Milestone 12/12B's architecture (unified variational surface transport, `use_aniso_surface=False`, no Ostwald removal/redistribution kernel, no prescribed `dV2/dt`) is preserved unmodified in history. Sink, hazard, RBM, and imposed stress/strain were never activated at any point in this milestone. The physical sinusoidal geometry (R2=80nm, wavelength=480nm, amplitude=24nm, overlap=20nm, W=20nm) was never changed.

This milestone added eight commits on top of `5c4ec10`, ending at full-suite **190/190** passing:

| commit | content |
|---|---|
| `ad6b9c2` | Section 9: rename mu-derived curvature proxy (`kappa_A`->`kappa_mu_effective`, `kappa_B`->`kappa_geom`) |
| `da35d8d` | Sections 10-11: tangent-cone constrained eta integrator + KKT reference tests |
| `4e1b97a` | Section 12: soften completeness claim on the eta free-energy derivation |
| `a01442b` | Section 8: neck control-volume boundary-face flux accounting + `wall_mean` coordinate bug fix |
| `c4e9f90` | Section 13 prep: fix eta rescale void-region convention mismatch |
| *(this report + campaign scripts)* | Sections 3-7, 13-15: pure surface-diffusion baseline, eta-mobility ladder, `MILESTONE_13_SURFACE_DIFFUSION_VS_GB_MIGRATION.md` |

## 1. Accepting the Milestone 12B finding

Milestone 12B's control-volume decomposition found `dV2/dt` at a representative state (t=0.015s) was ~99% driven by eta-ownership migration and only ~1% by conserved-f surface transport. This milestone treats that finding as the starting premise and asks the natural follow-up: what does conserved-f surface diffusion do **on its own**, with GB/ownership kinetics switched off entirely?

## 2. Pure surface-diffusion baseline (Sections 3, 5)

`scripts/m13_frozen_eta_campaign.py`: same State-A sinusoidal-substrate contact and unified isotropic transport as Milestone 12B, but the eta update is **never called** -- e1/e2/e3 stay frozen at their initial values for the entire campaign. Critically, `mu_isotropic(f, e1, e2, e3, s, p)` is still evaluated every step with the frozen eta fields, so the eta-f bulk coupling term stays exactly as production has it (Section 3's explicit requirement) -- only the GB/ownership **kinetics** are off, not the thermodynamic coupling.

With eta frozen, `f_weighted_ownership_volumes`'s `dV2/dt` is driven entirely by conserved-f transport by construction (the eta-migration term is exactly zero, confirmed to ~1e-27 relative in `tests/test_tangent_cone_eta.py`'s regression test) -- so in this baseline, `dV2/dt` **is** `dM_particle_f/dt`, Section 5's requested primary metric, with no decomposition needed.

Run at dx=2.5nm (primary) to t=0.1s and t=0.3s (Section 6), cross-checked at dx=5nm to t=0.1s.

## 3. Physical particle external-surface evolution (Section 4)

Tracked directly from `f=0.5` geometry, not V2:

- **`far_cap_x`**: the maximum X of the f=0.5 contour (how far the particle's outermost point extends). Monotonically **recedes** at both grids over the full window tested: dx=2.5nm, 382.874nm -> 382.763nm by t=0.3s (-111pm, never reverses); dx=5nm, 382.873nm -> 382.845nm by t=0.1s (-28pm, consistent rate).
- **`A_particle_geom`**: `sum(f * particle_side_mask)*dx^2`, where `particle_side_mask = (X - x_s(Y)) > 3W` -- a purely geometric particle-size proxy (no eta anywhere in it), used in place of a fully general particle-cap-area-bounded-by-free-surface-and-GB-chord construction (Section 4 explicitly allows "where possible"; the simpler proxy was chosen given this milestone's overall scope). Shows a more complex trajectory: rises slightly to a peak around t~0.01-0.02s, then declines to a minimum around t~0.16-0.18s, then turns and rises again by t=0.3s -- net change over the full window is small and non-monotonic, unlike `far_cap_x`.
- **`kappa_geom`** (particle-far/near, substrate-far/near, both TJs): tracked via `curvature_extraction.window_curvature` with the Section 8 `wall_mean` fix; qualitatively consistent with Milestone 12B's findings (particle-side curvature magnitude larger than substrate-side throughout).
- TJ coordinates, `L_contact_TJ_sub`, `L_GB_geom_sub`: tracked via the existing `compute_subgrid_contact` machinery, unchanged.

**`far_cap_x` is the cleanest, most direct "does the particle's free surface recede" signal**, and it says yes, consistently, at both grids, throughout the full tested window -- the OTHER geometric metric (`A_particle_geom`) and the ownership metric (`V2`, next section) both show more complex non-monotonic behavior, underscoring Section 14's point that these are genuinely different physical quantities that must be reported separately, not conflated.

## 4. Conserved particle-f transport rate (Section 5)

`V2(t)` in the frozen-eta baseline (dx=2.5nm, t=0 to 0.3s): starts at `2.04018e-14`, dips to a minimum `~2.04008e-14` around t=0.02-0.03s, then rises **monotonically** for the rest of the trajectory, reaching `2.04124e-14` by t=0.3s -- net **above** the starting value. dx=5nm shows the same qualitative dip-then-rise shape (minimum `~2.03978e-14` around t=0.02-0.03s, rising to `2.03996e-14` by t=0.1s). **Conserved-f transport alone, with GB kinetics off, does not shrink V2 over these windows** -- after an initial small dip it recovers and grows. This is the sharpest illustration of Milestone 12B's finding: essentially all of the earlier "particle shrinkage" required eta migration; conserved-f transport alone does something qualitatively different (and non-monotonic) to the ownership-weighted volume.

## 5. Neck f-mass evolution (Section 5-6)

`M_neck_f(t)` (direct `sum(f)*dx^2` within `neck_region_mask`) **increases monotonically at every sampled time, at both grids, over the full tested window** (dx=2.5nm: `3.9374e-15` -> `3.9503e-15` by t=0.3s; dx=5nm: `3.8446e-15` -> `3.8514e-15` by t=0.1s). `dM_neck_f/dt` stays strictly positive throughout but decelerates substantially: dx=2.5nm, `~2.71e-16` (early) -> `2.45e-17` by t=0.3s (~11x reduction); dx=5nm, `~1.99e-16` -> `4.04e-17` by t=0.1s (~4.9x reduction). **No sign crossing was found in either run** (`t_neck_flux_sign_change = None` at both grids, both durations). Per Section 6's explicit instruction not to automatically run to global equilibrium, and given the smooth, still-decelerating-but-clearly-still-positive trend at t=0.3s (10x the original Milestone 12B window), this was not extended further.

## 6. Direct particle-side / substrate-side neck flux balance (Sections 7-8)

Milestone 12B's cell-based half-split of the neck control volume returned an exactly-zero substrate-half contribution -- flagged as a degenerate, untrusted result in that milestone's report. Root cause found and fixed this milestone (Section 8, `pf_sintering/flux_closure.py`): `wall_mean` was computed with `model.initialize_fields`' CENTERED-coordinate formula (`(wall_frac-0.5)*Nx*dx`) but used with the UNCENTERED coordinate arrays every diagnostic tool in this line of work actually uses (`(arange(1,Nx+1))*dx`) -- a ~300nm systematic offset that canceled out in Milestone 12B's own branch classifier (a purely relative comparison) but broke any absolute-threshold classifier outright, exactly the neck-half split's failure mode.

`neck_boundary_face_flux_balance` replaces the cell-based split with the direct boundary-face accounting Section 8 explicitly requires: every face where a masked cell borders an unmasked cell is identified, its actual face-centered flux (matching production's own conservative-update discretization exactly) is signed into/out-of the control volume, and classified particle-side/substrate-side by the face-averaged substrate-baseline deviation at that face. Verified to reproduce the volume-integrated `-div(J)` exactly (`closure_residual ~1e-32`).

**Result, both grids, across the full 0-0.3s (dx=2.5nm) / 0-0.1s (dx=5nm) windows**: the substrate-facing boundary of the neck control volume carries essentially **zero net flux** at every sampled time -- `dM_neck_dt_substrate_side` stays in the `1e-22` to `1e-24` range throughout, **five to six orders of magnitude smaller** than the particle-facing boundary's `1e-17` to `1e-16` range (dx=2.5nm: particle-side `1.14e-16` -> `3.51e-17` decelerating; substrate-side `4.4e-22` -> `6.0e-22`, no clear trend, consistent with noise around zero). dx=5nm cross-check agrees: particle-side `1.37e-16` -> `6.2e-17`; substrate-side stays at `|value| < 2e-22` throughout, even flipping sign between samples (further confirming it is noise-level, not a genuine trend).

**This directly tests, and does not confirm, the user's proposed asymmetric-thinning hypothesis** (`J_neck_to_substrate > J_particle_to_neck` producing `dM_neck_f/dt < 0`). What is actually observed is close to the opposite: net inflow into the neck control volume is essentially **unidirectional from the particle side**, with the substrate-facing boundary carrying negligible net flux in either direction. The mechanism was measured, not imposed, and it did not emerge within the tested horizon.

## 7. mu(s), J_s(s), dJ_s/ds (Sections 7, 13-14 of Milestone 12B, reused)

`curvature_extraction.branch_mu_J_profile` traced on both branches at both TJs, periodically through the frozen-eta campaign. Representative snapshot (t=0.015s, dx=2.5nm, top TJ): particle branch `mu(s=0)=-2.266e8`, `mu(s=far)=-2.139e8` (mu decreasing away from the TJ along the particle branch); substrate branch `mu(s=0)=-2.266e8` (shared TJ value), `mu(s=far)=-6.075e7` (mu rising sharply away from the TJ along the substrate branch) -- the same non-monotonic mu(s) shape Milestone 12B found, now confirmed present even with GB kinetics entirely off (i.e. it is a property of the conserved-f transport law itself, not an artifact of eta migration). `J_tangent`: particle branch negative throughout (`-1.18e-9` at s=0 to `-1.03e-10` at s=far, flux directed toward the TJ -- particle donates toward the neck); substrate branch positive throughout (`+1.03e-9` at s=0 to `+2.11e-11` at s=far, flux directed away from the TJ into the substrate). Qualitatively unchanged from Milestone 12B's finding.

## 8. Correction of mu-equivalent vs. geometric curvature terminology (Section 9)

`kappa_A` (implicitly treated as curvature, validated only for a clean single-phase interface) renamed to `kappa_mu_effective`, explicitly documented as a mu-derived proxy not re-validated as a true curvature once eta-f coupling is present. `kappa_B` renamed to `kappa_geom` -- the actual geometric quantity, unaffected by eta-f coupling, to be used wherever physical curvature (not transport potential) is the question. `mu_mean` (the real transport potential) is unchanged and remains the quantity trusted for transport-direction reasoning. All call sites (`WindowCurvature`, `window_curvature`, `branch_window_report`, `delta_kappa_donor_receiver`, `scripts/m12b_grid_convergence.py`'s `curvature_snapshot` output keys, and tests) updated to match.

## 9. Tangent-cone constrained eta formulation (Section 10)

`pf_sintering/constrained_eta.py::tangent_cone_projected_velocity` + `constrained_tangent_cone_eta_update` replace the unconstrained-Euler-step-then-`model.reproject` pattern with a true constrained gradient flow: at every grid point, the unconstrained variational velocity is projected onto the exact tangent cone of the local simplex `{eta_i>=0, sum_i eta_i=f}` via a small active-set QP (box-lower-bound + single-equality structure, solved in at most N=2 or 3 iterations, vectorized over the whole grid). Constraints are built into the kinetic law itself -- velocities are feasible **before** the step is taken, not clipped back afterward, per Section 10's explicit requirement.

Two logically distinct corrections are tracked separately: `f_tracking_correction` (an expected, potentially large rescale reconciling eta's sum with the post-transport-step f -- an operator-splitting artifact, not eta physics) and `safety_correction` (a genuine numerical safety net for finite-dt overshoot).

## 10. Active-set tests (Section 11)

`tests/test_tangent_cone_eta.py` (13 tests): interior point, one-active-boundary (N=2 and N=3), two-active-boundaries (both freeze, and one-recovers), equal thermodynamic forces (zero velocity), strongly unequal forces -- every case cross-checked against a brute-force reference KKT solver (enumeration over all subsets of the active-boundary index set as the candidate "blocked" set, exact for N<=3), agreeing to `1e-9` absolute tolerance in every case. On the real production sharp-interface state, `f_weighted_ownership_volumes` conservation holds to `<1e-10` and ownership stays non-negative to `<1e-12`.

## 11. Proof post-step projection is negligible (Section 11)

Two corrections, tracked separately (Section 9 of Milestone 12B's report showed a ~0.84 projection/variational ratio for the OLD scheme; this compares against that same ratio for the NEW scheme's `safety_correction`, the genuinely comparable quantity):

`active_tol` (how close to zero counts as "at the boundary") had to be a **physically meaningful threshold, not machine epsilon**: at `active_tol=1e-12`, the safety correction was **~71% of the variational step** on the real sharp-interface initial condition's very first production step -- not negligible at all, and for the same underlying reason Milestone 12B's old scheme had a large ratio (the tangent cone is a statement about instantaneous velocity; a finite dt can carry an interior point past a boundary it wasn't yet flagged at). At `active_tol=1e-4` (physically negligible on this model's O(0.1-1) ownership-fraction scale), the safety correction drops to **true roundoff (~3e-11 relative)**, confirmed stable over 1000 real production steps on the sinusoidal-substrate geometry. This tolerance choice, not the projection algorithm's structure, was the actual fix needed.

A second, logically separate correction (`f_tracking_correction`, the rescale reconciling eta's sum with f after the f-substep) is intentionally NOT required to be small -- it is literally tracking conserved-f transport, not a numerical artifact -- and was measured separately throughout to avoid conflating the two (e.g. cumulative `f_tracking_correction ~152` vs. cumulative `variational_change ~4.3` vs. cumulative `safety_correction ~1.4e-10` over 1000 steps of the real campaign, confirming the safety net specifically is negligible while the f-tracking rescale, as expected, is not).

## 12. Eta-mobility mechanism ladder (Section 13)

`scripts/m13_eta_mobility_ladder.py`: same State-A initial condition, dx=2.5nm, t_end=0.03s, M_eta scaled via `dataclasses.replace` at `eta_ladder_scale = 0.0, 0.1, 1.0` (a mechanism-separation sweep, not parameter tuning), using the Section 10 tangent-cone integrator. At every sample, `dV2/dt` is decomposed into conserved-f transport vs. eta-ownership-migration contributions (`flux_closure.particle_volume_rate_decomposition`).

| scale | `dV2/dt` transport (early) | `dV2/dt` eta-migration (early) | cumulative transport | cumulative eta-migration | `V2` end | `M_neck_f` end |
|---|---|---|---|---|---|---|
| 0.0 | -1.40e-16 | ~0 (roundoff) | -1.50e-18 | ~0 | 2.040031e-14 | 3.940074e-15 |
| 0.1 | -1.40e-16 | -5.22e-16 | -1.50e-18 | -1.08e-17 | 2.038947e-14 | 3.940067e-15 |
| 1.0 | -1.40e-16 | -3.06e-15 | -1.47e-18 | -5.22e-17 | 2.034818e-14 | 3.940086e-15 |

Two findings:

1. **Eta migration dominates `dV2/dt` even at 10% of production mobility** (already ~7x the transport contribution cumulatively at scale=0.1, ~35x at scale=1.0 -- consistent with Milestone 12B's ~99% finding at full mobility).
2. **`M_neck_f`'s final value is essentially IDENTICAL across all three mobility scales** (`3.940074e-15`, `3.940067e-15`, `3.940086e-15` -- agreeing to the 6th significant figure). GB migration, across the full range tested, has **negligible feedback effect on the physical (conserved-f) neck-mass accumulation trajectory** -- its overwhelming effect is on the `V2` ownership bookkeeping, not on the actual mass redistribution. This is new evidence beyond Milestone 12B: not only does eta migration dominate the `V2` signal, restoring it (at any tested strength) does not appear to change the underlying physical coarsening trajectory this model produces.

## 13. Grain-size-change decomposition (Sections 5, 12, 15 of Milestone 12B, reused/extended)

Already covered in Section 12 above (the ladder's `particle_volume_rate_decomposition` output) and Section 4/5 (frozen-eta baseline's clean separation, since the eta-migration term is exactly zero there by construction). No eta-ownership change is reported as "mass transport" anywhere in this milestone's diagnostics; the two are tracked as separate dictionary entries throughout.

## 14. Ownership volume vs. physical morphology (Section 14)

Reported separately throughout, per Section 14's explicit instruction, rather than using `V2` as the sole particle-size metric:

- **`V2`** (eta-ownership-weighted): dips then rises under pure surface diffusion (Section 4); collapses rapidly toward the "V2 decreasing" story only once GB migration is switched on (Section 12).
- **`far_cap_x`** (purely geometric, no eta): recedes monotonically under pure surface diffusion at both grids, the whole way through the tested windows -- the cleanest evidence that the external free surface genuinely does retreat under conserved-f transport alone, independent of any grain-ownership bookkeeping.
- **`A_particle_geom`** (purely geometric, no eta): non-monotonic (rise, fall, rise) -- a genuinely different trajectory shape from both `V2` and `far_cap_x`, underscoring that even among purely geometric measures, the specific definition matters and a single number should not be over-interpreted as "the" particle size.

A grain can become smaller by GB migration while the external particle surface does not recede (or recedes at a completely different rate) -- confirmed directly here, not just asserted.

## 15. F1/F2/F3/F4 classification

**Primarily F2, with an important extension.** With eta frozen: the purely geometric `far_cap_x` metric shows the particle's outer free surface persistently receding (Section 3), while `dM_neck_f/dt` remains strictly positive throughout the full tested window at both grids (0-0.3s at dx=2.5nm, 0-0.1s at dx=5nm, Section 5) -- i.e. surface diffusion redistributes material away from the particle's far cap, but the neck continues to accumulate mass rather than thin, matching F2's description ("surface diffusion transfers material out of the particle but still fills the neck; the current geometry is not yet in the neck-thinning regime").

This is NOT a clean match to F2's literal premise, however: F2 as stated assumes "particle loses f mass," which is true of `far_cap_x` but explicitly **not** true of `V2` (which dips then recovers to end above its starting value, Section 4) or of `A_particle_geom` (non-monotonic, Section 3). The single cleanest statement of what actually happens is: **conserved-f surface diffusion alone causes the particle's outermost point to retreat and the neck to keep gaining mass (decelerating, not reversing, over a 10x-extended time horizon); it does not, by itself, produce a monotonic "particle shrinkage" in any of the tested volume-based metrics.**

The eta-mobility ladder (Section 12) adds a finding not anticipated by the F1-F4 list as literally written: **GB migration, at every mobility tested, leaves the physical (`M_neck_f`) coarsening trajectory essentially unchanged** while completely dominating the `V2` bookkeeping signal. This is closer to the opposite of F4 ("GB migration and surface diffusion are strongly coupled") for the specific quantity `M_neck_f` -- they appear only weakly coupled, at least for this geometry and mobility range. Restoring physical GB mobility (per F1's suggested next step) is therefore not expected, on this evidence, to change the neck-filling outcome -- only whether it is described in `V2` terms as "shrinkage."

## Section 16-17: no geometry change, no sink/hazard/RBM

R2, wavelength, amplitude, overlap, W were held fixed at their Milestone 12/12B values throughout every run in this milestone. Sink, hazard, RBM, and densification were never activated.

---

### Appendix: raw run artifacts

- `runs/m13_frozen_eta_dx2p5.json`, `runs/m13_frozen_eta_dx2p5_long.json` -- pure surface-diffusion baseline, dx=2.5nm, t=0.1s and t=0.3s.
- `runs/m13_frozen_eta_dx5.json` -- cross-check, dx=5nm, t=0.1s.
- `runs/m13_eta_mobility_ladder_dx2p5.json` -- eta-mobility ladder, scale=0/0.1/1, dx=2.5nm, t=0.03s.

### Stop conditions honored

Per Sections 16-17, this milestone did not activate sink, hazard, RBM, imposed stress, or imposed strain, and did not change any physical geometry parameter. The tangent-cone eta integrator and the boundary-face neck-flux accounting are new diagnostic-only code (`pf_sintering/constrained_eta.py`, `pf_sintering/flux_closure.py`), not wired into any production default path.
