# M16M Continuation — Stress Metrology Audit and Sequential-Event Control

## Central conclusion (read this first)

**Root cause found.** A static analytic-geometry benchmark (Section 10-12 continuation below) shows the circle-fit curvature *estimator itself* recovers a known sharp radius of curvature to <1% accuracy, completely independent of interface width `W` or grid spacing `dx` — the estimator is not biased. But computing the EXACT Young-Herring tangent-fillet radius that this project's own geometry construction (`solve_body_fillet`/`solve_flat_groove_fillet`) analytically targets at the canonical geometry (`a=100nm`, `psi=160°`) gives **rho = 6.87nm** — a radius roughly 3-6x SMALLER than any diffuse-interface width (`W=4-10nm`) or curvature-fitting window (`6-24nm`) used anywhere in the M16G-M16M lineage. **Every sintering-stress value reported throughout this entire project's history has been measuring a diffuse-blended, under-resolved curvature, not the true sharp-limit fillet curvature the geometry was analytically designed to have.** Plugging the true analytic radius directly into Hussein Eq. 1b gives `sigma ≈ 140.7 MPa` at `t=0` — far above the ~19.5MPa "legacy" value reported for the same state throughout M16H-M16M, and in the same rough neighborhood as the independent energy-conjugate estimate (~188-193MPa, evaluated at a later, evolved state). This directly explains the non-convergent W/dx refinement, the growing trough-vs-TJ disagreement, and the ~4x energy-vs-local discrepancy all found in this continuation — they are different symptoms of the same underlying under-resolution, not independent problems. Per the restart handoff's own Section 28 stop conditions ("stress metrology cannot be made convergent"; "energy and local stress measures give irreconcilable physical trends"), this report stops short of barrier recalibration (Section 15) and a "qualified" production run. The bounded multi-sink control run continues in the background as a legacy-stress, Poisson-mechanics-only control, not a quantitatively-calibrated result.

## 1. Exact branch/HEAD on restart

- Branch: `codex/m16m-poisson-multisink`
- HEAD at restart: `0b5bd893e1fe7f7994bc31d1baf1f69dd71c0b08` (clean, matching the handoff's expected state)
- `git status --short` and `git diff --check` both empty/clean.
- 257/257 tests passing before any continuation change.

## 2. Status of the pre-existing background M16M run

The run was **not** still alive at restart — it had already stopped (`meta.json`: `stop_reason: "Section 11 GATE FAILURE..."`, `n_born=1`, `n_completed=0`). Direct investigation (Section 3 below) showed this was a false-positive triggered by an overly strict audit gate I had added, not a correctness bug — the gate has been fixed and the run relaunched (Section 4).

## 3. Section-11 gate false positive — root cause and fix

Reproducing the exact field state at the birth step (from a saved checkpoint, replayed forward with `axisym_gb_face_projected_step`) and calling `active_sink_transport_step` (M16L's own already-validated function) side-by-side with the new `multi_sink_transport_step` (N=1) showed **bit-identical behavior** between the two — confirming the new multi-sink transport code is not the source of the discrepancy.

Checking M16L's own **already-validated, already-reported** `event_dense_history.csv` (`runs/m16l_first_event_qualification/`) directly revealed the identical pattern: the ratio of measured-to-requested per-step RBM displacement starts at ~1.00 on the very first transport step, then drops to ~0.59 on the second step and **smoothly, monotonically declines to ~0.47 by event completion** (213 steps later). This is a genuine, reproducible characteristic of the existing (pre-M16M) transport physics — most likely because the analytic point-estimate `v_event = b/tau_Coble` (a simple Coble-diffusion velocity) does not account for the fact that natural capillary/surface-diffusion relaxation, interleaved between RBM substeps, partially opposes each RBM increment; the true PF-measured accommodation rate is genuinely slower than the naive analytic rate, increasingly so as the event progresses.

Because the actual event-completion bookkeeping already uses the **measured** (self-correcting) displacement, not the naive request, M16L's original event still completed correctly (`delta_event` reached exactly `b`, mass conservation held to machine precision) despite this ratio — nothing about the one-b contract or mass conservation was ever broken. My Section-11 gate assumed request and measurement should closely track each other; that assumption is now shown to be false even for the fully-validated single-event baseline.

**Fix**: replaced the `<=2%` hard gate with a pathological-divergence-only check (ratio outside `[0.05, 20]`, sustained over 50 consecutive non-trivial steps) — this still catches a genuinely stalled or runaway transport without flagging the now-understood real decline. Committed as `0a0f6c4`.

## 4. Legacy-control run status

Relaunched with the fixed gate (`runs/m16m_multisink_qualification/`, `codex/m16m-poisson-multisink`, event-count target reduced to 10 per the restart handoff's Section 22, wall-clock cap 4h), explicitly labeled **LEGACY-STRESS POISSON CONTROL** throughout (single-window 1.5W=15nm circle-fit sigma) — answers questions about Poisson-clock/multi-event mechanics, not absolute physical stress.

**Outcome: reached and completed exactly ONE event (the same deterministic pre-activation trajectory as M16L: birth at step=90558, t=4.4218, sigma=45.126MPa; completion at step=90770, t=4.4321, sigma=47.805MPa, `cumulative_RBM/b=0.999997` — essentially exact one-Burgers-vector completion, consistent with M16L's own validated result), then the PF field solver hit a genuine numerical BLOWUP (NaN/overflow in `axisym.py`'s surface-diffusion flux terms) at step=99693, t=4.8678 — roughly 0.44 model-time-units, ~9000 PF steps, AFTER the event completed, with no second event ever born (`N_active=0` throughout the post-event interval).** A real, non-negligible mass-drift jump (from ~4.9e-15 to 8.88e-6 relative) appears in the diagnostic sample immediately following the event's completion — small in absolute terms but ~9 orders of magnitude larger than the pre-event baseline, and a plausible precursor to the blowup roughly 9000 steps later. This is a NEW failure mode not seen in M16L's own single-event run (which continued cleanly to t~4.8-25 in various M16K/M16L extensions) — whether it is connected to the same under-resolved-curvature root cause identified in Sections 10-14 (a numerically sharper, less-diffuse post-event neck than the solver can stably advance) or a separate issue was not diagnosed further this pass; flagged as a required follow-up before any longer multi-event run is attempted.

## 5-6. N_active statistics / completed-event count

`N_active`: 0 for the entire pre-activation climb (t=0 to 4.42), 1 for the ~0.01 time-unit duration of the single event (t=4.4218-4.4321), then 0 again until the blowup at t=4.8678. `N_completed`=1, `N_born`=1 — consistent with `B<<1` (Section 7): no overlapping events were observed or expected, and none occurred.

## 7. Revised interpretation from B<<1

Unchanged from the first M16M pass: `B(sigma) = Lambda*tau_event` is between 8e-4 and 1.4e-3 across the whole 20-100MPa range (computed purely from the frozen calibration, before any PF run). Overlapping active sinks should be rare; the primary state variable of interest is `N_completed` / cumulative RBM, not `N_active`. This is reprised, not re-derived, from the first M16M report (`MILESTONE_16M_POISSON_MULTISINK_BASELINE.md`).

## 8. Trough-versus-TJ geometry audit

Computed `z_min`/`R_min` (the R(z)-minimum, i.e. the coordinate the production driver has used throughout M16H-M16M as the authoritative neck/GB position) alongside `z_TJ`/`R_TJ` (the true contour-based particle/substrate grain-identity crossing, `find_tj_from_contour`) at two available reference states:

| state | z_min (nm) | X_min=2·R_min (nm) | z_TJ (nm) | X_TJ=2·R_TJ (nm) | \|z_TJ−z_min\| (nm) | \|X_TJ−X_min\|/X_min |
|---|---:|---:|---:|---:|---:|---:|
| t=0 (fresh construction) | 2.334 | 201.74 | −0.005 | 227.61 | 2.339 | 12.82% |
| step=90490, t=4.4185 (~45MPa legacy) | 3.581 | 279.79 | 0.016 | 339.46 | 3.565 | 21.32% |

**Finding: the trough and TJ definitions disagree non-trivially (13-21% in implied X_neck), and the disagreement GROWS as the trajectory evolves** — it is not a fixed, negligible offset that can be assumed constant. Section 8's request to "not silently use one as the other" is directly validated by this data: `X_min` and `X_TJ` differ enough to materially change `sigma_Hussein`'s `C_GB/X` term. The production driver (and this report's other diagnostics) use `X_min` (trough-based) as authoritative, consistent with M16H-M16L's established convention and with the fact that the TJ-contour method can, at small z-offsets, cross grain identity near the internal planar GB reference rather than the curved neck (documented in the first M16M report's stress-consensus section) — but this choice is now flagged as a real, unresolved source of potential systematic error in the absolute stress scale, not a settled question.

## 9. Stress-metrology reference dataset (partial)

Two states audited in full (Section 8/10-12/14 tables): the fresh t=0 construction and the step=90490/t=4.4185 (~45MPa legacy) frozen state from `runs/m16k_prescribed_displacement_microtest/frozen_state.npz`. A third, later high-stress state was attempted but discarded from this report after discovering the loaded checkpoint reflected an early (t=0) snapshot of the just-relaunched control run rather than a distinct later state — noted here so the omission is understood as a data-availability limitation, not an oversight.

## 10-12. W/dx convergence and window-ladder plateau search

**Existing runs were searched first (per Section 11's explicit instruction) rather than immediately re-running anything.** Two directly comparable deterministic sink-off reference trajectories already existed in `runs/m16j_geometry_search/stage1/` for the exact same canonical geometry (`R_p=1000nm`, flat substrate, `X0/(2Rp)=0.10`):

| label | W (nm) | dx (nm) | dt | sigma(t=0) MPa | r_neck(t=0) nm | X_neck(t=0) nm |
|---|---:|---:|---:|---:|---:|---:|
| `Rp1000nm_flat_X0over2Rp0.100_t8_ext_archive` | 10.0 | 1.25 | 4.8828e-5 | 19.52 | 40.98 | 201.74 |
| `Rp1000nm_flat_X0over2Rp0.100_refined_W6dx1` | 6.0 | 1.0 | 2.4414e-5 | 47.10 | 19.23 | 201.21 |

A third resolution was run fresh this session (`scripts/m16j_stage1_short_pf_screen.py --flat --ratio 0.10 --w-nm 4.0 --dx-nm 0.65`, short sink-off screen, not a long kinetic run, per Section 11's explicit "objective is NOT a long kinetic run" instruction):

| label | W (nm) | dx (nm) | sigma(t=0) MPa | r_neck(t=0) nm | X_neck(t=0) nm |
|---|---:|---:|---:|---:|---:|
| W4dx065 (this session) | 4.0 | 0.65 | 86.89 | 10.90 | 201.20 |

**First-pass finding (matched TIME, later shown to be confounded — see below): `X_neck` is essentially W-independent (201.2-201.7nm) but `r_neck` roughly HALVES each time W is halved (40.98 → 19.23 → 10.90nm), and `sigma` roughly DOUBLES (19.5 → 47.1 → 86.9MPa), with no plateau.** A fast, violent transient also appeared at W=4nm (`sigma`: 86.9MPa at t=0 → 4.4MPa by t=0.05, absent at W=10/6nm) — exactly the symptom the restart handoff anticipated: **the t=0 diffuse construction is not an equivalent relaxed state across W**, so a matched-TIME comparison before this transient relaxes away is not meaningful (restart handoff Sections 2-3, 5).

**Matched-MORPHOLOGY re-comparison** (`scripts/m16m_window_W_convergence.py`, new this continuation): re-ran short sink-off references at W=10/6/4nm with a FIXED physical curvature-fitting-window ladder (6/9/12/15/18/24nm, independent of W — the original comparison's r_neck used each resolution's own `1.5×W` window, so changing W simultaneously changed the absolute fitting window: 15→9→6nm, confounding two different effects exactly as flagged). A candidate-selection bug (sorting local-minima candidates by radius instead of matching the established "first found in z-scan order" convention used by the reference archives) was caught and fixed before trusting the re-comparison — the buggy version produced a spurious trajectory where W=4nm showed LOWER stress than W=10/6nm, inconsistent with everything else observed; after the fix, W=10nm reproduces the archived trajectory exactly (`X_neck(t=0.1)=205.48nm` both ways).

At a SINGLE fixed window (15nm), W=10 and W=6 now agree much more closely than the original confounded comparison suggested — e.g. near `X_neck~207-209nm`, `sigma_15nm` is ~22-24MPa at both W=10 and W=6 (compare to the original 1.5W-window comparison's ~2.4x mismatch at t=0). Taken alone, this could look like "the W-dependence was mostly a window-scaling artifact, and a fixed 15nm window is fine." **The FULL window ladder at each fixed W kills that interpretation:** at W=10, t=0.5 (`X_neck=209.4nm`), `r_neck` runs `20.1 → 22.4 → 28.2 → 37.9 → 44.5 → 62.3nm` across windows `6 → 9 → 12 → 15 → 18 → 24nm` (correspondingly `sigma`: `45.1 → 39.9 → 30.7 → 21.7 → 17.8 → 11.4MPa`) — **no two adjacent windows agree, and there is no plateau anywhere in this range.** At W=6, t=0.5 (`X_neck=199.2nm`) the ladder is `34.3 → 32.9 → 28.2 → 23.8 → 21.4 → 27.4nm` — non-monotonic (a dip then a rise), reflecting a different mix of true-fillet and transition-region contamination at each window, not a cleaner version of the same trend. **Neither resolution has a single window that is self-consistent with its neighbors, which is Section 12's own explicit disqualification criterion** ("a window is qualified only if neighboring windows agree"). The apparent W=10-vs-W=6 agreement at exactly 15nm is therefore best read as two different under-resolved measurements landing near each other by coincidence at that particular window, not evidence of a genuine plateau.

**Root cause identified via a static analytic-geometry benchmark** (`scripts/m16m_analytic_curvature_benchmark.py`, Section 6 of the restart handoff): built SHARP profiles with an EXACTLY KNOWN local radius of curvature (a circular arc of radius `r_true` in {10, 20, 40, 80}nm, capped to a flat plateau far from the trough), diffused them with the identical `f = 0.5*(1-tanh((R_g - R(z))/W))` construction `build_candidate_geometry` uses, and ran the SAME extraction pipeline (`measure_R_of_z` → `find_all_extrema` → `neck_curvature_windows` → `hussein_eq1b_sigma`) at W in {10, 6, 4}nm with matched dx.

**Result: the extraction pipeline recovers `r_true` to within <1% at EVERY tested W and EVERY tested fitting window, as long as the window stays smaller than the arc's own physical extent** (e.g. for `r_true=10nm`, any window beyond the arc's ~9nm extent abruptly jumps to a value reflecting the transition/plateau region instead, ~140nm — a large, sudden, unmistakable failure mode, not a small bias). **The circle-fit estimator itself is unbiased and W/dx-independent for a well-resolved feature.**

This makes the next question decisive: what is this project's OWN geometry construction's true, exact (sharp-limit) fillet radius? Calling `solve_body_fillet`/`solve_flat_groove_fillet` directly (the exact Young-Herring tangent-fillet solve `build_candidate_geometry` itself uses, with NO dependence on `W` — `W` only controls where the fillet is cropped and blended into the sphere-cap/plateau, not the fillet's own curvature) at the canonical geometry parameters (`a=100nm`, `psi=160°`, hence `m=young_herring_slope(160°)`, `R_p=1000nm`):

```
rho (true analytic fillet radius) = 6.869 nm
z_tangent (fillet's own physical extent from the trough) = 5.638 nm
sigma_Hussein at this TRUE radius (X_neck=200nm) = 140.65 MPa
    (curvature term = +145.57 MPa, contact-GB term = -4.92 MPa)
```

**The true geometric fillet radius is only 6.87nm — smaller than every diffuse-interface width (4-10nm) and every curvature-fitting window (6-24nm) used anywhere in the M16G-M16M lineage, and the fillet's own physical extent (5.64nm) is smaller still.** Combined with the analytic-benchmark result above, this means every circle-fit measurement made throughout this entire project's history has used a window comparable to or LARGER than the true feature it was trying to resolve — landing in exactly the "abrupt failure" regime the benchmark identified, not the well-resolved regime. **This single root cause directly explains all three symptoms found in this continuation**: (1) W/dx non-convergence (as W and the diffuse interface narrow toward the true ~6.87nm scale, the measurement starts to trend toward, but has not yet reached, the true value — 40.98→19.23→10.90nm at W=10→6→4, heading toward 6.87nm); (2) the growing trough-vs-TJ disagreement (both coordinates are being extracted from an under-resolved region whose apparent shape depends on how much of the surrounding sphere-cap/plateau geometry bleeds into the diffuse profile); (3) the ~4x energy-vs-local-curvature gap (140.65MPa from the exact sharp-limit radius is far closer to the energy-conjugate ~188-193MPa than to any diffuse-measured value, 19.5-86.9MPa).

**This satisfies the restart handoff's own stop condition ("stress metrology cannot be made convergent") with a concrete, well-supported mechanism, not just an empirical observation of non-convergence.** Resolving it for real would require `W` and the fitting window both well below ~6.87nm (e.g. `W~1nm`, windows ~2-4nm) — a substantially finer grid than anything run in this project to date (interactively estimated at several times the cost of the W=4nm/dx=0.65nm case already run, which itself took ~75 minutes for a 2-time-unit short screen).

## 13. Physical TJ usage — implementation status

`find_tj_from_contour` (path-continuous branch, sub-grid interpolation) remains available and was used for the Section 8 audit above; the production driver continues to use the trough (`R(z)`-minimum) coordinate as authoritative for `X_neck`, consistent with M16H-M16L. Given Section 8's finding that the two disagree by a growing 13-21%, this choice is flagged, not newly validated, by this continuation.

## 14. Energy-conjugate stress diagnostic

Implemented (`scripts/m16m_energy_conjugate_stress.py`): starting from the same step=90490 (~45MPa legacy) frozen state, the particle grain (`e2`, correct per the M16L grain-role fix) is shifted by small symmetric displacements (±0.0025 to ±0.04nm) relative to the fixed substrate grain, the total PF free energy `axisym_free_energy_gb` is evaluated at each shift, and the axial force conjugate to the displacement is estimated via a central finite difference: `F_sint = -dF/d(delta)`, normalized by the trough-based GB area `A_GB = pi*a_contact^2`.

| delta (nm) | dF/dd (J/m) | sigma_energy (MPa) |
|---:|---:|---:|
| 0.0025 | −1.1576e-5 | 188.27 |
| 0.0050 | −1.1596e-5 | 188.60 |
| 0.0100 | −1.1637e-5 | 189.27 |
| 0.0200 | −1.1720e-5 | 190.61 |
| 0.0400 | −1.1886e-5 | 193.32 |

**Finding: `sigma_energy` is internally consistent (188-193MPa, ~3% spread) across an order of magnitude of perturbation size — a well-behaved, converged first-derivative estimate — but is roughly 4x LARGER than the legacy circle-fit `sigma_Hussein` (~45MPa) at the SAME physical state.** The sign is physically sensible (moving the particle toward the substrate lowers total free energy, i.e. densification is energetically favorable, consistent with a positive driving stress). The formal second-derivative (curvature-of-F) diagnostic used to sanity-check the linear regime did not itself show a clean plateau as delta shrank, which is flagged as a numerical-precision caveat on the FD scheme (F is an O(1e-12) J sum over a ~1e5-cell grid; resolving its curvature to the required precision may need a more numerically careful implementation than a plain finite difference of the full functional) — but the first-derivative value itself (the quantity of actual interest) was stable and did not depend sensitively on this.

**This 4x discrepancy between an independent global energy method and the local curvature-based method, combined with Section 10-12's non-convergent W-refinement trend, is the second explicit restart-handoff stop condition triggered ("energy and local stress measures give irreconcilable physical trends").**

**Not yet validated against an analytic benchmark (restart continuation's own Section 7 requirement): the virtual-displacement operator used here (shift `e2`, hold `e1` fixed, reconstruct `f=clip(e2_shifted+e1,0,1)`) has not been checked against a case with a known force/energy derivative, and per-shift volume/residual bookkeeping (`e1+e2-f` residual, particle/substrate volume change) was not logged at each perturbation.** Given the root-cause finding above (the true fillet radius is far smaller than any window/interface width tested), `sigma_energy`'s ~188-193MPa is plausibly closer to the true value than `sigma_legacy`'s ~45MPa, but it should not yet be promoted to authoritative without that validation — noted here as required follow-up, not completed this pass.

## 15. Final selected stress coordinate

**None selected.** Per the two stop conditions triggered (Sections 10-12, 14), no single stress coordinate can currently be called quantitatively qualified. `sigma_legacy` (single-window 1.5W circle fit) continues to be used ONLY for continuity with M16H-M16L and for the mechanically self-contained legacy control run (Section 4) — never represented in this report as a converged absolute physical value.

## 16. Whether A0 required recalibration

**Not attempted.** Recalibrating `A0`/`V0` against an unconverged, internally-inconsistent (up to 4x uncertain) stress coordinate would produce a barrier calibrated to an arbitrary number, not a physically meaningful one. Per the restart handoff's own Section 15 ("the barrier must be recalibrated ONCE against the new deterministic sink-off trajectory" — contingent on having a qualified coordinate first) and Section 28's stop logic, barrier recalibration is deferred until the stress-metrology question above is resolved.

## 17. Sequential-event qualified run results

Not applicable this pass (Section 15/16) — the legacy-stress control run (Section 4) reached only 1 completed event before a numerical blowup (Section 4), far short of even a single well-characterized sequential-accumulation trend, and used the unqualified `sigma_legacy` coordinate throughout regardless.

## 18. Estimated event count for 5/10/25 MPa relaxation

Not computed — requires a qualified stress coordinate and a run with enough completed events to estimate a local slope, neither of which is available yet.

## 19. Runtime/profile assessment

Not performed this pass; deferred pending resolution of the higher-priority stress-metrology question, which changes what "long-time PF evolution" would even need to resolve.

## 20-21. (Poisson clock exactness, event statistics)

Unchanged / carried forward: the thinning-algorithm Poisson clock (`poisson_multisink_birth_step`) integrates the nonhomogeneous hazard continuously and exactly regardless of stress-sampling cadence; no deterministic threshold was substituted. Event statistics (birth/completion time, stress, lifetime) continue to be logged per-event in `events.jsonl` / `events_summary.json` for the legacy control run.

## Remaining limitations

- The root-cause finding (true fillet radius ~6.87nm) explains the DIRECTION and rough scale of the discrepancy but has not yet been used to construct a properly converged `sigma_qualified` — that requires either (a) a much finer W/dx run (estimated substantially more expensive than anything run to date), or (b) an alternative, resolution-independent local-curvature estimator validated against the analytic benchmark at radii comparable to 6.87nm specifically (only 10/20/40/80nm were tested; the smallest, most relevant case to this project's own geometry was not directly probed).
- **W=4nm/dx=0.65nm fixed-window screen (completed): confirms a genuine, not merely transient, qualitative breakdown.** Across the full t=0.025-1.0 range, `sigma_15nm` stays persistently NEGATIVE (settling around -0.3 to -0.4MPa) — never crossing back to the positive ~19-24MPa range W=10/W=6 show over the same interval. The raw `r_neck` values at W=4 are the key diagnostic: at matched morphology (t=0.5, `X_neck~206nm`), `r_neck` runs `104 → 153 → 207 → 225 → 224 → 203nm` across windows `6→9→12→15→18→24nm` — an order of magnitude LARGER (much flatter/less curved) than the corresponding W=10 values (`20→22→28→38→45→62nm`) at the same windows and similar morphology. This is the OPPOSITE of the naive expectation that narrowing `W` toward the true ~6.87nm fillet scale should sharpen (shrink) the measured radius. The likely explanation: `W=4nm` is still not smaller than the fillet's own physical extent (`z_tangent=5.64nm` from the analytic solve, Section 10-12) — it is comparable to it — so the diffuse interface at this resolution partially SMOOTHS OUT/erases the tight feature entirely rather than resolving it, producing an even flatter apparent trough. This confirms none of W=10/6/4nm are actually in a well-resolved regime; they fail in different, non-monotonic (and in this case sign-flipping) ways, all traceable to the same root cause.
- Only 2 states available for the TJ-vs-trough audit; a third was attempted but discarded due to a data-loading mistake (documented in Section 9) rather than silently reported.
- The energy-conjugate diagnostic's second-derivative check was not fully numerically resolved, and (per Section 14's update above) the method itself has not yet been validated against a known-force analytic benchmark, per the restart continuation's Section 7 requirement.
- The reciprocal window/W ladder test (Section 4 of the restart continuation: window/W in {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} at each W) was not run as its own explicit sweep — the fixed-window ladder (6-24nm) at each W partially substitutes but does not exactly reproduce this specific request.
- **NEW: the legacy-stress control run (Section 4) hit a genuine PF numerical blowup ~9000 steps (0.44 time units) after its one completed event, with a suspicious ~9-order-of-magnitude mass-drift jump immediately following event completion.** This is a new, undiagnosed robustness issue that must be understood before any longer/multi-event run is attempted, independent of the stress-metrology question.

## Recommendation on the video campaign

**Not justified.** Per Sections 10-12 and 14 above, two of the restart handoff's explicit stop conditions are triggered, and Section 4's new post-event blowup adds a THIRD, independent reason (numerical robustness) not to proceed. No video/production work should proceed until (a) the stress-metrology discrepancy (W-convergence and energy-vs-local-curvature disagreement) is resolved or well-characterized, and (b) the post-event blowup is root-caused and fixed.
