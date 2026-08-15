# M16M Continuation — Stress Metrology Audit and Sequential-Event Control

## Central conclusion (read this first)

**The stress-metrology audit requested in this continuation found that the current sintering-stress coordinate is NOT numerically convergent, and an independent global (energy-conjugate) stress estimate disagrees with the local circle-fit (Hussein Eq. 1b) measure by roughly a factor of 4 at the same reference state.** Per the restart handoff's own Section 28 stop conditions ("stress metrology cannot be made convergent"; "energy and local stress measures give irreconcilable physical trends"), this report stops short of barrier recalibration (Section 15) and a "qualified" production run, and instead lays out precisely what was found and what remains to resolve it. The bounded multi-sink control run (Section 16 of the restart handoff) is still permitted to run and is reported below as a legacy-stress control, not a quantitatively-calibrated result.

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

Relaunched with the fixed gate (`runs/m16m_multisink_qualification/`, `codex/m16m-poisson-multisink`, event-count target reduced to 10 per the restart handoff's Section 22, wall-clock cap 4h). Explicitly labeled **LEGACY-STRESS POISSON CONTROL** throughout (single-window 1.5W=15nm circle-fit sigma, per Section 2 of the restart handoff) — it answers questions about Poisson-clock/multi-event mechanics, not absolute physical stress. Status at time of writing: climbing toward the ~45MPa legacy activation point (same deterministic pre-activation trajectory as M16L, since geometry/hazard parameters are unchanged and the RNG seed is identical). Results will be appended to this report once the run reaches its stopping condition.

## 5-6. N_active statistics / completed-event count

Pending run completion (Section 4). Given `B<<1` (Section 8 below reprises the M16M overlap finding), `N_active<=1` is expected almost always.

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

**Finding: `X_neck` (trough-based contact width) is essentially W-independent (201.2-201.7nm across all three resolutions) — but `r_neck` (and therefore `sigma_Hussein`, which is dominated by the `1/r_neck` term) is NOT converged: it roughly HALVES each time W is halved (40.98 → 19.23 → 10.90nm), and correspondingly `sigma` roughly DOUBLES each time (19.5 → 47.1 → 86.9MPa).** There is no sign of a plateau across the three resolutions tested — the trend is consistent with the diffuse interface systematically under-resolving (blurring/smoothing) the true sharp-interface neck curvature, with `r_neck` continuing to shrink (curvature continuing to sharpen) as `W` narrows further. **This directly satisfies the restart handoff's own stop condition: "stress metrology cannot be made convergent."** A genuine plateau, if one exists, was not reached at any resolution tested this session; reaching it would require substantially finer `W`/`dx` than practical to run interactively within this session (each halving of `W` costs roughly `2-4x` the wall-clock of the previous resolution at fixed physical domain size, and the trend shows no sign of slowing between the three points collected).

An additional, independent warning sign surfaced while the W=4nm/dx=0.65nm reference screen continued running: `sigma` at this resolution swung from 86.9MPa (t=0) to 4.4MPa at t=0.05 — a factor-of-20 change in one twentieth of a time unit, far more violent than the smooth, gradual loading seen at W=10nm and W=6nm over the same interval (both climb smoothly from their t=0 values without any comparable transient). This is consistent with either a genuine fast interfacial-equilibration transient specific to the narrower, less-resolved interface, or the neck tracker locking onto a different, spurious local minimum at this resolution — either way, it is further evidence that the finer-W measurements are not yet in a numerically trustworthy regime, reinforcing rather than undermining the non-convergence finding above.

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

## 15. Final selected stress coordinate

**None selected.** Per the two stop conditions triggered (Sections 10-12, 14), no single stress coordinate can currently be called quantitatively qualified. `sigma_legacy` (single-window 1.5W circle fit) continues to be used ONLY for continuity with M16H-M16L and for the mechanically self-contained legacy control run (Section 4) — never represented in this report as a converged absolute physical value.

## 16. Whether A0 required recalibration

**Not attempted.** Recalibrating `A0`/`V0` against an unconverged, internally-inconsistent (up to 4x uncertain) stress coordinate would produce a barrier calibrated to an arbitrary number, not a physically meaningful one. Per the restart handoff's own Section 15 ("the barrier must be recalibrated ONCE against the new deterministic sink-off trajectory" — contingent on having a qualified coordinate first) and Section 28's stop logic, barrier recalibration is deferred until the stress-metrology question above is resolved.

## 17. Sequential-event qualified run results

Not applicable this pass (Section 15/16) — the legacy-stress control run (Section 4) continues as a mechanics/Poisson-clock check, not a quantitatively qualified production run.

## 18. Estimated event count for 5/10/25 MPa relaxation

Not computed — requires a qualified stress coordinate and a run with enough completed events to estimate a local slope, neither of which is available yet.

## 19. Runtime/profile assessment

Not performed this pass; deferred pending resolution of the higher-priority stress-metrology question, which changes what "long-time PF evolution" would even need to resolve.

## 20-21. (Poisson clock exactness, event statistics)

Unchanged / carried forward: the thinning-algorithm Poisson clock (`poisson_multisink_birth_step`) integrates the nonhomogeneous hazard continuously and exactly regardless of stress-sampling cadence; no deterministic threshold was substituted. Event statistics (birth/completion time, stress, lifetime) continue to be logged per-event in `events.jsonl` / `events_summary.json` for the legacy control run.

## Remaining limitations

- Only 3 resolutions tested for W/dx convergence (10, 6, 4nm); no plateau reached. A genuine convergence study would need at least one or two further halvings, which is expensive at fixed domain size.
- Only 2 states available for the TJ-vs-trough audit; a third was attempted but discarded due to a data-loading mistake (documented in Section 9) rather than silently reported.
- The energy-conjugate diagnostic's second-derivative check was not fully numerically resolved; only the first-derivative (the quantity that actually matters for `sigma_energy`) was verified stable.
- The legacy-stress multi-sink control run's full results (N_active/N_completed statistics, cumulative RBM, per-event stress response) were still in progress at the time of writing and are appended below once available.

## Recommendation on the video campaign

**Not justified.** Per Sections 10-12 and 14 above, two of the restart handoff's explicit stop conditions are triggered. No video/production work should proceed until the stress-metrology discrepancy (W-convergence and energy-vs-local-curvature disagreement) is either resolved or at minimum well-characterized enough to assign a defensible uncertainty band to the ~20-190MPa range spanned by the methods tested so far.
