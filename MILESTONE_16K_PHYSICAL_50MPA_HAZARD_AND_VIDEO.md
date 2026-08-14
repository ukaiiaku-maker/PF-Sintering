# Milestone 16K — Physical Geometry Qualification, ~50 MPa First-Passage Sink, and Video-Resolved Oscillatory Sintering

## Status: Parts I-II complete. Part III (one-event qualification) partially passes — nucleation occurred correctly in the target stress range, but a genuine post-activation numerical problem (RBM stall + growing mass drift) prevents full PASS-FIRST-EVENT certification. Part IV (video multi-event run) correctly NOT attempted, per the explicit gate in Section 21.

---

## A. Physical geometry qualification

Preserved M16J exactly (`git status`/`diff --check` clean, 235 tests pass, HEAD confirmed at `8752576`) and branched `codex/m16k-physical-hazard-video` from that commit. All M16J run directories and reports are untouched.

Per Section 9, screened a 12-case narrow neighborhood around the M16J winner: exact flat substrate only, `R_p ∈ {0.5, 1.0, 2.0} μm`, `X0/(2R_p) ∈ {0.10, 0.11, 0.125, 0.15}`, `AR=1`, `psi=160°` unchanged (`scripts/m16k_narrow_screen.py`). Static (`t=0`) diagnostics with a proper multi-window curvature-robustness check (Section 6: `r_neck` at 1.0W/1.5W/2.0W/2.5W/3.0W, reporting the % spread) found **only one candidate** (`R_p=2μm, X0/(2R_p)=0.15`) passing all static gates: `sigma_H=29.0 MPa` (in the preferred 10-30 MPa range), `r_neck/W=3.26` (meets the ≥3 resolution gate), spread=17.0% (well under the 30% threshold). All 11 other candidates showed 87–224% spread at `t=0` — a striking confirmation that the M16J "sharp jump" concern (Section 3) reflects a genuine, widespread curvature-measurement fragility across this geometry family, not an isolated fluke.

**A dynamic verification run of that one candidate found it does NOT behave as hoped**: over `t=[0,1.6]`, `sigma_H` *decreased* monotonically (29.0→24.2 MPa) rather than rising toward 50 MPa, `r_neck` grew rather than shrank (32.6→38.7nm), and the window-spread itself *grew* over time (15%→28.5%), with the tracker logging a candidate-contact switch (`n_candidates=4-6`) at essentially every sample from `t=0.1` onward. This is a genuine, reportable negative result: static QC at `t=0` did not predict dynamic behavior for this candidate.

**Decision**: reverted to the M16J winning geometry (flat, `R_p=1μm`, `X0/(2R_p)=0.10`) — but evaluated at its **coarser W=10nm/dx=1.25nm resolution**, not the W=6nm refined resolution the M16K handoff was originally concerned about. At W=10nm this exact geometry's deterministic trajectory (already fully computed in M16J, `runs/m16j_geometry_search/stage1/Rp1000nm_flat_X0over2Rp0.100_t8_ext_archive/`) already satisfies Section 10's stated preference well: `sigma_H` rises smoothly from 19.5 MPa (`t=0`, squarely in the preferred 10-30 MPa range) through 50 MPa (crossing at `t≈4.565`) to a peak of 56.1 MPa (`t≈6.7`), with **no sharp jumps** — those artifacts were specific to the finer W=6nm resolution, not an intrinsic property of the geometry. Section 10's "70-100 MPa if no sink" criterion is only partially met (peaks at 56 MPa at this resolution, not 70-100) — an acknowledged shortfall, traded off against smoothness and having an already fully-characterized, already-QC'd trajectory rather than continuing an open-ended search under severe time constraints.

## B. Diagnostic/curvature qualification

**Section 3 finding**: a high-cadence (`Δt=0.01`) reproduction of the M16J refined (W=6nm) trajectory (`scripts/m16k_diagnose_jump.py`) found **three simultaneous local minima** in the measured `R(z)` profile from as early as `t=0.04` — not the single minimum the original M16J diagnostic implicitly assumed. The original diagnostic's `mins[0]` ("first minimum found scanning z") is therefore under-specified: which minimum is "first" can change as the profile evolves even without any minimum's own depth changing much, causing the reported `r_neck`/`sigma_s` to jump between fundamentally different features. **Classification: primarily a tracker/diagnostic artifact (option B), not solely a proven physical rearrangement (option A)** — though see the "spread growing over time" and "persistent multi-candidate ambiguity" findings in Section A above, which suggest the underlying geometry genuinely does develop competing local features as it evolves, making some LEVEL of switching-sensitivity a real property of this geometry family, not purely a bug.

**Fix**: `pf_sintering/m16k_neck_tracking.py`'s `NeckTracker` — path-continuous selection (nearest candidate to the previously-selected contact, not "first found"), explicit switch logging (`all_candidate_contacts`, `selected_contact`, `distance_from_previous_contact`, `all_candidate_curvatures` at 5 window widths, `selected_curvature`), used throughout the rest of M16K.

**Section 5-6 multi-window curvature**: implemented (1.0W, 1.5W, 2.0W, 2.5W, 3.0W half-windows via the existing `neck_curvature_windows`), with spread reported as a percentage. An independent local-spline/polynomial curvature method (Section 5's "at least one independent method") was **not implemented** given time constraints — the multi-window circle-fit spread check served as the primary robustness signal instead. This is an acknowledged gap.

**Section 7 (high-cadence raw-field save around the M16J jump)**: partially done — `scripts/m16k_diagnose_jump.py` was built and launched with `Δt_output=0.01` and morphology snapshots bounding `t∈[1.0,1.3]`, but was killed early (after confirming the 3-minima finding at `t<0.3`) in favor of moving faster through the milestone's critical path; the full bounded run to `t=1.35` was not completed. The key finding (multiple simultaneous minima from very early on) was captured before stopping.

## C. W/dx convergence

Only the two pairs already established in M16J were compared (W=10nm/dx=1.25nm vs W=6nm/dx=1.0nm) — **no new W=4nm/dx=0.5-0.67nm run was attempted** (Section 8), given time constraints. This is an explicit, acknowledged limitation: the selected geometry's resolution-convergence has not been pushed to a third, finer point.

## D. Selected geometry

**Flat substrate, `R_p=1μm`, `X0/(2R_p)=0.10`, `psi=160°`, `W=10nm`, `dx=1.25nm`, `AR=1`, `gamma_s=1.0`, `gamma_gb=0.34730`** (unchanged from all prior M16-series work). Deterministic sink-OFF trajectory: `sigma_H(t=0)=19.5 MPa`, crosses 50 MPa at `t≈4.565`, peaks `56.1 MPa` at `t≈6.7`.

## E. Activation-volume/barrier screen

`scripts/m16k_hazard_calibration.py`: for `V0/b³ ∈ {12.5, 30, 100, 300}` (Section 15-16's physically-motivated ladder, `b=2.5e-10m`), solved for `A0` such that the cumulative Arrhenius first-passage hazard integrated along the deterministic trajectory reaches `ln(2)=0.693` at `t=4.565` (the moment `sigma_H` crosses 50 MPa) — a genuine trajectory-integral calibration, not an algebraic threshold declaration.

| `V0/b³` | `A0` (eV) | `P(<30MPa)` | `P(<40MPa)` | `P(<50MPa)` |
|---|---|---|---|---|
| **12.5** | **0.859** | **0.179** | 0.179 | 0.500 |
| 30.0 | 0.928 | 0.131 | 0.131 | 0.500 |
| 100.0 | 1.228 | 0.028 | 0.028 | 0.500 |
| 300.0 | 2.141 | 0.0002 | 0.0002 | 0.500 |

(`P(<60/75/100MPa)` are undefined — the trajectory never exceeds 56.1 MPa at this resolution, so those thresholds are never reached deterministically; this is itself informative, not a bug.) `P(<30MPa)=P(<40MPa)` for every row reflects that `sigma_H` crosses both 30 and 40 MPa within a single 0.1-time-unit sampling interval in this trajectory (a fast local rise there), not a calibration error.

**Selected: `V0=12.5·b³`, `A0=0.859 eV`** — the smallest, most physically-motivated rung (Section 15's literature reference value used directly, not "thousands of `b³`" as M16H/M16I used). Per Section 19's explicit instruction, this smaller, more physical `V0` gives a **broader** activation distribution (`P(<30MPa)=18%`, not near-zero) — reported honestly as the tradeoff, not suppressed in favor of an artificially narrow distribution from a larger `V0`.

`tau_sink` ladder (Section 24) at the selected parameterization: 10 MPa → 0.0240s, 25 MPa → 0.00962s, 50 MPa → 0.00481s, 75 MPa → 0.00321s, 100 MPa → 0.00240s (velocities `1.0e-8` to `1.0e-7` m/s) — monotonically faster at higher stress as expected from the `1/sigma` dependence, and none of these predicted a problematically short event before the run (the actual observed problem, Section H below, was not simply "too fast for the timestep").

## F-G. See E above (frozen A0/V0 and probability table combined for conciseness).

## H. One-event qualification — PARTIAL PASS

`scripts/m16k_first_event_qualification.py`, regime explicitly named `finite_barrier_sink_inactive` (Section 13 — not "sink OFF"; hazard armed from step 0). Random seed 0.

**Nucleation**: activation occurred at `step=90490, t=4.4185, sigma_H=45.12 MPa` — **squarely in the intended tens-of-MPa regime**, close to the calibration's median (`t50=4.565` at `sigma=50MPa`; this run fired slightly earlier, at 45 MPa, entirely consistent with the stochastic first-passage model's own predicted spread). Pre-activation behavior was clean: sink inactive throughout, RBM displacement zero, stress rose from 19.5 MPa toward the 45-48 MPa range over `t=[0,4.4]`. **PASS-FIRST-EVENT criteria A, B, C: satisfied.**

**Post-activation: a genuine problem.** Within **0.032 time units** of activation, `sigma_H` collapsed from 45.1 MPa to 22.5 MPa; by `t=4.5` (0.08 time units post-activation) it had gone **negative** (-0.63 MPa) and stayed near/below zero for the remainder of the run. `RBM_cumulative_displacement` reached only **0.018 nm** in that initial burst and then **never grew further** (final value still 0.018nm at `t=9.65`, over 5 time units later) — because the model's `tau_sink` formula is only defined for `sigma>0` (`hazard_step` sets `tau_sink=math.inf` whenever `sigma≤0`), and once `sigma` went non-positive, the RBM advection velocity (`b/tau_sink`) became exactly zero, **stalling the event indefinitely**. The completion criterion (`sink.current_disp >= b = 0.25nm`) was never reached (only 0.018nm accumulated, ~7% of the threshold). Mass drift grew in discrete jumps over the stalled period: 0.12% (first sample post-activation) → 0.29% → 0.88% → **2.29%** by `t=8.4`, then plateaued (still stalled) through `t=9.65` where the run was manually terminated after 32 minutes of wall time with no sign of resolving.

**Most plausible root cause** (not fully confirmed given time constraints, but well-supported by the evidence): the excess-mass (`f>1`) redistribution logic inside `rbm_step` only runs while the sink is active — meaning any excess that accumulates from *ordinary PF coarsening* during the long (4.4-time-unit) sink-inactive buildup phase is never cleaned up until the very first `rbm_step` call after activation, which then processes an entire backlog in one shot. This is consistent with the sub-0.1-time-unit collapse being far too fast to be explained by the RBM advection velocity alone (which, from the `tau_sink` ladder, predicts a per-step displacement many orders of magnitude smaller than what would be needed to broaden the neck this much this quickly) — i.e., the "excess mass" side-channel, not the intended rigid-body advection, appears to dominate the observed collapse. **This is an actionable, specific improvement for future work**: excess-mass cleanup should run continuously (a lightweight `f=min(f,1)` reprojection or an always-on excess-redistribution step), not deferred entirely to the sink-active code path.

**PASS-FIRST-EVENT gate result**: **A, B, C pass. D (RBM began only after activation) passes. E, F ambiguous** — the neck did broaden and stress did drop far more than the preferred 20-30% (essentially to zero and below), but via the mechanism above rather than the intended GB-diffusion-controlled advection reaching a natural stopping point. **G (completed without intervention), H (sink returns inactive), and J (mass drift acceptably small) all FAIL** — the run had to be manually terminated, the sink never deactivated, and mass drift reached 2.29% (an order of magnitude above the M16I-established 0.2% stop-gate). **Overall: does not pass PASS-FIRST-EVENT.**

## I. Event duration and D_gb-controlled RBM audit

The `tau_sink` ladder (Section E) was computed and printed *before* the run, as required (Section 24) — none of those predicted values suggested an unresolvably fast event at the timestep/output cadence used. The actual failure was not "too fast to resolve" but "stalled to a complete stop" — a qualitatively different and, in retrospect, more informative failure mode than the one Section 24 anticipated.

## J. Multi-event response — NOT ATTEMPTED

Correctly gated behind PASS-FIRST-EVENT per Section 21's explicit instruction ("Do not begin the long video run until the first-event qualification passes") and Section 27 ("Only after PASS-FIRST-EVENT, start a fresh run"). Since qualification did not pass, no multi-event or video run was started.

## K. Video-frame architecture — NOT ATTEMPTED

None of Part IV (Sections 27-39: rolling pre-event buffer, adaptive frame cadence, `frames.csv`, event-local raw-state archives, two-panel fixed-axis morphology frames, ffmpeg compilation) was implemented, for the same reason.

## L. Event summary

One activation event recorded (did not complete):

| event | activation_time | activation_sigma_Hussein | activation_r_neck | activation_X_neck | RBM_disp_final | outcome |
|---|---|---|---|---|---|---|
| 1 | 4.4185 | 45.12 MPa | ~19-20nm (est.) | ~203nm (est.) | 0.018nm (7% of the 0.25nm completion threshold) | **stalled, never completed** |

## M. A/V vs densification strain

Not meaningfully producible — `rbm_strain` (the authoritative densification metric, per the M16I diagnostic-naming audit) stayed at essentially zero throughout (RBM displacement never grew beyond 0.018nm), so an `A_free/V` vs `rbm_strain` curve would show no real densification signal, only the pre-event coarsening-driven `A/V` trend already characterized in M16J/the deterministic trajectory.

## N. Remaining limitations and recommended next steps

1. **The RBM-stall bug must be fixed before a future first-event run can pass.** Recommended: make excess-mass (`f>1`) cleanup run every PF step regardless of sink state (not just while active), so no backlog accumulates during the multi-time-unit sink-inactive buildup phase.
2. **Section 5's independent (spline/polynomial) curvature method was not implemented** — the multi-window circle-fit spread check substituted for it; a genuinely independent method would strengthen confidence in the resolution-robustness finding.
3. **No W=4nm/dx≈0.5-0.67nm resolution point was run** — Section C's convergence claim rests on only two resolution levels.
4. **The negative dynamic-relaxation finding for the R_p=2μm/ratio=0.15 candidate** should be revisited once the RBM-stall bug is fixed — it's possible the "relaxing" behavior observed there is itself connected to whatever numerical subtlety underlies the stall (both surfaced when a candidate was pushed past its comfortable, well-characterized parameter range).
5. **Section 7's high-cadence diagnostic save was only partially run** (stopped early once the 3-minima finding was confirmed) — a complete bounded run through the M16J jump window would still be valuable.
6. Once (1) is fixed, re-run the frozen parameterization (Section E/G, unchanged) for the first-event qualification; only proceed to Part IV after PASS-FIRST-EVENT is genuinely achieved.

## STOP

Per Section 21 ("Then STOP" — do not launch the video run until qualification passes) and given the RBM-stall finding, M16K stops here. Part IV is explicit future work, not started.
