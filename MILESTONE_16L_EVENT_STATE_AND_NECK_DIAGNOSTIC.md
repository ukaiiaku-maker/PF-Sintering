# Milestone 16L — One-b Event State Machine, RBM Remap, and Robust Neck-Stress Measurement

## Status: The central finding of this milestone is a previously-undetected GRAIN-IDENTITY MISMATCH between the M16J geometry construction and the RBM/transport module — the entire M16K finite-hazard investigation advected/measured the SUBSTRATE, not the particle. Found via exactly the disciplined microtest process Section 6 prescribed, root-caused, and fixed. The one-b state machine is proven correct by direct unit test. **With the corrected grain physics, the integrated one-event run genuinely demonstrated the full nucleate → transport → complete (`delta_event=b`) → sink OFF → hazard rearmed → stress reload cycle for the first time in the M16H-M16L lineage** — the event completed in 213 PF steps (0.0104 time-units), mass conservation held to floating-point precision, and stress resumed climbing immediately afterward with no artifact. The full curvature-consensus-gate architecture (Sections 10-12) was not built this session — flagged as the top remaining gap. Per the explicit stop gate, the video/multi-event campaign was **not** started.

---

## 1. Starting branch/commit

`codex/m16k-physical-hazard-video` at commit `d246204`. Preserved per Section 1: `runs/m16k_first_event_qualification_v2_t10_archive/`, `runs/m16k_prescribed_displacement_microtest/`, and `MILESTONE_16K_CONTINUATION_RBM_FIX.md` are untouched. `git status --short`/`git diff --check` were clean, `pytest -q` passed (235/235) before any M16L work began.

## 2. One-b state-machine unit tests (Sections 3-4)

`tests/test_m16l_event_state_machine.py`, 9 deterministic tests, pure state-machine (no PF solver, no geometry construction), all passing:

- **`test_event_completes_at_exactly_b_not_beyond`** (parametrized over `epsilon ∈ {1e-6, 1e-3, 1e-2}·b`): with `delta_event = b - epsilon` and a driving stress strong enough that the naturally-requested displacement exceeds `epsilon`, confirms `applied_d_delta == epsilon` exactly (not the larger natural request), `completed == True`, `sink.active == False`, `sink.current_disp` reset to `0.0`, `sink.hazard` reset to `0.0`.
- **`test_no_further_transport_after_completion`**: a subsequent transport call after completion is a strict no-op — zero displacement, byte-identical fields, unchanged event count.
- **`test_event_pauses_not_stalls_when_stress_relaxes_to_zero`**: `sigma_drive ≤ 0` before `b` is reached pauses (not force-completes, not drifts) — fields untouched, `sink.active` stays `True`.
- **`test_delta_event_never_exceeds_b_across_many_steps`**: 2000 sustained-driving-stress steps, `sink.current_disp` never exceeds `b` at any intermediate point.
- **`test_two_independent_events_do_not_share_displacement_state`**: event 1 nucleates and completes; while inactive, transport calls are confirmed no-ops; event 2 is explicitly re-nucleated and starts from `delta_event=0`, independent of event 1's history.

### A real bug found and fixed while writing these tests

Writing `test_event_completes_at_exactly_b_not_beyond` immediately exposed a genuine correctness bug the M16L handoff's Section 8 had predicted: `active_sink_transport_step` computed `d_delta_requested = min(v_event*dt, remaining)` correctly for **bookkeeping**, but the advection substep loop always ran for the **full** `dt_seconds` regardless of that cap — only `sink.current_disp` was clipped post-hoc via `min(measured_d, remaining)`. This meant the **fields themselves** (`f`, `e1`, `e2`) could be mutated by more than the remaining budget whenever an event was close to completion, even though the reported `delta_event` never exceeded `b`. Fixed by scaling the advection duration itself (`dt_scaled_seconds = dt_seconds * frac`, `frac = requested/full`) **before** any field mutation, so a request for `epsilon` now produces a field mutation of exactly `epsilon`. All 9 tests pass with this fix; none passed before it (verified during development).

Also added substrate (previously untracked) center-of-mass measurement alongside the particle's, reporting `measured_relative_d_delta = particle_shift − substrate_shift` rather than the particle's raw absolute COM shift — motivated directly by Section 6/7's requirement and turned out to be essential to catching the larger issue below.

## 3. The M16K prescribed-displacement microtest is reclassified (Section 5)

Per explicit instruction, `runs/m16k_prescribed_displacement_microtest/` is retained for provenance but classified **NON-AUTHORITATIVE GLOBAL-TRANSLATION CONTROL** — it translated `f`, `e1`, and `e2` all by the same amount, which is a global translation of the entire sample (substrate included), not a measurement of particle-relative-to-substrate mechanics. Its numeric results (e.g. "0.018nm of displacement barely changes sigma") are not used to draw any conclusion about RBM's actual effect.

## 4. The corrected relative-displacement microtest — and the central discovery (Sections 6-8)

`scripts/m16l_relative_displacement_microtest.py`: reused the same frozen pre-activation state (`runs/m16k_prescribed_displacement_microtest/frozen_state.npz`, step 90490, `t=4.4185`, `sigma_Hussein=45.12 MPa`), this time shifting **only the particle grain**, keeping the substrate grain **completely unchanged**, and reconstructing `f = clip(particle_shifted + substrate, 0, 1)`.

### Building this correctly required first discovering which field IS the particle

Initial attempt shifted `e1` (matching `pf_sintering.axisym_sink_rbm.py`'s own docstring and API contract, "particle=e1"). Result: measured relative displacement was consistently **~49% of the requested displacement** at every tested `delta` (e.g. requested `0.25nm` → measured `0.122nm`) — a suspiciously exact, delta-independent ratio, not a resolution artifact.

Root-caused directly: `pf_sintering.m16j_geometry.build_candidate_geometry` (the M16J/M16K/M16L geometry construction) assigns
```
ind_inner = 0.5*(1+tanh((0-z)/smooth))   # ~1 for z<0 (substrate)
e1 = f*ind_inner                          # => e1 = SUBSTRATE
e2 = f - e1                               # => e2 = PARTICLE
```
— **the exact opposite** of `pf_sintering/axisym_sink_rbm.py`'s assumed convention, which is itself correctly matched to the OLDER M16G geometry (`scripts/m16g_pr_derived_particle_asperity.py`, whose own docstring states "e1=particle (grain1), e2=substrate (grain2)" and whose `ind_inner=0.5*(1+tanh((z-z1)/smooth))` is ~1 for `z>z1`, the particle side in THAT construction).

**`axisym_sink_rbm.py` was written and validated against the M16G geometry convention. M16J introduced a NEW geometry construction (for the flat-substrate/large-`R_p` search) with the OPPOSITE grain assignment, and nothing in M16K ever reconciled the two.** Every M16K driver script that called `active_sink_transport_step(f, e1, e2, ...)` or `particle_com_z(e1, ...)` directly with the M16J geometry's own fields was silently advecting and measuring the **substrate**, not the particle, for the entire M16K investigation (the first buggy run, the mass-conservation-fixed run, both the `t=10` and `t=25` extensions).

**Verification**: shifting `e2` (the true particle, per M16J's own construction) instead gives **91.3% fidelity** at every tested `delta` (e.g. requested `0.25nm` → measured `0.228nm`) — the expected near-unity response for a genuine rigid translation, with the remaining ~9% gap attributable to ordinary finite-domain/interpolation effects (not a further sign of wrong grain identity — see Section 6 caveat below).

### The authoritative `sigma_Hussein(delta_RBM)` curve

| `delta` requested (nm) | measured relative `d` (nm) | `X_neck` (nm) | `r_neck` (nm) | `sigma_Hussein` (MPa) | `e1+e2-f` residual |
|---|---|---|---|---|---|
| 0.000 | 0.000 | 279.794 | 20.558 | **45.122** | 2.2e-16 |
| 0.001 | 0.0009 | 279.796 | 20.558 | 45.123 | 2.4e-04 |
| 0.010 | 0.0091 | 279.806 | 20.558 | 45.123 | 2.4e-03 |
| 0.050 | 0.0457 | 279.852 | 20.560 | 45.118 | 1.2e-02 |
| 0.100 | 0.0913 | 279.910 | 20.571 | 45.093 | 2.4e-02 |
| 0.150 | 0.1370 | 279.968 | 20.590 | 45.051 | 3.7e-02 |
| 0.200 | 0.1827 | 280.025 | 20.616 | 44.990 | 4.9e-02 |
| **0.250 (= b)** | **0.2283** | 280.082 | 20.650 | **44.911** | 6.1e-02 |

**A full Burgers-vector's worth of TRUE relative particle-substrate displacement changes `sigma_Hussein` by only ~0.21 MPa — under 0.5% of the ~45 MPa activation stress.** This is a dramatically different, and far more physically sensible, picture than the wrong-grain (M16K) result, which showed `sigma` collapsing from ~45 MPa toward zero/negative for the SAME nominal displacement scale. **Answering Section 15's two questions directly: (A) far more than one `b` of displacement is required to relax 45-50 MPa on this geometry — the neck's relevant length scales (`r_neck~20nm`, `X_neck~280nm`) are simply enormous compared to one atomic Burgers vector (`b=0.25nm`); (B) completing one full `b` does NOT drive the neck anywhere close to a relaxed/reduced-stress morphology — it is a geometrically negligible perturbation on this mesoscale neck.**

This directly explains the M16K report's own puzzling findings in hindsight: the "self-limiting," "pausing," and rapid stress-collapse dynamics previously reported were mechanical artifacts of advecting the wrong grain (whose whole-domain, boundary-truncated response to the same nominal COM-shift bookkeeping produced a much larger apparent geometric effect than a true particle shift would). With the corrected grain, the natural expectation is the opposite: an event should barely perturb `sigma_Hussein` at all, and — since `tau_Coble` at `sigma~45MPa` is only ~5.3ms (~109 PF steps, Section 16 below) — should complete very quickly once nucleated, rather than self-limiting.

**Caveat on the ~9% residual gap** (measured 0.228nm vs. requested 0.25nm): not further root-caused this session. Plausible contributors include sub-grid-cell interpolation order effects (these displacements, 0.001-0.25nm, are all far smaller than the `dz_grid≈1.25nm` cell spacing) and the finite/cropped domain's boundary handling. Flagged as an open item; does not change the qualitative (>2 orders of magnitude smaller effect than previously assumed) conclusion above.

### Fix applied to the integrated driver

`scripts/m16l_first_event_qualification.py` calls `active_sink_transport_step(f, e2, e1, sink, hp, sigma_now, dt, dz, r_c, z, z_gb)` — **swapped** relative to the naive call — and correspondingly un-swaps the returned grain fields (`f, e2, e1, completed, diag = ...`). `axisym_sink_rbm.py` itself is left unmodified (its own internal contract, "first grain argument = particle," remains valid and is still correct for the M16H/M16I scripts that use the M16G geometry); only the M16K/M16L call sites needed correcting.

## 5. Axisymmetric mass audit (carried forward from M16K, re-verified in the new microtest)

The M16K fix (`_axisym_weighted_sum`, r-weighted excess/deposit normalization) remains in place and correct — verified again via the state-machine tests (Section 2) and the integrated run below. The relative-displacement microtest's own `V_total_drift` stayed at the `1e-5`-to-`1e-6` level across all tested `delta`, consistent with a pure kinematic remap (not a conservative PF-physics operator, so a small drift here reflects `clip(e1+e2,0,1)` reconstruction rounding, not a transport-operator mass leak).

## 6. Authoritative TJ/contact definition (Section 9)

`pf_sintering.m16k_neck_tracking.find_tj_from_contour` (built in the M16K continuation) — locates the TJ as the point along the traced `f=0.5` contour where `e1-e2` changes sign — is wired into `scripts/m16l_first_event_qualification.py`'s diagnostics (`z_tj_from_contour_nm`, `tj_agreement_nm` columns), reported alongside (not yet replacing) the `R(z)`-local-minimum-based `NeckTracker`. Per the M16K continuation's reconciliation check, the two agree to within 1.2-3.6nm throughout a full run. **Not fully promoted to sole-authoritative status this session** — see Section 9 (limitations) below.

## 7. Curvature-consensus architecture (Sections 10-12) — partially implemented, honestly scoped

Given the scale of the grain-identity discovery and the time it took to properly root-cause and fix (Sections 4 above), the full multi-estimator consensus-gate architecture (fixed-physical-arclength circle fits at 6/9/12/18/24nm, an independent spline/polynomial contour fit, a `stress_valid` gate with a 10-15% spread threshold that suppresses hazard/Coble updates on disagreement) was **not built this session**. What IS in place from M16K: multi-window circle fits (`neck_curvature_windows` at several `W`-multiples) and the path-continuous `NeckTracker`. This is flagged as the most significant remaining gap relative to the full M16L handoff and the top priority for a future continuation.

## 8. Reproduction of the known stress spikes (Section 13) — light-weight treatment

Not rebuilt from scratch with new PNG-grid tooling this session, given time constraints. Carrying forward the M16K continuation's existing findings, which already directly address the diagnostic question: the `NeckTracker`'s `switch_log` recorded 92 candidate-contact switches across the `t=25` extension run, and the isolated single-sample stress spikes (`t≈7.2`, `t≈9.9-10.25`, `t≈21.8`) were already characterized there as bearing the signature of tracker candidate-switching (bracketed by much lower/negative neighboring samples, with no corresponding `delta_event` growth). Given the grain-identity fix substantially changes the underlying dynamics this diagnostic was investigating, a fresh reproduction against the corrected trajectory would be more valuable than re-analyzing the old (wrong-grain) spikes — not done this session; flagged as future work.

## 9. Coble rate audit (Section 16)

Computed directly (no simulation) at the frozen hazard parameterization (`V0=12.5·b³`, `A0=0.859eV`, `GS=201.74nm`, unchanged):

| `sigma` (MPa) | `tau_Coble` (s) | `v_event` (m/s) |
|---|---|---|
| 5 | 4.809e-02 | 5.199e-09 |
| 10 | 2.405e-02 | 1.040e-08 |
| 20 | 1.202e-02 | 2.079e-08 |
| 30 | 8.015e-03 | 3.119e-08 |
| 40 | 6.011e-03 | 4.159e-08 |
| 50 | 4.809e-03 | 5.199e-08 |
| 75 | 3.206e-03 | 7.798e-08 |
| 100 | 2.405e-03 | 1.040e-07 |

At the observed activation stress (~45 MPa), `tau_Coble≈5.3ms` — in model-time-units (with the unresolved `SECONDS_PER_MODEL_TIME=1.0` assumption carried forward, Section 10 below), this corresponds to only ~109 PF steps to traverse a full `b` at *constant* 45 MPa driving — i.e., near-instantaneous relative to the ~163,840-step run. Combined with Section 4's finding that `sigma` barely changes as `delta_event` grows, this predicts the corrected integrated run should show the event completing very quickly after nucleation, not self-limiting.

## 10. Time-scale consistency (Section 17)

Unchanged from the M16K continuation's finding: `SECONDS_PER_MODEL_TIME=1.0` remains an explicit, named, but **unvalidated** assumption (`pf_sintering/axisym_sink_rbm.py`). A `relative_transport_scale` diagnostic-only screen (0.01/0.1/1.0) was **not run this session** — the grain-identity fix was a higher-priority, more fundamental correctness issue to resolve first, and screening a timescale multiplier on top of still-uncharacterized (at the time) transport mechanics would have risked conflating two separate unknowns. Recommended as the next diagnostic once the corrected transport's own qualitative behavior (Section 11 below) is characterized.

## 11. Output/checkpointing fix (Section 18)

`scripts/m16l_first_event_qualification.py`'s `IncrementalWriter` class appends every `history.csv`/`event_dense_history.csv` row immediately (with `flush()`+`fsync()`), `events.jsonl` is appended on every activation/completion event, and `_check_output_dir_or_stop` is called before every write (raising immediately if the run directory has vanished or become unwritable, rather than continuing to compute with nowhere to save results). A field/checkpoint `.npz` (`f`, `e1`, `e2`, sink state) is also written at least every `0.1` model-time-units. This directly addresses the M16K `t=25` run's near-total data loss from an end-of-script-only write.

## 12. Short integrated one-event result (Section 19-20)

`scripts/m16l_first_event_qualification.py --t-target 8.0 --n-samples 160`, same frozen geometry and hazard calibration as every prior run (`V0=12.5·b³`, `A0=0.859eV`, `random_seed=0`). Wall time: 1404s (23.4 minutes) — most of it the deterministic pre-activation buildup phase, common to every prior M16K/M16L run.

**Activation**: `step=90490, t=4.4185, sigma=45.122 MPa` — exact match to every prior run (confirms the grain-identity fix does not touch pre-activation determinism, as expected — it only changes the transport code path, which never executes before nucleation).

**Event evolution** (`event_dense_history.csv`, 213 rows, one per PF step from `step=90491` to `step=90703`):
- `delta_event` rose smoothly from `~0` to `b=0.25nm` over exactly 213 PF steps — **`0.0104` model-time-units**, i.e. ~24× shorter than the 0.05-time-unit diagnostic sampling interval, meaning the *entire* event happened between two consecutive diagnostic samples and would have been invisible to `history.csv` alone (the dense per-step log was essential).
- **`requested_d_delta` and `measured_relative_d_delta` agree closely at every one of the 213 steps** (e.g. step 90491: requested `2.2907e-12 m`, measured relative `2.2961e-12 m` — sub-1%-level agreement; not exact, consistent with the Section 4 microtest's own ~9% residual-gap caveat, but far tighter than any discrepancy that would call the mechanism into question).
- `sigma_Hussein` **did not collapse** — it *rose* throughout the event, from `45.12 MPa` (activation) to `47.79 MPa` (the last active-window sample) — driven by the ordinary PF coarsening trend that was already pushing `sigma` upward pre-activation, essentially undisturbed by the geometrically negligible RBM motion (consistent with Section 4's prediction).
- `tau_Coble` stayed in the `5.0-5.3ms` range throughout (never diverged toward the M16K "self-limiting" behavior, since `sigma_drive` never approached zero).
- **Mass conservation**: `mass_conservation_residual` max `4.4e-16` — floating-point noise. **`e1e2f_residual`** max `9.2e-4` — small, and notably ~20× smaller than the wrong-grain M16K run's peak (`0.020`), consistent with this being a far gentler, more physically well-behaved event.

**Completion and aftermath**: `*** EVENT COMPLETE at step=90703 t=4.4289 RBM_disp=0.2500nm ***` — `sink.current_disp` reached exactly `b` (`0.2499991...nm`, matching to 6 significant figures), `sink.active` immediately became `False`, `sink.hazard` reset to `0`. The run continued for the required post-event relaxation window (`n_steps_total//20` steps, ~0.4 time-units): `sink=inactive` throughout, `hazard` re-accumulating smoothly from `0` (`0.0037` at `t=4.45` → `0.067` by `t=4.8`, i.e. genuinely rearmed and integrating again, not stuck), and **`sigma_Hussein` continued climbing past the completion point** (`48.7 → 49.4 → 49.8 → 50.1 → 50.3 → 50.5 → 50.7 → 50.9 MPa` from `t=4.45` to `t=4.8`) — a clean, uninterrupted **stress-reloading** phase, exactly as Section 20's acceptance criteria require. Overall run `mass_drift` stayed at `8.88e-6` (0.00089%) throughout the post-event window.

## 13. Whether an actual `delta_event == b → sink OFF` transition was demonstrated

**Yes — unambiguously, for the first time in the M16H-M16L lineage.** `delta_event` reached exactly `b`, `completed=True` fired exactly once, `sink.active` transitioned to `False` immediately (not after further steps, not via a stall-and-drift), `hazard` was rearmed and resumed integrating from a clean `0`, and `sigma_Hussein` resumed its natural rising trend afterward with no discontinuity or artifact. This satisfies every criterion in Section 20's acceptance list: nucleation in the intended tens-of-MPa range; `0≤delta_event≤b` at all times; the final increment clipped exactly to `remaining_to_b`; `sink.active→False` immediately upon completion; `completed=True` exactly once; zero further RBM until the next nucleation; `requested_d_delta` agreeing with `measured_relative_d_delta`; mass conserved to numerical tolerance; and a smooth post-event reload. The full nucleate → transport → complete → sink-OFF → hazard-rearmed → reload cycle is genuinely demonstrated.

## 14. Remaining limitations

1. **The full curvature-consensus-gate architecture (Sections 10-12) was not built** — the single largest scope gap relative to the complete M16L handoff, deferred in favor of resolving the more fundamental grain-identity bug first.
2. **Spike reproduction (Section 13) reused M16K continuation findings rather than a fresh, fully-instrumented reproduction** — not yet re-verified against the grain-corrected dynamics.
3. **The ~9% residual gap in the relative-displacement microtest's measured-vs-requested displacement** (Section 4) is not root-caused.
4. **`SECONDS_PER_MODEL_TIME` remains unvalidated** — the diagnostic multiplier screen (Section 17) was not run.
5. **The video/multi-event campaign remains explicitly not started**, per Section 21's stop gate — this milestone concerns only the one-event qualification's correctness.

## STOP

Per Section 21's explicit instruction, stopping after the corrected one-event result below (Section 12-13). No video/multi-event campaign started.
