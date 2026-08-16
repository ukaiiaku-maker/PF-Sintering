# M16N — Resolvable Geometry + Correct One-b Event Contract

**Status: Sections A, C, F, G complete. Section D (short sink-off evolution) and Section H (post-event blowup diagnostic) in progress at the time of writing; this report will be updated with their results. Per the explicit stop gate, this report stops before Sections I (barrier recalibration) and J (Poisson sequential events) — no barrier recalibration or multi-event run has been started.**

## A. Analytic geometry scan (no PF)

`scripts/m16n_analytic_geometry_scan.py` scans `X0/(2Rp)` at fixed `R_p=1000nm`, `psi=160°`, `gamma_s=1 J/m²`, flat substrate, using the exact Young-Herring tangent-fillet solve (`solve_body_fillet`/`solve_flat_groove_fillet`/`young_herring_slope`) directly — no PF field construction, no PF stepping. Full table in `runs/m16n_analytic_geometry_scan/analytic_scan.csv` (49 ratios, 0.020-0.260).

The scan is clean and monotonic: `rho` (true fillet radius) grows from 0.25nm at `ratio=0.02` to 59.3nm at `ratio=0.26`, and `sigma_Hussein_sharp` falls correspondingly from ~4000MPa down to ~15MPa. The current production geometry (`ratio=0.10`) sits at `rho=6.87nm`, confirming the M16M continuation's finding from a completely independent direction (this scan never touches PF fields at all).

**Candidates with `rho` in the target [25,40]nm band** (8 found, `ratio=0.185-0.220`):

| ratio | rho (nm) | z_tangent (nm) | X_neck (nm) | sigma_sharp (MPa) | rho/W (W=4nm) | rho/W (W=6nm) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.185 | 26.57 | 21.38 | 370 | 34.98 | 6.64 | 4.43 |
| 0.190 | 28.24 | 22.69 | 380 | 32.82 | 7.06 | 4.71 |
| 0.195 | 29.97 | 24.05 | 390 | 30.84 | 7.49 | 5.00 |
| 0.200 | 31.78 | 25.45 | 400 | 29.01 | 7.94 | 5.30 |
| 0.205 | 33.65 | 26.90 | 410 | 27.32 | 8.41 | 5.61 |
| 0.210 | 35.59 | 28.40 | 420 | 25.75 | 8.90 | 5.93 |
| 0.215 | 37.61 | 29.95 | 430 | 24.30 | 9.40 | 6.27 |
| 0.220 | 39.69 | 31.55 | 440 | 22.96 | 9.92 | 6.62 |

All 8 comfortably exceed the required `rho/W >= 3` minimum (and the preferred `>=4-5`) at both `W=4nm` and `W=6nm`. Aspect-ratio variation was not explored (the ratio scan alone reached the target band, per Section A's own "only if useful" instruction).

## B. Target resolvable geometry — candidate selection

Selected 3 candidates spanning the qualified band for Section C/D screening: `ratio ∈ {0.185, 0.200, 0.215}` (initial `sigma_sharp` = 35.0, 29.0, 24.3MPa respectively — bracketing the "20-35MPa" initial target, with `ratio=0.185` closest to needing the least additional loading to reach ~50MPa and `ratio=0.215` offering the most headroom).

## C. Static PF representation qualification (no evolution)

`scripts/m16n_static_pf_qualification.py`: built PF fields at `t=0` ONLY (no PF stepping) for all 3 candidates at both `W=6nm` (`dx=0.85nm`) and `W=4nm` (`dx=0.55nm`), measured with the fixed window ladder (6-36nm), and required agreement with the analytic `rho` target within 10%.

**Result: all 6 (candidate × W) combinations qualify, with best-window agreement of 1.2-1.4%** — a dramatic contrast with the old `rho=6.87nm` geometry, where no window at any tested resolution came close. Full window-ladder data for `ratio=0.200` (representative):

| window (nm) | window/rho | r_neck W=6 (nm) | agreement | r_neck W=4 (nm) | agreement |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.19 | 31.33 | 1.41% | 31.37 | 1.29% |
| 9 | 0.28 | 31.34 | 1.38% | 31.31 | 1.46% |
| 12 | 0.38 | 31.31 | 1.46% | 31.27 | 1.58% |
| 15 | 0.47 | 31.28 | 1.57% | 31.23 | 1.73% |
| 18 | 0.57 | 31.21 | 1.80% | 31.17 | 1.90% |
| 24 | 0.76 | 31.18 | 1.88% | 31.11 | 2.11% |
| 30 | 0.94 | 33.84 | 6.49% | 33.53 | 5.52% |
| 36 | 1.13 | 40.47 | 27.36% | 40.10 | 26.20% |

**This is a genuine plateau**: `r_neck` stays essentially flat (31.1-31.4nm, agreement 1.3-2.1%) across windows 6-24nm at BOTH `W=6nm` and `W=4nm` simultaneously, then breaks down cleanly once the window approaches/exceeds the fillet's own physical extent (`window/rho` crossing ~0.9-1.1) — exactly the behavior the M16M analytic benchmark predicted, and exactly what the old `rho=6.87nm` geometry never showed at any window tested. This new geometry family is genuinely resolvable at practical `W`/`dx`.

## D. Short sink-off evolution

**In progress at time of writing.** `scripts/m16n_sinkoff_screen.py` uses the path-continuous `NeckTracker` (M16K/M16L/M16M's established convention) rather than naive "first found" candidate selection, tracking `r_neck`/`X_neck`/`sigma_Hussein` via a fixed 12nm window (comfortably inside the Section C plateau for all 3 candidates).

**Cost note**: the larger `a` (185-215nm vs the old geometry's 100nm) roughly doubles the domain size (`Nz=597, Nr=738` vs `248x413` for `ratio=0.200`), making each short screen substantially more expensive than the M16M continuation's window-convergence runs (~85s per 0.025-time-unit sample vs ~8s previously). A full `t_target=8` run (matching the historical W=10 archive) was not practical within this session; a bounded `t_target=1.0` screen for the cheapest/most-promising candidate (`ratio=0.185`) was run instead.

**Preliminary observation** (from a short `ratio=0.200` probe, `t_target=0.1`): `sigma` drifted slightly DOWN (29.5→27.0MPa) over the first 0.1 time units, and the tracker saw up to 6 simultaneous candidate minima almost immediately (`t=0.025` onward) — consistent with, and not a new problem beyond, M16K's own earlier documented finding of multiple early local minima in this general flat-substrate construction family. This window is too short to characterize the loading trend; the `ratio=0.185` run in progress will be the primary Section D result reported.

## E. Energy stress as independent check (not re-validated this pass)

At the OLD geometry: `sigma_Hussein_sharp ≈ 140.7MPa` (Section 10-12 of the M16M continuation) vs `sigma_energy ≈ 188-193MPa` — **as instructed, this is now correctly characterized as an order-30% difference once the unresolved legacy curvature estimate is discarded, not the earlier factor-of-4 comparison (which compared the energy estimate against the UNRESOLVED `sigma_legacy≈45MPa`, not the resolved sharp-limit value).** The two measures are not characterized as irreconcilable.

`sigma_energy` has NOT yet been validated against an analytic geometry with a known capillary-force/energy derivative (still required before treating it as authoritative for the hazard, per Section E's explicit instruction) — not attempted this pass given time constraints; noted as required follow-up. For the new resolvable candidate, both `sigma_Hussein` and `sigma_energy` will be reported side-by-side, without averaging, once Section D's evolution run identifies a specific reference state to evaluate.

## F. Corrected one-b event progress definition

**Implemented and verified.** Both `pf_sintering/axisym_sink_rbm.py::active_sink_transport_step` (M16L, single-event) and `pf_sintering/m16m_multisink.py::multi_sink_transport_step` (M16M, multi-event) previously accumulated the event's completion quota (`sink.current_disp` / `event.delta`) from the MEASURED particle-relative-substrate COM response, even though the field itself was always advected by the analytically-requested Coble-rate amount (already correctly capped at the remaining budget). This conflated two genuinely different quantities.

**Fixed**: introduced the explicit separation Section F requires:
- **`delta_sink`** (`d_delta_requested` / `requested[event_id]`): the deterministic, analytic Coble-rate quantity the field is actually advected by this step, already capped so `0 <= delta_sink <= b` per event. **This now drives completion** — `sink.current_disp`/`event.delta` accumulate `delta_sink`, not the measured response.
- **`delta_COM`** (`measured_relative_d` / `delta_COM_this_step`): the measured particle-relative-substrate COM response, reported as a diagnostic output only, never fed back into the completion counter.

Both functions' docstrings and diag dicts were updated accordingly (`delta_sink_this_step`, `delta_COM_this_step` are new explicit diag keys; `applied_d_delta` is kept, now aliasing `delta_sink_this_step`, for backward compatibility with existing readers). One existing test (`test_G_total_applied_equals_sum_of_per_event_increments`) asserted the OLD behavior and was corrected to assert the new, intentionally-different contract. **257/257 tests passing** after the fix.

## G. Transport-only microtest

`scripts/m16n_transport_only_microtest.py`: on the frozen M16K state (step=90490, ~45MPa legacy stress), with PF evolution completely DISABLED between transport calls, applied the corrected one-b contract incrementally at the real production `dt=4.8828e-5`:

- **`delta_sink` reached `b` in exactly 110 steps, ratio=1.00000000 (floating-point exact)** — the corrected bookkeeping is exact, not merely approximately capped.
- **No transport after completion**: a subsequent call is a strict no-op (fields bit-identical, `completed=False`, `active=False`).
- **`delta_COM` totalled only 56.7% of `b`** by the time `delta_sink` reached `b` — confirming this is NOT a bug: `delta_COM` is legitimately a different, smaller quantity.
- **Mass drift jumps to ~8.87e-6 (relative) at event completion** — small but ~9 orders of magnitude above the pre-event floating-point baseline (~1e-15). This value reproduces almost exactly in the independent M16M legacy-control run (8.877e-6), confirming it is a genuine, reproducible characteristic of the RBM/excess-mass-redistribution mechanism at this `b` scale, not run-specific noise.
- **Phase 2 (PF-only continuation from the post-event state, 500 steps, NO sink activity at all): the particle-relative-substrate COM drifted by 0.139nm — 55.5% of a full Burgers vector — from ordinary capillary relaxation ALONE.** This is the direct explanation for why `delta_COM` undershoots `delta_sink` during an active event: natural capillary drift is comparable in magnitude to the sink's own contribution, so `delta_COM` was never a faithful proxy for "how much sink-mediated transport occurred." This directly validates Section F's design decision.

## H. Post-event blowup diagnostic

**In progress at time of writing.** `scripts/m16n_post_event_blowup_diagnostic.py` reconstructs the post-event state (via the corrected contract) and runs PF-only continuations (no hazard, no further RBM) at `dt, dt/2, dt/4, dt/8` out to a matched total time (0.5 model-time-units, comfortably past the ~0.446-time-unit interval where the M16M continuation's legacy-control run blew up), tracking `max/min f`, `max|mu_f_gb|`, free energy, mass drift, and the first nonfinite step.

**Preliminary result: at the ORIGINAL production `dt=4.8828e-5`, the PF-only continuation ran STABLY through the full matched interval (10240 steps, no blowup)** — meaning the earlier blowup does NOT simply reproduce from "the pre-event `dt` is unstable immediately after the RBM remap" in this simplified reconstruction. **Important caveat**: this microtest's `build_post_event_state()` uses a hardcoded `GB_z_hint=0.0` for the excess-mass redistribution, rather than the real driver's tracker-derived `z_gb` (~3.58nm at this state) — since the redistribution's Gaussian deposit is centered on this hint, the reconstructed post-event field state may not be bit-identical to what the real legacy-control run produced, so a clean "stable" result here does not yet fully rule out the real run's specific blowup cause. `dt/2, dt/4, dt/8` results, and a corrected re-run with the proper `z_gb` hint, will be added once available.

## Remaining limitations / next steps (per the stop gate)

- Section D's fuller sink-off trend (beyond the short bounded screen) and the resulting geometry/energy comparison (Section E) were not completed this pass due to the substantially higher per-step cost of the larger-domain candidates.
- Section H's `dt/2, dt/4, dt/8` results and the `GB_z_hint` correction are pending.
- Sections I (barrier recalibration) and J (Poisson sequential events) were explicitly NOT started, per the stop gate.

## Recommendation

Per the explicit stop gate ("STOP after: analytic geometry scan, PF representation qualification, one corrected one-b event, post-event stability test, and report before a long multi-event/video campaign"), this report stops here pending completion of Sections D/H's in-progress runs, and does not proceed to barrier recalibration or any multi-event run without further direction.
