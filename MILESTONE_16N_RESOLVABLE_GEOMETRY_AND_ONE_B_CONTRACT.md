# M16N — Resolvable Geometry + Correct One-b Event Contract

**Status: Sections A, C, F, G, H all complete (H confirmed by both an initial pass and a corrected rerun). Section D found a significant, unexpected result requiring attention before proceeding (see below). Per the explicit stop gate, this report stops before Sections I (barrier recalibration) and J (Poisson sequential events) — no barrier recalibration or multi-event run has been started.**

**Headline finding from Section D: for the new, properly-resolved geometry, sink-OFF sintering stress is DECREASING under capillary relaxation over a full time unit, not loading toward 50MPa.** `ratio=0.185` (initial `sigma_sharp=35.6MPa`) declined smoothly and monotonically to `27.6MPa` over the COMPLETE `t=0-1.0` run (a decelerating but still-negative trend at the end, `r_neck` growing 26.2→33.1nm — the neck steadily blunting, not sharpening). This is the physically-expected direction for ordinary surface-diffusion-driven capillary relaxation (which generally reduces curvature/stress over time), and is the OPPOSITE of the trend the old, unresolved `rho=6.87nm` geometry appeared to show (`sigma` climbing from ~19.5MPa toward ~50MPa over `t=0-8`) throughout the entire M16H-M16M lineage. This raises the possibility that the "loading" narrative built into this project's hazard framework was, at least in part, an artifact of measuring an under-resolved feature with a window/tracker setup that happened to drift in a particular direction as the (poorly-measured) geometry evolved — not necessarily genuine physical stress accumulation. This is flagged prominently rather than pursued further this pass (see Section D below and Remaining limitations).

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

`scripts/m16n_sinkoff_screen.py` uses the path-continuous `NeckTracker` (M16K/M16L/M16M's established convention) rather than naive "first found" candidate selection, tracking `r_neck`/`X_neck`/`sigma_Hussein` via a fixed 12nm window (comfortably inside the Section C plateau for all 3 candidates).

**Cost note**: the larger `a` (185-215nm vs the old geometry's 100nm) roughly doubles the domain size (`Nz=597, Nr=738` vs `248x413` for `ratio=0.200`), making each short screen substantially more expensive than the M16M continuation's window-convergence runs (~85-130s per 0.025-time-unit sample vs ~8s previously). A full `t_target=8` run (matching the historical W=10 archive) was not practical within this session.

**Result (`ratio=0.185`, `W=6nm`, `dx=0.85nm`, `t=0` to `1.0` -- COMPLETE run, 40 samples, `runs/m16n_sinkoff_screen/ratio0.185_W6dx0.85.csv`):**

| t | X_neck (nm) | r_neck (nm) | sigma (MPa) | n_candidates |
|---:|---:|---:|---:|---:|
| 0.000 | 372.99 | 26.17 | 35.58 | 1 |
| 0.100 | 374.81 | 27.81 | 33.33 | 4 |
| 0.200 | 375.24 | 28.85 | 32.04 | 4 |
| 0.300 | 375.40 | 29.68 | 31.06 | 6 |
| 0.400 | 375.49 | 30.36 | 30.31 | 6 |
| 0.500 | 375.55 | 30.94 | 29.70 | 5 |
| 0.600 | 375.62 | 31.44 | 29.18 | 7 |
| 0.700 | 375.68 | 31.90 | 28.73 | 8 |
| 0.800 | 375.74 | 32.32 | 28.32 | 8 |
| 0.900 | 375.79 | 32.72 | 27.94 | 5 |
| 1.000 | 375.85 | 33.08 | 27.61 | 4 |

**`sigma` declines smoothly and monotonically over the ENTIRE run, 35.6→27.6MPa (-22.5% over `t=0-1.0`), while `r_neck` grows (26.2→33.1nm, the neck steadily BLUNTING, not sharpening).** The decline rate decelerates throughout (per-0.025-step drops shrink from ~-0.9MPa near `t=0` to ~-0.08MPa by `t=1.0`) but had NOT reached a firm plateau by the end of the run — still declining, just more slowly. `n_candidates` grew to as many as 9 simultaneous local minima during the run (the path-continuous tracker handled this without erratic jumps in the reported trajectory, but this is a real, worsening ambiguity flagged by Section D's own reject criteria). This is the OPPOSITE direction from Section B's target ("sink-OFF capillary evolution should be capable of sharpening the neck toward ~18-20nm / ~45-50MPa") and from the qualitative trend the OLD, unresolved geometry appeared to show throughout M16H-M16M.

**This is now a robust, complete-run finding, not a short transient**: over a full time unit, with a decelerating-but-still-negative slope and no sign of reversal, the most defensible extrapolation is that this geometry approaches some plateau stress somewhat below 27.6MPa (or continues declining slowly) — not that it turns around and climbs toward 45-50MPa. Reaching a firm asymptote, or ruling out an eventual turnaround, would require substantially more wall-clock time than was practical this session (the observed rate implies many further time units at a cost of ~3200s each).

An additional Section D reject criterion is also triggered: the tracker sees 4-6 simultaneous candidate minima from `t=0.05` onward (`n_candidates` column) — the path-continuous selection handles this without erratic jumping (the trajectory above is smooth), but per Section D's own explicit list ("Reject cases showing: ... tracker switching"), this candidate does not cleanly qualify on that criterion either.

**Interpretation, not yet resolved**: this is the physically-expected direction for ordinary capillary/surface-diffusion relaxation (which reduces curvature over time, moving toward a smoother, lower-energy neck shape) — the WELL-RESOLVED measurement is behaving exactly as basic capillary theory predicts. The fact that the OLD, unresolved geometry appeared to show the opposite trend (loading up toward 45-50MPa) throughout the entire M16H-M16M lineage is now a live, unresolved question: was that apparent loading trend a genuine physical effect specific to the very different (rho=6.87nm, deeply under-resolved) geometry, or was it in some part an artifact of measuring an unresolved feature with a tracker/window setup whose apparent value happened to drift upward as that particular geometry evolved? This was not distinguished this pass. Given the wall-clock cost, `ratio=0.200` and `ratio=0.215` were not run to a comparable duration to check whether they show the same declining trend (a very short `ratio=0.200` probe, `t=0-0.1`, showed a consistent decline: 29.5→27.0MPa).

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

`scripts/m16n_post_event_blowup_diagnostic.py` reconstructs the post-event state (via the corrected contract) and runs PF-only continuations (no hazard, no further RBM) at `dt, dt/2, dt/4, dt/8` out to a matched total time (0.5 model-time-units, comfortably past the ~0.446-time-unit interval where the M16M continuation's legacy-control run blew up), tracking `max/min f`, `max|mu_f_gb|`, free energy, mass drift, and the first nonfinite step.

**Result (first pass, `GB_z_hint=0.0` -- see caveat below): ALL FOUR dt values ran completely STABLY through the full matched interval, with no blowup and floating-point-level mass drift (1e-15 to 1e-16, non-growing) throughout:**

| dt | n_steps | first_nonfinite | final mass_drift | final max\|mu\| |
|---|---:|---|---:|---:|
| dt_base (4.8828e-5) | 10240 | none | -7.99e-16 | 2.504e+10 |
| dt_base/2 (2.4414e-5) | 20480 | none | -1.14e-16 | 2.525e+10 |
| dt_base/4 (1.2207e-5) | 40960 | none | -2.85e-15 | 2.535e+10 |
| dt_base/8 (6.1035e-6) | 81920 | none | -4.22e-15 | 2.540e+10 |

**This means the earlier blowup does NOT simply reproduce from "the pre-event `dt` is unstable immediately after the RBM remap"** — even 8x finer than production `dt`, stability holds well past where the real blowup occurred.

**Caveat addressed — corrected rerun confirms the same conclusion.** The first pass used a hardcoded `GB_z_hint=0.0`; a corrected rerun with the real tracker-derived `z_gb=3.581363283165118e-9` produced essentially IDENTICAL results (uncorrected results preserved in `runs/m16n_post_event_blowup_diagnostic_zgb0_uncorrected/`):

| dt | first_nonfinite (corrected) | final mass_drift (corrected) | final max\|mu\| (corrected) |
|---|---|---:|---:|
| dt_base | none | -1.14e-15 | 2.504e+10 |
| dt_base/2 | none | -6.85e-16 | 2.525e+10 |
| dt_base/4 | none | -2.97e-15 | 2.535e+10 |
| dt_base/8 | none | -4.34e-15 | 2.540e+10 |

**All four dt values are STABLE through the full matched interval in BOTH the uncorrected and corrected reconstructions.** This closes the caveat: the `GB_z_hint` choice does not change the conclusion. **The earlier legacy-control run's blowup is NOT explained by "the pre-event dt becomes unstable immediately after the RBM remap"** — a clean, isolated PF-only reconstruction of the post-event state remains numerically stable at the production dt (and 8x finer) for at least 0.5 time units past event completion, well beyond the ~0.446-time-unit interval where the real run failed. The real blowup must therefore originate from something specific to the FULL production driver's operation beyond pure PF stepping from this state — candidates not yet investigated: interaction with the continuously-running Poisson/hazard bookkeeping, the path-continuous tracker's accumulated internal state across many more calls than this microtest exercises, or some other difference between this isolated reconstruction and the real driver's full per-step sequence of operations. Root-causing the actual blowup remains open.

## Remaining limitations / next steps (per the stop gate)

- **Central open question (Section D)**: does the properly-resolved geometry genuinely fail to load toward 45-50MPa under pure sink-off relaxation, or would it eventually turn around and load after a longer blunting transient than the full `t=0-1.0` unit observed (vs the old geometry's full `t=0-8` trajectory)? The decline was still decelerating-but-negative at `t=1.0`, with no turnaround visible. `ratio=0.200`/`0.215` were only probed very briefly (both also declining, consistent with `0.185`) and not run long enough for a full comparison. This is the single most consequential open item — it determines whether this geometry family can support the project's existing ~45-50MPa physical target at all, or whether a different geometry, mechanism, or target stress needs to be reconsidered.
- Section E (energy-vs-local stress comparison for the new candidate, and energy-method analytic validation) was not completed — blocked on first establishing a reference state from Section D.
- **Section H's real blowup cause remains unidentified**: dt-refinement (up to 8x finer) does not reproduce or explain it, ruling out the simplest hypothesis. The actual cause requires comparing the isolated microtest against the full production driver's operation more directly (e.g. instrumenting the real driver to save state incrementally through the blowup window, rather than reconstructing a simplified proxy).
- Sections I (barrier recalibration) and J (Poisson sequential events) were explicitly NOT started, per the stop gate.

## Recommendation

Per the explicit stop gate, this report stops before barrier recalibration or any multi-event run. **Beyond the stop gate itself, the Section D finding (sigma declining, not loading, under sink-off relaxation for the first properly-resolved geometry tested) is significant enough that it should be resolved — either by running `ratio=0.185` (or another candidate) substantially longer to see whether it plateaus, turns around, or continues declining, or by reconsidering the geometry/mechanism search — before any further investment in barrier recalibration against this specific candidate.**
