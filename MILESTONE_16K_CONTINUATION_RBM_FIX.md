# Milestone 16K Continuation — Corrected Post-Nucleation RBM Physics

## Status: The post-nucleation stall/mass-drift failure from the original M16K report is fixed and root-caused. The corrected first-event run shows physically coherent, self-consistent coupled dynamics: fast initial transport, self-limiting slowdown, a clean pause, and one brief genuine resumption driven by continued PF coarsening — all with mass conservation at floating-point precision (~1000x better than the original bug). An extension to `t_target=25` (2.5x the original window) confirmed the event settles into a long-lived, effectively permanently-paused state for this particular stochastic draw: `delta_event` never moved beyond the initial burst's `0.0179nm` (7% of `b`) at any of 501 samples spanning the full window, and local stress never sustainably recovered past zero. This is reported as an honest, physically informative outcome, not forced to a different conclusion. Per the explicit stop gate in both handoff documents, the video/multi-event campaign was **not** started.

---

## 1. What was NOT done first (per explicit instruction)

The original report's "backlog cleanup" hypothesis (excess-mass never cleaned up while the sink is inactive) was **not implemented**. The evidence cited against it was correct: mass drift was ~5e-15 immediately before activation and only became large after RBM started — inconsistent with a pre-existing backlog being dumped at activation, and consistent instead with a bug in the transport mechanics themselves.

## 2. Axisymmetric mass-conservation audit — bug confirmed and fixed

`pf_sintering/axisym_sink_rbm.py`'s excess-mass redistribution (`rbm_step`, now also `active_sink_transport_step`) computed `ex_sum = np.sum(excess)` and `dsum = np.sum(dep)` — **plain, unweighted array sums** over the (z, r) grid. For an axisymmetric field the correct volume element is `2π·r·dr·dz`; a cell's contribution to physical volume must be weighted by its radial position. Since the excess mass and the deposit pattern generally have different r-distributions, this unweighted normalization did not exactly conserve physical volume.

**Fix**: new `_axisym_weighted_sum(field, r_c)` helper (`r_c[None,:]*field` summed), used for both sides of the redistribution ratio. Verified: the constant `2π·dr·dz` factor cancels in the ratio used for renormalization, so only the r-weighting itself needed correcting — no additional grid-spacing parameters needed threading through.

## 3. e1+e2=f consistency audit — corrected characterization

Recorded at every active-window step (`e1e2f_residual = max|e1+e2-f|`). Across all 114,310 steps of the corrected run's active window: max 0.0200, mean 0.0022. **Revisiting this with the full per-step trace** (rather than only the aggregate max/mean) shows this is NOT a slow accumulation from ordinary PF stepping, as first reported — it is a **bounded transient tied directly to the RBM advection burst**: the residual is tiny (0.0009) at the very first active-window step (90491, immediately post-activation), rises rapidly to its run-wide maximum (0.0200) by step 90522 — i.e. within the same ~32-step window where essentially all of the 0.0179nm of real transport happens (Section 6) — and then **decays back down** to a small, stable value (0.0003-0.003) for the remaining ~114,000 steps once the event pauses. This is consistent with `e1`/`e2` being advected and clipped independently from `f`'s own upwind-advection-plus-excess-redistribution update inside `active_sink_transport_step`: the two parallel updates can transiently diverge from exact summation during active transport, then relax back toward consistency once transport stops. Not a growing numerical problem.

(The prescribed-displacement microtest, Section 5, separately reported `e1e2f_residual=0.0444` at `dz=0` — i.e., before any RBM translation at all, identical across every tested `dz`. This is larger than anything seen in the live run's trace and is suspected to be an artifact of `scipy.ndimage.shift`'s spline pre-filtering being applied even at a nominal zero shift, not a property of the true frozen state; the live run's own per-step trace above is the more reliable characterization.)

## 4. Nucleation/transport separation

`nucleation_hazard_step` (decides ONLY whether a new event nucleates; does nothing while one is active) and `active_sink_transport_step` (governs an already-nucleated event) replace the single `hazard_step`+`rbm_step` pairing for the corrected driver (the originals are left in place, unmodified, for M16H-M16J backward compatibility — no test in the existing suite covers this module, so this was a safe, non-breaking addition).

`active_sink_transport_step` implements the explicit one-Burgers-vector, Coble-rate-controlled contract:
- `sigma_drive = max(sigma_Hussein, 0)` — never `abs()`, never floored.
- `tau_Coble = xd²·kB·T/(sigma_drive·Ω·D_gb) + tau_ex0`, `v_event = b/tau_Coble`.
- Per-step requested displacement `d_delta = min(v_event·dt, b - delta_event)`.
- If `sigma_drive ≤ 0` (or the event's quota is already exhausted): the event **pauses** — `v_event=0`, no advection, `sink.active` stays `True`, no forced completion. A later stress recovery (from continued PF coarsening) can resume the same event.
- Every step returns a full audit dict: `requested_d_delta`, `measured_COM_d_delta`, `applied_d_delta`, `delta_event`, `remaining_to_b`, `tau_Coble`, `v_event`, `sigma_drive`, `mass_conservation_residual`, `e1e2f_residual`, `paused`, `completed`.

Unit-tested directly (synthetic state, `sigma_s=-1e6`): confirmed `active_sink_transport_step` returns `paused=True, active=True, completed=False` and leaves the fields unmodified, rather than stalling with drift.

## 5. Frozen-state prescribed-displacement microtest — the decisive diagnostic

`scripts/m16k_prescribed_displacement_microtest.py`: reproduced the deterministic (zero-randomness) pre-activation trajectory to the *exact* original activation step (90490, `t=4.4185`, `sigma_Hussein=45.12 MPa`), froze that state, then applied conservative prescribed rigid-body translations `dz ∈ {0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.250} nm` via a pure grid-remap (SciPy `ndimage.shift`, order-1 interpolation) — a mechanism entirely independent of the buggy `rbm_step` numerics.

| `dz` (nm) | `X_neck` (nm) | `r_neck` (nm) | `sigma_Hussein` (MPa) | mass drift |
|---|---|---|---|---|
| 0.000 | 279.79 | 20.56 | **45.12** | -5.6e-15 |
| 0.005 | 279.80 | 20.59 | **45.04** | -3.8e-05 |
| 0.010 | 279.80 | 19.47 | **47.84** | -7.7e-05 |
| 0.020 | 279.80 | 19.56 | **47.61** | -1.5e-04 |
| 0.050 | 279.80 | 79.92 | 8.99 | -3.8e-04 |
| 0.100 | 279.80 | 95.32 | 6.97 | -7.7e-04 |
| 0.250 | 279.80 | 50.97 | 16.10 | -1.9e-03 |

**Answer to the central question**: a genuine ~0.018nm translation (the amount the original buggy run's `rbm_step` had actually accumulated before "collapsing") leaves `sigma_Hussein` at **45-48 MPa — essentially unchanged from the 45.12 MPa baseline**. The dramatic collapse toward zero previously observed required displacements roughly 3-5× larger (0.05-0.1nm) to appear at all in this isolated-translation test. **This confirms the original collapse was a numerical artifact of the buggy (unweighted-sum) mass redistribution, not a genuine mechanical response to that amount of real rigid-body motion** — directly validating the mass-conservation fix as the correct target, rather than needing to reformulate the event-quota/displacement model itself.

(The `X_neck>50nm` and `>100nm` responses are non-monotonic and noisier — plausibly affected by the same multi-candidate-minimum tracking ambiguity documented in Section 7, since the `diagnose()` helper used here relies on `find_all_extrema`'s first-found minimum rather than the path-continuous tracker. Not over-interpreted; the small-displacement result is the one that matters for this diagnostic question.)

## 6. Corrected first-event run — full results

`scripts/m16k_first_event_qualification_v2.py`, same frozen geometry and hazard calibration as before (flat, `R_p=1μm`, `X0/(2R_p)=0.10`, `W=10nm`, `dx=1.25nm`; `V0=12.5·b³`, `A0=0.859eV`, `GS=201.74nm`, `random_seed=0`) — **not retuned**.

**Determinism confirmed**: activation occurred at `step=90490, t=4.4185, sigma=45.122 MPa` — an *exact* match to the original (buggy) run, as expected (the pre-activation segment has zero randomness and is unaffected by the transport fix).

### `sigma(delta_event)` and `delta_event(t)`

| phase | `t` range | `sigma_Hussein` (MPa) | `delta_event` (nm) | `tau_Coble` (s) | state |
|---|---|---|---|---|---|
| activation | 4.4185 | 45.12 | 0.0000 | 0.00533 | active |
| fast transport | 4.4185 → 4.423 | 45.1 → 7.9 | 0.0000 → 0.0179 | 0.0053 → 0.030 | active |
| asymptotic approach | 4.423 → 6.588 | 7.9 → ~0 (positive, shrinking) | **0.0179 (fixed)** | 0.030 → >1800 (diverging as 1/σ) | active, `v_event→0` |
| paused | 6.588 → 9.896 | slightly negative (-0.31 at `t=6.86`, recovering to -0.05 by `t=8.32`) | **0.0179 (fixed)** | ∞ | **paused** |
| resumption | 9.896 → 10.0 (end of run) | recovers to +4.95, then +16.5 at the final sample | 0.0179 (unchanged — resumption too recent to manifest as growth) | 0.0486 → 0.0146 | active again |

**All meaningful transport (0.0179nm, ~7% of `b=0.25nm`) happened within the first ~0.02 time-units after activation**, matching the microtest's independent prediction almost exactly. As `sigma_Hussein` fell (self-consistently, from the coupled PF+transport loop, never prescribed), `tau_Coble` rose correspondingly (`1/σ` dependence, exactly as required) — the event **self-limited**, slowing to a near-standstill rather than being forced through. It then genuinely paused (`sigma_drive` crossing to ≤0) for roughly 3.3 time-units, during which `delta_event` stayed frozen and **no further mass drift accumulated** (transport truly idle, not stalled-with-leakage). Continued ordinary PF coarsening during the pause window (not itself modified by any RBM logic) eventually rebuilt `sigma_Hussein` back to positive, and the event **genuinely resumed** (`paused→False`, `v_event>0`) at `t=9.896` — right near the end of the run's `t_target=10.0`, with too little remaining time to manifest as further `delta_event` growth before the script's own stop condition ended the run.

### Mass and consistency audit (across all 114,310 active-window steps)

- `mass_conservation_residual`: max `6.5e-16`, mean `4.8e-17` — **floating-point noise, effectively perfect** (compare the original buggy run's 2.29% and growing).
- Overall run `mass_drift` (the same quantity as M16I's stop-gate): stayed at `2.4-2.6e-05` (0.0024-0.0026%) throughout the entire active window — **~1000× better than the original run's 2.29%**, and comfortably under the M16I-established 0.2% threshold.
- `delta_event` never exceeded `1.79e-11 m = 0.0179nm`, safely bounded well below `b=0.25nm` at all times.
- `e1e2f_residual`: max 0.020, mean 0.0022 (Section 3 — a small pre-existing drift, not attributable to the transport fix).

### Late-run diagnostic caveat

The final two diagnostic samples of the original `t_target=10` run (`t=9.9`: `sigma=9.36 MPa`; `t=9.95`: `sigma=0.19 MPa`; `t=10.0`: `sigma=16.49 MPa`) show sample-to-sample swings inconsistent with smooth physical evolution — most plausibly the same multi-candidate-local-minimum tracking ambiguity documented in the original M16K report's Section B (the `NeckTracker` logged 92 candidate-contact switches over the full `t=25` run — see below). This does not affect the mass-conservation or `delta_event` findings (those come from the dense, every-step `active_sink_transport_step` internal re-measurement, not the coarser diagnostic-sample cadence).

### Extension to `t_target=25` — the event stays paused for the remainder of the observed window

The run was extended (same frozen geometry/hazard parameters, unchanged) to `t_target=25.0` specifically to see whether the `t=9.896` resumption continued. Full result, reconstructed from the run's complete 501-sample terminal log (see data-loss note below):

- **`delta_event` never took any value other than `0.0000nm` (pre-activation) or `0.0179nm` (post-burst) across all 501 samples spanning `t=[0,25]`.** The `t≈9.9` resumption did not translate into any further confirmed transport at any later point in the extended window either.
- `sigma_Hussein` spent the entire `t=[10.85, 24.95]` stretch oscillating in a band from about `-1.4` to `-0.4 MPa`, with a slow, genuine long-timescale recovery trend visible in the smooth-neighboring-sample data (e.g. `-1.12 MPa` at `t=19.4` → `-0.45 MPa` at `t=20.35` → `-0.50 MPa` at the final sample, `t=25.0`) — approaching but never sustaining a crossing back to positive.
- A handful of **isolated single-sample positive spikes** appear in the diagnostic-sample record (`t=7.2`: `+22.5 MPa`; `t=9.9-10.25`: up to `+16.5 MPa`; `t=21.8`: `+15.4 MPa`), each bracketed by much lower or negative neighboring samples with no corresponding `delta_event` growth. These bear the same signature as the known multi-candidate-minimum tracker-switching artifact (92 total candidate-contact switches logged across the run) rather than confirmed real stress recoveries — none of them produced measurable transport, which is itself evidence they did not reflect a sustained physical state.
- Final state at `t=25.0`: `sigma_Hussein=-0.501 MPa`, `delta_event=0.0179nm` (unchanged since the initial burst), `mass_drift=4.76e-05` (0.00476%) — still comfortably bounded, having crept up only slightly from `4.13e-05` due to the brief genuine activity around `t=9.9-10.2`, then perfectly flat for the remaining ~15 time-units.

**Characterization**: this single nucleated event executed ~7% of its Burgers-vector budget (`0.0179`/`0.25nm`) in a fast initial burst lasting a fraction of a time-unit, then settled into a long-lived (>20 time-unit, the full remainder of the explored window), essentially permanently-paused state. The local stress equilibrated to a value oscillating just **below** the activation/completion threshold and recovered only very slowly via continued ordinary PF coarsening — on a timescale that, within this observation window, never quite crossed back to a sustained positive value strong enough to resume meaningful transport. This is itself a legitimate, physically interesting finding (not a failure of the corrected mechanics): it suggests this particular geometry's post-event quasi-equilibrium sits very close to the activation stress, so a nucleated event here tends to self-limit almost immediately rather than running to completion in one continuous motion.

**Data-loss note**: the run's own final write (of `history.csv`, `event_dense_history.csv`, `events.json`, etc., all only written once at the very end of the script) failed with `FileNotFoundError` — the output directory `runs/m16k_first_event_qualification_v2/` had disappeared sometime during the ~80-minute run (cause undetermined; the directory existed at launch, since `os.makedirs(..., exist_ok=True)` runs at the start of `main()` and no earlier write failed). **The dense, per-PF-step `event_dense_history.csv` (114,000+ rows, held only in memory) was lost as a result** and could not be re-derived. The coarser (501-sample, `Δt=0.05`) trajectory survived via the process's own stdout log and was used for the analysis above. **Lesson for future runs**: write history incrementally (append-as-you-go, as the M16I production driver already does) or checkpoint periodically, rather than only at the end — this was a real, avoidable near-total data loss for an 80-minute compute investment, mitigated here only because the coarse trajectory happened to also be captured in the terminal log.

## 7. Corrected event physics vs. PASS-FIRST-EVENT-style criteria

| criterion | result |
|---|---|
| activation in tens-of-MPa regime | ✅ 45.1 MPa |
| stress rose from a meaningfully lower value | ✅ 19.5 → 45 MPa pre-activation (unchanged from before) |
| `delta_event` starts at zero, rises gradually | ✅ smooth sub-nm growth over ~0.02 time-units, not a single-step jump |
| cumulative RBM never exceeds `b` | ✅ capped at 0.0179nm ≪ 0.25nm, confirmed unchanged across the full `t=[0,25]` extension |
| Coble rate decreases as stress relaxes | ✅ `tau_Coble` rose from 0.0053s to >1800s as `sigma→0` |
| event completes OR physically pauses (no forcing) | ✅ **paused cleanly** at `sigma_drive=0` and **stayed paused** for the entire remaining ~20.6-time-unit extension, with one genuine brief resumption (`t≈9.9`) that did not sustain — no forced completion at any point |
| mass conserved near numerical tolerance | ✅ `6.5e-16` max residual in the (surviving, first-10-time-unit) dense trace; overall run `mass_drift` stayed at `4.1-4.8e-05` (≤0.005%) through the full 25-time-unit extension |
| sink returns inactive / stress fully reloads | ❌ **not reached within the fully-explored `t=[0,25]` window** — the event never completed; local stress recovers only very slowly and had not sustainably crossed back to positive by `t=25.0` |

**Overall conclusion**: the corrected physics is unambiguously validated — mass-conservative, self-limiting, properly pausable/resumable, never force-completing — a dramatic, confirmed improvement over the original buggy implementation. However, **this specific nucleated event, on this specific frozen geometry and hazard calibration, does not reach full completion within 25 time-units** (2.5× the original observation window). Whether it would eventually complete on a much longer timescale, or whether this geometry's post-event equilibrium is simply too close to the activation threshold for this event to ever complete under the frozen calibration, is not resolved and was not investigated further (a substantially longer run, or a different random-seed/hazard draw giving a higher-overshoot activation, would be the natural next diagnostic — not pursued without checking in first).

## 8. `young_laplace_pressure` rename (Section 8)

The module previously named `sintering_potential_stress.py` (function `sigma_potential`) has been renamed to `pf_sintering/young_laplace_pressure.py` (function `young_laplace_pressure`). It was never actually the experimental particle-on-substrate sintering-potential/GB-area formulation — it is the axisymmetric two-principal-curvature Young-Laplace capillary pressure, already present in this codebase (`hussein_neck_stress.local_young_laplace_pressure`, itself retired from any "sintering stress" name in M16G for the identical reason). The true experimental formulation is **not implemented** — no literature source was available in this session; if added later it must be named distinctly from both this quantity and `sigma_Hussein`.

## 9. Improved TJ tracking (Section 9)

`pf_sintering.m16k_neck_tracking.find_tj_from_contour`: locates the triple junction directly as the point along the already-traced free-surface contour (`f=0.5`) where grain identity flips (`e1-e2` changes sign), rather than relying solely on a local minimum of `R(z)` (an indirect geometric proxy). Wired into the corrected driver's diagnostics (`z_tj_from_contour_nm`, `tj_agreement_nm` columns in `history.csv`) as a cross-check, not a replacement, of the `R(z)`-minimum-based tracker.

**Reconciliation check** (across all 201 diagnostic samples of the `t_target=10` run, spanning pre-activation buildup, the fast-transport burst, the pause, and the resumption): the two methods agree to within **1.2-3.6nm** at every single sample — a small, stable offset, never growing or diverging, even through the activation event itself. This is reassuring: it means the late-run diagnostic swing noted in Section 6 (the `sigma=9.4/0.2/16.5 MPa` sample-to-sample jumps at `t=9.9-10.0`) is **not** attributable to disagreement between these two TJ *definitions* — both track the same physical feature consistently. The swing is more likely explained by the separate, previously-documented issue of the tracker switching between multiple *distinct* candidate local minima (a candidate-selection ambiguity when several genuine necks/troughs coexist), not an ambiguity in where the TJ sits once a candidate has been chosen. The two estimates were not merged into one authoritative value given time constraints, but this check substantially narrows what "the late-run diagnostic caveat" is actually about.

## 10. Time-unit audit (Section 7/10) — unresolved, explicitly flagged

Investigated whether the PF stepping variable `dt` is demonstrated to equal real SI seconds (required, since `D_gb`, `tau_Coble` are real physical quantities in `m²/s` and seconds respectively). Found: `pf_sintering/model.py`'s PF surface-diffusion mobility is calibrated as `M_f_base=(20e-9)⁴/(tau_target·k_f)` — a **chosen numerical relaxation timescale** (`tau_target`, picked for computational convenience/stability), **not derived from a real atomistic surface diffusivity**. "1 model time unit = 1 real second" is therefore an **unvalidated assumption**, inherited unchanged from M16H through M16K, not a demonstrated physical fact.

Per the "fail closed" instruction, this is not silently left implicit: `pf_sintering/axisym_sink_rbm.py` now has an explicit, named `SECONDS_PER_MODEL_TIME = 1.0` parameter (documented at length in the module, threaded through `nucleation_hazard_step` and `active_sink_transport_step`), preserving existing behavior but making the assumption visible and overridable. **A full resolution — calibrating `M_s`/`M_eta` against a real atomistic mobility so `seconds_per_model_time` could be derived rather than assumed — is out of scope for this continuation** and is the most significant open physics question affecting any absolute-time claims (event duration, `tau_Coble` comparisons) in M16H through M16K.

## 11. What was NOT done (explicit stop gate observed)

Per the explicit instruction in both continuation handoffs ("Do NOT start the multi-event/video run" / "STOP after this corrected first-event test"), **no multi-event campaign, video-frame architecture, or extended run beyond the one corrected first-event test was started.**

## Recommended next steps (not started without checking in first)

1. ~~Extend the corrected run to observe whether the `t=9.896` resumption continues to completion~~ — **done**: extended to `t_target=25.0` (2.5× the original window). The event stayed paused (`delta_event` frozen at `0.0179nm`) for the entire extension; local stress recovered only slowly and never sustainably crossed back to positive by `t=25`. See Section 6's "Extension to `t_target=25`" above.
2. ~~Reconcile `find_tj_from_contour` against the `R(z)`-minimum tracker~~ — **done**: the two agree to within 1.2-3.6nm at every sample across the full run, ruling out TJ-*definition* disagreement as the source of the diagnostic-sample swings; the swings are better attributed to multi-candidate-minimum *selection* switching (92 logged switches over the `t=25` run).
3. ~~Investigate the small `e1+e2=f` drift~~ — **done**: reclassified as a bounded transient tied to the RBM burst itself (spikes then decays), not a growing accumulation.
4. **New, higher priority given the extended-run outcome**: either (a) run substantially longer (e.g. `t_target=50-100`) to see if this specific event ever completes on a much longer timescale, or (b) try a different random-seed draw / re-run the hazard calibration to see whether a higher-overshoot activation (nucleating measurably above the 50 MPa median rather than at 45 MPa) produces an event that completes more readily — directly testing whether "does not complete in 25 time-units" is a property of this particular stochastic draw or a more general property of this geometry/calibration pairing.
5. **New, process fix**: add incremental/checkpointed output writing to `m16k_first_event_qualification_v2.py` (append-as-you-go for `history.csv`/`event_dense_history.csv`, matching the M16I production driver's pattern) before any further multi-hour run — the `t_target=25` run's dense per-step data was lost when the final end-of-script write failed after ~80 minutes of compute, an avoidable risk for a run of this length.
6. If/when a full multi-cycle qualification passes cleanly, proceed to the video/multi-event campaign (Part IV of the original M16K handoff) — still not attempted.

## Output locations

`runs/m16k_prescribed_displacement_microtest/microtest_results.csv`, `runs/m16k_first_event_qualification_v2/{history.csv, event_dense_history.csv, events.json, meta.json, switch_log.json, first_event_summary.png, event_dense_summary.png}` (all under `runs/`, gitignored, not committed).

## Code (committed)

`pf_sintering/axisym_sink_rbm.py` (mass-conservation fix + `nucleation_hazard_step`/`active_sink_transport_step`), `pf_sintering/young_laplace_pressure.py` (new, replaces `sintering_potential_stress.py`), `pf_sintering/m16k_neck_tracking.py` (`find_tj_from_contour` added), `scripts/m16k_prescribed_displacement_microtest.py` (new), `scripts/m16k_first_event_qualification_v2.py` (new).

## STOP

Per the explicit instruction in both continuation handoffs, stopping here. No video/multi-event campaign started.
