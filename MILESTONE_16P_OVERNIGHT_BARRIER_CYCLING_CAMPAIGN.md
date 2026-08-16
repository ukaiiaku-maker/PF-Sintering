# M16P — Overnight Finite-Barrier Cycling Campaign

**Central scientific test**: (1) does initial geometry/topology determine whether the sink-OFF system naturally loads or relaxes (established by M16O), and (2) in a naturally-loading geometry, does a finite nucleation barrier produce repeated loading/nucleation/relaxation/reloading cycles, while a zero barrier suppresses that intermittent response via continuous accommodation?

**Status: in progress.** This is a genuinely multi-hour-to-multi-day campaign (stochastic multi-event PF production runs plus a high-cadence movie campaign). This report documents real, honest progress against the full 35-section request, updated as background computation completes across this session. Sections not reached by the practical limits of this session are explicitly flagged as follow-up, not silently omitted.

## 1. Starting state / provenance

- Branch: `codex/m16m-poisson-multisink`
- HEAD at start: `8845ece` (M16O final)
- `git status --short` / `git diff --check`: clean
- Baseline: 261/261 tests passing, matching expected.

## 2. Physics frozen (carried forward)

M16N's corrected one-b contract (`delta_sink` drives completion, `delta_COM` diagnostic-only), independent Poisson nucleation, no hysteresis/avalanche/branching/barrier-lowering/sink-sink-interaction/refractory-period -- all unchanged.

## 7. RBM mass-conservation repair (completed FIRST, before any multi-event production, per explicit gate)

**Root cause found and fixed.** The excess-mass redistribution in both `active_sink_transport_step` and `multi_sink_transport_step` only ever compensated the `f>1` excess side of `np.clip(f, 0.0, 1.0)` -- small negative undershoots from the upwind advection scheme (a standard artifact at sharp gradients) were silently clipped UP to 0 every substep, adding uncompensated mass. This single mechanism accounted for the ENTIRE previously-observed ~8.8e-6 relative mass residual per one-b event (verified via a direct instrumented replay, `scripts/m16p_mass_residual_microtest.py`, matching the prior measurement to 4 significant figures once correctly reproduced).

**Fix**: track both `excess=max(0,f-1)` and `deficit=max(0,-f)` before clipping, apply ONE combined (excess-minus-deficit) correction via the same Gaussian-weighted redistribution mechanism (not two separate corrections, which could overshoot).

**Result**: mass drift after one full one-b event: **8.87e-6 -> -2.3e-16** (floating-point precision, ~10 orders of magnitude improvement). New regression test `tests/test_m16p_mass_conservation.py`: 10 repeated one-b events back-to-back (both single-event and multi-sink code paths) show no systematic accumulation, max residual well under the preferred `<=1e-10` gate. **263/263 tests passing.**

## 4. Natural loading range (chi=1.5 extended trajectory)

**In progress.** Extending the deterministic sink-OFF `chi=1.5`, `ratio=0.185` trajectory from `t=0.2` (M16O's endpoint, `sigma=73.79MPa`) toward the Section 4 stopping criteria (plateau, `sigma>=85MPa`, `t~0.6-1.0`, or a numerical issue). This section will be completed with the full extended dataset, the observed ceiling/plateau, and the `L_r`/`L_X` decomposition over the extended range once the run completes.

## 5-6. Finite-barrier target selection and calibration

**Pending** completion of Section 4's extended trajectory -- the target cannot be responsibly chosen before the natural ceiling is known (per the explicit "do NOT choose a target above the reachable deterministic ceiling" instruction).

## Remaining sections (8-34)

Not yet reached. Will be completed or explicitly deferred with reasoning as this session's practical compute budget allows, following the priority order: (8-10) finite-barrier cycling qualification for the loading geometry, (11-16) movie campaign if qualification passes, (17-22) zero-barrier and relaxing-geometry controls completing the 2x2 matrix, (24-27) comparison tables/figures/cycle statistics/Poisson audit, (34) final synthesis.

Given the realistic wall-clock cost observed so far in this project (a single short `t=0-0.2` sink-off screen at this domain size takes ~20-27 minutes; a full stochastic multi-event run with the calibrated barrier will need to integrate through the SAME natural-loading timescale repeatedly across multiple cycles, each cycle plus reload potentially requiring a comparable or greater duration) -- the full 2x2 matrix plus an 300-800-frame movie campaign is realistically a multi-hour-to-multi-day undertaking. This report will honestly reflect how far the campaign actually got within the practical bounds of this session, rather than claiming completion of stages not actually run.
