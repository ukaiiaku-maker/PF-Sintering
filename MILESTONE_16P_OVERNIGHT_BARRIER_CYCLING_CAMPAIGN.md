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

**Ceiling confirmed: sigma peaks at ~73.82MPa around t~0.22-0.24, then declines extremely slowly (73.816->73.814->73.810MPa from t=0.24 to t=0.26)** -- matching the earlier geometric-decay extrapolation (74-76MPa) closely, and confirming Section 5's first guidance case ("if the sink-OFF trajectory plateaus around 74-76MPa, choose a demonstration median activation near 72-74MPa"). The calibration target of 72MPa (Section 5-6 below) sits comfortably inside `sigma_initial(69.64MPa) < target(72MPa) < ceiling(~73.8MPa)`. The extended run continued past this peak in the background (`scripts/m16o_topology_screen.py --chi 1.5 --ratio 0.185 --t-target 0.8`) to characterize the post-peak behavior for the report's completeness.

## 5-6. Finite-barrier target selection and calibration

**Target selected: 72MPa**, comfortably inside `sigma_initial(69.64MPa) < target < extrapolated_ceiling(~74-76MPa)`, per Section 5's guidance for a trajectory expected to plateau around 74-76MPa ("choose a demonstration median activation near 72-74MPa").

**Calibration** (`scripts/m16p_calibrate_barrier.py`, against the M16O `chi=1.5` deterministic trajectory, `t=0-0.2`): solved `A0` such that the integrated Arrhenius hazard `H(t) = integral Lambda(sigma(t'),T) dt'` reaches exactly `ln(2)` at `t_target=0.03` (the first time the deterministic trajectory crosses 72MPa) -- the median first-passage condition.

```
V0 = 12.5*b^3         (unchanged, frozen since M16K)
A0 = 0.466794 eV        (solved)
r0 = 1e12
GS = 201.74nm          (unchanged)
T  = 1000K
target stress = 72MPa, reached deterministically at t=0.03
H(t_target) = 0.693146  (ln(2) = 0.693147, matches to 6 significant figures)
```

This satisfies the essential requirement (`sigma_initial < typical nucleation stress < natural sink-OFF ceiling`) so the system visibly loads (`t=0 to t~0.03`, `sigma: 69.6->72.0MPa`) before the median nucleation event, without exceeding the reachable ceiling (subsequently confirmed at ~73.8MPa, Section 4 above). Not repeatedly retuned; frozen for Run A and Run C (the finite-barrier controls) per the explicit "one recalibration, then freeze" instruction.

## Discrete-grid neck-position artifact (found and fixed during Run A)

Run A's first launch produced a genuine correctness anomaly requiring investigation before trusting further results, per Section 31's explicit fail-closed instruction: after the first birth/completion (a real, large single-event relaxation, `sigma: 72.00MPa -> ~46-48MPa` -- see below), the reported `sigma(t)` showed a discontinuous **+20MPa jump within a single PF step, with `N_active=0`** (no sink activity active to explain it physically).

**Root cause** (verified via a direct instrumented replay, `scripts/m16p_barrier_run.py`'s `operative_sigma`/`NeckTracker` reproduced step-by-step): `n_candidates=1` throughout the jump (NOT the previously-documented M16K/M16M multi-candidate tracker-switching artifact) -- instead, the DISCRETE grid-cell location of a single, genuine physical R(z) minimum hopped by exactly one grid spacing (`dz=0.85nm`) between two consecutive PF steps, because `find_all_extrema`/`NeckTracker` only ever compare discrete grid points, with no sub-grid interpolation. This `chi=1.5` geometry's very tight neck (`r_neck~13-15nm`, only ~16-18 grid cells at `dx=0.85nm`) makes the minimum shallow/near-degenerate at grid resolution -- a genuinely new numerical regime relative to the gentler necks explored in M16H-M16O.

**Fix**: added a standard 3-point quadratic (parabolic) sub-grid interpolation around the discrete minimum (clipped to +/-1 grid cell for safety), applied to both the curvature-window evaluation and the `GB_z_hint` fed back into the RBM excess-mass redistribution. Verified via a second instrumented replay: completely eliminates the jump in the exact case that exposed it (smooth, continuous `sigma` before/after/through the region that previously jumped). Run A relaunched with the fix.

## Remaining sections (8-34)

Not yet reached. Will be completed or explicitly deferred with reasoning as this session's practical compute budget allows, following the priority order: (8-10) finite-barrier cycling qualification for the loading geometry, (11-16) movie campaign if qualification passes, (17-22) zero-barrier and relaxing-geometry controls completing the 2x2 matrix, (24-27) comparison tables/figures/cycle statistics/Poisson audit, (34) final synthesis.

Given the realistic wall-clock cost observed so far in this project (a single short `t=0-0.2` sink-off screen at this domain size takes ~20-27 minutes; a full stochastic multi-event run with the calibrated barrier will need to integrate through the SAME natural-loading timescale repeatedly across multiple cycles, each cycle plus reload potentially requiring a comparable or greater duration) -- the full 2x2 matrix plus an 300-800-frame movie campaign is realistically a multi-hour-to-multi-day undertaking. This report will honestly reflect how far the campaign actually got within the practical bounds of this session, rather than claiming completion of stages not actually run.
