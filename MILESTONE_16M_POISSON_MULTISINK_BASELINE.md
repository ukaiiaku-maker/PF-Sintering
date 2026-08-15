# M16M — Independent Poisson Multi-Sink Baseline

**Status (in progress): infrastructure complete, unit-tested, and committed. The bounded multi-event PF qualification run (Section 20) is executing in the background; this report will be completed with its results once the run reaches a stopping condition.**

## 1. Starting commit and test status

- Started from branch `codex/m16k-physical-hazard-video`, commit `52477c2` (M16L's final, qualified state).
- Verified clean (`git status --short` empty, `git diff --check` clean) and 244/244 tests passing before any M16M change.
- New work developed on branch `codex/m16m-poisson-multisink`.
- After adding M16M's infrastructure (Sections 2-4 below) and its own test suite: **257/257 tests passing** (244 M16L baseline + 13 new M16M state-machine/role tests).

## 2. Explicit grain-role architecture

M16L's central finding was that `pf_sintering.m16j_geometry.build_candidate_geometry` assigns `e1=substrate, e2=particle` — the opposite of what `pf_sintering.axisym_sink_rbm.py`'s functions (validated against the older M16G geometry) assume. M16K's entire investigation silently advected/measured the wrong grain as a result.

To prevent a repeat of this class of bug, `pf_sintering/grain_roles.py` introduces:

- `GrainRoles` (dataclass: `particle`, `substrate`) and `roles_from_m16j_geometry(e1, e2)`, which performs the swap explicitly and by name immediately after geometry construction.
- `assert_particle_above_substrate(particle, substrate, z, r_c)`: a runtime guard (mass-weighted COM comparison) that raises `AssertionError` if particle/substrate are accidentally passed in swapped. Verified via `tests/test_m16m_grain_roles.py` both on synthetic fields and on the real M16J-constructed geometry.

All M16M code (the multi-sink module, the qualification driver) receives explicitly named `particle`/`substrate` arguments throughout — no function in the new code path accepts raw, ambiguous `e1`/`e2`.

## 3. Multi-event state-machine tests

`tests/test_m16m_multisink_state_machine.py` (13 tests) covers Section 13's items A-H on synthetic fields (no PF solver), plus two additional audits:

- **A.** Two events with different remaining displacement both progress and cap independently at `b`.
- **B.** One near-complete event finishes while a fresh event, in the same step, makes independent nonzero progress without also being force-completed.
- **C.** Two events completing in the same PF step are both correctly detected and deactivated.
- **D.** The Poisson birth clock fires new births while multiple `SinkEvent`s are already active — confirms the clock never gates on `N_active`.
- **E.** Completing one event leaves two other active events untouched (still active, still independently progressing).
- **F.** `N_active` traced through 0→1→2→3→…→1 without corrupting any event's cumulative displacement or exceeding `b`.
- **G.** The sum of all per-event applied increments in one transport call exactly equals the field's own measured total displacement (proportional-distribution bookkeeping is exact, not merely approximate).
- **H.** No individual event's `delta` ever exceeds `b` under 3000 steps of sustained multi-event driving.
- **`lambda_birth` audit**: confirms the birth intensity is a pure function of `(sigma, hp)` only — never implicitly scaled by call count or active-event count (Section 4/5's explicit requirement).
- **Poisson integrated-hazard sanity** (Section 24, light-weight): at constant stress, the mean of consumed hazard-per-birth (`clock.consumed_thresholds`) converges toward the Exp(1) mean of 1.0, as expected for a correctly-implemented nonhomogeneous-Poisson thinning algorithm.

Two real numerical lessons surfaced while building these tests (both fixed before commit, not swept under the rug):

1. An initial `dt=1.0`/`dt=0.01` test design accidentally forced *every* fresh event's requested displacement to be capped at its full `b` budget in a single step (because `v_event*dt` at 100MPa vastly exceeds `b` at those `dt` values) — this made tests B/E/F/G spuriously fail (multiple events completing simultaneously when the test intended only one to). Fixed by choosing `dt` values (`1e-4`) that keep the natural per-step request comfortably below a fresh event's full budget, matching the realistic per-PF-step displacement scale later observed in the actual qualification run (~`1e-3` of `b` per step near activation).
2. The `dt=0.05`/`sigma=60MPa` Poisson-diagnostic test initially requested 500 births within a budget that, at this milestone's realistic `Lambda(60MPa)≈0.21`/s, would need ~2350 simulated seconds — far more than a fast unit test should take. Reduced the target to 300 births at a larger `dt` (the birth clock is a pure hazard-integral calculation here, with no field mutation, so `dt` can be made arbitrarily large without any physical-accuracy cost for this test).

## 4. Poisson birth-clock implementation

`pf_sintering/m16m_multisink.py::poisson_multisink_birth_step(clock, sigma_s, dt, hp, rng, seconds_per_model_time)`:

- Maintains a persistent `PoissonBirthClock` (`hazard`, `threshold`, `consumed_thresholds`) **independent of any `SinkEvent`'s state** — it is never consulted against, or gated by, `N_active`.
- Integrates `dH = Lambda(sigma,T)*dt_seconds` into `clock.hazard` every call.
- Fires births via a **while-loop** (not an `if`): `while clock.hazard >= clock.threshold: clock.hazard -= clock.threshold; draw new threshold; n_births += 1`. This is the standard thinning-algorithm form for simulating a nonhomogeneous Poisson process, and correctly handles more than one birth occurring within a single PF step (Section 6's explicit requirement) — verified by `test_D_birth_clock_fires_while_events_already_active`.
- Each consumed `threshold` is recorded, giving a direct sample of the inter-birth integrated hazard for the Section 24 Poisson diagnostic.

## 5. Confirmation: hazard calibration NOT multiplied by any site count

`lambda_birth(sigma_s, hp)` is **byte-for-byte the same formula** as M16L's `nucleation_hazard_step`'s per-call `r_nuc`:

```
Lambda(sigma,T) = r0 * (b/GS)^3 * exp(-max(0, A0 - sigma*V0) / (kB*T))
```

with the frozen M16L calibration (`b=2.5e-10`, `V0=12.5*b^3`, `A0=0.8589eV`, `GS=201.74nm`, `r0=1e12`, `T=1000K`) carried forward **unchanged**. `test_lambda_birth_not_multiplied_by_site_count` confirms the function is a pure function of `(sigma, hp)` alone. The only behavioral change relative to M16L is that the "no new nucleation while one event is active" restriction is removed (Section 6) — the clock runs continuously regardless of `N_active`.

## 6-7. Lambda(sigma), tau_event(sigma), and B(sigma)

Computed via `scripts/m16m_overlap_diagnostic.py` (`runs/m16m_overlap_diagnostic/overlap_table.csv`), purely from the frozen calibration — **not tuned to obtain any particular value**:

| sigma (MPa) | Lambda (1/s) | mean wait (s) | tau_event (s) | B=Lambda·tau_event |
|---:|---:|---:|---:|---:|
| 20 | 0.1185 | 8.44 | 0.01202 | 1.424e-3 |
| 30 | 0.1365 | 7.33 | 0.00802 | 1.094e-3 |
| 40 | 0.1572 | 6.36 | 0.00601 | 9.451e-4 |
| 45 | 0.1688 | 5.93 | 0.00534 | 9.017e-4 |
| 50 | 0.1811 | 5.52 | 0.00481 | 8.710e-4 |
| 60 | 0.2086 | 4.79 | 0.00401 | 8.361e-4 |
| 75 | 0.2580 | 3.88 | 0.00321 | 8.270e-4 |
| 100 | 0.3674 | 2.72 | 0.00240 | 8.834e-4 |

**Interpretation: `B(sigma)` is between 8×10⁻⁴ and 1.4×10⁻³ across the ENTIRE 20-100MPa stress range — always `B<<1`.** Under the current calibration, mean waiting times between births (2.7-8.4 seconds) are roughly 1000× longer than a single event's own diffusion-controlled lifetime (2.4-12ms). This is a strong, first-principles prediction, made **before** running the expensive PF simulation, that at this calibration **overlapping active sinks (`N_active>1`) should be extremely rare** — the kinetics predict temporally isolated events (Regime A), not an overlapping Poisson burst (Regime C), essentially independent of exactly where in 20-100MPa the trajectory sits.

## Section 30: time-scale caveat

`B(sigma)=Lambda(1/s)*tau_event(s)` is computed entirely from SI-second quantities and is therefore **independent of the unvalidated `SECONDS_PER_MODEL_TIME=1.0` mapping** carried over from M16K/M16L — changing that mapping would rescale how many PF steps are needed per event (and per waiting interval) but would not change `B(sigma)` itself. This is stated explicitly rather than left implicit; the caveat itself remains otherwise unresolved this milestone (no new calibration against a real atomistic mobility was attempted).

## 8. Robust TJ/curvature-consensus implementation and stress-validity statistics

`pf_sintering/m16m_stress_consensus.py` implements the full architecture M16L flagged as its single largest remaining scope gap:

- **TJ/X_neck**: supplied by the caller from the SAME path-continuous `NeckTracker` M16K/M16L validated (never a stale position against a fresh morphology, per Section 19) — all three curvature estimators below are evaluated against this one shared `(z_gb, a_contact)`.
- **Method 1 — circle-fit family**: Kasa algebraic circle fits (existing `neck_curvature_windows`) at fixed physical half-windows {6, 9, 12, 18, 24} nm; consensus value = median across windows.
- **Method 2 — local polynomial fit**: independent fitting basis (quadratic Taylor expansion of R(z) in a fixed window, analytic curvature from the fit coefficients).
- **Method 3 — diffuse-interface curvature**: computed directly from the phase field's own gradient (`-div(grad f/|grad f|)`, axisymmetric form) — the only one of the three that does **not** use the traced R(z) contour at all.
- **`stress_valid`**: True only if ≥2 methods give a finite positive `r_neck` and the resulting `sigma_Hussein` values agree within `spread_tol` (15% initial).

**Honest finding from calibration testing** (against the M16K/M16L frozen ~45MPa state): the local-polynomial method (Method 2) proved numerically unstable at windows ≥9nm for this specific groove geometry (non-monotonic, order-of-magnitude swings across window choice) and was retuned to a smaller, more locally-accurate 6nm window; the diffuse-interface method (Method 3) required both Gaussian pre-smoothing and a finite-difference offset comparable to the other methods' window scale (a 1-2 grid-cell offset is dominated by the diffuse interface's own intrinsic thickness, not the mesoscale neck curvature). Even after these fixes, the three methods still show a substantial **~40% spread near the ~45MPa activation point**, and a much larger (order 100-250%) spread at other points in the trajectory (e.g. t=0). **This means a strict, literal 15% consensus gate — used as a hard blocking condition on every kinetic step — would pause the large majority of samples and prevent essentially all multi-event physics from being exercised.**

This is reported as a genuine, first-class finding of this milestone (the pre-existing single-window sigma measurement used throughout M16H-M16L carries a real, previously under-characterized method/window-dependent uncertainty band for this specific geometry), not swept under the rug. The **disclosed, deliberate choice** made for this first qualification pass: the already-validated single-window circle-fit measurement (identical to M16L's own convention, 1.5×W=15nm window via the path-continuous tracker) is used as the **operative** kinetic driver every step; the full 3-method consensus is run as a **periodic audit overlay** (every `diag_every * 15` steps) whose spread/`stress_valid` statistics are logged and reported, not as a run-blocking gate. Stress-validity statistics from the actual run are reported in Section 9 below.

## 9-17, 19. [PENDING — bounded multi-event qualification run in progress]

The bounded PF run (`scripts/m16m_multisink_qualification.py`, launched from the canonical M16J geometry with the frozen hazard calibration unchanged) is executing in the background against Section 20's stopping criteria (earliest of: 20 completed events; cumulative RBM = 20·b; a clear ≥10MPa stress decrease from a local post-nucleation maximum; a Section 11 request/measurement-gate failure; or a pragmatic 4-hour wall-clock budget — the last explicitly NOT a scientific criterion). Given the wait-time/lifetime figures in Section 6-7 above (mean wait 5.5-5.9s of *model time* near the activation stress, translating to tens of minutes of wall-clock per event at this run's ~40-65 PF-steps/second throughput), reaching even a handful of completed events is expected to take on the order of hours; this section will be completed with the actual `N_active(t)`, cumulative event count, cumulative RBM/b, sigma-vs-cumulative-RBM/b trajectory, waiting-time/integrated-hazard diagnostics, and the observed Regime (A/B/C/D per Section 22) once the run reaches a stopping condition.

## 18. Time-scale caveat (carried forward)

Unchanged from M16K/M16L: `SECONDS_PER_MODEL_TIME=1.0` remains an explicit, unvalidated assumption (the PF model's `dt` is treated as literal SI seconds when combined with real-physics `tau_Coble`/`D_gb`). Section 6-7 above already establishes that the overlap number `B(sigma)` itself does not depend on this mapping; the wall-clock/step-count estimates for the qualification run's expected duration DO depend on it, and are reported as such.

## 20. Recommendation on the video campaign

**Not yet applicable — the qualification run has not reached a stopping condition.** Per the explicit standing instruction (Sections 31/34), the video/multi-event campaign will not be started regardless of this run's eventual outcome without a further explicit go-ahead; this report will be updated with a concrete recommendation once Sections 9-17 above are filled in with real results.
