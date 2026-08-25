# Stochastic PR avalanche v1 root-state replay

## Scope and provenance

Development branch: `development/stochastic-pr-avalanche-v1`

Frozen parent: annotated tag `stochastic-pr-production-v1`, peeled commit
`d2aafbf7df99728acb67af941f2c078126455e1f`.

The production root-event layer was not changed. The replay retains the exact
1830.15 K root EXP-floor barrier, local geometric root activation stress,
`N_sites = 2*pi*r_TJ/b`, `b = 0.25 nm`, the qualified finite one-b PF event,
`C4 = 0.05`, transport physics, and the frozen physical-time mapping. The
external barrier export SHA-256 is
`fa7c9a2fb30e55596d9cdf47d2b03d530ef073cbe15f6210282285b2fcee7f37`.

The new code is an external D2-style controller only. It adds:

- an absolute, non-additive descendant barrier envelope;
- one serialized descendant queue;
- descendant hazard integration during every finite one-b transit;
- multiple threshold crossings during one transit;
- a fixed post-transit correlation window for natural extinction;
- no DDD or disconnection elastic-stress facilitation.

The root transit is included in avalanche size `S`, so total displacement is
`Q_avalanche = S*b`.

## Single-parameter calibration

Twenty archived production event histories spanning 67.727--92.092 MPa were
used without repeating their sink-OFF waits. The fixed correlation interval is
the median duration of the 20 verified production one-b events:

`tau_corr = 1.947861854319964e-6 s`.

Only `G0_step_eV` was calibrated. A deterministic reduced queue replay used
256 calibration draws per archived root and a calibration-only seed of
20260824, independent of every full PF root-replay seed. It selected:

`G0_step = G0_eff = 3.760085178265583 eV`.

The authoritative root values used by the descendant envelope were
`G0 = 5.3646516047156325 eV`, `Gfloor = 2.9476002588459194 eV`,
`a = 0.9682214719449614`, `sigma_hat = 1.6972329282808514 GPa`, and
`n = 0.46101864227254197`. The root floor fraction was preserved exactly.

The reduced calibration median was 7.5, but its 34.6% cap-100 censoring rate
already indicated a broad near-critical tail. This diagnostic did not replace
the subsequent full PF replay.

## Full PF root replay

Ten stress-stratified archived roots were selected independently of descendant
thresholds. Descendant RNG seeds were derived once from
`SHA256("sintering-avalanche-v1|production_seed|root_cycle")`; no seed was
inspected, rejected, or changed. Each full one-b event used the unchanged
production event solver with a hard `C4 == 0.05` assertion. The declared replay
cap was 20 completed transits.

| Case | Root stress (MPa) | S | Status | Q (nm) | Duration (us) | Final local stress (MPa) | Local drop (MPa) |
|---|---:|---:|---|---:|---:|---:|---:|
| R08_E01 | 67.727 | >=20 | right-censored | >=5.00 | >=161.490 | 29.792 | >=37.935 |
| R07_E02 | 70.804 | 2 | extinct | 0.50 | 8.283 | 55.071 | 15.733 |
| R07_E01 | 71.383 | >=20 | right-censored | >=5.00 | >=107.505 | 28.312 | >=43.071 |
| R09_E01 | 74.148 | 2 | extinct | 0.50 | 8.373 | 56.086 | 18.062 |
| R10_E01 | 75.260 | 6 | extinct | 1.50 | 23.014 | 44.650 | 30.610 |
| R03_E05 | 79.517 | 4 | extinct | 1.00 | 6.847 | 60.357 | 19.160 |
| R07_E03 | 81.967 | 4 | extinct | 1.00 | 11.215 | 60.463 | 21.504 |
| R09_E02 | 84.809 | 6 | extinct | 1.50 | 13.101 | 57.957 | 26.853 |
| R09_E04 | 89.833 | 6 | extinct | 1.50 | 11.571 | 67.483 | 22.349 |
| R07_E05 | 92.092 | 3 | extinct | 0.75 | 6.834 | 77.862 | 14.230 |

There were eight exact self-terminating avalanches, two right-censored tails,
and zero event-operator blockers. Exact finite sizes were
`2, 2, 3, 4, 4, 6, 6, 6`, giving mean 4.125, median 4, and S95 6. Their exact
empirical probabilities were `P(2)=0.25`, `P(3)=0.125`, `P(4)=0.25`, and
`P(6)=0.375` conditional on observed extinction.

When the two capped cases are retained correctly as lower bounds, the observed
lower-bound mean is 7.3 and median is 5.0. The discrete Kaplan-Meier median is
4 (survival reaches 0.5 at S=4), and 20% survival remains beyond S=20. The
experiment-scale 5--10b response is therefore present in several unbiased
realizations and in the lower-bound median, while the distribution is broad:
the exact-extinction median alone is below the desired interval and two tails
are not resolved. This replay demonstrates finite, self-terminating avalanches
without an event-solver blocker, but it does not justify hiding or treating the
two long tails as finite S=20 events.

For the finite subset, avalanche duration spans 6.834--23.014 us (median
9.794 us), cumulative displacement spans 0.50--1.50 nm, and local stress drop
spans 14.230--30.610 MPa (median 20.332 MPa).

## Outputs

All outputs are under:

`runs/pr_current_head_regression/coarsening_driven_fourier/avalanche_v1_root_replay`

Important files:

- `barrier_preflight.json` and `barrier_preflight_profiles.csv`;
- `replay_manifest.json`;
- `avalanche_results.csv` and `avalanche_summary.json`;
- `avalanche_size_distribution.csv` for exact finite `P(S)`;
- `avalanche_size_observations.csv` for exact/censored status;
- `avalanche_size_kaplan_meier.csv` for censor-aware survival;
- `representative_avalanche.png/.pdf` with all requested channels;
- `avalanche_calibration_summary.png/.pdf` with censored tails marked;
- per-root `result.json`, `subevents.csv`, hazard crossings, time traces,
  PF contour histories, and full event checkpoints.

## Verification

All three new Python files compile. Five focused controller tests pass through a
direct Python test harness. The local `pytest` launcher exits with signal 11
even for an existing production test, so pytest itself could not provide a
valid pass/fail result in this environment.

No larger avalanche production ensemble was launched.
