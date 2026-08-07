# Operator-resolved audit of coarsening-driven neck widening

Scope: the operator-resolved ledger, spatial Ostwald audit, O0/O1/O2 mechanism
decomposition, and CH flux analysis for the primary case (wider geometry,
`coarsening=3x, surface=0.3x`), plus a compact cross-check on the baseline
(5nm-overlap) geometry. **Stopping here for review, as instructed** — no
production physics was modified.

## A. Git / test state

- Branch: `codex/coarsening-stress-buildup`.
- Starting commit (Milestone-5 checkpoint, made at the start of this session):
  `04730ed` "feat: natural-coarsening rate-competition benchmark and diagnostics".
- New commits this session (A-D per the requested split):
  1. `3093587` operator-resolved geometry ledger infrastructure.
  2. `2c13e5f` Ostwald spatial-transfer diagnostics + O0/O1/O2 decomposition.
  3. `8cb296f` CH chemical-potential / flux diagnostics.
  4. (this commit) audit driver + tests + this report.
- Tests: **49/49 passing** (`pytest ./PF-Sintering/tests -q`) — 12 new this
  session (`test_operator_ledger.py` x4, `test_ostwald_diagnostics.py` x5,
  `test_ch_flux_diagnostics.py` x3), none modified.

## B. Active operator call chain (verified, not assumed)

```
model.evolve_f is parity_kernels.evolve_f            -> True
model.ostwald_substrate is parity_kernels.ostwald_substrate -> True
model.evolve_eta                                      -> model.py's own implementation
                                                          (parity_kernels defines no evolve_eta,
                                                           so nothing monkey-patches it)
structural_projection.project_eta_mass_preserving     -> single implementation, never patched
```

This matches the existing `test_parity_wiring.py` contract and was
independently re-verified by identity check at the start of this session.
Both new diagnostic mirrors (`ostwald_substrate_diagnostic`,
`evolve_f_diagnostic`) were verified to reproduce **bit-for-bit identical**
`(f, e1, e2, e3)` output to the live production kernels for the same input
(`tests/test_ostwald_diagnostics.py::test_diagnostic_kernel_matches_production_exactly`,
`tests/test_ch_flux_diagnostics.py::test_diagnostic_kernel_matches_production_exactly`)
before being used for anything else.

## C. Operator ledger

Primary case: `dev` preset, `R2=80nm`, `aspect_ratio=2.0`,
`contact_orientation=short_plane`, `initial_overlap=20nm`,
`coarsening_rate_scale=3.0`, `surface_mobility_scale=0.3`,
`eta_mobility_scale=1.0`, fixed `dt=7.2624e-6s` (same convention as the
Milestone-5 campaign). Target `|dV2|/V20 ~ 3e-4` (found necessary after an
initial `1e-4` run gave a correctly-behaved but very small signal — closure
was already exact there too, just harder to read in absolute terms).
Sink/RBM never called anywhere in this audit.

**276 steps**, `dV2/V20 = -3.006e-4`. Per-operator cumulative contribution to
each total observed change (closure error is **exactly 0.0** for every
quantity, every run — a telescoping sum by construction, not an
approximation):

| operator | d_x_neck (nm) | % of total | dA_GB (nm^2) | dV2 | dsigma (MPa) | % of total sigma |
|---|---:|---:|---:|---:|---:|---:|
| CH | +0.000000 | 0.0% | +0.0000 | 0 | -0.17011 | 90.3% |
| post-CH projection | +0.006502 | 34.6% | +2.5219 | ~0 | -0.00655 | 3.5% |
| Ostwald | +0.001702 | 9.1% | +0.0481 | **-5.990e-18 (100%)** | -0.00098 | 0.5% |
| eta relaxation (raw) | +0.008126 | 43.2% | +1.3432 | ~0 | -0.00818 | 4.3% |
| post-eta projection | +0.002474 | 13.2% | +1.3485 | ~0 | -0.00249 | 1.3% |
| **total** | **+0.018805** | 100% | **+5.2616** | -5.990e-18 | **-0.18831** | 100% |

Two things are exact, not approximate, and independently confirm the
qualified mass-preserving physics contract is intact: (1) **CH's raw step
cannot change `x_neck` or `A_GB` at all** -- both are pure functions of
`(e1, e2)`, which `evolve_f` never touches, so their `CH` row is `0.0` to
the bit, verified directly rather than assumed; (2) **100% of the `V2`
change is attributed to `Ostwald`** -- every other operator's `V2`
contribution is at floating-point roundoff (`~1e-28`-`1e-29`), confirming
CH/projection/eta are not secular grain-volume-change mechanisms here
either, exactly as `README.md`'s physics contract requires.

The **x_neck and A_GB widening signal is dominated by post-CH-projection +
eta relaxation + post-eta-projection (91% combined)**, not by Ostwald's own
direct transfer (9%). The **sigma relaxation signal is dominated by raw CH**
(90%) -- a different geometric quantity (curvature of the `f=0.5` contour,
which raw CH does change) responding differently than `x_neck`/`A_GB`
(functions of `e1,e2`, which raw CH cannot touch).

## D. Spatial Ostwald transfer

At the initial state (and materially unchanged through the 276-step
trajectory -- see table below), the exposed `removal_candidate`
(`surf*e2*incl`, pre-convolution) and applied `remove`/`add` fields were
plotted (`runs/operator_audit_plots/primary_step*.png`, not committed) and
inspected directly:

- **Removal** is concentrated in a **ring around the particle's entire
  exposed free-surface perimeter** (the whole `e2` boundary, both near and
  far from the neck), not preferentially at the neck -- the neck-exclusion
  factor (`incl`) does visibly thin the ring slightly right at the TJ
  locations, but most of the ring is elsewhere on the particle's cap.
- **Addition** is concentrated in a **vertical stripe spanning the entire
  substrate wall face** (`e1`'s own free surface, the full `y`-extent of the
  domain), not localized to the contact region either -- the neck occupies
  only a small fraction of that stripe's total length.

Quantitatively (near-neck fractions, `frac_within_NW` = fraction of the
field's mass within `N` interface widths of the nearer TJ):

| step | frac removed <=1W | <=2W | <=3W | <=5W | frac added <=1W | <=2W | <=3W | <=5W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.40% | 6.28% | 13.34% | 30.25% | 1.62% | 5.37% | 10.32% | 23.31% |
| 92 | 1.40% | 6.29% | 13.34% | 30.26% | 1.63% | 5.37% | 10.33% | 23.31% |
| 184 | 1.47% | 6.45% | 13.59% | 30.41% | 1.82% | 5.82% | 10.99% | 24.35% |
| 276 | 1.47% | 6.46% | 13.60% | 30.41% | 1.82% | 5.83% | 10.99% | 24.36% |

**~70% of removed mass and ~76% of added mass is more than 5 interface
widths from either TJ**, essentially unchanged over this trajectory.
Total removed = total added = `8.682e-4` (raw eta units) exactly (mass
conservation residual `0.0`), confirming O0's transfer is closed as
implemented.

## E. O0/O1/O2 mechanism comparison

All three variants run for the **same 276 steps** (O0's step count to reach
`|dV2|/V20=3e-4`), from the identical starting state:

| variant | d_x_neck (nm) | d_A_GB (nm^2) | V2/V20 | external/artificial mass | sigma (MPa) |
|---|---:|---:|---:|---:|---:|
| **O0** normal | **+0.0188** | +5.262 | 0.999699 | -- | 49.652 |
| **O1** removal-only | **+0.0136** | +5.115 | 0.999699 (same as O0) | reservoir += 0.2396 | 49.656 |
| **O2** addition-only | **+0.0223** | +5.360 | 1.000000 (e2 untouched) | artificial += 0.2396 | 49.649 |

This is **Outcome B from the handoff's decision tree, not Outcome A**: `O1`
(no deposition anywhere -- removed material tracked to an external
bookkeeping reservoir instead) **still widens the neck**, at 73% of O0's
full magnitude. `O2` (deposition with no corresponding removal) widens even
*more* than O0 (119%) in isolation. **Neither side of the transfer is
individually necessary or solely responsible** -- both independently push
in the same (widening) direction, and O0's combined result sits between
them (`0.0188`, close to but not exactly the mean of `0.0136` and `0.0223`,
consistent with a mostly-independent, mildly sub-additive combination). This
directly answers Q3 below: it is **not primarily a removal-topology or a
destination-topology problem** -- particle shrinkage by itself, with no
local deposition at all, already reproduces most of the response.

## F. CH flux analysis

Sign convention: `J_tangent > 0` means flux directed **away from the TJ**,
into the bulk of that free-surface branch (matching `tj_force`'s own
"outward tangent" convention, verified in
`test_surface_flux_tangent_points_away_from_tj`) -- i.e. **positive means CH
is transporting solid material away from the neck**, not into it.

| state | top J_tangent | top J_normal | bottom J_tangent | bottom J_normal |
|---|---:|---:|---:|---:|
| initial | **+3.150e-9** | -3.633e-9 | **+1.270e-9** | +3.074e-9 |
| after 276 steps | **+2.289e-9** | -3.679e-9 | **+1.223e-9** | +3.080e-9 |

`J_tangent` is **positive at both TJs, at both sampled times** -- CH's
conserved flux consistently moves solid material *away* from the neck along
the free surface, never toward it. Combined with Section C's finding that
raw CH cannot change `x_neck` directly (it only reshapes `f`), the
mechanistic picture is: CH relaxes curvature by pulling material away from
the sharp near-TJ region toward the smoother far cap; this changes the
*local solid capacity* available to `(e1, e2)`, and it is the **subsequent
mass-preserving projection** (Section C, `post-CH projection`, 34.6% of the
total) that turns that capacity change into an actual `x_neck` increase, not
CH's flux "aiming at" the neck.

## G. Eta-relaxation contribution

Quantified directly in Section C: `eta_raw` alone contributes **43.2%** of
the total `x_neck` widening in the primary (wider) geometry -- the single
largest individual contributor there -- and **13.1%** in the baseline
geometry (Section I). Always positive (widening) in both geometries, never
negative, consistent with the Milestone-5 finding that faster GB/eta
relaxation *amplifies* (not suppresses) the widening response.

## H. Projection contribution

Also quantified directly in Section C: `post-CH projection` + `post-eta
projection` together contribute **47.8%** of total `x_neck` widening in the
primary geometry and **83.8%** in the baseline geometry (Section I) -- not
negligible in either case, and dominant in the baseline case. This is **not**
a mass-accounting artifact -- `V2` accounting is exact (Section C) -- it is
the mechanism by which `f`'s CH-driven capacity change (which cannot itself
move `x_neck`) and Ostwald's own transfer get translated into an actual
`(e1, e2)` geometry change. The projection is doing real geometric work here
and should not be assumed neutral just because it is mass-neutral.

## I. Baseline-geometry cross-check

Same rates (`coarsening=3x, surface=0.3x, eta=1x`), `initial_overlap=5nm`,
same target `|dV2|/V20=3e-4`, ledger table only (276 steps, closure exact):

| operator | d_x_neck (nm) | % of total | dA_GB (nm^2) | dsigma (MPa) | % of total sigma |
|---|---:|---:|---:|---:|---:|
| CH | +0.000000 | 0.0% | +0.0000 | -0.16703 | 71.98% |
| post-CH projection | +0.032945 | 56.3% | +2.9855 | -0.03674 | 15.83% |
| Ostwald | +0.001828 | 3.1% | +0.0409 | -0.00176 | 0.76% |
| eta relaxation (raw) | +0.007661 | 13.1% | +1.0832 | -0.00854 | 3.68% |
| post-eta projection | +0.016113 | 27.5% | +1.5130 | -0.01797 | 7.74% |
| **total** | **+0.058547** | 100% | **+5.6226** | **-0.23204** | 100% |

**Same qualitative mechanism, different quantitative split.** In both
geometries: `CH` direct is exactly `0`; `Ostwald` direct is the smallest
contributor (`9.1%` wider vs. `3.1%` baseline); the combination of
projection + eta relaxation dominates (`91%` wider vs. `97%` baseline). What
differs is *which* of `post-CH projection` and `eta_raw` is individually
larger -- `eta_raw` dominates in the wider geometry, `post-CH projection`
dominates (more strongly) in the baseline geometry. The baseline geometry
also shows a substantially larger total response (`0.0585nm` vs. `0.0188nm`
for the same `dV2/V20`), consistent with the Milestone-5 finding that the
narrower starting geometry produces a stronger widening response overall.

## J. Mechanistic conclusion

**Direct answers to the handoff's Q1-Q10:**

- **Q1** (largest single contributor): geometry-dependent. `eta relaxation`
  in the wider geometry (43.2%); `post-CH projection` in the baseline
  geometry (56.3%). Never `Ostwald` itself in either case.
- **Q2** (does Ostwald directly widen the neck in its own substep?): yes,
  but only modestly (`9.1%` wider / `3.1%` baseline of the total).
- **Q3** (removal or destination topology?): **neither alone** -- O1
  (removal-only, Section E) reproduces 73% of the full response with *zero*
  deposition anywhere; O2 (addition-only) reproduces 119% with *zero*
  removal. Both independently widen; the transfer's spatial topology is not
  the primary lever.
- **Q4** (does CH transport material toward the neck afterward?): **no** --
  measured tangential flux is directed *away* from both TJs, consistently
  (Section F).
- **Q5** (sign of CH tangential flux near each TJ): **positive** (away from
  the TJ) at both TJs, both initially and after the trajectory.
- **Q6** (does eta/GB relaxation independently widen the contact?): **yes**,
  substantially and always in the widening direction (43.2% wider / 13.1%
  baseline).
- **Q7** (how much geometry change from the mass-preserving projection?):
  **substantial, not negligible** -- 47.8% (wider) to 83.8% (baseline) of
  total `x_neck` change, while remaining exactly mass-neutral for `V2`.
- **Q8** (same mechanism at both overlaps?): **qualitatively yes**
  (Ostwald direct always smallest; CH direct always exactly zero; the
  projection+eta combination always dominates), **quantitatively no** (which
  of `post-CH projection` vs. `eta_raw` is individually larger flips between
  the two geometries).
- **Q9** (Ostwald topology / CH capillary response / eta relaxation /
  projection artifact / combination?): **a combination, with Ostwald
  reservoir topology the *least* implicated of the four.** The removal and
  destination topologies each independently produce widening when isolated
  (Section E), so neither is a clean "smoking gun"; direct Ostwald transfer
  is consistently the smallest ledger contributor in both geometries
  (Section C, I); the dominant, geometry-dependent contributors are CH's
  curvature-driven capacity change (realized through the post-CH
  mass-preserving projection) and eta/GB structural relaxation. The
  projection itself is not an "artifact" in the sense of leaking mass (`V2`
  accounting is exact throughout), but it is where a large fraction of the
  actual geometric widening signal is realized, and should not be assumed
  geometrically neutral going forward.
- **Q10** (smallest physically motivated correction, not implemented): given
  that neither Ostwald's removal nor its destination topology is solely
  responsible (Q3), and that the dominant, geometry-dependent contributors
  are CH's capacity-driven projection response and eta/GB relaxation, a
  change to *where* Ostwald deposits material (e.g. an external reservoir,
  Section 24 of the handoff) would likely reduce but is **not obviously
  sufficient by itself** to reverse the sign -- O1 alone already reproduces
  73% of the full widening with zero deposition. The more promising
  direction suggested by this audit is to examine the CH capillary/curvature
  response and the eta/GB structural-relaxation kinetics themselves (the
  operators actually responsible for the bulk of the signal), rather than
  the Ostwald transfer's spatial pattern. This is a recommendation for
  review, not a plan being executed.

## K. Recommended next step (not implemented)

Given the mechanism is dominated by CH/eta response rather than Ostwald
topology, the most information-dense next diagnostic (still not a
production-physics change) would be to isolate the CH-only and eta-only
mechanism the same way Section E isolated Ostwald: run "post-CH-projection
only" and "eta+post-eta-projection only" variants from the same checkpoint
(analogous to Section 18 of the handoff's one-step isolation, extended to
match Section E's full-trajectory comparison) to see whether either, in
isolation from Ostwald entirely, still produces the same widening sign and
similar magnitude split. That would show whether the mechanism is really
"coarsening removes volume, and *any* subsequent capillary/structural
relaxation response widens the neck regardless of source," rather than
something specific to the Ostwald-shaped perturbation used here. This is
suggested for review, not started.

---

**Stopping here for review**, as instructed. No production Ostwald, CH, or
eta physics was modified; O1/O2 remain clearly-labeled diagnostic-only code
paths, not proposed replacements.
