# Constrained relaxation at fixed (V2, L): G*, F_L, and the V2 sweep

Scope: the constrained-relaxation benchmark and (conditionally) the constraint-release
test, per the updated handoff. **Stopping here for review.** The constraint-release
test was **not** run — see Section H for why.

## A. What was preserved / what is new

- The direct TJ capillary-vector construction (`tj_force.py`) and signed-curvature
  audit (`signed_curvature.py`) are unchanged and reused as-is — no `psi_eq` was
  reintroduced anywhere.
- The experimental static-neck-geometry family (`static_neck_geometry.py`) was **not**
  further refined. It is now explicitly labeled EXPERIMENTAL/DIAGNOSTIC in its
  docstring and used only as a (already-precise) initial-condition generator for the
  new constrained relaxation — never as a constraint during evolution, and x0 from it
  is not treated as physical (per the prior report's own conclusion).
- New this session: `pf_sintering/separation_constraint.py` (the explicit L/V2
  corrector), `pf_sintering/constrained_relaxation.py` (the relaxation driver),
  `scripts/constrained_G_star_benchmark.py` (the benchmark), and this report.
- The work was checkpointed on `codex/coarsening-stress-buildup` in 5 commits
  *before* this new work began (Milestone 1/2, TJ-force diagnostic + tests,
  signed-curvature audit, static-neck family + tests, force-map benchmark + report),
  exactly as requested. This session's new files are not yet committed — left for
  review alongside this report.

## B. Tests and status

30/30 passing (`pytest ./PF-Sintering/tests -q`): the 26 from before, plus 4 new in
`tests/test_constrained_relaxation.py` covering the L-corrector (no-op when already
within tolerance; corrects an induced offset; holds both V2 and L over repeated
calls) and a short (200-step) constrained relaxation (finite diagnostics, both
constraints held tightly). No existing test was modified.

## C. The explicit L/V2 constraint (as requested: verify to a *stated* tolerance)

`separation_constraint.enforce_V2_and_L` alternates two corrections until both
converge or a small iteration cap is hit:

1. **L**: a pure geometric correction (`enforce_separation`) that reuses exactly the
   same conservative upwind transport stencil `model.rbm()` uses (confined to the
   e2-dominant region via the same `e2/(e1+e2+e3)` weighting), applied as small
   corrective substeps with the centroid re-measured after each one. It carries **no**
   Sink/event bookkeeping and is never called from production code.
2. **V2**: the existing, already-qualified `structural_projection.project_eta_mass_preserving`.

Neither alone leaves the other exactly satisfied: the L-correcting advection leaks a
small amount of grain mass (clipping in the upwind scheme), and the mass-preserving
projection's local redistribution can itself shift the e2 centroid slightly. They are
alternated until both are within tolerance.

**A real bug was found and fixed while building this**: an early version re-pinned V2
to the *external*, session-start target on every single call, unconditionally. Over
thousands of steps this measurably **raised** `G_interface` (a slow but real energy
increase from otherwise-unnecessary redistribution on steps where nothing had actually
drifted) — caught by comparing `G_interface` at a mid-run checkpoint against a later
one during convergence-trend inspection. Fixed by reprojecting to the masses measured
*immediately before* each `enforce_separation` call (self-referential, matching how
the main relaxation loop's own CH/AC-step projections already work) rather than to a
fixed external target on every call.

**Verified tolerance** (not merely "approximately fixed"): across the primary
relaxation and every warm-started sub-relaxation in the full benchmark run (8000 +
2x2x4000 + 3x3x4000 = 44,000 total constrained steps), the final residuals were:

| run | \|dL\| final (nm) | \|dV2/V2\| final |
|---|---:|---:|
| primary state (seed 30nm) | 3.44e-4 | 2.10e-14 |
| stationarity-check state (seed 45nm) | 3.77e-4 | 1.77e-14 |
| F_L, dL=+1nm / -1nm sub-relaxations | 1.82e-4 / 3.78e-4 | (same order) |
| F_L, dL=+2nm / -2nm sub-relaxations | 3.34e-4 / 2.00e-4 | (same order) |

`L` is held to `~3-4e-4 nm` out of `100 nm` (relative `~3-4e-6`) and `V2` to
`~1-2e-14` relative, in every run, regardless of how far the underlying shape
relaxation itself has converged (Section D). The constraint machinery itself is
working correctly and precisely; the open problem is entirely in the *relaxation*
converging, not in holding the constraints.

## D. Single relaxed state and stationarity — **the verification criterion fails**

Reference: `dev` preset, `R2=80nm`, `aspect_ratio=2.0`, `V2 = V20 = pi*R2^2 =
2.011e-14`, `L0 = 100nm`. Practical step budget: 8000 steps for the primary/
stationarity states (chosen after a convergence probe described in Section E).

Two independent starting seeds (`x_neck_seed = 30nm` and `45nm`, both from the
experimental static-family generator, used only as initial guesses) were relaxed for
the *same* 8000-step budget:

| | seed x_neck=30nm | seed x_neck=45nm |
|---|---:|---:|
| G* (J) | 1.728866e-06 | 1.700039e-06 |
| x_neck* (nm) | 29.91 | 40.84 |
| psi_top* / psi_bottom* (deg) | 54.78 / 38.02 | 73.66 / 76.67 |
| \|F_TJ_top\| / \|F_TJ_bottom\| | 0.778 / 1.008 | 1.022 / 0.680 |

**These do not agree.** G* differs by 1.7% (not noise), x_neck* by 11nm (37%
relative), and the two states aren't even close to each other in shape (psi_top and
psi_bottom are nearly equal for seed 45 but very different for seed 30). **The
stationarity requirement — "G* is stationary with respect to unconstrained shape
perturbations" — is not met at this step budget.** Both `|F_TJ|` values also remain
large (comparable to the individual `xi` vector magnitudes, which are O(gamma_s) =
O(1) throughout) rather than small, so **the second verification requirement also
fails**: the two Young-Herring residuals are *not* small compared with the individual
interfacial vector magnitudes.

## E. Why: an intrinsic convergence-rate problem, investigated directly

A convergence probe (checkpointing G*, psi, F_TJ at steps 1000/3000/7000/15000) showed:
`G_interface` decreases substantially and monotonically through ~7000 steps, then
**increases again** by step 15000 (`1.7288e-6 -> 1.7313e-6`), while `x_neck` reverses
direction (`29.8nm -> 32.0nm`) over the same interval. A second probe, run with the
L/V2 corrector *removed entirely* (pure CH+structural relaxation, L allowed to drift
freely), showed the *same* substantial, monotonic G* decrease over the first 3000
steps and confirmed real, physical (uncorrected) L drift of order 1nm per 1000 steps —
i.e. the corrector is doing real, necessary work, and is not itself the primary source
of the longer-horizon non-monotonicity (though it cannot be fully ruled out as a
contributor without a much longer isolated run than time allowed here).

The likely root cause is a genuine numerical-stiffness property of the explicit
Cahn-Hilliard scheme, not a bug: explicit 4th-order-parabolic (CH) time-stepping
requires a number of stable steps that scales like `(system_size / reference_length)^4`.
The model's own mobility calibration (`M_f` in `build_params`) is referenced to a
20nm length scale; this particle's own radius (`Rx ~= 113nm`) is `~5.6x` larger,
implying a relaxation timescale roughly `5.6^4 ~= 980x` longer than the reference —
i.e. potentially requiring a step count far beyond what is practical to run
repeatedly (multiple states, each needing independent verification) within this
session, even though each individual 8000-step run itself only took `~20-25s`.

**Given this, `n_steps=8000` (primary/stationarity) and `n_steps=4000` (warm-started
sub-relaxations, which start closer to their own local trajectory) were used as a
practical, fixed budget for the rest of this benchmark, on the understanding — proven
by Section D above, not assumed — that these are *substantially relaxed but not
converged* states. Results below should be read with that caveat throughout.**

## F. F_L via centered finite difference

Computed from the `x_neck_seed=30nm` lineage (warm-started translations of the primary
state, ±dL, each re-relaxed for 4000 steps):

| dL | G*(L0+dL) | G*(L0-dL) | F_L = -[G*(L0+dL)-G*(L0-dL)]/(2 dL) |
|---:|---:|---:|---:|
| 1nm | 1.731139e-06 | 1.723526e-06 | **-3.807** |
| 2nm | 1.732192e-06 | 1.718026e-06 | **-3.541** |

Same sign, same order of magnitude at both `dL` (a ~7% difference between them,
plausibly within the noise expected given Section E's non-convergence, not a clean
Richardson-extrapolation-quality check). **Sign interpretation**: `F_L < 0` means
`dG*/dL > 0` at this state — G* increases with L — so the generalized force conjugate
to L already points toward **smaller L**, i.e. toward densification, even at
`V2 = V20`. This is a directionally sensible, non-degenerate result (not pinned at
zero, not a sign flip between the two `dL` values), computed with **no** hazard
reconnection and **no** kinetics tuning, exactly as instructed. It should still be
read subject to the non-convergence caveat in Section E.

## G. V2 sweep at fixed L0 — **the principal physics test fails, in every component**

All four states warm-started sequentially from the `x_neck_seed=30nm` lineage (each
`V2` step warm-started from the previous, 4000 relaxation steps each; `F_L` recomputed
at each `V2` with `dL=1nm` only, for cost):

| V2/V20 | G* (J) | x_neck* (nm) | A_GB* (m^2) | psi_top* (deg) | psi_bottom* (deg) | F_L | \|F_TJ_top\| | \|F_TJ_bottom\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 1.728866e-06 | 29.91 | 1.579e-15 | 54.78 | 38.02 | -3.807 | 0.778 | 1.008 |
| 0.995 | 1.729481e-06 | 31.03 | 1.638e-15 | 54.62 | 38.58 | -3.662 | 0.680 | 1.057 |
| 0.990 | 1.730375e-06 | 31.76 | 1.686e-15 | 55.04 | 34.60 | -3.556 | 0.773 | 0.931 |
| 0.980 | 1.731269e-06 | 32.17 | 1.718e-15 | 54.56 | 31.71 | -3.615 | 0.766 | 1.051 |

The handoff's principal physics test asks, at fixed L, whether V2 decreasing
coincides with **G* decreasing**, **x_neck* decreasing**, and **|F_L| increasing**.
**None of the three hold here**:

- **G* increases** monotonically as V2 shrinks (`1.7289e-6 -> 1.7313e-6`, +0.22% over
  a 2% V2 reduction) — the opposite of the required direction.
- **x_neck* increases (widens)** monotonically as V2 shrinks (`29.9nm -> 32.2nm`) —
  again the opposite direction, and this is now the **third independent method**
  (after the Milestone-1/2 sink-off coarsening runs and the Milestone-3 static
  force-map) to find neck-widening rather than neck-narrowing as V2 shrinks at
  (approximately) fixed separation. Three different constructions agreeing is
  meaningful, but see the caveat below.
- **|F_L| does not grow**: it moves `3.807 -> 3.662 -> 3.556 -> 3.615`, essentially
  flat to slightly *decreasing*, not increasing.

**This entire table must be read together with Section D's stationarity failure.**
Every state in this sweep is warm-started from (and therefore shares the relaxation
history/basin of) the same `x_neck_seed=30nm` trajectory, which Section D showed sits
in a measurably higher-energy, less-relaxed configuration than the `x_neck_seed=45nm`
trajectory reaches from the same budget. That means this table is **internally
consistent along one relaxation lineage**, but it has **not** been shown to represent
converged, path-independent values of a true state function `G*(V2, L)` — the
stationarity check that would establish that failed. The qualitative agreement with
the two earlier, methodologically distinct findings (neck widening, not narrowing) is
suggestive of a real model behavior worth taking seriously, but this specific
benchmark cannot rigorously confirm it is the *equilibrium* answer rather than an
artifact of an insufficiently relaxed, seed-dependent trajectory.

## H. Constraint-release test: not run

Per the handoff's own conditional instruction ("if the constrained-coarsening
sequence builds |F_L|, take one high-force state and..."), and since Section G found
**|F_L| does not build** with shrinking V2 (it is flat-to-decreasing across the tested
range), the constraint-release test was not performed. Running it from the
`V2/V20=0.980` state (the nominal "highest" state) would not test the intended
question, since that state does not represent a genuinely elevated densification
force relative to `V2/V20=1.000` — if anything `|F_L|` is smallest there among the
sweep with `dL=1nm`. Given Section D and E's findings, the more scientifically honest
next step is to resolve the underlying convergence problem first (Section I), not to
proceed to a constraint-release test on an unreliable state.

## I. Numerical pathology / ambiguity notes and recommended next step

1. **Primary finding**: within a practical (single-session) step budget, this
   explicit-time-stepping constrained relaxation does not converge tightly enough to
   treat `G*` as a well-defined function of `(V2, L)` — two different starting shapes
   at the *same* `(V2, L)` reach measurably different states after the same step
   budget (Section D), and the longer-horizon trend is not even reliably monotonic in
   the diagnostic proxy `G* = E_surf + E_gb` (Section E). This is very likely an
   intrinsic stiffness property of explicit 4th-order CH time-stepping at this
   particle-to-reference-length ratio, not a bug in the new constraint machinery
   (which was independently verified to hold both V2 and L to tight, stated tolerance
   throughout, Section C).
2. `E_surf + E_gb` is a total-variation/area-based *proxy* for interfacial energy
   (documented as such in `diagnostics.py`), not the exact CH/AC free-energy
   functional (which also includes the double-well and eta-f coupling terms). The
   true functional the dynamics gradient-descends could behave more cleanly than this
   proxy; the observed non-monotonicity in the proxy does not by itself prove the
   underlying dynamics is non-thermodynamic, but it does mean **the proxy is not a
   safe convergence/stopping criterion on its own** at long horizons — this benchmark
   used a fixed step count instead, precisely because of this finding.
3. **Recommended next step, not attempted here**: either (a) a much longer relaxation
   budget run outside an interactive session (the timescale estimate in Section E,
   `~1000x` the 20nm-reference relaxation time, suggests this could require very many
   more steps than were practical here), or (b) a fundamentally faster convergence
   method for this specific sub-problem — e.g. a semi-implicit/spectral CH solver, or
   direct energy minimization (L-BFGS/conjugate-gradient on the level-set/phase
   fields subject to the V2, L constraints) rather than explicit gradient-flow time
   stepping — since this benchmark only needs the *minimum-energy shape* at fixed
   (V2, L), not a physically time-accurate trajectory, an accelerated minimizer would
   answer the same question far more cheaply than continuing to extend the explicit
   time-stepping horizon.
4. The `separation_constraint.py` mass-leak-then-reproject bug (Section C) is a
   reminder that even small, well-intentioned "constraint enforcement" code can
   quietly perturb a conserved diagnostic if applied unconditionally rather than only
   when actually needed — worth keeping in mind if any future correction/constraint
   code is added elsewhere in this codebase.

---

**Stopping here.** The constrained-relaxation and F_L/V2-sweep benchmark is complete,
but its central physics conclusion is inconclusive rather than confirmatory: the
"principal physics test" requested (G* down, x_neck* down, |F_L| up as V2 shrinks at
fixed L) was **not** observed — instead all three moved in the opposite direction,
consistent with two earlier, independent findings but not yet established as a
converged result given the stationarity failure. No constraint-release test, no
stochastic nucleation, no rate sweeps, and no long trajectories were attempted.
