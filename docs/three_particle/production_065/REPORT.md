# Selected 0.65 qualification — reload and one-b mechanics passed; descendants pending

The selected 0.65 no-sink reload passes 30 physical seconds. The forced LEFT
one-b mechanical test now completes in 0.121635 s, with chain strain +0.009925%,
LEFT stress 17.151 -> 5.263 MPa, RIGHT stress 17.906 MPa, relative material error
-6.07e-14, unchanged field bounds and a resolved 32.769 nm center span.
Center volume grows 0.02942% during this event and tracked PF energy rises
0.14036%; neither should be conflated with the shrinking, energy-decreasing
no-sink reload. See [completed-event figure](forced_completed_event.pdf).

The checkpoint-hashed [one-b review](one_b_review.json) qualifies this single
mechanical event for the prescribed conditional descendant study. The retained
q=0.9 within-event warning is a restricted-domain maximum, not an observed
unmasked local peak at the mask edge. The original three-completed-event monitor
remains active. Conditional descendants are running with seed 20260911;
**genuine stochastic Phase B remains DISABLED** until their qualification.
The predeclared genuine-root seed 20260910 has not been drawn.


Rc/Ro = 0.65, Ro = 119.999 nm, W = 4 nm, dr = 0.5 nm and the aligned
dz are fixed. The original CMC mapping, 200-step cleanup, surface and GB
energies, passive mobility and temperature are retained. Broad screening stops.
The earlier probability-before-field-guard criterion is superseded. With two
rates near 0.06 /s and independent 1.25 Exp(1) root thresholds, the characteristic
first-root time is 1.25/(Gamma_LEFT + Gamma_RIGHT), about 10.4 s. An 8.35%
probability by 0.864 s does not disqualify this shrinking-center geometry.

## Mathematical field-bound audit

The implemented energy is
E = integral [W_f f^2(1-f)^2/2 + k_f |grad f|^2/2 + f g_GB(phi)] dV,
with W_f = 12 gamma_s/W and k_f = 3 gamma_s W. Its chemical potential is
W_f f(1-f)(1-2f) - k_f Laplacian(f) + g_GB. This is a finite polynomial
in f, not an obstacle or logarithmic potential for f. Ownership phi has its
own obstacle construction; that does not turn the polynomial f energy into
an obstacle energy.

However, the evolution has mobility proportional to f^2(1-f)^2 and a
regularized tangential projector. Its double zeros preserve pure endpoints
in the formal smooth continuum equation. Therefore **finite energy outside
[0,1] alone does not justify admitting overshoot**. A rigorous invariant-region
theorem for this exact projected operator has not been established here.
The live bicrystal transfer capacities also assume bounded f, and its separate
ConservativeBoundedPhaseProjector clips and redistributes. That projector is
not used in this three-particle repair.

The actual discrete defect is identifiable: native face mobility evaluates
q at the arithmetic mean of neighboring fields. It can transport material
into a cell already at f=1. The late-state boundary vector-field probe sets
only the sampled peak to exactly 1 in an off-trajectory diagnostic copy:
the old RHS there is +2.084319180110499e-6 per model-time unit; the harmonic
cell-mobility RHS is zero. This probe is not an accepted trajectory or a
clipping operation. See bound_quadrature_audit.json.

The new optional HarmonicSurfaceDiffusion uses the harmonic mean of cell
mobilities on each face, retaining the original tangent projection, energy,
physical M_s, conservative divergence and current-field implicit solver.
The same face rule is used in both the linear solve and conservative update.
This is a discretization change, not a kinetic or field-guard change. Zero
mobility at a pure endpoint blocks its face flux. The original native/default
implicit path and old failed histories remain available unchanged.

A fixed-geometry native overlap at dr = 0.5, 0.375 and 0.25 nm gives old/new
field infinity differences 9.10e-7, 5.39e-7 and 2.58e-7; local stress differences
are -37.96, -19.18 and -6.96 Pa. This is consistent with a vanishing spatial
quadrature difference; it does not alone qualify a long implicit trajectory.
See mobility_spatial_refinement.json. The long no-sink test retains the original
1e-8 guard, mass and topology gates and embedded timestep error control.

For mathematical context, Elliott and Garcke's degenerate-mobility analysis
[SIAM 1996](https://epubs.siam.org/doi/10.1137/S0036141094267662) concerns bounded
weak solutions of a related equation. It is not invoked as a theorem for this
regularized tangential operator.

## Three-grain event extension

The existing current_state_mass_transfer_event accepts an optional transfer
callback; its binary default, current-state masks, physical transport clock,
step rejection, rollback and exactly-once progression are retained. LEFT uses
L/C and RIGHT uses C/R. Pair source masks are weighted by the active material
fraction when the inactive grain is present, so a broad donor mask cannot
remove inactive material. Its eta field is bitwise unchanged by the source;
active eta fields close pointwise to f minus inactive eta. With absent inactive
material, the adapter calls the original binary transfer directly and matches
bitwise. Fast surface evolution may change inactive eta through f; the pair
ownership update holds inactive phi fixed and reduces to the binary kernel.

The forced LEFT trajectory from the reproducible post-transient state is
labelled FORCED_EVENT_FOR_MECHANICAL_QUALIFICATION, never stochastic nucleation.
It must complete q/b = 1, conserve material, preserve topology and produce
positive chain strain and a LEFT local stress drop before it passes. Both
contacts and curvature are measured independently afterward. Descendant and
full renewal qualification remain pending this result. No barrier, attempt
frequency, site count, formation penalty or threshold has been changed.

The center curvature watcher uses the complete center branch between contacts,
excluding 3W near both TJs. The earlier half-center window has no remaining
samples after its exclusions at this 8.2W thickness. The production watcher
formulas are retained; only the three-particle branch support is extended.

Phase B: DISABLED pending qualification. No clipping. No fitted correction.
The 0.70 case and all old 1e-8 failures remain preserved. A scalar instantaneous
center-volume flux cancellation does not imply a stationary full PF field.

## Completed 30-second reload

The events-off trajectory reaches 30.000 physical seconds in 483 recorded states.
Center volume decreases 0.9856407%; max f remains exactly 1.0 and min f remains
at inherited -1.37488e-108 roundoff. Maximum relative mass error is 1.11e-15,
maximum mirror error 1.41e-10, and every recorded free-energy increment is
negative. Sparse topology checks pass. The instantaneous two-contact first-root
timescale is 9.5996 seconds at the end. Integrated hazards are 1.93757 per
contact; these are deterministic quadratures, not sampled or localized roots.
See [reload figure](reload.pdf), reload_summary.json and the complete harmonic
history in ../production_screen/harmonic_continuation_0.65_119.999.json.

Independent late-field refinement starts at 9.7977 s and advances 0.04 s with
4, 8 and 16 fixed steps. Differences to the finest field are 9.91e-9 and
2.42e-9; LEFT stress differences are -0.0621 and -0.00739 Pa. Both refinements
retain max f = 1.0 and conserve mass to floating-point precision. The selected
reload is qualified for its several-to-10-second loading interval, with no
claim of a general positivity theorem or arbitrary long-time stability.

All 60 focused three-particle and relevant binary regressions passed at this
milestone. Hash verification of 97 historical run files found no changes.

## Renewal semantics and implementation status

The two-contact clock core is tested for full-field crossing bisection, trial
rollback, RIGHT-first activation and exact RNG restart. Both pristine root
clocks freeze while a serialized avalanche source is alive. At extinction,
only the activated contact resets hazard and draws another 1.25 Exp(1)
threshold; the unselected contact retains its threshold and accrued hazard.
An exact numerical tie has deterministic contact ordering. No alternation,
simultaneous events, or threshold selection is imposed.

The gated renewal driver is implemented but not yet qualified end to end.
It uses the existing descendant controller with 1.5 eV lowering, 0.70 retention
and a 9 ms source window. Descendant hazards freeze during each complete 1b
transit and resume only after completion, as in the live bicrystal. Source-window
PF evolution retains the active-pair ownership flow; after extinction the new
ownership is pinned. Production seed 20260910 was declared before any production
threshold draw. The launch guard was exercised and correctly refused to start.
Single-event mechanical qualification has since passed; descendant and complete renewal qualification remain pending.

## First mechanical stop and continuing numerical audit

The original 60-block fast-relaxation budget stops the forced event at
q/b = 0.4075 (0.0427644 physical seconds). LEFT stress drops from 17.1508 to
13.0855 MPa and chain strain is +4.61587e-5. This is **not** a completed one-b
qualification. Material error is -9.88e-15 and topology remains valid, but
independently reconstructed ownership develops 1.68754e-14 pointwise closure
error, above the 5e-15 closure target. The unmodified stopped output and its
[figure](forced_event.pdf) are preserved.

A discarded copy of the rejected minimum increment meets exactly the original
fast convergence tolerances at block 61. This identifies an iteration-budget
limit, not demonstrated physical nonconvergence. Reconstructing one active
ownership as the complement of the other active and inactive fractions prevents
roundoff drift, matching the binary simplex construction. f is untouched and
inactive eta remains unchanged by the source. A 2,000-step roundtrip test passes;
the actual rejected increment now closes to 1.11e-16, still converges at block
61, and changes local stress by only -1.12e-7 Pa relative to the original audit.

The same event continues from its exact checkpoint with a 240-block computational
ceiling and a finer 0.005b maximum increment. Convergence tolerances, physical
coefficients, field guard, source construction and accumulated physical event
clock are retained. The finer ceiling reduces rejected doubled trials; near q/b=0.8 the adaptive solver accepts 0.0025b and still rejects attempted 0.005b trials.
See continuation_chain.json and the completed checkpoint-hashed one-b review.

Actual-field reflection of an accepted 0.0025b increment agrees to 3.3e-15 in
material fields and 2.1e-14 relatively in its physical clock. Active-source PF
window evolution agrees with 100 native steps to 1.39e-9 in fields and 0.70 Pa
in LEFT stress. The three-grain topology interval now uses both actual boundary
coordinates, including asymmetric states; its translation-invariance test passes.
Curvature monitoring can use the two measured field TJ positions.

The fast implicit-flow prototype fails its native event overlap and is not
promoted. A separate storage-only native optimization preserves block count,
fields to 3.9e-15 and the physical clock to 3.3e-14; the ongoing reference event
still uses the original native path. The four-thread test is slower than one
thread and is not selected. All these are numerical tests on discarded copies.

## Distinguish reload and event energetics/volume changes

At the stopped q/b=0.4075 checkpoint the center gains 0.00865% volume during the
event, despite positive chain strain. Its tracked PF energy increases 0.03456%.
These observations must not be conflated with the energy-decreasing,
center-shrinking no-sink reload. The corresponding archived *qualified bicrystal*
current-state events also increase tracked PF energy, by about 0.00051–0.00169%.
See bicrystal_event_energy_audit.json with original source hashes. The prescribed
current-state event source is not a guaranteed descent direction of the tracked
PF functional. No new activation work or barrier correction is introduced to
change that observation.

The requested volume-loss/strain correlation remains a question to test across
the complete renewal trajectory, not a conclusion inferred from a partial
forced stress drop. Descendant qualification and genuine stochastic production
remain gated on a completed, audited one-b event.

At the follow-up audit milestone, 62 focused regressions pass. The forced event
has passed q/b=0.8 but is still running. Both descendants and genuine Phase B
remain disabled; the descendant launch guard was exercised and refused entry.
No production random thresholds have been drawn.

The late increment audit at q/b=0.8 advances 0.0025b using one, two and four
subincrements, with the original fast convergence tolerances. All complete.
Field differences to the finest result decrease from 2.44e-6 to 1.35e-6;
local-stress differences decrease from 1.85 kPa to 0.521 kPa. The coarse
physical-clock difference is 0.00126%. This is a local refinement result,
not a full-one-b convergence claim. See event_increment_refinement.json.

## Late-event morphology review and source-window audit

At q/b=0.9 the LEFT center-side watch raises a **within-event analogue** of
its progressive-growth flag: gradients at q/b=0.7, 0.8 and 0.9 are
6.54, 10.31 and 12.83e14/m2 and their restricted maxima lie near 3W.
This is not the established three-completed-event test. The unmasked profile
contains no local absolute-gradient peak within 0.75W of 3W; the flagged maximum
is the first sample outside the TJ exclusion. Thus the flag alone cannot
separate a transfer-mask artifact from a broad TJ-core tail in this short,
two-contact center. The warning is retained, not automatically cleared.
See forced_curvature_profiles.pdf and mask_edge_detail.pdf/json.

A discarded q/b=0.8 field evolved for a full 9 ms without another source
increment raises LEFT stress from 7.469 to 9.259 MPa. Its chain strain recovers
by 1.9163e-5, center volume increases 0.01030%, and tracked PF energy decreases
0.02850%. This is an off-trajectory numerical window audit, not a completed
root/descendant or a sampled avalanche. It demonstrates why immediate-event
strain and subsequent window relaxation must be recorded separately.

Optional small-step preconditioner reuse retains the full current-field
linear operator, the original residual limits and conservative flux update.
Its two new equation/conservation regressions pass; the complete focused suite
now has 64 passing tests. The full 9 ms cached result agrees with its uncached reference to 1.61e-15 in fields.
The optional cache is enabled only in the still-gated descendant/renewal
source-window drivers; solver defaults and the ongoing native event are unchanged.

The cached implicit-fast prototype still stops at its timestep floor. A variant
with 50 native startup steps completes its test increment, but differs from
the native reference by 2.99e-5 in fields, exceeding the previously declared
2e-5 overlap limit. Neither prototype is promoted. Their original negative
results, inputs and outputs are preserved separately.

The full 9 ms window partition audit completes with one, two and four outer
partitions. Fields agree within 4.22e-6 and LEFT stress within 62 Pa; these
are adaptive solves at the same error tolerances, not a fixed-step order test.
All retain the unchanged field guard and roundoff-level conservation/closure.

## Completed native performance and clock audits

The buffered native implementation preserves the exact stencil formulas and
checks after each ten-step block. At the retained q/b=0.8 increment it agrees
with native fields to 3.14e-15, has identical local stress and the same 126
blocks, and agrees in physical clock to 3.38e-14 relatively. Its nontrivial
ownership-flow regression preserves the inactive fraction exactly. It is
selected for the conditional descendant study; no physical coefficient or
convergence tolerance changes.

Before the first descendant draw, the launch audit aligned the new source
window with the existing bicrystal's 1e-12 s crossing tolerance and 0.025
model-time hazard subinterval. Root crossing keeps its separate original
1e-4 model-time tolerance. A microsecond-crossing regression passes; all 66
focused tests pass. The first conditional descendant threshold is drawn once
from the predeclared seed and is not selected by its firing time.

## Root quadrature and analysis readiness

A separate one-second current-field audit from the predeclared post-transient
checkpoint compares 0.2, 0.1 and 0.05 s root-hazard intervals without drawing
random thresholds. LEFT integrated hazards are 0.0632673502, 0.0632682297
and 0.0632684422. Relative differences to the finest are 1.73e-5 and
3.36e-6, below the predeclared 1e-4 limit. Fields differ by at most 1.20e-5,
below the declared 2e-5 overlap limit. These adaptive field differences do
not establish a monotone temporal order. The 0.2 s root quadrature is now
qualified; full-field crossing localization retains its original tolerance.
See root_quadrature_refinement.json and its retained NPZ fields.

The analysis driver separates passive reload, active one-b transit and
facilitated source-window relaxation. It reports interval rates and complete
event changes in center volume and chain strain, with explicit incomplete
event labels and no independent-sample interpretation of serial correlations.
Local activation, center mean, cluster, integral and Cannon–Carter diagnostics
are retained separately. The conditional worker launched before the reporting
update uses Cannon–Carter fits about its original pinned planes; its plots
label that convention. The prospective genuine driver uses measured current
contact positions for those diagnostic fits. Neither changes activation.

This reporting and quadrature milestone does not enable Phase B. The
conditional descendant family is still in progress; no completed descendant
or genuine stochastic-root result is claimed here.

## Continuation batching and quota endpoint

A discarded-copy audit at forced q/b=0.8 raises only the computational
iteration ceiling to 1024 and compares 0.01b, 0.005b and 0.0025b increments
for a total 0.01b advance. The 0.01b trial is rejected by the unchanged
acceptance checks; its returned input is not a valid overlap comparison.
The 0.005b result agrees with 0.0025b within 8.19e-6 in fields, 5.47 kPa
in local stress and 6.47e-5 relatively in clock. It uses 499 rather than
504 native blocks and offers no material speed advantage. The selected
0.0025b increment, 240-block ceiling and all convergence tolerances remain
unchanged. See event_batch_audit.json.

A separate regression reproduces a quota-roundoff bug: 400 additions of
0.0025b can leave q infinitesimally below one and falsely hit the 400-state
cap. The final trial endpoint now includes the accumulated floating-point
remainder before source and clock construction. This transfers the full
remaining quota; it neither clips a field nor declares untransferred material
complete. All 66 focused regressions pass, including exactly-once restarts
and the new exact 400-increment endpoint assertions.

The active conditional descendant was intentionally checkpointed at
q/b=0.2025, time since the forced-root start 0.1529799409 s, and resumed with
this fix. Its accepted fields, base fields, adaptive continuation state,
physical event clock, source controller, hazard, threshold and RNG state
are restored. Immutable pre-restart files and hashes are retained in
forced_descendant_family/restart_before_quota_fix. No accepted source is
replayed and no threshold is redrawn. This is a computational restart,
not a physical failure or a second descendant.

## First descendant: quarter-b warning and local timestep audit

The first conditional descendant reaches q/b=0.25 with LEFT stress
3.094 MPa. Its unmasked center profile has a new gradient maximum at 3.418W,
within 0.75W of the 3W transfer boundary. Its magnitude is 1.736e15 /m2
and prominence 7.32e13 /m2, about 4.2% of that magnitude. This shallow local
peak was absent at the descendant's initial state. It is retained as an
additional warning; the first descendant is incomplete and repeated-event
morphology is not qualified. The established three-completed-event monitor
is unchanged. Immutable plots and profiles are in descendant_quarter_b_review.

A discarded 0.0025b continuation increment from that retained q/b=0.25
state compares native dt, dt/2 and dt/4 while using 10, 20 and 40 steps
per convergence-check block. Thus the physical check cadence and all
convergence tolerances are fixed. All three stop at 210 blocks with
unchanged field bounds. Nominal/fine and half/fine field differences are
2.79e-9 and 9.30e-10; local-stress differences are 1.43 and 0.475 Pa,
and relative event-clock differences are 1.13e-8 and 3.77e-9. This is strong
local timestep agreement. It does not test refinement of the morphology
already accumulated before the checkpoint, nor qualify a complete event.
See native_timestep_audit.json and the retained input/output fields.

The active-restart precision audit additionally finds a one-ULP
(1.39e-17 s) global event-origin offset from subtracting the saved local
event time from the global checkpoint time. The local event clock, RNG
and prior history are preserved, but the reconstructed global origin is
not bitwise identical. Future active restores use the unique saved source
crossing timestamp directly and validate it against the checkpoint clock.
The existing offset is far below the unchanged 1e-12 s crossing tolerance.

## First descendant: retained 240-block failure and audited continuation

At q/b=0.4225 the next 0.0025b trial fails the 240-block fast-relaxation
ceiling and rolls back. Its rejection reason is fast_manifold; no field,
mass, ownership or topology guard fails. The complete stopped history,
controller/RNG and restart fields are retained under
forced_descendant_family/fast_budget_240_stop.

A discarded-copy retry from that exact failed checkpoint converges at
block 241 with every original convergence tolerance unchanged. The audit
then intentionally hits a one-accepted-increment cap at q/b=0.425; that
cap is not a failed trial or a completed one-b event. The production
continuation ceiling is raised to 512 blocks, which does not affect a
trial that already converges earlier. The 0.0025b increment, field guard,
physical coefficients, source law and stochastic parameters are unchanged.
The live event resumes from the rolled-back checkpoint, not from the
discarded audit output. Its original threshold and source crossing remain
in force; the failed trial and rejection count remain in provenance.
Phase B is still disabled. See descendant_fast_budget_audit.json.

The live retry with the 512-block ceiling reaches q/b=0.425 and matches
the discarded 1024-ceiling audit bit-for-bit in fields and local event
clock. The original controller and RNG state are unchanged, and the
fast_manifold rejection remains recorded. The continuation uses the saved
source crossing directly for its global clock origin. See
descendant_fast_budget_live_overlap.json.

## Half-b conditional morphology

At q/b=0.5 of the first descendant, LEFT stress is 1.447 MPa. The
center-side unmasked gradient peak outside the 3W exclusion moves from
3.418W to 3.543W. Its gradient magnitude rises from 1.736e15 to
1.981e15 /m2 and its prominence from 7.32e13 to 2.61e14 /m2
(4.2% to 13.2% of the peak magnitude). Another local peak at 2.426W
lies inside the TJ exclusion; it is recorded separately rather than
conflated with the feature outside 3W.

Applying the original growth rule to the three within-event snapshots
q=0, 0.25 and 0.5 does not raise a flag. This remains an additional
within-event comparison, not the required three-completed-event test.
The increasingly distinct outside-3W peak remains a warning, and the
descendant and Phase B remain unqualified. Immutable profiles and the
phase-resolved analysis snapshot are in descendant_half_b_review. The
analysis snapshot extends slightly past the exactly retained half-b field
and labels the descendant incomplete.

## Fixed-q stopping-tolerance sensitivity

A discarded copy of the half-b field is relaxed at fixed quota with
nominal, half and quarter fast-convergence tolerances. The nominal test
stops after five blocks, changing fields by 7.48e-6. Both tighter tests
reach the 2048-block cap without convergence. They therefore follow the
same native trajectory to the same final field; their agreement is not
evidence of tolerance convergence.

That capped trajectory represents 0.389 ms of additional native relaxation,
with zero added event quota and no advancement of the transport-event or
stochastic clocks. Fields change by 2.72e-3, LEFT stress by +40.4 kPa,
PF energy by -0.00297%, and center volume by +0.000832%. Field bounds and
roundoff-level conservation/ownership closure remain sound. The outside-3W
peak persists, moving from 3.543W to 3.667W; its prominence changes from
2.61e14 to 2.52e14 /m2. See fixed_q_relaxation_audit.json/pdf and its NPZ.

Convergence under tighter stopping tolerances is not demonstrated. The
nominal stopping test is not a proof that the full PF vector field vanishes.
These copies are not accepted states, and neither live tolerances nor
qualification gates change. The stationary-null result remains a separate
validation question, not a renewed production prerequisite.
