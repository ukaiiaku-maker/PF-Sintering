# Selected 0.65 production qualification — reload passed, event pending

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
Mechanical, descendant and complete renewal qualification are still pending.
