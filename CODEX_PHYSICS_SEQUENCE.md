# Codex physics sequence: coarsening-driven stress buildup and nucleation-limited densification

## Scientific target

The purpose of this campaign is to recover the experimentally observed sequence:

1. No active sink -> no rigid-body densification and the particle/GB separation is fixed.
2. Coarsening continues and the shrinking particle/contact geometry evolves toward a higher local sintering-stress state.
3. The neck/contact width and load-bearing GB region decrease while the total interfacial free energy can continue to fall.
4. When the local nucleation hazard fires, a finite sink becomes active.
5. Interface-normal transport produces rigid-body motion/densification, the neck/contact broadens, and the local sintering stress relaxes.
6. The sink exhausts its finite displacement/strain quota, becomes inactive, and stress accumulation can begin again.

This is a directional-physics campaign, not a long-time production campaign. Start from geometries near the anticipated critical state and measure the sign and relative magnitude of the short-time response.

## Important physical constraints

- Do **not** require the starting geometry to be Plateau-Rayleigh-like. With no active sink, the center-to-center/particle-substrate separation is fixed. Coarsening of the shrinking particle can drive the constrained geometry toward a PR/de-sintering-like state as particle size decreases without densification.
- Do **not** replace the current local neck/triple-junction stress calculation with `sintering potential / GB area`. The stress is nonuniform on the GB plane and the quantity of interest is the local stress state at the neck/triple junction. Keep or improve the more rigorous local capillary/Cahn-Hoffman/TJ construction.
- Preserve the newly qualified mass accounting. CH, structural relaxation, and RBM must not become secular grain-volume-change mechanisms. Explicit Ostwald/reservoir exchange is the permitted background grain-volume-change mechanism.
- Avoid long waiting-time simulations. Where necessary, create controlled benchmark states near the critical geometry or near event activation so the direction of the response is visible in short runs.
- Do not tune the stochastic hazard merely to force a visually desired trajectory. Geometry/stress generation and event response must be qualified independently before coupling them.

## Leading hypotheses to test

### H1. The current Ostwald/reservoir operator protects the neck too strongly

The existing substrate Ostwald operator suppresses source removal near the contact through a neck-exclusion factor. That may allow the particle to lose volume while retaining a disproportionately large contact, preventing the contact/neck from shrinking and the local stress from increasing.

Test whether the reservoir should prescribe only the **net grain-volume loss rate**, while the surface-diffusion/phase-field dynamics determine where the free surface recedes.

### H2. Solid-solid interface / structural relaxation is too fast

Even after making structural relaxation volume-neutral, a high GB/eta mobility can allow the contact and triple-junction geometry to continuously relax toward a low-stress state while the particle coarsens. This can suppress accumulation of geometric frustration and local stress.

Treat GB/eta mobility independently from free-surface transport and explicitly test the limit of very small or zero GB mobility during sink-off evolution.

### H3. Relative rates of free-surface transport, reservoir/Ostwald exchange, and GB relaxation select the response

Once H1/H2 are isolated, determine which dimensionless ordering gives stress accumulation. Do not start with a broad parameter sweep. Use short directional perturbations around one near-critical state.

## Milestone 0 — freeze current numerical invariants

Before changing physics:

- Keep all existing mass-preserving structural-projection tests passing.
- Add/retain checks that total solid `f` is conserved to numerical tolerance.
- In a no-event run, confirm that integrated grain-volume change equals the explicit Ostwald/reservoir contribution.
- In an RBM-only benchmark, require net grain-volume drift after constrained projection to be negligible.

Do not modify these invariants to make later benchmarks pass.

## Milestone 1 — add a directional state diagnostic

Create a compact diagnostic that reports, at every requested sample:

- time and step,
- shrinking-grain volume `V2/V20`,
- fixed particle-substrate or center-to-center separation,
- rigid-body displacement / densification strain,
- neck/contact width,
- GB/contact measure,
- local neck/triple-junction sintering stress,
- the individual terms entering that local stress calculation when available,
- local curvature(s), TJ/dihedral angle(s), and Cahn-Hoffman force contribution when anisotropy is active,
- total free-surface energy, GB energy, and total interfacial energy,
- active/inactive sink state, hazard, threshold, and quota progress.

Also report short-time directional derivatives/signs such as

`dV2`, `dL`, `dx_neck`, `dA_GB`, `dSigma_local`, `dG_interface`, `dstrain`.

The diagnostic should make it possible to classify a 10–1000 step test without relying on movies.

## Milestone 2 — sink-off directional benchmark

Purpose: determine whether coarsening alone drives the geometry in the required direction when densification is forbidden.

Use one substrate geometry already close enough to the anticipated critical regime that a small amount of coarsening produces a measurable directional response. Disable hazard and RBM completely. Keep particle-substrate separation fixed.

Run only until a small prescribed amount of grain-volume change has occurred, e.g. `|Delta V2|/V20 ~ 1e-4 to 1e-3`, or an equivalent short fixed horizon. Do not run to pinch-off.

Evaluate a minimal 2x2 mechanism matrix:

1. **Current reservoir localization + current GB mobility** (reference).
2. **Neck-unprotected / geometry-driven reservoir + current GB mobility**.
3. **Current reservoir localization + very low/zero GB mobility**.
4. **Neck-unprotected / geometry-driven reservoir + very low/zero GB mobility**.

For the neck-unprotected version, do not arbitrarily remove material at the neck. Prefer a reservoir formulation that sets the net volume loss / chemical-potential bias and lets the conserved surface-diffusion dynamics decide where the surface recedes.

### Required direction for the experimental mechanism

For at least one physically reasonable sink-off case, require simultaneously:

- `dV2 < 0` (coarsening/shrinkage),
- `dL ~ 0` (no RBM/densification),
- `dx_neck < 0` and/or decreasing load-bearing contact,
- `dSigma_local > 0`,
- `dG_interface < 0`.

The last two must be allowed to coexist: total free energy can decrease while the local stress/chemical-potential concentration increases.

If no case produces this direction, inspect the local stress construction and surface-transport thermodynamics before changing the hazard model.

## Milestone 3 — compact PR/de-sintering sanity check only if needed

This is not a new primary model and should not become a diversion.

Only if Milestone 2 remains ambiguous, construct the smallest possible benchmark that tests whether the existing surface-diffusion formulation drives a constrained neck in the expected instability direction near a known critical geometry. Prefer reusing existing Cartesian/substrate machinery or an analytically prepared near-critical profile.

Do **not** build a full axisymmetric production solver unless Milestone 2 demonstrates that dimensional geometry is the limiting issue.

The objective is only to verify the sign of the capillarity-driven neck evolution, not to reproduce a complete Plateau-Rayleigh paper campaign.

## Milestone 4 — forced event-response benchmark

Once a sink-off state with rising local stress is available, save a checkpoint just below the desired activation condition.

First test event mechanics independently of stochastic waiting:

- Start from the same near-critical checkpoint.
- Activate one sink explicitly in a benchmark-only mode.
- Run only long enough to observe a fraction or one completion of the finite `b` quota.

Required event direction:

- rigid-body displacement / densification strain increases,
- particle-substrate separation decreases,
- neck/contact width increases,
- local sintering stress decreases,
- total solid mass remains conserved,
- grain-volume change beyond explicit background Ostwald exchange is negligible.

If the event does not relax stress and broaden the neck, fix RBM/transport geometry before reconnecting the stochastic hazard.

## Milestone 5 — stochastic activation near the critical state

After Milestones 2 and 4 pass independently, reconnect the integrated hazard.

Do not wait through a long incubation trajectory. Initialize from a saved geometry already near the critical stress. Use the physical hazard law and ordinary random threshold; if needed for a diagnostic-only test, choose a seed/threshold giving an event in a short interval rather than changing the barrier physics.

Required coupled sequence:

1. sink inactive,
2. coarsening continues,
3. neck/contact decreases,
4. local stress and integrated hazard increase,
5. nucleation fires,
6. sink becomes active,
7. RBM/densification occurs,
8. neck/contact broadens,
9. local stress drops,
10. quota completes and sink returns inactive.

A single clean event is sufficient for qualification. Do not run a long multi-event campaign yet.

## Milestone 6 — short relative-rate map

Only after the directional sequence is working, vary the competing rates locally around the qualified state.

Primary independent controls:

- free-surface transport mobility,
- reservoir/Ostwald volume-loss rate,
- GB/eta mobility,
- active sink completion time / GB transport rate,
- optionally GB-to-surface energy ratio and anisotropy strength after the isotropic/weak-anisotropy response is understood.

Use multiplicative perturbations around the baseline (for example 0.1x, 1x, 10x), but run only for the same small target `Delta V2/V20` or equivalent short horizon. Construct a directional response table, not a long trajectory sweep.

Quantities of interest:

- sign and magnitude of `dx_neck/dV2`,
- sign and magnitude of `dSigma_local/dV2`,
- stress gain before activation,
- neck broadening per unit RBM displacement,
- stress drop per event,
- whether GB relaxation erases the accumulated stress before activation.

The goal is to identify the kinetic ordering that produces stress accumulation followed by discrete relaxation, not to calibrate every rate yet.

## Stop conditions / anti-diversion rules

- Do not run long 20 s / 2500 ms campaigns while a short directional test can answer the question.
- Do not broaden into a full PR study unless the minimal sanity check shows it is necessary.
- Do not replace the local stress measure with a mean `potential/area` stress.
- Do not change the hazard barrier to compensate for a geometry that fails to build stress.
- Do not change the qualified mass-preserving structural projection unless a conservation test proves it is wrong.
- Keep each milestone independently testable and commit after each physically meaningful result.

## Immediate Codex task

Start with Milestones 1 and 2 only.

1. Add the directional diagnostic and tests without changing physics.
2. Expose independent runtime controls for GB/eta mobility and for the reservoir neck-exclusion/localization behavior.
3. Implement the smallest physics-neutral option that allows a net grain-volume-loss reservoir without artificially protecting the neck.
4. Construct one near-critical sink-off initial state and execute the 2x2 short directional matrix.
5. Report the sign/magnitude table for `dV2`, `dL`, `dx_neck`, `dA_GB`, `dSigma_local`, and `dG_interface`.
6. Stop after this diagnosis and report which of H1 and H2 is supported. Do not proceed to long runs or stochastic-event tuning until reviewed.
