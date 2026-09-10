# Live bicrystal call-path audit

Authority is the running resume-13 launch manifest copied beside this file.
The driver chain is resume13 -> resume11 -> current-state-transfer production
-> slow-GB production -> avalanche renewal. Historical module defaults are
not the live configuration. Read-only live source comparison is recorded in
isolation_audit.json.

## Two production stress calculations

`pr_coarsening_stochastic_two_event.measure` calls
`measure_experimental_pr_state`, which calls `experimental_geometry_record`,
`_particle_silhouette` and `_side_first_stresses`.

The physical TJ is the simultaneous intersection of f=0.5 and the adjacent
ownership fields. On each side, the original quadratic fit to the f=0.5
branch over 3W gives meridional curvature k_m and the measured side-equivalent
angle theta. `_side_first_stresses` evaluates, independently on each side:

    sigma_local,side = gamma_s [-k_m,side + 3 sin(theta_side/2)/r_TJ]

The second production stress uses continuous endpoint tangent turning divided
by the full free-branch arclength in place of k_m. The fitted TJ tangent and
resolved final chord are exactly `_continuous_endpoint_turning`; raw polyline
turning remains a separate diagnostic. The production serializer selects
`sigma_integral_continuous_Pa` for its integral-stress history.

Both contact scalars are the means already implemented inside the bicrystal
`_side_first_stresses`. They are not a newly introduced cluster combination.
The new evaluator calls that function directly and preserves negative/positive
one-sided values. LEFT maps negative/positive to L/C; RIGHT maps them to C/R.
A center-grain branch ends at the other field TJ, never beyond that contact.
Local fits and the full-branch turning calculation retain their original code.
Each contact uses the same particle-positive bicrystal coordinate frame: the
center is the particle and the adjacent outer grain is the substrate. RIGHT
therefore reflects the axial coordinate and arrays before calling the original
functions, then maps side names back to physical C/R order. The original
positive-branch pole closure and negative-branch convention are retained in
that local frame. This is a coordinate/indexing change, not a sign change to
the stress formula. A reflection regression verifies that mirror contacts give
identical integral and local scalars. A first global-z implementation exposed
an artificial signed-integral asymmetry; its diagnostic runs are archived as
rejected provenance, and replacement continuations use the validated frame. MW/N silhouette diagnostics are outside the two
requested stress families and are not used for ranking or activation.

The absent-third-grain regression supplies an asymmetric two-grain field and
compares against the original full `experimental_geometry_record` plus
`_side_first_stresses`, including both sides and both contact scalars. Results
are numerically identical. A separate test checks the root rate against the
original production `rate` configured exactly as the live run.

## Root activation and hazard

`pr_coarsening_stochastic_two_event.measure` passes **sigma_local_Pa** to
`rate`. It does not pass integral stress, Cannon–Carter stress, a cluster
average, GB affinity, or a null-subtracted quantity.

    G_fit(sigma) = G_floor + (G0-G_floor) exp[-a (max(sigma,0)/sigma_hat)^n]
    G_root = G_fit + root_formation_penalty
    Gamma = (2 pi r_TJ / b_event) nu0 exp[-G_root/(kB T)]

The exponential stress dependence is already the empirical activation-work
coupling. There is no additional sigma*A*b subtraction. The output
`barrier_reduction_from_zero_eV` is G0-G_fit, a diagnostic of this same law.
The live root formation penalty is 0.9803507282790153 eV, nu0=1e12/s,
T=1830.15 K and b_event=0.25 nm. The external creep export lists a different
b=0.36 nm; production's explicit event/site-count b=0.25 nm is retained.
The model-time clock is 0.01557994316955921 seconds per model time.

The live threshold is 1.25 times an Exp(1) draw. There is no deterministic
stress threshold. At constant state E[t_wait]=1.25/Gamma; for two independent
contacts P(any root by t)=1-exp[-(H_LEFT+H_RIGHT)/1.25]. This expression is a
probability diagnostic, not a replacement event sampler. Sparse saved-frame
quadrature cannot supply an exactly localized stochastic crossing.

The live code also retains a sparse copy-only reaction-coordinate probe in
`pr_corrected_normalized_ownership_production.sparse_reaction_coordinate_probe`:
Fq=-(E_after-E_before)/dq and delta_mu_q=Fq/(dV_source/dq), using an infinitesimal
volume-conserving event on a copied state. This is not either serialized
production stress above and does not feed `rate`. Its three-grain version
requires a validated three-grain event operator; it has not been substituted
with affinity*A*b or claimed implemented by this events-off screen.

## Current-state event work and avalanche architecture

`corrected_pr_thermodynamics.make_corrected_state_evaluator` samples the actual
variational mu on the field-TJ GB center plane with radial partial-cell area
weights to r_TJ-3W (original support fallbacks retained). It separately averages
surface mu over 1W–3W on each adjacent free branch and then uses the existing
two-side mean. `_node_state` sets the fundamental delivery affinity to
mu_GB-mu_TJ_local. The zero-storage node partitions delivery between branches;
it does not redefine that thermodynamic affinity or the root activation stress.
The three-grain evaluator uses its existing pairwise energy derivative and
reuses these exact plane/branch sampling helpers with contact indexing.

`production_mass_transfer_event.current_state_mass_transfer_event` rebuilds
masks from current fields, transfers A_GB(current)*dq with zero net source,
relaxes the actual PF state, rejects failed trials and advances time using
`transport_clock_increment`. The latter uses trapezoidal reciprocal delivery
rates; `ModelTimeGBTransport` supplies qdot from the instantaneous affinity.
The recorded affinity*A_GB*b is the corresponding one-b work diagnostic,
not an additional term in G_root. The binary transfer function preserves
normalized ownership pointwise. Its capacity calculation bounds auxiliary
arrays, but it explicitly does not clip the accepted live field.

The active production descendant law is
max(G_floor,G_fit(sigma_local)-h*1.5 eV), with h initially 1 and h<-0.70 h after
each completed descendant. The facilitated-source lifetime is 9 ms; root
formation penalty is excluded from descendant barriers. No cluster-specific
facilitation law has been added.

The present screen does not yet implement or launch three-grain events.
Any subsequent event implementation must generalize the binary ownership
bookkeeping, preserve the current-state material-transfer equations, validate
contact-local crossing/restart and other-contact bookkeeping, and then record
morphology, strain and both contacts after each accepted increment. Neither a
perfect stationary null nor large global stress oscillations is required.

## Aggregate output convention

The center diagnostic is the mean of LEFT-positive and RIGHT-negative local
stresses. The cluster diagnostic weights each existing contact scalar by its
current pi*r_TJ² area. Both are labeled diagnostics and neither feeds a root
rate or event affinity. The events-off screen cannot supply activated strain
or avalanche markers; it does not manufacture these outputs.
