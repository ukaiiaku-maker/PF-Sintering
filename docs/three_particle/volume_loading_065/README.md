# Events-off loading as a function of center-volume loss

This study continues the exact saved 30 s harmonic/current-field PF state for
Rc/Ro=0.65 and Ro=119.999 nm. The center-loss coordinate is
x = 1 - V_center/V_center,post-cleanup, using the original 200-step-cleaned
screen state, not the 30 s restart volume. The existing forced-root and
conditional-descendant studies remain separate and unchanged.

The exact original radius expression `119.999 * 1e-9` is retained. A launch
assertion caught the roundoff-level grid mismatch from the alternative literal
`119.999e-9` before any field was advanced. The corrected launch requires
bitwise equality of the saved grid, pinned GB positions and ownership arrays.

No events, stochastic draws, hazard accumulation or hazard stopping are used.
The bicrystal stress/activation definitions, mobility, temperature, W=4 nm and
original dr=0.5 nm/aligned dz are unchanged. All accepted states evolve the full
current field with the established harmonic mobility and adaptive implicit
solver. The existing field, mass, symmetry and topology/resolution limits remain
active. No accepted-field clipping, geometry replay or fitted correction occurs.

Checkpoints are retained near 1, 2, 3, 5, 7.5 and 10% loss. Trial time increments
are shortened when necessary to approach a milestone; field values and volumes
are never prescribed or interpolated into a checkpoint. The actual achieved
loss and physical time are reported. The near-milestone tolerance is 0.001
percentage point in center loss.

A plateau may stop the extension only after 2% total loss and a complete
1-percentage-point loss window. In both contacts, each of four consecutive
0.25-point secant slopes must have absolute magnitude <=0.05 MPa per percentage
point, and the stress range across the entire window must be <=0.05 MPa.
This is a predeclared operational plateau definition, not proof of an
infinite-time asymptote. The numerical criteria and all slopes are retained.
A physical/topology or numerical stop is reported separately, not relabeled as
an asymptotic plateau.

For each side, the diagnostic decomposition is exactly
sigma_kappa = -gamma_s*kappa1 and sigma_TJ = 3*gamma_s*sin(theta/2)/r_TJ.
Their sum is checked against the existing one-sided production value, and the
two-side mean against the unchanged within-contact scalar. LEFT-positive and
RIGHT-negative correspond to the center grain in global coordinates. No angle
average is substituted before applying the sine.

`loading_vs_volume.pdf` contains stresses, rates, their volume-coordinate
slope, stress contributions, measured curvatures/angles/radii, grain volumes and
center resolution. The displayed derivative is a centered secant over 0.05
percentage points, excluding the earliest x<0.1% transient. It is diagnostic
only; the plateau uses its separately specified windows. `loading_diagnostics`
contains physical time, energy, extrema, material/symmetry errors and the
instantaneous two-contact waiting scale. All accepted extension states retain
the underlying one-sided measurements in `report.json`.

The 2–5% milestones, if reached while resolved, actively ripening and materially
loading, will be assessed as candidate initial microstructures. No stochastic
production start is selected automatically, and the separate descendant
morphology qualification gate remains unchanged.


## Preserved symmetry stop and temporal-stability audit

The first extension stopped at 47.56371371 s and 1.17289794% center loss when
reflection error reached 1.32359e-8. Mass error remained 1.11e-15, f_max=1,
energy decreased, and both contact stresses agreed within 2.43e-5 Pa. Stress
was still increasing; this is not a plateau finding. The complete failed
history, checkpoint and figures remain in the original run and
first_extension_symmetry_stop provenance.

A discarded-copy probe at that field over the same 20 model-time interval
amplified reflection error by 1.7475 with one step, but reduced it to 0.6470
with two half-steps and 0.01525 with four quarter-steps. All embedded absolute
error estimates were below one. Thus the embedded norm alone missed this
small-amplitude temporal instability. No accepted field was symmetrized.

The continued study restarts from the exact saved 1% milestone, not the
out-of-guard field. Existing invariant limits now reject a trial before its
state/time are committed and retry at smaller time step. Guard values,
physical evolution, grid and error tolerances are unchanged. The coherent
accepted prefix through the 1% source is retained, with hashes linking to the
original history. The new lineage is runs/three_particle_volume_loading_065_guarded.

The separate 5 s ceiling comparison at the 1% state passed its declared local
overlap bounds, but used 47 versus 45 accepted states (20 versus 200 model-time
ceilings). It does not establish long-time stability. The study retains the
original ceiling of 20; no larger-ceiling result is inserted into its trajectory.

The guarded lineage has passed the original stopping time: a coherent accepted
checkpoint at 49.042395826 s is retained in guard_recovery_check. At least one trial
was rejected by the unchanged reflection guard and smaller time steps restored
symmetry within that bound. This is numerical continuation evidence, not a
plateau or a production-start selection.


## Full native-flux Jacobian continuation

The guarded frozen-mobility lineage remained within every accepted-state bound,
but needed increasingly small time steps in the dilute exterior tail. It was
superseded at 57.4841285095524 s and 1.25344213910603% center loss. Its complete
accepted history and exact field remain in the guarded run; handoff.json hashes
the coherent checkpoint/report. Its RUNNING report is the original snapshot;
the handoff records that this worker was retired. No failed trial was inserted.

At the separately preserved 54.501293839 s accepted state, stricter linear
solves stalled, and an algebraically equivalent potential formulation retained
the same reflection amplification. Those discarded tests are not promoted.
The linear-accuracy audit uses restart=80/maxiter=6 for every tolerance, whereas
the original solver uses restart=40/maxiter=3. The potential probe's --strict
flag reproduces its separate 1e-11 tolerance experiment.

For the observed odd mode, the native finite-difference instantaneous response
converges to a Rayleigh quotient near -3.82898 per model time. The frozen A H
approximation gives only -0.23408. The full flux derivative includes the
mobility and tangential-projector derivatives and gives -3.82898421. This is
local mode-response evidence, not a global eigenvalue or stability proof.
Whole-field finite differences in the dilute tail are limited by nonlinearity
and roundoff; the independent moderate-field regression verifies second-order
finite-difference convergence of the complete derivative.

The opt-in FullJacobianSurfaceDiffusion solves (I-h F'(f)) delta = h F(f)
and applies the increment through the common-face conservative native flux.
It changes the implicit numerical linearization, not F, the mobility law,
energy, grid, or physical parameters. It retains the established Richardson
estimate, linear tolerances, and original invariant/energy acceptance guards.
There is no clipping, reflection projection, fitted correction, or state replay.

A 64-model-time refinement failed its coarsest linear solve. That failure is
preserved in full_jacobian_refinement.json; its failed-row field is the input,
so differences involving that row are not equal-time convergence evidence.
The ceiling was NOT promoted. A separate 32-model-time audit, with 1/2/4
partitions and the same predeclared limits, passed. Coarse-to-finest field
error is 2.08450e-6 (bound 2e-5), contact stress difference 0.654981 Pa
(bound 5000 Pa), and relative grain-volume difference 1.37427e-10
(bound 1e-5). Refinement reduces field error to 2.42783e-7. Every completed
step passes embedded-error, energy, reflection, and mass checks. This is local
qualification; the live trajectory still checks every trial and may reduce h.

The active lineage is runs/three_particle_volume_loading_065_full_jacobian.
Its launch requires the passing audit, records source/audit/solver hashes, and
uses ceiling 32 model time with an initial trial of 20. It restarts the exact
accepted guarded handoff, retaining the earlier accepted history and 1%
milestone. All audit outputs are discarded copies, never trajectory states.
The focused loading/contact/mobility/implicit/full-Jacobian suite passes 20 tests.

The volume-loading decision remains pending. PF-native stationary null: NOT
YET FOUND. Phase A: NOT FULLY QUALIFIED. Phase B: DISABLED. Unequal 0.70 case
and field guard: unchanged. No clipping. No fitted correction. An instantaneous
scalar cancellation dot(V_center)=0 does not imply the full-field F(f)=0.
The separate conditional-descendant morphology gate remains not qualified.
