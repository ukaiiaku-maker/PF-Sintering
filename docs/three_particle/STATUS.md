# Three-particle ripening development

```text
PF-native stationary null: NOT YET FOUND
Phase A: NOT FULLY QUALIFIED
Phase B: DISABLED
Unequal 0.70 case: unchanged
Field guard: unchanged
No clipping
No fitted correction
```

## Current stationary-null follow-up

The implicit milestone is committed and pushed as `9b188cf`.
The prescribed finite-width bracket gives a zero-initial-flux root at
0.7428751867, but it is not stationary: the center gains 0.0277% and the
unchanged field guard stops the run at 0.8723 s. The thermodynamic virtual-work
estimates remain basis-dependent after the prescribed 200-step cleanup.
A qualified PF-native stationary null is still outstanding. Phase B is disabled.

See [null-study report](stationary_null/REPORT.md),
[plots](stationary_null/study_plots.pdf), and
[native timestep bound audit](stationary_null/native_bound_refinement.json).

## Previous implicit follow-up

The current-field implicit solver passes both the initial and late native
PF overlaps. The paired long-time comparison observes sustained additional
stress loading after the null's local relaxation criterion, in minutes of
wall time. However, the sharp-CMC matched-potential null develops finite-width
material transfer, and both trajectories reach the existing native f upper
bound before the 0.1% target. Full Phase A and Phase B remain gated.

See [implicit results](implicit/RESULTS.md), [plots](implicit/long_time_plots.pdf),
and [solver validation](implicit/README.md). No physical parameters, accepted
CMC checkpoints, production source, worker or watcher were changed.

## Previous CMC follow-up

The CMC initializer and short events-off ripening-direction/control tests now pass.
Full Phase A and Phase B remain gated. See [CMC report](cmc/REPORT.md),
[plots](cmc/analysis/phase_a_short_interval_plots.pdf), and
[morphology movie](cmc/analysis/morphology.gif). The old projected initializer is
disabled; its unmodified outputs are preserved and hashed in
`cmc/rejected_study_manifest.json`.

## Historical rejected projected-initialization study

The statements below describe the earlier study, before the CMC follow-up.

Phase A is NOT QUALIFIED. Phase B is gated off.

Base: `stochastic-pr-current-state-production-v1`, commit
`0023db8509df151622ae94fec140e714b967ef31`. See `THREE_PARTICLE_BASE_AUDIT.json`.
The original dirty production worktree has not been edited. The complete
tracked diff is saved externally; its unrelated historical changes were not
applied. An AST audit of 80 transitive source files used by the live worker and
monitor found no missing untracked dependency to copy. No results were copied.

Production uses one conserved solid fraction f, normalized binary ownership,
and axisymmetric cell volumes 2*pi*r*dr*dz. Three ownership fractions are needed
for three separately measured grains; the binary dynamic ownership update
cannot be used unchanged. The new fixed-GB Phase-A representation uses three
compact obstacle fractions summing to one, with eta_i=f*phi_i.

The production `axisym_numba_kernel.flux_kernel` accepts only f and mu,
not grain masks. It projects chemical-potential gradients tangentially on all
internal faces; only external boundary fluxes vanish. Thus it permits transport
across BOTH TJs. There is no per-grain volume correction in this kernel.
Its radial-face transverse derivatives wrap axially even under noflux; preserve
large vacuum margins and check boundary clearance. Do not use the historical
Cartesian `model.py` threeparticle initializer for this axisymmetric chain.

The finite-neck seed has nominal cap radii 100/70/100 nm, W=10 nm,
dx=1.25 nm, neck radius 28 nm, nominal angle 160 degrees. Each neck-to-cap join
matches radius and two derivatives using a quintic. It is a geometric seed,
not a relaxed surface; no seed transient may be called ripening. Nominal cap
radii differ from volume-equivalent radii, which must be measured separately.

Next: initialization-only constrained energy relaxation at fixed grain volumes
and centers, followed by actual PF surface diffusion and all three deterministic
controls. Equal nominal radii do not themselves prove equal chemical potential:
the center has two contacts and outer grains only one; test this rather than
subtracting a null drift or forcing its sign.

## Qualification result — 2026-09-09

**PHASE_A_NOT_QUALIFIED.** No canonical initial checkpoint exists; no stochastic
site has been implemented or launched. See `phase_a_gate_report.json` and
`initialization_candidate.png`. The full requested model remains unfinished.

Implemented and tested: three-grain finite-neck geometry; pairwise extension
of the normalized-ownership energy (exact binary reduction away from the
production near-vacuum ownership cutoff); fixed-GB native projected PF flux;
volume/centroid-constrained initialization experiment; per-contact measured
curvature, dihedral angle, Cannon–Carter force/stress and the existing PF local
geometric stress; four-branch reuse of the cumulative curvature watcher's
metrics; topology-resolution checks; bounded PF timing and transport-off tools.
The watcher extension reports instantaneous branch extrema. Its repeated-event
progressive flags and two-contact event records remain Phase-B work.

### Actual results

- Focused suite: **28 passed**, including production transfer, avalanche,
  corrected dynamics, and restart regressions.
- Unequal initialization: 30,000 total descent iterations across two calls,
  approximately 144 seconds. Grain-volume error `3.19e-14`; mirror error
  `4.12e-14`. Residual `5.72e-4` fails the `1e-5` screening threshold.
  The nominal 28 nm neck grows to 59.05 nm. Cubic extrapolation over
  0.5–2 interface widths gives 164.88 degrees, outside the 160 ± 3 degree gate.
- Equal initialization: 20,000 iterations, approximately 107 seconds.
  Grain-volume error `2.62e-14`; mirror error `5.36e-14`; residual `5.95e-4`;
  measured angle 164.69 degrees. This also fails initialization.
- In the unequal candidate, material in cells with `f<0.1` reaches 9.65% of
  total volume. This is not accepted as local surface equilibration. The
  initialization script now rejects a doubling of this fraction relative to
  the seed; this screening guard was added after the above exploratory runs.
  It prevents continuing those runs silently. It is an explicit numerical
  rejection criterion, not a physical vapor-solubility model.
- Angle metrology is sensitive to window and polynomial degree: quadratic
  fits on this candidate range from 170.3 degrees (0–1W) to 157.5 degrees
  (2–3W). Do not tune the window to select a passing angle. The current
  extrapolation requires calibration against a resolved equilibrium groove.
- Surface-weighted PF chemical potentials of the unequal candidate are
  15.89 MPa in the center and 19.21 MPa in either outer grain. This does NOT
  demonstrate the proposed center-to-outer mechanism: initialization is
  unqualified, and nominal radius alone does not fix the chemical potential
  of a constrained finite chain. No rate/sign correction was imposed.
- Native PF timing probe: 1,000 steps at the actual passive `M_s=6e-34`,
  `dt=4.8828125e-5` model-time, 1.174 seconds wall time on one thread,
  approximately 240 MB peak RSS, compressed field checkpoint 594,353 bytes.
  Cost: 24.04 wall seconds/model-time unit, or 1,543 wall seconds/physical
  second using the frozen clock mapping. Compilation excluded.
  This probe evolved an **unequilibrated seed** for timing only; its center
  gained material transiently. Its rate is not a ripening qualification or
  an estimate of long-time physical performance.
- Surface-transport OFF: 1,000 calls from the unequal unqualified candidate
  leave every field value and all grain volumes bitwise identical. This
  qualifies operator stationarity, not the canonical initial geometry.

### Scientific conventions and remaining work

Curvature is outward-convex positive, with a sphere giving `kappa=2/R`.
Cannon–Carter is implemented directly from the formula supplied in the task:
`Sigma=2*pi*Rb*gamma*sin(psi/2)-pi*kappa*gamma*Rb^2`. The frozen PF local
geometric estimator is called independently and retained separately. They
are not assumed identical. PF chemical potential is reported per volume
(Pa), without silently substituting geometric curvature for its variational
bulk/gradient/GB terms. Surface-weighted grain averages include the contact
region and are not the far-cap-only Gibbs–Thomson value.

The initialization-only projected energy descent is **an unsuccessful
candidate**, not promoted production physics. Its global grain constraints
permit unwanted dilute-phase material storage despite conserving the totals.
The next development step is a locality-preserving initialization method,
with independent curvature/angle verification and fixed volume/centroid
constraints. The current morphology also warrants examining how the chosen
GB spacing and finite-chain end conditions affect each grain's chemical
potential. Do not compensate with fitted transport rates or forced signs.

Required unequal ripening and equal-size null trajectories, all physical-time
Phase-A plots, and the morphology movie have **not** been generated from a
canonical state. They remain gated by initialization. No reduced ripening
solver or immutable geometry replay was introduced. Phase B must wait for
all Phase-A scientific gates and a committed/pushed passing state.

### Production isolation recheck

Original path: `/Volumes/Data/Data/PF-sintering/PF-Sintering`.
Original branch remains `diagnostic/pr-geometry-conservation-audit` at
`0023db8509df151622ae94fec140e714b967ef31`. The SHA-256 of its current tracked
diff and the original saved binary patch both equal
`ae7ec5aa058325a6f93dbf04308f60c573637422a13f2270861b7d7dddd99a2c`.
Worker PID 49267 and monitor PID 49324 were still alive at the final recheck.
Only shared Git metadata needed to add the experimental worktree/branch was
changed; no production source file, branch checkout, process, or run output
was changed by this development.
