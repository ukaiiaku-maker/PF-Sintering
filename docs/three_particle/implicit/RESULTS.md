# Long-time current-field Phase A result

**Acceleration succeeds, and sustained additional late stress loading is
observed. Full Phase A and Phase B remain gated by the drifting analytical
null and the native field-bounds limit.** No stochastic events were enabled.

The accepted CMC geometries, W=4 nm, aligned mesh, surface/GB energies, passive
mobility, temperature, time conversion and fixed ownership are unchanged.
Every step evolves the actual field. There is no geometry-versus-volume
replay, grain-volume projection, global mass correction, clipping, or imposed
symmetrization. The frozen contact-angle estimator is unchanged.

## Scientific comparison

The preregistered null criterion requires two complete doubling-time intervals
with changes below 0.05 MPa in both stresses, 0.05 degrees in contact angle,
and 50000/m in contact curvature. Both coarse and refined histories first
satisfy it at **0.398847 physical seconds**
(25.6 model time). The passing intervals are 6.4–12.8 and 12.8–25.6 model time.
This operational decay criterion does not imply zero long-range flux or a
fully stationary diffuse-field null.

Use the common verified interval **0.398847–0.875177 s**.
The table shows one contact; the other is symmetry-related, and both are
recorded in the accompanying histories.

| Quantity | Unequal 0.70 | Analytical matched-μ null |
|---|---:|---:|
| Center-volume change from canonical state, at common end | -0.080668% | +0.027829% |
| CC stress at analysis start | 0.834007 MPa | 1.426949 MPa |
| CC stress at common end | 0.930164 MPa | 1.474665 MPa |
| Post-transient CC increase | 0.096157 MPa | 0.047716 MPa |
| Post-transient PF geometric-stress increase | 0.097955 MPa | 0.047234 MPa |

The unequal-minus-null differences of these increments are
**0.048441 MPa (CC)** and
**0.050722 MPa (PF)**.
Both stress measures have positive slopes in all three consecutive thirds of
the analysis interval. The unequal center continues losing material, each
outer grain gains half its loss, and its projected μC−μouter remains positive.
Thus the late increase is resolved beyond the initial shared TJ transient.

The null also gains center material and increases its stress. At the common
endpoint its center has gained about 0.028%, so it is no longer the quiet
long-time control implied by the original short interval. Its projected
potential contrast becomes negative during continued diffuse-field evolution.
The stress-increment difference above is a measured comparison between two
different geometries; it is not a calibrated causal subtraction or a proof
that all additional stress originates solely from center-volume loss.
Do not promote this result to a clean stationary-null qualification.

## Stopping and resolution

The coarse unequal trajectory stops at 60.111984 model time; the refined
trajectory continues to **60.456580 model time (0.941910 s)**,
reaching **0.084096% center loss**.
The 0.1% target was not reached. The clear late stress trend provides the
alternative stopping result requested in the numerical development task.

The coarse/refined null stops occur at 56.364605/56.173293 model time. At all
four terminal checkpoints, the next unchanged native explicit step also
violates the existing upper bound f≤1+1e-8. The last accepted fields remain
within that bound. No failed trial is promoted to a checkpoint, and no bound
is relaxed. The stop is timestep-sensitive at the sub-percent level and the
overshoot is tiny; it must not be presented as a physical instability or
extinction event. It calls for a separate review of the native bounds policy
and bulk/GB behavior before extending the campaign.

The center GB separation stays **8.50594W** (unequal) and **10.13383W** (null).
These GBs are fixed in this Phase-A model: their separation cannot shrink
under the current equations. The material-core/neck guards remain satisfied,
outer ownership cores have zero overlap, and the unequal interior radial
size remains about 116 nm. The stop is not caused by two GB cores merging.

## Numerical evidence

- All eleven initial native observations pass for both cases, including
  ΔVL, ΔVC, ΔVR, both contact curvatures/radii/stresses and projected μ contrast.
- At approximately one million equivalent native steps, a further 10000-step
  native restart overlap passes for both states: maximum field difference
  below 7e-9 and stress difference below 0.13 Pa.
- Four-times tighter field tolerance and smaller macrosteps preserve the
  null analysis start and both late stress trends. At actual coarse observation
  times, maximum post-transient stress error is below 29 Pa and relative
  center-volume error below 5.0e-8. All original late-refinement limits pass.
- Exact-time endpoint checks give CC errors of 36.6 Pa (unequal) and 15.2 Pa
  (null), with center-volume differences 4.58e-9 and 4.97e-8 relative.
- Comparing two linearly interpolated histories on an arbitrary uniform time
  grid adds coarse-sampling error to tiny volume differences (about 0.10–0.11
  ppm). The qualification instead evaluates actual coarse observations against
  the finer history and additionally checks identical-time checkpoints. The
  uniform-grid diagnostic is retained in the JSON report, not silently ignored.
- Maximum global relative volume error in the paired coarse histories:
  4.44e-16; maximum mirror error:
  3.23e-13. These are conservative native face-flux updates,
  not volume corrections or mirror projections.
- 35 focused tests pass, covering native operator algebra and boundary
  stencils, time consistency, mass, mirror symmetry, actual-state restart,
  reused-preconditioner equivalence, transient detection, CMC geometry and
  unchanged production transfer/avalanche/restart regressions.

## Cost and scope

The accepted coarse trajectory stages total **7.03 minutes
(unequal)** and **7.75 minutes (null)**. Native per-step
benchmarks extrapolate to about 45.7 and
41.1 wall hours for the same respective endpoints:
approximately **390× / 318×** acceleration.
Those are computational-cost extrapolations, not assumed linear ripening rates.
Refinement, solver scouts and native-overlap validation add separate wall cost.
No tens-of-hours explicit continuation was launched.

Full Phase A remains unqualified because the long-time null is not stationary
and the existing field-bounds policy stops the model short of the volume
change target. Phase B remains disabled. The next qualification questions
are the finite-width stationary null and the native bounds behavior; neither
has been bypassed to obtain a desired stress curve.

## Artifacts and reproduction

- [Four-page long-time plots](long_time_plots.pdf)
- [Numeric comparison and refinement gates](long_time_comparison.json)
- [Initial native overlap](overlap.json)
- [Late native overlap](late_native_overlap.json)
- [Coarse/refined native guard audit](guard_audit.json)
- [Actual-state file hashes](run_manifest.json)
- [Method](README.md) and [pre-extension protocol](PROTOCOL.md)

Numeric coarse and refined histories are included here as CSV files. Full
fields/checkpoints are retained locally under `runs/three_particle_implicit`,
separate from all CMC initialization and native-reference outputs.

Use the repository venv with NUMBA_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and a
writable MPLCONFIGDIR. The driver refuses long extension without a passing
initial overlap artifact. Each output directory must be new. Continuations
use `--resume` with actual f/ownership/grid/time/next-h checkpoints; numerical
factorizations are disposable caches. Run `three_particle_implicit_late_check.py`,
`three_particle_implicit_guard_audit.py` and `three_particle_implicit_analysis.py`
to reproduce the final verification and plots from the recorded states.
