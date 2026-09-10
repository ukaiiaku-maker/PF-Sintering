# CMC initializer and deterministic ripening follow-up

**The analytical initializer and short-interval ripening/control checks pass.**
Full Phase A is not yet qualified: sustained ripening-driven stress buildup,
long-time null stability, and efficient long-time integration remain open.
Phase B has not been implemented or launched.

All work is in `/Volumes/Data/Data/PF-sintering/PF-Sintering-3particle`, on
`exp/three-particle-ripening-v1`. The base remains production freeze `0023db8`.
The old projected-energy initializer is disabled. Its output files were not
resumed, replaced, or copied into the new trajectories; hashes are recorded
in `rejected_study_manifest.json`.

## Mathematical construction and compatibility

The solver uses the supplied CMC equation with **sum-principal curvature**
`K=1/(r sqrt(1+r'^2))-r''/(1+r'^2)^(3/2)`; a sphere has `K=2/a`.
The center is solved on a symmetric half-domain with an integrated half-volume
state. Its full volume is `2*pi*integral_0^L r^2 dz`.

The isotropic, planar, Young-balanced 160-degree contact has one-sided slopes
`-tan(10 degrees)` and `+tan(10 degrees)` on the center and right outer branch.
The left contact is its mirror. Both free surfaces meet at one contact radius.
The regular one-pole outer CMC segment is a spherical cap. Its volume fixes
its sphere radius, contact radius, and centroid once the angle is specified.

This imposes a compatibility condition that cannot be omitted: independently
solving two segments at arbitrary GB spacing does not make their contact radii
match. The scan explicitly records that mismatch. A coupled BVP then solves
for the center half-length that satisfies the prescribed volumes, slopes,
and shared radius. No gaps are filled with a fillet or receiver mask.

Similarly, the outer centroid is an output of this CMC construction, not an
additional independent boundary condition. The new calculated positions are
held fixed during cleanup to tight tolerances. The old seed's 160.156 nm
center spacing was not preserved. Prescribing arbitrary centroids as well as
volume, GB position, pole regularity, and angle generally overconstrains the
CMC problem; an active centroid constraint would instead introduce an axial
Lagrange-multiplier term in the curvature equation.

The numerical method implements the CMC equation and angle condition provided
in the user brief. It is not represented as a line-by-line reproduction of an
unavailable Josell paper. The Cannon–Carter formula is retained as supplied;
its bibliographic reference is *Interplay of Sintering Microstructures, Driving
Forces, and Mass Transport Mechanisms*, [DOI](https://doi.org/10.1111/j.1151-2916.1989.tb07705.x).

## Geometry map and selected state

The 42-case analytical spacing scan and six compatible-chain solves take
about 0.12 seconds. Radius ratios 0.4, 0.5, 0.6, 0.7, 0.8 and the equal-volume
comparison are included. Each ratio scans seven center half-lengths; solved
but disconnected profiles are never accepted as three-grain geometries.
The scan follows a regular branch; it is not an exhaustive search for every
possible unduloid branch or a full second-variation stability proof.

| Volume-radius ratio | Compatible half-length (nm) | Kc (10^6 /m) | Ko (10^6 /m) | Desired ordering |
|---|---:|---:|---:|---|
| 0.4 | 3.219 | 62.589 | 17.135 | yes, under-resolved |
| 0.5 | 6.267 | 36.335 | 17.135 | yes, under-resolved |
| 0.6 | 10.781 | 24.716 | 17.135 | yes, under-resolved at W=10 nm |
| **0.7** | **17.012** | **18.789** | **17.135** | **yes** |
| 0.8 | 25.185 | 15.441 | 17.135 | no |
| 1.0 | 48.061 | 12.061 | 17.135 | no |

Here radius means **volume-equivalent grain radius**, not a spherical cap's
radius or a grain's maximum radial extent. At outer equivalent radius 100 nm,
the outer cap sphere radius is about 116.717 nm, and the shared contact radius
is **114.944 nm**. The selected 70 nm-equivalent center is a thin lens with
GBs at **±17.01187 nm**, not a small spherical bulge connected by 28 nm necks.
Its sharp outer-grain centroid spacing from the chain center is about
70.009 nm. This geometry change is the consequence of the CMC compatibility
conditions, not imposed ripening kinetics.

The selected analytical driving difference is **1.65380 MPa** in `gamma*(Kc-Ko)`,
or **1.65380e-23 J/atom** using the unchanged model atomic volume `1e-29 m^3`.
No transfer rate, barrier, diffusivity, temperature, or stochastic parameter
was tuned to produce that sign.

The primary null is a spherical central zone with exactly the outer cap's
sphere radius. Its volume-equivalent ratio is **0.7428902075**, and its GBs are
at **±20.267653 nm**. The independently solved BVP matches Kc and Ko to about
`7e-10` relative error. The equal-volume case is a secondary comparison with
opposite analytical ordering.

## Mapping, metrology, and short cleanup

The user approved **W=4 nm** and nominal grid spacing **0.5 nm**. At W=10 nm,
the selected 34.024 nm center violates the retained `GB separation >= 8W`
resolution guard. W=4 nm gives 8.506 widths. Smaller scanned ratios are not
substituted because their compatible center regions are even thinner.

The physical energies and passive mobility are unchanged:
`gamma_s=1 J/m^2`, `gamma_gb=0.3472963553 J/m^2`, `M_s=6e-34`, and `T=1830.15 K`.
The numerical energy coefficients are recalculated from the approved W.
Production's planar profile convention is `2*sqrt(k_f/W_f)=W`, so mapping uses
`tanh(d/W)`. Inserting an extra sqrt(2) would change that code's interface width.
The initial field is a direct signed-distance mapping. There is no global
correction of f or deposition into dilute vapor. Diffuse versus sharp grain
volumes differ by less than 0.4%; those differences are measured and retained.

Both GBs are placed exactly on finite-volume faces by using
`dz=L/ceil(L/0.5nm)`, while `dr=0.5 nm`. Actual dz is approximately 0.48605 nm
for the unequal case and 0.49433 nm for the null. This preserves the exact
analytical GB positions. An earlier nearest-face blocking experiment was
rejected: it moved the null's effective boundary and narrowly failed cleanup.

The angle estimator is **f=0.5, quadratic fit on each side over 1W–2W,
extrapolated to the fixed GB plane**. It was selected on an analytical
ratio-1 CMC mapping, excluding the first W of diffuse-TJ core before any
transport test. Error is 0.0432 degrees on training and 0.0382 degrees on a
held-out ratio-1.1 mapping. It is unchanged across all three production-resolution
cases, cleanup scouts, and trajectories. The initial selection allowing the
TJ core was excluded before the first PF cleanup; that rule was never used
to accept a trajectory. No per-result fitting-window tuning is performed.

The initial 2,000-step-duration scout was stopped at 800 steps when it crossed
the angle guard. A separate 40-iteration ownership-only cleanup reused the
existing fixed-geometry ownership routine, conserved ownership amounts, and
held f unchanged; it did not materially improve that result and is not used
for the selected state. Both scouts remain saved separately.

The common cleanup duration is **200 native PF steps**, with cross-GB flux
blocked at the exact GB faces. This duration was selected from the recorded
scout to limit cleanup to a discretization adjustment. A 5% contact-curvature
check was added before applying this common protocol to the three cases.
All three aligned cases pass the same fixed limits:

| Maximum change/error | Unequal | Matched null | Equal-volume comparison |
|---|---:|---:|---:|
| Grain volume, relative | 1.46e-8 | 1.08e-11 | 1.60e-8 |
| Centroid displacement / Ro | 2.00e-8 | 1.96e-8 | 1.56e-8 |
| Neck radius, relative | 3.72e-4 | 3.75e-4 | 3.75e-4 |
| Angle error (degrees) | 0.408 | 0.343 | 0.310 |
| Contact curvature error | 3.97% | 3.15% | 3.13% |

Mapped seeds, rejected scouts, and guard-passing numerical initial checkpoints
are distinct. `canonical_initial.npz` denotes the chosen CMC-based numerical
initial state, not proof that every diffuse-TJ chemical-potential mode is at
equilibrium. The field-potential profiles still show a local TJ transient;
full Phase-A qualification remains open for that reason and the stress-loading
limitations below.

## Actual native-PF transport result

After releasing both GB faces, all evolution uses the existing tangential
surface-flux kernel with fixed three-grain ownership. No event deposition,
GB sink/advection, or reduced ripening model is used.

| Case | Relative center-volume change | Relative outer change, each | Total-volume error |
|---|---:|---:|---:|
| Unequal 0.70 | **−2.52298e-6** | **+4.32035e-7** | −1.11e-16 |
| Matched-potential null | +1.89441e-10 | −3.8776e-11 | −4.44e-16 |
| Equal-volume comparison | +2.82047e-6 | −1.40806e-6 | −3.33e-16 |
| Surface transport OFF | exactly zero | exactly zero | exactly zero |

The unequal case loses center volume at every recorded interval. Each outer
gains `1.8142424e-27 m^3`, half the center loss `3.6284848e-27 m^3`, to about
`5e-10` relative accuracy. The mirror-field error stays around `1e-15`.
GB positions are fixed, and no topology stop is approached in this short run.
The null change is more than four orders of magnitude below the unequal
signal. All checkpoints pass a bitwise field/ownership save-load roundtrip;
the untouched production restart regressions also pass. This is not a new
qualification of two-contact stochastic restart behavior.

The three cases each use 10,000 steps with the same Courant scaling, so their
physical durations differ slightly with their aligned dz. Unequal duration:
**0.0111626 model-time = 173.913 microseconds** using the frozen seconds mapping.
The transport-off case uses 1,000 steps and is exactly stationary.

Raw PF `mu(f=0.5)` and q-weighted grain means are not the sharp-interface
chemical potential. On a curved tanh profile they carry different profile
factors. The analysis additionally computes `integral mu df` across the radial
interface, away from 2W around each GB, which tends to `gamma*K`. Its initial
unequal center/outer values are **18.742 / 17.090 MPa**; the null initially
matches to about **267 Pa**. At the end, the null's projected means differ by
about 0.166 MPa despite almost zero net exchange. This finite-width/local
relaxation effect is recorded, not offset or fitted away. Long-time null
stationarity is not established by the short interval.

## Curvature, stress, and performance limits

The full profile plots retain both contact regions. They show the physical
rounded groove and a decaying field-potential transient around each TJ.
The preexisting curvature watcher is applied to four branches; ordinary
meridional-curvature maxima outside the core grow about 0.17% on the outer
branches and 1.48% on the center branches. No mirror-breaking artifact appears.
The center gradient changes from a nearly zero baseline, so its large relative
growth factor is not treated as a spike by itself. The full raw profiles are
saved, and W-scale-smoothed profiles are plotted. Pole extraction has a stable
baseline discretization feature; center branches offer only a short region
outside the watcher's 3W core. These metrology limits prevent claiming a
complete long-time artifact qualification.

Cannon–Carter and existing PF geometric stresses are reported separately.
For the two unequal one-sided free surfaces at each contact, the diagnostic
uses the arithmetic mean of their independently fitted total curvatures as
`kappa_b`; this averaging convention is explicit and is not a new force law.
Unequal CC stress changes from approximately −0.164 to −0.133 MPa; PF geometric
stress changes from 16.969 to 16.979 MPa. The matched null has a comparable
common stress transient. This short interval therefore **does not establish
sustained ripening-driven stress buildup**. No monotonic force law was imposed.

The unequal trajectory costs **30.53 wall seconds** including periodic
metrology, with **442.9 MB peak RSS** and a **1241578-byte compressed checkpoint**.
Each report records peak RSS, compressed checkpoint size, steps, model/physical
time, and wall time. The measured short-interval center loss is about
**0.030% per wall hour**. A linear extrapolation would put 1% change near
34 hours, but rates and morphology need not remain constant.

The benchmark supports investigating a current-field implicit or multirate
surface-flux integrator: resolve local curvature/TJ modes, then take larger
error-controlled steps for long-range flux while updating the actual morphology.
The observed local relaxation interval is thousands of times shorter than an
extrapolated percent-level ripening interval. No such optimization is yet
qualified, and no immutable geometry-versus-volume replay is introduced.

## Artifacts and reproduction

- `geometry_scan.json`, `chemical_potential_null.json`
- `analytical_morphologies.png`, `cmc_geometry_map.png`
- `angle_calibration.json`
- `*_short_aligned_cleanup_report.json`
- `analysis/joint_qualification.json`
- `analysis/phase_a_short_interval_plots.pdf` and individual PNGs
- `analysis/morphology.gif` — actual fields at true scale; only ppm volume change
- `runs/three_particle_cmc/*_short_aligned/canonical_initial.npz`
- `runs/three_particle_cmc/ripening_*/`: histories, full profiles, frames,
  final checkpoint, and detailed report

Run with the repository Python environment and `NUMBA_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`. Scripts: `three_particle_cmc_scan.py`,
`three_particle_cmc_figures.py`, `three_particle_cmc_calibrate.py`,
`three_particle_cmc_cleanup.py --case CASE --canonical-study`,
`three_particle_cmc_ripening.py --case CASE`, and
`three_particle_cmc_analysis.py`. The frozen estimator is already stored;
recalibration is not part of repeating a trajectory.

Focused tests: **29 passed**, including CMC curvature and volume quadrature,
contact matching, exact spherical null, grid-aligned mapping, and preserved
production transfer/avalanche/restart behavior. The rejected projected descent
was not invoked by this follow-up test selection.
