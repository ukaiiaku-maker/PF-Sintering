# Finite-width PF stationary-null study

```text
PF-native stationary null: NOT YET FOUND
Phase A: NOT FULLY QUALIFIED
Phase B: DISABLED
Unequal 0.70 case: unchanged
Field guard: unchanged
No clipping
No fitted correction
```

**The approved implicit milestone was pushed as `9b188cf`. This follow-up did
not produce a qualified PF-native stationary null. Phase B remains disabled.**

The prescribed compatible-CMC scan and exactly 200-step cleanup produce an
accurate zero-initial-grain-flux root, but its actual field is not locally
equilibrated and its released trajectory repeats the analytical null's drift.
No correction was fitted to the unequal case or its stress response.

## Construction and independent tests

Fresh compatible CMC solves at 0.72, 0.73, 0.7428902074688286, 0.755 and 0.77
all pass the existing cleanup guards. Every candidate uses the same signed-
distance map, W=4 nm, dr=0.5 nm, GB-aligned dz rule, energies, mobility,
temperature, fixed ownership and frozen contact-angle estimator. The aligned
coordinates change as required by each solved GB separation; resolution rules
do not change. Each cleanup contains exactly 200 native steps with both GB
faces blocked. The unequal initial state and all its trajectories are preserved.

Root finding on the released ownership-weighted center-volume rate gives

    Rc/Ro = 0.7428751866977
    (1/Vc) dVc/dt_model = 2.031e-16

A native first step changes center volume by only 4.4e-16 relative, while the
field changes by 7.44e-6. After 1000 native steps the field has changed by
0.00289 although the net center-volume change is still only −4.06e-11.
The initial field-rate maximum is **6.232618 per model time**.

These are different conditions:

    dVc/dt = integral(phi_center * F(f)) = 0
    stationary field: F(f) = 0.

The first is one scalar cancellation and does not establish the second.
The rate definition is identical to the accepted grain-volume diagnostics.
Sharp-GB-plane flux is also logged; it differs from the ownership-weighted
rate inside a diffuse GB and is not silently substituted for it.

## Why no thermodynamic potential root was asserted

For virtual variations b_i=w(f)*phi_i, form G_ki=dV_k/db_i and e_i=dE/db_i,
and solve G^T lambda=e. Three interface-localized weights test whether these
multipliers are independent of the admissible variation basis.

At the initial flux root the center-minus-outer estimates are approximately:

| Variation weight | Potential contrast |
|---|---:|
| f(1−f) | −0.02559 MPa |
| [f(1−f)]² | −1.53721 MPa |
| W times the field-gradient magnitude | −0.00484 MPa |

The spread is **1.53237 MPa**, and weighted KKT residuals are 8.7–10.4 MPa.
The far-TJ normal-projected potential contrast is about +247 Pa; it is a
separate diagnostic, not a derivative of minimized PF energy.

A direct, total-volume-preserving finite difference of the actual PF energy
agrees with the virtual-work derivative to 5.4e-7 relative at perturbation
1e-4. Thus the discrepancy is not a missing energy derivative. At this
unequilibrated state the virtual-work estimates cannot be promoted to unique
constrained thermodynamic chemical potentials. Selecting the basis with the
most convenient zero would be an unjustified calibration.

Basis independence here tests the fixed-ownership energy-minimization
interpretation. It is not necessary for every stationary state of a
semidefinite tangent-projected mobility: that mobility can have a nontrivial
nullspace. The decisive independent test is the actual native/released field
evolution, which is nonstationary. No claim is made that stationary PF states
are impossible in general, or that this small bracket exhausts all states.

## Released long-time qualification

The accelerated current-field run attempted 0.9 physical seconds. It stopped
at **0.8723046 s**, after **320.0 wall seconds**, at the unchanged
f≤1+1e-8 guard. No clipping, global mass correction, stress correction,
geometry replay or bound relaxation was used.

| Gate | Measured result | Outcome |
|---|---:|---|
| Reach 0.9 s | 0.8723046 s | fail |
| Relative center drift <1e-6 | 0.0002772848 (+0.02773%) | fail |
| Post-0.3988465-s projected-μ change <5000 Pa | 105945 Pa | fail |
| Post-transient CC change <5000 Pa | 47577 Pa | fail |
| Post-transient PF stress change <5000 Pa | 47096 Pa | fail |
| Relative total mass error <1e-11 | 3.33e-16 | pass |
| Mirror error <1e-8 | 1.15e-13 | pass |
| Resolved topology | existing guards satisfied | pass |

The new candidate and old analytical null nearly coincide in the trajectory
plots. Comparing the unchanged unequal trajectory against this failed
candidate gives descriptive excess increments of about 0.0481 MPa (CC) and
0.0504 MPa (PF), essentially preserving the previous finding. This is **not**
the requested stationary-null-controlled ripening calibration.

Initial and terminal native curvature-watcher outputs are archived. The
actual-field curvature plot uses fixed 3W TJ exclusions and excludes r<6W.
Pre-existing contour-extraction oscillations and pole-related errors remain
visible and largely coincide between the two states. An artifact-free
stationary-null claim is not made; the drift and bounds gates already fail.

## Separate native bound audit

The audited refined unequal terminal field has

    max(f) − 1 = 9.99973681637e-09
    maximum near (z,r) = (-19.1991, 100.25) nm
    native df/dt_model there = 2.3627711e-06 > 0.

The reflected point is symmetry-related. Native timestep refinement factors
1, 2, 4, 8 and 16 were tested both for one step and for an identical
**2.226086-microsecond** window (128 baseline steps).
These are rejected diagnostic trial arrays, not accepted continuation or
restart states. The solver guard remains unchanged.

A sufficiently small *single* step can stay just below the guard. Over the
same finite time, however, the overshoot remains about **1.0337e-8**, with an
additional outward change about **3.37e-10**. Temporal refinement does not
eliminate this finite-time crossing from this starting field.

The one-step dt→0 limit necessarily equals the inherited overshoot. Neither
that limit nor this short terminal-state audit proves that a spatially
converged PF equilibrium requires f>1. A compatible pre-guard trajectory and
spatial-error analysis are still needed before reformulating the guard.
There is no evidence here justifying an arbitrary larger bound or clipping.

## Outcome and remaining qualification

A PF-native stationary null has **not** been found. The missing ingredient is
a certified local-equilibrium state before taking constrained chemical-
potential derivatives or interpreting a single zero grain rate as equilibrium.
The prescribed 200-step cleanup is geometrically well behaved but does not
provide that certification. The bounds behavior also remains an independent
limitation. The accepted unequal case, physical parameters and Phase-B gate
have not been changed to manufacture a passing result.

The next state-construction problem must establish local stationarity of the
actual PF field, then determine its volume chemical-potential balance. The
present scalar root and drifting trajectory must not be used as that null.
This report preserves a failed calibration study, not a completed stationary-
null implementation.

## Evidence and reproduction

- [Five-page plots](study_plots.pdf)
- [Bracket and root](bracket.json)
- [Qualification gates](qualification.json)
- [Released trajectory and curvature watch](released_report.json)
- [PF energy derivative check](energy_derivative_check.json)
- [Native bound refinement](native_bound_refinement.json)
- [Initial protocol](PROTOCOL.md)
- [Actual-state hashes](run_manifest.json)

Scripts: `three_particle_stationary_null.py`,
`three_particle_stationary_null_evolve.py`,
`three_particle_native_bound_refinement.py`, and
`three_particle_stationary_null_analysis.py`. Use the repository venv,
NUMBA_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and a writable MPLCONFIGDIR.
The root's CMC geometry and actual f are saved under
`runs/three_particle_stationary_null`; the released output directory must be new.

33 targeted tests passed in this study. They include virtual-work equilibrium
basis independence, nonequilibrium contrast dependence, exact native-RHS
consistency, the implicit solver, CMC geometry and production transfer/restart
regressions. The actual-energy directional check is additionally recorded.
