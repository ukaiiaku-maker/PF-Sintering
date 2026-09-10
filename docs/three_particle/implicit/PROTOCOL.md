# Current-field implicit Phase A protocol

All CMC checkpoints, W=4 nm, aligned grid spacings, ownership, surface and GB
energies, passive mobility, time conversion and temperature are unchanged.
Events remain disabled. No volume-to-geometry replay or mass correction.

The first-order linearly implicit base method evaluates the native mobility
and surface projector at the current field and uses the exact chemical-
potential Hessian. Full/half step doubling with Richardson extrapolation
provides a second-order current-field update. Its error controller includes
field error (2e-5), fixed-estimator angle error (0.03 degrees), meridional
curvature error (30000/m), and maximum field increment (0.025). The final
updates use conservative native face fluxes. Dropped matrix coefficients
occur only in the ILU preconditioner; GMRES solves the complete operator.
Every accepted macrostep recomputes the contour, contact geometry and actual
chemical potential. Bounds failures and nonlinear-response errors reduce
rather than clip a step. There is no stochastic or ownership evolution.

Before any extension: both eleven-frame overlaps must pass the explicit
absolute limits in three_particle_implicit_qualify.py. All three grain volume
changes, both contact curvatures/radii/stresses, and projected potential
contrast are compared. A near-zero null uses an absolute volume-error floor.

Before inspecting the extended trajectories, define substantial null-TJ
relaxation as TWO consecutive doubling-time intervals with changes below
0.05 MPa in both contact stresses, 0.05 degrees in contact angle, and
50000/m in fitted contact curvature. This is an operational plateau criterion,
not proof that every diffuse mode has equilibrated. Start the stress-loading
comparison only at the END of the second interval. If the null never meets
this criterion, do not claim a post-transient loading regime.

First extension: at most 1,000,000 equivalent native steps for each case,
with a 900-second wall budget per invocation. Follow-ups use actual field
checkpoints. No 30-hour explicit continuation. A bounded later native overlap
and/or tighter-tolerance implicit refinement must verify long-step results.
Stop at 0.1% center-volume change, a demonstrable asymptotic stress trend,
or a resolution guard. Keep Phase B gated pending the comparison.

The fixed-GB Phase A equations cannot shrink the axial GB separation. Track
separation/W anyway (8.506 unequal, 10.134 null), together with actual material
core and neck resolution. Do not interpret a constant separation as evidence
of moving-GB extinction resolution. Stop if the existing 8W/core guards fail.
