# C2 rigid-frame kinematics and instantaneous virtual work

> **Subsequent interpretation:** the literal signed result in this directory is
> retained as the contact-cusp audit. The physical densifying diagnostic is
> the positive unilateral derivative, decomposed in
> [`../c2_one_sided_force_decomposition/REPORT.md`](../c2_one_sided_force_decomposition/REPORT.md).
> No projected simplex-tangent force is used as the event force.

## Scope

This audit corrects the mechanical coordinate for the preserved first RIGHT
root of the C2 trajectory. It does not modify, relax, or restart that
trajectory. It does not alter the root law, barrier, site density, temperature
dependence, or ordinary surface-diffusion evolution.

For the RIGHT contact, the imposed rigid frames are

\[
U_L=U_C=0,\qquad U_R=-u,\qquad q_L=0,\qquad q_R=u.
\]

The ownership planes are derived from those body frames. The LEFT plane is
fixed and the RIGHT midpoint plane moves by \(-u/2\). Material centroids are
diagnostics only and are never used to impose or recover frame motion.

## Energy-free kinematic qualification

Discarded states were constructed at \(u/b=0,0.1,0.5,1\), before any energy
was evaluated. All kinematic gates pass:

- the LEFT and CENTER frame translations are exactly zero;
- the RIGHT frame translation is \(-u\);
- the LEFT ownership plane is fixed and the prescribed RIGHT-plane shift is
  \(-u/2\);
- all three independent rigid-frame tracers reproduce their prescribed shifts;
- total material is conserved to the reported numerical precision;
- material-centroid shifts and free-surface shapes are recorded separately;
- no symmetry projection, centroid constraint, or global minimization is used.

The combined record is in `kinematic_audit.csv`. The same observables are
separated into `rigid_frames.csv`, `ownership_planes.csv`,
`material_centroids.csv`, and `free_surface_geometry.csv`. The discarded fields
are under `runs/c2_corrected_rigid_frame_audit/`.

The measured diffuse ownership-plane position does not move by exactly the
prescribed subcell midpoint displacement because it is recovered afterward
from a sampled diffuse profile. The maximum recovery error is reported in
`kinematic_summary.json`; it is a diagnostic interpolation error and is not
fed back into the rigid-frame coordinate.

## Literal signed virtual-work test

Starting from the preserved stochastic RIGHT-root snapshot, the test constructs
both \(+\delta u\) and \(-\delta u\) by translating only the RIGHT body/frame.
The LEFT and CENTER frames remain fixed. The root morphology is not globally
stationarized, and no centroid constraint is applied.

The direct signed construction exposes a contact cusp. Under refinement to
\(\delta u/b=9.765625\times10^{-6}\), the one-sided forces converge separately
but do not approach one another:

\[
F_R^{+}=1.65823202\times10^{-7}\ {\rm N},\qquad
F_R^{-}=-7.08105565\times10^{-7}\ {\rm N}.
\]

Their jump is \(8.73928766\times10^{-7}\) N. Therefore the two-sided
instantaneous derivative \(-\partial G/\partial u\) is not defined for this
literal signed event construction at the preserved root. The symmetric
quotient is retained only as a diagnostic and is not reported as a physical
conjugate force.

The corrected Cannon--Carter comparison uses the far-field global-CMC pressure
difference and, separately, the full-domain KKT multipliers. It does not use
diffuse-TJ curvature. The global-CMC force is
\(4.87514950\times10^{-7}\) N. The converged positive-approach PF force is
65.99% smaller. The KKT construction gives \(3.67713296\times10^{-7}\) N, but
the instantaneous root has a projected KKT residual of 0.0210, so that KKT
pressure estimate is not qualified; the approach force is 54.90% smaller than
it.

The decision is therefore:

```text
C2_INSTANTANEOUS_DERIVATIVE_NOT_DEFINED
UNRESOLVED_EVENT_THERMODYNAMICS
finite event path: NOT AUTHORIZED, NOT LAUNCHED
stationary barrier: NOT INTERPRETED
```

The corrected kinematics remove the ambiguity between rigid-frame motion and
material-centroid motion. They do not establish instantaneous PF/sharp-interface
conjugacy, so the stationary barrier from the globally relaxed ownership-frame
branch has no qualified contact-densification interpretation.

## Event evolution contract

If a later model closes the instantaneous conjugacy gate, a finite RIGHT event
must keep \(U_L=U_C=0\) and advance \(U_R=-u\). Surface redistribution then
evolves over physical time, while material centroids respond naturally and
ordinary center-particle ripening remains a separate process.

For a later LEFT event, the coordinate is

\[
U_L=+u,\qquad U_C=U_R=0.
\]

For simultaneously active contacts, it is

\[
U_L=+u_L,\qquad U_C=0,\qquad U_R=-u_R.
\]

No general multiparticle mechanical solver is introduced by this audit.

## Future contact networks

For a larger contact network, rigid-body translations must be solved globally
from contact compatibility. Freely connected particles may translate
collectively. External constraints and loops generally require elastic
deformation and stress evolution. The present three-particle qualification
avoids that problem by fixing the center frame and moving only the selected
outer frame toward it.
