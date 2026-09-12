# Three-particle production campaign status

Updated: 2026-09-11

## Current phase

Production source selection and renewal-driver hardening. No production random
threshold has been drawn yet.

## Selected initial microstructure

- Lineage: symmetry-released full-domain `Rc/Ro = 0.65`, `Ro = 119.999 nm`,
  `W = 4 nm`, `dx = 0.5 nm` events-off continuation.
- Selected checkpoint: `runs/three_particle_volume_loading_065_symmetry_released/loss_2pct.npz`
- Checkpoint SHA-256: `9360b587f169d71ab5ff2859df0e2b824c30e3bfe70c4d6e6eb5adb8a166694a`
- Physical time: `305.76900020092603 s`
- Center-volume loss: `0.019997071838334213`
- LEFT/RIGHT local stress: `18.5069380 / 18.5069227 MPa`
- LEFT/RIGHT root rate: `0.06607302242 / 0.06607298206 s^-1`
- Characteristic two-contact first-root time: `9.4592341623 s`
- Center span: `8.2 W`; topology healthy.

The checkpoint was selected from deterministic field and rate diagnostics before
any production threshold draw. Continued loading through 2.121% loss changed the
characteristic wait only to 9.445 s, so the additional cost did not materially
improve sampling. The preferred 0.5--5 s range was not reached naturally at this
healthy point; no kinetic parameter was changed.

## Morphology decision

The previous center-side curvature warning is classified as a **non-blocking,
monitored structure** for this campaign. The unmasked local peak moved from
`3.418 W` at descendant quota `q/b = 0.25` to `3.543 W` at `q/b = 0.50`, then
was absent at `q/b = 0.75` and `1.0`; after the complete transit and source
window it appeared at `3.792 W`. This is inconsistent with a feature pinned to
the `3 W` diagnostic mask edge. Native timestep refinement changed the field by
at most `2.79e-9`, local stress by `1.43 Pa`, and the event clock by
`1.13e-8` relatively. The state remained bounded and conserved, and mobility
spatial refinement reduced solver differences monotonically. It has not been
shown to corrupt activation, conservation, or topology.

The feature remains recorded at each completed event. A new progressive narrow
feature that becomes grid/mask-position dependent or materially changes contact
stress remains a stopping condition.

## Stochastic state

- Fixed production seed: `20260910`
- LEFT/RIGHT thresholds: not drawn
- LEFT/RIGHT hazard ratios: `0 / not-drawn`, `0 / not-drawn`
- Active avalanche/event: none
- Completed avalanches: 0
- Cumulative production strain: 0
- Geometric strain: 0 at the selected production reference
- Symmetry enforcement: disabled
- Reflection: diagnostic only
- Phase B: authorized for this end-to-end campaign after driver validation

## Numerical health

Material conservation, field bounds, energy sanity, topology/resolution,
ownership closure, and nonlinear accuracy guards remain active. No clipping,
fitted correction, symmetry averaging, or symmetry projection is permitted.
The passive production integrator is being moved to the overlap-qualified full
native-flux Jacobian with preconditioner reuse. Atomic restart is required before
the genuine draw and launch.

## Latest checkpoint

`runs/three_particle_volume_loading_065_symmetry_released/loss_2pct.npz`

