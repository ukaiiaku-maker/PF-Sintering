# M16O — Geometry-Selected Sintering-Stress Pathways

**Central question**: Can we predict, from the initial geometry and Hussein analytical mechanics, which resolved morphologies will relax and which will develop rising sintering stress under sink-OFF coarsening?

**Status: in progress.** This report is being built incrementally as background PF screens complete. Sink OFF, hazard OFF, RBM OFF throughout — no barrier recalibration or Poisson production has been touched.

## Scope decision: topology family used

Section 4 of the handoff requested a literal sinusoidal-height substrate family (`H(r) = H0 + A*F(2*pi*r/lambda + phi)`) with a new exact concave-tangency fillet solve satisfying Young-Herring at the TJ. The project's existing `solve_body_fillet` only implements EXTERNAL tangency to a convex neighbor sphere (this project's existing finite-`chi=R_s/R_p` construction); a genuine trough/concave-bowl contact needs a different (internal-tangency) derivation that does not exist in the codebase yet and was not practical to derive, implement, and numerically validate from scratch within the available time.

**Substitution used instead: this project's already-implemented, already-validated finite-`chi` (neighbor-particle-curvature) family**, spanning:
- `chi -> 0`: small/sharp neighbor (strongly curved local contact)
- `chi = 1`: symmetric neighbor (particle-side and neighbor-side analytic fillet radii are exactly equal by construction)
- `chi -> infinity`: flat substrate (M16N's already-qualified relaxing control)

This is not an arbitrary shortcut: the handoff's own Section 20 decision logic says "if ALL sinusoidal candidates relax, pivot to the explicitly coarsening unequal-particle / pearl-necklace / Plateau-Rayleigh topology" — the finite-`chi` construction IS that topology, reached directly.

## Analytic pre-screen note (Section 6-7)

An initial attempt to use the analytic single-fillet radius (`solve_body_fillet`, chi-independent for the particle side, chi-dependent for the neighbor side) as a predictive proxy for the PF-measured neck curvature was checked empirically and found unreliable for the finite-`chi` family specifically: the two independently-solved fillets (particle-side `rho1`, neighbor-side `rho2`) meet at the TJ with a slope discontinuity (matching, not smoothly blending, at (a,0)) unless `chi=1` exactly — meaning the diffused, PF-measured curvature at finite `chi` is NOT simply either fillet's own radius. Verified directly: at `ratio=0.185`, `chi=1.0`, both `rho1=rho2=26.57nm` analytically, but the actual PF-measured `r_neck` at `t=0` is `16.30nm` — substantially tighter than either individual fillet, consistent with a corner/kink effect at the TJ that a finite measurement window partially resolves as extra curvature. Given this, **the analytic fillet radius is used only as a resolvability/domain-sizing check (Section 1), not as a quantitative sigma predictor for this geometry family** — PF-measured curvature (via the same validated NeckTracker + fixed-window circle-fit pipeline used throughout M16K-M16N) is authoritative.

## Initial-state (t=0) stress gradient across chi (ratio=0.185 fixed)

| chi | r_neck (nm) | X_neck (nm) | sigma (MPa) |
|---:|---:|---:|---:|
| 0.5 | 19.25 | 370.12 | 49.29 |
| 1.0 | 16.30 | 370.00 | 58.69 |
| 1.5 | 13.83 | 370.04 | 69.64 |
| flat (M16N) | 26.17 | 372.99 | 35.58 |

A clean monotonic trend already emerges at `t=0` alone: smaller `chi` (sharper relative neighbor curvature) → smaller `r_neck` → higher initial `sigma`, spanning 35.6-69.6MPa just from this one parameter at fixed `ratio`. Evolution trajectories (loading vs. relaxing) for each are in progress; this section will be completed with the full dataset.
