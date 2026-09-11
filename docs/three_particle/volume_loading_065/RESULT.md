# Volume-controlled loading extension: numerical stop

The requested A/B distinction is **inconclusive**. The actual events-off PF
trajectory reached **1.4137265543% center loss** at
**82.9423895733 s**, then reached the unchanged reflection guard despite
step reduction to the numerical floor. Only the 1% requested milestone was
reached; there are no 2–5% candidate initial microstructures from this study.
The calculation and its milestone watcher exited normally after preserving
the terminal report. No production realization was launched.

| Final accepted quantity | Value |
|---|---:|
| LEFT / RIGHT production stress | 18.27121711 / 18.27121711 MPa |
| LEFT one-sided negative / positive stress | 18.23214615 / 18.31028807 MPa |
| Root rate per contact | 0.06564827555 s⁻¹ |
| Instantaneous two-contact first-event scale | 9.52043286 s |
| Center span / W | 8.19999712 |
| Maximum f | 1.0 |
| Minimum f | −1.37488e−108 |
| Relative material error | −1.99840e−15 |
| Reflection error | 9.9999998745e−9 |
| Topology/resolution stop | false |

The wait scale is 1.25/(Gamma_LEFT+Gamma_RIGHT) at the frozen instantaneous
state, not a realized waiting time or accumulated hazard. No threshold was
drawn and no cumulative hazard was evaluated.

Stress increased by **0.24556252 MPa** from the ~1% checkpoint to the
terminal state. Of this increase, **0.20227658 MPa** comes from the curvature
term becoming less negative and **0.04328593 MPa** from the TJ term. The exact
within-contact decomposition at the endpoint is −3.36285360 + 21.63407071 =
18.27121711 MPa. Each side is evaluated separately before taking the production
mean; no mean-angle substitution or fitted correction is used. This supports
continued modest loading over 1–1.414% loss, but does not establish the requested
behavior at 2–5% or rule out a later plateau.

## Why the extension stopped

The terminal limit is **reflection**, not the field bound. The limiting pair
is in the dilute exterior tail at z=±0.248485 nm, r=156.25 nm, with f around
1.107e−4. Accepted f remains bounded, energy decreases, material is conserved,
and topology remains healthy. Failed trials were rejected before state/time
commit; the original field and reflection limits remain unchanged.

Discarded probes at h=0.001, 0.0005 and 0.00025 model time give reflection-error
growth per model time of 2.31670965e−10, 2.31669501e−10 and 2.31668851e−10,
converging to the native RHS value 2.31668047e−10. Their embedded errors are
below 5.3e−7, but each exceeds the reflection bound. The observed odd mode's
instantaneous Rayleigh quotient is positive: native RHS 0.0229977114,
full Jacobian 0.0229976333, and centered finite difference at epsilon=1e−9
0.0229976335 per model time. This later mode differs from the earlier damped
mode at 54.5 s, which motivated the full-Jacobian correction.

Thus the final crossing persists in small-step native evolution; further local
timestep reduction is not a remedy. This does **not** prove a continuum physical
instability or an eigenvalue result. Spatial/discrete-operator interpretation
remains unqualified. A symmetrized copy was used only to measure native
reflection-equivariance error (2.34e−14 in RHS); no accepted or restart field
was symmetrized. The observed odd mode was never removed or clipped.

A separately attempted 3 s adaptive-ceiling comparison aborted on the baseline
reflection guard. It supplies no equal-duration comparison and does not qualify
the proposed larger ceiling. Its input and failure report are retained.

## Preserved status

PF-native stationary null: NOT YET FOUND. Phase A: NOT FULLY QUALIFIED.
Phase B: DISABLED. Unequal 0.70 case: unchanged. Field guard: unchanged.
No clipping. No fitted correction. An instantaneous scalar cancellation
of dot(V_center) does not imply F(f)=0 for the complete PF field.

The forced-root/conditional-descendant payload remains separate and unchanged.
Its morphology gate remains not qualified. The focused numerical and contact
suite passes 21 tests. The exact terminal checkpoint/history, requested
one-sided measurements, decomposition, rates, grain volumes, diagnostics,
figures, and discarded native-mode probes are retained. No merge into main or
production is performed.
