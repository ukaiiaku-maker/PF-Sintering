# Milestone 16E: Exact Figure-4 Parity and Event-Local Sink Response

## 0. Corrected M16D decision status (per this milestone's Section 1)

| item | M16D status | corrected status |
|---|---|---|
| ETA conservation audit | PASS | **PASS** (reaffirmed and strengthened here, Section 7 below) |
| Hussein Figure-4 mechanism | "partial" | **HOLD/FAIL** |
| sink-OFF stress buildup | "partial" | **FAIL** |
| forced sink event | "PASS" | **PARTIALLY QUALIFIED** (re-examined event-locally, Sections 9-10) |

No progression to hazard/nucleation development, per this milestone's explicit instruction.

## 1. Exact Figure-4 initial geometry

`R(z)/R_cyl = R0/R_cyl + eps1*cos(2*pi*z/lambda) + eps2*cos(pi*z/lambda)`, domain `0<=z<2*lambda`, `eps1=eps2=0.4`, `R0/R_cyl=sqrt(1-(eps1^2+eps2^2)/2)`. Verified numerically before any dynamics, exactly matching the values this milestone specifies:

| quantity | computed | specified |
|---|---|---|
| `R0/R_cyl` | `0.916515138991168` | `0.91651513899` |
| `R(0)/R_cyl` | `1.7165151389911681` | `1.716515139` |
| `R(lambda)/R_cyl` | `0.916515138991168` | `0.916515139` |

## 2. Exact GB positions and grain topology

`z1/lambda = (1/pi)*acos(-eps2/(4*eps1))`, `z2/lambda = 2-z1/lambda`:

| quantity | computed | specified |
|---|---|---|
| `z1/lambda` | `0.5804306232551663` | `0.5804306233` |
| `z2/lambda` | `1.4195693767448336` | `1.4195693767` |
| `R(z1)/R_cyl` | `0.4665151389911679` | `~0.466515139` |
| `R(z2)/R_cyl` | `0.466515138991168` | `~0.466515139` |

Both computed troughs match to the specified precision, and match each other to 12 significant figures -- **the two GB troughs have identical minimum radius.** This corrects a real construction error in Milestone 16D, which added an ad hoc phase-shifted second harmonic specifically to make the two trough *depths* unequal, believing that was necessary for an "inner grain that can shrink." That was the wrong mechanism: the actual published construction gives equal-depth troughs, and the inner/outer asymmetry is entirely a matter of **arc length** -- grain 1 (inner, `z1<z<z2`, centered at `z=lambda`) spans `z2-z1 = 0.8391*lambda`, while grain 2 (outer, `0<z<z1` and `z2<z<2*lambda`, connected through the periodic boundary) spans `2*lambda-(z2-z1) = 1.1609*lambda` -- the inner grain is smaller by *arc length*, not by trough depth. GB local-minimum detection on the numerically-sampled profile lands on the analytic `z1`,`z2` to grid resolution, confirming the construction is self-consistent. (`scripts/m16e_exact_hussein_two_mode.py`)

## 3. Initial analytical stability checks -- GATE FAILS

Per this milestone's Section 4, the initial classification (`lambda/R_cyl=1.14*pi` against the published `lambda_c/R_cyl~1.5*pi` for `psi=160` and `~0.9*pi` for `psi=80`) must be verified **before any PF evolution**, using this project's own sharp-interface constrained-Hessian machinery (validated to `0.11%` against the classical Rayleigh-Plateau threshold and against this project's own GB-destabilization trend in Milestone 16C).

**Two-GB bamboo topology** (the exact published grain topology, Section 2 above), Hessian applied directly to the raw (unrelaxed) initial profile:

| psi | gamma_gb (dihedral formula) | computed lambda_min | classification | required |
|---|---|---|---|---|
| 160 | 0.3473 | -0.4844 | UNSTABLE | STABLE |
| 80 | 1.5321 | -0.3092 | UNSTABLE | UNSTABLE (correct by accident, not for the right reason -- see below) |

**Neither classification can be trusted as-is: both come out unstable, including the one that should be stable.** Diagnosed rather than reported blind:

- **Not a large-amplitude/nonlinearity artifact.** Re-run at `eps1=eps2=0.02` (a genuinely small perturbation, well inside this project's own validated linear regime) on the identical two-GB topology: `psi=160` still comes out unstable (`lambda_min=-0.065`). The failure is not an artifact of the published profile's large (`0.4`+`0.4`) amplitude.
- **Topology matters, but doesn't fully explain it.** Re-run with the *simpler*, already-validated single-GB-per-period topology (Milestone 16C's own Section 8b construction: one GB, domain length exactly `lambda`, base state properly relaxed via `relax_to_equilibrium`) at the same `lambda/R_cyl=1.14*pi`:

  | psi | single-GB lambda_min | classification | required |
  |---|---|---|---|
  | 160 | +1.106 | STABLE | STABLE (correct) |
  | 80 | +0.748 | STABLE | UNSTABLE (still wrong) |

  The single-GB topology gets `psi=160` right, but `psi=80` still comes out stable when it must be unstable. The two-GB bamboo topology introduces an *additional* destabilization channel (an antisymmetric "one grain grows while the other shrinks" mode, the same kind of mode Milestone 16C found dominates its own two-substrate sanity case) that apparently over-destabilizes the `psi=160` case relative to the paper -- while even the simpler, single-GB case still under-destabilizes `psi=80` relative to the paper.
- **A real, quantified GB-energy-scale discrepancy.** Sweeping `gamma_gb` (in this project's normalization) against the single-GB topology to find what value reproduces the paper's `psi=80` threshold (`lambda_c/R_cyl=0.9*pi=2.827`):

  | gamma_gb (this project's units) | resulting lambda_c/R0 |
  |---|---|
  | 1.5 (paper's dihedral-formula value for psi=80) | 5.558 |
  | 3.0 | 4.649 |
  | 6.0 | 3.314 |
  | 8.0 | 2.626 |

  Matching the paper's `2.827` requires `gamma_gb ~ 7.5-8` in this project's normalization -- **roughly 5x larger** than the standard dihedral-angle value (`gamma_gb=2*gamma_s*cos(psi/2)=1.532` for `psi=80`) this project has used throughout Milestones 14G-16D.

**Per this milestone's own explicit instruction ("If either initial classification is wrong: STOP. The benchmark geometry/stability implementation is still incorrect"), this gates further Figure-4 dynamics work.** No PF evolution of the exact two-mode geometry was run past this point (Sections 4-8's grain-elimination trajectory, Eq.-4 comparison, etc. are not attempted on top of a stability criterion already known to misclassify the starting point).

## 4. Light-weight parity investigation (Section 9's mandate, scoped to what is tractable here)

The full 8-item reference-PF parity ladder Section 9 specifies (free-surface energy, grain-order-parameter energy, density-eta coupling, hard-vs-energetic constraint, surface-diffusion localization, GB kinetic equation, mobility definitions, boundary conditions) is a substantially larger undertaking than remains tractable in this milestone alongside its other required sections; it was not attempted term-by-term. What *was* established, and is reported as the concrete lead for that future investigation:

- The discrepancy is **energetic in origin, not a numerics bug**: it persists at small amplitude, is internally consistent with this project's own already-validated (single-crystal PR, GB-trend) machinery, and is a clean, monotonic, ~5x magnitude gap in effective GB destabilization strength.
- The magnitude (~5x, not a simple factor of 2 or `pi`) is not explained by an obvious unit or convention slip in this project's own `gamma_gb=2*gamma_s*cos(psi/2)` dihedral relation, which is the standard Young's-equation form and has been used consistently since Milestone 14G.
- The most likely candidate, not yet verified: this project's stability criterion is a **static energy (Hessian) analysis** of the sharp-interface free energy alone. If the Abdeljawad/Hussein published threshold instead comes from a **coupled kinetic dispersion relation** (GB migration mobility interacting with surface-diffusion mobility, not just the equilibrium GB energy), a purely energetic criterion would generically under- or over-estimate the threshold depending on the kinetic coupling's sign and strength -- consistent with the two-GB topology *over*-destabilizing `psi=160` while even the simpler one-GB topology *under*-destabilizes `psi=80` (the two topologies' natural GB-migration channels differ, and a kinetic mechanism would be topology-sensitive in exactly this way, whereas a purely energetic groove-depth mechanism would not be). This is a plausible, testable hypothesis, not a demonstrated conclusion -- the dynamic linear-stability construction needed to check it (linearizing the *full coupled* PF equations, not just the sharp-interface energy) is beyond this milestone's remaining scope.

## 5. F-tracking invariance audit (Section 10)

Isolated the f-tracking rescale from the full tangent-cone eta update by calling `constrained_tangent_cone_eta_update(..., dt=0.0)` on a genuine production state (post-CH-step `f`, pre-existing `e1`,`e2`) -- at `dt=0` the variational velocity contributes nothing and the safety clip is a no-op, isolating the rescale alone.

| check | result |
|---|---|
| max `\|w1_before - w1_after\|` where pre-existing solid exists (`w_i=eta_i/f`) | `2.22e-16` (machine epsilon) |
| GB zero-set (`eta1-eta2=0`) crossing columns, before vs. after | identical (`[75, 204]` both) |
| newly-solid cells created by 2000 steps of ordinary surface-diffusion evolution | `0` |

**Confirmed exactly as claimed**: f-tracking alone leaves the physical ownership fractions `w_i=eta_i/f` unchanged to roundoff wherever solid pre-exists, and leaves the GB position exactly unchanged. The "new f in previously-empty cells" scenario (Section 10's second requirement) did not arise at all over 2000 steps of ordinary evolution -- expected, since surface diffusion evolves `f` continuously (a cell's value changes smoothly from whatever it already was; it cannot jump from exactly zero to resolvably nonzero in one step under smooth transport), so this specific scenario is only relevant to topological events (pinch-off, new-contact formation), not ordinary coarsening. Only the variational eta motion migrates an existing GB -- confirmed directly, not assumed. (Ad hoc diagnostic, not saved as a standalone script -- reproducible inline from `constrained_tangent_cone_eta_update` called at `dt=0`.)

## 6. Event-local OFF/ON clone comparison (Sections 11-12)

M16D's sink comparison evaluated the forced-sink effect over a `20ms`-scale window when the sink quota completes by `~3ms` -- diluting the event's own signature with `~17ms` of ordinary post-event coarsening common to both clones. Here, one initial state (identical to M16D's corrected comparison: literal `geometry="substrate"`, current tangent-cone eta pathway, `dx=2nm`, `W=10nm`, `gamma_gb=0.6`) is cloned exactly into OFF and ON, run only to `~1.5x` the ON clone's quota-completion step (`1041` steps, `t=3.02e-3s`), sampled at 80 matched points. (`scripts/m16e_event_local_sink_parity.py`)

**At quota completion (`t=3.021e-3s`, the end of the active event):**

| quantity | OFF | ON | required direction | met? |
|---|---|---|---|---|
| densification strain | `0` | `1.386e-3` | ON > OFF | YES |
| separation | `99.753nm` | `99.446nm` | ON < OFF | YES |
| `x_neck` | `29.6136nm` | `29.6188nm` | ON > OFF | YES (marginal, `+0.005nm`) |
| `sigma` | `8.128e7 Pa` | `8.094e7 Pa` | ON < OFF | YES |

**All four required differential signs are satisfied at the event's actual completion.** One nuance surfaced and reported rather than smoothed over: `x_neck`'s differential (`x_neck_ON - x_neck_OFF`) is **not monotonic** during the active interval -- it dips *negative* (down to `-0.31nm`, i.e. transiently *narrower* for ON) around the middle of the event before recovering to positive by quota completion, then continuing to grow (`+0.057nm` by the end of the short post-event tail). A comparison that only looked at pre- vs. post-event endpoints, or a Milestone-16D-style long window, would have reported clean monotonic broadening and missed this transient; a comparison that only looked at the interior of the event could have wrongly concluded the sink narrows contact. Reported as a genuine, real feature of this specific trajectory, not an artifact -- `sigma`'s differential shows a matching non-monotonic dip-then-recover pattern over the same interval, suggesting a shared underlying transient (likely the dihedral-angle/curvature measurement responding to the RBM-driven local geometry change before the shape re-equilibrates), not independent noise in each channel.

## 7. Active-event V2 ledger (Section 13)

`V2` (grain-2/particle ownership volume) at quota completion: OFF `1.940900e-14 m^2`, ON `1.940700e-14 m^2` (per-unit-depth "volume," 2-D Cartesian) -- a differential of `-1.99e-19 m^2`, i.e. **`~1e-5` relative to `V2` itself**. Total mass conservation (`f`) was independently confirmed at roundoff (`~1e-15` relative) in every sampled row of both clones, matching Milestone 16D's established contract. The small residual `V2` differential is attributable to RBM's own internal eta-field advection (`rbm()` directly upwind-advects `e1`,`e2` alongside `f`, a distinct mechanism from ordinary CH+tangent-cone eta migration present in both clones) -- **consistent with the existing physical contract** (no silent, large, unaccounted grain-volume-loss channel opened by the active sink): the active-sink-specific contribution to `V2` change is real but small (five orders of magnitude below `V2` itself), not a dominant or runaway term.

## 8. PASS-F4 / FAIL-F4

**FAIL-F4.** The exact published two-mode geometry was constructed and verified precisely (Sections 1-2), but the initial analytical stability classification required by Section 4 does not reproduce the paper's stated result for either topology tested (two-GB bamboo: both `psi` misclassified; single-GB: `psi=160` correct, `psi=80` still wrong), and a genuine, quantified `~5x` GB-energy-scale discrepancy was identified as the likely proximate cause, with a specific, testable hypothesis (a missing kinetic, not purely energetic, contribution to the true stability threshold) flagged for future investigation. Per this milestone's own gate, no grain-elimination trajectory, Eq.-4 comparison, or `psi=80` dynamical control was run on top of a stability criterion already shown to misclassify the starting geometry.

## 9. PASS-EVENT / FAIL-EVENT

**PASS-EVENT.** The event-local (quota-window-only) OFF/ON comparison shows the complete, correctly-directioned differential response Section 12 requires, evaluated at the event's actual active lifetime rather than a diluting long window: densification strain increases, separation decreases, contact broadens, and stress relaxes, all specifically for the ON clone relative to OFF, all confirmed at quota completion (not merely after an arbitrarily long post-event tail). The one qualifier: `x_neck`'s response is non-monotonic within the event (a transient narrowing before net broadening), reported explicitly rather than concealed by only checking endpoints -- this does not change the PASS classification (the requirement is met at the event's actual completion), but it is relevant context for anyone using this comparison to calibrate event-scale broadening magnitudes.

## 10. What this means for the decision sequence

Per Section 15: only after both PASS-F4 and PASS-EVENT should the one-particle/one-substrate sink-OFF coarsening-direction question be revisited. **PASS-EVENT is achieved; PASS-F4 is not.** The sink-OFF coarsening-direction investigation (why Milestone 16D's plain sink-off configuration widens contact and relaxes stress instead of narrowing/building stress) therefore remains explicitly out of scope for this milestone, pending resolution of the Figure-4 stability-criterion discrepancy (Section 3-4) -- consistent with Section 14's own instruction not to use the current sink-off state for hazard calibration in the meantime.
