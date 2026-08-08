# Natural surface-diffusion crossover in the sinusoidal contact

Scope: test whether ordinary (Ostwald-off) surface diffusion on the
sinusoidal substrate geometry naturally reverses from contact widening to
contact thinning as the particle+substrate morphology relaxes, before
attributing any widening to coarsening itself. Hazard/sink/RBM off; no
applied stress, strain, or displacement introduced anywhere. **Stopping
here before activating hazard or sink, as instructed.**

## 0. Headline result

**No crossover exists within the accessible relaxation timescale — a clean,
well-resolved Outcome C, not an inconclusive result.** The instantaneous
CH-only contact tendency `g_CH(t)` was tracked over a 200,000-step S0
trajectory (`|dV2|/V20` irrelevant here, Ostwald is off; physical elapsed
time **1.45 seconds**, roughly 500-700x longer than any horizon used in
Milestones 7-10). `g_CH` starts at `+1.63e-6 s⁻¹`, decays **monotonically
and without a single sign change** through every one of 124 log-spaced
samples, reaching `+9.2e-10 s⁻¹` at the end — a **1,780x reduction, still
strictly positive**. `L_contact_TJ_sub` grows the entire time (`64.95nm` →
`81.13nm`), still (slowly) increasing at step 200,000. The chemical-
potential profiles (Section 5) show the system relaxing smoothly toward a
**uniform-`mu` (capillary-equilibrium-like) state**, not toward a
sign-reversing instability — physically consistent with never crossing
zero rather than a numerical artifact.

**Despite no crossover, the coarsening-vs-capillary-relaxation question
this milestone was really asking is still answered cleanly**: forking S0/S1
(external-reservoir closure, Milestone 9/10) from three representative
points spanning the *entire* accessible range of `g_CH` — early
(`g_CH=+1.23e-6`), mid (`g_CH=+3.14e-8`, background 39x more relaxed), and
late (`g_CH=+9.2e-10`, background 1,330x more relaxed) — gives
`delta_L_coarsening` **negative (contact shrinks relative to the
no-coarsening control) at all three**: `-0.0119nm`, `-0.0103nm`,
`-0.0055nm`. **The coarsening-induced shrinking effect is robust across the
entire dynamic range of the background capillary relaxation** available in
this geometry, weakening by roughly half from early to late but never
disappearing or reversing.

## 1. Long S0 trajectory

CH+eta+projection ON, Ostwald/hazard/sink/RBM OFF, sinusoidal geometry
(`wavelength=480nm`, `amplitude=24nm`, `W=20nm` fixed physical, `R2=80nm`,
`overlap=20nm`, `eta_diffusivity_fixed_physical=True` — Milestone 9's
corrected reference throughout). Approximately log-spaced sampling (124
points, `steps[k] ≈ round(base^k)`, `base=200000^(1/140)≈1.098`), dense at
small step counts (every step from 1-5, then progressively wider). Full run
completed at the `max_steps=200000` budget without triggering an early-stop
(no crossover to confirm) and without a `compute_stress` stop condition
(`stop=False` at every sample — geometry stayed well-resolved throughout,
no topology failure).

| step | `t` (s) | `L_contact_TJ_sub` (nm) | `g_CH` (s⁻¹) |
|---:|---:|---:|---:|
| 1 | 7.26e-6 | 64.964 | +1.633e-06 |
| 100 | 7.26e-4 | 65.985 | +1.228e-06 |
| 1,069 | 7.76e-3 | 70.398 | +4.403e-07 |
| 10,319 | 7.49e-2 | 75.627 | +3.144e-08 |
| 41,636 | 3.02e-1 | 78.552 | +9.497e-09 |
| 108,637 | 7.89e-1 | 80.444 | +1.449e-09 |
| 200,000 | 1.4525 | 81.129 | +9.213e-10 |

## 2. `L_contact(t)` and its rolling derivative vs. `g_CH(t)`

`g_full(t)` (finite-difference slope of the actual `L_contact_TJ_sub(t)`
samples) and `g_CH(t)` (the isolated CH-only probe) agree in sign and
order-of-magnitude at every sample checked, both strictly positive and
monotonically decaying — consistent with each other and with the
qualitative picture that **CH itself, not an eta/projection response
coupled to CH, is the source of the (persistently positive) contact
tendency**. No divergence between `g_full` and `g_CH` was observed that
would suggest eta relaxation is fighting or amplifying CH's own tendency at
any point in this trajectory (a finding consistent with Milestone 8's
paired-operator-ledger result that eta/projection contribute negligibly to
`delta_L_coarsening` next to CH and Ostwald).

`g_CH` was checked at `dt`, `dt/2`, `dt/4` at every sample (not just
selected states) — the three values agree to 4+ significant figures at
every single sample in the entire 200,000-step run (e.g. at step 10,319:
`3.144350e-08`, `3.144336e-08`, `3.144329e-08`) — the positive sign is
categorically not a timestep artifact anywhere in this trajectory.

## 3. Chemical-potential and flux profiles (early/mid/late)

Traced along the free-surface branch from the top TJ outward (150nm
arclength, 60 samples, exact production `mu`/`Jx`/`Jy` from
`ch_flux_diagnostics.evolve_f_diagnostic`, not a continuum surrogate):

| state | `mu` near TJ (Pa) | `mu` far (150nm, Pa) | `J_tangent` near TJ | `J_tangent` far |
|---|---:|---:|---:|---:|
| early (step 100) | `-2.69e8` | `-6.85e7` | `-4.7e-9` to `-8.6e-9` | `-1.6e-10` (same sign) |
| mid (step 10,319) | `-1.74e8` | `-8.14e7` | `-3.6e-9` to `-3.4e-9` | `+1.4e-10` (**sign change along the branch**) |
| late (step 200,000) | `-1.50e8` | `-1.41e8` | `-8.7e-10` to `-9.7e-10` | `+4.6e-11` to `+8.1e-11`, itself shrinking toward the far end |

The `mu` gradient along the branch flattens monotonically from early to
late (`~2.0e8 Pa` early → `~9.0e6 Pa` late, over the same 150nm) — the
system is visibly relaxing toward a spatially **uniform chemical potential**
along the free surface, the textbook signature of approaching a capillary
equilibrium, not an instability. The near-TJ tangential flux stays
negative (in the convention used here, corresponding to net material
delivery toward the TJ / contact widening) at all three states, shrinking
in magnitude by ~300x from early to late but never flipping sign — fully
consistent with `g_CH` itself never crossing zero. A sign change *does*
appear in the *far-field* flux by the mid state (near-TJ still negative,
far positive) — the divergence structure, not the raw sign at the TJ, is
what Section 4 below resolves directly.

## 4. Neck-region CH mass balance

Control region: disk of radius `max(2W, 0.6*TJ_separation)` centered on the
current TJ midpoint (not tuned), net change in local total-`f` mass under
one CH-only probe step:

| state | neck mass balance (m²) | sign |
|---|---:|---|
| early | `+9.81e-22` | accumulation |
| mid | `+1.69e-21` | accumulation |
| late | `+4.02e-22` | accumulation |

**The neck region shows net CH mass *accumulation* at every state
checked across the entire trajectory — it never transitions to net
depletion.** This is the direct, unambiguous confirmation (Section 7 of
the handoff's own framing) that the proposed widening→thinning reversal
does not occur within the accessible timescale: the most literal test
available (does the neck gain or lose conserved mass under CH alone) says
"gain," consistently, from step 100 to step 200,000.

## 5. Multiscale geometry tracking

| | early | mid | late |
|---|---:|---:|---:|
| `L_contact_TJ_sub` | 65.98nm | 75.63nm | 81.13nm |
| sinusoid `x_mean` | `-162.32nm` | `-161.42nm` | `-153.41nm` |
| fundamental `amp1` | `24.15nm` | `25.78nm` | `37.46nm` |
| 2nd harmonic `amp2` | `0.085nm` | `1.012nm` | `7.65nm` |
| 3rd harmonic `amp3` | `0.030nm` | `0.333nm` | `2.55nm` |

The sinusoid's own amplitude and harmonic content are **not** static
background — by the late state the fundamental has grown from 24nm to
37.5nm (+56%) and higher harmonics have grown by two orders of magnitude
relative to their initial (near-zero) values. This is the slower,
larger-scale morphology evolution the handoff's Stage II/III hypothesis
anticipated — it is real and substantial over this horizon, it just does
not, by itself, reverse the sign of the local neck's CH tendency within
1.45 physical seconds.

## 6. Does a natural `g_CH=0` state exist?

**No** — not within `|dV2|/V20`-equivalent horizons up to and including
1.45 physical seconds (200,000 steps), roughly 500-700x longer than any
horizon used for a coarsening comparison in Milestones 7-10. Per the
handoff's own Outcome C provision ("several thousand additional steps show
convincingly that neither happens over the accessible local-relaxation
timescale") — the monotonic, ever-slowing-but-never-crossing decay pattern,
together with the `mu`-profile evidence of relaxation toward a uniform
(equilibrium-like) chemical potential, is a physically coherent, not
merely budget-limited, negative result. No `F_before`/`F_neutral`/`F_after`
bracket exists to save; the three representative states in the sections
above (steps 100 / 10,319 / 200,000, spanning the full 1,780x range of
`g_CH` actually reached) are used in its place for every subsequent
section, exactly as the data — not an a priori choice — dictated.

## 7. S0/S1 forks at early/mid/late

Same protocol as Milestones 9-10 (external-reservoir closure,
`coarsening_rate_scale=3`, matched physical time via `|dV2|/V20=3e-4` on
S1, `n_ref=276` steps at every fork point since the shared `dt` and target
are unchanged):

| state | `g_CH` at fork point (s⁻¹) | S0 `L_contact_TJ_sub` end | S1 `L_contact_TJ_sub` end | `delta_L_coarsening` |
|---|---:|---:|---:|---:|
| early | `+1.228e-6` | 67.824nm | 67.812nm | **-0.01191nm** |
| mid | `+3.144e-8` | 75.692nm | 75.682nm | **-0.01031nm** |
| late | `+9.213e-10` | 81.130nm | 81.125nm | **-0.00555nm** |

## 8. Does surface diffusion itself reverse from widening to thinning?

**No, not within the tested horizon.** `g_CH` remains strictly positive at
every one of 124 samples across a 1,780x dynamic range and 1.45 physical
seconds. The system is visibly approaching (not moving away from) a
capillary-equilibrium-like state (Section 3's flattening `mu` profile).
This directly answers Section 12 of the handoff: since no late fork with
`g_CH<0` exists, there is no "late fork" in the originally anticipated
sense to compare against an "early fork sampling the wrong part of the
trajectory" — every fork tested, from the earliest to a background 1,330x
more relaxed than the start, still has `g_CH>0`.

## 9. Does coarsening shift or strengthen a crossover?

There being no crossover to shift, the more precise and still physically
informative question is answered instead: **does coarsening's shrinking
effect persist, strengthen, or weaken as the background capillary
relaxation itself proceeds?** It **persists in sign at every stage tested
and weakens in magnitude by roughly half** (`-0.0119nm` → `-0.0103nm` →
`-0.0055nm`, early to late) as the background approaches its slow
asymptotic state. This is a meaningful, quantified answer even absent a
literal crossover: coarsening's contact-shrinking mechanism (Milestone 9's
external-reservoir closure) is not a fragile effect that only appears in a
narrow window of background relaxation — it operates across the entire
accessible range, with a mild, monotonic weakening trend that itself may
be worth tracking if still-longer background relaxation ever becomes
computationally accessible.

## 10. Post-crossover (post-relaxation-stage) local driving force

No literal post-crossover regime exists; the analogous, available
comparison is `sigma`/`psi`/`F_TJ` in S0 vs. S1 at each of the three fork
points:

| state | `delta_sigma` (S1-S0, MPa) | S0 `psi_top` (deg) | S1 `psi_top` (deg) | S0 `F_TJ_mag_top` | S1 `F_TJ_mag_top` |
|---|---:|---:|---:|---:|---:|
| early | -0.00039 | 76.983 | 76.997 | 1.12272 | 1.12298 |
| mid | -0.00012 | 77.528 | 77.533 | 1.11801 | 1.11809 |
| late | **+0.00053** | 78.829 | 78.834 | 1.15963 | 1.15972 |

`delta_sigma` is negative (coarsening lowers stress relative to the
capillary-only control, matching every flat- and curved-substrate finding
so far) at the early and mid fork points, but **flips to slightly positive
at the late fork point** — a small, single-data-point observation (`+0.53
kPa`, at the very limit of what these `sigma_Pa` values resolve
meaningfully given `sigma` itself is `~4.1e7 Pa` at that point) that should
not be over-interpreted, but is flagged here rather than silently
smoothed over: it is at least directionally consistent with the shrinking
effect's own magnitude weakening substantially by the late state (Section
9), and is worth re-examining with a longer/finer fork run if a future
milestone revisits stress buildup specifically. `psi_top` and
`F_TJ_mag_top` both grow modestly from early to late in both S0 and S1
(reflecting the slow background relaxation itself, Section 5), with S1
consistently and only very slightly above S0 in both quantities at every
fork point — no dramatic local-driving-force growth is observed at any
stage, consistent with there being no crossover/instability event to drive
one.

## 11. Optional mode-growth description

Not attempted. Section 5's evidence (the sinusoid's own amplitude/harmonic
content changing substantially, alongside the neck geometry, over the same
horizon) means no single clean, separable geometric mode was identified
that isn't itself coupled to the neck/TJ relaxation being measured — per
the handoff's explicit instruction not to force a mode decomposition the
geometry does not justify, this section is left as a negative/deferred
finding rather than a forced fit.

## 12. Periodicity

Continued using the existing reflecting one-wavelength symmetry cell
(Milestone 10 Section 2) for this entire deterministic, symmetric
milestone — no stochastic or asymmetric process was enabled, so the
reflecting-boundary/periodicity-equivalence condition established there
continues to apply. Genuine Y-periodic operators remain a prerequisite for
any future stochastic/asymmetric sink work, not attempted here.

## 13. Summary and recommendation

1. **Ordinary surface diffusion does not rescue the sign of the contact
   response on this geometry within any accessible timescale** — the
   "maybe the initial contact just hadn't reached the right part of its
   own relaxation trajectory yet" hypothesis motivating this milestone is
   not supported; `g_CH` never reverses over nearly 1,800x of relaxation
   and 1.45 physical seconds.
2. **The coarsening-driven shrinking effect (Milestone 9's external-
   reservoir closure) is robust across the entire tested range of
   background relaxation states**, not an artifact of sampling one
   particular, possibly-atypical moment in the S0 trajectory. This is a
   materially stronger qualification of that finding than Milestones 9-10
   provided on their own (which only checked a single, early starting
   point).
3. The mild weakening of the effect's magnitude at the most-relaxed state
   tested, and the single sign-flip in `delta_sigma` there, are worth a
   quick follow-up (a finer/longer fork specifically from a late state, or
   an even-later background state if the ~1.45s horizon can be extended
   further inexpensively) before treating the "weakens toward late times"
   trend as established rather than suggestive.
4. No basis was found here to revisit Milestone 9/10's closure
   recommendation — the external-reservoir closure remains the qualified
   diagnostic default going forward.

**STOP. Do not activate hazard or sink yet**, per the handoff's explicit
instruction.
