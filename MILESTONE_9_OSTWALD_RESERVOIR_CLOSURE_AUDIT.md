# Ostwald receiver/reservoir closure audit

Scope: (1) correct a normalization bug in Milestone 8's fixed-eta-diffusivity
diagnostic; (2) determine whether the substrate ("grain 1") is intended as a
finite receiving particle or an effectively macroscopic/external reservoir;
(3) trace the Milestone 8 finding (Ostwald's destination-side addition
drives the wrong-sign widening) down to *where* the addition field's mass
matters, and test whether a globally mass-conserving external-reservoir
closure restores the physically expected direction. **Stopping here for
review, as instructed** — no production Ostwald/CH/eta/projection equation
was modified; all closures (Models B/C) are diagnostic-only alternatives,
compared but not defaulted to.

## 0. Headline result

**A globally mass-conserving external-reservoir closure restores the
physically correct direction, cleanly and without tuning.** Diverting the
Ostwald addition field's mass to an explicit external-reservoir variable
instead of redepositing it locally (Model B: `phi_local=0`) flips
`delta_L_coarsening` from Milestone 8's `+0.0054nm` (widening) to
**`-0.0113nm` (shrinking — the desired direction)** at `|dV2|/V20=3e-4`, and
from `+0.0314nm` to **`-0.0545nm`** at `1.5e-3`, in both cases while holding
global mass (`local_f + reservoir`) exactly constant (residual at
floating-point roundoff, `~1e-17` relative). The sign flip is not a
free parameter: `phi_local` was tested at 0.0, 0.1, and 0.5, and the sign
already flips to shrinking by `phi_local=0.5` — the crossover is not
delicately tuned to a specific small value.

**Section 6's spatial decomposition explains why**: partitioning the exact,
unrenormalized production addition field by distance to the nearest
continuous sub-grid TJ shows that **only the addition mass within ~1
interface width of a TJ has any measurable effect on `L_contact_TJ_sub`,
even after 276 steps** — every band beyond that (1W-3W, 3W-5W, `>5W`)
converges to essentially the *same* result as removal-only (O1)/Model B,
regardless of how much mass (23-38% each) each band actually carries. This
is Case **R1** (Section 8): the near-TJ addition band is what drives the
widening; far-field addition is dynamically inert on the timescales studied.

**Section 4's semantic audit** independently supports Model B over Model A
(current production): the substrate ("grain 1") is constructed as a
semi-infinite half-plane wall (`e1=0.5*(1-tanh((X-wall)/W))`) with **no
finite-particle radius parameter at all** — `R1` is loaded into `Params`
but never referenced anywhere in the substrate-geometry branch of
`initialize_fields`. There is no explicit statement anywhere in the model of
what larger physical receiver the local window represents a fraction of,
so **no principled `phi_local` can be derived from current model semantics**
(Section 10) — Model C's `phi_local` values tested here are illustrative
only, not physically derived.

Section 2's normalization bug is fixed; Section 13's corrected fixed-physics
grid check confirms Model A's sign is grid-stable (as Milestone 8 found),
and additionally shows **Model B's sign and magnitude are grid-stable to a
much tighter tolerance** (1.6% vs. Model A's 164% spread between dx=5nm and
dx=2.5nm) — a secondary point in Model B's favor.

## 1. Checkpoint / test state

- Branch `codex/coarsening-stress-buildup`, starting HEAD `cae5265`
  (Milestone 8), clean working tree, 112/112 tests passing — verified
  before any change.
- New this session: one bug-fix to `model.py`'s
  `eta_diffusivity_fixed_physical` construction (Section 2, replacing
  Milestone 8's buggy diagnostic, not adding a new one),
  `pf_sintering/ostwald_receiver_closures.py`,
  `scripts/ostwald_reservoir_closure_audit.py`, and two new test files.
- Full suite after all changes: **118/118 passing**.

## 2. Corrected eta fixed-physics normalization

Milestone 8's `eta_diffusivity_fixed_physical=True` dropped `M_eta`'s
`/dx**2` factor entirely (`M_eta = M_f_base*0.01*eta_mobility_scale`),
giving `M_eta*k_eta ~ 1.599e-33` at every resolution — dx-independent, but
**not** the qualified baseline's actual rate (`~6.396e-17` at the dx=5nm
production default), a ~4e16 reduction that effectively turned structural
relaxation off for any calculation using that diagnostic.

**Fixed** (`pf_sintering/model.py`, `_eta_diffusivity_reference`): compute
`D_eta_ref = M_eta*k_eta` at the qualified reference point (`dx=5nm`,
`W=20nm`, same `theta_mis_deg`/`eta_mobility_scale` as the actual config —
`gamma_s`/`tau_target` are fixed `Params` defaults, not configurable, so
they are automatically shared), then set `M_eta = D_eta_ref / k_eta` using
the *actual* `k_eta` (whatever `W` the current call uses). Since `D_eta_ref`
doesn't depend on the actual `dx`/`W`, this makes `M_eta*k_eta` equal to the
qualified baseline's rate **exactly**, at every resolution — verified to 9+
significant figures at dx=5/2.5/1.25nm (all three give `M_eta*k_eta =
6.395675783e-17`, matching the baseline to `test_eta_diffusivity_fixed_physical_preserves_qualified_dx5_baseline_rate`).
Also added the explicitly-requested empirical test: the same physical
smooth eta perturbation gives the same `d(eta)/dt` at dx=5nm and dx=2.5nm to
within 2% (pure `lap9` discretization error) —
`test_eta_diffusivity_fixed_physical_gives_equal_rate_on_same_physical_field`.
Production default (`eta_diffusivity_fixed_physical=False`) is completely
unchanged.

## 3. CH exoneration wording (corrected, no code change)

Milestone 8's Section 13 showed the **isotropic** discrete CH operator
exactly dissipates its own exact discrete free energy at every timestep
tested. The production model uses **anisotropic** CH by default, and that
functional was not independently re-derived or checked in Milestone 8. Per
this milestone's explicit instruction, the exoneration is stated at the
precision actually established, everywhere in this report and going
forward: *"No isotropic discrete-CH numerical inconsistency has been found,
and CH is not the dominant direct source of the paired widening"* — not
"CH is completely exonerated." Section 14 explains why the anisotropic
check is deferred rather than attempted this session.

## 4. Receiver/substrate semantic audit

**The substrate is architecturally a semi-infinite half-plane, not a finite
particle.** `initialize_fields`'s substrate branch:

```python
wall = (p.substrate_wall_frac - .5) * p.Nx * p.dx
e1 = .5 * (1 - np.tanh((X - wall) / W))
```

`e1` is a smooth step function of `X` alone, extending uniformly across the
full domain height and to the domain's negative-`X` edge — there is no
radius, no curvature, no finite extent parameter anywhere in this
construction. Contrast with the `threeparticle` branch, which builds every
grain (including grain 1) from an explicit `R1` via
`.5*(1-np.tanh((np.hypot(X-c,Y)-r)/W))` — a genuine finite circular particle.
**`R1` is loaded into `Params` (`p.R1=r1`, default 100nm for the `dev`
preset) but is referenced nowhere in the substrate branch of
`initialize_fields`** (confirmed by grep — the only other use of `R1` is
`p.GS1=r1+r2`, an unrelated legacy grain-size bookkeeping term feeding
`Stress.GS`/hazard, not geometry). Domain sizing (`nx`, `wall_room`,
`margin`) is derived entirely from `r2`/`aspect_ratio`; `r1` plays no role.

This matches `PHYSICS_BACKGROUND.md` Section 7's own language: Ostwald
"changes the net amount of a shrinking grain by moving mass to or from an
**external/remote reservoir**," and "the reservoir should ideally specify
the *net mass/chemical-potential exchange*, while local surface evolution
determines where the surface recedes" — text written before this audit,
already describing the intended semantics as an external reservoir, not a
finite locally-complete receiver. `README.md` corroborates: the substrate
path is "the current production target" (distinct from the `threeparticle`
path, whose "complete two-GB kinetics" are explicitly *not yet* production-
ready) — i.e. the substrate was never meant to need the same finite-particle
completeness a two/three-particle geometry would.

**Conclusion**: grain 1/the substrate is intended as an effectively
macroscopic/external reservoir of which the computational window shows only
a local slice, not a finite receiving particle whose entire surface is
represented. Section 11 revisits this given the closure results.

## 5. Mass-closure accounting

One ordinary Ostwald step, primary geometry, uncapped (`diag.capped=False`):

| quantity | value |
|---|---:|
| mass removed from grain 2 | `8.6837e-04` (sum units) = `2.1709e-20 m^2` |
| mass added to grain 1 (local) | `8.6837e-04` = `2.1709e-20 m^2` (exactly equal — uncapped) |
| change in total local `f` | `-1.598e-31 m^2` (floating-point roundoff — exactly conserved) |
| receiving (`e1`) free-surface contour length | `355.00nm` |
| particle (`e2`) free-surface contour length | `513.28nm` |
| mean normal advance on receiver, spread over its **full** contour | `6.115e-05nm` |
| mean normal recession on particle, spread over its **full** contour | `-4.230e-05nm` |

Spread uniformly over each field's own *entire* represented contour, the
receiver's mean advance and the particle's mean recession are the **same
order of magnitude** (`6.1e-5nm` vs. `4.2e-5nm`) — a naive whole-contour
mass-closure comparison does **not** by itself reveal an extreme size
mismatch. The real effect is not a *bulk* over-concentration onto too small
a total receiving length; it is a **spatial** one (Section 6): most of the
locally-deposited mass is not where the whole-contour average implies, it
is concentrated in specific places relative to the interface geometry.
Section 6 supplies the actual spatial breakdown; Section 5's whole-contour
average was a necessary check (ruling out a simpler "receiver-too-small"
explanation) but not the full story.

## 6. Spatial partial-addition decomposition (trajectory-level, N_ref=276)

The **exact** production `add(x,y)` field (from `ostwald_substrate_diagnostic`,
never recomputed or renormalized), partitioned by distance to the nearest
continuous sub-grid TJ, applied region-by-region for the full 276-step
primary trajectory (omitted mass sent to an external-reservoir bookkeeping
variable, never redeposited):

| region | mass fraction (one-step) | `delta_L_coarsening` (nm) |
|---|---:|---:|
| 0W-1W (nearest TJ) | 5.8% | **+0.005546** |
| 1W-3W | 23.7% | **-0.011440** |
| 3W-5W | 32.6% | **-0.011282** |
| `>5W` | 37.9% | **-0.011282** |

Only the 0W-1W band reproduces the production sign (widening, and even
slightly exceeds O0's `+0.005388nm`); **every band beyond ~1W converges to
essentially the same value as O1/Model B** (`-0.011284nm`,
Section 12) regardless of carrying anywhere from 24% to 38% of the total
added mass. This is not a smoke-test coincidence — bands 3W-5W and `>5W`
match to 4 significant figures after 276 real production steps.

## 7. One-step differential test and its search-radius caveat

Applying each variant **once** to the initial state (no subsequent CH/eta/
projection) gives the same qualitative pattern at much smaller magnitude
(`~1e-5nm`, appropriate for one step vs. 276): `region_0W_1W` matches
`O0_normal` exactly (`+1.623e-05nm`); `region_1W_3W`, `region_3W_5W`, and
`region_>5W` all match `O1_removal_only` exactly (`-4.053e-05nm`).

**Important methodological caveat, checked directly**: `locate_neck_tjs_subgrid`'s
own candidate-intersection search radius for this geometry is
`min(3.0*interface_width, 0.4*neck_height) = min(60nm, 24nm) = 24nm = 1.2W`
(computed directly from `tj_subgrid.py`'s own formula for this state). A
perturbation entirely outside that radius cannot be seen by a **single**
call to the locator, by construction — so the one-step test is *structurally
biased* toward attributing effect only to addition within ~1.2W, independent
of the underlying physics. This is why Section 6's **trajectory-level**
result (276 steps, during which the TJ is relocated fresh every step from
the evolving field) is the decisive evidence, not the one-step test alone:
if far-field addition mattered on any timescale relevant to this diagnostic
horizon, 276 steps of subsequent CH/eta relaxation would have had ample
opportunity to transport its effect back into the locator's window, and it
did not (Section 6's 3W-5W and `>5W` bands remain locked to O1's value to 4
significant figures). The one-step test is retained here as a fast,
cheap confirmation of *what happens immediately*, not as the primary
evidence for the spatial conclusion.

## 8. R1 vs. R2 classification

**Case R1 (addition near the TJs dominates)** — decisively, per Sections 6-7.
Material redeposited within about one interface width of a TJ directly and
measurably widens the contact; material redeposited farther away, even
carrying the majority of the transferred mass, has no measurable effect on
`L_contact_TJ_sub` over the entire diagnostic horizon tested (up to `|dV2|/V20
=3e-4`, 276 steps). This is **not** case R2 (far-surface addition
reconstructing the field in a way that advances the TJ without local
deposition) — the far bands are dynamically inert for this metric, not
merely indirect.

## 9. External-reservoir mass-conservation test

Model B (`phi_local=0`, all addition diverted), 50-step primary-case run:

| | value |
|---|---:|
| `local_f`: initial → final | `7.30077576e-14` → `7.30066722e-14` (`m^2`) |
| change in `local_f` | `-1.085435e-18 m^2` |
| `M_external_reservoir`: 0 → final | `+1.085435e-18 m^2` |
| combined residual (`local_f + reservoir - initial`) | max `2.524e-29`, i.e. `3.5e-16` relative |

`M_local_solid + M_external_reservoir` is conserved to floating-point
roundoff throughout — the closure is genuinely globally mass-conserving,
not merely locally checked at the endpoints. No empirical source/sink is
introduced; `reservoir_delta` is exactly the mass the production kernel's
own `diag.add` field would have deposited, simply not deposited.

## 10. Physically scaled local-receiver test (Model C)

Per Section 4, **no explicit statement exists anywhere in the current model
of what larger physical receiver volume/surface the computational window
represents a fraction of** — the substrate has no finite size parameter to
take a fraction of (Section 4). Per the handoff's explicit instruction, no
`phi_local` is invented; the values tested here (`0.1`, `0.5`) are
**illustrative only**, to characterize the trend, not claims about the
physically correct fraction:

| `|dV2|/V20` target | Model A (`phi=1.0`) | Model C (`phi=0.1`) | Model C (`phi=0.5`) | Model B (`phi=0.0`) |
|---|---:|---:|---:|---:|
| 3e-4 | +0.005388nm | -0.009614nm | -0.002949nm | -0.011284nm |
| 1.5e-3 | +0.031352nm | -0.045929nm | -0.011531nm | -0.054533nm |

The sign flips well before `phi_local` reaches its physically-motivated
lower bound of 0 — already negative (shrinking) at `phi_local=0.5` at both
targets. **If a genuine receiver-geometry fraction were ever derived, it
would not need to be small for the qualitative conclusion (external
mass loss shrinks the contact) to hold** — but deriving that fraction
requires an explicit receiver geometry the current substrate model does not
provide (Section 11).

## 11. Is grain 1 a finite particle?

No — Section 4 established this architecturally (no `R1`-derived geometry
in the substrate branch), and Section 9/10/12 are consistent with treating
it as an open/external system: diverting addition mass entirely to an
external reservoir (Model B) is the closure that restores the experimentally
expected direction, and does so without requiring any locally-represented
receiver surface to "absorb" it. **If a future explicit two/three-particle
geometry is used instead** (where grain 1 genuinely is a finite, fully-
represented particle — the `threeparticle` path `README.md` already flags
as not yet production-ready), the correct closure there is a separate
question: full local redeposition (Model A) may be appropriate *for that
geometry*, since the entire receiver surface and its curvature would then
actually be represented. The finding here is specific to the **substrate**
topology as currently constructed, not a universal statement that local
redeposition is always wrong.

## 12. Paired trajectory comparison of closures

Full paired C0/C1 runs (same initial state, same grain-2 removal physics,
matched physical time), primary 20nm-overlap geometry:

| target | closure | n_steps | `delta_L_coarsening` | `delta_L_GB_geom_sub` | `delta_sigma` | reservoir | mass residual |
|---|---|---:|---:|---:|---:|---:|---:|
| 3e-4 | A (`phi=1.0`) | 276 | +0.005388nm | +0.006624nm | -0.008796MPa | 0 | -2.5e-29 |
| 3e-4 | B (`phi=0.0`) | 276 | **-0.011284nm** | -0.011269nm | -0.000115MPa | 5.99e-18 | (exact) |
| 3e-4 | C (`phi=0.1`) | 276 | -0.009614nm | -0.009478nm | -0.000982MPa | 5.39e-18 | (exact) |
| 3e-4 | C (`phi=0.5`) | 276 | -0.002949nm | -0.002324nm | -0.004454MPa | 3.00e-18 | (exact) |
| 1.5e-3 | A | 1378 | +0.031352nm | +0.038307nm | -0.043612MPa | 0 | -1.0e-28 |
| 1.5e-3 | B | 1378 | **-0.054533nm** | -0.054395nm | +0.000670MPa | 2.99e-17 | (exact) |
| 1.5e-3 | C (`phi=0.1`) | 1378 | -0.045929nm | -0.045113nm | -0.003740MPa | 2.69e-17 | (exact) |
| 1.5e-3 | C (`phi=0.5`) | 1378 | -0.011531nm | -0.007993nm | -0.021416MPa | 1.49e-17 | (exact) |

**Success criteria** (Section 12 of the handoff): `V2` decreases in every
case (by construction, all share the same grain-2 removal physics); no RBM
anywhere (sink/RBM never called, unchanged from every prior milestone); and
**the physical TJ-to-TJ contact decreases relative to C0** — satisfied by
Models B and C at both targets and both `phi_local` values tested, without
tuning any hazard or rate parameter.

## 13. Corrected fixed-physics grid check

Same `interface_width_override=20nm`, `eta_diffusivity_fixed_physical=True`
(now correctly preserving the baseline rate), independently dt-checked at
each grid (dx=5nm at natural dt, dx=2.5nm at dt/4, matching Milestone 8's
converged choices — unaffected by the Section 2 fix, since `eta_raw`'s
contribution to `delta_L_coarsening` was already established as ~0.04% of
the total in Milestone 8):

| closure | dx=5nm | dx=2.5nm | ratio |
|---|---:|---:|---:|
| Model A (production) | +0.005388nm | +0.014202nm | 2.64x |
| Model B (external reservoir) | -0.011284nm | -0.011468nm | **1.02x** |

**Both closures keep the same sign under mesh refinement** (Q7). Model B is
additionally far better resolution-converged than Model A (1.6% magnitude
spread vs. 164%) — an incidental but notable secondary point in Model B's
favor: removing the locally-redeposited addition term also removes a
source of resolution-sensitive spatial redistribution.

## 14. Anisotropic energy check — deferred

Per the handoff's explicit permission ("If it becomes a major project,
defer it. It is no longer the leading explanation for the wrong-sign
geometry"): deferred this session. The reservoir-closure result (Sections
6-13) is now the leading, well-evidenced explanation, and re-deriving the
anisotropic Kobayashi-type discrete functional (`ch_exact_energy.py`'s
isotropic derivation does not generalize trivially — the anisotropic
gradient term's own `a(theta)`/`a'(theta)` dependence on the field itself
introduces additional variational cross-terms not present in the isotropic
case) would be a substantial undertaking disproportionate to its current
priority. Section 3's corrected wording stands in its place.

## 15. Q1-Q7

**Q1. Why does addition-only widen the physical contact?** Because a
measurable fraction of the redeposited mass lands within about one
interface width of a TJ (Section 6: 5.8% of the one-step total, and this
band is the only one that shows any lasting trajectory-level effect),
directly growing the receiving grain right at the contact.

**Q2. Which spatial region of `add(x,y)` produces the widening?** The 0W-1W
band (nearest either TJ). Every band beyond ~1W is dynamically inert for
`L_contact_TJ_sub` over the full 276-step diagnostic horizon (Section 6-8).

**Q3. Does the operator force 100% of grain-2 mass loss into a locally
represented receiver whose physical size is inconsistent with that
normalization?** Not exactly as a bulk-size mismatch (Section 5: whole-
contour mean advance/recession are comparable order of magnitude) — but yes
as a **spatial concentration** mismatch: the receiver has no explicit
finite geometry (Section 4) to justify redepositing *any* fixed fraction
locally, let alone concentrating enough of it within 1W of the TJ to
directly widen the contact.

**Q4. Is the local domain intended to be closed or open w.r.t. Ostwald mass
transport?** Open (Section 4): the substrate is architecturally semi-
infinite with no finite receiver geometry represented; `PHYSICS_BACKGROUND.md`
independently describes Ostwald as exchanging with an "external/remote
reservoir."

**Q5. Does external-reservoir bookkeeping restore V2 down + L_contact down
without an artificial rule?** Yes (Sections 9, 12) — Model B (`phi_local=0`,
the direct, untuned open-system limit) gives both simultaneously at both
targets tested, with global mass conserved to floating-point roundoff.

**Q6. If the receiver should instead be finite/explicit, what geometry is
needed?** An explicit finite grain-1 geometry (radius, curvature, its own
free surface) analogous to grain 2's — i.e. the `threeparticle`/two-particle
topology `README.md` already flags as not yet production-ready, not the
current substrate construction.

**Q7. Is the direction stable under mesh refinement after the eta-rate
correction?** Yes for both Model A and Model B (Section 13); Model B is
additionally far tighter in magnitude convergence.

## 16. Recommended smallest physically defensible model change (NOT implemented)

Given Sections 6-13, the most direct, well-evidenced next step is to
introduce an **explicit external-reservoir bookkeeping variable** into the
production Ostwald operator's diagnostic-adjacent accounting — not by
deleting the addition term, but by making *where the removed mass goes* an
explicit, examined choice rather than an implicit "100% back onto whatever
of grain 1 is visible in this window." Concretely: adopt Model B
(`phi_local=0`, all addition diverted to `M_external_reservoir`) as the next
diagnostic default for the sink-off directional campaign specifically —
it is the only closure tested that (a) restores the experimentally required
`dV2<0`/`L_contact` down direction, (b) conserves mass globally without any
new empirical rule, and (c) is better resolution-converged than production.
Before promoting it further: confirm the direction survives the same
extended-trajectory and coarsening-rate-series checks Milestone 7 ran for
the original (wrong-sign) case, and revisit whether `phi_local=0` is too
extreme once/if an explicit finite receiver geometry (Q6) is ever built —
until then, `phi_local=0` is the physically defensible default precisely
*because* no finite receiver is represented, not because it was chosen to
produce the desired sign.

**STOP. Do not alter production physics before review.**
