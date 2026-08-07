# Physical contact-geometry audit

Scope: measure the ACTUAL physical solid-solid contact (not the eta-only proxy)
through each operator, for the exact Milestone-6 primary case plus the
baseline-geometry cross-check. **Stopping here for review, as instructed** —
no production physics was modified.

## 0. Headline result

The new physical metrics revealed a real and important limitation before
they could answer the intended question: **`L_contact_TJ` (and, mostly,
`L_GB_geom`) are anchored to `tj_force.locate_neck_tjs`'s discrete,
integer-grid-indexed TJ rows, so they can only change in units of roughly
one grid cell (`dx = 5nm`) — never continuously.** At the `|dV2|/V20 ~ 3e-4`
scale used throughout this investigation, that quantization step is
**~250x larger** than the entire eta-metric signal (`x_neck_eta` totals
`0.019–0.059nm` over the same window). Once that was understood and
explicitly separated out (Section 6), a consistent, if statistically thin,
picture emerged: every resolvable discrete change in the physical contact
length was in the **widening** direction, matching the sign of the
eta-based `x_neck_eta` metric — corroborating, not contradicting, Milestone
6's conclusion, but not on the strength originally hoped for from these new
metrics.

## 1. Definitions of every new metric

All implemented in `pf_sintering/contact_geometry.py`, reusing
`tj_force.locate_neck_tjs` / `tj_force.compute_neck_tj_forces` without
modification (see that module's docstring for the full derivation):

- **`t_GB`, `n_GB`** (mean GB tangent/normal): built from the two TJs' own
  `v_gb` (each already validated to point away from its TJ into the bulk
  GB). Since the two point roughly toward each other, `t_GB =
  normalize(v_gb_bottom - v_gb_top)` combines them into one direction;
  `n_GB` is its 90-degree rotation. Not assumed vertical — measured from the
  actual resolved GB branches every time.
- **`L_contact_TJ = abs(dot(r_top - r_bottom, t_GB))`**: physical TJ-to-TJ
  distance projected onto the local mean GB tangent.
- **`d_n_TJ = abs(dot(r_top - r_bottom, n_GB))`**: TJ separation
  perpendicular to the mean tangent (a genuine diagnostic, not assumed
  zero — measured at `~1.6e-23 m` for this near-vertical geometry, i.e.
  effectively zero, but *measured* to be so, not hard-coded).
- **`L_GB_geom`**: arc length of the connected `e1=e2` contour segment
  (traced with `skimage.measure.find_contours` on `e1-e2`, masked to
  `f>0.5` so it depends on both grain identity and physical solid
  occupancy, *not* `integral(e1*e2)`) between the two TJs.
- **`L_f_farfield`**: x-position (same centered convention as
  `model.center()`/`diagnostics.wall_x0()`) of the farthest point of the
  `f=0.5` contour from the wall, minus `wall_x0` — tracks the tip of the
  particle cap, which by construction of this geometry is always far
  (`>150nm`, dozens of interface widths) from the neck, so local TJ
  reshaping cannot masquerade as a change here.
- **`L_eta_centroid`**: unchanged from the existing `center(e2,p) -
  wall_x0` separation diagnostic, reused under this name for direct
  comparison against `L_f_farfield`.
- **`V2_f_eta = integral(f * e2/(e1+e2+e3+eps))`**: solid-occupancy-weighted
  grain-2 volume; differs from the legacy `V2_eta = integral(e2)` whenever
  `e1+e2+e3 != f` locally.

`x_neck_eta`, `A_GB_eta`, `V2_eta` are the *unchanged* legacy metrics
(`model.contact_width`, `diagnostics.contact_area`, `integral(e2)`),
explicitly relabeled here for contrast — none of their code was touched.

Test coverage: `tests/test_contact_geometry.py` (7 tests) validates
`L_contact_TJ` and `L_GB_geom` agree to within 2% as two independently
derived measures, `d_n_TJ` is small-but-measured, `t_GB`/`n_GB` are unit and
orthogonal, `L_f_farfield` is farther from the wall than `L_eta_centroid`,
and `V2_f_weighted` reduces exactly to `V2_eta` when `e1+e2=f` exactly
(synthetic check) while differing measurably for the real analytic
initializer.

## 2. Operator ledger for `L_contact_TJ` and `L_GB_geom`

Primary case (unchanged from Milestone 6: `overlap=20nm, coarsening=3x,
surface=0.3x, eta=1x`, `276` steps, `dV2/V20=-3.006e-4`, closure exact
(`0.0`) for every quantity as before):

| operator | d_L_contact_TJ (nm) | d_L_GB_geom (nm) | d_x_neck_eta (nm) |
|---|---:|---:|---:|
| CH | **+4.995294** | **+5.070187** | +0.000000 |
| post-CH projection | +0.000015 | +0.001928 | +0.006502 |
| Ostwald | +0.000003 | +0.000380 | +0.001702 |
| eta relaxation (raw) | -0.000004 | +0.000291 | +0.008126 |
| post-eta projection | -0.000013 | +0.000240 | +0.002474 |
| **total** | **+4.995295** | **+5.073026** | **+0.018805** |

Baseline cross-check (`overlap=5nm`, same rates, `276` steps,
`dV2/V20=-3.006e-4`, closure exact):

| operator | d_L_contact_TJ (nm) | d_L_GB_geom (nm) | d_x_neck_eta (nm) |
|---|---:|---:|---:|
| CH | **+9.987231** | **+5.771125** | +0.000000 |
| post-CH projection | -0.000050 | +0.009068 | +0.032945 |
| Ostwald | +0.000002 | -0.000675 | +0.001828 |
| eta relaxation (raw) | -0.000003 | -0.002736 | +0.007661 |
| post-eta projection | -0.000005 | +0.000662 | +0.016113 |
| **total** | **+9.987174** | **+5.777444** | **+0.058547** |

At face value, `CH` appears to dominate `L_contact_TJ`/`L_GB_geom` by
~1000x over every other operator combined. Section 6 shows why this
reading is misleading and how to correct it.

## 3. TJ coordinates through the trajectory

Primary case: `tj_top` stayed at `(150.0nm, 350.0nm)` for the first 103
steps, then **snapped in a single step (step 104, within the `CH` stage)
to `(150.0nm, 355.0nm)`** — an exact `+1*dx` (`5nm`) change in the y
coordinate — and stayed there for the remaining 172 steps. `tj_bottom`
did not move at all over this window (`290.0nm` throughout).
`x_neck_eta` was essentially unchanged across that same step
(`41.4816nm -> 41.4816nm`), confirming the jump is a discrete relocation
of the *detected* TJ row, not a visible change in the continuous
eta-based neck measure. Baseline case: two such single-cell jumps
(steps 122 and 242), both also confined to the `CH` stage, both positive.

## 4. Comparison against `x_neck_eta` and `A_GB_eta`

`x_neck_eta`'s total (`+0.019nm` primary, `+0.059nm` baseline) is
**2-3 orders of magnitude smaller** than `L_contact_TJ`'s raw total
(`+5.0nm` / `+10.0nm`). Once the jump-dominated `CH` row is set aside
(Section 6), the *other four* operators' `L_contact_TJ` contributions are
all `<1e-4 nm` — i.e. below what this discrete-TJ metric can resolve at
all, while the *same* four operators' `x_neck_eta` contributions are
`0.0004-0.033nm` — clearly resolved, non-zero, and (with one exception,
baseline `post_CH_projection` being the largest single contributor)
consistently positive. **The two families of metrics are not simply
disagreeing — they are operating at genuinely different resolutions**:
`x_neck_eta` (a continuous, area/peak-ratio-based measure) resolves
sub-grid-cell change that the discrete-TJ metrics cannot see at all in
this regime.

## 5. `L_eta_centroid` versus `L_f_farfield`

| | primary total (nm) | baseline total (nm) |
|---|---:|---:|
| `L_eta_centroid` | -0.0676 | -0.0803 |
| `L_f_farfield` | +0.1028 | +0.1040 |

These two **disagree in sign**. `L_eta_centroid` (mass-weighted centroid of
`e2`) drifts slightly *toward* the wall (negative), matching the small
shape-driven drift already documented in `MILESTONE_1_2_REPORT.md`.
`L_f_farfield` (position of the particle-cap tip, far from the neck) drifts
slightly *away* from the wall (positive) by a comparable magnitude. Both are
tiny (`<0.11nm` over this window) and both are consistent with **shape
redistribution, not rigid-body translation**: `strain = 0` exactly
throughout (confirmed in every `stage_samples` entry, as in all prior
sink-off audits), so neither drift can be RBM. The sign difference itself is
informative: the particle's *centroid* moves one way while its *far tip*
moves the other, meaning the particle's mass distribution is becoming
asymmetric (more material redistributing toward the near/neck side, pulling
the centroid inward, while the cap tip itself edges slightly outward) rather
than the whole shape translating together. This directly demonstrates why
`L_eta_centroid` alone (the metric used throughout Milestones 1-5) is not a
rigorous rigid-body diagnostic on its own — it conflates translation with
this kind of internal redistribution, exactly as the handoff anticipated.

## 6. `V2_eta` versus `V2_f_eta`

| | V2_eta total | V2_f_eta total | ratio |
|---|---:|---:|---:|
| primary | -5.990e-18 | -5.696e-18 | 0.951 |
| baseline | -6.229e-18 | -5.353e-18 | 0.859 |

Both track the same coarsening direction (both negative, both attributed
almost entirely to `Ostwald` per the Milestone-6 ledger), but `V2_f_eta`'s
magnitude is `86-95%` of `V2_eta`'s — the solid-occupancy-weighted measure
changes *less* than the raw eta sum. This means part of what looks like
"grain-2 volume loss" in `V2_eta` is being compensated by a shift in how
much of the *existing* solid capacity (`f`) is fractionally attributed to
grain 2 via `q2 = e2/(e1+e2+e3)`) versus left as unclaimed slack or shifted
to grain 1 — i.e. exactly preserving `integral(e2)` does require some
redistribution of *fractional* ownership that a purely mass-conserving
check cannot see. This is consistent with, and adds a second independent
line of evidence for, Section 7's finding that the mass-preserving
projection does real geometric (not just bookkeeping) work.

## 7. What raw CH physically does to the contact

**Ambiguous at the resolution these specific metrics provide, but leaning
toward Outcome B/D, not Outcome A or C.** The literal ledger entries
(Section 2) show CH producing a large, positive change in both
`L_contact_TJ` and `L_GB_geom` — but Section 3 showed this is a single
discrete `+1*dx` grid-quantization jump, not a smoothly resolved trend.
De-quantizing (excluding any single-step delta `>= 0.5*dx`, `runs/contact_geometry_audit.json`,
`scripts/contact_geometry_audit.py::dequantized_totals`):

| geometry | dequantized CH `d_L_contact_TJ` (nm) | dequantized CH `d_L_GB_geom` (nm) |
|---|---:|---:|
| primary | **0.000000** (0/276 non-jump steps show any change) | **0.000000** |
| baseline | **0.000000** | **+0.751699** (277 non-jump steps) |

In the primary geometry, raw CH's effect on the physical contact is
**exactly zero except for that one grid-cell jump** — the discrete-TJ
metric literally cannot resolve anything smaller. In the baseline geometry,
`L_contact_TJ` is likewise exactly zero outside its jumps, but `L_GB_geom`
(which also incorporates the *contour path shape* between the two TJs, not
just their endpoint separation) shows a real, positive, non-jump residual
of `+0.75nm` — smooth evidence, independent of the quantization artifact,
that raw CH's continuous component *also* widens the physical GB contour in
that geometry.

**A longer supplementary run** (primary geometry, extended to
`|dV2|/V20~1.5e-3`, `1378` steps, not part of the required Milestone-6-scope
ledger but run to get better jump statistics) found exactly **2** jump
events, both attributed to `CH`, both positive: `+4.995nm` (step 104) and
`+4.983nm` (step 511). Combined with baseline's 2 CH-attributed jumps
(`+4.997nm`, `+4.990nm`), **all 4 discrete jump events observed across both
geometries and both trajectory lengths are positive (widening); none are
negative.** With only 4 events this is not strong statistical evidence, but
it is 4-for-4 in the same direction as `x_neck_eta`'s conclusion, and zero
evidence of the opposite sign.

**Conclusion for this section**: within the resolution these metrics
actually provide, raw CH shows no evidence of *shrinking* the physical
contact (Outcome A/C are not supported); what discrete and smooth signal is
resolvable is consistently in the *widening* direction (weak support for
Outcome B), and in the primary geometry specifically, `L_contact_TJ`/`L_GB_geom`
are so close to perfectly quantization-locked that most of Milestone 6's
eta-based widening signal there is, by construction of *these* metrics,
invisible to them (partial support for Outcome D's letter, though not
necessarily its full interpretation -- see Section 9).

## 8. What projection does afterward

Both projection steps (`post-CH` and `post-eta`) show `L_contact_TJ`
contributions below `1e-4 nm` in both geometries (no jumps, nothing to
de-quantize) — i.e. **unresolvable** at this metric's precision, neither
confirming nor contradicting a physical effect. For `L_GB_geom`, both
projections show small but nonzero, geometry-dependent contributions
(`post-CH`: `+0.0019nm` primary / `+0.0091nm` baseline; `post-eta`:
`+0.0002nm` primary / `+0.0007nm` baseline) — small and consistently
positive, but two orders of magnitude below their `x_neck_eta` counterparts
(`+0.0065-0.0329nm` / `+0.0025-0.0161nm`). The mass-preserving projection is
not obviously pushing the physical contact geometry in the *opposite*
direction from what `x_neck_eta` reports (Q7/B in the handoff is not
supported here) — but its effect on the metrics that can actually resolve
it (`x_neck_eta`, and `V2_f_eta` per Section 6) remains real and, per
Milestone 6, substantial.

## 9. What eta relaxation does afterward

`eta_raw`'s `L_contact_TJ` contribution is effectively zero in both
geometries (`-0.000004nm` primary, `-0.000003nm` baseline — sign is
negative but far below the `~dx` resolution floor, not a meaningful
narrowing signal). Its `L_GB_geom` contribution is small and
**geometry-dependent in sign**: `+0.0003nm` (primary, widening) versus
`-0.0027nm` (baseline, narrowing) — the only operator whose sign flips
between the two geometries for this metric. Given Milestone 6 attributed
`43.2%` of the *legacy* `x_neck_eta` widening to `eta_raw` in the primary
geometry (the single largest contributor there), this new metric's
near-total insensitivity to `eta_raw` is notable: **eta relaxation appears
to act primarily by broadening/redistributing the diffuse `e1`/`e2`
overlap band (which `x_neck_eta`'s area/peak-ratio measure is sensitive to)
rather than by moving the discrete TJ locations or the traced GB contour
length in a consistent way (which `L_contact_TJ`/`L_GB_geom` are sensitive
to).** That is a real, physically meaningful distinction, not just more
resolution noise -- it says eta relaxation's contribution to the legacy
metric's widening is more about the diffuse-interface bookkeeping than
about the physical TJ positions moving.

## 10. Outcome classification

None of the four outcomes as literally stated fit cleanly; the honest
classification is a qualified version of **Outcome D, with an important
caveat**, plus elements of **B**:

- **Not Outcome A or C**: no evidence anywhere (jump or dequantized) of raw
  CH *shrinking* the physical contact.
- **Partial Outcome D** ("`L_contact_TJ` and `L_GB_geom` are nearly
  stationary while `x_neck_eta` grows substantially"): true at face value
  for the primary geometry's non-CH operators and for `L_contact_TJ`
  everywhere — but the *reason* is not necessarily that the physical
  contact is truly unchanged; it is that these two metrics are
  grid-quantized at a resolution `~250x` coarser than the signal being
  measured, and are therefore simply **blind** to it rather than
  positively confirming its absence.
- **Weak Outcome B**: every one of the 4 observed discrete jump events
  (both geometries, both trajectory lengths) was positive (widening), and
  baseline's dequantized `L_GB_geom` residual was also positive
  (`+0.75nm`) — consistent with, not contradictory to, `x_neck_eta`'s
  conclusion that raw coarsening drives the neck toward widening rather
  than narrowing, though the statistical strength of this evidence (n=4
  discrete events plus one smooth residual) is modest.

**Overall**: this audit did not find evidence that Milestone 6's
conclusion (natural coarsening widens the neck under the tested rate
regime) is a pure eta-bookkeeping artifact. It also did not (yet) obtain a
high-resolution, fully independent physical confirmation of the same
magnitude/timing, because the physical-contact metrics built directly on
the discrete TJ locator are not fine-grained enough at this `|dV2|/V20`
scale to do so on their own.

## 11. Smallest physically defensible next step (not implemented)

The clean fix is **not** a change to production physics but a
**resolution upgrade to the diagnostic**: extend `tj_force.locate_neck_tjs`
(or add a companion function alongside it, without modifying the validated
original) to report a **sub-grid-interpolated** TJ row position — e.g.
linearly interpolate the row index where the column's `f` profile crosses
`0.5`, rather than snapping to the nearest integer row via
`np.flatnonzero(f[:,nc]>0.5)`. That single change would let
`L_contact_TJ`/`L_GB_geom` resolve continuous, sub-`dx` motion the same way
`x_neck_eta` already does, and would let Section 7's question be answered
at full strength rather than qualified by a resolution floor. This is
recommended for review, not started.

---

**Stopping here for review.** No production CH, Ostwald, eta, or
projection physics was modified; `contact_geometry.py` and the extended
ledger (`operator_ledger.run_ledger_trajectory_v2`) are additive
diagnostics that reuse, and do not alter, every previously validated
module.
