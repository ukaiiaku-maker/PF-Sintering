# Continuous sub-grid TJ tracking and resolved contact evolution

Scope: replace the discrete, grid-quantized `L_contact_TJ` / `L_GB_geom` from
Milestone 6B with a genuinely continuous, sub-grid-in-both-axes TJ locator,
and use it to answer the standing question — does the actual physical
solid-solid contact shrink or widen under raw sink-off Cahn-Hilliard (CH)
coarsening? **Stopping here for review, as instructed** — no production
physics was modified.

## 0. Headline result

With grid quantization removed, the answer is unambiguous and **the
opposite of what would resolve the original problem**: **raw CH physically
widens the solid-solid contact**, not just the eta-derived proxy metric.
The signal is smooth (no cell-sized jumps anywhere), overwhelmingly
CH-dominated (CH's contribution is ~55–95x every other single operator's,
and always the same sign), and **sign-stable** across three run lengths
(`|dV2|/V20` = 3e-4, 1.5e-3, 3e-3), a 2x finer grid (dx = 2.5nm vs 5nm), and
a much narrower starting geometry (5nm vs 20nm overlap). This is
**Outcome B**: conserved CH capillary evolution itself favors widening for
this state, at least at these coarsening/mobility rate settings — it is not
an artifact of the eta/projection representation. See Section 10.

A secondary finding sharpens Milestone 6B's suspicion about the eta proxy:
the `eta_raw` operator moves `L_contact_TJ_sub` and `x_neck_eta` in
**opposite directions** at every stage checked. `eta_raw` growing
`x_neck_eta` is largely diffuse-interface bookkeeping, not real TJ
separation — see Section 8.

## 1. The 2-D level-set intersection algorithm

Implemented in `pf_sintering/tj_subgrid.py`, new file, does not modify or
replace `tj_force.locate_neck_tjs` (kept as-is, used only as a seed/contrast).
A TJ is defined as the point satisfying **both** `f(x,y) = 0.5` and
`g(x,y) = e1(x,y) - e2(x,y) = 0` simultaneously — a true 2-D intersection of
two level-set contours, not a 1-D interpolation along a fixed integer row
or column.

1. **Seed neighborhood**: call the legacy `locate_neck_tjs` once per (f, e1,
   e2) frame purely to get an integer-grid TJ estimate and a `neck_height`
   scale — used only to size the local search window, never as the answer.
2. **Sub-pixel contours**: trace `f = 0.5` and `e1 - e2 = 0` with
   `skimage.measure.find_contours` (marching squares — sub-pixel-accurate
   polylines via linear interpolation along grid edges), converted to the
   module's uncentered `(index+1)*dx` coordinate convention.
3. **Candidate intersections**: keep only the polyline segments within
   `search_radius` of the legacy seed; compute all pairwise segment×segment
   intersections between the two contour families (`_segment_intersection`).
4. **Near-miss fallback**: if no segment pair literally crosses (the true
   intersection can sit close to a grid vertex where two independently
   traced polylines pass near but not through each other), fall back to the
   closest-approach point between the two local polyline sets
   (`_closest_approach`), but only if the gap is < 0.5·dx — otherwise reject
   rather than guess (Section 3, finding c).
5. **Newton refinement**: starting from the candidate, solve
   `F1 = f_interp(x,y) - 0.5 = 0`, `F2 = e1_interp(x,y) - e2_interp(x,y) = 0`
   with bilinear interpolation (`tj_force._sample_bilinear`) and a
   finite-difference Jacobian using a **0.02·dx** stencil (Section 3,
   finding a). Reject candidates whose refined solution leaves the local
   neighborhood, or that fail to converge.
6. **Ambiguity handling**: if more than one distinct (non-mergeable)
   candidate survives, report `resolved=False, reason="ambiguous"` rather
   than silently picking one.

Every result carries: `x_sub, y_sub`, `f_residual`, `gb_residual`,
`dist_from_legacy`, `contour_angle_deg` (angle between the two contours'
gradients — 90° = well-conditioned, near 0°/180° = tangent/degenerate),
`jacobian_det`, `n_candidates`, `newton_converged`,
`used_closest_approach_fallback`, and `resolved`/`reason`.

Continuous contact metrics (also in `tj_subgrid.py`, all diagnostic only):

- `L_contact_TJ_sub = |dot(r_top_sub − r_bottom_sub, t_GB_sub)|`
- `d_n_TJ_sub = |dot(r_top_sub − r_bottom_sub, n_GB_sub)|`
- `L_GB_geom_sub`: arc length of the `e1 = e2` contour between the two
  **continuous** TJ coordinates — the contour's own terminal vertices are
  replaced by the continuous coordinates directly (no snapping to a
  marching-squares vertex).
- `t_GB_sub`, `n_GB_sub`: built from continuous local tangents fit to the
  sub-grid `e1 − e2 = 0` contour near each TJ (`_gb_tangent_at`), not the
  discrete force-balance tangent vectors.

## 2. Synthetic sub-grid validation (31 tests, `tests/test_tj_subgrid.py`, all pass)

Fractional-grid translation of an otherwise-identical wedge TJ through
`frac·dx` for `frac ∈ {0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90}`, plus
dihedral angle, GB tilt, mild curvature, and mirrored top/bottom sweeps.
Representative numbers (dx = 2nm test grid):

**x-translation (perpendicular to GB, the harder direction) vs. legacy
integer-grid locator:**

| frac | sub-grid err/dx | legacy err/dx | contour angle |
|---|---|---|---|
| 0.00 | 0.00075 | 0.400 | 90.0° |
| 0.10 | 0.09959 | 0.412 | 87.8° |
| 0.20 | 0.18017 | 0.447 | 86.6° |
| 0.35 | 0.26023 | 0.532 | 86.9° |
| 0.50 | 0.28748 | 0.640 | 90.0° |
| 0.70 | 0.23924 | 0.500 | 86.5° |
| 0.90 | 0.09959 | 0.412 | 87.8° |

The sub-grid error grows and shrinks continuously with `frac` (peaking
mid-cell, a bilinear-interpolation discretization bias between the
continuous analytic construction and its discretized grid representation —
not a locator defect); it never exceeds ~0.29·dx. The legacy locator's
error is a sawtooth that never drops below ~0.40·dx even at
grid-aligned points, because it snaps to the nearest whole grid row/column
regardless of the true offset.

**y-translation (along GB, the easier direction)**: sub-grid error stays
below 0.0015·dx at every fraction — one case (frac=0.35) needed the
closest-approach fallback (Section 3c) and still resolved to err/dx =
0.00067.

**Dihedral angle sweep** (sharper angles are harder): err/dx = 0.494 at
80°, 0.350 at 100°, 0.242 at 120°, 0.154 at 140°, 0.075 at 160° — monotonic,
always sub-grid, `jacobian_det` well away from zero (~−2.6 to −3.8e15)
throughout.

**Tilted GB (15°), mild curvature, mirrored top/bottom**: all resolve with
err/dx < 0.35, confirming the locator is not tied to a purely vertical GB or
to the specific top/bottom polarity of the synthetic construction.

**Degenerate/tangent case** (`e1 − e2` built exactly proportional to `f`, so
the two contours coincide along a line rather than crossing at a point —
the textbook singular-Jacobian case): the locator correctly refuses to
resolve (`resolved=False, reason="Newton did not converge"`,
`jacobian_det=0.0`) rather than returning a confident but meaningless
answer. `n_candidates=1` shows a candidate was found geometrically but
correctly rejected once Newton failed to converge on the singular system.

## 3. Numerical residuals and conditioning findings

**a) Newton convergence rate is stencil-width sensitive.** The default
gradient stencil `h = 0.5·dx` (tj_force's own default) produced
damped-oscillatory convergence (alternating-sign geometric decay, ratio
~0.69/step) instead of quadratic — never tightening below a plateau within
a practical iteration budget. Using `h = 0.02·dx` specifically inside
`_newton_refine` restores standard quadratic Newton convergence
(machine-precision residuals within ~8 iterations). This only matters for
the *derivative* estimate used inside Newton; the *field* interpolation
itself (`_sample_bilinear`) is unaffected.

**b) Real f/gb residuals at convergence** are at floating-point roundoff:
`f_residual`, `gb_residual` ≈ 1e-16–1e-15 in every synthetic case checked
(Section 2 table). On the real sintering trajectory (Section 5), all
276–2759 steps resolved both TJs with the same tight residuals — zero
unresolved steps anywhere.

**c) Near-miss segment intersection.** One y-translation case (frac=0.35)
initially failed entirely (no direct segment crossing — the two
independently-traced marching-squares polylines pass close to, but not
literally through, each other when the true crossing sits very near a grid
vertex). Fixed by the closest-approach fallback described in Section 1,
step 4, gated at < 0.5·dx to avoid pairing unrelated contours.
`used_closest_approach_fallback=True` is exposed on the result for
transparency; it was used in 1 of the 31 synthetic tests and did not
compromise accuracy (err/dx = 0.00067 in that case).

**d) Narrow-neck search-radius overlap (real-trajectory bug, not a
synthetic-test bug).** With the initial fixed `search_radius_widths=3.0 ×
interface_width` (60nm at dx=5nm, interface_width=20nm), the 5nm-overlap
baseline geometry (`neck_height≈40nm`) had each TJ's neighborhood reach
past the midpoint and pick up the *other* TJ's contour segments as a second
candidate — both TJs then correctly, but unhelpfully, reported "ambiguous"
every single step. Fixed in `locate_neck_tjs_subgrid` by capping the radius
at `min(3.0·interface_width, 0.4·legacy["neck_height"])` (floored at
`0.5·interface_width`), the same adaptive convention `tj_force` already
uses internally. After the fix, both TJs in the baseline case resolve
cleanly at every step (`contour_angle_deg≈26°` — more marginal than the
primary case's ~44°, reflecting the narrower/less-separated geometry, but
never degenerate). Full test suite (92/92) re-verified clean after this fix
(commit `587c73d`).

## 4. Discrete vs. sub-grid TJ trajectories

Over the full 1379-step primary trajectory (dx=5nm, `|dV2|/V20` up to
1.5e-3), `L_contact_TJ_sub` never changes by more than **0.00099·dx** in a
single physical step (max observed step-to-step change: 0.00495nm against
dx=5nm) — orders of magnitude below the "cell-sized jump" failure threshold
the handoff explicitly flagged as a stop condition. No such jump was ever
observed at any target, grid, or geometry.

The contrast with the legacy discrete metric is stark and itself
informative. Cumulative `L_contact_TJ` (legacy) vs. `L_contact_TJ_sub` (new)
over the primary case's three run lengths:

| `|dV2|/V20` target | n_steps | `L_contact_TJ` (legacy, nm) | `L_contact_TJ_sub` (new, nm) |
|---|---|---|---|
| 3e-4 | 276 | +4.995 | +1.231 |
| 1.5e-3 | 1378 | +9.9999977 | +3.570 |
| 3e-3 | 2759 | +9.9999969 | +4.919 |

The legacy metric jumps almost immediately to a small number of whole-`dx`
values and then **saturates at exactly 2·dx (10nm)** for the two longer
runs — it has stopped tracking the physics entirely, reporting the same
number whether coarsening has proceeded for 1378 or 2759 steps. The new
sub-grid metric keeps growing smoothly and monotonically over the same
window, tracking the physical contact continuously rather than in a handful
of discrete grid-locked increments. This is the direct, real-trajectory
confirmation of what Milestone 6B diagnosed as a quantization artifact.

## 5. Primary case operator ledger (`|dV2|/V20` = 3e-4, the Milestone 5/6/6B comparison point)

`preset=dev, overlap=20nm, dx=5nm, coarsening_rate_scale=3, surface_mobility_scale=0.3, eta_mobility_scale=1`, 276 steps, `dV2/V20 = -3.0062e-04` (exact target hit):

| operator | `dL_contact_TJ_sub` (nm) | `dL_GB_geom_sub` (nm) | `dx_neck_eta` (nm) |
|---|---|---|---|
| CH (raw) | **+1.2180585** | +1.2428017 | +0.0000000 |
| post-CH projection | +0.0214399 | +0.0232943 | +0.0093829 |
| Ostwald | +0.0051214 | +0.0063243 | +0.0085078 |
| eta_raw | **−0.0090016** | −0.0087689 | +0.0082388 |
| post-eta projection | −0.0046783 | −0.0047488 | +0.0028636 |
| **TOTAL** | **+1.230940** | +1.258903 | +0.028993 |

Closure is exact to floating-point roundoff (`d_n_TJ_sub` closure error
~1.8e-27; all other keys exactly 0.0). CH alone accounts for **99.0%** of
the total contact-length change; every other single operator is under 2% of
CH's contribution, consistent with the dominance ratio the v3 test suite
now asserts (`test_raw_CH_dominates_subgrid_contact_change_in_this_regime`:
every non-CH operator's magnitude < 0.1·CH's).

## 6. Raw-CH contact-motion sign

**Positive at every target checked, without exception**:

| case | `dL_contact_TJ_sub` from CH (nm) | run `|dV2|/V20` |
|---|---|---|
| primary, dx=5nm | +1.2180585 | 3.0e-4 |
| primary, dx=5nm | +3.4869816 | 1.5e-3 |
| primary, dx=5nm | +4.7468010 | 3.0e-3 |
| fine grid, dx=2.5nm | +5.5189722 | 3.0e-4 |
| baseline, overlap=5nm | +1.4898425 | 3.0e-4 |

Raw CH widens the physical contact in every configuration tested. This
directly answers the milestone's central question: measured by contact
endpoint motion rather than inferred from flux sign alone, **CH capillary
evolution widens the neck**, it does not shrink it.

Fitting `L_contact_TJ_sub` (total, all operators) vs. `|dV2|/V20` linearly
across the 1379-point 1.5e-3 trajectory gives slope ≈ 2260 nm per unit
`|dV2|/V20`, R² = 0.96 — a good but not perfect linear fit; the
CH-only totals ratio (`dL_CH / |dV2/V20|`) is 4051, 2325, and 1582 nm/unit
at the 3e-4, 1.5e-3, and 3e-3 targets respectively, i.e. mildly concave
(the rate of contact growth per unit further densification slows somewhat
as coarsening accumulates) rather than perfectly linear — worth noting for
any future model fit, but the **sign** is unambiguous throughout.

## 7. CH flux vs. measured TJ motion

Milestone 6 established that the CH surface flux `J_tangent` near both TJs
points away from the TJ along the free surface, but could not on its own
determine what that implies for contact width. With continuous TJ
positions now available, the geometric link is direct and, in the primary
case's local frame (`t_GB_sub ≈ (0, 1)`, i.e. the GB runs along y; top TJ
starts above bottom TJ), verified to floating-point precision over the
first 5 steps:

| step | `v_top` (tangent, normal) m/s | `v_bottom` (tangent, normal) m/s | `dL_contact_TJ_sub` observed (nm) | `dL_contact_TJ_sub` from `(v_top−v_bottom)·t_GB·dt` (nm) |
|---|---|---|---|---|
| 1 | (+3.10e-7, −6.04e-8) | (−3.76e-7, −7.38e-8) | 0.004977 | 0.004977 |
| 2 | (+3.10e-7, −6.03e-8) | (−3.75e-7, −7.34e-8) | 0.004976 | 0.004976 |
| 3 | (+3.11e-7, −6.04e-8) | (−3.74e-7, −7.31e-8) | 0.004975 | 0.004975 |
| 4 | (+3.12e-7, −6.04e-8) | (−3.73e-7, −7.28e-8) | 0.004970 | 0.004970 |
| 5 | (+3.12e-7, −6.05e-8) | (−3.72e-7, −7.25e-8) | 0.004966 | 0.004966 |

The two TJs move in **opposite tangential directions**, away from each
other along the GB (top moves +t, bottom moves −t) — this, not flux
direction alone, is what fixes the sign: with both `J_tangent_top > 0` and
`J_tangent_bottom > 0` (both fluxes "away from TJ" in the established
convention), the material each TJ sheds locally along the free surface
corresponds to that TJ's own position receding outward along the GB. Two
TJs both receding outward from each other is, geometrically, contact
**widening**, not recession. This resolves Milestone 6's open question:
"flux away from TJ" in this regime means the contact grows, not shrinks.
The normal-direction (`d_n_TJ_sub`) component is small and consistent
between top and bottom (both TJs also drift slightly inward in x together
— ordinary symmetric neck migration, not a contact-width effect).

## 8. Eta/projection overlap change vs. endpoint motion

Comparing `dx_neck_eta` (the legacy eta-based proxy) against
`dL_contact_TJ_sub` (the new physical endpoint metric), stage by stage in
the primary 3e-4 ledger (Section 5):

| operator | `dL_contact_TJ_sub` (nm) | `dx_neck_eta` (nm) | same sign? |
|---|---|---|---|
| post-CH projection | +0.0214 | +0.0094 | yes |
| Ostwald | +0.0051 | +0.0085 | yes |
| **eta_raw** | **−0.0090** | **+0.0082** | **no** |
| **post-eta projection** | **−0.0047** | **+0.0029** | **no** |

`eta_raw` and its immediate post-projection step consistently **shrink**
the physical contact (`L_contact_TJ_sub` decreases) while simultaneously
**growing** the eta-derived proxy (`x_neck_eta` increases) — opposite signs,
at every stage, in every run length checked (the 1.5e-3 and 3e-3 tables
show the same pattern, e.g. at 3e-3: eta_raw `dL_contact_TJ_sub=−0.0765nm`
vs `dx_neck_eta=+0.0796nm`). This is a clean, reproducible dissociation: the
structural/eta relaxation operator's effect on `x_neck_eta` is **not**
tracking the same physical event as its effect on the true TJ separation.
The most consistent interpretation is that `eta_raw` is substantially
broadening the diffuse e1/e2 overlap region near the neck (which
`x_neck_eta`, an eta-threshold-based proxy, picks up directly) while the
actual level-set TJ crossing point it is not moving in the same direction —
i.e., at least part of the legacy widening signal in `x_neck_eta` is
structural/diffuse-interface bookkeeping rather than genuine physical TJ
motion, exactly as Milestone 6B suspected but could not confirm without a
continuous locator.

## 9. Extended trajectory (1.5e-3 and 3e-3)

Both extensions preserve the primary case's sign and CH-dominance pattern
(tables above, Sections 4 and 6). `L_GB_geom_sub` tracks `L_contact_TJ_sub`
closely throughout (e.g. at 3e-3: 4.919nm vs. 5.055nm total — the GB itself
is not developing significant curvature between the two continuous TJ
endpoints, consistent with a roughly straight, uniformly-receding GB rather
than a locally pinned or buckling one). `sigma_Pa` (from the pre-existing
stress diagnostic sampled alongside these metrics) decreases mildly and
monotonically over the trajectory, from 5.951e7 to 5.913e7 Pa at 3e-3 —
consistent with a widening (lower local curvature/stress-concentration)
contact rather than a narrowing one, corroborating the sign found directly
from TJ positions through an independent diagnostic channel.

## 10. Finer-grid directional check (dx = 2.5nm vs. 5nm)

Same 3e-4 target, physical dimensions (r2, overlap, aspect ratio) held
fixed, interface width scaled consistently with dx via the existing
`build_params` convention:

| | dx=5nm | dx=2.5nm |
|---|---|---|
| n_steps | 276 | 276 |
| `L_contact_TJ_sub` total (nm) | +1.231 | +5.613 |
| CH contribution (nm) | +1.218 | +5.519 |
| `x_neck_eta` total (nm) | +0.029 | **−0.105** |

The sign of raw-CH contact widening is **unchanged** at the finer grid — if
anything the effect is larger in absolute nm (though also a larger fraction
of a now-smaller dx, so this magnitude comparison is not the point; the
sign is). Notably, the legacy eta proxy `x_neck_eta` actually **flips
sign** between the two grids (+0.029nm at dx=5nm vs. −0.105nm at
dx=2.5nm) while the new sub-grid physical metric does not — direct evidence
that `x_neck_eta` is grid-resolution-sensitive in a way the continuous
physical contact length is not, reinforcing that the eta proxy is an
unreliable stand-in for the true contact trajectory.

## 11. Baseline 5nm-overlap cross-check

Same rates, dx=5nm, 3e-4 target, `initial_overlap=5nm` instead of 20nm
(narrow-neck geometry that originally triggered the search-radius bug fixed
in Section 3d):

| | 20nm (primary) | 5nm (baseline) |
|---|---|---|
| n_steps | 276 | 276 |
| `L_contact_TJ_sub` total (nm) | +1.231 | +1.523 |
| CH contribution (nm) | +1.218 (99.0%) | +1.490 (97.9%) |
| `x_neck_eta` total (nm) | +0.029 | +0.065 |

**Same sign, same CH dominance, at both starting overlaps.** No
neck-stability sign crossover was observed between 5nm and 20nm initial
overlap under these rate settings — both geometries widen under raw CH,
ruling out Outcome D for this particular rate-parameter point (though a
crossover at some other overlap or rate combination cannot be excluded from
two points alone).

## 12. Outcome classification

**Outcome B: raw CH physically widens the contact**, confirmed by direct
sub-grid TJ tracking (not merely inferred from flux sign), sign-stable
across:

- three run lengths (3e-4, 1.5e-3, 3e-3 `|dV2|/V20`),
- a 2x change in grid resolution (dx=5nm and dx=2.5nm),
- a 4x change in initial neck geometry (20nm and 5nm overlap),
- and overwhelmingly dominant (CH is 96–99% of the total physical contact
  change at every target checked; no other single operator exceeds ~2% of
  CH's contribution).

This rules out Outcome A (the earlier working hypothesis that the eta
diagnostic was hiding an underlying CH-driven neck-shrinkage tendency) —
the physical contact endpoint itself, tracked continuously, widens under
raw CH, not just the discretized/eta-based proxies used in earlier
milestones. Outcome C is only partially supported: eta/projection *does*
move the physical TJ (Section 8), and even opposes CH's widening in the
eta_raw stage specifically — but its magnitude is far too small (≤2% of
CH's per step) to be the primary driver, and its sign relative to
`x_neck_eta` is inconsistent with `x_neck_eta` being a faithful physical
proxy in the first place. Outcome D (geometry-dependent sign) is not
supported by the two geometries tested here (Section 11), though it remains
untested outside this rate-parameter point.

## 13. Recommended next physics step (NOT implemented)

Given Outcome B, plain conserved CH surface diffusion under this sink-off
protocol does not, on its own, reproduce the experimentally expected
neck-narrowing/stress-buildup direction — so the nucleation-relevant
driving quantity likely should **not** be defined as "raw contact-length
change under natural coarsening." The smallest next diagnostic step,
building directly on data this milestone already generates rather than
starting a new investigation: **compute the trend of `F_TJ` (the
Cahn-Hoffman force-imbalance magnitude/normal component from
`tj_force.compute_neck_tj_forces`, already available and unmodified) over
the same trajectories analyzed here, and compare its sign/trend against
`L_contact_TJ_sub`.** Force imbalance at the TJ and physical contact width
are logically separate quantities — it is entirely possible for the
resolved solid-solid contact to widen geometrically while the local force
imbalance (or the signed curvature / configurational force at the
continuous sub-grid TJ position) still grows in a way that is directly
relevant to nucleation stress. This should be checked, purely as a
diagnostic against the same runs already produced, before any change to the
hazard/stress coupling model is considered.

**STOP. Do not alter production physics before review.**
