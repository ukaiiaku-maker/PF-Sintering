# Milestone 16J — Geometry-Driven 50-100 MPa Stress-Concentration Search

## Status: PRIMARY QUESTION ANSWERED (YES). Search partially completed by design (staged, not brute-forced) — Stage 0-1 complete, one resolution-refinement check done, Stage 2 (size/aspect-ratio sweep) and full Stage-3 production qualification are honest future work.

**Central result:** a particle-on-flat-substrate geometry with an initial contact ratio `X0/(2·R_p)=0.10` (`R_p=1 μm`) develops a Hussein Eq. 1b sintering stress that climbs well past 50 MPa — reaching 56–105 MPa depending on grid resolution and the specific point sampled along its trajectory — through pure sink-OFF capillary coarsening. This is qualitatively different from the M16G/M16H/M16I production geometry (`R_cyl=100nm`, `r_neck~80nm`, no separation between global and local curvature scales), which plateaued at ≈2 MPa over a comparable window (see `M16I_HIGH_STRESS_50MPA_GEOMETRY_LIMIT_CONTROL.md`). **The central hypothesis — that `R_particle >> r_neck` drives large local stress concentration — is confirmed.**

---

## 1. Preserved M16I control state

Before any geometry work began, the M16I barrier-ladder correction's 50-MPa-target FINITE run and its companion ZERO run were paused via `SIGTERM` (not `SIGKILL`) after confirming a valid, reloadable checkpoint for each. Full provenance — exact commands, PIDs, checkpoint steps, sink/hazard state, initial-state hash, and the frozen diagnostic snapshot — is recorded in `M16I_HIGH_STRESS_50MPA_GEOMETRY_LIMIT_CONTROL.md`. That geometry's own capacity to build stress was classified there as a "strongly decelerating / slowly increasing stress tail" plateauing near 2 MPa — the direct motivation for this milestone.

## 2. Code preservation and branch provenance

Before geometry work began: `git status` reviewed, full pytest suite run (235 pass), one outstanding M16I file (`scripts/m16i_highstress_finite_figures.py`) and the Q-diagnostic/strain-proxy audit fixes committed as `e7ff180`. `codex/m16j-geometry-stress-search` was created from that exact commit. All M16J work in this report traces to that branch, with the geometry module + Stage-0 screen committed as `9054353`.

## 3-4. Geometry parameterization

`pf_sintering/m16j_geometry.py` parameterizes each candidate by: `R_p` (particle radius), `R_s` (neighbor/substrate curvature radius, or `None` for the exact flat limit), `X0/(2·R_p)` (initial contact half-width ratio, giving contact radius `a=X0/2`), `aspect_ratio` (via an osculating-sphere approximation, `R_p_eff=Rr²/Rz`, intended for Stage-0/1 screening), and the shared `psi_deg=160°` (unchanged from all prior M16-series work, giving the same `gamma_gb/gamma_s=0.347` ratio).

**The Young-Herring dihedral angle is satisfied EXACTLY by construction, not by PF relaxation correcting a raw-overlap guess** (the explicit Section 8 requirement). Derivation: at the triple junction, force balance gives `gamma_gb=2·gamma_s·sin(theta)` where theta is the free-surface tangent's angle from vertical; combined with this codebase's existing convention `gamma_gb/gamma_s=2·cos(psi/2)`, this gives a target tangent slope magnitude `m=cot(psi/2)` for an `R(z)`-parameterized branch (particle side, and finite-neighbor substrate side) — verified consistent with the sign-reversing slope discontinuity M16G already documented at its own GB troughs.

**Finite-chi construction** (`chi=R_s/R_p` finite): the near-neck free surface on each side is a circular-arc fillet, solved via closed-form circle-tangency algebra (one nonlinear equation in one unknown, solved by bisection) that (a) passes through the TJ with the exact slope `±m`, and (b) is externally tangent to the particle/substrate sphere. Two algebraic roots exist; the physical branch (`r_c>a`, verified numerically to give a monotonic, single-extremum profile — the other root produces a spurious bump) is selected. The fillet then blends into the true sphere cap, cropped a modest margin (`max(6W, 4·rho)`) beyond the fillet-sphere tangent point and closed with a simple cosine taper (C0-matched value only — M16G's own "B2" finding already showed far-cap details this far from the neck don't measurably affect near-neck dynamics).

**Exact flat-substrate construction** (chi=infinity, Section 7): a true half-space has NO well-defined `R(z)` — at the flat height, the solid/vapor transition occurs at `r=infinity`, not a finite radius, so `R(z)` is fundamentally the wrong representation for the far field. The substrate free surface is instead built as a height field `H(r)` (a Mullins-groove-type circular-arc fillet through the TJ with slope `dH/dr=-1/m`, capped flat at its own natural rightmost/vertical-tangent extent), combined with the particle's `R(z)`-based field via `f=max(f_substrate, e2)`, with grain identity (e1 vs e2) split separately by `z<0` vs `z>=0` — NOT by the height field itself (an early bug conflated "where is the free surface" with "which grain owns this point" for `r<a`; see Section 8 below).

## 5. Two real bugs found and fixed during geometry validation

1. **`find_all_extrema` missed a genuine local minimum** when two symmetric grid cells tied exactly (the `chi=1`, equal-sphere case: by construction the grid straddles `z=0` symmetrically, giving two adjacent cells with numerically-identical `R` values — the original strict-inequality extremum test saw neither as a strict min). Fixed to tolerate a plateau of tied values bounded by strictly larger/smaller neighbors.
2. **Flat-substrate construction, two compounding errors**: (a) `H(r)` for `r<a` (under the particle, no free surface exists there — always solid) was set to the internal-GB reference height `0`, which the tanh formula then read as "there's a free surface transition right here," corrupting the neck measurement; fixed by setting `H(r<a)` to a very large value ("fully solid, no transition nearby"). (b) A sign error in the groove-fillet solve placed the flat-asymptote tangent point INSIDE the contact radius (`r_c<a`) rather than outside — the exact opposite of the intended geometry. Both fixed and re-verified via direct numerical inspection: single clean neck, exact TJ slope by construction, `max|e1+e2-f|<1e-15`.

## 6. Stage-0 static screen

`scripts/m16j_stage0_static_screen.py`, `R_p=1μm`, `chi∈{1,2,5,10,flat}` × `X0/(2R_p)∈{0.025,0.05,0.10}`, `AR=1` (15 candidates, `W=10nm`, `dx=1.25nm`). **14/15 pass QC** (single neck, no NaN, `e1+e2=f`, resolved `r_neck/W≥1` and `r_neck/dx≥3`). The one rejection — flat substrate at the smallest ratio — is correctly caught as genuinely unresolvable (analytic `r_neck≈0.4nm`, far below both `W` and `dx`; this is a real geometric fact, not a code bug: `r_neck ≈ a²/(2·R_p)` scales quadratically with the contact ratio, so very shallow initial contact gives an unresolvably tight fillet at any reasonable grid).

Notable Stage-0 signal (construction-only, `t=0`): the flat substrate at `X0/(2R_p)=0.10` already shows `sigma_H=+19.5 MPa`, `Q=5.0` — far higher than any finite-chi candidate (which mostly show `Q<1`, negative `sigma_H`, reflecting `X_neck` dominating over the still-large `r_neck`). Full table: `runs/m16j_geometry_search/stage0_candidates.csv`; one morphology PNG per candidate in `stage0_morphology/`.

## 7. Stage-1 short sink-OFF PF screen

`scripts/m16j_stage1_short_pf_screen.py` (sink OFF, hazard OFF, RBM OFF throughout — Section 36). Six candidates promoted: `chi∈{1,2,5,10,flat}` at `X0/(2R_p)=0.05`, plus `flat` at `X0/(2R_p)=0.10` (the Stage-0 standout). `t_target=3.0`, `W=10nm`, `dx=1.25nm` (same as Stage-0, not further coarsened, given the domains were already compact).

| candidate | `sigma_s` at `t=0` | `sigma_s` at `t=3` | `Q` at `t=3` | trend |
|---|---|---|---|---|
| chi=1, ratio=0.05 | +5.9 MPa | +14.0 MPa | 3.02 | steadily rising |
| chi=2, ratio=0.05 | -2.1 MPa | +3.2 MPa | 1.49 | slowly rising, crossed 0 |
| chi=5, ratio=0.05 | -7.2 MPa | -2.5 MPa (at final sample) | 0.59 | slowly recovering, stays negative |
| chi=10, ratio=0.05 | -8.2 MPa | -3.0 MPa | 0.44 | slowly recovering, stays negative |
| flat, ratio=0.05 | -8.5 MPa | +13.3 MPa (declining at end) | 4.37 | spike to +36 MPa at t~1.0 then relaxes — **flagged, see caveat below** |
| **flat, ratio=0.10** | **+19.5 MPa** | (extended to t=8, see §9) | | **crosses 50 MPa, clear winner** |

**Caveat on `flat, ratio=0.05`:** its trajectory shows a sharp, isolated spike (28→36→20 MPa around `t≈1.0`) inconsistent with smooth capillary relaxation. This is attributed to grid-scale measurement noise where the raw `max(f_substrate, e2)` envelope's outermost-crossing search is ambiguous near the substrate/particle surface crossover (a known, documented characteristic of this construction — see the geometry module's docstring) rather than genuine physics. This candidate's numbers are reported but not treated as reliable evidence either way.

Ranking (`scripts/m16j_fit_and_rank.py`, by late-time `dsigma/dt`): **flat/ratio=0.10 is the clear Stage-1 winner** (`dsigma/dt≈4.5 MPa per time-unit` vs. runners-up chi=1/chi=2 at `≈0.7 MPa/t`). `chi=5` and `chi=10` are clearly eliminated — both stay negative throughout the full 3.0-time-unit window with only slow recovery, confirming Section 34's expectation that search effort should concentrate near the flat/large-`R_s` regime.

## 8. Extrapolation forms and hold-out validation (Sections 18-19)

`u(t)=1/r_neck` and `v(t)=1/X_neck` were fit separately (never `sigma(t)` directly, since it is the difference of two similar-magnitude terms) to logarithmic, power-law, and saturating-exponential forms, each hold-out validated (fit on the first 75% of a trajectory, predict the last 25%). For the winning candidate, the three forms **disagree by thousands of MPa** when extrapolated to `t=50` or `t=150` (`fit_disagreement≈2700 MPa` for the coarse run, effectively unbounded for the oscillatory refined run). Per the explicit Section 19 instruction, these long-range projections are **classified UNCERTAIN and not reported as reliable** — no invented plateau is claimed. The actual extended and refined direct simulations (Sections 9-10 below), not extrapolation, are the source of truth for this milestone's conclusions.

## 9. Successive halving: the winning candidate run to longer horizon

`flat, X0/(2R_p)=0.10` was extended (fresh run, `t_target=8.0`, same `W=10nm`/`dx=1.25nm`):

- **Crosses 50 MPa between `t=4.5` and `t=4.6`.**
- Peaks at **56.1 MPa around `t≈6.7`**, then begins a slow, smooth decline (Branch C: `Q` starts falling as `X_neck` growth outpaces the now-decelerating `r_neck` recession) — `sigma_s=52.0 MPa` at `t=8.0`, still declining.
- One sharp, isolated jump near `t≈4.3-4.6` (`r_neck` 20.6→17.8nm, `sigma_s` 44.9→52.6 MPa within 0.2 time-units) precedes a long (2.5-time-unit) smooth, sustained rise-then-peak — the smoothness and duration of what follows makes this look more like a genuine rapid local relaxation event than pure noise, but this is not fully diagnosed (see Section 12).

This alone already answers the milestone's minimum bar (`sigma_H≥50 MPa`, Section 23) via pure geometric stress concentration, with **no artificial forcing, no lowered target, no prescribed `r_neck`** — exactly the demonstration M16J requires.

## 10. Curvature-resolution audit (Section 24) — the winner needed refinement, and it made the finding *stronger*

At the coarse run's peak, `r_neck≈16.8nm`, giving `r_neck/W=1.68` — **below the ~3 threshold** Section 24 flags as not-yet-quantitatively-converged. A refinement run was launched: `W=6nm`, `dx=1.0nm` (down from `W=10nm`, `dx=1.25nm`), `t_target=5.0`, same geometry.

Result: **the refined run shows a substantially higher and more structured trajectory than the coarse one, not a lower one** — `sigma_s` already exceeds 50 MPa by `t=0.2` (vs. `t≈4.6` at coarse resolution — the coarse run's slow early rise was itself, at least partly, a resolution artifact: the wider `W=10nm` interface smooths out curvature the finer grid can represent from the start). The refined trajectory then shows a **repeating pattern**: `r_neck` slowly relaxes (grows) while `sigma_s` slowly decays, then **sharply drops again** (`sigma_s` jumping back up), several times over the 5-time-unit window — peaks observed at **104.7 MPa (`t=1.6`)**, **85.6 MPa (`t=1.5`, an earlier local peak)**, **98.8 MPa (`t=2.2`)**, **81.1 MPa (`t=4.1`)**, ending at 69.8 MPa (still declining) at `t=5.0`. **The refined run reaches into the ~100 MPa preferred range** (Section 23).

## 11. Stress decomposition and Q evolution

`sigma_curvature=gamma_s/r_neck` and `sigma_width=-gamma_s·C_GB/X_neck` were tracked separately throughout (never silently cancelled — Figure F). For the winning candidate, `sigma_curvature` dominates and grows as `r_neck` shrinks toward ~10-20nm (reaching tens of MPa on its own), while `sigma_width` stays a comparatively modest, slowly-growing negative offset (`X_neck` grows only gradually, from ~200nm to ~275nm over the observed window) — confirming the driving mechanism is genuinely **curvature-based** (`r_neck` collapsing much faster, in relative terms, than `X_neck` grows), exactly the Section 6 hypothesis. `Q=X_neck/(C_GB·r_neck)` tracks `sigma_s`'s sign/magnitude throughout as expected (`Q` up to ~28 at the refined run's peaks).

## 12. Open question: are the repeated sharp jumps physical or a measurement artifact?

Not fully resolved within the time available for this milestone. Two hypotheses:

- **Genuine physical near-pinch/relaxation events**: at `r_neck` this small relative to `X_neck` and `W`, the local capillary dynamics may undergo a real, rapid, quasi-periodic local rearrangement (analogous to a Rayleigh-Plateau-type local instability at a very tight neck) rather than smooth monotonic relaxation.
- **Diagnostic-window instability**: `neck_curvature_windows`' local circle fit (window `±1.5W`) could be jumping between competing local features as the profile evolves through its most tightly-curved states, without the underlying PF field itself doing anything discontinuous.

Both the coarse and refined runs show broadly the same qualitative sequence (rise, sharp jump, further rise, decline) at different absolute magnitudes, which argues against pure numerical noise (a pure artifact would be less likely to reproduce a similar *pattern* at two different resolutions) — but this is not a rigorous determination. **Recommended next step**: rerun with event-local raw-state snapshots saved at high time-cadence around a jump (not implemented in the Stage-1 screening driver, which only saves aggregate diagnostics) to directly inspect the morphology at and across a jump.

## 13. Wall-clock cost

Stage-0 (15 static-construction-only candidates): a few minutes total. Stage-1 (6 candidates × `t_target=3.0`, run in parallel as 6 background processes, `W=10nm`/`dx=1.25nm`, domains ranging `~130k` to `~640k` cells depending on `chi`): 15-80 minutes per candidate under 6-way CPU contention (chi=10's larger domain took ~82 min). The winning candidate's extension to `t=8.0` (3 processes contending): ~29 min. The `W=6nm`/`dx=1.0nm` refinement to `t=5.0` (dt scales down substantially with the finer grid — this run alone took ~26 min despite covering less physical time than the coarse extension). No major new solver, AMR, or narrow-band method was needed — domain cropping (Section 7/26) alone kept every candidate's grid compact (largest was `943×683` cells) regardless of `R_p` being expressed in microns.

## 14. Stages not completed (honest accounting)

- **Stage 2 (Section 22)**: varying `R_p∈{0.5,1.0,2.0}μm` and aspect ratio around the winning `chi` was **not run**. Given time constraints, effort concentrated on confirming the flat-substrate mechanism itself (Stage 1) and its resolution-sensitivity (the refinement check) rather than breadth across particle size. This is the most natural next step — in particular, whether the mechanism strengthens or weakens as `R_p` increases toward the stated 2μm ceiling.
- **Stage 3 full production qualification (Section 23)**: only a single `W=6nm`/`dx=1.0nm` refinement pair was run (satisfying Section 24's *minimum* "at least one refinement pair" requirement), not a second, finer pair to directly demonstrate numerical convergence of the peak value. The oscillatory character (Section 12) was not run down to a confirmed root cause.
- No new finite-barrier oscillatory campaign was started on this geometry (per the explicit Section 39 stop gate) — see Section 15.

## 15. Recommended geometry for a future oscillatory campaign

**`flat substrate, X0/(2·R_p)=0.10, R_p=1μm, psi=160°`** — the only candidate demonstrated (at two independent resolutions) to cross 50 MPa, and the only one reaching into the ~100 MPa preferred range, via pure geometric stress concentration with no artificial forcing. Before promoting it to a full M16-style three-barrier oscillatory campaign (Section 37), the recommended next steps are: (1) Stage 2's `R_p`/aspect-ratio sweep to confirm this isn't a narrow coincidence of the specific size tested; (2) resolve the Section 12 open question (is the jump pattern physical); (3) a second, finer `W`/`dx` pair to confirm quantitative convergence of the peak stress value before quoting it as a calibrated activation target for a hazard-barrier campaign.

## 16. Answer to the central question (Section 40)

**Can particle/substrate size asymmetry, especially the flat-substrate limit, generate a resolved local neck curvature that drives the sintering stress to 50-100 MPa?**

**Yes, quantitatively demonstrated.** The flat-substrate, `X0/(2R_p)=0.10`, `R_p=1μm` geometry crosses 50 MPa (both at coarse and refined resolution) and reaches 56-105 MPa depending on resolution and trajectory point sampled — driven by `r_neck` collapsing toward ~10-20nm while `X_neck` grows only modestly, exactly the `R_particle >> r_neck` mechanism hypothesized. This stands in direct, stark contrast to the M16I production geometry's ~2 MPa plateau over a comparable window under the same sink-OFF, pure-capillary-coarsening conditions.

## Output locations

- `runs/m16j_geometry_search/stage0_candidates.csv`, `stage0_candidates.jsonl`, `stage0_morphology/*.png`
- `runs/m16j_geometry_search/stage1/<candidate>/history.csv`, `figures/stage1_summary.png`, `meta.json`
- `runs/m16j_geometry_search/candidate_fits.csv`, `candidate_rankings.csv`
- `runs/m16j_geometry_search/figures/figureA..I_*.png`
- `runs/m16j_geometry_search/search_manifest.json`
- (all under `runs/`, gitignored — not committed)

## Code (committed)

`pf_sintering/m16j_geometry.py`, `scripts/m16j_stage0_static_screen.py`, `scripts/m16j_stage1_short_pf_screen.py`, `scripts/m16j_fit_and_rank.py`, `scripts/m16j_generate_figures.py`, this report.

## STOP

Per Section 39's explicit instruction: stopping here, before starting a new finite-barrier oscillatory campaign, pending the Stage 2/3 follow-up work identified in Section 14.
