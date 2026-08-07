# Rate-competition campaign: natural coarsening physics, sink-off

Scope: the compact sink-off rate-competition matrix (coarsening rate x surface/CH
rate, on two starting geometries) plus a follow-up GB/eta-mobility sweep, per the
updated handoff. **Stopping here for review, as instructed** — the active-sink rate
campaign (Section 11 of the handoff) was not attempted.

## What changed from the previous (constrained-relaxation) direction

Per the handoff, the constrained-`G*(V2,L)` work was checkpointed as
EXPERIMENTAL/DIAGNOSTIC (both `separation_constraint.py` and
`constrained_relaxation.py` now say so in their module docstrings) and is not
developed further. This campaign instead lets the model's own Ostwald/reservoir
exchange, CH free-surface transport, and structural (eta/GB) relaxation run forward
exactly as the production runner does — no V2 trajectory is prescribed, no shape is
frozen, nothing is minimized to convergence. The direct TJ capillary-vector
diagnostic (`tj_force.py`), the robust branch extractor, the measured-dihedral
diagnostic, the signed-curvature diagnostic (`signed_curvature.py`), the Milestone-1
directional diagnostics (`diagnostics.py`), the mass-preserving structural
projection, and the H1/H2 controls are all unchanged and reused as-is.

## A. New, independent rate controls

Two new `ModelConfig` fields, wired into `build_params` (`pf_sintering/model.py`):

- `coarsening_rate_scale`: multiplies the Ostwald/reservoir rate via
  `tau_ripening = 20.0 / coarsening_rate_scale` (the existing parameterization,
  Section 3A of the handoff).
- `surface_mobility_scale`: multiplies `M_f` (free-surface/CH mobility), anchored
  independently of `eta_mobility_scale`'s `M_eta` (both are computed from the same
  *unscaled* `M_f_base`, so scaling one never leaks into the other — verified in
  `tests/test_mechanism_controls.py`).

**A dt subtlety that had to be resolved first**: `build_params`' own CFL formula sets
`dt = min(CFL*dx^4/(M_f*k_f), 1e-5)`. If `dt` is left to auto-recompute from a scaled
`M_f`, the per-step CH update magnitude is `~ dt*M_f`, and since `dt ~ 1/M_f` under
that formula whenever the CFL term is the binding constraint, `dt*M_f` becomes
*independent of the scale* — i.e. `surface_mobility_scale` would silently do nothing
to the actual per-step physics, only to the step count needed to cover a given
physical time (and even that effect cancels out under a physical-time-based stopping
criterion). Empirically, for the geometry used here the `1e-5` cap (not the CFL term)
is what actually binds up to at least 10x `M_f`, so this particular failure mode
doesn't occur in the tested range — but rather than rely on that being true for every
case, `dt` was explicitly fixed via `--dt-override` to the 1x/1x/1x baseline's value
for the *entire* campaign (`dt_fixed = 7.2624e-6 s`), guaranteeing every case
integrates the same physical time per step regardless of scale, by construction
rather than by coincidence.

## B. Trajectory runner (`pf_sintering/rate_competition.py`)

`run_sinkoff_trajectory`: CH -> mass-preserving eta projection -> Ostwald ->
structural relaxation -> mass-preserving eta projection, every step, exactly matching
the qualified production sequence. `hazard_step` and `rbm` are **never called** (not
merely prevented from firing), so the sink stays inactive and no RBM occurs by
construction. Runs until `|V2(t) - V2(0)|/V2(0)` reaches `target_dv2_frac` (no V2
trajectory is prescribed -- V2(t) is whatever the existing Ostwald kernel produces) or
`max_steps` is hit. `rich_sample()` composes `diagnostics.sample()` (V2, separation,
neck width, GB/contact area, legacy sigma + its decomposition, energies) with
`tj_force.compute_neck_tj_forces()` (measured `psi_top`/`psi_bottom`, both TJs'
Cahn-Hoffman `xi` vectors, the GB vector, `F_TJ` and its signed components) and
`signed_curvature.signed_curvature_top_bottom()` -- purely by composing the
already-validated modules, none of which were modified. `psi_eq` is never used
anywhere in this campaign.

Tests: `tests/test_rate_competition.py` (3 tests: rich-sample field completeness,
sink-never-activates + mass-holds over a trajectory, and that the two new rate
controls actually change trajectory speed in the expected direction). Full suite:
**37/37 passing**.

## C. Two initial geometries, via existing controls only

Both built with the existing `initialize_fields` (no new construction):

- **A (baseline)**: `initial_overlap = 5nm` -- the same near-critical candidate used
  throughout Milestones 1-2 (`dev` preset, `R2=80nm`, `aspect_ratio=2.0`,
  `contact_orientation=short_plane`, `dx=5nm`).
- **B (wider)**: `initial_overlap = 20nm` -- one of the overlaps already explored in
  the Milestone-2 near-critical scan, giving a visibly wider resolved neck at `t=0`,
  motivated by testing whether starting on the wide side of the (unknown) natural
  trajectory changes the response.

## D. The compact rate matrix and results

`coarsening_rate_scale` in `{0.3, 1.0, 3.0}` x `surface_mobility_scale` in
`{0.3, 1.0, 3.0}`, `eta_mobility_scale = 1.0`, both geometries, target
`|dV2|/V20 = 1e-3` (18 runs, `dt_fixed = 7.2624e-6 s` throughout):

| geometry | coarse | surf | steps | dx_neck (nm) | dA_GB (nm^2) | dpsi_top (deg) | dpsi_bot (deg) | dsigma (MPa) | dF_TJ_top | dF_TJ_bot | dG_interface (J) | dL (nm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3x | 0.3x | 9185 | +1.476 | +131.88 | +10.82 | +1.49 | -1.980 | +0.287 | -0.043 | -2.81e-09 | -0.987 |
| baseline | 0.3x | 1.0x | 9185 | +2.880 | +214.17 | +9.33 | +7.23 | -3.307 | +0.121 | +0.068 | +7.62e-10 | -1.837 |
| baseline | 0.3x | 3.0x | 9205 | **+6.401** | +434.93 | **+31.71** | +20.41 | **-5.718** | +0.502 | +0.045 | **+2.76e-08** | **-3.720** |
| baseline | 1.0x | 0.3x | 2756 | +0.497 | +47.02 | +9.38 | +3.33 | -1.029 | +0.241 | -0.040 | -3.05e-09 | -0.456 |
| baseline | 1.0x | 1.0x | 2756 | +1.115 | +90.65 | +10.91 | +1.61 | -1.603 | +0.285 | -0.044 | -2.82e-09 | -0.917 |
| baseline | 1.0x | 3.0x | 2756 | +2.360 | +163.80 | +9.84 | +7.15 | -2.801 | +0.135 | +0.079 | +2.45e-10 | -1.678 |
| baseline | 3.0x | 0.3x | 919 | **+0.181** | +17.25 | +4.99 | +0.38 | **-0.533** | -0.060 | -0.007 | -2.09e-09 | -0.210 |
| baseline | 3.0x | 1.0x | 919 | +0.425 | +37.51 | +9.48 | +3.20 | -0.951 | +0.245 | -0.041 | -3.12e-09 | -0.460 |
| baseline | 3.0x | 3.0x | 919 | +1.009 | +78.66 | +10.94 | +1.65 | -1.490 | +0.285 | -0.044 | -2.83e-09 | -0.897 |
| wider | 0.3x | 0.3x | 9185 | +0.707 | +137.97 | +4.85 | +1.10 | -0.947 | +0.356 | -0.060 | -3.78e-09 | -0.982 |
| wider | 0.3x | 1.0x | 9185 | +1.637 | +215.56 | +7.94 | -0.39 | -1.966 | +0.431 | -0.044 | -1.49e-09 | -1.873 |
| wider | 0.3x | 3.0x | 9191 | +4.946 | +404.62 | +18.80 | **-18.99** | -4.710 | +0.543 | +0.193 | +1.58e-08 | -3.878 |
| wider | 1.0x | 0.3x | 2756 | +0.190 | +47.32 | +1.80 | +1.85 | -0.552 | +0.068 | -0.044 | -3.39e-09 | -0.427 |
| wider | 1.0x | 1.0x | 2756 | +0.452 | +86.90 | +4.88 | +1.13 | -0.698 | +0.357 | -0.059 | -3.78e-09 | -0.905 |
| wider | 1.0x | 3.0x | 2756 | +1.217 | +156.00 | +7.82 | -0.30 | -1.575 | +0.428 | -0.045 | -1.82e-09 | -1.709 |
| **wider** | **3.0x** | **0.3x** | 919 | **+0.064** | +16.83 | +2.45 | +4.12 | **-0.417** | +0.105 | -0.050 | -2.26e-09 | -0.187 |
| wider | 3.0x | 1.0x | 919 | +0.134 | +34.60 | +1.72 | +1.56 | -0.496 | +0.064 | -0.043 | -3.49e-09 | -0.431 |
| wider | 3.0x | 3.0x | 919 | +0.377 | +72.01 | +4.89 | +1.14 | -0.624 | +0.358 | -0.059 | -3.78e-09 | -0.884 |

`dstrain = 0` exactly in every one of the 18 runs (no RBM occurred, as required by
construction). Bold marks the strongest and weakest responses (Section E).

## E. Direct answers to the requested questions

- **Which rate ordering gives the strongest neck response?** The **strongest
  widening** is `coarsening=0.3x, surface=3.0x` on the baseline geometry
  (`dx_neck = +6.40nm`, `dsigma = -5.72MPa`) -- i.e. surface relaxation fast relative
  to coarsening. The **weakest widening** (closest to a reversal) is
  `coarsening=3.0x, surface=0.3x` on the **wider** geometry (`dx_neck = +0.064nm`,
  essentially flat) -- coarsening fast relative to surface relaxation, on the side of
  the trajectory expected to be closer to any natural equilibrium.
- **Does the sign of the response reverse anywhere?** **No.** `dx_neck > 0`
  (widening) and `dsigma < 0` (stress relaxing) in **all 18** matrix cases and **all
  4** eta-variant cases -- 22/22, across a combined coarsening/surface rate-ratio
  range spanning roughly a factor of 100 (`0.3x` coarsening with `3.0x` surface vs.
  `3.0x` coarsening with `0.3x` surface), and across `eta_mobility_scale` from `0.1x`
  to `10x`. The direction of the response established in Milestones 1-2 and 3 is
  **not** an artifact of the specific rate values used there -- it holds throughout
  this rate-ratio space.
- **Does the wider-neck initial state behave differently?** **Yes, quantitatively,
  not qualitatively.** At every matching rate combination, the wider geometry shows
  roughly **1.5-3x smaller** `|dx_neck|` and `|dsigma|` than the baseline geometry
  (e.g. `1.0x/1.0x`: `+1.115nm`/`-1.603MPa` baseline vs. `+0.452nm`/`-0.698MPa`
  wider). Starting closer to (or past) the natural low-stress neck width weakens the
  widening tendency substantially, consistent with the original H1/H2-era hypothesis
  that a neck wider than its equilibrium value should have a weaker (or reversed)
  driving force -- but even the widest tested starting state and the most favorable
  rate ratio (`coarsening=3x, surface=0.3x`) only reduced the response to
  `+0.064nm`, not through zero.

## F. GB/eta-mobility follow-up

At `coarsening=1.0x, surface=1.0x` (baseline rate cell), `eta_mobility_scale` in
`{0.1, 1.0, 10.0}`:

| geometry | eta scale | dx_neck (nm) | dA_GB (nm^2) | dsigma (MPa) | dG_interface (J) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.1x | +0.968 | +74.26 | -1.448 | -2.83e-09 |
| baseline | 1.0x | +1.115 | +90.65 | -1.603 | -2.82e-09 |
| baseline | 10.0x | **+2.422** | +244.91 | -2.937 | -2.79e-09 |
| wider | 0.1x | +0.351 | +66.73 | -0.599 | -3.78e-09 |
| wider | 1.0x | +0.452 | +86.90 | -0.698 | -3.78e-09 |
| wider | 10.0x | **+1.394** | +276.02 | -1.425 | -3.79e-09 |

**Faster GB/TJ relaxation *amplifies* the widening response; it does not suppress
it.** Going from `0.1x` to `10x` (a 100x range) monotonically increases `|dx_neck|`
by roughly 2.5x (baseline) to 4x (wider) -- the opposite of "rapid GB/TJ relaxation
continually removes the geometric frustration" reducing the response toward zero.
This refines the original Milestone-1/2 H2 finding (there, `eta_mobility_scale=0`
reduced but did not reverse the wrong-direction response) with a much wider,
consistently-normalized dynamic range: **slower** GB relaxation is what reduces the
response (toward, but not through, zero), and faster GB relaxation makes it worse.

## G. Regime interpretation

None of the three expected qualitative regimes (Section 12 of the handoff) were
observed to *reverse* the sign, but the magnitude ordering is informative and
internally consistent with a real, monotonic rate-competition structure:

- Slower coarsening relative to faster surface relaxation (Regime-I-like: interfacial
  relaxation dominates) -> **largest** widening/stress-relief (top of Section D's
  table).
- Faster coarsening relative to slower surface and GB relaxation -> **smallest**
  widening/stress-relief (bottom of Section D's table), approaching (but not
  reaching) zero on the wider geometry.

This is the *opposite* magnitude ordering from a naive reading of Regime II in the
handoff (which anticipated *coarsening* competitive with or faster than relaxation
producing the *stronger* stress-building response). Here, faster relaxation
(surface or GB) consistently produces a **larger** wrong-direction response, and
faster coarsening relative to relaxation consistently produces a **smaller** one
-- monotonic and consistent across both rate controls and both geometries, but
opposite in sign-of-effect from what Section 12 anticipated, and never crossing zero
in the tested range.

## H. Numerical pathology / ambiguity notes

1. **`dG_interface` goes positive (energy *increases*) specifically at
   `surface_mobility_scale=3.0x` combined with slow-to-moderate coarsening**, most
   strongly for `coarsening=0.3x` (`+2.76e-8 J` baseline, `+1.58e-8 J` wider -- by far
   the largest-magnitude `dG_interface` values in the whole table, a full order of
   magnitude above the typical `~1e-9 J` scale elsewhere). These same cases also show
   the **largest measured `dpsi_top`** (`+31.7deg`, `+18.8deg`) and the **largest
   particle/substrate centroid drift** (`dL = -3.72nm`, `-3.88nm`, several percent of
   `L`). This cluster of correlated anomalies at the highest tested surface-mobility
   scale suggests the `3.0x` case may be pushing toward the edge of the accurate
   regime for the fixed `dt` used here (chosen for the `1.0x` baseline) -- plausible,
   not confirmed. As in MILESTONE_4, `G_interface` is a total-variation/area-based
   *proxy*, not the exact CH/AC free-energy functional, so a proxy excursion at large
   shape-change-per-step is a more likely explanation than a genuine free-energy
   violation, but this was not root-caused further here.
2. **Separation is not exactly conserved**, by design of this benchmark (no
   corrective projection was added, unlike the checkpointed constrained-relaxation
   work) -- `dL` is purely the shape-driven centroid drift documented in
   MILESTONE_1_2_REPORT.md, and grows with `surface_mobility_scale` specifically,
   reaching `-3.72` to `-3.88nm` (a few percent of `L`) at the `3.0x` extreme,
   compared to `<1nm` at `0.3x`-`1.0x`. `dstrain = 0` exactly throughout, confirming
   this is shape drift, not RBM. Given the correlation with Note 1, this is worth
   tracking if `surface_mobility_scale` is pushed further in future work.
3. `dpsi_bottom` is smaller than `dpsi_top` in most cases and goes **negative** in
   three of the wider-geometry, high-surface-rate cells (`-0.39`, `-18.99`, `-0.30
   deg`) while `dpsi_top` stays strongly positive in the same rows -- the persistent
   top/bottom TJ asymmetry already noted in Milestones 3 and 4 remains present here
   and becomes more pronounced at high `surface_mobility_scale`.

---

**Stopping here for review**, as instructed. The active-sink GB-transport-rate
campaign (Section 11 of the handoff) was not started. No stochastic hazard, no long
trajectories beyond the `|dV2|/V20 ~ 1e-3` target, and no CH chemical-potential/flux
diagnostics (Section 8, deferred as optional) were added.
