# Fixed-physics discretization and paired operator causal audit

Scope: (1) verify Milestone 7's grid comparison represented one fixed
diffuse-interface physical model, not a confound between mesh refinement and
regularization-length/mobility changes; (2) find which production operator,
and which physical mechanism within it, actually generates
`delta_L_coarsening` — not just that CH's absolute trajectory is large.
**Stopping here for review, as instructed** — no production CH, Ostwald,
eta, projection, hazard, or RBM equation was modified; two new
diagnostic-only, default-preserving `ModelConfig` fields were added
(Section 3/4).

## 0. Headline result

**The Milestone 7 finding survives fixed-physics scrutiny, and its cause is
now identified concretely.** Milestone 7's dx=5nm vs dx=2.5nm comparison
confounded mesh refinement with a genuine physical-model change: the
production default ties both the diffuse-interface width `W` and the
eta/structural-relaxation diffusivity to `dx`. Under a corrected comparison
that holds both fixed (Section 6), `delta_L_coarsening` stays **positive at
both resolutions** (+0.0054nm at dx=5nm, +0.0142nm at dx=2.5nm) — the sign
is robust, and the magnitude ratio between grids shrinks from Milestone 7's
7.5x to 2.6x, meaning part of Milestone 7's grid sensitivity really was the
W/dx confound, though a real (smaller) spatial-discretization sensitivity
remains.

The paired operator ledger (Section 8/9) shows the coarsening-attributable
widening is generated **overwhelmingly directly by the Ostwald operator
itself** (95.1% of the total `delta_L_coarsening`, from the first step
onward), not by a delayed CH response to Ostwald-perturbed geometry — though
a real, non-negligible, and *growing* secondary CH contribution is also
present (5% of the total, nearly tripling from the trajectory's first half
to its second half). The O0/O1/O2 mechanism-isolation decomposition
(Section 11) then identifies **which half of the Ostwald operator is
responsible**: removal alone would *shrink* the contact
(`delta_L_coarsening = -0.0113nm`, the physically desired direction), while
addition alone *widens* it even more than normal operation
(`+0.0167nm` vs. normal's `+0.0054nm`). **The destination-side
redeposition pattern, not the source-side removal, is what drives the
widening.** H1 (neck-protection) is confirmed negligible at full scale
(Section 7, ~4% effect, same sign). CH's own numerics are confirmed
thermodynamically sound (Sections 12-13): the isotropic discrete CH operator
exactly dissipates its own exact discrete free energy at every timestep
tested — the problem is not a CH implementation defect, it is the physical
state Ostwald's addition step hands to CH.

## 1. Checkpoint / test state

- Branch `codex/coarsening-stress-buildup`, starting HEAD `f62f064`
  (Milestone 7), clean working tree, 99/99 tests passing — verified before
  any change.
- New this session: two diagnostic `ModelConfig` fields in `model.py`
  (Section 3/4), `pf_sintering/paired_operator_ledger.py`,
  `pf_sintering/ch_exact_energy.py`, an `ostwald_fn` parameter added to
  `differential_coarsening._step_once`/`run_single_trajectory` (default
  preserves production behavior), `scripts/fixed_physics_paired_operator_audit.py`,
  and four new test files.
- Full suite after all changes: **112/112 passing**.

## 2. Every parameter that changes when `dx` changes (production default config)

`build_params`: `W = interface_cells*dx` (default `interface_cells=4.0`),
so refining the mesh **also refines the physical diffuse-interface width**
in the default configuration — confirmed numerically:

| `dx` | `interface_width` | `W/dx` | `k_f` | `W_f` | `k_eta` | `W_cpl_f` | `M_f` | `M_eta` | `M_f*k_f` | `dt` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.00nm | 20.00nm | 4.0 | 6.000e-08 | 6.000e+08 | 5.996e-08 | 1.799e+09 | 8.000e-25 | 1.067e-09 | 4.8e-32 | 7.2624e-06s |
| 2.50nm | 10.00nm | 4.0 | 3.000e-08 | 1.200e+09 | 2.998e-08 | 3.598e+09 | 1.600e-24 | 8.533e-09 | 4.8e-32 | 7.2624e-06s |
| 1.25nm | 5.00nm | 4.0 | 1.500e-08 | 2.400e+09 | 1.499e-08 | 7.195e+09 | 3.200e-24 | 6.827e-08 | 4.8e-32 | 1.4775e-06s |

`k_f`/`k_eta` scale linearly with `dx` (via `W`); `W_f`/`W_cpl_f`/`M_f`
scale as `1/dx`; `M_f*k_f` is exactly `dx`-independent (Milestone 7 Section
9, re-confirmed); **`M_eta` scales as `1/dx^3`** (Section 4). Ostwald's
exclusion-window cell counts (`max(5,round(2W/dx))`, `max(5,round(3W/dx))`)
stay fixed at 8/12 cells since `W/dx=4` is fixed by `interface_cells`, but
their **physical** width shrinks proportionally with `dx` (40nm/60nm at
dx=5nm down to 10nm/15nm at dx=1.25nm) — Section 5. Domain dimensions
(`Nx*dx`, `Ny*dx`) stay approximately fixed (~600x355nm) since `Nx`/`Ny` are
sized from the same physical `R2`/`aspect_ratio` at every `dx`.

**Conclusion**: Milestone 7's dx=5nm vs. dx=2.5nm comparison changed the
physical diffuse-interface width, the physical Ostwald exclusion-window
size, and (Section 4) the physical eta-relaxation rate simultaneously with
mesh resolution — not a controlled mesh-refinement test of one fixed
physical model.

## 3. Diagnostic fixed-physical-interface-width control

Added `ModelConfig.interface_width_override: float | None = None` (default
`None` preserves `W = interface_cells*dx` exactly — `test_interface_width_override_default_is_none_and_preserves_production_w`).
When set, `build_params` uses `W = interface_width_override` directly. Since
essentially every downstream length-scale-dependent routine in the codebase
(`model.py`'s `curvature`/`measure_dihedral`/`_vapor_normal`/`compute_stress`'s
burrow check, `signed_curvature.py`, `tj_force.py`'s `_radii_for_tj`,
`tj_subgrid.py`'s search radii, `contact_geometry.py`'s snap tolerance, and
Ostwald's exclusion windows) reads `p.interface_width` directly rather than
re-deriving it from `dx`, fixing this one field automatically gives every
one of those routines the same fixed physical support at every resolution —
verified directly for the Ostwald exclusion windows (Section 5) and used
throughout Section 6.

## 4. `M_eta` dx-scaling: not required, and not physically consistent

**Empirically and algebraically confirmed**: the production `M_eta =
M_f_base/dx**2 * 0.01 * eta_mobility_scale` makes the physical
structural-relaxation diffusivity `M_eta*k_eta` scale as `1/dx**2`,
regardless of whether `W` tracks `dx` or is held fixed (`k_f` substitution
shows `W` cancels out of `M_eta*k_eta` entirely — see
`ch_exact_energy`-adjacent derivation in the commit; confirmed by
`test_default_config_m_eta_times_k_eta_scales_as_inverse_dx_squared`).
Direct numerical test: apply `evolve_eta`'s update to the SAME physical
smooth field (fixed wavelength, fixed domain) at three resolutions — `lap9`
itself converges cleanly to the same continuum Laplacian value at every
`dx` (3.915e14, 3.940e14, 3.946e14 m⁻², approaching the analytic
3.948e14 m⁻² as `dx→0`, exactly as a consistent finite-difference operator
should), but the resulting physical rate `M_eta*k_eta*lap9` **quadruples**
each time `dx` halves (0.0250, 0.1008, 0.4038 s⁻¹) — the eta/GB-relaxation
rate for a fixed physical geometry is **not** grid-independent in the
production default: halving `dx` makes structural relaxation happen 4x
faster in physical (nanosecond) time for the same field, a genuine artifact.

**This is not a required consequence of `lap9`'s own discretization**
(which already handles its own `1/dx**2` normalization correctly, as the
convergence above shows) — it is an unwarranted extra factor in `M_eta`'s
own construction. Per instruction, production behavior is unchanged;
`ModelConfig.eta_diffusivity_fixed_physical: bool = False` is added
(default `False` preserves `M_eta = M_f_base/dx**2*...` exactly), and when
`True`, `M_eta = M_f_base*0.01*eta_mobility_scale` (no extra `/dx**2`).
Combined with a fixed `interface_width_override`, `M_eta*k_eta` is then
**exactly `dx`-independent** to 9+ significant figures at dx=5/2.5/1.25nm
(`1.598919e-33` at every resolution — `test_eta_diffusivity_fixed_physical_makes_m_eta_k_eta_dx_independent`).

## 5. Ostwald (and other) length-scale audit under fixed `W`

Under `interface_width_override=20nm`, the exclusion-window cell counts
scale up exactly to keep physical support fixed: `2W/dx` and `3W/dx` cells
are 8/12 at dx=5nm, 16/24 at dx=2.5nm, 32/48 at dx=1.25nm — physical widths
**40.000nm/60.000nm at every resolution tested**, none of them near the
`max(5, ...)` floor (would only bind if `W/dx < 2.5`, far below anything
tested here). Since `p.interface_width` is the single field every other
length-scale-dependent routine reads (Section 3), this same fixed-physical-
support property holds automatically for `tj_force._radii_for_tj`,
`tj_subgrid`'s search radii, `signed_curvature`'s search radius, and
`contact_geometry`'s snap tolerance — confirmed by grep, not just Ostwald.

## 6. True fixed-physics paired grid test

`interface_width_override=20nm`, `eta_diffusivity_fixed_physical=True`,
same physical domain/geometry/rates as the Milestone 7 primary case, `dt`
independently checked (dt, dt/2, dt/4 on the paired differential quantity,
`n_bench=100` natural-dt-equivalent steps) at each grid before the main
comparison:

| dx | dt sensitivity (`delta_L_coarsening`, nm) | chosen dt | n_steps | `dV2/V20` | `delta_L_coarsening` | `delta_L_GB_geom_sub` | `delta_sigma` |
|---|---|---|---:|---:|---:|---:|---:|
| 5nm | dt=0.001750, dt/2=0.001761, dt/4=0.001754 (flat, <0.7% spread) | natural (1.0x) | 276 | -3.0062e-4 | **+0.005397nm** | +0.006635nm | -0.008814MPa |
| 2.5nm | dt=0.007213, dt/2=0.007218, dt/4=0.007241 (flat, <0.4% spread) | dt/4 | 1102 | -3.0007e-4 | **+0.014218nm** | +0.015708nm | -0.008717MPa |

**Same sign at both resolutions**, and — notably — both grids' dt-sensitivity
benchmarks are *already* flat at the natural (clamped) dt under fixed-W (a
tighter convergence than Milestone 7's non-fixed-W dx=2.5nm case showed,
which needed dt/4 to flatten). The magnitude ratio between grids is 2.6x
(14.218/5.397), down from Milestone 7's 7.5x under the confounded
comparison — genuine partial convergence improvement, but **not full
magnitude convergence**: a real spatial-discretization sensitivity remains
even with the physical model held fixed, and any future quantitative
(non-directional) claim from this diagnostic should still be checked at a
third resolution before being trusted numerically.

## 7. Compact H1 confirmation (paired sub-grid metric, full scale)

Primary grid/geometry, `|dV2|/V20=3e-4`, same C0, two C1 variants:

| | `delta_L_coarsening` (nm) |
|---|---:|
| protected (`reservoir_neck_unprotected=False`, default) | +0.005388 |
| unprotected (`reservoir_neck_unprotected=True`) | +0.005191 |

A **3.7% reduction**, same sign, not a qualitative change. This confirms
(now with the qualified continuous sub-grid metric, not the legacy
eta-based proxy the earliest H1 test used) that **neck-exclusion/protection
alone is not the primary mechanism** — consistent with the handoff's note
that the earlier raw H1 test already found almost no difference. H1 is
retired as the leading explanation; Section 11 below identifies the real
driver within the Ostwald operator.

## 8/9. Paired operator ledger and the P1/P2/P3 causal distinction

Applying `operator_ledger._delta`'s own generic dict-subtraction machinery
to the *already-paired* (C1-C0) stage series (rather than to either
branch's absolute state) gives, directly, each production operator's
contribution to `d(delta_L_coarsening)` itself. Full 276-step primary
trajectory, exact closure (all closure errors 0.0):

| operator | `d(delta_L_contact_TJ_sub)` total (nm) | share of total |
|---|---:|---:|
| CH | +0.0002686 | 5.0% |
| post_CH_projection | -0.0000033 | -0.06% |
| **Ostwald** | **+0.0051214** | **95.1%** |
| eta_raw | -0.0000024 | -0.04% |
| post_eta_projection | +0.0000040 | 0.07% |
| **TOTAL** | **+0.005388** | 100% |

**This is CASE P1 (direct Ostwald effect), dominantly**: the operator that
literally differs between C0 and C1 (Ostwald is on in C1, off in C0)
generates 95% of `delta_L_coarsening` directly, in the same step it acts,
not on a delayed subsequent step. This is the opposite pattern from
Milestone 6's single-branch decomposition (where raw CH dominated 96-99% of
the *absolute* `L_contact_TJ_sub` trajectory) — the two decompositions
answer different questions: CH acts nearly identically on both branches
each step (so its direct contribution to the *difference* is small), while
Ostwald is exactly the operator that differs by construction.

**However, a real, growing CASE P2 (delayed CH response) component is also
present** — splitting the trajectory in half:

| | first half (nm) | second half (nm) | growth |
|---|---:|---:|---:|
| CH | +0.000072 | +0.000197 | **2.7x** |
| Ostwald | +0.002412 | +0.002710 | 1.12x |

CH's direct contribution to `delta_L_coarsening` nearly triples from the
first half of the trajectory to the second, while Ostwald's grows only
modestly — CH is a genuine, strengthening secondary channel, not just
numerical noise, even though it remains a small minority of the total at
this horizon (`|dV2|/V20=3e-4`). No evidence for CASE P3 (structural/eta):
`eta_raw` and `post_eta_projection` totals are both below 0.1% of the
total and have no consistent sign relationship to it.

## 10. CH mu/curvature/flux causal chain near the TJs

Tracking C1-C0 differences in `mu`, signed curvature, and CH flux `J`
(bilinearly sampled at each branch's own continuous sub-grid TJ location)
immediately before each of the first 10 CH updates, alongside that step's
direct CH contribution to `delta_L_contact_TJ_sub`:

| step | `delta_mu_top` | `delta_kappa_top` (1/m) | `delta_J_tangent_top` | `d_delta_L_coarsening_CH` (m) |
|---:|---:|---:|---:|---:|
| 2 | -275.2 | -11.90 | +6.14e-15 | +1.97e-17 |
| 4 | -820.3 | -35.70 | +1.89e-14 | +5.95e-17 |
| 6 | -1366.9 | -59.51 | +3.12e-14 | -2.34e-16 |
| 8 | -1901.2 | -83.33 | +3.80e-13 | +1.41e-16 |
| 10 | -2405.9 | -107.10 | -3.46e-13 | -1.98e-16 |

Both TJs show a **consistent, smoothly growing negative `delta_mu`** (C1's
local chemical potential runs increasingly below C0's near both TJs) and a
**consistent, smoothly growing negative `delta_kappa`** (C1's TJ-local
curvature becomes increasingly more concave/neck-like than C0's) at every
step from step 2 onward — a clean, monotonic, non-noisy signal, unlike the
much smaller and noisier `delta_J_tangent`/`delta_J_normal`, which stay 2-4
orders of magnitude below the absolute flux scale (`J_tangent ~ 9.2e-10`)
at every one of the 10 steps. This is consistent with — and mechanistically
explains — Section 8/9's finding that CH's *direct*, single-step
contribution stays at the floating-point-noise level (`~1e-16` to `~1e-17`
m) even while `delta_mu`/`delta_kappa` grow substantially: the CH-flux
perturbation induced by Ostwald's mu/curvature disturbance is real and
growing, but small in *relative* terms at any single step, only becoming
the 5%-and-rising contribution documented in Section 8/9 by accumulating
over hundreds of steps. The mechanistic statement the handoff asked for,
stated at the precision the data supports: **Ostwald's redeposition lowers
the local chemical potential and curvature near both TJs in the coarsening
branch relative to the capillary-only control, and CH responds to that
lowered local `mu`/curvature with a small but persistent, accumulating
tangential-flux bias that itself widens the contact over many steps** — a
real but secondary channel next to Ostwald's own direct, one-step
contribution.

## 11. O0/O1/O2 decomposition with the qualified sub-grid metric

Full 276-step horizon (`N_ref` from the primary case), same shared C0
reference for all three:

| variant | `dV2/V20` | `delta_L_coarsening` (nm) |
|---|---:|---:|
| O0 (normal Ostwald) | -3.006e-04 | **+0.005388** |
| **O1 (removal-only, no redeposition)** | -3.006e-04 | **-0.011284** |
| **O2 (addition-only, no source removal)** | ~0 | **+0.016664** |

**The sign contrast is the key finding**: removing grain-2 mass with no
local redeposition (O1) would, by itself, *shrink* the physical contact —
the physically desired direction — while depositing the same material
pattern onto the receiving grain (O2) *widens* it, more than normal (O0)
operation does. (O1+O2 does not equal O0 by construction — both are
deliberately nonphysical, non-mass-conserving isolations, not a linear
decomposition — but their *signs* are directly informative.) **The
destination-side redeposition pattern of the Ostwald operator, not the
source-side removal, is what drives the observed widening** — a sharper,
more actionable finding than H1 (Section 7), which only tested whether
removal is spatially *excluded* near the neck, not what happens to the
material once it's redeposited.

## 12. Exact isotropic discrete free-energy derivation and verification

Milestone 7's isotropic energy check used a central-difference `|grad f|^2`
gradient term that is not the exact discrete Euler-Lagrange conjugate of
`lap9` (the compact 9-point stencil production `evolve_f` actually
differentiates with) — flagged there as an approximation, not verified.
Fixed here (`pf_sintering/ch_exact_energy.py`):

1. **Symmetry check**: `lap9` is confirmed symmetric under the plain dot
   product, `sum(a*lap9(b,dx)) == sum(b*lap9(a,dx))`, to `~6e-16` relative
   error for random arrays — holds despite the mixed periodic-x/clamped-y
   boundary treatment (`test_lap9_is_symmetric_under_plain_dot_product`).
2. Given that symmetry, `E_grad(f) = -(k_f/2)*dx^2*sum(f*lap9(f,dx))` has
   discrete directional derivative `dx^2*sum((-k_f*lap9(f,dx))*q)` for any
   perturbation `q` — exactly production's isotropic gradient term.
3. Combined with the (already-exact, pointwise) bulk term:
   `F_exact = sum[(W_f/2)f^2(1-f)^2 + Wc*eta2*(f^2/2-f) - (k_f/2)*f*lap9(f,dx)] * dx^2`.
4. **Finite-difference directional-derivative test**:
   `[F(f+eps*q)-F(f-eps*q)]/(2*eps)` vs. `dx^2*sum(mu_isotropic*q)` for a
   smooth random `q`, relative error `3.1e-5` at `eps=1e-2` → `2.4e-8` at
   `eps=1e-4` — clean `O(eps^2)` central-difference convergence down to the
   float64 roundoff floor (`test_exact_energy_directional_derivative_matches_mu_isotropic`).
   **`F_exact` is now genuinely exact**, not approximately so.

## 13. CH-only discrete dissipation test

Isotropic mode, eta/Ostwald/projection all disabled (`evolve_f` only), one
step from a realistic initial state at six geometrically-halved `dt`
(natural down to `dt/32`):

| `dt` | `F_0` | `F_1` | `F_1 <= F_0`? | `(F_1-F_0)/dt` |
|---:|---:|---:|---|---:|
| 1.0000e-05 (natural) | -5.1025184646e-05 | -5.1025222328e-05 | **True** | -3.768174e-06 |
| 5.0000e-06 | same | -5.1025203517e-05 | **True** | -3.774144e-06 |
| 2.5000e-06 | same | -5.1025194089e-05 | **True** | -3.777129e-06 |
| 1.2500e-06 | same | -5.1025189370e-05 | **True** | -3.778621e-06 |
| 6.2500e-07 | same | -5.1025187009e-05 | **True** | -3.779368e-06 |
| 3.1250e-07 | same | -5.1025185828e-05 | **True** | -3.779741e-06 |

`F_1<=F_0` holds at **every** `dt` tested, including the full natural
(largest, un-refined) `dt` — no violation anywhere. `(F_1-F_0)/dt` converges
smoothly and monotonically toward a finite limit (~`-3.7801e-6`) as
`dt→0`, consistent with first-order (explicit-Euler) truncation error
vanishing as expected, not with any instability or sign inconsistency.
**The isotropic discrete CH operator exactly dissipates its own exact
discrete free energy — there is no variational inconsistency in the CH
implementation.**

## 14. Interpretation

**A. Is the positive coarsening differential robust under mesh refinement
for the SAME physical model?** Yes (Section 6) — same sign at dx=5nm and
dx=2.5nm under fixed `W` and fixed eta-diffusivity, with improved (though
not yet complete) magnitude convergence relative to Milestone 7's confounded
comparison.

**B. Is H1 still negligible under the qualified paired sub-grid metric?**
Yes (Section 7) — unprotected reservoir changes `delta_L_coarsening` by
only ~4%, same sign.

**C. Which operator stage first generates `delta_L_coarsening > 0`?**
Ostwald, directly, from the first step (Section 8/9: 95.1% of the total).

**D. Removal, addition, or subsequent CH response?** Primarily destination-
side **addition** (Section 11: O2 alone exceeds O0's widening; O1 alone
reverses the sign to shrinking). CH's subsequent response is real, growing
(nearly tripling first-half to second-half), but a minority contributor
(~5% of the total) at this horizon.

**E. What local change causes the TJs to move apart?** A consistently
negative and growing `delta_mu` and `delta_kappa` at both TJs following
each Ostwald application (Section 10), producing a small, accumulating
tangential-CH-flux bias — individually near the noise floor, cumulatively
real over hundreds of steps.

**F. Does isotropic CH dissipate its own exact discrete energy?** Yes,
unconditionally in the tests run (Section 13) — CH's own numerics are
thermodynamically sound; the problem is entirely in the physical state
Ostwald's addition step produces, not in CH's implementation.

## 15. Recommended smallest physically defensible model change (NOT implemented)

Given Sections 8/9 and 11, the next diagnostic (not yet a physics change)
should isolate **what specifically about the Ostwald addition/redeposition
spatial pattern** drives widening — the destination-side `snk`/`add` field
in `ostwald_substrate`, which distributes redeposited mass following
`surf*e1*incl` (i.e., proportional to the *receiving* grain's own local
free-surface curvature-density `surf=16f²(1-f)²`, weighted toward wherever
`e1`'s surface already is, not explicitly toward or away from the TJ).
Concretely: **compute the near-neck fraction of the O2 addition field**
(`ostwald_diagnostics.near_neck_fractions`, already available and unused
here) to determine whether redeposition is disproportionately occurring
*near* the neck/TJ region (locally growing the receiving grain right where
the contact is measured) versus far from it (only indirectly affecting the
TJ through subsequent CH relaxation). If addition is concentrated near the
neck, the fix belongs in reshaping *where* `add` deposits mass (e.g.
weighting by distance from the TJ, not just by the receiving grain's
existing curvature); if addition is spread over the whole receiving grain
and the effect is still large, the coupling is more likely in how
uniform-grain-growth interacts with the TJ's own force balance, and the
next audit should turn to the F_TJ/configurational-force diagnostics
already validated in `tj_force.py`, per Milestone 6C's original
recommendation. Either way, the fix belongs in the Ostwald
addition/redeposition formulation specifically, not in CH, eta, or the
projection machinery, all three of which are now independently qualified
(Sections 12-13, and Milestone 6C/7's prior audits).

**STOP. Do not alter production physics before review.**
