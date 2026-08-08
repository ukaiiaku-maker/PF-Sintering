# Differential-coarsening audit: separating coarsening's effect from background capillary relaxation

Scope: isolate the coarsening-*induced* perturbation to the physical
solid-solid contact from the background capillary relaxation the same
non-equilibrated initial geometry would undergo anyway, via paired
trajectories (C0 = capillary-only control, C1 = capillary + coarsening)
started from an identical initial state and compared at matched physical
time. **Stopping here for review, as instructed** — no production CH,
Ostwald, eta, projection, hazard, or RBM equation was modified (one
config-construction guard fix is documented and flagged in Section 0b).

## 0. Headline result

**Outcome B.** With the common capillary-relaxation trajectory subtracted
out, coarsening itself produces additional, real contact widening — not
just an unresolved-geometry artifact. `delta_L_coarsening(t) =
L_contact_TJ_sub_C1(t) - L_contact_TJ_sub_C0(t)` is **positive at every
sampled step of the entire 2759-step trajectory** (never once negative),
grows smoothly and very nearly linearly with `|dV2|/V20` (R² = 0.9985),
scales monotonically (apparently linearly) with `coarsening_rate_scale`
across a 33x range (0.3x to 10x), keeps the same sign at a 2x finer grid
once the timestep is independently converged at that grid, and keeps the
same sign at a 4x narrower starting contact (5nm vs 20nm overlap). This
directly answers the question the milestone was designed to ask: coarsening
is not merely coincidental with the widening Milestone 6C observed — it is
a genuine additional driver of it, on top of whatever capillary relaxation
the same (non-equilibrated) initial geometry would do on its own.

A second, equally important finding: `delta_sigma(t)` (the legacy local
stress, C1 minus C0) is **consistently negative** throughout — coarsening
lowers the local neck stress relative to the capillary-only control, the
opposite of the direction `PHYSICS_BACKGROUND.md` Section 4 requires
(`dSigma_local > 0` under sink-off coarsening). Section 21's Outcome B
interpretation therefore applies directly: **the existing coarsening + CH
thermodynamics genuinely drive additional contact widening, and the
CH/Ostwald coupling itself needs auditing** — not just the diagnostic used
to measure it (which Milestone 6C already ruled out).

### 0b. One config-construction fix required to run this milestone

`coarsening_rate_scale=0.0` (needed for the C0 control) raised
`ZeroDivisionError` in `build_params`'s `p.tau_ripening=20./c.coarsening_rate_scale`
— Python's float division, unlike IEEE-754/numpy, raises rather than
returning `inf`. This is a parameter-construction guard, not a change to any
evolution equation: `tau_ripening=math.inf` is exactly the semantics the
field's own docstring implies ("multiplies the Ostwald/reservoir transfer
rate"), and Section 12 below verifies analytically and numerically that it
makes `ostwald_substrate` an *exact*, bit-for-bit no-op (`tr =
min(V*dt/inf, .002*V) = 0.0` exactly, so every downstream quantity derived
from `tr` is exactly zero). Fixed with a one-line guard in
`pf_sintering/model.py` (`p.tau_ripening=math.inf if
c.coarsening_rate_scale==0 else 20./c.coarsening_rate_scale`), covered by
new tests, full 99/99 suite re-verified passing after the change.

## 1. Checkpoint / test state

- Branch `codex/coarsening-stress-buildup`, starting HEAD `d3e4207`
  (Milestone 6C), clean working tree, 92/92 tests passing — verified before
  any change.
- New this session: `pf_sintering/differential_coarsening.py` (paired-
  trajectory machinery), `tests/test_differential_coarsening.py` (7 tests),
  the one-line `model.py` guard above, `scripts/differential_coarsening_audit.py`.
- Full suite after all changes: **99/99 passing**.

## 2. Exact C0/C1 definitions

Both trajectories use `pf_sintering.differential_coarsening._step_once`,
the identical production operator sequence used everywhere in this project
(`operator_ledger.run_ledger_step_v3` / `rate_competition.run_sinkoff_trajectory`):
CH → mass-preserving eta projection → Ostwald → structural (eta) relaxation
→ mass-preserving eta projection, sink and RBM never called (not merely
non-firing).

- **C0 (control)**: `coarsening_rate_scale=0.0` → `tau_ripening=inf` →
  `ostwald_substrate` is an exact no-op (Section 12). CH, eta relaxation,
  and both mass-preserving projections remain fully active.
- **C1 (physical)**: `coarsening_rate_scale=3.0` (the Milestone 6/6B/6C
  primary-case rate), everything else identical.

Both share every other `ModelConfig` field (`preset=dev`, `R2=80nm`,
`aspect_ratio=2`, `short_plane`, `overlap=20nm`, `surface_mobility_scale=0.3`,
`eta_mobility_scale=1.0`, `dx=5nm`), and both are stepped from **two
independent `.copy()`s of one shared `initialize_fields()` call** — never
initialized separately (`differential_coarsening.run_paired_trajectory`
takes a single `f0,e1_0,e2_0,e3_0` and copies it once for each branch).

`p_c0.dt == p_c1.dt` is asserted (raises `ValueError` otherwise, tested in
`test_paired_trajectory_rejects_mismatched_dt`) rather than assumed:
`build_params`'s `dt = min(CFL*dx**4/(M_f*k_f), 1e-5)` depends on
`surface_mobility_scale` and geometry only, not on `coarsening_rate_scale`,
so C0/C1 share `dt` by construction and stepping both the same number of
times is sufficient for matched physical time.

## 3. Matched-time paired trajectories

Primary case run to `|dV2|/V20 = 3e-3` in one paired pass (2759 steps,
sampled every step for both branches), with the 3e-4/1.5e-3/3e-3 milestones
read off that single trajectory rather than run independently:

| `\|dV2\|/V20` target | step idx | actual `dV2/V20` | `L_contact_TJ_sub` C0 (nm) | C1 (nm) | `delta_L_coarsening` (nm) |
|---|---:|---:|---:|---:|---:|
| 3e-4 | 276 | -3.0062e-04 | 74.87892 | 74.88430 | **+0.005388** |
| 1.5e-3 | 1378 | -1.5000e-03 | 77.19179 | 77.22314 | **+0.031352** |
| 3e-3 | 2759 | -3.0010e-03 | 78.50603 | 78.57255 | **+0.066526** |

(`delta_L_GB_geom_sub` tracks the same pattern and sign throughout:
+0.006624, +0.038307, +0.081503nm at the same three milestones.)

## 4. `delta_L_coarsening(t)` and Table A/B (raw C0(t), C1(t), and their difference)

Sampled every one of the 2759 steps; representative rows:

| step | `dV2/V20` (C1) | `L_contact_TJ_sub` C0 (nm) | C1 (nm) | `delta_L_coarsening` (nm) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 73.65336 | 73.65336 | +0.000000 |
| 276 | 3.006e-4 | 74.87892 | 74.88430 | +0.005388 |
| 552 | 6.011e-4 | — | — | +0.012028 |
| 828 | 9.016e-4 | — | — | +0.018939 |
| 1104 | 1.202e-3 | — | — | +0.025266 |
| 1380 | 1.502e-3 | — | — | +0.031394 |
| 1656 | 1.802e-3 | — | — | +0.037228 |
| 1932 | 2.102e-3 | — | — | +0.044398 |
| 2208 | 2.402e-3 | — | — | +0.052124 |
| 2484 | 2.702e-3 | — | — | +0.059443 |
| 2759 | 3.001e-3 | 78.50603 | 78.57255 | +0.066526 |

`delta_L_coarsening` is **monotonically non-decreasing over the entire
trajectory** (verified programmatically, no exceptions) and never negative
even once — Tables A and B are effectively the same clean, gap-free curve;
there is no point in the trajectory where subtracting the capillary-only
background flips the sign or introduces noise large enough to obscure the
trend.

## 5. Table C: `delta_L_coarsening` vs. `1 - V2_C1/V20`

Linear fit over the full 2759-point trajectory: slope = **22.20 nm per unit
`|dV2|/V20`**, intercept = -0.0014nm (i.e. passes very close to the origin,
as it must since both branches start identical), **R² = 0.9985** — an
excellent linear fit, not merely a same-sign correlation. This is a
substantially cleaner, more decisive signal than any single-run raw
`L_contact_TJ_sub(t)` trace could give, precisely because subtracting the
shared capillary-relaxation background removes the dominant common-mode
trend and leaves only the coarsening-attributable perturbation.

## 6. Compact coarsening-rate series (against the same C0 reference)

`coarsening_rate_scale ∈ {0, 0.3, 1, 3, 10}`, all other rates fixed
(`surface_mobility_scale=0.3`, `eta_mobility_scale=1.0`), all run for the
**same** `N_ref=276` steps (the step count at which the primary C1=3x case
reaches `|dV2|/V20=3e-4`) from the same shared initial state — matched
physical time by construction, one shared C0 reference reused for every
comparison rather than five independent controls:

| `coarsening_rate_scale` | `dV2/V20` | `L_contact_TJ_sub(N_ref)` (nm) | `L(rate) - L(rate=0)` (nm) | ratio to rate |
|---:|---:|---:|---:|---:|
| 0 (reference) | +2.2e-15 | 74.87892 | — (reference) | — |
| 0.3 | -3.007e-05 | 74.87952 | +0.00054 | 0.00179/unit |
| 1 | -1.002e-04 | 74.88074 | +0.00179 | 0.00179/unit |
| 3 | -3.006e-04 | 74.88430 | +0.00539 | 0.00180/unit |
| 10 | -1.002e-03 | 74.89681 | +0.01790 | 0.00179/unit |

The incremental widening is **monotonically increasing with rate and
strikingly close to linear in `coarsening_rate_scale`** over this 33x range
(the ratio `[L(rate)-L(0)]/rate` is 0.00179-0.00180nm/unit at every one of
the four nonzero rates tested, essentially constant). Faster coarsening
produces a proportionally larger positive (widening) departure from the
zero-coarsening trajectory — directly answering the "coarsening rate vs.
interfacial response rate" question posed in Section 9 of the handoff: at
this `surface_mobility_scale`, there is no sign reversal or saturation
anywhere in the tested range, only a proportionally growing widening bias.

## 7. Local psi/curvature/TJ-force differences

At the 3e-4/1.5e-3/3e-3 milestones (`delta_X = X_C1 - X_C0`):

| milestone | `delta_sigma` (MPa) | `delta_F_TJ_mag_top` (N/m) | `delta_F_TJ_mag_bottom` | `delta_F_gb_normal_top` | `delta_F_gb_normal_bottom` |
|---|---:|---:|---:|---:|---:|
| 3e-4 | **-0.00880** | +9.005e-04 | -2.560e-04 | +1.020e-03 | +7.491e-05 |
| 1.5e-3 | **-0.04361** | +3.952e-03 | -1.153e-03 | +4.116e-03 | +2.955e-04 |
| 3e-3 | **-0.08698** | +7.093e-03 | -2.373e-03 | +7.588e-03 | +8.374e-04 |

`delta_sigma` grows monotonically **more negative** — coarsening
systematically lowers the legacy local stress relative to the capillary-only
control, consistent with (and quantitatively explaining) the widening
contact: `sigma_lt ~ 1/x_neck`-type scaling means a wider contact reads as
lower stress. The two TJs respond asymmetrically: the top TJ's force
imbalance magnitude grows under coarsening (`delta_F_TJ_mag_top>0`) while
the bottom TJ's shrinks (`delta_F_TJ_mag_bottom<0|`) — coarsening is not
symmetric between the two junctions even though the geometry itself is
close to top/bottom symmetric at `t=0`.

`delta_psi_top` stays at or below ~0.01deg (median `|delta_psi|` over all
2760 samples = 0.0039deg) for the great majority of the trajectory, with one
notable exception: a persistent step-change of **+2.09deg** appears in the
C1 branch only, starting at step 1030 and persisting afterward (31/2760
samples, ~1.1%, show `|delta_psi|>0.1deg`, essentially all clustered near
this one event). Inspecting the raw values shows this is a genuine
discontinuity in the *measured* `psi_deg_top` (a jump from ~66.48deg to
~68.57deg between two adjacent steps, then smoothly varying afterward), not
a smoothly accumulating physical rotation — consistent with a resolution
transient in `tj_force`'s circle-crossing dihedral measurement (documented
in that module as capable of resolution failure/reassignment at the branch
level) rather than a real ~2deg instantaneous physical reorientation. It
does not appear in `L_contact_TJ_sub` at the same step (no
correspondingly-sized jump there), so it does not affect Sections 4-6's
conclusions; flagged here as a diagnostic-robustness note for `psi`
specifically, not a finding about the physics.

## 8. `F_TJ` interpretation

Per the handoff's explicit instruction, `F_TJ = xi_s1+xi_s2+xi_gb` and its
components are reported above strictly as the local Young-Herring
junction-force-balance residual — a measure of how far the TJ sits from
local capillary equilibrium — and are **not** used here as, or substituted
for, the nucleation-relevant sintering stress. The asymmetric top/bottom
`delta_F_TJ_mag` response (Section 7) is noted as a geometric/mechanistic
detail worth carrying into any future force-conjugate-to-translation
formulation, not as evidence for what that formulation should be.

## 9. CH dt/dx stability audit

`build_params`: `p.k_f = 3*gamma_s*W` with `W = interface_cells*dx`
(`interface_cells=4` by default), so `k_f` is linear in `dx`. `M_f_base =
(20e-9)**4/(tau_target*k_f)`, so `M_f = M_f_base*surface_mobility_scale` is
linear in `1/dx`. The product `M_f*k_f = (20e-9)**4 * surface_mobility_scale
/ tau_target` therefore **cancels every dependence on `dx` and `W`
entirely** — verified numerically: `M_f*k_f = 4.800000e-32` exactly, at both
`dx=5nm` and `dx=2.5nm`.

Since `dt = min(CFL*dx**4/(M_f*k_f), 1e-5)`, and `M_f*k_f` is dx-independent,
the *un-clamped* CFL value scales as `dx**4`, exactly the fourth-order
spatial-stability scaling expected for a Cahn-Hilliard-type (biharmonic)
evolution operator:

| `dx` | `dt_raw` (pre-clamp, pre-aniso) | 1e-5s clamp active? | final `dt` |
|---:|---:|---:|---:|
| 5.00nm | 5.2083e-04s | **yes** | 7.262370e-06s |
| 2.50nm | 3.2552e-05s | **yes** | 7.262370e-06s |
| 1.25nm | 2.0345e-06s | no | 1.477533e-06s |

**Why `dt` was unchanged from `dx=5nm` to `dx=2.5nm` in Milestone 6C**: not
because of any explicit dx-appropriate rule, but because the raw CFL-derived
value at *both* resolutions (5.2e-4s and 3.3e-5s) still exceeds an unrelated
fixed `1e-5`s ceiling, so `min(...)` returns that ceiling at both — which is
then divided by the same fixed (dx-independent) anisotropy factor `fac`,
giving the identical final `dt=7.262370e-06s` bit-for-bit at both
resolutions purely by coincidence of both raw values exceeding the same
cap, not by design. **The solved crossover** where the CFL formula alone
first reaches the 1e-5s ceiling is `dx ≈ 1.86nm` — below that (e.g.
`dx=1.25nm` above), the natural `dx**4` stability scaling finally becomes
the binding constraint and `dt` starts shrinking with resolution as
intended. This is not obviously a bug (a fixed conservative cap that happens
to dominate is still a valid, stable choice), but it does mean **the actual
integration timestep at `dx=5nm` and `dx=2.5nm` was never chosen for
resolution-appropriate accuracy** — it must be independently verified for
time-step convergence at each resolution before its output is trusted
(Section 10).

## 10. Fixed-grid timestep-sensitivity (matched physical time, `t_final=1.0894e-3`s, `n_bench=150` steps at natural `dt`)

**Raw C1 trajectory, dx=5nm** (`L_contact_TJ_sub`, nm): dt=74.359428,
dt/2=74.359190, dt/4=74.359079, dt/8=74.359022 — successive differences
-0.000238, -0.000111, -0.000057nm (ratio ~0.48-0.51, consistent with
first-order/explicit-Euler `O(dt)` convergence). Total spread across all
four dt's: 0.0004nm out of ~74.36nm (~5e-4%) — **already well converged at
the natural (clamped) dt.**

**Raw C1 trajectory, dx=2.5nm**: dt=70.746153, dt/2=70.743605,
dt/4=70.742314, dt/8=70.741657nm — differences -0.002548, -0.001291,
-0.000657nm (ratio ~0.51 each halving, same first-order pattern) — **not
yet fully converged at the natural dt**: Richardson-extrapolating the
geometric series suggests a residual error at dt/8 of roughly another
0.0006-0.0007nm, i.e. the natural-dt value (70.746153nm) differs from the
true converged limit by about 0.0046nm (~0.0065%) — small in absolute terms
but non-negligible relative to some of the differential signals in Section
7.

**Paired differential `delta_L_coarsening`, dx=5nm**: 0.002710, 0.002711,
0.002714, 0.002711nm across dt, dt/2, dt/4, dt/8 — flat to the 4th decimal
nm at every dt tested, i.e. converged even more tightly than the raw
trajectory already was.

**Paired differential `delta_L_coarsening`, dx=2.5nm**: 0.029798, 0.030050,
0.030061, 0.030124nm — differences +0.000252, +0.000011, +0.000063nm: not a
clean monotone geometric series like the raw trajectory, but every value is
within ~1% of the dt/8 value already by dt/2, and within ~0.2% by dt/4. This
confirms the handoff's expectation (Section 14): **the paired subtraction
cancels most of the raw trajectory's common discretization error**, so the
coarsening-attributable signal converges materially faster in `dt` than the
absolute contact-length trajectory does, even at the resolution (dx=2.5nm)
where the raw trajectory itself is not yet fully time-converged.

## 11. dt-converged coarse/fine grid comparison

Based on Section 10, `dx=5nm` at the natural (clamped) `dt` is adequate;
`dx=2.5nm` was re-run at `dt/4` (adequate per the paired-convergence
evidence above), same `|dV2|/V20=3e-4` target:

| grid | `dt` used | n_steps | `dV2/V20` | `delta_L_coarsening` (nm) | `dL_contact_TJ_sub`(C1)`/dt` (m/s) |
|---|---:|---:|---:|---:|---:|
| dx=5nm | 7.2624e-06s (natural) | 276 | -3.0062e-04 | **+0.005388** | 6.14e-07 |
| dx=2.5nm | 1.8156e-06s (dt/4) | 1102 | -3.0007e-04 | **+0.040462** | 2.80e-06 |

**Same sign at both resolutions, once each grid's own dt is independently
adequate** — the ~7.5x magnitude difference between grids (consistent in
direction, though not literally reproducing Milestone 6C's raw ~4.5x ratio,
since 6C compared un-time-converged absolute magnitudes rather than this
milestone's dt-converged differential quantity) is a separate, genuine
spatial-resolution sensitivity that should be kept in mind for any future
quantitative (not merely directional) claim, but it does not affect the
sign-stability conclusion this milestone needed.

## 12. Isotropic-mode exact free-energy check

Reconstructed for `use_aniso_surface=False` only (the anisotropic
production default's gradient-energy functional is a standard 2D
orientation-dependent/Kobayashi-type construction consistent with the
code's `div(k_f*(a²gx-a·ap·gy), k_f*(a²gy+a·ap·gx))` formula, but was not
independently re-derived and verified to closure here — flagged rather than
asserted, per the handoff's explicit permission to skip an unclean
reconstruction):

`F = sum[(W_f/2)*f²*(1-f)² + Wc*eta2*(f²/2-f) + (k_f/2)*|grad f|²] * dx²`

Derivation: `d/df[(W_f/2)f²(1-f)²] = W_f*f(1-f)(1-2f)` and
`d/df[Wc*eta2*(f²/2-f)] = Wc*eta2*(f-1) = -Wc*eta2*(1-f)` together reproduce
`mu0 = W_f*f(1-f)(1-2f) - Wc*eta2*(1-fb)` exactly (away from the `[0,1]`
clip boundary); `-k_f*lap9(f,dx)` (the production gradient term) is the
Euler-Lagrange derivative of `(k_f/2)|grad f|²` in the continuum, though
**not exactly** of this particular discrete Dirichlet-energy sum computed
via central differences (`lap9` is a 9-point compact/rotated-Laplacian
stencil, not the discrete Euler-Lagrange operator of the naive
nearest-neighbor `gx²+gy²` sum) — the check below is therefore expected to
hold only to good approximation, not floating-point closure.

Ran 80 CH steps of the isotropic C0 control (`dx=5nm`): `F[0]=-5.21117042e-05`
→ `F[79]=-5.21105580e-05`, a net **increase** of `+1.146e-9` (~2.2e-5
relative). Every one of the 80 individual steps shows a tiny increase
(largest single-step: `+1.679e-11`, ~3.2e-7 relative to `|F|`) rather than
occasional random-sign noise — consistent with a small, systematic
`lap9`-vs-naive-gradient-energy stencil mismatch exactly as flagged above,
not a violation of the underlying continuum gradient-flow structure (which
is a model-independent identity, `dF_continuum/dt = -∫M|grad(mu)|²dx ≤ 0`
for any `M≥0`, itself satisfied here since `M = min(M_f*(16f²(1-f)²)²,
M_f) ≥ 0` everywhere by construction). Magnitude is utterly negligible next
to the `~1e-4` to `~1e-3` relative `dV2` changes under study. `total_f`
drift over the same 80 steps: exactly `0.0` (relative), confirming mass
conservation holds independent of this energy-bookkeeping caveat.

## 13. 5nm-overlap paired cross-check

Same rates, same `|dV2|/V20=3e-4` target, `initial_overlap=5nm` instead of
20nm:

| | 20nm (primary) | 5nm (baseline) |
|---|---:|---:|
| n_steps | 276 | 276 |
| `dV2/V20` | -3.0062e-04 | -3.0062e-04 |
| `delta_L_coarsening` (nm) | +0.005388 | **+0.004928** |
| `delta_L_GB_geom_sub` (nm) | +0.006624 | +0.005353 |

**Same sign, similar magnitude, at both starting overlaps** — no
neck-stability sign crossover between 5nm and 20nm initial overlap under
these rate settings (consistent with Milestone 6C's baseline cross-check
finding for the raw, non-differential quantity).

## 14. Outcome classification

**Outcome B**, decisively: `delta_L_coarsening(t) > 0` throughout the entire
3e-3 trajectory (never negative, R²=0.9985 linear in `|dV2|/V20`), growing
monotonically with `coarsening_rate_scale` across a 33x range, same sign at
a dt-independently-converged 2x finer grid, and same sign at a 4x narrower
starting contact. This rules out the concern that motivated this milestone
— that Milestone 6C's widening was purely a non-equilibrated-geometry
capillary-relaxation artifact common to any trajectory regardless of
coarsening (that would have been Outcome C, `delta_L_coarsening≈0`, or
would have required occasional sign changes inconsistent with what was
observed). Coarsening is a genuine, quantitatively resolvable, additional
driver of contact widening in this model, and (Section 7) it does so while
lowering, not raising, the local `sigma` relative to the capillary-only
control — the wrong direction for `PHYSICS_BACKGROUND.md`'s required
`dSigma_local > 0` sink-off signature.

## 15. Recommended next physics step (NOT implemented)

Given Outcome B, the next diagnostic step should audit the specific
mechanism by which coarsening (Ostwald mass removal) feeds back into CH's
local behavior near the neck — not the hazard/stress model, which remains
out of scope until the geometry mechanism itself is understood.
`PHYSICS_BACKGROUND.md` Section 8 (H1) already names the leading suspect:
**the current Ostwald operator's neck-exclusion/localization factor**
(`ostwald_substrate`'s `incl` mask, active by default —
`reservoir_neck_unprotected=False` throughout every run in this milestone,
including the one that produced the Outcome B result above). The smallest,
most direct next test: **re-run this exact paired C0/C1 differential
experiment (the machinery is already built and validated —
`pf_sintering.differential_coarsening.run_paired_trajectory`, one new
`ModelConfig(reservoir_neck_unprotected=True, ...)` per branch) and check
whether `delta_L_coarsening(t)` stays positive, shrinks, or flips sign.**
If disabling neck protection removes or reverses the coarsening-attributable
widening, that directly implicates H1 (the reservoir preferentially removing
material away from the neck, leaving the neck itself to relax/widen under
ordinary CH surface diffusion once released from local mass balance) as the
mechanism, and the fix belongs in the Ostwald/reservoir formulation, not in
CH or the eta/projection machinery audited so far. If it does not change the
sign, H2 (GB/eta mobility, already independently controllable via
`eta_mobility_scale`) or the CH/Ostwald free-energy coupling itself should
be audited next, in that order, per `CODEX_PHYSICS_SEQUENCE.md`'s
H1-then-H2 sequencing.

**STOP. Do not alter production physics before review.**
