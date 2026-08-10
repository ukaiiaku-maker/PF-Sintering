# Milestone 15C: Hazard-Driving-Variable Qualification

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup` @ `e778532` (Milestone 15B,
235/235 tests, clean worktree) — verified before starting. Preserves the
unified Milestone-14G free energy, physical `M_GB`, `face_projected`
surface diffusion, and `constrained_tangent_cone_eta_update` GB migration
throughout. sink, hazard, RBM, anisotropy, and independent `M_TJ` stayed
OFF.

## 2. Symmetric exact ownership initializer

Milestone 15B's one-sided cap (`eta2=min(f*phi0, e2_raw)`) is replaced
with a symmetric admissible-band clip:

    eta2_desired = f*phi0
    eta2_min = max(0, f-e1_raw)
    eta2_max = min(f, e2_raw)
    eta2 = clip(eta2_desired, eta2_min, eta2_max)
    eta1 = f - eta2

Algebraic check (also verified numerically): this clip range is always
well-posed (`eta2_min<=eta2_max` for any `e1_raw,e2_raw>=0`,
`f=max(e1_raw,e2_raw)`), and guarantees `0<=eta1<=e1_raw` and
`0<=eta2<=e2_raw` automatically, in *both* directions — Milestone 15B's
one-sided cap only guaranteed `eta2<=e2_raw`, leaving `eta1` potentially
able to exceed `e1_raw` in principle (a narrow imperfection flagged but
not pursued in that milestone).

**For the primary campaign geometry** (A=100nm, lambda=320nm, R2=80nm,
aspect=2, overlap=20nm, W=20nm, gamma_gb=1.0, dx=2.5nm), the symmetric
clip is **bit-identical** to Milestone 15B's one-sided cap: `max(|eta1_new
-eta1_old|)=0.0`, `max(|eta2_new-eta2_old|)=0.0` (0 of 36352 cells
differ). This confirms the one-sided-cap imperfection never actually
triggered for this geometry/parameter regime (the `eta2_min>0` band only
activates where `e2_raw>e1_raw`, which for this geometry only occurs deep
inside the particle, where `phi0` is already saturated near 1 and the
`min(f,e2_raw)` cap was already binding to the same value). The symmetric
form is retained as the more general, more robust construction for other
geometries/parameter regimes not exercised here.

Verified directly (both `"substrate"` and `"sinusoidal_substrate"`
geometries): `sum(eta_i)-f` max abs `=1.1e-16` (roundoff); `eta_i>=0`;
`eta1<=e1_raw`, `eta2<=e2_raw` exactly (`max(eta_i-e_i_raw)=0.0`); `f`,
particle contour, substrate contour, and the `phi=0.5` GB location are all
unchanged (identical `e1_raw`/`e2_raw`/`f` construction, only the
split changed).

## 3. Bounded dynamic non-regression

Because the initial condition is bit-identical to Milestone 15B's for the
primary geometry (Section 2), and no other physics code changed, a fresh
500-step run was used as a direct proof rather than re-running full
trajectories: `run_trajectory` at `dx=2.5nm`, `M_GB=10*M_GB_ref` to
`t=0.005s` reproduces Milestone 15B's own FAST trajectory exactly
(`F=-3.098801e-7`, `L_GB=64.52nm`, `sigma_sint_app=3.2163e7` Pa, matching
to all reported digits). Given bit-identical initial conditions and an
unchanged dynamics pipeline, **Milestone 15B's full FROZEN/FAST/M_s-
perturbation trajectories are exact reproductions of what a fresh Section-
3 rerun would produce** and are reused directly rather than recomputed:
the G2 classification, TJ reversal, GB-length overshoot/recovery, and
`M_s` ordering all survive unchanged (Milestone 15B Sections 4-5).

## 4. Endpoint capillary-force derivation

`capillary_stress.capillary_force_endpoint_form(top_particle_dir,
bot_particle_dir, gamma_s)` (already implemented, Milestone 14C) computes
exactly the Frenet-telescoping identity `F_cap = gamma_s*(t(L)-t(0))`
along the particle-side arc walked continuously from the top TJ to the
bottom TJ: `t(0)=top_particle_dir` (departing the top TJ) and
`t(L)=-bot_particle_dir` (arriving at, not departing from, the bottom TJ,
hence the sign flip) — giving `F_cap_endpoint = -gamma_s*(top_particle_dir
+ bot_particle_dir)`. This is the SAME function used for the "endpoint
form" cross-check in Milestones 15/15B; per Section 4 it is now
**promoted to the primary loading observable**, with
`sigma_sint_endpoint = -(F_cap_endpoint . n_GB)/L_contact`
(`apparent_sintering_stress`, unchanged) as the primary apparent-stress
construction. The curvature-integral form remains a diagnostic
cross-check (Milestone 15B Section 8's grid-convergence audit already
showed it converges but retains a smoothing-kernel-scale residual
relative to the endpoint form) — **not averaged with it**, per Section 4's
explicit instruction. `scripts/m15_gb_surface_rate_competition.py`'s
`sample_state` was extended to also record the raw
`top_particle_tangent`/`bot_particle_tangent` vectors directly (previously
only derived quantities were stored).

## 5. Absolute grid convergence of endpoint stress

GB-FAST (`M_GB=10*M_GB_ref`, `surface_mobility_scale=0.3`), `dx=5/2.5/
1.25nm`, at three physical states (defined from Milestone 15B's own
sigma(t) trajectory): **A** `t=0.03s` (near the curvature-form's
post-transient minimum), **B** `t=0.125s` (near the rebound peak), **C**
`t=0.25s` (late recovery). `dx=1.25nm` to `t=0.25s` took 122880 steps,
~46 minutes wall time.

| state | quantity | dx=5nm | dx=2.5nm | dx=1.25nm | rel. change 5->2.5 | rel. change 2.5->1.25 |
|---|---|---|---|---|---|---|
| A (t=0.03s) | `F_cap_endpoint,x` | -1.5565 | -1.6261 | -1.6397 | +4.47% | +0.84% |
| A | `sigma_sint_endpoint` | 2.716e7 Pa | 2.826e7 Pa | 2.840e7 Pa | +4.04% | +0.52% |
| A | `L_contact` | 57.31nm | 57.54nm | 57.73nm | +0.42% | +0.32% |
| A | `psi_top` | 104.60deg | 106.85deg | 107.88deg | +2.25deg | +1.03deg |
| B (t=0.125s) | `F_cap_endpoint,x` | -1.6333 | -1.6095 | -1.6287 | -1.46% | +1.19% |
| B | `sigma_sint_endpoint` | 2.805e7 Pa | 2.790e7 Pa | 2.819e7 Pa | -0.56% | +1.04% |
| B | `L_contact` | 58.22nm | 57.70nm | 57.79nm | -0.90% | +0.15% |
| B | `psi_top` | 109.69deg | 108.59deg | 109.21deg | -1.11deg | +0.63deg |
| C (t=0.25s) | `F_cap_endpoint,x` | -1.65163 | -1.65157 | -1.65149 | -0.004% | -0.004% |
| C | `sigma_sint_endpoint` | 2.828e7 Pa | 2.840e7 Pa | 2.838e7 Pa | +0.43% | -0.07% |
| C | `L_contact` | 58.41nm | 58.16nm | 58.20nm | -0.43% | +0.06% |
| C | `psi_top` | 112.53deg | 111.40deg | 111.44deg | -1.12deg | +0.04deg |

`F_cap_endpoint,y` is `~1e-13` or smaller at every state/resolution (the
symmetric-geometry cancellation this milestone's construction predicts,
confirming the tangent-vector `y`-components cancel to floating-point
precision, not just approximately). The raw tangent vectors themselves
(`top_particle_tangent`, `bot_particle_tangent`) converge in lockstep with
`F_cap_endpoint` and `psi` (they are the same underlying measurement).

**Convergence is real and required no inference from `force_rel_err`**:
every quantity (`F_cap_endpoint`, `L_contact`, `sigma_sint_endpoint`,
`psi`) was compared directly across `dx`. State C shows the tightest
convergence (`<0.5%`/`<0.07deg` between `dx=2.5` and `dx=1.25nm`); states
A and B show larger but still small and shrinking discretization error
(`<1.2%` between the two finest resolutions everywhere). No monotonic
divergence, no sign changes, no order-of-magnitude jumps at any state.
**This satisfies Section 5's convergence requirement for the endpoint
quantity itself.**

## 6. Delta_sigma_load and Delta_sigma_GB (endpoint stress)

Using Milestone 15B's full FROZEN/FAST trajectories (Section 3: exact
reproductions), re-extracted with `sigma_sint_app_endpoint_form` (not
curvature-form):

`t_sigma_min` for FAST (post-transient minimum, endpoint form) = **t=0.075s**,
`sigma=2.744e7 Pa` — later than the curvature-form's `t=0.03s`
(Milestone 15B); the two stress constructions do not locate the minimum
at exactly the same time, an estimator-dependent detail noted here rather
than papered over.

    Delta_sigma_load(t) = sigma_endpoint_FAST(t) - sigma_endpoint_FAST(t_sigma_min)

reaches `+9.60e5 Pa` (~3.5% of its own minimum) by `t=0.25s` — smaller in
relative magnitude than the curvature-form's `~5%` finding (Milestone
15B), but the same sign and qualitative existence of a post-transient
rebound.

    Delta_sigma_GB(t) = sigma_endpoint_FAST(t) - sigma_endpoint_FROZEN(t)

grows essentially monotonically from 0 at `t=0` to `4.62e6 Pa` (~19% of
`sigma_FROZEN`'s own late-time value) by `t=0.25s` — closely matching the
curvature-form's independently-computed `5.24e6 Pa` (~20%) from Milestone
15B. **The matched-trajectory excess `Delta_sigma_GB` is the more robust
of the two derived quantities** (consistent in both sign and
approximate magnitude between the two stress estimators); `Delta_sigma_
load`'s exact magnitude and timing are estimator-sensitive at the ~30-40%
relative level, though its sign and existence are not. Per Section 6,
neither an Arrhenius barrier coupling nor any other hazard-rate
construction is defined here — only these two kinematic observables.

## 7. Status of local curvature

Unchanged from Milestone 15B's finding (Section 9 there): the far window
(3W-5W) reproduces consistently across independent estimators
(`trace_particle_arc`, Kasa `window_curvature`, `branch_mu_J_profile`) at
every state tested, early or late. The near-TJ window (1.5W-3W,
partially also 2W-4W) develops a genuine, time-and-window-dependent SIGN
disagreement at later/deformed states — not a fixed, always-valid
convention. Per Section 7's explicit instruction, **no attempt was made
in this milestone to tune the three methods into agreement** (Milestone
15B already established this is a real reconstruction difficulty near
the TJ, not a simple convention mismatch). Far-window curvature is
retained strictly as a morphology diagnostic; **it is not promoted to a
hazard variable**, and near-TJ curvature (any of the three estimators)
remains explicitly disqualified for that purpose.

## 8. Finite-L_GB/W_GB energy benchmark

A controlled family with two free-surface/GB junctions was built by
reusing the flat-`"substrate"`-geometry initializer (Section 2's exact,
symmetric construction) with a large-radius circular grain-2 particle
(`R2=400nm`, `aspect_ratio=1`, `dx=2.5nm`, `W_GB=20nm`,
`gamma_gb=1.0`) and varying `initial_overlap` to control `L_GB`. `E_GB_
excess` was measured with the already planar-qualified background-
subtracted `gb_excess_energy` (Milestone 15B Section 11, `<1%` error on
an isolated planar GB) directly on the freshly-constructed (unrelaxed,
`t=0`) state -- deliberately isolating the geometric/energetic question
from any dynamical relaxation.

| overlap | L_GB | L_GB/W_GB | E_GB_excess/(gamma_GB*L_GB) |
|---|---|---|---|
| 4nm | 127.3nm | 6.4 | 0.795 |
| 6nm | 149.3nm | 7.5 | 0.737 |
| 8nm | 169.4nm | 8.5 | 0.706 |
| 12nm | 204.2nm | 10.2 | 0.689 |
| 16nm | 233.1nm | 11.7 | 0.705 |
| 24nm | 279.6nm | 14.0 | 0.779 |
| 32nm | 319.6nm | 16.0 | 0.867 |
| 48nm | 384.7nm | 19.2 | 1.036 |
| 64nm | 439.6nm | 22.0 | 1.160 |
| 96nm | 524.7nm | 26.2 | 1.309 |
| 128nm | 589.8nm | 29.5 | 1.382 |

The ratio is **not monotonic and does not plateau at 1**: it dips to a
minimum (~0.69) around `L_GB/W_GB~10`, then rises through 1.0 near
`L_GB/W_GB~19`, and **continues climbing** to `1.38` by `L_GB/W_GB~30`
(the largest tested) with no sign of leveling off.

## 9. W-sensitivity test

At fixed `overlap=20nm` (`L_GB~256-258nm` roughly constant), varying
`W_GB` instead (`dx` scaled to keep `>=8` cells through `W`):

| W_GB | dx | L_GB | L_GB/W_GB | ratio |
|---|---|---|---|---|
| 20nm | 2.5nm | 258.4nm | 12.9 | 0.736 |
| 10nm | 1.25nm | 256.6nm | 25.7 | 0.944 |
| 5nm | 0.625nm | 255.8nm | 51.2 | 1.237 |

**The same pattern recurs**: the ratio rises through 1.0 and keeps
climbing (to `1.24` at `L_GB/W_GB~51`, the largest value tested in
either sweep) rather than saturating at 1. This is independent evidence
(a completely different control parameter) reproducing Section 8's
finding, which rules out an `L_GB`-construction-specific artifact as the
sole explanation.

## 10. Interpretation of the ~0.5 (and non-plateauing) neck GB-energy ratio

Neither sweep supports a clean "`E_GB_excess/(gamma_GB*L_GB) -> 1` as
`L_GB/W_GB -> infinity`" conclusion within the tested range (up to
`L_GB/W_GB~30-51`) — the ratio does not merely recover from below 1, it
**overshoots and keeps rising**. Two candidate explanations, neither
confirmed nor ruled out within this milestone's scope:

1. **Particle/neck curvature confound**: this benchmark's single control
   parameter (`overlap`, or `W_GB` at fixed overlap) simultaneously
   changes `L_GB` (or `L_GB/W_GB`) AND the local curvature of the
   particle boundary near each TJ (a larger overlap engages a
   proportionally larger fraction of the R2=400nm particle's own
   curvature into the neck region) — a purpose-built, curvature-
   decoupled two-TJ construction (e.g. a saturating-profile GB segment
   like Milestone 14G's Young-Herring wedge, mirrored top and bottom)
   was not built in this milestone (time-limited) and would be needed to
   cleanly isolate the `L_GB/W_GB` effect from this confound.
2. **Unrelaxed-state limitation**: measurements were taken on the
   freshly-constructed (not dynamically relaxed) state. The calibrated
   obstacle profile is only guaranteed to be at local equilibrium far
   from the TJs (Milestone 14G); near a curved TJ, an unrelaxed profile
   may carry excess energy unrelated to the true equilibrium finite-GB
   energy, and this excess need not scale simply with `L_GB/W_GB`.

**Per Section 10's explicit instruction, `gamma_GB*L_GB` is documented
here as NOT an exact (or even clearly asymptotically-exact, within the
range tested) decomposition of the coupled diffuse free energy for these
constructed states** — this milestone does **not** attempt to "correct"
the free energy or force the ratio to 1, and does not classify the ~0.5
neck value as resolved-and-understood. **PASS-B (Section 12) is not
achieved**; this remains an open question requiring a cleaner benchmark
construction and/or relaxed (not freshly-initialized) reference states.

## 11. Whether endpoint sigma_sint is qualified for hazard design

**Yes, qualified for RELATIVE/ordering use**, per PASS-A (Section 12):
`sigma_sint_endpoint` is grid-converged (Section 5, every quantity,
`<1.2%` between the two finest resolutions at every state tested) and the
G2 ordering/rebound signatures survive the symmetric initializer (Section
3, bit-identical trajectories). `Delta_sigma_GB(t)` (the matched FAST-
vs-FROZEN excess) is the most robustly qualified derived observable — it
agrees in sign and approximate magnitude between BOTH stress estimators
and is grid-converged via the endpoint form's own convergence. `Delta_
sigma_load(t)` (rebound relative to a trajectory's own minimum) is
qualified for existence/sign but its precise magnitude and timing
(`t_sigma_min`) are estimator-sensitive and should be treated as
approximate, not exact, inputs to any future hazard design. Absolute Pa
values still carry whatever physical-parameter uncertainty the isotropic
model itself carries (`gamma_s=1.0` J/m^2 etc. are calibration choices,
not measured material constants) — this milestone qualifies the
NUMERICAL construction, not a physical-units claim.

## 12. Whether a future diffuse configurational-force diagnostic is needed

Not implemented here, per Section 11's explicit instruction (endpoint
stress was not found insufficient — it passed PASS-A). Given Section 7's
finding that local curvature remains unresolved near the TJ and Section
10's finding that the finite-GB energy decomposition is not yet clean,
**a future control-volume configurational-force construction (a diffuse
capillary/Korteweg/Eshelby stress integrated around a contour enclosing
the TJ, derived from the same `F[f,eta]`) remains a reasonable candidate
for a genuinely LOCAL TJ driving force**, should the global/contact-
averaged `sigma_sint_endpoint` prove insufficient for the eventual sink
hazard (e.g. if the hazard needs to distinguish which of several TJs in a
multi-particle system is most loaded). This is documented as a live
option for a future milestone, not designed or implemented now.

## Test suite

235/235 passing throughout (Section 2's change is bit-identical for the
tested geometry, so no regressions were possible; confirmed by full-suite
rerun after the initializer change).

## New/modified files

- `pf_sintering/model.py` — `initialize_fields`'s symmetric admissible-
  band eta clip (Section 2).
- `scripts/m15_gb_surface_rate_competition.py` — `sample_state` records
  raw particle-side tangent vectors (Section 4/5).
- `scripts/m15c_endpoint_stress_grid_check.py` — Section 5.

## STOP

Per Section 13, sink/hazard/RBM/anisotropy were not activated. This
report ends the milestone.
