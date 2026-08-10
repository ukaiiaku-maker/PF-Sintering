# Milestone 15B: Stress and Initialization Qualification

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup` @ `eb8c538` (Milestone 15, 233/233
tests, clean worktree) — verified before starting. Preserves the
Milestone-14G/15 unified free energy and physical `M_GB` mapping
throughout; sink, hazard, RBM, anisotropy, and independent `M_TJ` stayed
OFF.

## 2. Exact signed-distance eta initializer

Milestone 15's eta ownership construction (`eta1=e1_raw*t1`,
`eta2=e2_raw*t2`) is replaced with the exact identity `eta1=f*(1-phi_GB)`,
`eta2=f*phi_GB` (`f=max(e1_raw,e2_raw)` already computed), making
`eta1+eta2=f` hold identically by construction, for **any** `phi_GB`.

`phi_GB` is built in two steps:

1. **Exact signed distance** `d_GB(x,y)` to the GB reference curve
   `x=x_s(y)` (`pf_sintering/gb_signed_distance.py`, new module): for the
   flat `"substrate"` geometry, `X-wall` already is the exact signed
   distance (no correction needed). For `"sinusoidal_substrate"`, the raw
   horizontal difference `X-x_s(Y)` is NOT the normal distance wherever
   the curve's slope is nonzero — significant at this milestone's
   A=100nm/lambda=320nm geometry, where the slope reaches ~2.0 near the
   TJs. The exact nearest-point Euclidean distance is computed via a
   coarse grid search (locating the right basin among candidate `Y'`
   offsets) followed by Newton refinement on the stationarity condition
   `(X-x_s(Y'))*x_s'(Y') + (Y-Y') = 0` — naive Newton from `Y'=Y` was
   found to diverge for this curve's curvature; the coarse-search seed
   fixes that. Verified against brute-force fine sampling of the curve to
   machine precision (~1e-15 m, `tests/test_sinusoidal_substrate.py::
   test_signed_distance_matches_brute_force_nearest_point`).
2. `phi_GB = min(obstacle_profile(d_GB, ell), e2_raw)`, `ell=obstacle_ell
   (p.k_eta, p.W_cpl_f) = W_GB/pi` (Milestone 14G's calibrated
   compact-support sine). **The cap at `e2_raw` was necessary**: an
   uncapped `phi_GB=obstacle_profile(d_GB,ell)` is a function of position
   relative to the GB reference curve alone, with no knowledge of where
   the particle actually reaches in `Y` — applied unconditionally, it
   assigns spurious grain-2 "ownership" deep in pure-substrate bulk far
   from the particle (wherever `X` merely happens to be close to the
   *infinite* reference line), where the particle's own raw level set
   `e2_raw` is essentially zero. This was caught empirically as a >5x
   inflation of `contact_geometry.contact_width`'s contact-region estimate
   during this milestone's own test-suite requalification (Section 4) —
   not a hypothetical concern. Capping at `e2_raw` restores `eta2->0`
   there while leaving the calibrated obstacle-profile shape intact near
   the true crossing (the cap does not bind there, since `e2_raw~f`).
   (The analogous `eta1` cap at `e1_raw` is not applied symmetrically — a
   possible, much narrower imperfection immediately at the TJs where
   `e2_raw` could locally exceed `e1_raw` on the nominal substrate side;
   not pursued further, out of this milestone's scope.)

## 3. Initialization invariants

Verified on the primary geometry (A=100nm, lambda=320nm, R2=80nm,
aspect=2, overlap=20nm, W=20nm, gamma_gb=1.0, dx=2.5nm) by direct
comparison against a reconstruction of Milestone 15's own (tanh-split)
initializer on the identical raw level sets:

| invariant | result |
|---|---|
| `f_new == f_old` exactly | **True** (bit-identical) |
| total f unchanged | exact (11049.6446 both) |
| `sum(eta_i)-f` max abs | **1.1e-16** (new) vs **0.322** (old, a real deficit) |
| `eta_i>=0` | holds both |
| grid-based TJ positions (`locate_neck_tjs`) | identical, `tj_top=[242.5,182.5]nm`, `tj_bottom=[242.5,137.5]nm`, both |
| sub-grid TJ positions (`compute_subgrid_contact`) | agree to ~0.23nm (~9% of one dx cell) |
| `L_contact` | 49.75nm (new) vs 49.68nm (old), 0.14% apart |
| `L_GB` | 56.24nm (new) vs 54.95nm (old), 2.3% apart (expected: the new construction correctly reflects the calibrated profile's own width/shape, not a tanh-contaminated approximation) |

Also required by Section 2: `tests/test_sinusoidal_substrate.py` gained/
updated 6 tests covering the exact split, the signed-distance function
itself, `sum(eta_i)=f` everywhere (not just in a solid core, as Milestone
15's weaker invariant required), non-negativity, and TJ-position
invariance.

**First-0.01s transient, before/after** (same coupled production dynamics,
`variational_surface_diffusion_step` + `constrained_tangent_cone_eta_update`,
`M_GB=M_GB_ref`, `surface_mobility_scale=0.3`):

| t | F (old init) | F (new init) |
|---|---|---|
| 0.000s | -1.369e-7 | -3.038e-7 |
| 0.001s | -2.645e-7 (Delta=-1.276e-7) | -3.056e-7 (Delta=-1.8e-9) |
| 0.010s | -2.747e-7 | -3.099e-7 |

The first-100-step energy drop shrinks from `-1.276e-7` to `-1.8e-9` — a
**~70x reduction** in the artificial initial ownership-relaxation
transient, exactly as Section 4 anticipated. (The absolute `F` values are
not comparable between old/new — the deficit-vs-exact constructions carry
different reference energies — only the *size of the initial jump*
relative to each trajectory's own subsequent smooth evolution is the
relevant comparison.)

## 4. Bounded FROZEN/REF/FAST requalification

Rerun at the primary geometry, `dx=2.5nm`, baseline `M_s`
(`surface_mobility_scale=0.3`), `t=0.25s`, with the corrected initializer.
All of Milestone 15's qualitative findings **survive**:

- **FAST TJ reversal**: continuous sub-grid top-TJ x-position goes
  242.5nm -> 239.5nm (minimum, t~0.01s) -> 244.5nm (t=0.25s, PAST its own
  starting point) — confirmed.
- **FAST GB-length peak/turnover**: `L_GB` rises 56.2nm -> 65.6nm (peak,
  t~0.02s) -> 60.1nm (t=0.25s) — confirmed.
- **FROZEN/REF/FAST ordering**: `L_GB(0.25s)` = 85.3 / 75.6 / 60.1nm
  (FROZEN > REF > FAST, GB migration increasingly shortens the GB);
  `sigma_sint_app(0.25s)` = 2.67e7 / 2.87e7 / 3.20e7 Pa (FAST > REF >
  FROZEN, opposite ordering — GB migration increasingly elevates apparent
  stress) — both orderings confirmed, both consistent with Milestone 15.
- Total `F` decreases monotonically at every sample in all three
  trajectories (secular descent maintained); mass drift stays `<6e-15`
  relative throughout.

## 5. Bounded M_s perturbation requalification

Fixed `M_GB=10*M_GB_ref` (FAST), `M_s/3`, `M_s`, `3*M_s`,
`t=0.25s`:

| M_s scale | L_GB(0.25s) | sigma_sint_app(0.25s) |
|---|---|---|
| M_s/3 | 57.3nm | 3.34e7 Pa |
| M_s (ref) | 60.1nm | 3.20e7 Pa |
| 3*M_s | 63.6nm | 3.13e7 Pa |

Slower `M_s` -> shorter `L_GB` (more mismatch) and higher stress; faster
`M_s` -> longer `L_GB` and lower stress, approaching the FROZEN/REF range —
the same monotonic, clean ordering Milestone 15 found. **The G2 (coupled/
bounded) classification survives the corrected initializer unchanged**;
the primary M_GB campaign was not fully re-run beyond this bounded check
per Section 5's explicit instruction (no qualitative change was found that
would require it).

## 6. Post-transient stress definition

Apparent stress must not be read as monotonically increasing from `t=0`
(that portion is initialization-transient, much reduced but not zero even
with the corrected initializer — Section 3). Defining `t_sigma_min` as the
first local minimum of `sigma_sint_app(t)` after the initial transient:

| case | t_sigma_min | sigma(t_sigma_min) | later behavior |
|---|---|---|---|
| FROZEN | not reached by t=0.25s | still slowly decreasing | no rebound — no GB migration, nothing to drive one |
| REF | ~0.05s | 2.834e7 Pa | rebounds to ~2.87-2.92e7 Pa (75-100% of the way) by t~0.1s, stays flat after |
| FAST | ~0.03s | 3.080e7 Pa | rebounds to ~3.20-3.23e7 Pa (peak ~t=0.125-0.15s), 4-5% above its own minimum |

`Delta_sigma_load(t) = sigma(t) - sigma(t_sigma_min)`: FROZEN ~0 throughout
(monotonic decline only); REF peaks at ~+0.09e7 Pa (~3% of its own
minimum); FAST peaks at ~+0.15e7 Pa (~5% of its own minimum) — a real,
GB-migration-scaled rebound, growing with `M_GB` exactly as the G2 picture
predicts.

`Delta_sigma_GB(t) = sigma_FAST(t) - sigma_FROZEN(t)` (the matched-
trajectory excess): grows essentially monotonically from 0 at t=0 to
`5.24e6 Pa` (~20% of `sigma_FROZEN`'s own late-time value) by t=0.25s, with
a decreasing growth rate (consistent with the extended t=0.6s run from
Milestone 15 showing this eventually saturates/partially recovers over
longer times — not re-run here per Section 5's bounded-campaign
instruction, since nothing in this bounded window contradicts it).

**The scientifically supported statement, per Section 6, is exactly this**:
GB/surface mismatch produces a measurable POST-TRANSIENT stress rebound
(FAST: +5% over its own minimum) and a growing matched-trajectory excess
over the frozen-GB control (+20% by t=0.25s) — not simply "stress
increases from the initial constructed geometry" (most of *that* signal,
Section 3 showed, was the initialization artifact, now much reduced but
still present in the first ~0.005s).

## 7. Both apparent-stress estimators

`sigma_sint_app` was computed both ways (`capillary_force_curvature_form`
integral vs. `capillary_force_endpoint_form` tangent-difference) at every
sample of every trajectory in Sections 4-5. Representative values at
t=0.25s:

| case | sigma_curvature-form | sigma_endpoint-form |
|---|---|---|
| FROZEN | 2.672e7 Pa | 2.377e7 Pa |
| REF | 2.873e7 Pa | 2.557e7 Pa |
| FAST | 3.196e7 Pa | 2.840e7 Pa |

**Ordering agrees between the two estimators**: FAST > REF > FROZEN, for
both, at every sampled time from t=0.005s onward. The `M_s` ordering
(Section 5) likewise agrees between both estimators at every sample (not
tabulated here for space; both show M_s/3 > M_s > 3*M_s in stress,
monotonically). The sign and approximate timing of the post-transient
rebound (Section 6) also agree between the two forms.

**Common-mode offset**: the curvature-form value is *consistently* ~10-15%
higher than the endpoint-form value across every case and every time
sampled (e.g. at t=0.25s: FROZEN +12.4%, REF +12.4%, FAST +12.5% — a
strikingly *constant* relative offset). This matches `force_rel_err`
(Section 8) and is best read as a shared methodological bias (traced in
Section 8 to the arc-smoothing kernel), not a physical disagreement about
which trajectory has more or less stress. Per Section 7's explicit
instruction: the qualitative claims here (ordering, rebound) are NOT
resting on any single 8-12%-scale difference from one estimator alone —
every comparison in this report that matters for the G2 classification is
a factor of ~1.15-1.5x or a clear sign, well outside this common-mode
band.

## 8. Capillary-force identity audit

The `~13%` `force_rel_err` flagged in Milestone 15 was diagnosed by grid
refinement at two states:

**Static (t=0, undeformed) state**, `dx=5/2.5/1.25nm`:
`force_rel_err` = 0.065 / 0.046 / 0.014 — **shrinks cleanly, ~3-4x per
halving**, consistent with `O(dx)`-`O(dx^2)`-ish discretization
convergence, not a fixed bug. `closest_approach_dist` (arc-tracing quality)
scales linearly with dx (5.6/2.4/1.1nm) as documented.

**Dynamic (t=0.02s, deformed GB-REF) state**, same three `dx`:
`force_rel_err` = 0.191 / 0.119 / 0.108 — still **decreasing** with grid
refinement (ruling out a fixed sign/branch/orientation bug, which would
not shrink with `dx`), but the improvement **slows markedly** between
`dx=2.5` and `dx=1.25nm` (0.119->0.108, far less than the 2-3x reduction
seen in the static case). This residual, only weakly `dx`-dependent
component is consistent with Milestone 14C's own documented caveat: the
particle-arc's smoothing kernel width is fixed at `1.5*interface_width`
(a *physical*, not discretization, length scale) — "both smaller... and
larger smoothing scales give worse agreement" per that module's own
docstring, i.e. a genuine, non-vanishing (as `dx->0` at fixed `W`)
methodological trade-off, not a bug.

Checked directly (not just inferred): particle-arc branch selection
(`classify_branches`, confirmed picking the bulging/particle branch, not
substrate or GB, at every state tested); endpoint tangent orientation and
arc traversal direction (both forms use the SAME `top_particle_dir`/
`bot_particle_dir` from the SAME `classify_branches` call, so branch/
orientation cannot be the source of a *relative* discrepancy between the
two forms); the TJ diffuse core is excluded consistently by both forms
(the endpoint form uses the branch tangent AT the TJ from `compute_tj_force`,
never entering the diffuse core; the curvature-form arc is truncated at
closest approach to the opposite TJ, `closest_approach_dist` reported and
small).

**Conclusion**: the discrepancy converges with `dx` (ruling out an
implementation bug) but has a genuine, bounded residual tied to the fixed
physical smoothing scale, not to discretization. **Not tuned to force
agreement** — both forms are reported as computed.

## 9. Three-way curvature reconciliation

`trace_particle_arc`'s windowed `kappa(s)`, `window_curvature`'s Kasa
`kappa_geom`, and `branch_mu_J_profile`'s `kappa(s)` were compared at the
SAME branch, orientation, windows (1.5W-3W, 2W-4W, 3W-5W), and physical
arclength coordinate, at all four canonical states (Section 12):

| state | window | arc | kasa | mu_J | agreement |
|---|---|---|---|---|---|
| A (t=0.005s) | 1.5-3W | -4.6e6 | -7.1e6 | -6.8e6 | same sign, ~35% spread |
| A | 3-5W | -5.1e6 | -4.9e6 | -4.9e6 | good (<4%) |
| B (t=0.03s) | 1.5-3W | -5.0e6 | -6.3e6 | -4.7e6 | same sign, ~25% spread |
| B | 3-5W | -5.1e6 | -4.8e6 | -4.8e6 | good |
| **C (t=0.125s)** | **1.5-3W** | **-3.4e6** | **+1.2e7** | **+3.2e6** | **SIGN DISAGREEMENT** |
| C | 3-5W | -5.0e6 | -4.9e6 | -4.8e6 | good |
| **D (t=0.25s)** | **1.5-3W** | **-3.8e6** | **+1.2e7** | **+4.3e6** | **SIGN DISAGREEMENT** |
| D | 2-4W | -4.7e6 | -6.5e6 | -1.9e6 | same sign, large spread (2.5x) |
| D | 3-5W | -5.1e6 | -4.6e6 | -4.7e6 | good |

**Reconciled**: the far window (3W-5W) agrees well (within ~5-10%) at
every state tested, early or late.

**NOT reconciled**: the near-TJ window (1.5W-3W) agrees (same sign,
20-35% magnitude spread — tolerable) at the two EARLY states (A, B) but
develops an outright **sign disagreement** between `kasa`/`mu_J` (both
flip positive) and `arc` (stays negative) at the two LATE, more-deformed
states (C, D). The 2W-4W window shows a similar, milder pattern (same
sign but a widening magnitude spread by D). This is exactly the failure
mode Section 9 says must be diagnosed as real, not papered over: the
disagreement is **not a fixed, time-independent sign convention** (which
would be an acceptable, documented flip) — it **changes with time and
with window**, meaning it reflects a genuine reconstruction difficulty
specific to the near-TJ region once the geometry has deformed
substantially (plausibly: the Kasa circle fit becoming poorly conditioned
on a short, curvature-varying arc segment very close to a TJ that has
itself moved and reshaped the local geometry; `branch_mu_J_profile`'s
`kappa(s)` uses a small local-point window per Milestone 14G's own
diagnostic and may share this sensitivity). This was not resolved within
this milestone's scope; it needs its own targeted investigation (see
Section 10).

## 10. Local curvature not used as a hazard input

Per Section 9's finding, `gamma_s*kappa_local` is **not qualified** for
use as a quantitative sink-hazard input, specifically in the near-TJ
window(s) at later/deformed states — the far-window value is fine, but a
hazard construction cannot self-consistently pick "only the far window"
without further justification this milestone doesn't establish. The
geometric `M_GB`/`M_s` coupling result (Sections 4-5, the G2
classification) uses `L_GB`, TJ position, and `sigma_sint_app` — none of
which depend on local `kappa`, so it is unaffected and remains usable.

## 11. Physical GB excess-energy ledger

`energy_ledger`'s raw `E_GB` (eta-gradient term only) and `E_coupling`
(the raw `Wc*eta2*(0.5f^2-f)` bulk term) are NOT, individually or summed,
the physical GB interfacial excess energy — both carry a nonzero
single-grain "background" (e.g. the coupling integrand at `f=1`, one
grain, is `-0.5*Wc*f^2`, not zero). A background-subtracted excess
(`gb_excess_energy`, new function) is derived: subtracting the LOCAL
single-grain-at-the-same-`f` reference from each term and using
`e1+e2=f` (now exact, Section 2) gives

    excess_coupling = Wc*e1*e2*f*(2-f)     (-> Wc*phi*(1-phi) at f=1)
    excess_grad     = -0.5*k_eta*[(e1*Lap(e1)+e2*Lap(e2)) - f*Lap(f)]

using `lap9_bc` with the SAME `bc_x`/`bc_y` the coupled dynamics actually
use (**not** the periodic-only `lap9` `energy_ledger` uses for its own,
different purpose) — using the periodic form on a non-periodic planar test
was checked BY HAND to diverge as `dx->0` (a spurious wraparound
artifact, ~4x too large at dx=2.5nm growing to ~80x at dx=0.1nm on a
one-sided profile), confirming the BC-aware form is required, not
optional, for this diagnostic.

**Planar qualification** (isolated calibrated GB, `f=1`, reflecting BC
both axes, `gamma_gb=1.0`): `E_GB_excess/L_GB` / `gamma_GB` = 0.9936 /
0.9984 / 0.9997 at `dx=2.5/1.25/0.5nm` — **clean convergence to 1**,
confirming the derivation and its discretization are correct.

**Applied to the neck geometry** (same four canonical states):

| state | E_GB_excess/L_GB | ratio to gamma_GB=1.0 |
|---|---|---|
| A (t=0.005s) | 0.613 | 0.613 |
| B (t=0.03s) | 0.550 | 0.550 |
| C (t=0.125s) | 0.513 | 0.513 |
| D (t=0.25s) | 0.509 | 0.509 |

Rather than recovering `gamma_GB` after relaxation, the measured ratio
**decreases** over the trajectory and appears to be settling near ~0.5,
not 1.0. This is a genuine, reproducible finding, not a bug in the
formula (which passed the planar benchmark cleanly) — plausible physical
readings include: the GB's local width/shape in the coupled neck genuinely
departs from the calibrated planar equilibrium profile under the
competing GB-migration/surface-reconstruction dynamics (curvature from
the two TJs, or compression/stretching as `L_GB` itself evolves,
Section 4); or a residual mismatch between the sub-grid `L_GB` metric's
own effective width convention and the energy integral's spatial extent.
**This was not resolved within this milestone's scope.** Per Section 11's
instruction, the raw coupling/gradient terms are NOT interpreted as
physical GB energy on their own; this background-subtracted ledger is the
correct construction, and it is reported honestly as unqualified for
quantitative use in the neck geometry pending further investigation.

## 12. Corrected canonical states

Saved to `runs/m15b_canonical/` for the corrected GB-FAST trajectory
(`dx=2.5nm`, `M_GB=10*M_GB_ref`, `surface_mobility_scale=0.3`), labeled by
the ACTUAL post-transient loading trajectory (Section 6), not by GB-length
peak timing (Milestone 15's approach, which mislabeled `t=0.03s` "stress
onset" when `sigma_sint_app` was, per the corrected trajectory, still
decreasing there):

- `A_post_transient_t0.005s.npz` — after the (now much smaller,
  Section 3) initialization transient has settled.
- `B_stress_minimum_t0.03s.npz` — `sigma_sint_app`'s post-transient
  minimum, the true start of the "loading" phase.
- `C_post_transient_peak_t0.125s.npz` — the stress rebound's peak.
- `D_late_recovery_t0.25s.npz` — partial relaxation from the peak.

Each `.npz` contains `f`, `e1`, `e2`, `e3`, and both TJs'
`mu(s)`/`J_tangent(s)`/`J_normal(s)`/`kappa(s)` profiles
(`branch_mu_J_profile`); `meta.json` carries the full trajectory row
history. No runaway/depinning state exists to save (Milestone 15's
extended run showed recovery, not divergence — unchanged conclusion, not
re-verified at the full 0.6s horizon in this bounded milestone).

## 13. Decision gate

| criterion | result |
|---|---|
| 1. exact eta initialization removes the artificial first-step correction | **PASS** (~70x reduction, Section 3) |
| 2. M15's G2 geometric coupling survives | **PASS** (Section 4-5, all signatures confirmed) |
| 3. both apparent-stress estimators agree qualitatively (ordering, M_GB/M_s, rebound sign/timing) | **PASS** (Section 7); common-mode offset (~12%) quantified separately, not conflated with a physical claim |
| 4. curvature methods reconciled | **FAIL** — far window reconciled at all times; near-TJ window(s) develop a genuine, time-and-window-dependent sign disagreement at later/deformed states (Section 9) |
| 5. GB excess energy passes the planar benchmark | **PASS for the planar case** (Section 11, <1% at fine dx); does **NOT** extend to the neck geometry (persistent ~50% gap, not resolved) |

**Overall: FAIL/HOLD**, per Section 13's own fallback: the G2 geometric-
coupling result (Sections 4-5) is retained as well-qualified — it does not
depend on local curvature or the neck GB-excess-energy estimate, only on
`L_GB`, TJ position, and `sigma_sint_app` (both forms, consistent
ordering). Local `kappa` and the neck-geometry GB-excess-energy value are
**not** qualified for use in a future sink-hazard construction. The
apparent-sintering-stress observable (`sigma_sint_app`, either form) is
qualified for RELATIVE/qualitative use (ordering, rebound existence and
approximate timing) but not for its absolute Pa value at the ~12%
common-mode-offset level, nor beyond the bounded `t<=0.25s` window
actually re-verified here.

## Test suite

233/233 passing at the starting checkpoint (`eb8c538`). `pf_sintering/
structural_projection.py`'s `_add_with_capacity` per-cell headroom mask
was fixed (`headroom>tol` -> `headroom>0`) after this milestone's exact
initializer removed an incidental slack `runner.py`'s SinteringModel path
had been implicitly relying on to find thinly-spread redistribution
capacity within its iteration budget — a genuine, if narrow, pre-existing
robustness gap, exposed rather than caused by the initializer fix.
`tj_subgrid.py`'s Newton `max_iter` was raised 20->100 after the same fix
shifted some transient states' `e1-e2` field enough to land in a
near-singular-Jacobian (genuinely, slowly convergent, not divergent)
regime for the legacy `operator_ledger_v3` test geometry.
`tests/test_contact_geometry.py`, `tests/test_operator_ledger_v3.py`, and
`tests/test_static_neck_geometry.py` had reference values/tolerances
recalibrated to the corrected (now bug-free, ~5x narrower) `contact_width`
estimate; `tests/test_sinusoidal_substrate.py` gained/updated 6 tests
replacing Milestone 15's weaker invariant tests with the exact ones this
milestone establishes. Full final run: **235/235 passing**.

## New/modified files

- `pf_sintering/gb_signed_distance.py` — exact nearest-point signed
  distance to the sinusoidal GB curve (Section 2).
- `pf_sintering/model.py` — `initialize_fields`'s eta construction
  (Section 2); `structural_projection.py`'s headroom mask; `tj_subgrid.py`'s
  Newton iteration budget.
- `scripts/m15_gb_surface_rate_competition.py` — `gb_excess_energy`
  (Section 11) added to the existing driver.
- `scripts/m15b_capillary_identity_grid_check.py` — Section 8.
- `scripts/m15b_save_canonical_states.py` — Section 12.
- `tests/test_sinusoidal_substrate.py`, `tests/test_contact_geometry.py`,
  `tests/test_operator_ledger_v3.py`, `tests/test_static_neck_geometry.py` —
  updated for the corrected initializer.

## STOP

Per Section 14, sink/hazard/RBM/anisotropy were not activated and no
independent `M_TJ` was added. This report ends the milestone.
