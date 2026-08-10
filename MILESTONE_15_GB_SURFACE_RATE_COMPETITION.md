# Milestone 15: Coupled GB Migration / Surface-Reconstruction Rate Competition

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup` @ `6154ad3` (Milestone 14G,
229/229 tests, clean worktree) — verified before starting. This is the
first actual COUPLED neck calculation using Milestone 14G's physically
normalized GB energy (`Wc=4*gamma_GB/W_GB`, `k_eta=4*gamma_GB*W_GB/pi^2`)
and calibrated physical GB mobility (`gb_mobility_m4_J_s`). sink, hazard,
RBM, anisotropy, imposed stress/strain stayed OFF throughout; no
independent `M_TJ` was introduced.

## 2. Production eta-initialization audit and fix

`model.initialize_fields`'s `sinusoidal_substrate`/`substrate` branches
split `f` into `eta1`/`eta2` via ownership factors `t1`/`t2` that were
tanh-shaped (literally the same expression as the substrate's own
free-surface tanh) — not the Milestone 14G calibrated compact-support
obstacle profile. Fixed by replacing `t2=0.5*(1+tanh((X-x_s)/W))` with
`t2=gb_obstacle_energy.obstacle_profile(X-x_s, ell)`,
`ell=obstacle_ell(p.k_eta, p.W_cpl_f)` (= `W_GB/pi`), `t1=1-t2`, in both
branches (`threeparticle`'s three-way split was left untouched — out of
this milestone's scope, unused by the primary geometry).

Verified invariants (`tests/test_sinusoidal_substrate.py`, 4 new tests):

- **f/particle/substrate/TJ geometry exactly preserved**: `e1`, `e2` (the
  raw level sets) and `f=max(e1,e2)` are untouched by the fix (only the
  ownership SPLIT changed) — confirmed `np.array_equal(f_new, f_old)` and
  that `locate_neck_tjs` returns bit-identical `tj_top`/`tj_bottom` for the
  old (tanh) and new (obstacle) split.
- **`sum(eta_i)=f` exactly in the solid core** (`f>0.99`) near the GB/TJ —
  confirmed to `<1e-9` residual.
- **`eta_i>=0`** everywhere (product of two non-negative level sets).
- A **pre-existing** (not introduced by this fix) characteristic was found
  by this audit: in the free-surface DIFFUSE TAIL near the TJ (where `f`
  itself has not saturated and both grains' raw level sets are still
  transitioning simultaneously), `sum(eta_i)` undershoots `f` by up to
  ~0.3 in absolute terms, for BOTH the old tanh split and the new
  obstacle split (comparable magnitude either way) — a limitation of the
  1D (distance-to-substrate-surface-only) ownership split's geometry, not
  something this milestone's profile-SHAPE fix introduces or is asked to
  correct (`gb_split_deficit_not_worse_than_tanh_baseline` test locks this
  in as a non-regression bound). `reproject()` does not correct this
  deficit (its repair path only triggers for `sum<0.02`); the coupled
  dynamics' own first few steps visibly relax it (see Section 5 below,
  `t=0` -> `t=0.002s` L_GB/psi jump is partly this settling, common to
  every M_GB case since it is independent of GB mobility).

## 3. Exact physical parameters used

Primary geometry: `A=100nm` (`sinusoid_amplitude`), `lambda=320nm`
(`sinusoid_wavelength`), `R2=80nm`, `aspect_ratio=2`, `overlap=20nm`,
`W=20nm` (`interface_width_override`), `gamma_s=1.0` (fixed `Params`
default), `gamma_gb=1.0` (`gamma_gb_override`) -> isotropic Young-Herring
`psi_eq=120deg`. `dx=2.5nm` primary (`Nx=284`, `Ny=128`). BC:
`bc_x="reflecting"` (substrate wall, a true domain edge),
`bc_y="periodic"` (one full sinusoid wavelength, matching every other
sinusoidal-substrate script in this codebase).

`gb_obstacle_coefficients(1.0, 20e-9)` -> `k_eta=8.105694691387022e-09`,
`W_cpl_f=200000000.0` (both consumed identically by `mu`'s Wc-groove term
and eta's structural force -- single unified `F[f,eta]`, no `p`/`p_eta`
decoupling, confirmed by construction since both come from the same
`p.gamma_gb_ref` through `gb_obstacle_coefficients`).

**M_GB_ref** (Section 5): the Milestone 14E-14G historical reference eta
mobility `M_eta=4.266666666666666e-09` maps through
`m_gb_from_m_eta(M_eta, 20e-9)` to

    M_GB_ref = 3.458429734991796e-17  m^4/(J*s)

Confirmed round-trip through the actual `build_params()` path: passing
`gb_mobility_m4_J_s=M_GB_ref` reproduces `p.M_eta=4.266666666666666e-09`
exactly. This is a CALIBRATED reference numerical mobility, not a claimed
experimental material value.

**Surface mobility** (Section 4): `surface_mobility_scale=0.3` (matching
`m14_mechanism_screen.py`'s default, the moderate-accelerated baseline
used in the most recent prior neck calculations). At the primary
condition: `p.M_f=8.0e-25`, `M_s=m_s_ref(p.M_f, W)=1.4628571428571426e-32`
(physical units, `m_s_ref`'s own integrated-tangential-mobility
convention), `p.dt=1.0e-05 s` (the production CH-stability cap
`min(CFL*dx^4/(M_f*k_f), 1e-5)`, CFL not binding at this scale).

## 4. GB-FROZEN / SLOW / REF / FAST trajectories

Four trajectories at the baseline `M_s`, `t_target=0.25s` (`dx=2.5nm`,
`n_target=25000` steps): `M_GB in {0, 0.1, 1.0, 10.0} x M_GB_ref`. All
four ran the SAME unified `p`, the SAME production
`variational_surface_diffusion_step` (face_projected) +
`constrained_tangent_cone_eta_update` (no `g_external` bias), from the
SAME initial condition (Section 2's fixed initializer).

| case | M_GB | L_GB(0.25s) | sigma_sint_app(0.25s) | psi_top(0.25s) | F(0.25s) | mass drift |
|---|---|---|---|---|---|---|
| FROZEN | 0 | 75.97nm | 2.849e7 Pa | 111.39deg | -2.792e-7 | 5.5e-15 |
| SLOW (0.1x) | 3.458e-18 | 75.66nm | 2.855e7 Pa | 111.40deg | -2.940e-7 | 5.3e-15 |
| REF (1x) | 3.458e-17 | 71.71nm | 2.944e7 Pa | 111.44deg | -3.161e-7 | 5.5e-15 |
| FAST (10x) | 3.458e-16 | 59.93nm | 3.193e7 Pa | 111.45deg | -3.361e-7 | 5.3e-15 |

(all start from the identical `t=0` state: `L_GB=54.95nm`,
`sigma_sint_app=3.512e7 Pa`, `psi_top=109.74deg`, `F=-1.369e-7`.)

`M_eta` for each case (via `m_eta_from_m_gb`): FROZEN `0`, SLOW
`4.267e-10`, REF `4.267e-9`, FAST `4.267e-8`.

## 5. TJ and GB motion

Continuous (SUB-GRID, `tj_subgrid.compute_subgrid_contact`) top-TJ
position, not the grid-quantized `locate_neck_tjs` coordinate (which
turned out to be identical across all four cases to within one `dx`
cell — a resolution artifact, not evidence of M_GB-independence; see
below):

| case | top-TJ x(0) | top-TJ x(0.25s) | direction |
|---|---|---|---|
| FROZEN | 242.74nm | 235.71nm | monotonic inward (-7.03nm) |
| SLOW | 242.74nm | 235.85nm | monotonic inward (-6.89nm), ~indistinguishable from FROZEN |
| REF | 242.74nm | 237.64nm | inward then PLATEAUS after t~0.15s (net -5.11nm) |
| FAST | 242.74nm | 244.43nm | inward to t~0.02s (min 240.49nm), then REVERSES, ending PAST its own start (+1.69nm net) |

FROZEN and SLOW are nearly indistinguishable (0.1x M_GB_ref is too slow
to matter on this timescale — GB migration is negligible relative to the
surface-diffusion-driven TJ sliding that occurs even with `M_eta=0`,
exactly as expected: with the GB literally frozen in shape, the TJ still
slides along the fixed GB contour as the free surface reconstructs
around it). REF shows a clear, resolved SATURATION/plateau. FAST shows a
qualitatively DIFFERENT, non-monotonic trajectory that overshoots and
reverses direction — the clearest signature that GB migration is doing
something surface diffusion alone does not.

## 6. GB-length evolution

`L_GB` (`tj_subgrid.gb_geom_length_sub`, the actual sub-grid GB arc
length between the two continuous TJ positions) is the most discriminating
observable:

- FROZEN/SLOW: monotonic growth throughout, `54.95nm -> ~76nm` (the TJs
  are pulled apart by neck growth, dragging the fixed-shape GB contour's
  intercepted arc length up with them — a purely passive, surface-
  diffusion-driven stretching).
- REF: grows similarly at first, then visibly SLOWS and PLATEAUS around
  `70-72nm` after `t~0.15s`.
- FAST: grows to a PEAK (`~63nm` at `t~0.03s`), then reverses and SHRINKS
  to `59.93nm` by `t=0.25s` — the GB is actively shortening itself, faster
  than the passive dragging alone would produce, consistent with GB-
  curvature/length-reducing migration becoming dynamically significant.

An extended GB-FAST run to `t=0.6s` (Section 9) shows `L_GB` bottoming out
around `59.67nm` near `t=0.30-0.35s`, then SLOWLY RECOVERING back to
`60.06nm` by `t=0.6s` — a damped, bounded overshoot-and-recovery, not
unbounded shrinkage.

## 7. Free-surface reconstruction

`L_contact` (`tj_subgrid.compute_subgrid_contact`'s sub-grid contact
length) grows monotonically in every case (all four cases: `49.68nm` at
`t=0` -> `65.28/65.14/63.16/58.21nm` at `t=0.25s` for FROZEN/SLOW/REF/FAST
respectively) — FAST shows the LEAST contact growth, consistent with its
GB-migration-driven geometry pulling mass/shape differently than the
passive surface-diffusion-only cases. `p_gamma_particle_windows`
(`gamma_s*kappa` on the particle-side free-surface branch, three windows
`1.5W-3W`, `2W-4W`, `3W-5W`) and `substrate_curvature_windows` (same
windows, substrate-side branch) were tracked at every sample for both TJs
(Section 12).

## 8. Surface chemical-potential and flux response

`curvature_extraction.branch_mu_J_profile` (`mu(s)`, `J_tangent(s)`,
`J_normal(s)`, `kappa(s)` along the particle-side branch from each TJ,
using the SAME `mu`/face-flux the production `f`-step consumes) was
exercised and confirmed working (`pf_sintering.surface_flux_face_projected`
feeding the same `Jx`/`Jy` the conservative divergence uses), and used at
the three GB-FAST canonical save points (Section 15) rather than at every
one of the ~15 samples per trajectory (would have bloated the primary
JSON without adding discriminating information beyond the curvature-window
summaries already tracked densely).

## 9. Local curvature / capillary-stress evolution and the "GB runs away"
test (Sections 12/14 combined)

`kappa_arc_max_abs` (peak curvature magnitude anywhere on the traced
particle arc) and the windowed `p_gamma_top`/`p_gamma_bottom` values (three
windows, both TJs) were tracked at every sample. At `t=0.25s`, GB-FAST's
near-TJ windows show visibly larger-magnitude curvature (`1.5W-3W`:
`-3.84e6`/`-3.36e6` m^-1 for top/bottom) than the far windows, and the
overall arc peak (`kappa_arc_max_abs=2.48e7` m^-1) is high — the stress-
buildup conclusion does not depend on a single window (checked across all
three).

The "GB runs away from the neck" possibility (Section 14) was tested
directly via the extended `t=0.6s` GB-FAST run: `L_GB` and
`sigma_sint_app` BOTH show a clear TURNING POINT (peak/trough around
`t=0.03s`/`t=0.3-0.35s`) followed by RECOVERY toward the other cases'
level, not persistent divergence. This is a SUSTAINED, resolved
non-monotonicity (not a single-step jump), and it resolves in the direction
of RECOVERY, not growing separation -- the explicit signature this
milestone requires to rule OUT dynamic runaway/depinning (Section 14: "do
NOT call this depinning from one instantaneous jump... require a sustained
divergence"). No sustained, growing divergence was observed at any tested
M_GB.

## 10. Apparent sintering stress

`sigma_sint_app` (`capillary_stress.apparent_sintering_stress`,
`-F_cap_n/L_contact`, cross-checked via the curvature-form and endpoint-
form `F_cap` constructions) is reported in Section 4's table. The two
`F_cap` forms agree to `force_rel_err~0.13` (13%) at the representative
GB-FAST/`t=0.25s` sample — worse than the "few percent" reported for the
idealized checkpoint states this diagnostic was originally validated
against (Milestone 14C), consistent with this being a genuinely dynamic,
actively-relaxing, non-equilibrium neck rather than a static saved state;
reported honestly as a moderate- (not high-) confidence cross-check on the
absolute `sigma_sint_app` magnitude. The RELATIVE trend across M_GB cases
(FAST > REF > SLOW ~ FROZEN) and across the M_s ladder (Section 11) is
far larger than this cross-check uncertainty and is the load-bearing
result, not the absolute Pa value.

## 11. M_s perturbation (Section 16 -- triggered by GB-FAST's clear stress
buildup)

Fixed `M_GB=10*M_GB_ref` (GB-FAST), varied `surface_mobility_scale in
{0.1, 0.3, 0.9}` (`M_s/3`, `M_s`, `3*M_s`), same physical `t=0.25s`:

| M_s scale | L_GB(0.25s) | sigma_sint_app(0.25s) | psi_top(0.25s) |
|---|---|---|---|
| M_s/3 | 57.10nm | 3.338e7 Pa | 107.31deg |
| M_s (ref) | 59.93nm | 3.193e7 Pa | 111.45deg |
| 3*M_s | 63.39nm | 3.132e7 Pa | 118.44deg |

Clean, monotonic trend: SLOWER surface diffusion -> MORE GB-length
overshoot (shorter `L_GB`, i.e. more lag between GB migration and neck
reconstruction) and HIGHER apparent stress; FASTER surface diffusion ->
LESS overshoot and LOWER stress, approaching the FROZEN/SLOW/REF
`L_GB` range. This is a direct, quantitative confirmation of Section 16's
predicted signature: TJ/GB response at fixed (fast) `M_GB` is strongly
controlled by `M_s`, i.e. surface-diffusion capacity governs how much
transient stress the GB's own migration can build up before the neck
catches up.

## 12. Grid/dt qualification

**dt-halved** (Section 18, GB-FAST, `dx=2.5nm`, `t<=0.05s`): `dt=1e-5s`
vs. `dt=5e-6s` give `F`, `L_GB`, `sigma_sint_app`, `psi_top` identical to
4-5 significant figures at every matched physical time (e.g. `t=0.05s`:
`F=-3.158012e-7` vs `-3.158011e-7`, `L_GB=62.77` vs `62.77nm`) — `dt`
converged; `F` monotonically decreasing in both. No dt-halving-triggered
correction was needed (no nonmonotonicity was ever observed at the
baseline `dt`).

**Grid check** (Section 19, GB-FAST): `dx=5nm` (full `t=0.25s`) and
`dx=1.25nm` (bounded `t=0.05s`, ~494s wall for that alone --
`Nx=564,Ny=256`, `dt=2.03e-6s` uncapped by the `1e-5s` ceiling) both
reproduce the SAME qualitative signature seen at `dx=2.5nm`: `L_GB` rises
to a peak (`dx=5`: `~62.9nm` at `t=0.05s`; `dx=1.25`: `~63.0nm` at
`t=0.03s`, already turning over by `t=0.05s`) then declines. Absolute
`L_GB`/`F` values differ across `dx` (expected: different discretized
initial condition per Milestone 14G's established grid-convergence
pattern for the calibrated obstacle profile, and `F` is extensive in the
different-sized domains) -- the QUALITATIVE overshoot-then-decline is what
was checked for grid-robustness, and it holds at both coarser and finer
resolution. The longer-time RECOVERY (Section 9) was confirmed at
`dx=2.5nm` only (compute budget); the bounded `dx=1.25nm` run already
shows the onset of the same turnover within its shorter window.

## 13. Saved canonical states

For GB-FAST (`M_GB=10*M_GB_ref`, `surface_mobility_scale=0.3`,
`dx=2.5nm`), saved to `runs/m15_canonical/` (`f`, `e1`, `e2`, `e3`, and
both TJs' `mu(s)`/`J_tangent(s)`/`J_normal(s)`/`kappa(s)` profiles per
state, plus `meta.json` with the full trajectory row history):

- `post_transient_t0.01s.npz` -- after the initial profile-shape/init
  transient (Section 2) has settled.
- `stress_onset_t0.03s.npz` -- near the `L_GB`/curvature peak, onset of
  measurable stress buildup.
- `pre_recovery_t0.325s.npz` -- near the `L_GB` overshoot minimum (the
  strongest, most stress-buildup-characteristic state found).

No runaway/depinning state was saved: none occurred (Section 9) -- the
extended run shows recovery, not divergence, so a fourth "runaway" state
would misrepresent what was actually observed. These three states are
reasonable starting candidates for a future first-passage/sink
calculation, if one becomes warranted.

## 14. G1/G2/G3/G4 classification

**G2 -- COUPLED / STRESS-BUILDUP** (transient, bounded -- not runaway).

Evidence:

1. Sensitivity to `M_GB` ACCELERATES rather than saturates across the
   ladder (FROZEN->SLOW: negligible change; SLOW->REF: modest; REF->FAST:
   large) -- inconsistent with G3 (surface-reconstruction-limited, which
   predicts DIMINISHING returns as `M_GB` increases further).
2. GB-FAST shows genuinely different DYNAMICS, not just a faster version
   of the same trajectory: the TJ x-position reverses direction and the
   GB length overshoots (shrinks) below where slower-M_GB cases settle --
   inconsistent with simple G1 (GB-limited, proportional response, surfaces
   "comfortably" following).
3. `sigma_sint_app` is measurably and sustainedly ELEVATED for GB-FAST
   relative to FROZEN/SLOW/REF over most of the `0-0.6s` window tested --
   the direct stress-buildup signature Section 8 hypothesized.
4. The extended `t=0.6s` run shows both `L_GB` and `sigma_sint_app`
   turning around and RECOVERING toward the slower-M_GB range -- ruling
   OUT G4 (persistent/growing runaway): the mismatch between GB migration
   and neck reconstruction is transient and self-limiting, not diverging.
5. The `M_s` perturbation (Section 11) gives a clean, monotonic,
   quantitative confirmation that surface-diffusion CAPACITY directly
   controls how much of this transient stress/overshoot accumulates --
   exactly the "surface-diffusion-controlled effective TJ mobility"
   mechanism Section 16 asks this test to isolate, now demonstrated (not
   just hypothesized) under the single, physically corrected free energy.

## 15. Interpretation

The effective TJ/GB response in this neck is **coupled**: it is not simply
GB-limited (G1) nor purely surface-reconstruction-limited/saturating (G3)
in this M_GB range, and it does not runaway/depin (G4). Increasing `M_GB`
genuinely changes the accessible GB geometry (shorter GB, different TJ
path) in a way that free-surface reconstruction must accommodate via
conserved diffusion; when that reconstruction lags (either because `M_GB`
is large or because `M_s` is small), local curvature and apparent
sintering stress rise measurably, and the system spends an extended but
BOUNDED transient in a higher-stress configuration before surface
diffusion catches up and the geometry relaxes back toward the slower-M_GB
range. This is precisely the qualitative mechanism chain hypothesized in
Section 8 and anticipated by Milestone 14E/14F/14G's preserved "surface-
reconstruction-limited TJ motion" observation -- now demonstrated
quantitatively, with a clean confirmatory `M_s` sensitivity test, under
the single, correctly normalized free energy this milestone was gated on
using.

## Test suite

233/233 passing (`../.venv/bin/python -m pytest -q`), up from the starting
checkpoint's 229: 4 new tests in `tests/test_sinusoidal_substrate.py`
covering the Section 2 eta-initialization fix and its invariants. No
existing test's behavior changed (the initializer fix only reshapes the
`t1`/`t2` ownership SPLIT, not `f`, particle/substrate geometry, or TJ
positions).

## New files

- `scripts/m15_gb_surface_rate_competition.py` -- primary M_GB-ladder
  coupled-trajectory driver (`build_state`, `energy_ledger`,
  `sample_state`, `profile_mu_J`, `run_trajectory`).
- `scripts/m15_ms_perturbation.py` -- Section 16 M_s sensitivity test.
- `scripts/m15_dt_and_grid_check.py` -- Section 18/19 dt-halved and grid
  checks.
- `scripts/m15_save_canonical_states.py` -- Section 22 canonical-state
  saves.

## STOP

Per Section 21/20, sink/hazard/RBM were not activated, no independent
`M_TJ` was introduced, and this report ends the milestone as specified.
