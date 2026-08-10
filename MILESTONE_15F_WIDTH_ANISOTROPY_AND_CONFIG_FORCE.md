# Milestone 15F — Width Convergence + Interfacial Anisotropy

Starting checkpoint: `1ef8a9b` (235/235 tests, clean worktree). Preserved: unified free energy,
physical `gamma_s`/`gamma_GB`/`M_GB`, `face_projected` conserved surface transport, constrained
tangent-cone GB migration. Kept OFF: sink, hazard, RBM, independent `M_TJ`. Anisotropy was
reintroduced only through the qualified free-energy path described in Sections 4-9 below — no
empirical multiplier was used anywhere.

## Bottom line

- **Q1 (width): NO.** Reducing the diffuse-interface width `W` from 20nm to 10nm to 7.5nm, while
  holding physical `gamma_s`, `gamma_GB`, integrated surface mobility, and physical `M_GB`
  invariant (confirmed exactly, not approximately), leaves `L_contact_reset` and `A_sigma`
  essentially unchanged (41.6-41.8nm and 1.01-1.02 respectively across all three widths at
  matched physical time). Width refinement alone does not materially change neck evolution.
- **Q2 (anisotropy): NO, at the orientation tested.** A physically-derived anisotropic surface
  energy (Cahn-Hoffman `xi(theta)`, correct dynamics, correct endpoint force — see Sections 4-9)
  produces only a small (a few percent), non-systematic change in `A_sigma`, and the free surface
  near both TJs sits near the anisotropy's **hardest** orientation (~40-45deg offset from the easy
  axis) for the entire 500ms trajectory tested, never drifting toward a facet. No contact
  narrowing, no facet-controlled TJ motion, no larger capillary resultant, no qualitative change
  in evolution was observed.
- **S75/S100: FAIL.** Best trajectory across the whole width x anisotropy campaign (25 screening
  cases + 4 long trajectories) reaches `A_sigma=1.037`, `sigma_peak=34.75MPa` — solidly **H1**
  (weak amplification), the same classification Milestones 15D/15E already established. `L_contact`
  is monotonically **broadening** (not narrowing) throughout every long trajectory run, at every
  width and anisotropy level tested.
- **Width x anisotropy interaction: WA1.** Anisotropy changes little, and the contact broadens at
  both W=20nm and W=10nm; if anything the W=10nm case showed *slightly less* stress variability
  than its W=20nm counterpart (`A_sigma` 1.010 vs 1.037 for the same overlap/orientation), not
  more — the opposite of WA3/WA4.
- **Grand-potential configurational force: LOCAL-HOLD persists.** A genuine, real bug was found
  and fixed along the way (Milestone 15E's benchmark relaxation used a BC convention inconsistent
  with the Eshelby-tensor evaluation), but re-running the same benchmarks with the fix still does
  not reproduce the correctly-signed Young-Herring residual. `mu_inf` was proven analytically to be
  exactly 0 for this free-energy functional, so the literal `omega=psi_hat-mu_inf*f` correction the
  milestone proposed is algebraically a no-op relative to Milestone 15E's already-tried
  construction. The root cause is deeper than either hypothesis tried so far and remains open.
- **Recommended next physics step:** neither width nor this anisotropy orientation moves the
  needle. Before trying further crystal orientations (Section 6's "B"/"C" cases were not reached
  this milestone — the facet diagnostic suggests a ~45deg rotation of `theta_mis_deg`, aligning the
  easy axis with the TJ-flank's actual normal direction, is the one directly-motivated untried
  lever), it would be more efficient to first resolve why `L_contact` broadens in essentially every
  configuration tried across Milestones 15-15F — that geometric trend, not the specific rate/width/
  anisotropy knobs turned so far, looks like the actual bottleneck.

## Section-by-section

### 1-2. Two central questions / width-refinement physical contract

Audited (`scripts/m15f_width_invariant_audit.py`) rather than re-derived: the existing
`build_config(interface_width_override=W_nm*1e-9, gb_mobility_m4_J_s=M_GB, surface_mobility_scale=...)`
contract (already used throughout Milestones 15-15E) was found to ALREADY implement Section 2's
exact physical-invariance requirements, with no code changes needed for the contract itself:

    k_f = 3*gamma_s*W,  W_f = 12*gamma_s/W                    (varies with W, by design)
    M_f = M_f_base*scale,  M_f_base=(20nm)^4/(tau_target*k_f)
      => M_f*k_f = (20nm)^4/tau_target * scale                (EXACTLY W-independent)
    M_s_int = surface_transport.m_s_ref(M_f, W) = M_f*W*SECH8_INTEGRAL
      = (20nm)^4*scale*SECH8_INTEGRAL/(tau_target*3*gamma_s)  (EXACTLY W-independent)
    k_eta = 4*gamma_GB*W/pi^2,  Wc = 4*gamma_GB/W              (varies with W, by design)
    M_eta = pi^2*M_GB/(4*W)                                    (so physical M_GB stays fixed)

Verified numerically across the full width ladder: `M_f*k_f`, `M_s_int`, and the physical `M_GB`
recovered via `m_gb_from_m_eta(M_eta, W)` are invariant to a relative spread of `<2e-16` — exact to
floating-point precision, not merely "close."

### 3. Width ladder

Base morphology: `A=120nm, lambda=320nm, overlap=10nm` (Milestone 15E's winning rate-competition
point, `gamma_GB/gamma_s=1.4, M_GB_scale=30, M_s_scale=1.0`; overlap dropped from 15E's 10-15nm
range to a uniform 10nm because `tj_subgrid`'s Newton-based contact locator turned out to be
sensitive to the specific `W`/`overlap` combination — confirmed a numerical-robustness issue, not a
geometric ill-posedness, by direct probe: overlap=10nm is the only value that reliably resolves at
all three width-ladder points).

| W | dx | sigma_reset | sigma_peak | A_sigma | L_contact_reset | wall |
|---|---|---|---|---|---|---|
| 20nm | 2.5nm | 33.53 MPa | 34.14 MPa | 1.018 | 41.84 nm | 76s |
| 10nm | 1.25nm | 33.63 MPa | 33.92 MPa | 1.009 | 41.63 nm | 1807s |
| 7.5nm | 1.25nm | 33.51 MPa | 33.86 MPa | 1.010 | 41.65 nm | 1956s |

All three converge to the same `sigma`/`L_contact` behavior to within ~1%, at matched t=0.15s
physical time. **Width refinement alone is not the missing ingredient.**

### 4. Anisotropic free-surface functional audit

Implemented (model.py's `use_aniso_surface`/`aniso_delta` LUT block, pre-existing): 4-fold
weakly-anisotropic law `a(psi)=1-d*cos(4*psi)`, `psi=(theta-theta0)%(pi/2)`, `gamma(theta) =
gamma_s*a(psi)`, `theta` = local interface **normal** angle (confirmed exactly equivalent to using
the tangent angle instead, since the law is 4-fold — a 90deg relabeling of `theta0` only).
Derived (not inferred from the parameter name):

    gamma'(theta)  = gamma_s*4*d*sin(4*psi)
    gamma''(theta) = gamma_s*16*d*cos(4*psi)
    gamma_tilde(theta) = gamma(theta)+gamma''(theta) = gamma_s*[1+15*d*cos(4*psi)]

**Critical positive-stiffness bound: `d < 1/15 = 0.06667`.** Confirmed both analytically and
numerically against the actual runtime LUT (`scripts/m15f_aniso_audit.py`, exact match to
roundoff). **The codebase's own `Params.aniso_delta` default (0.15) is already past this bound**
(min stiffness -1.25 there) — using it as-is would have silently entered the regularized/faceted
regime `model.py`'s own build_params already anticipates and handles via a Wulff-envelope
construction, but Section 16 explicitly keeps this milestone on the positive-stiffness side. Chosen
ladder (all positive-stiffness, confirmed): **AN0**=0, **AN1**=0.02 (ratio 1.04), **AN2**=0.045
(ratio 1.09), **AN3**=0.0647 (ratio 1.14, min stiffness 0.03 — right at the edge). AN3 was audited
but not run in the neck campaign (AN0/AN1/AN2 already showed a monotone-in-`d` trend with no hint
of a qualitative change, so the extra AN3 runs were not judged worth their cost — see Section 11).

### 4b. A silent-no-op finding, fixed before any production use

`p.use_aniso_surface`/the LUT are consumed ONLY by the older `model.evolve_f` Euler-step function
and diagnostics (`model.compute_stress`, `tj_force._aniso_gamma` usage) — **not** by the
M15-series production pathway (`ch_exact_energy.mu_isotropic` +
`surface_transport.variational_surface_diffusion_step`), which computes `mu` via `mu_isotropic`
UNCONDITIONALLY. Simply setting `use_aniso_surface=True` in an M15-series config would have been a
silent no-op on the actual dynamics. Fixed by adding `pf_sintering/aniso_flux.py::mu_anisotropic`
(Section 9), the anisotropy-aware equivalent, wired into a new `run_trajectory_f`
(`scripts/m15f_campaign_lib.py`) that dispatches to it exactly when `p.use_aniso_surface`.

A first version of `mu_anisotropic` computed the WHOLE gradient term via
`div_bc(k_f*(a^2*fx-a*a'*fy), ...)` (composed `grad_bc`+`div_bc`) and, compared against
`mu_isotropic` at `delta=0` (where they should coincide), showed up to **~50% local disagreement**
deep inside the diffuse-interface core — not roundoff: `div_bc(grad_bc(...))` is a naive 5-point
Laplacian composition, while `mu_isotropic` uses `model.lap9`'s higher-order 9-point compact
stencil (Milestone 12B's validated exact discrete adjoint), and at `W/dx~8` grid cells across the
interface the two discretizations are not close. Fixed by splitting `a^2=1+(a^2-1)` and reusing
`lap9_bc` for the dominant isotropic part, adding only the genuinely anisotropic correction via
`div_bc(grad_bc(...))` (exactly zero at delta=0) — see `pf_sintering/aniso_flux.py`'s docstring for
the full derivation. Verified: at delta=0, `mu_anisotropic` now reduces EXACTLY (0.0 difference) to
`mu0_bulk - k_f*lap9_bc(f,bc_x=reflecting,bc_y=periodic)`; the only remaining difference from
`mu_isotropic` is the BC convention itself (`bc_x=reflecting` — the M15 series' own physically
correct convention for this finite-width geometry — vs `mu_isotropic`'s hardcoded legacy
`bc_x=periodic`), confirmed a small (~0.3% mean, ~3% max) and physically-motivated residual, not a
discretization bug, and confirmed inert deep in bulk (the CH mobility `q(f)=(12/W)f^2(1-f)^2`
vanishes there regardless of `mu`'s value).

### 7-9. Anisotropic endpoint capillary force

Derived `pf_sintering/aniso_capillary.py::capillary_force_endpoint_form_aniso`, generalizing the
existing isotropic `capillary_force_endpoint_form` exactly along the path its own docstring
anticipated: `F_cap_aniso = xi_end - xi_start`, `xi(theta) = gamma(theta)*t + gamma'(theta)*n`,
reusing `model._aniso_gamma`/`model._vapor_normal` (not re-derived — these ALREADY implement the
standard Cahn-Hoffman vector and are already used, and already anisotropy-aware, in the validated
`tj_force.compute_tj_force`/`cahn_hoffman_vector`). Kinetic anisotropy was deliberately kept OFF
(Section 9): only `gamma(theta)` is anisotropic; surface mobility stays isotropic.

**Benchmarks (`scripts/m15f_aniso_endpoint_benchmarks.py`):**
- **A (isotropic limit): PASS**, exactly (0.0 difference, not just small) — at delta=0,
  `F_cap_aniso` reduces identically to the isotropic formula.
- **B (straight anisotropic surface): PASS** — constant `xi` along a translation-invariant flat
  interface, exactly (0.0 spread) at 5 sample points.
- **C (antipodal-symmetry force balance): PASS** — `xi(theta+pi,-t) = -xi(theta,t)` to `1e-16`
  relative, confirming the correct even/odd parity of `gamma`/`gamma'` under the formula.
- **D (anisotropic Young-Herring TJ): CHARACTERIZED, not binary pass/fail.** Reuses
  `tj_force.compute_tj_force` (already anisotropy-aware). Finding, verified analytically to 8
  significant figures: for a mirror-symmetric synthetic wedge, `gamma(theta_L)=gamma(theta_R)`
  (even) but `gamma'(theta_L)=-gamma'(theta_R)` (odd); combined with the mirrored `v`/`n`, the
  `gamma*v` terms cancel exactly but the `gamma'*n` "torque" terms REINFORCE:
  `F_TJ_x = 2*gamma'(theta_L)*n_Lx`, generically nonzero for any symmetric-psi wedge. A genuine
  anisotropic Young-Herring equilibrium generally requires an independent 2-DOF (asymmetric flank
  angle) search — out of scope for this bounded benchmark. The isotropic-limit reduction is exact,
  and A-C directly validate the underlying `xi(theta)` formula, so it was used in production with
  this caveat documented rather than blocked on D.

### 10. Isolated width x anisotropy qualification

W in {20,10} x AN in {AN0,AN1,AN2}, single short (t=0.02s) point each, orientation A
(`theta_mis_deg=0`). All 6 cases: `F_monotonic=True`, mass drift `<2e-15`, no construction/validation
failures. `sigma` varies mildly (34.2-36.3MPa) across the 6 combinations with no dramatic outlier —
the anisotropic response does not disappear or change qualitatively between W=20 and W=10, it stays
uniformly small.

### 11-13. Primary neck screen

Geometries G1 (A=100,lambda=320) / G2 (A=120,lambda=320), overlaps restricted to what actually
resolves per width (W=20nm: {5,10,15}nm both geometries; W=10nm: only overlap=10nm reliably
resolves — the expensive W=10nm leg was restricted to G2 only per Section 11's "do not run the full
Cartesian matrix" instruction), AN0/AN1/AN2, orientation A only (t=0.06s screen). **21/21 cases
resolved and ran cleanly** (0 rejections). All in H0/H1 (`A_sigma` 1.000-1.028). Best: `overlap=5nm,
W=20nm, AN0` (`A_sigma=1.028`, sharpest resolvable contact, `L_contact=35.2nm`) — narrower overlap
correlates with somewhat more amplification, consistent with Milestones 15D/15E; anisotropy's
effect within {AN0,AN1,AN2} is small and NOT monotonic in one direction across morphologies (weakens
`A_sigma` for the ov=5/ov=15 families, mixed for ov=10). No case showed the sustained
`d(sigma)/dt>0` + `d(L_contact)/dt<0` promotion signature from Section 12 — every case's `L_contact`
was still increasing at the end of its screening window.

### 14-15. Long trajectories

Promoted 4 candidates to t=0.5s (t=0.25s for the W=10nm case — a full 0.5s at W=10 would cost
~1.7 hours per the Section 3 ladder's timing, and the width ladder already showed W changes little
at matched time, so the marginal information did not justify the cost):

| case | W | AN | A_sigma | sigma_reset | sigma_peak | t_reset | t_peak |
|---|---|---|---|---|---|---|---|
| ov5, W20, AN0 | 20nm | AN0 | 1.023 | 37.27 MPa | 38.12 MPa | 0.40 | 0.50 (still rising) |
| ov5, W20, AN1 | 20nm | AN1 | 1.017 | 36.72 MPa | 37.35 MPa | 0.35 | 0.50 (still rising) |
| ov10, W20, AN0 | 20nm | AN0 | **1.037** | 33.53 MPa | 34.75 MPa | 0.05 | 0.50 (still rising) |
| ov10, W10, AN0 | 10nm | AN0 | 1.010 | 33.63 MPa | 33.98 MPa | 0.075 | 0.175 (turned over) |

All 4: `F_monotonic=True`, mass drift `~1e-14`. **`L_contact` is monotonically BROADENING
throughout every trajectory, at every width and every anisotropy level** (e.g. ov10/W20:
35.2nm->46.5nm over 0.5s; ov10/W10: 35.0nm->44.5nm over 0.25s) — the neck fillet is rounding, the
opposite of the contact-narrowing mechanism this milestone was looking for. Classification: **all
four are H1** (`1<A_sigma<1.5`), well short of H2 (`>=1.5`). Three of the four had not yet reached
their true peak by t=0.5s (still slowly rising) — the loading pattern remains the same very slow,
small-amplitude process established in Milestones 15D-15E, not a distinct anisotropy-driven
mechanism.

Facet diagnostic (Section 13): for the AN1 long trajectory, `facet_offset_top/bottom_deg` (angular
distance from the anisotropy's easy axis, 0deg=facet, 45deg=hardest direction) stayed in the
40-45deg band for the ENTIRE 500ms run, with only a small (+-2-3deg) wobble and no systematic drift
toward 0. **The free surface near the TJs sits near the anisotropy's hardest orientation
throughout, and never approaches a facet** at `theta_mis_deg=0` (orientation "A": easy axis aligned
with the substrate's own flat-wall normal). This is a concrete, actionable finding: a ~45deg
rotation of `theta_mis_deg` (aligning the easy axis with the TJ-flank's actual ~40-45deg normal
direction instead) is the directly-motivated untried orientation for a future milestone —
orientations "B"/"C" from Section 6 were not reached this campaign given the cost of the W=10nm leg
and the uniformly negative signal from orientation "A" alone.

### 16. Strong-anisotropy / faceting gate

Not triggered: AN1/AN2 showed no stronger amplification than AN0 (if anything weaker in most
morphologies), so AN3 was not run in the neck campaign and the negative-stiffness boundary was
never approached in production.

### 17. Grid check

**Skipped, with documented rationale.** Section 17 is a grid-robustness check for "the best
anisotropic candidate" — but no candidate from this campaign exceeded H1 (`A_sigma<=1.037`), so
there is no promising result to verify. Following the precedent Milestone 15E's Section 9 already
established ("do not spend the night on an ultra-fine run if the result is already clearly
negative"), and given a `dx=1.25nm->0.625nm` refinement would cost on the order of 16-25x the
already-substantial `dx=1.25nm` wall time (~30-60 min) observed in this campaign, this check was
not run.

### 18. Width x anisotropy interaction

**WA1**: anisotropy changes little (a few percent at most, non-systematic in direction), and the
contact broadens at both W=20nm and W=10nm. The matched-overlap comparison (ov=10nm, AN0, the only
pair run at both widths) shows W=10nm producing **slightly less** `A_sigma` variability (1.010) than
W=20nm (1.037), not more — the opposite of WA3/WA4 ("anisotropy becomes substantially stronger at
smaller W"). No qualitatively new contact-narrowing/high-stress trajectory emerged from the
width-anisotropy combination.

### 19. Global high-stress gate

**S75: FAIL. S100: FAIL.** Best trajectory across the entire campaign: `A_sigma=1.037`,
`sigma_peak=34.75MPa`, `sigma_reset=33.53MPa` — nowhere near the required `A_sigma>=2.5`/
`sigma>=75MPa` (S75) or `A_sigma>=3.33`/`sigma>=100MPa` (S100). `L_contact` broadens, never narrows,
in every long trajectory run.

### 20-21. Configurational-force track (parallel)

Before changing any code, did the required analytic step: for this project's free-energy
functional, the bulk chemical potential `mu0_bulk` (the actual `dF/df` production dynamics use)
was shown to vanish EXACTLY at both `f=0` and `f=1` for any `eta` configuration (the coupling term
`-Wc*eta2*(1-fb)` vanishes identically at `fb=1`, and the whole expression vanishes at `f=0`
trivially) — i.e. **`mu_inf=0` exactly** for this system's two bulk phases. This means the proposed
`omega=psi_hat-mu_inf*f` grand-potential correction is algebraically IDENTICAL to Milestone 15E's
already-tried, already-failed `psi_hat` (background-subtracted excess energy) construction when
`mu_inf=0` — so applying it literally would not have changed anything.

Rather than stop there, checked the REQUIRED analytic identity directly: computed `div(C)` (the
Milestone 15E Eshelby tensor) numerically on a genuinely relaxed equilibrium wedge, away from the
TJ. Found `div(C)` was NOT negligible at many points even in nominal "bulk" regions — traced this to
a real, separate bug: Milestone 15E's benchmark `relax()` used the OLDER `model.evolve_f`/
`model.evolve_eta` (hardcoded periodic-X/one-sided-Y BC) to pre-relax test profiles, but then
evaluated the Eshelby tensor with `bc_x=reflecting` (the M15 series' own correct convention for this
geometry) — a genuine BC mismatch, confirmed by finding e1 stuck at ~0.997 (not 1.0) exactly at the
x=0 boundary after 1500 mismatched relaxation steps, exactly where the largest `div(C)` outliers
were located. **Fixed** (`scripts/m15f_conf_force_benchmarks_v2.py::relax_bc_consistent`, using the
ACTUAL M15-series production dynamics — `mu_isotropic`+`variational_surface_diffusion_step`+
`constrained_tangent_cone_eta_update`, `bc_x=reflecting`/`bc_y=periodic` throughout, matching the
Eshelby-tensor evaluation exactly) and re-ran Milestone 15E's Benchmarks B and C.

**Result: the sign-tracking failure PERSISTS after the BC fix** (Benchmark C: `cos_sim=+1.0` at
psi=100deg but `cos_sim=-1.0` at psi=140deg, same as before; Benchmark B's plateau stays at
`~0.88*gamma_s`, not near zero). This is valuable negative information: the BC-mismatch bug was
real and is now fixed (documented, available for any future local-force work), but it was NOT the
root cause of Milestone 15E's benchmark failures. The actual root cause remains unidentified and is
deeper than either hypothesis tried across 15E and 15F.

**Per Section 21: LOCAL-HOLD remains.** The local configurational-force diagnostic was NOT used in
any production trajectory in this milestone.

## Files added/changed

- `pf_sintering/model.py`: added `ModelConfig.aniso_delta` (previously `Params`-only and
  unreachable through the normal config pipeline; default `None` preserves existing behavior
  exactly).
- `pf_sintering/aniso_flux.py`: `mu_anisotropic`, the anisotropy-aware production chemical
  potential (Section 4b/9).
- `pf_sintering/aniso_capillary.py`: `capillary_force_endpoint_form_aniso`, `xi_vector` (Section 7).
- `scripts/m15_gb_surface_rate_competition.py`: `build_config`/`run_trajectory` extended with
  optional `use_aniso_surface`/`aniso_delta`/`theta_mis_deg` kwargs (defaults reproduce exact prior
  behavior for every existing call site).
- `scripts/m15f_campaign_lib.py`: restartable campaign infrastructure extending Milestone 15E's
  pattern with `W_nm`/anisotropy parameters, `run_trajectory_f`, `energy_ledger_f`, `sample_state_f`
  (anisotropic endpoint force + facet diagnostics).
- `scripts/m15f_aniso_audit.py`, `m15f_width_invariant_audit.py`: Sections 2/4 audits.
- `scripts/m15f_aniso_endpoint_benchmarks.py`: Section 8 benchmarks A-D.
- `scripts/m15f_stage_w.py`, `_stage_n.py`, `_stage_long.py`, `_stage_grid.py`: Sections 3/10, 11-13,
  14-15, 17 (unused — see above).
- `scripts/m15f_conf_force_benchmarks_v2.py`: Section 21 re-run with BC-consistent relaxation.
- `runs/m15f_campaign/` (gitignored): all case results (manifest + per-case JSON).

## Status

- 235/235 tests still pass with all new code in place.
- No hazard, first-passage integration, sink action, or RBM was implemented or activated.
- Per Section 22: **STOP here.**
