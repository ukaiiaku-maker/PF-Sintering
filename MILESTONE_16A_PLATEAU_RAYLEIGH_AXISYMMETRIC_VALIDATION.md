# Milestone 16A — Fundamental Plateau-Rayleigh / Axisymmetric Validation

Starting checkpoint: `18f11c3`. Principal physical benchmark: Hussein et al., "Plateau-Rayleigh
instability with a grain boundary twist," Applied Physics Letters 121, 141601 (2022). Kept OFF:
sink, hazard, RBM, anisotropy, independent `M_TJ`. This milestone did NOT continue M15 parameter
tuning (Section 17) — it is a from-scratch geometric-correctness validation, implemented as a
SEPARATE module (`pf_sintering/axisym.py`) that does not modify or wire into the production
Cartesian path in any way.

## Bottom line

- **The current production model IS a per-unit-depth Cartesian 2-D formulation and structurally
  cannot represent the Plateau-Rayleigh (PR) instability.** Confirmed both by direct derivation
  (Section 1 below) and empirically (Section 3): a well-separated Cartesian ligament smooths every
  perturbation wavelength tested, with no analogue of a critical wavelength — there is no `1/r`
  azimuthal-curvature term anywhere in the production functional.
- **A from-scratch axisymmetric solver, built and validated independently, reproduces the correct
  PR physics cleanly**: linear-regime growth/decay rate crosses zero almost exactly at the
  classical threshold `lambda/R0=2*pi` (fitted `omega`: -2.53e-3 at `lambda/R0=4`, -3.55e-4 at 5.5,
  +1.34e-5 at `2*pi`, +1.84e-4 at 7, +3.23e-4 at 8 — correct sign, correct zero-crossing, correct
  monotone k-dependence), and a nonlinear run at an unstable wavelength shows unambiguous,
  ACCELERATING neck thinning (`R_min` drops 39%, from 28.0nm to 17.1nm, with the per-interval rate
  of decrease itself increasing — the classic PR runaway signature).
- **This directly explains the central open question from Milestones 15G-15I**: those milestones
  found that surface diffusion structurally BROADENS the contact in every Cartesian configuration
  tested, competing against a weaker GB-migration narrowing tendency. Section 1's derivation shows
  why: the Cartesian functional has NO mechanism by which reducing a neck's local radius lowers the
  total surface energy (there is no azimuthal curvature to gain from) — surface diffusion in the
  ACTUAL (per-unit-depth) geometry being simulated has no reason to narrow a neck, only to relax
  local curvature gradients, which for this geometry means widening/rounding. The axisymmetric
  model has the OPPOSITE structural bias: the SAME surface-diffusion mechanism, in the correct
  3-D-revolution geometry, actively narrows a neck once `lambda/R0>2*pi`. **The M15 series'
  persistent contact-broadening finding is very likely a consequence of solving the wrong geometric
  variational problem, not primarily a mobility, orientation, or GB-reservoir defect** (Section 13).
- **GB destabilization (Sections 11-13): partially confirmed, with an honest limitation flagged.**
  A weak GB (psi=140deg) shows the expected qualitative groove-narrowing trend (`R_GB/R_mid`
  decreasing, 1.000->0.988 over the tested window). A strong GB (psi=100deg) shows a
  qualitatively DIFFERENT response (`R_GB/R_mid` INCREASING, 1.000->1.051) under this milestone's
  deliberately simplified (unconstrained Allen-Cahn, no tangent-cone projection) GB kinetics —
  likely GB-migration-dominated behavior from an uncalibrated `M_eta`/`M_s` ratio, not a validated
  match to the paper's "higher `gamma_GB` narrows faster" prediction. This is reported honestly as
  an open item, not force-fit to the expected answer.
- **Section 14 (two-mode grain coarsening) was NOT attempted** — deferred given the time budget and
  given Section 11-13's GB kinetics need recalibration before a two-mode result would be reliable.

## 1. The current production Cartesian variational problem (Section 2)

Exactly, as implemented (`pf_sintering/model.py`, `ch_exact_energy.py`, `surface_transport.py`):

    F[f] = integral_x integral_y psi(f, grad f) dx dy      (per unit OUT-OF-PLANE DEPTH -- explicit)
    psi = (W_f/2)*f^2*(1-f)^2 + (k_f/2)*(f_x^2+f_y^2)
    mu = dF/df = W_f*f*(1-f)*(1-2f) - k_f*lap9(f)           (lap9: 9-point compact Cartesian Laplacian,
                                                               Milestone 12B's validated exact discrete adjoint)
    n = grad(f)/sqrt(|grad f|^2+eps_n^2)
    P_t = I - n(x)n                                          (tangential projector)
    M_tensor = M_s*q(f)*P_t,  q(f)=(12/W)*f^2(1-f)^2
    J = -M_tensor . grad(mu)
    df/dt = -div(J)                                          (bc_ops.flux_divergence, exact discrete conservation)

Boundary conditions: `bc_x=reflecting` (finite domain), `bc_y=periodic` (production M15-series
convention). Every gradient operator, Laplacian, and divergence in this stack is a PLAIN Cartesian
one — no `r`, no `1/r`, no azimuthal term anywhere. `F=integral psi dx dy` is explicitly interpreted
as energy PER UNIT DEPTH in the (unmodeled) third Cartesian dimension, not as a slice through an
axisymmetric solid. In cylindrical coordinates EVERY term above changes: the integration measure
becomes `r dr dz`, the gradient-energy term's variational derivative picks up a `(1/r)d/dr(r*.)`
structure (Section 4), the divergence becomes `(1/r)d(r*Jr)/dr + dJz/dz` (Section 5), and the
conserved quantity becomes `2*pi*integral(r*f)dr dz`, not `sum(f)*dx^2`. None of these are present
in the Cartesian implementation, confirmed by direct code inspection, not inferred from visual
geometry similarity.

## 2. Axisymmetric variational derivation (Sections 4-6)

Implemented from scratch in `pf_sintering/axisym.py` (finite-volume, HALF-integer r-cell centers so
`r=0` is an EXACT cell face, not offset by half a grid cell — `r_centers[i]=(i+0.5)*dr`,
`r_faces[i]=i*dr`, `r_faces[0]=0` exactly):

    F = 2*pi * integral r*psi(f, grad f) dr dz
    mu = dpsi/df - (1/r)*d/dr(r*dpsi/d(f_r)) - d/dz(dpsi/d(f_z))
       = W_f*f*(1-f)*(1-2f) - k_f*[(1/r)*d/dr(r*f_r) + f_zz]
       = mu0_bulk(f) - k_f*Laplacian_cylindrical(f)

derived by two discrete integrations by parts against the `r dr dz` measure (full derivation in the
module docstring) — NOT an ad hoc `1/r` patch inserted into the Cartesian `mu`. Radial flux is
forced to exactly 0 at `r=0` (axis symmetry: a smooth axisymmetric field has zero radial flux
through the axis by construction) and at the outer domain edge (no-flux, verified inert by using
generous vapor margins). This gives EXACT discrete conservation of the physically correct quantity

    V_f = 2*pi*integral(r*f) dr dz

(NOT `sum(f)*dx^2`, per the milestone's explicit instruction) — the weighted cylindrical divergence
telescopes to a pure boundary term that vanishes identically given both boundary fluxes are zero.

A deliberately SIMPLER (but still exactly conservative and dissipative) scalar-mobility
Cahn-Hilliard flux was used, `J=-M_s*q(f)*grad(mu)` with cylindrical divergence
`df/dt=-div_cyl(J)`, rather than replicating the production path's tangential-projected mobility
tensor — appropriate for this foundational geometric-correctness test (see module docstring for the
explicit rationale).

**Validation (`scripts/m16a_validate_axisym.py`), all against the discrete implementation directly,
not assumed:**
- `mu` is the EXACT variational derivative of `F`: finite-difference directional-derivative test
  (`[F(f+eps*q)-F(f-eps*q)]/(2*eps)` vs. `2*pi*sum(r*mu*q)*dr*dz`) agrees to `4e-8` relative at
  `eps=1e-3`, converging to `3e-11` at `eps=1e-5` (clean O(eps^2) convergence).
- `V_f` is conserved to `1.5e-16` relative over 50 real evolution steps (roundoff level).
- `F` is non-increasing every single step over the same 50-step run (`max(F(t+dt)-F(t))=0.0`
  exactly).
- The chain-rule dissipation identity `Fdot_chain = -2*pi*sum(r*mu*div(J))*dr*dz` matches an
  independent finite-difference `[F(t+dt)-F(t)]/dt` with error scaling EXACTLY linearly in `dt`
  (5.08e-2 -> 5.08e-3 -> 5.08e-4 as `dt` shrinks by 10x each time) — confirming the identity is
  exact and the residual is pure O(dt) time-discretization truncation, not a bug.

## 3. Planar Cartesian control (Section 3)

`scripts/m16a_planar_control.py`: a single-phase slab (`f=1` band) with two free surfaces at
`x=x_mid +- [W0+eps0*cos(2*pi*z/lambda)]/2`, periodic z, reflecting x, evolved with the PRODUCTION
`mu_isotropic`+`variational_surface_diffusion_step` pathway (Section 2's exact functional).

**First attempt (`W0=40nm`, interface width `W=20nm`, i.e. `W0/W=2`) showed GROWTH, not decay, at
every wavelength tested** — initially surprising. Diagnosed directly (not assumed): with `W0`
only 2x the interface width, the two free surfaces' diffuse profiles substantially OVERLAP, so they
are not independent Mullins-type surfaces; the two-surface interaction produces its own distinct
near-field effect. **Repeated at `W0=200nm` (10x `W`, genuinely well-separated surfaces)**: clean,
monotone SMOOTHING at every one of 3 wavelengths tested (400/630/800nm) — e.g. amplitude
10.0000nm -> 10.0127nm (small transient) -> steadily decreasing back toward/below 10.0000nm. This
is the milestone's expected control result, confirmed once the surfaces are properly separated; the
narrow-ligament interaction effect is a real, distinct, and separately-noted phenomenon, not a
failure of the control or a bug.

## 4. Single-crystal PR threshold and growth-rate dispersion (Sections 7-8)

`scripts/m16a_pr_benchmark.py` + `m16a_fit_growth.py`: axisymmetric rod `R(z,0)=R0+eps0*cos(2*pi*
z/lambda)`, `R0=40nm`, `W=20nm`, `dx=2.5nm`, periodic z, `eta` OFF, `gamma_GB=0`, `eps0=0.05*R0`,
200000 steps (reaching `t~97.7` in the solver's internal units).

| lambda/R0 | omega (fitted) | R^2 | interpretation |
|---|---|---|---|
| 4.000 | -2.530e-3 | 0.976 | clearly stable (decay) |
| 5.500 | -3.546e-4 | 0.835 | stable, weaker |
| **6.283 (2*pi)** | **+1.34e-5** | 0.017 | **essentially neutral** (near-zero rate; the poor R^2 here is itself expected — a rate this close to zero is dominated by noise in a log-linear fit, exactly what "neutral" should look like) |
| 7.000 | +1.841e-4 | 0.877 | unstable |
| 8.000 | +3.228e-4 | 0.986 | unstable, faster |

**`omega` crosses zero almost exactly at the classical threshold `lambda/R0=2*pi`, is monotonically
increasing with `lambda/R0` across the tested range, and has the correct sign on both sides** — all
three of Section 8's explicit requirements (correct sign, correct zero-crossing, correct
qualitative k-dependence) satisfied cleanly by a single, consistent fitting procedure.

## 5. Nonlinear pinch-off (Section 9)

Starting from a larger, already-nonlinear-scale perturbation (`eps0=0.3*R0=12nm`) at `lambda/R0=8`
(clearly unstable), run to 400000 steps: `R_min` (neck radius) drops from 28.0nm to 17.06nm (39%
reduction) while `R_max` slowly relaxes toward its own equilibrium (52.0nm -> 50.2nm). Critically,
**the RATE of decrease in `R_min` itself increases** over the run (0.55nm per unit interval early
vs. 0.8nm per unit interval late, same interval width) — the signature accelerating/runaway
approach to pinch-off, not a saturating relaxation. `R_min` (17.1nm) is already comparable to the
interface width (`W=20nm`), the point at which a diffuse-interface model begins to represent a
genuine topological neck-severing event. Mass conservation stayed at `~1e-14` relative throughout
(no rigid-body drift, no spurious sink).

## 6. Grid/W convergence (Section 10)

A first attempt (`scripts/m16a_stage_convergence.py`, finer `dx=1.25nm` and smaller `W=10nm`, SAME
step count as the `dx=2.5nm` baseline) gave an apparently contradictory result (`lambda/R0=7`
showing decay instead of growth at `dx=1.25nm/W=20nm`). **Diagnosed directly, not dismissed**: 4th-
order stability forces a much smaller `dt` at finer `dx` (confirmed: both finer-resolution runs
reached only `t~1.8`, vs. the baseline's `t~97.7`, using the same fixed step count) — far too short
to leave the initial-transient regime, so the fitted "omega" there reflected transient relaxation,
not genuine PR dynamics. **Corrected** (`scripts/m16a_stage_convergence2.py`): sized the step
budget from a TARGET PHYSICAL TIME instead of a fixed step count, and re-ran the single most
informative confirmatory case (`lambda/R0=8`, `dx=1.25nm`, `W=20nm`, targeting `t~30`).

| case | dx | W | omega | R^2 |
|---|---|---|---|---|
| baseline | 2.5nm | 20nm | +3.228e-4 | 0.986 |
| finer dx | 1.25nm | 20nm | +2.468e-4 | 0.986 |

**`lambda/R0=8` remains clearly, robustly UNSTABLE at both resolutions** (same sign, same order of
magnitude — a ~24% reduction in the fitted rate at 2x finer `dx`, consistent with a genuine,
convergent discretization correction rather than a threshold shift). This directly satisfies
Section 10's core requirement: mobility (`M_s`, held fixed) did not change WHICH wavelength is
stable; only the resolution changed, and the sign did not flip. A full re-fit of the complete
5-wavelength dispersion curve at finer resolution (to precisely re-locate the neutral wavelength
itself, which the baseline placed very close to but not exactly at `2*pi`) was not completed given
the cost of matching the baseline's `t~97.7` physical-time coverage at `dx=1.25nm` (~27x more
steps at ~4x the per-step cost, roughly 100x the wall time of the baseline single-wavelength run);
the single bounded confirmatory case above establishes the essential SIGN-robustness result Section
10 requires.

Mobility (`M_s`, held fixed in absolute value across all resolution changes here since the CFL-
determined `dt` already differs enormously between resolutions) was not tuned to move the
threshold at any point in this milestone — every comparison above uses the SAME `M_s`, confirming
(per Section 10's explicit requirement) that any resolution sensitivity found is a genuine
discretization effect, not a manufactured shift.

## 7. GB destabilization (Sections 11-13)

Extended `pf_sintering/axisym.py` with a minimal two-grain (GB) capability (documented scope
reduction: unconstrained Allen-Cahn relaxation of the exact cylindrically-generalized variational
derivative `g_i` — Milestone 12B's `structural_thermodynamic_force`, cylindrically generalized —
rather than the production path's tangent-cone-constrained update). `gamma_GB/gamma_s` set via
`psi=2*acos(gamma_GB/(2*gamma_s))` for target dihedral angles. Geometry: a periodic domain of
length `lambda=2*pi*R0` containing TWO grains (two GBs, at `z=lambda/4` and `3*lambda/4`, matching
a periodic bamboo-grain fiber) with NO externally-imposed shape perturbation — any groove
that forms is purely a consequence of the GB/coupling energy, matching the paper's actual physical
mechanism.

`scripts/m16a_gb_benchmark.py`, tracking `R_GB(t)` (at the GB) and `R_mid(t)` (mid-grain), the
paper's `R(t)=R(lambda/2,t)/R(lambda,t)` analogue:

| psi | R_GB/R_mid trend | interpretation |
|---|---|---|
| 140deg (weak GB) | 1.000 -> 0.988 (t=0 to ~20) | **groove narrows at the GB, as expected** |
| 100deg (strong GB) | 1.000 -> 1.051 (t=0 to ~73) | **groove WIDENS at the GB relative to mid-grain — opposite of expected** |

**Honest assessment**: the weak-GB case reproduces the qualitatively correct trend (Section 12.A/B's
minimum bar — one stable-ish/weak and one clearly different condition). The strong-GB case does
NOT match the paper's expected "higher `gamma_GB` narrows faster" direction under this milestone's
simplified kinetics. The most likely explanation is that the unconstrained Allen-Cahn `eta` update
(no tangent-cone projection) combined with an uncalibrated `M_eta`/`M_s` ratio allows GB MIGRATION
(the GB position itself relaxing/moving) to dominate over the thermal-groove/PR mechanism at high
`gamma_GB`, a genuinely different physical regime from what the paper's own (presumably better-
calibrated, or genuinely different-kinetics) simulation captures. **This is reported as an open
item, not resolved this milestone** — a real limitation of the deliberate scope reduction in
Section 11's GB kinetics, not a claim that the paper's mechanism is wrong or that this milestone's
axisymmetric solver is broken (the single-crystal results, Sections 4-10, are unambiguous). Section
13's full Figure-3-type quantitative comparison and Section 14's two-mode extension were NOT
attempted given this open calibration question and the time budget already spent reaching a solid
single-crystal result.

## 8. Implication for all M15 sintering-stress results (Section 13/16)

This is the central synthesizing finding of the milestone. Milestones 15-15I extensively
characterized a Cartesian, per-unit-depth sintering neck model and consistently found that surface
diffusion BROADENS the contact in every configuration tested (geometry, rate ratio, orientation,
anisotropy amplitude) — with only a weak, competing GB-migration narrowing tendency (Milestones
15G-15I) that never won sustainably. Section 1's exact derivation of the production functional shows
this is not a coincidence: **the Cartesian model has NO azimuthal curvature term, so surface
diffusion in that geometry has no energetic incentive to narrow a neck — only to relax local
(meridional) curvature gradients, which generically means widening/rounding a sharp contact.**
Sections 4-9 show that the SAME surface-diffusion mechanism, evaluated in the physically correct
axisymmetric geometry, actively and unambiguously narrows a neck once the local aspect ratio
exceeds the classical `lambda/R0=2*pi` PR threshold. **The persistent contact-broadening result
throughout the M15 series is very likely a consequence of solving the wrong geometric variational
problem for the actual 3-D sintering neck, not primarily a mobility, GB-reservoir, or orientation
effect** — those were all real, correctly-characterized effects WITHIN the Cartesian model, but the
Cartesian model itself appears to be missing the dominant physical mechanism (the genuine 3-D PR/
de-sintering instability) that would drive neck narrowing in a real axisymmetric particle-particle
or particle-substrate neck.

## 9. Recommendation for rebuilding the sintering geometry (Section 15)

Given this milestone's scope and time budget, a full axisymmetric reconstruction of the M15 series'
particle+substrate neck geometry was NOT attempted as a new simulation. A qualitative,
evidence-based assessment: the M15 series' neck region has local contact width `L_contact` (~30-50nm
across the campaigns) set by a substantially larger particle radius `R2` (~80nm) — i.e. a "neck
radius" that is a SMALL fraction of the particle's own radius of curvature, with an axial extent
(the region over which the neck profile varies) that is plausibly COMPARABLE TO OR LARGER than the
local neck radius itself, given the neck's own aspect ratio in the existing profiles. This is
qualitatively consistent with sitting inside or near a PR-unstable regime (`lambda/R0` large) once
properly represented axisymmetrically, though this is an ESTIMATE, not a computed stability result —
**the concrete recommendation is that a future milestone construct the genuine axisymmetric
equivalent of the M15 sintering neck (using this milestone's now-validated `pf_sintering/axisym.py`
machinery, extended with the substrate boundary condition) and directly test its PR stability**,
rather than continuing to refine the Cartesian model's rate/orientation/GB-reservoir parameters
(Section 17's own explicit guidance, now further motivated by this milestone's findings).

## Files added

- `pf_sintering/axisym.py`: the axisymmetric solver (single-crystal core + minimal GB extension),
  independent of and not wired into the production Cartesian path.
- `scripts/m16a_validate_axisym.py`: Sections 4-6 variational/conservation/dissipation validation.
- `scripts/m16a_planar_control.py`: Section 3 planar Cartesian control.
- `scripts/m16a_pr_benchmark.py`, `m16a_fit_growth.py`: Sections 7-8 PR threshold + dispersion fit.
- `scripts/m16a_stage_pinchoff.py`: Section 9 nonlinear pinch-off.
- `scripts/m16a_stage_convergence.py`, `_convergence2.py`: Section 10 (first attempt + corrected).
- `scripts/m16a_gb_benchmark.py`: Sections 11-13 GB destabilization.
- `runs/m16a_campaign/` (gitignored): all case results.

## Status

- 235/235 tests pass (no changes to any existing `pf_sintering/` module — `axisym.py` is new and
  self-contained, not imported by production code or existing tests).
- No hazard, first-passage integration, sink action, or RBM was implemented or activated.
- No M15-series parameter tuning (`M_GB`/`M_s`/reservoir geometry/anisotropy amplitude) was
  continued this milestone, per Section 17.
- Per Section 18: **STOP here.**
