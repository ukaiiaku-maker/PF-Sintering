# Milestone 14G: Single-Free-Energy GB Normalization

Starting checkpoint: `codex/coarsening-stress-buildup` @ `93986c7` (Milestone 14F,
219/219 tests). This milestone corrects Milestone 14E/14F's `GB_ENERGY_
CALIBRATION_FACTOR=19` normalization, which was derived from the energy of an
*imposed tanh* eta profile rather than the *equilibrium* profile of the actual
constrained two-grain free energy, and removes the script-level `p`/`p_eta`
thermodynamic decoupling Milestone 14F introduced to route around that error.
Per the hard stop in Section 18 of the handoff, this report ends the milestone:
no `M_GB` vs. `M_s` competition, G1-G4 classification, full coupled sintering,
or independent `M_TJ` are run here.

## 1. The obstacle-problem reduction

At `f=1`, `eta1+eta2=1`. Writing `phi=eta2`, `eta1=1-phi`, the existing
structural free energy

```
F = integral[ Wc*(eta1^2+eta2^2)*(f^2/2-f) + (k_eta/2)*sum_i|grad(eta_i)|^2 ] dV
```

reduces exactly (constant `-0.5*Wc` term dropped) to

```
F_GB[phi] = integral[ k_eta*(phi')^2 + Wc*phi*(1-phi) ] dV,      0<=phi<=1
```

via direct algebra: `-0.5*Wc*(eta1^2+eta2^2) = -0.5*Wc + Wc*phi*(1-phi)` and
`(k_eta/2)*(|grad eta1|^2+|grad eta2|^2) = k_eta*(phi')^2` (since
`eta1'=-phi'`, `eta2'=phi'`). This is a **double-obstacle** interfacial energy
(Wc*phi*(1-phi), not the double-well WGB*phi^2*(1-phi)^2), constrained by the
box `0<=phi<=1` rather than regularized by a quartic well — its equilibrium
profile is *not* a tanh.

## 2. Equilibrium profile: a compact-support sine, not a tanh

Inside the active interval, the Euler-Lagrange equation is
`-2*k_eta*phi'' + Wc*(1-2*phi) = 0`. With `ell=sqrt(k_eta/Wc)` and
`psi=phi-0.5`, this is the SHM equation `psi''=-psi/ell^2`, giving

```
phi(x) = 0.5*(1 + sin(x/ell)),   -pi*ell/2 <= x <= +pi*ell/2
phi(x) = 0 (x < -pi*ell/2),  phi(x) = 1 (x > +pi*ell/2)
```

The free boundary at `x=+-pi*ell/2` satisfies *both* `phi=0/1` and `phi'=0`
there (smooth-pasting) — confirming this is the exact obstacle-problem
solution with **compact support**, not an asymptotic tail. Hence the full
compact-support width is `delta_GB = pi*ell = pi*sqrt(k_eta/Wc)`.

Along the active interval, `k_eta*(phi')^2 = Wc*phi*(1-phi) = 0.25*Wc*cos^2(x/
ell)` (equipartition). Integrating (`integral cos^2(theta) dtheta` over
`[-pi/2,pi/2] = pi/2`):

```
integral (phi')^2 dx = pi/(8*ell)
gamma_GB_PF = (pi/4)*sqrt(k_eta*Wc)
```

Both relations, plus the profile's PDE-satisfaction and free-boundary
behavior, are verified analytically and numerically in
`tests/test_gb_obstacle_energy.py` (8 tests, all passing) — implemented in
the new module `pf_sintering/gb_obstacle_energy.py`.

## 3. Reinterpreting the old (tanh-calibrated) coefficients

For the old `k_eta=3*gamma_declared*W`, `Wc=36*gamma_declared/W`:

```
ell = W/sqrt(12)
delta_GB = (pi/sqrt(12))*W ~= 0.9069*W
gamma_GB_PF = (3*pi*sqrt(3)/2)*gamma_declared ~= 8.1621*gamma_declared
```

— different from *both* the naive "declared=true" assumption *and*
Milestone 14E/14F's `19*gamma_declared` tanh-profile finding. A tanh
initialized at that same width relaxes, under the production tangent-cone
integrator, from the tanh excess energy toward this compact-obstacle value —
demonstrated directly by the Section 9 planar-equilibrium runs below (every
tanh-family initial condition converges to the *same* attractor as the
analytic-obstacle initial condition, at every grid resolution tested).

## 4. The 19x factor is deprecated, not a physical calibration

`GB_ENERGY_CALIBRATION_FACTOR=19` and `gamma_gb_target_to_declared` are
**removed** from `pf_sintering/constrained_eta.py` (they must not remain as
the physical conversion, per Section 4 of the handoff). The numerical fact
itself (`19.0`, a tanh-profile *non-equilibrium* excess-energy factor) is
preserved, correctly labeled, as `gb_obstacle_energy.TANH_PROFILE_EXCESS_
FACTOR`, retained *only* for regression/documentation. The two Milestone-14F
tests built on the old factor
(`test_gb_energy_calibration_factor_matches_analytic_isolated_gb`,
`test_gamma_gb_target_to_declared_round_trips`) are removed from
`tests/test_constrained_eta.py`; their replacement coverage lives in
`tests/test_gb_obstacle_energy.py`.

## 5-7. One free energy, one coefficient provider

Milestone 14F's script-level decoupling (`p` for f-transport with `gamma_gb_
ref` at the *target* value, `p_eta` for eta with a *separately corrected*
`gamma_gb_ref`) is removed. The single new coefficient provider,
`gb_obstacle_energy.gb_obstacle_coefficients(gamma_gb_physical, gb_width_
physical, width_convention="compact_support")`, solves the exact obstacle
relations for the preferred `delta_GB=W_GB` convention:

```
Wc    = 4*gamma_GB/W_GB
k_eta = 4*gamma_GB*W_GB/pi^2
```

(derived by substituting `delta_GB=W_GB`, `gamma_GB_PF=gamma_GB_physical` into
Section 2's two relations and solving for `k_eta`, `Wc`). It returns `k_eta`,
`Wc`, and `W_cpl_f` (an alias, matching `Params.W_cpl_f`'s existing field
name).

An audit (`grep -rn` across `pf_sintering/*.py`) found the *active* `Wc`
computation duplicated as a hardcoded `36.0 * gl / W` literal in **five**
separate call sites, while `Params.W_cpl_f` itself was a dead field (written,
never read) — `ch_exact_energy.py::_wc`, `constrained_eta.py::local_wc`,
`parity_kernels.py`, `model.py::evolve_f`, `ch_flux_diagnostics.py`. All five
now call the same coefficient formula (`4.0*gl/W`, i.e.
`gb_obstacle_coefficients`'s `Wc`); `model.py::build_params` and
`_eta_diffusivity_reference` now derive `p.k_eta`/`p.W_cpl_f` from
`gb_obstacle_coefficients(p.gamma_gb, W)` directly. `mu=delta F/delta f` (via
`p.gamma_gb_ref`'s `Wc` inside `ch_exact_energy.mu0_bulk`) and
`g_i=delta F/delta eta_i` (via `p.k_eta`/`p.W_cpl_f` inside `constrained_eta.
structural_thermodynamic_force`) now consume the *same* `p.k_eta`/`p.W_cpl_f`
pair for every grain and every step — there is exactly one `Wc` value in the
production dynamics, not two.

## 8. Variational derivative reverification (hard gate)

`tests/test_ch_exact_energy.py::test_exact_energy_directional_derivative_
matches_mu_isotropic` (`mu=delta F/delta f`, central-difference directional
derivative, `O(eps^2)` convergence to `<1e-6` relative error) and
`tests/test_constrained_eta.py::test_structural_force_matches_full_free_
energy_derivative` (`g_i=delta F/delta eta_i`, same methodology, `<1e-8`) are
both coefficient-value-agnostic — they check self-consistency between the
force/potential and a *matching* free-energy functional built from the same
`p.k_eta`/`p.W_cpl_f`, not any specific numeric coefficient. Both pass under
the new coefficients (confirmed via a targeted rerun:
`pytest tests/test_ch_exact_energy.py tests/test_constrained_eta.py` -> 11
passed). This satisfies the Section 8 hard gate.

## 9. Planar equilibrium profile qualification

`scripts/m14g_planar_equilibrium_profile.py`: a single flat GB
(`bc_x="reflecting"`, exactly one boundary, no periodic-image ambiguity),
five initial conditions (tanh at the interface-width scale, a 3x-broader
tanh, a 3x-narrower tanh, tanh+smooth perturbation, and the analytic
compact-support obstacle profile itself), relaxed under the unmodified
production `constrained_tangent_cone_eta_update` (no bias). `gamma_measured`
is the *actual* excess energy integrated directly from the discretized field
(`k_eta*(dphi/dx)^2 + Wc*phi*(1-phi)`, divided by the domain's y-extent) —
not the closed-form target, which is a property of the parameters alone.
`width_fit` is a nonlinear least-squares fit of the relaxed field to the
analytic sine shape (`ell`, `x0` free) — i.e. a genuine shape check, not just
an energy check.

| dx (nm) | gamma_meas/target (all 5 ICs) | width_fit/target (all 5 ICs) |
|---|---|---|
| 5.0  | 0.8886 - 0.8891 | 0.8520 |
| 2.5  | 0.9738 - 0.9741 | 0.9597 |
| 1.25 | 0.9936 - 0.9940 | 0.9899 - 0.9901 |

All five initial conditions converge to the **same** ratio at each `dx` (to
within ~0.05%), and both ratios converge monotonically to 1.0 under grid
refinement. This is strong evidence that (a) the compact-support sine is the
unique attracting equilibrium of the production dynamics, independent of
initial condition, and (b) it converges to the analytically-derived target as
`dx->0`, i.e. Section 1-2's derivation correctly describes the continuum limit
of the discretized system.

## 10. Analytic M_eta -> M_GB mapping

The existing two-grain reduction of the constrained update,
`d(eta1)/dt = -(M_eta/2)*(g1-g2)`, is exactly `phi_dot = -(M_eta/2)*delta
F_GB/delta phi` by direct substitution — no change to the integrator, only to
how `M_eta` is physically interpreted. For a traveling front under a small
bulk bias `Delta_g` (added as `Delta_g*eta_i`, i.e. `g_i -> g_i+Delta_g` for
the biased grain), the standard translational-mode solvability projection
(multiply by `phi_0'(xi)`, integrate, use `integral(phi_0')dxi=1`) gives

```
v_GB = (M_eta/(2*I))*Delta_g,   I = integral(phi')^2 dx = pi/(8*ell)
M_GB = (M_eta/2)/I = 4*M_eta*ell/pi = 4*M_eta*delta_GB/pi^2
```

For the preferred `delta_GB=W_GB` convention: `M_GB = 4*M_eta*W_GB/pi^2`,
`M_eta = pi^2*M_GB/(4*W_GB)`. Implemented as `gb_obstacle_energy.
m_gb_from_m_eta` / `m_eta_from_m_gb`.

## 11. Circular-grain calibration (obstacle-profile IC)

`scripts/m14g_circular_grain_calibration.py`: rebuilds Milestone 14E's
curvature-shrinkage calibration, initializing the radial GB with the
calibrated obstacle profile (not tanh), and comparing the fitted `M_gb_eff
=-(1/(2*gamma_gb))*d(R^2)/dt` against the analytic `m_gb_from_m_eta` mapping.

| dx (nm) | M_gb_eff / M_gb_analytic | fit R^2 | mass drift |
|---|---|---|---|
| 5.0  | 0.8920 | 1.000000 | 0 |
| 2.5  | 0.9751 | 1.000000 | 0 |
| 1.25 | 0.9938 | 1.000000 | 0 |

Clean grid convergence, essentially bit-identical to the Section 9 planar
ratios at matched `dx` (0.889/0.974/0.994) — two independent measurement
methods (static shape/energy fit vs. dynamic curvature-shrinkage rate)
agreeing to within the discretization error at every resolution tested.

## 12. Planar-bias mobility calibration (production update, `g_external`)

`constrained_tangent_cone_eta_update` gained an optional `g_external`
parameter (default `None`, preserving the exact prior production code path —
confirmed via `tests/test_tangent_cone_eta.py::test_tangent_cone_g_external_
none_is_bit_identical_to_default`, checking both the implicit-default and
explicit-`None` calls against an explicit all-`None` per-grain list). A
second new test, `test_tangent_cone_g_external_biases_velocity_as_expected`,
confirms the exact linear-algebra prediction for an interior (no
active-boundary) state: a uniform bias `c` on one grain shifts that grain's
velocity by `-0.5*M_eta*dt*c` and the other's by `+0.5*M_eta*dt*c` (the exact
sum-zero mean-subtraction projection), matched to machine precision.

`scripts/m14g_planar_bias_mobility.py` replaces Milestone 14E's ad hoc
`eta+=dt*v; clip; renormalize` loop with this hook. A single flat GB
(`bc_x="reflecting"`) is initialized at the calibrated obstacle profile and
**pre-relaxed** at `Delta_g=0` for the same physical relaxation time used in
Section 9, then a constant bias `Delta_g` is applied to grain 2 via
`g_external=[None, Delta_g, None]` through the unmodified production update,
and the front position is tracked via bisection.

| dx (nm) | Delta_g (J/m^3) | M_gb_eff / M_gb_analytic | fit R^2 |
|---|---|---|---|
| 5.0  | 1e7 | 0.8459 | 0.998959 |
| 5.0  | 2e7 | 0.8771 | 0.999914 |
| 2.5  | 1e7 | 0.9710 | 0.999931 |
| 2.5  | 2e7 | 0.9733 | 0.999987 |
| 1.25 | 1e7 | 0.9886 | 0.999922 |
| 1.25 | 2e7 | 0.9901 | 0.999939 |

`M_gb_eff` is exactly proportional to `Delta_g` at fixed `dx` (confirmed
separately at `Delta_g=1e4,2e4,5e4` giving *identical* `M_gb_eff` to 6 sig
figs) — the required linear-response behavior.

## 13. Three-way M_GB consistency

| dx (nm) | analytic-vs-circular | analytic-vs-planar-bias | circular-vs-planar-bias |
|---|---|---|---|
| 5.0  | ratio 0.892 | ratio 0.846-0.877 | ~2-5% apart |
| 2.5  | ratio 0.975 | ratio 0.971-0.973 | ~0.4% apart |
| 1.25 | ratio 0.994 | ratio 0.988-0.990 | ~0.6% apart |

At `dx=2.5nm` the three independent methods (analytic mapping, circular
curvature-shrinkage, planar driven-boundary) agree with each other to well
under 1%, comfortably inside the `<=5%` target, with agreement *improving*
under refinement (dx=5nm spread ~5%, dx=1.25nm spread ~0.6%). Per Section 12's
gate ("Target `<=5%` disagreement at dx=2.5nm... If not: STOP"), this passes.

**`gb_mobility_m4_J_s`** (Section 13) is now a `ModelConfig` field: when set,
`p.M_eta = pi^2*M_GB/(4*W_GB)` (via `gb_obstacle_energy.m_eta_from_m_gb`),
overriding `eta_mobility_scale`/`eta_diffusivity_fixed_physical` entirely;
`None` (default) preserves existing behavior exactly.
`tests/test_model_gamma_gb_override.py` gained two tests confirming this
(`test_gb_mobility_m4_J_s_none_preserves_existing_behavior`,
`test_gb_mobility_m4_J_s_sets_m_eta_via_analytic_mapping`).

## 14. Static Young-Herring qualification

`scripts/m14g_young_herring_wedge.py` builds a **synthetic** wedge (not the
sintering-neck geometry): a triple junction where a vertical GB
(`v_gb=(0,-1)`) meets two free-surface branches at the prescribed local
tangents `v_s1=(-sin(alpha),cos(alpha))`, `v_s2=(sin(alpha),cos(alpha))`, so
psi=2*alpha is set *directly* by construction, with no simulation needed to
produce it. `tj_force.compute_tj_force` is called with the exactly-known TJ
location.

**Equilibrium check (Step 1)**: built at `alpha_eq=acos(gamma_gb/(2*gamma_
s))` using an exact infinite-line construction (matches the prescribed
tangent everywhere, giving the most accurate possible static measurement):

| gamma_gb/gamma_s | psi_eq (deg) | psi_measured (deg) | F_TJ_mag/gamma_s |
|---|---|---|---|
| 0.5 | 151.04 | 151.058 | 2.12e-4 |
| 1.0 | 120.00 | 120.040 | 6.00e-4 |
| 1.4 | 91.15  | 91.204  | 7.27e-4 |

All three ratios: sub-0.1-degree agreement and `F_TJ_mag/gamma_s` at the
`1e-4`-`1e-3` level — i.e. an analytically-constructed wedge at the
Young-Herring angle is, to within diffuse-interface discretization error, a
genuine force-balance equilibrium of the single unified `F[f,eta]`. **This
directly confirms the corrected free energy places its GB/free-surface
triple-junction equilibrium at the theoretically predicted angle** — the
core thermodynamic claim of this milestone.

**Perturbed dynamic relaxation (Step 2)**: attempted but **not cleanly
qualified** in this session. An infinite-straight-line construction produces
a strong, wrong-direction artifact (the box's total free-surface arc length
scales as `h_box/cos(alpha)`, a *spurious box-scale* energy gradient that
dominates the true local TJ force and drives `alpha` toward 0 regardless of
`gamma_gb/gamma_s`). Replacing it with a saturating profile (`h(x) = lambda*
cot(alpha)*(1-exp(-|x|/lambda))`, matching the prescribed local slope at the
TJ and flattening for `|x|>>lambda`) removes the confirmed box-truncation
bias (equilibrium-check accuracy unaffected) but the dynamic run *still*
drifted away from `psi_eq` over the ~6.5e-3s tested (`psi`: 77.7 deg -> 70.7
deg at `gamma_gb/gamma_s=1`, `F_TJ_mag/gamma_s` *increasing* over that
window). A control run with eta frozen (surface diffusion only) showed no
comparable drift, isolating the effect to the eta-f *coupling* rather than a
pure surface-diffusion bug — but the total free energy `F` was also
monotonically *decreasing* throughout (confirming a genuine, bug-free
gradient descent), and by a tiny fraction of its total value (~0.002%) while
psi moved substantially, meaning the wedge's overall energy landscape is
extremely flat in this direction at the tested scale and a slow shape-
relaxation transient in the arbitrarily-chosen saturating profile (not a true
zero-curvature-everywhere joint f-eta equilibrium at `t=0`) likely dominates
the short-time signal. Getting a quantitatively clean dynamic version of this
test right requires a properly pre-equilibrated joint (f,eta) initial
condition (e.g. an actual flat-far-field groove similarity solution) and/or
much longer run times to separate the local-TJ timescale from the
far-field-shape-relaxation timescale — left as follow-up work, not resolved
here.

**Verdict**: Section 18 gate D ("synthetic Young-Herring equilibrium
passes") is read as the *static* equilibrium check (explicitly what Section
14 asks to verify "first" and "directly"), which passes cleanly for all
three ratios. The dynamic extension is honestly reported as inconclusive
rather than claimed as a pass.

## 15. Question A status

Not re-run in this milestone (Section 0: do not extend the existing
Question-A trajectories, do not run the full coupled sintering problem).
Given Section 14's clean static pass but inconclusive dynamic extension, and
per Section 18's hard stop, Question A's TJ-vs-sintering-rate classification
remains **not yet re-qualified** under the corrected, unified free energy.

## 16. Preserved evidence: surface-reconstruction-limited TJ motion

Milestone 14F's qualitative observation — that increasing surface mobility
produced a markedly more mobile TJ/dihedral response — is preserved as a
hypothesis worth retesting once the dynamic Young-Herring test (Section 14,
Step 2) is properly qualified: `v_TJ` in an actual sintering neck is
geometrically coupled to the free surface (GB motion away from the neck
lengthens the GB unless the neck geometry reconstructs via conserved surface
diffusion), so `v_TJ` can saturate with `M_GB` and become `M_s`-controlled.
This remains a hypothesis, not a requalified quantitative result — the
mis-normalized/decoupled Milestone 14F psi trajectory that originally
suggested it is *not* treated as physically qualified (per Section 16 of the
handoff), and this milestone's own Section 14 Step 2 attempt does not yet
supply a clean quantitative replacement.

## 17. Impact on Milestones 14-14D

The old `Wc=36*gamma_gb/W` was not a physically normalized GB coupling (it
was calibrated so a tanh eta profile carries `19*gamma_declared` of excess
energy, not so that the constrained free energy's actual equilibrium carries
`gamma_declared`). Correcting the shared free energy changes both the `f`
transport chemical potential (`Wc` inside `mu0_bulk`) and the eta GB
thermodynamics (`k_eta`, `Wc` inside `structural_thermodynamic_force`)
together, since Milestone 14G ties both to the *same* `gb_obstacle_
coefficients(gamma_gb, W)` call. Any Milestone 14-14D result whose
quantitative value depended on the old `gamma_gb` coefficient formula (e.g.
groove strength, `psi(t)` trajectories under the old calibration) must
eventually be requalified against the new `Wc=4*gamma_gb/W`,
`k_eta=4*gamma_gb*W/pi^2` coefficients — the old groove strength is **not**
preserved as an invariant across this change, per Section 17's explicit
instruction. The narrow `A=100nm, lambda=320nm` sinusoidal-substrate geometry
remains a useful geometric candidate for future requalification.

## Test suite

229/229 passing (`../.venv/bin/python -m pytest -q`), up from the starting
checkpoint's 219: 8 new (`test_gb_obstacle_energy.py`), 2 new
(`test_tangent_cone_eta.py`'s `g_external` regression tests), 2 new
(`test_model_gamma_gb_override.py`'s `gb_mobility_m4_J_s` tests); 2 removed
(the deprecated 19x-factor tests), replaced by the `gb_obstacle_energy`
suite's coverage of the same tanh-vs-obstacle facts, correctly reframed.

## New files

- `pf_sintering/gb_obstacle_energy.py` — obstacle-profile derivation,
  unified coefficient provider, mobility mapping (Sections 1, 2, 6, 10).
- `tests/test_gb_obstacle_energy.py` — 8 tests verifying the above.
- `scripts/m14g_planar_equilibrium_profile.py` — Section 9.
- `scripts/m14g_circular_grain_calibration.py` — Section 11.
- `scripts/m14g_planar_bias_mobility.py` — Section 12.
- `scripts/m14g_young_herring_wedge.py` — Section 14.

## STOP

Per Section 18, this milestone stops here. Not run: `M_GB` vs. `M_s` rate
competition, G1-G4 classification, full coupled sintering, independent
`M_TJ`. Gates A (planar equilibrium `gamma_GB` correct), B (f and eta share
one free energy), and C (circular/planar `M_GB` agreement) are satisfied with
clean, quantitative, grid-converging evidence. Gate D (synthetic Young-
Herring equilibrium) is satisfied for the static force-balance check; its
dynamic extension is explicitly left unresolved rather than claimed. sink,
hazard, RBM, and anisotropy remain OFF throughout.
