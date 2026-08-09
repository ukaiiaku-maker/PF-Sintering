# Milestone 14E: physical grain-boundary migration calibration

Sharp-interface mobility mapping and triple-junction kinetic qualification

## 0. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `6d946d4` ("docs+feat: Milestone 14D narrow-substrate geometry stress-buildup screen"), confirmed clean working tree and **217/217** tests passing before any change. Sink, hazard, RBM, anisotropy stayed OFF throughout. The full coupled sintering calculation (surface diffusion + active eta migration in the actual sintering geometry, driving the trajectory forward) was NOT run -- every benchmark here is either an isolated eta-only geometry (circular grain, planar boundary) or a bounded diagnostic coupling of the two qualified kinetic laws for TJ characterization only (Section 6-8), never advanced to a "production" campaign. New: `pf_sintering/constrained_eta.py` was read and used AS-IS (not modified); three new scripts (`scripts/m14e_circular_grain_calibration.py`, `scripts/m14e_planar_driven_boundary.py`, `scripts/m14e_tj_force_balance.py`).

## 1. Current M_eta semantics

`constrained_eta.py`'s own module docstring documents the model precisely: at fixed `f`, `d(eta_i)/dt = -M_eta*(g_i - lambda)` with `g_i = -k_eta*lap(eta_i) + 2*Wc*eta_i*(f^2/2-f)` (the complete structural derivative of `F = integral[(W_f/2)f^2(1-f)^2 + Wc*eta2*(f^2/2-f) + (k_eta/2)*sum_i|grad(eta_i)|^2]`, `Wc=36*gl/W`, `gl=gamma_gb_ref` away from a GB) and `lambda=mean_i(g_i)` over active grains -- an exact `sum_i d(eta_i)/dt=0` identity. This milestone uses `constrained_tangent_cone_eta_update` throughout (the Milestone 13 exact-feasible-velocity version, consistent with Section 11's explicit "tangent-cone safety correction" terminology), not the older post-hoc-`reproject` `constrained_variational_eta_update`.

Two things established directly from the module's own docstring, confirmed important later: (a) the bulk coupling term is **linear** in `eta_i` (not a standard cubic/quartic double well), and is **unconditionally unstable** under forward-Euler at any fixed `dt` on its own; (b) the box constraint `0<=eta_i<=f` (enforced via `tangent_cone_projected_velocity`'s active-set projection) is therefore not a rare safety net but the mechanism that keeps the profile bounded AT ALL -- confirmed directly in this milestone's benchmarks by the active-set blocking being permanently engaged in bulk regions (eta_i exactly at its bound) rather than only occasionally near a moving front. This is NOT a standard Allen-Cahn phase-field GB model; it is a constraint-stabilized profile, and Section 4 below shows this distinction is not academic.

`M_eta` itself is currently set via `p.M_eta = M_f_base/dx**2*.01*eta_mobility_scale` (default path) or a fixed-physical-diffusivity alternative (`eta_diffusivity_fixed_physical`) -- **neither has ever been validated as a physical GB migration mobility**; this milestone's task.

## 2. Physical M_gb units and sharp-interface target

`v_n = M_gb*P_gb`, isotropic `P_gb = gamma_gb*kappa_gb`, so `v_n = M_gb*gamma_gb*kappa_gb`. Units: `[v_n]=m/s`, `[kappa_gb]=1/m`, `[gamma_gb]=J/m^2` (matching `gamma_s`'s existing convention throughout this codebase) `=> [M_gb] = m^4/(J*s)`. GB migration must not change total conserved `f` mass -- verified in every benchmark below (`mass_drift` at roundoff, `~1e-16` relative, in every run).

## 3. Circular-grain calibration

`f=1` everywhere, grain-2 circular inclusion (`R0=100nm`) in a grain-1 matrix, `W=20nm`, fully periodic BC, no free surface, no TJ (`scripts/m14e_circular_grain_calibration.py`). `constrained_tangent_cone_eta_update` at fixed `f`; `R(t)` measured via bisection on the `eta2=0.5` level set along 72 rays from the grain center. `M_gb_eff = -(1/(2*gamma_gb))*d(R^2)/dt` (linear least-squares fit).

To keep the tangent-cone safety correction negligible (Section 11's requirement) across a wide `M_eta` ladder without either using an impractically small `dt` for the largest `M_eta` or an impractically large step count for the smallest, `dt` was set via a fixed `M_eta*dt` product (`8e-14`, empirically tuned so `max_safety_frac ~ 1e-12`, true roundoff, at the largest `M_eta` tested) -- this also, as a side effect of `M_gb_eff` being linear in `M_eta` (confirmed below), gives every scale a comparable `R^2` shrinkage fraction in the same `n_steps=2000`, letting `dF` (structural free energy change) come out numerically IDENTICAL across the whole `M_eta` ladder at fixed `dx` (confirmed in the raw output) -- a strong internal-consistency check that the calibration procedure itself is behaving as designed.

**Results** (`M_eta_base=4.2667e-9`, `gamma_gb=1.0` J/m^2, `dx=2.5nm` primary):

| M_eta scale | M_eta | M_gb_eff (m^4/(J s)) | fit R^2 | max safety_frac | mass_drift |
|---|---|---|---|---|---|
| 0.5 | 2.133e-9 | 1.27496e-16 | 1.000000 | 2.9e-12 | 0.0 |
| 1 | 4.267e-9 | 2.54991e-16 | 1.000000 | 2.9e-12 | 0.0 |
| 2 | 8.533e-9 | 5.09983e-16 | 1.000000 | 2.9e-12 | 0.0 |
| 5 | 2.133e-8 | 1.27496e-15 | 1.000000 | 2.9e-12 | 0.0 |
| 10 | 4.267e-8 | 2.54991e-15 | 1.000000 | 2.9e-12 | 0.0 |
| 20 | 8.533e-8 | 5.09983e-15 | 1.000000 | 2.9e-12 | 0.0 |

`M_gb_eff/M_eta` is constant to 5+ significant figures at every scale (`5.9748e-8`) -- **exact linear proportionality**, `R^2(t)` linear to `fit_R2=1.000000` at every scale, `mass_drift` exactly `0.0` throughout (eta kinetics never touch `f`, confirmed to the last bit), `max_safety_frac~3e-12` (true roundoff) throughout.

**Grid convergence** (`M_eta` scale=1 fixed): `dx=5.0nm: M_gb_eff=2.52027e-16`; `dx=2.5nm: 2.54991e-16`; `dx=1.25nm: 2.55966e-16` -- monotonically converging, `1.6%` total spread across a 4x resolution range, differences shrinking `1.2%->0.4%` as expected for a converging sequence. **This isolated calibration is clean, robust, and grid-converged.**

## 4. Planar driven-boundary cross-check

Two flat GBs in a periodic-X domain (`f=1`, grain-2 a `200nm`-wide stripe in a `400nm` grain-1-periodic domain, so the domain closes with two symmetric flat fronts -- `scripts/m14e_planar_driven_boundary.py`). A small, **diagnostic-only** bulk free-energy bias `Delta_g` [J/m^3] was added directly in the benchmark script (added to `g2` before the tangent-cone projection, equivalent to `F_bias=Delta_g*eta2` -- never wired into `pf_sintering/constrained_eta.py`). Front position measured via bisection (tight `+-3W` bracket, required after an initial bug where a `+-10W` bracket crossed the periodic wrap/other-GB and silently corrupted the search via `_sample_bilinear`'s clip-not-wrap edge handling).

**Results** (`M_eta` scale=20, same as the largest circular-calibration point, `dx=2.5nm`):

| Delta_g (J/m^3) | v (m/s) | v/Delta_g (m^4/(J s)) | fit R^2 |
|---|---|---|---|
| 1e4 | -1.674877e-11 | -1.674877e-15 | 0.999954 |
| 1e5 | -1.674300e-10 | -1.674300e-15 | 0.999954 |
| 1e6 | -1.674243e-9 | -1.674243e-15 | 0.999954 |

**`v` linear in `Delta_g`** (consistent to 4 significant figures across a 100x range), `M_gb_eff(planar) = 1.674e-15` m^4/(J s) -- but this **disagrees with the circular-grain value at the same `M_eta`** (`5.09983e-15`) by a factor of **3.05x**.

**Root-cause investigation**: computed the ACTUAL implemented excess free energy of an isolated planar GB profile directly (analytic tanh, `gamma_gb_override=1.0`): `integral[(bulk+gradient energy density) - background] dx = 19.0` J/m^2 exactly (`18.0` from the bulk term + `1.0` from the gradient term, confirmed both on the real simulated two-GB field, `18.98`, and on a clean isolated analytic profile, `19.000` to 6 significant figures) -- **the declared `gamma_gb_override=1.0` parameter and the ACTUAL implemented interfacial energy of the eta profile differ by a factor of 19**, an exact, reproducible, analytically-confirmed miscalibration in the current `Wc=36*gamma_gb/W`, `k_eta=3*gamma_gb*W` construction (in contrast to `gamma_s`/`k_f`/`W_f`, whose corresponding physical calibration WAS validated for the surface-transport system in Milestones 12-13 via `m_s_ref`'s `SECH8_INTEGRAL` normalization -- no analogous validation exists for `gamma_gb`/`k_eta`/`Wc`).

Correcting for this (`gamma_gb_true = 19*gamma_gb_declared`): `M_gb_true(circular) = 5.09983e-15/19 = 2.684e-16` -- this does NOT resolve the discrepancy, it makes it WORSE in the other direction (`M_gb_true(planar)/M_gb_true(circular) = 1.674e-15/2.684e-16 = 6.24`, a residual ~6x gap even after the static-energy correction).

**Interpretation**: this residual disagreement is attributed to Section 1's finding that the profile shape is constraint-stabilized (box bound), not double-well-stabilized -- the standard sharp-interface reductions used to derive `R^2(t)=R0^2-2*M_gb*gamma_gb*t` and `v=M_gb*Delta_g` both implicitly assume a smooth, self-similar traveling-wave/curvature-following profile shape set by a genuine energy balance; a constraint-dominated profile need not preserve that shape identically under curvature-driven vs. bulk-bias-driven forcing, so the two "M_gb_eff" extractions are not guaranteed to agree even in principle for this specific formulation.

**Per Section 4's explicit instruction: STOP. `M_eta` is NOT used as a physical GB mobility in this milestone** (Section 5's `gb_mobility_m4_J_s` production-facing parameter was NOT added -- see Section 5 below).

## 5. M_eta <-> M_gb conversion (deferred)

**Not implemented.** Section 4's consistency check failed by a substantial, reproducible margin (a factor of ~3 before, and ~6 after, correcting for the independently-confirmed 19x static-energy miscalibration). Adding a production-facing `gb_mobility_m4_J_s` parameter now would present a false sense of calibrated confidence. Two concrete, actionable follow-up items are identified for whoever picks this up next: (1) fix the `Wc`/`k_eta` normalization so the declared `gamma_gb` matches the implemented profile's excess free energy (the required correction factor, `1/19`, is exact and reproducible for the current formula -- verify it holds across other `W`/`gamma_gb` values before hard-coding it, since the box-constraint-dominated character of this model may make the ratio configuration-dependent); (2) after that fix, redo the circular-vs-planar cross-check to determine whether the residual ~6x kinetic-mapping disagreement persists (if so, this model likely needs a genuine reformulation, e.g. a proper double-well bulk term, before a universal `M_gb` is meaningful at all).

## 6. Static Young-Herring TJ relaxation

`scripts/m14e_tj_force_balance.py`: reused the qualified sinusoidal-substrate/particle-contact geometry (`A=24nm`, `lambda=480nm`, `dx=2.5nm`, unmodified from Milestones 12-14D) but, uniquely in this milestone, turned eta kinetics ON (`constrained_tangent_cone_eta_update`) alongside the qualified `variational_surface_diffusion_step` surface transport -- both `f` and `eta` relax together from the unmodified analytic initial condition; `psi` was never prescribed. Three `gamma_gb` values (`1.4142136`, `0.9993243`, `0.5176381`, nominal Young angles `90.00/120.04/150.00deg`), `M_eta` scale=5, `t=0` to `0.20s`:

| gamma_gb | psi_Y_nominal | psi(t=0) | psi(t=0.20s) | F_TJ_mag(t=0) | F_TJ_mag(t=0.20s) |
|---|---|---|---|---|---|
| 1.4142 | 90.00deg | 77.87deg | 81.17deg | 0.787 | 0.507 |
| 0.9993 | 120.04deg | 77.87deg | 79.53deg | 0.807 | 0.689 |
| 0.5176 | 150.00deg | 77.87deg | 79.35deg | 1.099 | 1.074 |

**Measured `psi` stays clustered in a narrow ~76-81deg band across ALL THREE `gamma_gb` conditions, decoupled from every nominal target** (90/120/150deg) -- the same qualitative "psi does not track the nominal Young angle" finding from Milestones 14/14B, but this time with eta kinetics genuinely ACTIVE (confirmed by `V2`, the particle-grain ownership volume, decreasing monotonically and measurably, `-2.4%` over this window at `gamma_gb=0.9993`, not merely roundoff). **`F_TJ_mag/gamma_s` stays large** (`0.51-1.10`, i.e. 51-110% of `gamma_s=1.0`) at every `gamma_gb`, decreasing only modestly (`14-37%`) and NOT approaching zero within the tested window. Mass conservation: `mass_drift<=1.4e-15` relative in every run (roundoff). `F` decreases monotonically in every run (structural+surface free energy jointly relaxing, as expected).

## 7. Dynamic effective TJ mobility audit

Per Section 8's explicit instruction, no new TJ mobility law was added; instead the EXISTING diffuse TJ kinetics were probed for approximate `v_TJ ~ F_TJ` proportionality and sensitivity to `M_eta`/`dx`.

**`M_eta` sensitivity** (same geometry, `gamma_gb=0.9993243`, `M_eta` scale 5 vs. 20, a 4x change): `F_TJ_mag(t=0.20s)`: `0.689` (scale=5) vs. `0.682` (scale=20) -- **nearly identical**. `psi(t=0.20s)`: `79.53deg` (scale=5) vs. `79.38deg` (scale=20) -- **nearly identical**. Grain-2 volume change over the window: `-2.43%` (scale=5) vs. `-2.80%` (scale=20) -- a 4x increase in `M_eta` produced only a **15% increase** in actual migration extent, far short of the ~4x a mobility-rate-limited process would show. **The TJ/GB dynamics in this coupled system are NOT primarily rate-limited by `M_eta`** in the range tested.

**Grid (`dx`) sensitivity** (`gamma_gb=0.9993243`, `M_eta` scale=5, `dx=2.5nm` vs. `5.0nm`): `psi(t=0.20s)`: `79.53deg` vs. `80.96deg` (`1.4deg`, `~1.8%` relative difference); `F_TJ_mag(t=0.20s)`: `0.689` vs. `0.638` (`~7%` relative difference) -- a real but moderate grid sensitivity, smaller than the `M_eta`-insensitivity finding above but not negligible either.

## 8. TJ0/TJ1/TJ2 classification

**TJ2.** Neither TJ0 nor TJ1 is supported: `F_TJ/gamma_s` does NOT stay small (it stays at `0.5-1.1`, the same order as `gamma_s` itself, throughout every tested window -- ruling out TJ0's "F_TJ/gamma_s stays small"), and the dynamics are demonstrably NOT strongly tied to `M_gb`/`M_eta` (ruling out TJ1's core requirement -- a 4x `M_eta` change produced far less than a proportional response). Combined with `psi` staying decoupled from all three distinct nominal targets and a real, non-negligible `dx` sensitivity, this matches TJ2's description directly: "significant F_TJ persists purely because of diffuse kinetics" and "effective mobility changes with... M_gb" (here, is insensitive to `M_gb`, an equally disqualifying form of the same criterion). **Per Section 8: STOP -- an independently controllable TJ kinetic law (`v_TJ = M_TJ*F_TJ`, `psi_eq` never imposed directly) should be implemented and qualified before the full coupled surface+GB sintering calculation.**

## 9. Physical sources available for M_gb

No material-specific value was selected in this milestone (Section 9's explicit instruction, doubly reinforced by Section 4's failed self-consistency check -- there is no defensible `M_gb` to report yet regardless of source). Hooks for future sourcing, none pursued here: measured bicrystal mobility (e.g. capillary/curvature-driven boundary migration experiments on a specific alloy system), MD-derived mobility (atomistic GB migration simulations at the material's actual composition/misorientation), grain-growth-derived mobility (back-calculated from measured grain-growth kinetics `d(bar_D)/dt = k/bar_D`, requiring a separate, established `k`-to-`M_gb` conversion for the material system of interest), literature Arrhenius mobility (`M_gb=M_0*exp(-Q/kT)`, requiring `M_0`/`Q` for the relevant GB character/misorientation).

## 10. Recommended Lambda values for the later coupled run

`Lambda = tau_surface/tau_GB`, `tau_GB ~ L^2/(M_gb*gamma_gb)`, `tau_surface` from the already-qualified sinusoid-decay (Mullins) linearization `dA/dt=-M_s*gamma_s*k^4*A` (`k=2*pi/lambda`), giving `tau_surface=1/(M_s*gamma_s*k^4)`. At `lambda=480nm`, `gamma_s=1.0`: `tau_surface=2328s` (`surface_mobility_scale=0.3`) or `776s` (`scale=0.9`).

Because `M_gb` itself is not reliably calibrated (Section 4), `tau_GB`/`Lambda` are reported using the AS-MEASURED (uncorrected, `M_eta` scale=20, `gamma_gb_declared=1.0` J/m^2) `M_gb_eff` from BOTH the circular and planar methods, spanning the calibration uncertainty, at both requested length scales:

| L | M_gb_eff source | tau_GB (s) | Lambda (surf 0.3x) | Lambda (surf 0.9x) |
|---|---|---|---|---|
| L=65nm (contact width) | circular, declared gamma | 65nm^2/(5.0998e-15*1.0) = 0.828s | 2811 | 937 |
| L=65nm | planar, declared gamma | 65nm^2/(1.674e-15*1.0) = 2.524s | 923 | 308 |
| L=113nm (2*Ry, particle width) | circular, declared gamma | 113nm^2/(5.0998e-15*1.0) = 2.505s | 929 | 310 |
| L=113nm | planar, declared gamma | 113nm^2/(1.674e-15*1.0) = 7.647s | 305 | 102 |

**`Lambda>>1` (by roughly 2-3 orders of magnitude) across every combination tested** at this `M_eta` scale -- meaning at `M_eta` scale=20 (the value used for the largest-mobility circular/planar/TJ benchmarks in this milestone), GB migration would relax essentially instantaneously compared to surface diffusion if the two were coupled at their currently-configured rates. Since `M_gb_eff` was shown exactly proportional to `M_eta` (Section 3), reaching a BALANCED coupled regime (`Lambda~1`, comparable timescales, the physically interesting competition regime for a future coupled sintering run) requires reducing `M_eta` by roughly the same 2-3 orders of magnitude found above (e.g. `M_eta` scale `~0.01-0.1` of the value used here) -- but this recommendation should be treated as provisional pending Section 4/5's outstanding calibration-consistency issue.

## 11. Energy/mass checks

- **Total `f` exactly unchanged when surface diffusion is off**: confirmed to the last bit (`mass_drift=0.0` exactly) in every circular-grain and planar-boundary run (Sections 3-4), where surface transport was never invoked.
- **Eta evolution decreases the structural free energy**: confirmed in every circular-grain run (`dF<0`, e.g. `-4.15e-6` at `dx=2.5nm`, identical across the `M_eta` ladder by the `M_eta*dt=const` construction) and in every coupled TJ run (`F` monotonically decreasing throughout, Section 6 table).
- **No clip/projection-driven migration**: `max_safety_frac~3e-12` (true roundoff) in the isolated circular-grain benchmarks (Section 3); in the coupled TJ benchmarks (Section 6-7, a more complex, non-idealized system) `safety_fraction` is small but not roundoff-level (`~0.1-2.4%`, decreasing over the run) -- reported honestly as a real, bounded, non-negligible-but-small correction rather than claimed as negligible.
- **Tangent-cone safety correction remains negligible**: satisfied for the isolated calibration benchmarks (Section 3, `~3e-12`); NOT fully at roundoff for the coupled TJ benchmarks, but stays in the low-percent range and decreases over time, consistent with a stability-respecting `dt` per the module's own documented convention.

## 12. Summary and recommendation

Two independent, decisive STOP conditions were triggered in this milestone:

1. **Section 4**: circular-grain and planar-boundary `M_gb_eff` disagree by ~3x as configured, and by ~6x after correcting for an independently-confirmed, exact 19x static-energy miscalibration between the declared `gamma_gb` parameter and the eta profile's actual implemented interfacial energy. `M_eta` is not used as a physical GB mobility (Section 5 deferred).
2. **Section 8**: the TJ is classified **TJ2** -- `F_TJ/gamma_s` stays large (not small), and the dynamics are demonstrably insensitive to a 4x `M_eta` change, ruling out both TJ0 (quasi-equilibrated) and TJ1 (cleanly `M_gb`-tied). An independently controllable TJ kinetic law is recommended before any full coupled surface+GB sintering calculation.

**Recommended next steps, in order**: (a) fix the `Wc`/`k_eta` normalization (Section 5) and re-verify the circular-vs-planar consistency check; (b) if consistency is restored, re-run the TJ0/TJ1/TJ2 classification, since it may have been affected by the same miscalibration; (c) only then consider adding an independent `v_TJ=M_TJ*F_TJ` kinetic law if TJ2 persists; (d) use the `Lambda` framework (Section 10), not an arbitrary `M_eta` scale, once (a)-(c) are resolved, to select a physically balanced coupled-run configuration.

**STOP. Do not activate sink/hazard/RBM, and do not run the full coupled sintering calculation with active GB migration until the above is resolved.**
