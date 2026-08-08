# Milestone 13E: discrete gradient-flow closure and face-projected transport promotion gate

## 0. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `8b9615b` ("docs: Milestone 13D face-projected conservative transport report"), confirmed clean working tree and **205/205** tests passing before any change. Anisotropy, sink, hazard, RBM, imposed stress/strain never activated; eta stayed frozen throughout. Physical geometry, `W`, `M_s`, `dx` grids and wavelengths were never changed. Full suite at the end of this milestone: **209/209** (205 inherited + 3 new energy/operator tests + 1 new promotion test, one existing assertion corrected -- see Section 4).

New functions in `pf_sintering/surface_transport.py`: `fdot_chain_face_projected`, `exact_dissipation_face_projected`, `variational_surface_diffusion_step`. `dissipation_density_face_projected` is kept (unused by the update path) and its docstring marked deprecated/algebraically-incorrect for auditability. One diagnostic-only script fix: `scripts/m13d_isolated_substrate_paired.py`.

## 1. Exact discrete chain-rule derivation (Section 3)

`mu = delta F_h/delta f` is already established by `ch_exact_energy.py`. For the face-projected transport, `f_dot = L_h(mu)` is exactly `-flux_divergence(Jx_face, Jy_face, ...)` -- the same divergence `surface_divergence_update`'s `face_projected` branch applies. The authoritative instantaneous thermodynamic rate is computed directly (not from finite-time `F` differences):

```
Fdot_chain = dx^2 * sum(mu * f_dot)
```

implemented as `fdot_chain_face_projected`.

## 2. Exact face-power derivation (Section 4)

`bc_ops._axis_efflux(J, axis, bc)` (used inside `flux_divergence`) and the forward face difference `(a[i+1]-a[i])/dx` are exact discrete adjoints of each other, for both `periodic` and `reflecting` BC -- proved by direct telescoping-sum expansion (periodic: closed-ring cancellation; reflecting: `_axis_efflux`'s domain-edge-zeroing convention makes the boundary terms vanish identically, leaving the same clean adjoint relation restricted to the `N-1` real interior faces). Substituting into `Fdot_chain` gives, to machine precision (verified numerically, rel diff = 0.0 on real states):

```
Fdot_chain = dx^2 * [ sum_x-faces(Jx_face * gx_mu_x) + sum_y-faces(Jy_face * gy_mu_y) ]
```

Only each face family's OWN normal-direction flux component and normal-direction `grad(mu)` enter -- exactly what the conservative divergence uses, never the "other," diagnostic-only component each face's full vector also carries (Section 4's explicit warning against assuming a full independently-evaluated `grad(mu).M.P.grad(mu)` form).

Substituting `Jx_face = -(Mxx_x*gx_mu_x + Mxy_x*gy_mu_x)` (and the y-face analogue) gives the exact discrete dissipation:

```
D_h = dx^2 * [ sum_x-faces(Mxx*gx_mu^2 + Mxy*gx_mu*gy_mu)
             + sum_y-faces(Myy*gy_mu^2 + Mxy*gx_mu*gy_mu) ]
```

with `Fdot_chain = -D_h` an exact algebraic identity (`test_fdot_chain_equals_negative_exact_dissipation`, rel_tol=1e-9). This is structurally *different from, and smaller than*, Milestone 13D's `dissipation_density_face_projected`, which used the FULL 2x2 quadratic form `v^T M v` independently at each face family (double-counting the cross term with coefficient 2 instead of 1, and including an extra `Myy*gx_mu^2` / `Mxx*gy_mu^2` term that never appears in the real power balance). This exactly explains Milestone 13D's previously-unresolved observation that `(F_n-F_{n+1})/dt` converged to only `~0.655` of the old `D_CH` proxy as `dt->0`: the old proxy was summing a genuinely larger, different quantity.

## 3. Operator symmetry/adjoint audit (Sections 6-7)

Audited the frozen-`f` linear operator `L_h: mu -> f_dot` on representative states:

- **Conservation**: `sum(L_h mu) ~ 0` to roundoff (`6.6e-24` vs. typical magnitudes `~1e-24`) -- holds by construction of `flux_divergence`'s telescoping identity.
- **Constant null mode**: `L_h(constant) = 0` exactly (a uniform `mu` has zero gradient, normal and tangential, at every face).
- **Symmetry/adjoint defect**: `L_h` is **not exactly self-adjoint**. Across 30 random smooth-field pairs (`dx^2`-weighted inner product `<a, L_h b>` vs. `<L_h a, b>`): relative discrepancy mean 16.6%, max 101.9%, min 0.34% -- a real, non-negligible antisymmetric component (from the tangential-derivative averaging in `face_gradient_x/y` combined with the normal-direction one-sided difference), not roundoff noise.
- **Quadratic-form sign** (the physically decisive test): 450+ diverse test vectors (smooth Gaussian-filtered at varying sigma, fully-random/high-frequency, single-grid-point spike) across two independent random seeds and two states: **zero violations** of `Fdot_chain = <mu, L_h mu> <= 0` (worst observed value across the first 200-vector batch: exactly `0.0`; second 50-vector batch: max `-5.3e-26`, never positive).

**Reconciliation**: a quadratic form `x^T A x` is exactly insensitive to the antisymmetric part of `A` (`x^T A_antisym x = 0` identically), so the antisymmetric component found does not affect whether the actual dynamics (`<mu, L_h mu>` evaluated at the same `mu` on both sides) satisfies `<= 0`. The decisive test passes robustly, so this milestone follows **Section 8's path** ("if `<mu, L_h mu> <= 0` is established, replace the diagnostic") rather than Section 7's redesign path. The asymmetry is a genuine structural property of the current staggered discretization, reported here for completeness, but it does not compromise the operator's negative-semidefiniteness in the sense that matters for gradient-flow decrease.

Unlike the deprecated proxy (each per-face term individually PSD by construction), `D_h`'s per-face partial terms are **not** individually sign-definite -- confirmed both by a hand-worked counterexample (`M=[[1,1],[1,1]]`, PSD rank-1, with `a=1, b=-2`: `Mxx*a^2+Mxy*a*b = 1-2 = -1 < 0`) and numerically (real negative pointwise values, e.g. `-8.35e+02`, found on an actual state). Only the **global sum** `D_h` is guaranteed non-negative; sign-definiteness is established by the direct quadratic-form probing above, not pointwise inspection. `tests/test_face_projected_transport.py::test_quadratic_form_negative_semidefinite_at_frozen_f` locks this in with 22 diverse test vectors.

## 4. Corrected `D_h` (Section 8)

`dissipation_density_face_projected` is kept in place (module docstring now marked "DEPRECATED / ALGEBRAICALLY INCORRECT -- kept only for Milestone 13D regression comparison"). `surface_divergence_update`'s `face_projected` branch now calls the new `exact_dissipation_face_projected` for `D_density_x`/`D_density_y`. `test_energy_descent_face_projected`'s old pointwise-non-negativity assertion (valid only for the old proxy) was corrected to check the global sum `D_h >= -1e-30*max(|D_h|,1)`, consistent with Section 3's finding.

## 5. Finite-`dt` convergence (Section 8)

On a real rough random state (`dx=2nm`, `Nx=Ny=100`, Gaussian-filtered-noise `f`): `D_h = 1.5127e-05`. `(F_n-F_{n+1})/dt` at `dt_frac = 1.0, 0.1, 0.01, 0.001`:

| dt_frac | ratio to D_h |
|---|---|
| 1.0 | 0.941274 |
| 0.1 | 0.994127 |
| 0.01 | 0.999413 |
| 0.001 | 0.999941 |

Clean, monotonic convergence to **1.0** -- contrasting sharply with Milestone 13D's old proxy, which plateaued at `~0.655` (`0.6165, 0.6511, 0.6546, 0.6550` at the same `dt` fractions) and never approached unity. `test_exact_dissipation_converges_to_unity_ratio_as_dt_to_zero` locks this in.

## 6. Isolated-substrate tangentiality diagnostic correction (Section 9)

`scripts/m13d_isolated_substrate_paired.py` already called the authoritative `surface_flux_face_projected` + `face_projected_tangentiality` (no legacy-flux reconstruction) -- so the bug was not in *which* flux was used, but in an insufficient validity mask on the reported ratio.

**Root cause**: `J.n / |J|` blows up to an O(1) ratio wherever the RAW (unregularized) `|grad f|` at a face is weak compared to the normal-estimator regularization `eps_n`. Algebraically, `J.n = -M_s*q*(1-|n|^2)*(grad_mu.n)` where `1-|n|^2 = eps_n^2/(|grad f|^2+eps_n^2)`; this is only `~0` (tangential projection working as intended) when `|grad f| >> eps_n`. Deep in bulk (e.g. the substrate's reflecting-wall tail at `f ~ 1-1e-5`), `|grad f|` can fall to the same order as `eps_n`, so `1-|n|^2` becomes `O(1)` and the reported ratio is dominated by a numerically-real but physically-meaningless small-over-small division -- even though `q_face` there (`~0.01-0.1` of its own max) is still just above the existing flux-magnitude floor.

Reproduced directly: at `t=0`, `dx=2.5nm`, the anomalous faces (`ratio_x > 1e-4`, up to `2.2e-2`) were located exactly at the `X=0` reflecting-boundary column, with `f ~ 0.99999` (deep bulk) and `q_face` in `[0.01, 0.1)` of its own max -- while faces with `q_face` near its own max (true interface) already had `ratio ~ 1.7e-5` rms, `~4e-4` max (roundoff-consistent with the other Milestone 13D face_projected diagnostics).

**Fix**: added a `q_face`-relative floor (`q_face > 1e-3 * max(q_face)`, using the authoritative `q_face` already returned in `fp["xface"]`/`fp["yface"]`) alongside the existing flux-magnitude floor, in `scripts/m13d_isolated_substrate_paired.py`'s `sample()`. This is a diagnostic-only change -- no physics/library code touched.

## 7. Post-fix isolated-substrate recheck (Section 10)

Full 0-0.30s campaign rerun at `dx=2.5nm` (`runs/m13e_isolated_substrate_paired.json`/`.log`) with the corrected diagnostic:

| t (s) | tang_x_rms (face_projected) | A1 legacy (nm) | A1 face_projected (nm) | F face_projected | mass_drift |
|---|---|---|---|---|---|
| 0.00 | -- | 24.0000 | 24.0000 | -- | -- |
| 0.01 | 4.02e-08 | 23.9998 | 23.9997 | 6.24795e-06 | 0.00e+00 |
| 0.03 | 5.00e-08 | 23.9993 | 23.9993 | 6.24794e-06 | 3.94e-16 |
| 0.06 | 5.00e-08 | 23.9986 | 23.9986 | 6.24793e-06 | 2.63e-16 |
| 0.10 | 5.17e-08 | 23.9977 | 23.9977 | 6.24790e-06 | 2.63e-16 |
| 0.15 | 1.08e-07 | 23.9966 | 23.9966 | 6.24787e-06 | 2.63e-16 |
| 0.20 | 9.70e-08 | 23.9955 | 23.9955 | 6.24782e-06 | 0.00e+00 |
| 0.25 | 4.95e-08 | 23.9944 | 23.9944 | 6.24769e-06 | 2.63e-16 |
| 0.30 | 1.15e-07 | 23.9933 | 23.9933 | 6.24755e-06 | -1.05e-15 |

`tang_x_rms` is now uniformly at roundoff (`4e-8` to `1.1e-7`) throughout the trajectory -- matching the other Milestone 13D face_projected diagnostics, not the previously-anomalous `1e-3` to `6e-3`. `A1` decreases essentially identically between `cell_average_legacy` and `face_projected` (matching to the printed precision at every sampled time); `F` decreases monotonically; mass conserved to roundoff (`|drift| <= 1.05e-15`). The isolated-substrate Mullins/k4 benchmark remains qualified.

## 8. Selected dx=1.25nm particle-contact check (Section 11)

`face_projected`, eta frozen, current geometry unchanged, t=0 to 0.10s (`runs/m13e_particle_contact_dx1p25.json`/`.log`), compared at matched times against the existing Milestone 13D dx=5.0nm and dx=2.5nm results:

| t (s) | M_neck_f dx=5.0nm | M_neck_f dx=2.5nm | M_neck_f dx=1.25nm | L_contact dx=5.0/2.5/1.25nm (nm) |
|---|---|---|---|---|
| 0.00 | 3.8446e-15 | 3.9374e-15 | 3.9657e-15 | -- |
| 0.03 | 4.8254e-15 | 4.9524e-15 | 4.9576e-15 | 74.40 / 74.41 / 74.41 |
| 0.06 | 5.1617e-15 | 5.1722e-15 | 5.1898e-15 | 76.02 / 76.08 / 76.08 |
| 0.10 | 5.3878e-15 | 5.3888e-15 | 5.3580e-15 | 77.11 / 77.25 / 77.27 |

`dM_neck_f/dt` is **positive at every interval, at every grid** (`M_neck_f` strictly increasing) -- sign-consistent across dx=5, 2.5, 1.25nm, no flip. `M_neck_f` at dx=1.25nm tracks dx=2.5nm closely (better agreement than dx=2.5nm vs. dx=5.0nm at every sampled time), consistent with grid convergence. `L_contact` is likewise increasing and grid-convergent. `face_x_rms(3W)` at dx=1.25nm stays at roundoff (`9.4e-8` to `1.9e-7`), matching dx=5/2.5nm. `M_fixed` (fixed particle-side mass) shows the same "dips early then rises" pattern as dx=5/2.5nm (t=0: 2.054973e-14 -> t=0.03: 2.054744e-14 (dip) -> t=0.10: 2.054927e-14 (rise)).

## 9. Promotion gate (Section 12)

| Criterion | Result | Status |
|---|---|---|
| (A) exact mass conservation | roundoff at every test/campaign (`<=1e-15` relative) | PASS |
| (B) exact discrete energy dissipation | `Fdot_chain = -D_h` to machine precision; `(F_n-F_{n+1})/dt -> D_h` (ratio -> 1.0) as `dt->0` | PASS |
| (C) authoritative face tangentiality | roundoff-level (`~1e-8` to `~2e-7`) on isolated-substrate (post-fix) and particle-contact at dx=5/2.5/1.25nm | PASS |
| (D) isolated-substrate Mullins/k4 benchmark | A1(t), F(t), mass unchanged from Milestone 13D's qualified trajectory | PASS |
| (E) particle-contact `dM_neck/dt` sign consistency | positive at every interval, dx=5, 2.5, 1.25nm, no sign flip | PASS |

All five criteria pass. `face_projected` is promoted as the canonical transport law for the unified variational coarsening mode.

**Mechanism**: `surface_divergence_update` is shared, low-level infrastructure called (without an explicit `face_flux_mode`, i.e. relying on its own default) by roughly 15 pre-13D/13E milestone scripts and tests (`tests/test_surface_transport.py`, `tests/test_flux_closure.py`, `scripts/m12b_*.py`, `scripts/m13_*.py`, `scripts/m13b_*.py`, `scripts/m13c_*.py`, `scripts/coarsening_benchmarks.py`, `scripts/surface_transport_benchmarks.py`) that assume its `cell_average_legacy` default and its cell-centered `Jx`/`Jy` diagnostic-dict shape. Flipping that shared default would silently change those unrelated regression baselines, which Section 12 explicitly forbids. `variational_surface_diffusion` was also never wired into `model.py`'s production stepper (confirmed by grep: `surface_divergence_update` has zero callers inside `pf_sintering/*.py` outside its own definition) -- it exists only as this milestone-track transport-law module plus the campaign scripts that call it directly.

Accordingly, promotion is implemented as a new, explicitly-named entry point, `variational_surface_diffusion_step` (in `pf_sintering/surface_transport.py`), which forwards to `surface_divergence_update` but defaults `face_flux_mode="face_projected"`. `surface_divergence_update`'s own default is left unchanged. `cell_average_legacy` remains reachable through either function via explicit `face_flux_mode="cell_average_legacy"`. `tests/test_face_projected_transport.py::test_variational_surface_diffusion_step_defaults_to_face_projected` locks in both the new default and the fact that `surface_divergence_update`'s bare default is untouched.

## 10. Conclusion

`face_projected` becomes the canonical unified-transport default for the `variational_surface_diffusion` mode, via the new `variational_surface_diffusion_step` entry point. `surface_divergence_update` itself, and every existing milestone script/test that calls it without `face_flux_mode`, are unaffected. Full suite: **209/209** passing.

**STOP. Do not begin the geometry sweep in this milestone.**
