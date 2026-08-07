# Milestone 1 + 2 report: coarsening-driven stress buildup, sink-off directional benchmark

Scope: Milestones 1 and 2 only, per `CODEX_PHYSICS_SEQUENCE.md` / handoff Sections 7–11 / 18. No hazard tuning, forced events, long runs, parameter sweeps, or PR model were attempted. **Stopping here for review, as instructed.**

## A. Baseline commit and branch

- Branch: `codex/coarsening-stress-buildup`
- Baseline commit (branch start / qualified state): `374578ac39c1e2fca63dc3ca3d90f8683cca3d62`
- A newer commit `b4a19de` (adding `PHYSICS_BACKGROUND.md`) was pulled in mid-session; no conflicts.
- One pre-existing uncommitted change was found on disk at session start (a diagnostic decomposition of `Stress` — `sigma_lt`/`sigma_curv`/`psi`/`psi_eq`/`Fx_cahn_hoffman` fields and their computation in `compute_stress`). It matched exactly what Milestone 1 asks for, so it was kept and built upon rather than discarded.

## B. Tests run and pass/fail status

- Baseline (before any change): `8 passed` (`pytest ./PF-Sintering/tests -q`).
- After all Milestone 1/2 changes: `14 passed` (8 original + 6 new: `tests/test_diagnostics.py` ×3, `tests/test_mechanism_controls.py` ×3).
- No existing test was modified or weakened. The mass-preserving structural-projection contract and MATLAB-parity-kernel-wiring tests are untouched and still pass.

## C. Selected near-critical initial geometry

A cheap t=0 scan (`scripts/scan_near_critical_geometry.py`, no time evolution — just `initialize_fields` + one `compute_stress` call per candidate) was run over `aspect_ratio ∈ {1.5, 2.0, 2.5}` × `initial_overlap ∈ {3, 5, 8, 12, 16, 20} nm`, `contact_orientation = short_plane`, on the README's canonical small-grid geometry (`dev` preset, `nx=96, ny=128, dx=5 nm, r2=80 nm`).

All 18 candidates were resolved (not disconnected, not burrowed) with finite `sigma ∈ [46.6, 54.7] MPa`. None had a *measured* (as opposed to equilibrium-fallback) dihedral angle at `t=0` — expected, since the analytic tanh initial condition has not yet relaxed into a resolvable meniscus/triple-junction shape; `psi` falls back to `psi_eq` for every candidate, which is a documented behavior of `compute_stress`, not a bug.

Selected candidate: **`aspect_ratio = 2.0`, `initial_overlap = 5 nm`** (`r2 = 80 nm`, `dx = 5 nm`, `nx=96, ny=128`, `GS1 = 180 nm`). This is the same geometry already used as the project's canonical small-test example in `README.md`. It gives a resolved, finite-stress, non-degenerate contact (`x_neck/GS1 = 0.219`, `sigma = 52.6 MPa`) without being pathologically over- or under-lapped. Robustness of the Milestone-2 result (below) was cross-checked against `overlap ∈ {3, 20} nm` × `aspect_ratio ∈ {1.5, 2.5}` (4 additional geometries); the qualitative sign of the response was identical in every case, so no further search or "controlled continuation" run was needed to reach a usable near-critical state.

## D. Confirmation that sink/RBM were disabled

Per `CODEX_PHYSICS_SEQUENCE.md` Milestone 2 ("Disable hazard and RBM completely"), the benchmark loop (`scripts/run_sinkoff_stress_matrix.py`) **never calls `hazard_step` or `rbm`** — not merely prevents activation. Each step only runs: CH (`evolve_f`) → mass-preserving `eta` projection → `ostwald_substrate` → structural relaxation (`evolve_eta`) → mass-preserving `eta` projection → `compute_stress` (diagnostic only). This reproduces the qualified per-operator sequence in `pf_sintering/runner.py` so the mass-accounting contract in `README.md` is preserved.

Verified by construction and by output:
- `Sink.cumulative_strain` stays exactly `0.0` in every case (never touched).
- `Sink.active` stays `0` throughout (never touched).
- Total solid mass `f` drift over 827 steps: `+1.8e-15` to `+4.6e-15` relative (floating-point roundoff) in every one of the 4×5 = 20 runs — i.e. conservation holds to machine precision with no RBM/hazard contribution.

One documented side effect: because `hazard_step` is the only place that updates `Sink.g_ex` (dynamic GB-excess energy), `g_ex` stays pinned at its initial value of `0` for the whole benchmark, so `effective_gamma(s, p) == gamma_gb_ref` throughout. This is a simplification specific to this isolation benchmark (matching the explicit "disable hazard completely" instruction), not a defect; it should be revisited when the hazard is reconnected at Milestone 5.

**Separation tolerance.** "Fixed separation" was checked two ways:
1. *Rigid-body translation* (the mechanism `dL ≈ 0` is meant to rule out): exactly zero by construction, since `rbm()` — the only function that translates the fields — is never called.
2. *Centroid drift from asymmetric shape evolution* (a distinct, smaller effect: material removed from the particle's far cap and re-deposited near the neck shifts the mass-weighted x-centroid of `e2` slightly, with no translation involved): measured directly as `separation = center(e2,p) − wall_x0`. Over the 827-step / `|ΔV2|/V20 ≈ 3.0e-4` benchmark, `|ΔL| = 0.30–0.56 nm` against `L0 ≈ 89–133 nm` across the 5 geometries tested (0.31–0.43%). This is small and is reported explicitly in the table below (`Δseparation`) rather than assumed to be zero.

## E. Exact definitions used (new/clarified for this diagnostic)

- **Neck/contact width** `x_neck`: unchanged, existing `Stress.x_neck` from `model.compute_stress` (half-width of resolved solid at the `e1·e2` overlap column).
- **GB/contact measure** `A_GB`: `Σ(4·e1·e2)·dx²` — domain integral of the same `4·e1·e2` weighting `model.contact_width` already uses for `x_neck`, so it is dimensionally an area and independent of the single-column neck-width measure. (`pf_sintering/diagnostics.py::contact_area`)
- **Local sintering stress** `sigma`: unchanged, existing `Stress.sigma = sigma_lt + sigma_curv` from `compute_stress`, where `sigma_lt` is the line-tension/dihedral (Cahn-Hoffman-corrected when anisotropy is active) term and `sigma_curv = gamma_s·kappa` is the curvature term. **Not** replaced with a uniform potential/area stress, per instructions.
- **Total interfacial energy** `G_interface = E_surf + E_gb`:
  - `E_surf = gamma_s · Σ|∇f| · dx²` (total-variation estimate of interfacial length × surface energy).
  - `E_gb = effective_gamma(s,p) · A_GB` (effective GB energy density × GB/contact area, same `A_GB` as above).
- **Separation** `L = center(e2,p) − wall_x0`, where `wall_x0 = (substrate_wall_frac − 0.5)·Nx·dx` is the fixed geometric constant `compute_stress` already uses internally for its "particle burrowed into substrate" check — a run-configuration constant, not a field-derived quantity, so it cannot itself drift.

All definitions are implemented once in the new `pf_sintering/diagnostics.py` module (`sample()` / `summarize()`), covered by `tests/test_diagnostics.py`.

## F/G. 2×2 mechanism matrix and results

Both mechanism controls are new, independent runtime knobs that default to the previously qualified baseline behavior:

- **H1 control** — `ModelConfig.reservoir_neck_unprotected` / `Params.reservoir_neck_unprotected` (default `False`): when `True`, `ostwald_substrate` (in `parity_kernels.py`, the active MATLAB-parity implementation, and mirrored in `model.py`'s shadowed copy) sets the neck-exclusion weight `incl = 1` instead of the Gaussian exclusion `1 − exp(−½(...))`, so removal/deposition is governed purely by the `surf·e2` / `surf·e1` interface weighting with no artificial neck protection. Verified independently in `tests/test_mechanism_controls.py`: at `t=0` this roughly doubles the fraction of `e2` removal occurring within 3 interface-widths of the neck column (14.5% → 28.3% for the selected geometry).
- **H2 control** — `ModelConfig.eta_mobility_scale` (default `1.0`), multiplies `Params.M_eta` only; `M_f` (free-surface mobility) is untouched. `eta_mobility_scale=0.0` reproduces the "very low/zero GB mobility" limit exactly (`M_eta = 0.0`).

Benchmark: same starting fields for all 4 cases, `827` steps (`dt ≈ 7.26e-6 s`), reaching `|ΔV2|/V20 ≈ 3.0e-4` (within the requested `1e-4`–`1e-3` window), on the selected geometry (`aspect_ratio=2.0`, `overlap=5 nm`).

| Case | Reservoir | GB mobility | ΔV2/V20 | Δseparation (nm) | Δstrain | Δx_neck (nm) | ΔA_GB (nm²) | Δσ_local (MPa) | ΔE_surf (J) | ΔE_gb (J) | ΔG_interface (J) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | current (protected) | current | −3.00e-4 | −0.411 | 0 | **+0.381** | +34.07 | **−0.903** | −2.880e-9 | +3.41e-17 | −2.880e-9 |
| B | unprotected | current | −3.00e-4 | −0.405 | 0 | **+0.380** | +34.25 | **−0.903** | −2.878e-9 | +3.42e-17 | −2.878e-9 |
| C | current (protected) | zero | −3.00e-4 | −0.374 | 0 | **+0.325** | +27.86 | **−0.841** | −2.884e-9 | +2.78e-17 | −2.884e-9 |
| D | unprotected | zero | −3.00e-4 | −0.367 | 0 | **+0.323** | +28.04 | **−0.841** | −2.882e-9 | +2.80e-17 | −2.882e-9 |

Mass conservation (`f`) held to `~2–5e-15` relative in all four cases. `Δstrain = 0` exactly in all four (RBM never called).

Robustness check (same 2×2 matrix, `overlap ∈ {3,20} nm × aspect_ratio ∈ {1.5,2.5}`, 4 more geometries, 16 more runs): **identical qualitative signs** in every single case — `x_neck` always widens, `sigma` always falls, `A_GB` always grows, as `V2` shrinks under sink-off coarsening. No geometry in the tested set reproduces the required direction.

## H. Directional derivatives

For the selected geometry, case A (reference):

- `dx_neck/dV2 = (last.x_neck − first.x_neck)/(last.V2 − first.V2)` → **negative** (x_neck rises as V2 falls) — wrong sign vs. required.
- `dSigma_local/dV2` → **positive** (sigma falls as V2 falls) — wrong sign vs. required.
- `dG_interface/dV2 = ΔG_interface/ΔV2 > 0`, i.e. **`G_interface` decreases as coarsening proceeds** (`ΔG_interface < 0` throughout) — this part is correct and consistent with spontaneous coarsening in every one of the 20 runs.

(`summarize()` in `pf_sintering/diagnostics.py` computes these directly from the sample trajectory; exact values are in `runs/sinkoff_stress_matrix*.json`, not committed per repo convention — `runs/` is gitignored.)

## I. Conclusion: is H1 or H2 supported?

**Neither H1 nor H2 is supported as the primary suppressor**, and this appears robust rather than geometry-specific:

- Removing the reservoir's neck-exclusion protection entirely (**A→B**, **C→D**) changes the outcome only marginally (e.g. `Δx_neck`: `+0.381→+0.380 nm`, `Δσ`: `−0.903→−0.903 MPa`) even though the control itself is verified to materially change *where* Ostwald removal happens at the operator level. **H1 is not supported.**
- Driving GB/eta mobility to exactly zero (**A→C**, **B→D**) reduces the magnitude of the wrong-direction response somewhat (`Δx_neck: +0.381→+0.325 nm`, `Δσ: −0.903→−0.841 MPa`) but does not reverse its sign. **H2 has a measurable but insufficient effect — not primarily supported either.**
- All four cases fail the required directional signature (`dx_neck<0` and/or `dA_GB<0`, `dSigma_local>0`) while correctly satisfying `dV2<0`, `dL≈0` (up to the small centroid-drift tolerance above), `dstrain=0`, and `dG_interface<0`.

Per Section 11 of `CODEX_PHYSICS_SEQUENCE.md` / Section 8 of `PHYSICS_BACKGROUND.md`: since **none of the four cases build stress**, the next step should be to inspect free-surface transport / capillary thermodynamics / the local stress construction, and only then consider the compact PR/de-sintering sanity check — **not** to tune the hazard. This is a recommendation for the next reviewed step, not something executed in this session.

## J. Numerical pathology / ambiguity notes

1. **Dihedral angle never resolves to a measured value** (`psi == psi_eq` fallback) across every candidate geometry tried, including after 827 steps of sink-off coarsening in all 20 benchmark runs. `sigma_lt` is therefore always computed from the *equilibrium* dihedral angle plus the `1/x_neck` line-tension scaling, not from a geometrically measured triple-junction angle, and the anisotropic Cahn-Hoffman branch (`Fx_cahn_hoffman`) was never exercised in these runs either (`flanks` from `measure_dihedral` was always empty). This is worth auditing directly: `measure_dihedral()`'s TJ/branch-detection logic may need a more relaxed/patient geometry, or a longer conditioning run, before it resolves — otherwise the "local dihedral/TJ geometry" and "anisotropic Cahn-Hoffman contribution" parts of the local-stress decomposition (Section 4/6 of the handoff) are effectively dormant in this regime.
2. **The consistent widening of `x_neck` alongside falling `sigma`** as `V2` shrinks (in every one of 20 runs across two mechanism controls and 5 geometries) suggests the discrepancy is not in H1/H2 but likely upstream, in how `ostwald_substrate`'s source/sink weighting (`surf·e2`/`surf·e1`, independent of the `incl` exclusion) or the CH free-surface kernel redistributes/deposits mass near the contact region as the particle's outer cap recedes — i.e. exactly the "surface-transport thermodynamics" and "local stress construction" that Section 11 flags as the next things to inspect.
3. Centroid drift (`Δseparation` up to ~0.5% of `L0`) from asymmetric shape evolution, while small, is nonzero and should be tracked explicitly (not assumed zero) in any future longer-horizon benchmark, since it is a real (if minor) geometric effect distinct from RBM.

---

**Stopping here.** No hazard tuning, forced-event benchmark, long runs, broad parameter sweep, or PR/de-sintering model were attempted, per Section 18 / Milestone-3+ gating.
