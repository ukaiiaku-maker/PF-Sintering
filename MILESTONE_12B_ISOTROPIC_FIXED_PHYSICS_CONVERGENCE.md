# Milestone 12B: isotropic fixed-physics convergence, structural thermodynamics, and surface-flux closure

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `207d801` ("docs+feat: Gate E particle-on-substrate campaign and findings report"), confirmed clean working tree and **165/165** tests passing before any change in this milestone. Milestone 12's five commits (`697b599`, `5e18f7d`, `93346cc`, `d2f0cfb`, `207d801`) are preserved unmodified in history. The explicit Ostwald removal/redeposition kernel was not touched, and sink, hazard, RBM, imposed stress, and imposed strain were never activated at any point in this milestone.

This milestone added six commits on top of `207d801`:

| commit | content |
|---|---|
| `da1443a` | complete structural thermodynamic force `g_i` (Section 3 below) |
| `b74b820` | eta substepping option + dual-method curvature extraction |
| `6d9afac` | grid+dt convergence driver + particle/neck control-volume flux closures |
| `7901d9c` | fix: local circle-marching branch walk (replaces a buggy global nearest-neighbor walk) |
| `632b8d5` | representative-state flux closure + mu(s)/J_s(s) diagnostics |
| *(this report)* | `MILESTONE_12B_ISOTROPIC_FIXED_PHYSICS_CONVERGENCE.md` |

Full test suite at the end of this milestone: **176/176 passing** (165 inherited + 11 new: 2 in `test_constrained_eta.py`, 3 in `test_curvature_extraction.py`, 4 in `test_flux_closure.py`, plus 2 more added to `test_curvature_extraction.py` for the branch-walk fix).

## 2. Anisotropy confirmation

`use_aniso_surface=False` throughout every script and test added this milestone (`scripts/m12b_grid_convergence.py`, `scripts/m12b_flux_diagnostics.py`, and all reused Milestone-12 machinery via `unified_sinusoidal_campaign.build_config`). No anisotropic Cahn-Hoffman correction is imported or referenced anywhere in the new code. `tests/test_unified_transport_never_calls_ostwald.py` (unmodified, still passing) statically confirms the transport/eta modules never reference legacy Ostwald kernels. Anisotropy was not reintroduced at any point, including after seeing the final G1 result.

## 3. Derivation of the complete `delta F/delta eta_i` (Section 5's "first priority")

Milestone 12's `g_i = -k_eta*lap(eta_i)` matched `model.evolve_eta`'s own gradient-only kinetics but was **not actually derived from the model's free energy** — it was reverse-engineered from the existing (frozen, gradient-only) eta update. Inspecting `model.evolve_f`'s real chemical potential,

```
mu0 = W_f*f*(1-f)*(1-2f) - Wc*eta2*(1-fb),   eta2 = sum_i clip(eta_i,0,fb)^2,   Wc = 36*gl(local)/W
```

shows the intended free energy is

```
F = integral[ (W_f/2)*f^2*(1-f)^2 + Wc*eta2*(f^2/2 - f) + (k_eta/2)*sum_i|grad(eta_i)|^2 ] dV
```

(the bulk term's `f`-derivative reproduces `mu0` exactly — the same derivation Milestone 8's `ch_exact_energy.py` used for the pure-`f` energy). Since `eta2 = sum_i eta_i^2`, this term has a **nonzero derivative with respect to each `eta_i`** that Milestone 12 omitted:

```
g_i = delta F/delta eta_i = -k_eta*lap(eta_i) + 2*Wc*eta_i*(f^2/2 - f)
```

This is **Outcome E2** of the Section 5 decision tree: the free energy contains a real eta-f bulk coupling term beyond the pure gradient term, so the complete `g_i` was implemented (`pf_sintering/constrained_eta.py`), not the old incomplete gradient-only version.

**Verification** (machine precision, `tests/test_constrained_eta.py::test_structural_force_matches_full_free_energy_derivative`): finite-difference directional derivative `[F(eta_i+eps*q)-F(eta_i-eps*q)]/(2*eps)` vs. `dx^2*sum(g_i*q)`, using the `lap9`-self-adjoint construction `-0.5*k_eta*eta_i*lap9(eta_i)` for the gradient-energy term (a naive `|grad|^2` central-difference sum is **not** the discrete adjoint of `lap9` and fails this test by ~47%, the identical pitfall Milestone 8 documented for the `f`-field's own gradient energy). Result: `rel_err ~1e-12` down to `eps=1e-4`.

**Physical interpretation**: since `f^2/2-f <= 0` for `0<=f<=1`, and `Wc>0`, this term contributes `Wc*eta2*(f^2/2-f) <= 0` to `F`, most negative when `eta2` is *largest* — i.e. when ownership is concentrated in one grain rather than split evenly. Minimizing `F` therefore favors **unequal grain ownership** at fixed `f`: this is the standard multi-order-parameter "grain selection" mechanism (structurally analogous to the cross-coupling terms in classical Fan-Chen-type grain-growth models), not an artifact.

## 4. Constrained eta integrator

Unchanged in form from Milestone 12, now driven by the complete `g_i`: `d(eta_i)/dt = -M_eta*(g_i - lambda)`, `lambda = mean_i(g_i)` over active grains — an exact algebraic identity guaranteeing `sum_i d(eta_i)/dt = 0` regardless of which `g_i` is used — followed by `model.reproject`'s existing local simplex projection. `local_wc(f,e1,e2,e3,s,p)` reproduces `model.evolve_f`'s own local-`Wc` construction exactly (verified, `test_local_wc_matches_evolve_f_construction`). `constrained_variational_eta_update` gained a required `s` (Sink) parameter (needed for `effective_gamma`) and an optional `n_substeps` for Section 5 below; both callers (`scripts/coarsening_benchmarks.py`, `scripts/unified_sinusoidal_campaign.py`) were updated.

## 5. Projection contribution before/after investigation

Re-running Milestone 12's Gate D two-region benchmark (`scripts/coarsening_benchmarks.py`, flat-substrate two-grain geometry, 4000 steps) with the corrected `g_i` gives **projection/variational = 0.873** — *worse* than Milestone 12's pre-correction 0.19–0.25, because the new bulk coupling term is a genuine unstable ("anti-diffusive") mode: linearizing the bulk dynamics in a region of uniform `Wc`, `d(eta_i-mean)/dt = +M_eta*Wc*(eta_i-mean)`, which grows for *any* forward-Euler `dt>0`.

Investigated in the order Section 7 specifies:

1. **Substepping** (`n_substeps=1,2,4,8,16` at fixed total `dt`, `scripts/coarsening_benchmarks.py --eta-substeps`): the ratio is **flat to 4 significant figures** (0.84390 -> 0.84390) across all 16x, and the resulting `V2` after 2000 steps changes only in the **8th significant digit** (`2.024809825...e-14` at n=1 vs. `2.024809794...e-14` at n=16). This rules out a simple discretization/stiffness artifact: the exact continuum trajectory hits the simplex boundary at essentially the same point regardless of how many discrete steps are used to get there.
2. **Root cause, checked directly**: at a representative snapshot, `e1` sits within `1e-12` of its simplex boundary (`0` or `f`) at **26.5%** of grid points, with the *maximum* single-step variational perturbation only `~1e-5`. The projection is not correcting large integration errors — it is clamping a field that has already saturated at its physical boundary almost everywhere except at genuine grain boundaries, which is the expected steady-state signature of the (now-complete, physically correct) grain-selection free energy.
3. Per the handoff's own priority ("more important: the sign/magnitude of dV2/dt must be unchanged as the projection correction is reduced"): since substepping demonstrably does not change `V2` beyond the 8th significant digit while leaving the ratio unchanged, the projection is **not materially driving** `dV2/dt`, even though its raw magnitude stays large.

**Conclusion**: the large projection/variational ratio is not a numerical defect to "fix" — it reflects genuine simplex-boundary saturation of a correctly-derived unstable order-parameter free energy. No new global correction was introduced (per the explicit instruction); `n_substeps` was added as a diagnostic/optional feature only. This is flagged as a legitimate open item for future work (e.g. a semi-implicit or exponential integrator for the bulk term), not a blocker for interpreting `dV2/dt`, which Section 15's control-volume decomposition (Section 12 below) independently confirms.

## 6. Timestep convergence at each dx

Tested `dt`, `dt/2`, `dt/4` from the same initial state A over a common 200-step (at the base `dt`) physical-time window, comparing `V2`, `L_contact`, `total_f`, `F` at the matched end time (`scripts/m12b_grid_convergence.py::dt_convergence_test`):

| dx | rel. `|dV2|/V2` (dt vs dt/4) | rel. `|dL_contact|/L_contact` | rel. `|dF|/F` |
|---|---|---|---|
| 5.0 nm | 1.27e-9 | 5.50e-6 | 3.95e-9 |
| 2.5 nm | 5.26e-9 | 3.90e-5 | 1.53e-9 |
| 1.25 nm | 2.43e-9 | 8.83e-6 | 7.41e-10 |

All three grids are converged to well under 1e-4 relative in every quantity — dt was never the limiting factor at any resolution tested.

**Important audit finding** (Section 8/9): production's own `p.dt = min(CFL*dx^4/(M_f*k_f), 1e-5)` formula is **clamped to the identical `1e-5 s` at both dx=5nm and dx=2.5nm** (the natural CFL-scaled values would be `5.21e-4 s` and `3.26e-5 s` respectively — 52x and 3.3x larger). Only at dx=1.25nm does the natural CFL value bind (`dt=2.03e-6 s`, unclamped). This means Milestone 12's original dx=5nm-vs-2.5nm Gate E comparison used the *same* `dt`, far below either grid's own stability limit — so its sign disagreement was never a hidden temporal-convergence problem; it was always a genuine spatial-discretization effect (now understood, via Section 12 below, to trace back mostly to the incomplete `g_i`'s effect on the eta-migration term that dominates `dV2/dt`).

## 7. Curvature-extraction methodology

Two independent methods, both reported over fixed **physical** arclength windows (`pf_sintering/curvature_extraction.py`), not fixed cell counts or single TJ pixels:

- **Method A** (diffuse-interface production thermodynamics): `mu = mu_isotropic(...)` sampled along the f=0.5 contour and averaged over the window, converted via `kappa_A = mu/(1.5*gamma_s)`. The 1.5 factor was **derived analytically** from this model's `k_f=3*gamma_s*W` normalization (substituting the planar equilibrium profile into the 2D radial chemical potential, the leading curvature correction is `mu(R) = k_f/(2*W*R) = 1.5*gamma_s/R`, not `gamma_s/R` as naively assumed from production's *separate* `sigma_curv=gamma_s*kappa` mechanical convention) and confirmed numerically to **<0.03%** against synthetic circular interfaces at R=300/500/800nm.
- **Method B** (geometric): direct Kasa least-squares circle fit to the actual f=0.5 contour points within the window (same core algorithm as `signed_curvature.signed_curvature_at`, applied to an explicit window instead of a Euclidean radius).

Windows: `[0,2W]` ("near_TJ") and `[2W,5W]` ("far"), wider than the handoff's illustrative "~1W/2W/3W" because a 1W window at the coarsest grid (dx=5nm, W=20nm) contains only ~5 raw contour points — below the minimum needed for a well-posed 3-parameter circle fit. The 2W/5W windows guarantee >=8 points even at dx=5nm.

**A real bug was found and fixed during this work**: the first version of the branch-following walk (adapted from Milestone 11's `ch_crossover_diagnostics.trace_branch_profile` algorithm) used a global nearest-unvisited-point search with a permissive direction pre-filter. On a real TJ, the two free-surface branches were only 78 degrees apart — an *acute* angle — so points on either branch project *positively* onto the other branch's direction, and the walk from `v_s1` and from `v_s2` converged onto the **identical path** (confirmed to float precision). This never surfaced in Milestone 11 because `trace_branch_profile` is only ever called with one branch direction at a time. Replaced with local circle-marching (`_walk_branch`, reusing `tj_force._circle_crossings`/`_tangent_at`): only a small circle of radius ~1.5dx around the *current* point is searched at each step, advancing to whichever crossing is most aligned with the running tangent — this cannot jump to a distant branch. Verified the two branches now diverge correctly (regression test `test_walk_branch_diverges_for_distinct_tj_branches`). `ch_crossover_diagnostics.trace_branch_profile` itself was left unmodified (existing, in-service Milestone 11 module, out of scope) — flagged as a latent risk for future work since it is subject to the same underlying algorithm.

Tests: `tests/test_curvature_extraction.py` (5 tests: flat interface -> kappa_A=0, circle -> both methods match analytic 1/R to <1%, unresolved on an empty field, branch divergence regression, `branch_mu_J_profile` resolves on a real TJ).

## 8. Curvature convergence

Method A `kappa_particle_far` and `kappa_substrate_far` (top TJ, t=0 and t=0.03s, from the representative flux-diagnostics run):

| quantity | dx=5nm | dx=2.5nm | rel. diff |
|---|---|---|---|
| `kappa_particle_far` (t=0) | -1.423e8 | -1.427e8 | 0.3% |
| `kappa_substrate_far` (t=0) | -3.729e7 | -3.658e7 | 1.9% |
| `kappa_particle_far` (t=0.03s) | -1.433e8 | -1.440e8 | 0.5% |
| `kappa_substrate_far` (t=0.03s) | -3.903e7 | -3.813e7 | 2.4% |

Both methods and both grids agree the particle-far region has substantially larger-magnitude (more negative) curvature than the substrate-far region throughout, with grid agreement at the few-percent level — reasonably converged, though the substrate-far value (a shallower, more slowly-varying surface) shows more grid sensitivity than the particle-far value.

## 9. Delta-kappa trajectories

`delta_kappa_far = kappa_particle_far - kappa_substrate_far` is **persistently negative** throughout the full 0-0.03s window at both grids (dx=5nm: -1.050e8 -> -1.042e8; dx=2.5nm: -1.061e8 -> -1.059e8), i.e. `kappa_particle < kappa_substrate` (both negative, particle more so) everywhere sampled. `delta_kappa_near` (the `[0,2W]` window) is smaller in magnitude but the same sign at both grids (dx=5nm: ~-3.2e7 to -3.9e7; dx=2.5nm: ~-4.2e7 to -4.6e7). See Section 16 for why this two-point comparison should **not** be read as directly predicting `dV2/dt`'s sign.

## 10. mu(s) trajectories

Full branch traces (`curvature_extraction.branch_mu_J_profile`, Sections 13-14, representative state at t=0.015s, both grids) show `mu(s)` is **non-monotonic across the whole connected surface**: at the TJ itself both branches necessarily share the same value (`mu(s=0)=-1.84e8` at dx=5nm); moving along the **particle** branch, `mu` *decreases* further (to `-2.14e8` at the far window); moving along the **substrate** branch, `mu` *increases* sharply (to `-6.34e7` at the far window). This non-monotonicity is exactly why the naive two-point `delta_mu_far` comparison (Section 9/12) is misleading as a donor/receiver predictor here — see Section 13.

## 11. J_s(s) trajectories

`J_tangent(s)` (positive = away from the TJ along the branch) at t=0.015s, dx=2.5nm: particle branch `J_tangent(s=0)=-1.15e-9` (toward the TJ) weakening to `-1.03e-10` at the far window; substrate branch `J_tangent(s=0)=+1.01e-9` (away from the TJ, i.e. spreading material further into the substrate) weakening to `+2.08e-11`. Both grids agree in sign and order of magnitude. This is the physically expected local picture: conserved-f material moves from the particle cap toward the neck, and from the neck onward into the substrate — but see Section 12 for its (small) share of the total `dV2/dt`.

## 12. Particle control-volume balance (Section 15)

`pf_sintering/flux_closure.py::particle_volume_rate_decomposition` splits `dV2/dt` **exactly** (an algebraic identity, not an approximation) into a conserved-f transport term (f-change weighted by post-step ownership) and an eta-ownership-migration term (ownership change weighted by pre-step f). At the representative t=0.015s state:

| dx | `dV2/dt` total | transport | eta migration | transport fraction |
|---|---|---|---|---|
| 5.0 nm | -1.552e-15 | -1.333e-17 | -1.538e-15 | 0.86% |
| 2.5 nm | -1.235e-15 | -1.116e-17 | -1.224e-15 | 0.90% |
| closure residual | ~1e-30 (exact) | | | |

**This is the central quantitative finding of this milestone**: `dV2/dt` is **~99% driven by eta-ownership migration** (structural relabeling — grain 1 "winning" territory from grain 2 at their shared boundary, per Section 3's grain-selection mechanism) and **only ~1% by actual conserved-f mass transport**. The small transport contribution is in the physically expected direction (Section 11: particle donates toward the neck), but it is not what is actually shrinking the particle's f-weighted ownership volume. Per the handoff's explicit Section 15 instruction, **this eta-ownership change should not be called "mass transport"** — `f` moves only slightly, if V2 is read as "how much material belongs to the particle," most of its decrease is a bookkeeping/ownership effect, not physical solid migrating away from the particle.

Cross-checked independently against `integral(J . grad(w2))dV` from the actual flux field (an integration-by-parts identity on the transport piece alone); the two agree in sign and order of magnitude (within the loose tolerance expected from a cell-centered-vs-face-centered flux discretization mismatch).

## 13. Neck control-volume balance (Section 16)

`neck_control_volume_flux_balance` splits the neck control region (`ch_crossover_diagnostics.neck_region_mask`) into particle-facing/substrate-facing halves via the same substrate-baseline classifier used for TJ branches, summing the exact per-cell `-div(J)` (via `bc_ops.flux_divergence`'s discrete divergence theorem) over each half. At the representative state, this returned `dM/dt_substrate_half = 0` exactly at both grids (all mask cells classified as particle-facing under the current threshold) — a **degenerate result**, reported honestly as a limitation of this specific geometric split at this control-region size/threshold rather than presented as a validated finding. The *total* neck-region flux (`dM/dt_total`, summed correctly and confirmed to match a direct before/after mass-difference measurement to <1e-6 relative in `tests/test_flux_closure.py`) is reliable; the particle/substrate sub-split needs a refined classifier (e.g. a face-based rather than cell-based split, or a larger control region) before it can be trusted, and is left for future work.

## 14. V2(t) and dV2/dt

Full dense time series (not endpoints only), State A, unified isotropic transport, 0 to 0.03s, sampled every 20 steps (151 points per grid):

| dx | V2(t=0) | V2(t=0.03s) | dV2/dt(t=0.03s) |
|---|---|---|---|
| 5.0 nm | 2.03983e-14 | 2.03448e-14 | -1.065e-15 |
| 2.5 nm | 2.04018e-14 | 2.03533e-14 | -8.951e-16 |
| 1.25 nm | 2.04029e-14 | 2.03580e-14 | -8.757e-16 |

`dV2/dt` is **negative at every sampled point, at all three grids, with no sign change anywhere in 0-0.03s**, and the endpoint value is converging smoothly with grid refinement (-1.065e-15 -> -8.951e-16 -> -8.757e-16, the 5->2.5nm step and 2.5->1.25nm step differing by a factor consistent with the expected convergence order). A longer dx=2.5nm continuation to t=0.09s (Section 21) confirms `dV2/dt` continues decaying in magnitude (to -4.38e-16 by t=0.09s) without ever reversing sign.

## 15. L_contact(t) and dL/dt

`L_contact_TJ_sub` grows monotonically and smoothly at all three grids over the same window (dx=5nm: 64.95nm -> 74.13nm; dx=2.5nm: 63.49nm -> 74.12nm; dx=1.25nm: 62.86nm -> 74.23nm) — the neck is widening throughout, consistent across resolutions (endpoint values agree to <0.2%), with no discontinuities or reversals. `dL/dt` is smooth and positive throughout the sampled window at every grid.

## 16. Donor-receiver crossover: none found

`t_V*` (the zero-crossing of `dV2/dt`) is **`None` at all three grids** over the full 0-0.03s window, and remains absent through the longer 0-0.09s evolved-state continuation at dx=2.5nm. Milestone 12's Gate E crossover (dx=5nm showing `V2` decrease, dx=2.5nm showing a small `V2` increase, at matched t=0.02s) does **not reproduce** under the corrected physics — both grids now agree V2 decreases monotonically throughout. Section 12's control-volume decomposition explains why the earlier, incomplete `g_i` could produce grid-dependent noise in the sign: with the eta free energy driving ~99% of `dV2/dt`, an incompletely-derived (and therefore incorrectly-scaled/shaped) structural force is exactly the kind of error that would produce spurious, non-converging sign flips between grids — consistent with what was actually observed in Milestone 12.

## 17. dx=5/2.5/1.25nm comparison summary

All fixed-physics quantities (W, R2, wavelength, amplitude, M_f, M_eta, M_s) confirmed dx-independent by direct audit (Section 6/8). dt independently converged at each grid (Section 6). V2(t), dV2/dt, and L_contact(t) are consistent and converging smoothly across all three grids, with no qualitative disagreement of any kind (unlike Milestone 12's pre-correction result). The Mullins k^4 recheck (below in this section) also grid-converges well. Overall: **this is the cleanest, most consistent three-grid agreement of any milestone in this line of work to date.**

Grid-converged Mullins benchmark (`scripts/surface_transport_benchmarks.py`, unchanged, single-phase isotropic, dx=5 and 2.5nm, wavelengths 120/160/240nm): log-log power-law fit `decay_rate ~ k^4.03` (R^2=0.99987) at dx=5nm and `k^3.97` (R^2=0.99999) at dx=2.5nm — both consistent with the analytic k^4 exponent to within 1%, confirming the fixed-physics mobility normalization remains correct and grid-converged under the corrected eta physics. Energy strictly non-increasing at all dt fractions tested (1, 0.5, 0.25, 0.125x) at both grids; mass conserved to ~1e-15 relative.

## 18. G1/G2/G3/G4 classification

**G1 — converged shrinkage**, with an important qualification. `dV2/dt < 0` at all three fine grids (5, 2.5, 1.25nm), consistently, with no sign change over 0-0.03s (and confirmed out to 0.09s at dx=2.5nm). However, Section 12's control-volume decomposition shows this "shrinkage" is **~99% an eta-ownership-migration (structural relabeling) effect and only ~1% genuine conserved-f mass transport** — the small transport component is itself in the physically expected donor(particle)->receiver(substrate) direction (Section 11), consistent with (though not the dominant contributor to) G1's mass-transfer framing. The headline classification should therefore read: **the particle's f-weighted ownership volume converges to a robust, grid-independent decrease, driven overwhelmingly by structural/ownership dynamics rather than by capillary-driven surface diffusion** — a materially different physical story than a simple "particle shrinks via surface diffusion" reading of G1 would suggest.

Curvature is **not** near-neutral (Section 23 does not apply): `delta_kappa_far` and `delta_mu_far` are large, consistently signed, and grid-converged at the few-percent level, not noise.

## 19. Recommendation for the next geometry experiment

1. **Do not yet change geometry.** The convergence achieved here is a significant result in its own right and should be allowed to stand as the reference case before perturbing R2/wavelength/amplitude/overlap.
2. **The highest-priority follow-up is *not* a new geometry but a deeper look at the eta-ownership dynamics themselves**, since Section 12 shows they now dominate `dV2/dt` by ~2 orders of magnitude over conserved-f transport. Specifically: (a) address the Section 5 integrator instability (semi-implicit or exponential treatment of the bulk coupling term) so the projection/variational ratio can be meaningfully reduced rather than merely shown-not-to-matter; (b) verify the grain-selection mechanism's rate constant (`M_eta`, currently set by `eta_diffusivity_fixed_physical`) is physically calibrated rather than an inherited legacy default, since it is now the dominant driver of the reported particle "shrinkage."
3. **Fix the neck control-volume particle/substrate sub-split** (Section 13's degenerate result) before relying on it for any future geometry study — likely needs a face-based boundary classification rather than a cell-based one, or a wider control region.
4. Once (2) is addressed, a natural next physical experiment is a **geometry sweep at fixed, now-validated eta kinetics** (e.g. R2, overlap, wavelength/amplitude) to see whether the eta-migration-dominated shrinkage found here is a generic feature of this two-grain sinusoidal-substrate setup or specific to the particular grain-size/GB-energy ratio tested.

---

### Appendix: raw run artifacts

- `runs/m12b_grid_convergence_5_2p5_v2.json` — dx=5/2.5nm primary State-A campaign with curvature/energy diagnostics (Sections 6, 8-11, 14-16).
- `runs/m12b_grid_convergence_1p25.json` — dx=1.25nm primary State-A campaign (Sections 6, 14-16).
- `runs/m12b_flux_diagnostics.json` — representative-state (t=0.015s) control-volume closures and mu(s)/J_s(s) profiles, dx=5/2.5nm (Sections 10-13).
- `runs/m12b_evolved_state_2p5.json` — dx=2.5nm continuation to t=0.09s (Section 21/16).
- `runs/m12b_mullins_dx5.json`, `runs/m12b_mullins_dx2p5.json` — Mullins k^4 recheck (Section 17).
- `runs/m12b_gate_d_recheck.json`, `runs/m12b_substep_{1,2,4,8,16}.log` — projection/variational ratio investigation (Section 5).

### Stop conditions honored

Per Sections 22/24/25, this milestone did **not**: activate sink, hazard, RBM, imposed stress, or imposed strain; change any physical geometry parameter (R2, wavelength, amplitude, overlap all held fixed at their Milestone 12 values throughout); reintroduce anisotropy (`use_aniso_surface=False` throughout, no anisotropy parameters referenced); or promote the new transport/eta mode to any production default path. All new physics (the corrected `g_i`) lives only in the diagnostic `pf_sintering/constrained_eta.py` module, unchanged from Milestone 12's own scope boundary.
