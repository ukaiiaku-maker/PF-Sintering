# Milestone 14F: TJ thermodynamic qualification attempt (Section 17 follow-up to Milestone 14E)

## 0. Starting checkpoint and scope

Branch `codex/coarsening-stress-buildup`, HEAD `8073484` ("docs+feat: Milestone 14E physical GB mobility calibration -- STOP triggered"), confirmed clean working tree and **217/217** tests passing before any change. Sink, hazard, RBM, anisotropy stayed OFF throughout. Per Section 17C's explicit ordering ("Question A... MUST pass after gamma_GB normalization" before Question B is meaningful), this milestone stayed on **Question A** (thermodynamic TJ qualification) once it became clear Question A was not yet cleanly resolved -- Section 17D's rate-competition campaign (new geometry, M_gb ladder), Section 17F's M_s perturbation, Section 17G's full energy ledger, and the G1-G4/M_TJ decision were **not** run at scale; see Section 6 below for what was and was not attempted and why.

New: two functions added to `pf_sintering/constrained_eta.py` (`GB_ENERGY_CALIBRATION_FACTOR`, `gamma_gb_target_to_declared`, both diagnostic-only, matching that module's existing scope), two new tests, one new script (`scripts/m14f_tj_thermodynamic_qualification.py`).

## 1. The gamma_GB normalization fix (Milestone 14E Section 4/5's identified follow-up)

Milestone 14E found, numerically, that the declared `gamma_gb` parameter and the eta profile's actual implemented excess free energy differ by a factor of ~19. This milestone derived the **exact** value analytically (not just numerically): for an isolated planar GB at `f=1` with the natural profile `eta1=0.5*(1-tanh(x/W))`, `eta2=0.5*(1+tanh(x/W))`,

```
integral(bulk excess) dx  = 0.5*Wc*W = 18*gamma_gb_declared
integral(gradient excess) dx = k_eta/(3*W) = 1*gamma_gb_declared
gamma_gb_implemented = 19*gamma_gb_declared   -- EXACT, W-independent
```

(both terms individually reduce to a pure multiple of `gamma_gb_declared` with no residual `W` dependence, confirming the factor is not grid/resolution-specific). Implemented as `constrained_eta.GB_ENERGY_CALIBRATION_FACTOR = 19.0` and `gamma_gb_target_to_declared(gamma_gb_target) = gamma_gb_target/19`, both with the full derivation in the docstring. Verified by two new tests: a direct numerical integration of the analytic profile (`test_gb_energy_calibration_factor_matches_analytic_isolated_gb`, matches to `<1e-6` relative) and a round-trip check (`test_gamma_gb_target_to_declared_round_trips`). `tests/test_constrained_eta.py`: **11 -> 13** tests, all passing.

## 2. A second, deeper calibration issue found while applying the fix (Section 17's own warning, realized in practice)

Applying the fix naively (passing `gamma_gb_target_to_declared(target)` as `gamma_gb_override` to `build_params`, so `p.gamma_gb_ref` carries the corrected value everywhere) produced a striking, diagnostic result: running the coupled TJ benchmark (Section 3 below) at THREE different `gamma_gb` targets (nominal Young angles 90/120/150deg) gave **nearly identical `psi(t)` trajectories regardless of target** -- and a control run at `gamma_gb_target=0.02` (nominal Young angle 178.85deg, i.e. essentially no GB energy at all) gave the SAME trajectory too. This proved `gamma_gb` was not meaningfully influencing the dynamics at all.

**Root cause**: `gamma_gb` serves two genuinely different roles in the existing code, both currently read from the same `p.gamma_gb_ref`:

- (a) the F-field's own groove-coupling strength, `Wc=36*gamma_gb_ref/W` inside `ch_exact_energy.mu0_bulk` -- the mechanism Milestones 14/14B/14C established, characterized, and used extensively (always with eta FROZEN, so this was the ONLY active `gamma_gb`-dependent mechanism in all of those milestones), calibrated using the DECLARED value directly;
- (b) the eta profile's own static/kinetic energy inside `constrained_eta.py`'s `structural_thermodynamic_force` -- the role Section 1 of this report (and Milestone 14E Section 4) corrects with the `/19` factor.

Applying the `/19` correction via `gamma_gb_override` fixes (b) but silently WEAKENS (a) by the same 19x, since both read the identical `p.gamma_gb_ref`. Since gamma_gb targets comparable to `gamma_s=1.0` (needed for a 90-150deg Young angle) become tiny (`0.03-0.07`) once divided by 19, the groove-coupling mechanism that Milestones 14/14B/14C showed was the dominant, real, `gamma_gb`-sensitive driver of neck/TJ geometry became negligible relative to `gamma_s`'s own surface-tension terms -- explaining the observed target-independence directly.

**Fix**: decouple the two roles at the script level (no production/library-adjacent change beyond the diagnostic-only Section 1 addition). `scripts/m14f_tj_thermodynamic_qualification.py` keeps `p.gamma_gb_ref` at the TARGET (uncorrected) value for the f-transport step (matching Milestones 14-14D's established convention), and constructs a separate `Params` copy via `dataclasses.replace(p, gamma_gb_ref=gamma_gb_target_to_declared(gamma_gb_target))` passed ONLY to the eta update call. A side effect, also newly found and worth flagging: `tj_force.gb_vector`/`effective_gamma` (which computes `F_TJ`'s GB capillary vector `xi_gb`) also reads `p.gamma_gb_ref` directly -- with the decoupled `p` (target value), `F_TJ` is automatically evaluated against the correct target GB energy with no further correction needed (an earlier attempt that used a single globally-corrected `p` required an ad hoc `F_TJ_corrected` reconstruction from `xi_s1+xi_s2+19*xi_gb`, since discarded once the cleaner decoupled-`p` approach was adopted).

## 3. Question A: thermodynamic TJ qualification (Section 17C)

Deliberately idealized setup (NOT the rate-competition question): reused the qualified sinusoidal-substrate/particle-contact geometry (`A=24nm`, `lambda=480nm`, `dx=2.5nm`) with eta kinetics genuinely ON, but `surface_mobility_scale` boosted 33x above the Milestone 14 baseline (`10.0` vs. `0.3`) and `M_eta` scale=20, so free-surface relaxation is fast/non-rate-limiting -- isolating the thermodynamic question from Section 17A/B's surface-diffusion-limited kinetic question. `psi` never prescribed.

**First (undecoupled) run**: confirmed the surface-diffusion-limited hypothesis directly -- with `M_s` boosted, `psi` moved substantially (`77.87deg -> 98deg` over `t=0.15s`), a completely different (much more mobile) regime than Milestone 14E's un-boosted run (`psi` stuck at `76-81deg` over the same window). This is a genuine, useful confirmation of Section 17A/B's core physical claim.

**Decoupled run** (Section 2's fix applied), three `gamma_gb` targets, `t=0` to `0.30s`:

| gamma_gb target | psi_Y nominal | psi(t=0.02s) | psi(t=0.30s) | direction vs. target (initial psi=77.87deg) |
|---|---|---|---|---|
| 1.0 | 120.00deg | 77.97deg | 72.15deg | WRONG (should increase toward 120, decreased) |
| 1.4142 | 90.00deg | 75.25deg | 63.52deg | WRONG (should increase toward 90, decreased) |
| 0.5176 | 150.00deg | 80.59deg | 92.87deg | CORRECT (increased toward 150) |

**The three targets now give clearly DIFFERENT trajectories** (no longer locked together) -- confirming the decoupling fix restored genuine `gamma_gb` sensitivity. But only the weakest-GB-energy case (target `0.5176`, the largest nominal angle) moves in the direction its Young-Herring target requires; the two stronger cases move AWAY from their targets over the full `0.30s` window tested.

**Interpretation (not fully resolved)**: a plausible, physically-grounded explanation is transient groove formation -- classical grain-boundary-grooving theory has a stronger GB energy initially digging a deeper, sharper groove before the profile widens toward its eventual equilibrium angle over a longer timescale; the two stronger-`gamma_gb` cases may simply not have been run long enough to pass through this transient. This is plausible but **not confirmed** -- the runs were not extended far enough to see a reversal.

**Total free energy is not strictly monotonic** in any of the three runs (a structural consequence of the decoupling, not a numerical error): `F` is reported using the F-field's own energy (target-strength `Wc`), but the eta sub-step is a gradient flow of a DIFFERENT, weaker-`Wc` landscape (`p_eta`) -- so the reported `F` is not a strict Lyapunov functional for the eta sub-step. Confirmed directly: `F` increases in 1-2 sampled intervals per run (`gg1`: `t=0.02-0.06s`; `gg90`: `t=0.02-0.04s`; `gg150`: `t=0.20-0.30s`), small relative to the total `F` range but real, not roundoff. Mass conservation stayed intact throughout (`mass_drift` at the `1e-14` relative level, consistent with the surface-transport-only conservation guarantee -- eta kinetics never touch `f` by construction).

**Verdict on Question A: NOT yet conclusively passed.** The decoupling fix is a real, necessary, and now-verified correction (Section 2), and the surface-diffusion-limited hypothesis (Section 17A/B) is directly confirmed. But `psi -> Young-Herring equilibrium` was only demonstrated in the direction-correct sense for one of three tested targets within the available time budget, and the reported `F`'s non-strict-monotonicity (Section 2's decoupling side effect) means even the "energy decreasing" check is not yet on fully solid footing.

## 4. Why Question B / the G1-G4 campaign / M_s perturbation / energy ledger / M_TJ decision were deferred

Section 17C is explicit: Question A "MUST pass after gamma_GB normalization" before Question B is meaningful. Section 17H's four-condition gate for adding an independent `M_TJ` requires (1) "the calibrated free energy gives correct Young-Herring equilibrium in a simple benchmark" as a prerequisite -- not yet established (Section 3). Running the full Section 17D-G campaign (new-geometry rate-competition ladder, `M_s` perturbation, energy ledger, G1-G4 classification) on top of an unresolved thermodynamic foundation would produce numbers that could not be confidently interpreted -- if the underlying calibration is still not self-consistent, an apparent "G2/G3 coupled regime" finding could just as easily be another calibration artifact as genuine physics, exactly the trap this milestone's Section 17 was written to avoid. These sections are deferred, not skipped by oversight.

## 5. Energy/mass checks actually performed

- **Mass**: `mass_drift` stayed at the `1e-14`-relative level (roundoff-adjacent, growing slowly and smoothly, no discontinuities) in every Question A run -- consistent with the surface-transport-only conservation guarantee, unaffected by any of the calibration issues found.
- **Total F**: NOT strictly monotonic in any of the three Question A runs, for the structural reason identified in Section 3 (the decoupled `p`/`p_eta` energies) -- this is the one required check (Section 17G, "the total F must decrease") that did **not** cleanly pass, and is flagged as an open issue rather than glossed over.
- **No clip/projection-driven migration**: not explicitly re-audited in this milestone (deferred along with the rest of the Section 17D-G campaign); Milestone 14E's own finding (small but non-roundoff `safety_fraction`, `~0.1-2.4%`, in the analogous coupled-TJ setting) should be assumed to still apply until re-checked.

## 6. Recommended next steps

1. **Extend Question A's stronger-`gamma_gb` runs (targets `1.0`, `1.4142`) well past `t=0.30s`** to determine whether the observed "wrong-direction" `psi` trend is a genuine transient (groove-deepening phase, expected to reverse) or a persistent, unresolved problem. This is the single most important open question from this milestone.
2. **Resolve the `F` non-monotonicity** before trusting any energy-ledger check: either accept that `F` (F-field-only, target-`Wc`) is not the right Lyapunov functional for the decoupled system and construct a joint functional that IS conserved/decreasing for the actual implemented dynamics (accounting for both `p`'s and `p_eta`'s different effective `Wc`), or reconsider whether the decoupling itself (Section 2) should be replaced by a more principled single-energy formulation.
3. Only once (1)-(2) give a clean, confirmed "Question A passes" result should Section 17D's rate-competition campaign (new `A=100nm/lambda=320nm` geometry, `M_gb` ladder at fixed `M_s`), Section 17F's `M_s` perturbation test, Section 17G's full energy ledger, and the G1-G4/`M_TJ` decision be attempted.
4. Sink/hazard/RBM/anisotropy were not activated; the full coupled sintering calculation was not run.

**STOP. Question A is not yet conclusively resolved -- do not proceed to Question B/G1-G4/M_TJ decisions on this foundation.**
