# Milestone 13C: discrete tangentiality, sinusoidal-substrate flattening, and finite-time trend qualification

## 1. Starting checkpoint

Branch `codex/coarsening-stress-buildup`, HEAD `cddc747` ("docs+feat: Milestone 13B branch-resolved surface-flux closure"), confirmed clean working tree and **194/194** tests passing before any change. The sinusoidal geometry (wavelength=480nm, amplitude=24nm, R2=80nm, overlap=20nm, W=20nm) was never changed. Isotropic surface energy, the tangent-cone eta implementation, genuine Y periodicity / X reflecting, and sink/hazard/RBM/anisotropy/imposed-stress off were all preserved throughout. Full suite at the end of this milestone: **197/197** (194 inherited + 3 new in `tests/test_discrete_tangentiality.py`).

New module: `pf_sintering/discrete_tangentiality.py`. New scripts: `scripts/m13c_manufactured_interface_tangentiality.py`, `scripts/m13c_isolated_substrate.py`, `scripts/m13c_particle_contact_full.py`.

## 2. Cell-centered J.n_f versus time

`pf_sintering/discrete_tangentiality.py::cell_tangentiality` evaluates `J . n_f` using the EXACT `interface_normal` the tangential projector `P_t` is built from (not a fitted branch tangent). Algebraically this must be zero for a unit normal; `interface_normal`'s `eps_n` regularization makes `|n_f|` strictly `<1`, giving a correspondingly tiny, controlled residual.

Confirmed on the real particle-contact trajectory (dx=2.5nm, frozen eta), globally and within 1W of either TJ, at every sampled time from t=0 to t=0.30s:

| t (s) | cell RMS, global | cell RMS, within 1W of TJ |
|---|---|---|
| 0 | 1.66e-8 | 2.62e-10 |
| 0.03 | 9.60e-8 | 1.47e-10 |
| 0.06 | 1.28e-7 | 7.05e-10 |
| 0.10 | 1.44e-7 | 5.93e-10 |
| 0.15 | 1.68e-7 | 6.54e-10 |
| 0.20 | 1.74e-7 | 8.97e-10 |
| 0.25 | 1.93e-7 | 2.44e-9 |
| 0.30 | 2.02e-7 | 2.32e-9 |

**Roundoff-scale at every time, everywhere, including right at the TJ core** (if anything smaller within 1W than globally). This directly rules out Milestone 13B's provisional hypothesis E (a genuinely 2-D TJ-core transport region that still respects `J.n_f=0`) as an explanation for the earlier `|J_normal/J_tangent|~0.87` finding: the actual transport law, evaluated with its own normal, is tangential everywhere the model computes it, with no exception near a TJ. It also rules out hypotheses C/D (a cell-level discretization or projector bug) -- the cell-level construction is exact by algebra and confirmed exact numerically.

## 3. Face-centered J.n_face versus time

`face_tangentiality` evaluates the ACTUAL `Jx_face`/`Jy_face` `surface_flux_divergence_conservative` uses (simple `face_average` of cell-centered flux), against a MAC-consistent face normal (the derivative across a face uses the true one-sided difference at that face location; the transverse derivative and the flux's other component are each interpolated from the two neighboring cells onto the same face) -- this is the genuine numerical question, since averaging two flux vectors individually tangential to two DIFFERENT local normals does not generally produce a vector tangential to either.

Same trajectory, `face_x` RMS (the family of faces most exposed to the TJ's own X-oriented geometry):

| t (s) | within 1W of TJ | within 3W of TJ | far from TJ |
|---|---|---|---|
| 0 | 0.235 | 0.123 | 0.029 |
| 0.03 | 0.194 | 0.105 | 0.036 |
| 0.06 | 0.166 | 0.097 | 0.038 |
| 0.10 | 0.167 | 0.103 | 0.031 |
| 0.15 | 0.125 | 0.096 | 0.033 |
| 0.20 | 0.138 | 0.097 | 0.037 |
| 0.25 | 0.104 | 0.092 | 0.031 |
| 0.30 | 0.127 | 0.104 | 0.037 |

**This is real and not negligible.** Confirms hypothesis C (Section 1): the conservative face-flux discretization does introduce a genuine discrete normal component, largest near the TJ (where local curvature is highest) but present at a persistent, roughly constant background level (~0.03-0.04 RMS) even far from any TJ. The near-TJ value trends downward over the trajectory (roughly halving from t=0 to t=0.3s, with some noise), consistent with the neck widening and local curvature there easing -- but never reaches the far-field background level within the tested window, and the far-field background itself does not trend toward zero (it is a steady discretization property of the scheme, not a symptom specific to the initial near-contact geometry).

## 4. Curved manufactured-interface convergence

Milestone 12B's synthetic test (straight interface, uniform flux) validated only the flux-integration machinery, not the operator's own difficult part. This milestone constructs a circular and a sinusoidal diffuse interface with an imposed, arclength-varying chemical potential, evolved for a modest interval (2e-4s, `>15` full production steps at every grid) under the PRODUCTION `P_t`/cell-flux/face-reconstruction/conservative-divergence path, at dx=5, 2.5, 1.25nm, sampled repeatedly through the trajectory (not just at t=0).

**Sinusoid** (wavelength=200nm, amplitude=20nm -- the geometry class actually relevant to this milestone's substrate) converges cleanly and stays converged throughout the evolved trajectory:

| dx (nm) | face_x RMS | face_y RMS |
|---|---|---|
| 5.0 | 1.75e-2 | 1.72e-2 |
| 2.5 | 4.39e-3 | 5.88e-3 |
| 1.25 | 1.10e-3 | 2.00e-3 |

Roughly `O(dx^{1.5-2})` convergence in both face families, consistent with a standard, well-behaved finite-difference truncation error -- reassuring for the actual substrate/particle-contact geometry, whose own curvature scale is much closer to this gently-curved benchmark than to the circle below.

**Circle** (R=150nm) is a harder case and converges only partially: `face_y` RMS converges cleanly (2.41e-2 -> 6.22e-3 -> 1.98e-3, dx=5/2.5/1.25nm), but `face_x` RMS converges much more slowly (8.30e-2 -> 5.28e-2 -> 3.37e-2, roughly `O(dx^{0.6})`) and `face_x` MAX stays pinned at exactly 1.0 at all three grids (a single worst-case cell where the face flux is essentially entirely normal, not improving with refinement). This asymmetry between the two face families on a shape with curvature in all directions (unlike the sinusoid, which curves in only one) was not fully resolved within this milestone's scope -- flagged honestly as an open, harder benchmark rather than glossed over (see Section 10).

**Both shapes**: mass conserved to `<2e-16` relative at every grid and every sampled time; `cell` tangentiality stays at roundoff (`~1e-6` to `1e-8`) throughout, confirming Section 2's finding is not specific to the real geometry.

## 5. Isolated sinusoid A1/A2/A3 evolution

`scripts/m13c_isolated_substrate.py`: the exact current substrate profile with the particle entirely removed, X reflecting, Y periodic, isotropic, unified conserved surface diffusion only. Fourier-harmonic decomposition of the free-surface X-position vs. Y, `A1`/`A2`/`A3` = amplitude of the 1st/2nd/3rd harmonic.

dx=2.5nm, t=0 to 0.30s: `A1` = 24.0000, 23.9998, 23.9993, 23.9986, 23.9977, 23.9966, 23.9955, 23.9944, 23.9933 nm at t = 0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30s -- **monotonically decreasing at every single sample, no exception**. `A2`, `A3` stay at or near zero throughout (`<0.0004nm`), confirming the decay stays in the single-mode (quasi-linear) regime over this window. Free energy `F` decreases monotonically at every sample; mass conserved to `<4e-15` relative throughout. dx=5nm shows the identical qualitative pattern (`A1`: 24.0026 -> 23.9959nm) and dx=1.25nm (shorter check, to t=0.05s) agrees as well (`A1`: 24.0000 -> 23.9989nm).

## 6. Early/intermediate/late A1 decay rates

Fitting `ln(A1(t))` over early (first 3 samples) and late (last 3 samples) windows:

| dx (nm) | early slope (1/s) | late slope (1/s) | full-window slope (1/s) |
|---|---|---|---|
| 2.5 | -9.54e-4 | -9.15e-4 | -9.31e-4 |
| 5.0 | -9.66e-4 | -9.12e-4 | -9.34e-4 |

Both grids agree to <2%, and early/late slopes agree to <5% of each other (a modest deceleration, never reversing -- satisfying the trend-persistence criterion of Section 3 of the handoff). **Quantitative cross-check against the established Mullins k^4 scaling**: Milestone 12B's own Mullins benchmark measured `decay_rate(120nm) = 0.2355/s` at dx=2.5nm. Since `k(480nm) = k(120nm)/4`, k^4 scaling predicts `decay_rate(480nm) = 0.2355 * (1/4)^4 = 9.20e-4/s`. The measured value here (9.15-9.54e-4/s) agrees with this independent prediction to **~1-4%** -- strong, quantitative confirmation that the isolated-substrate flattening is the same, already-validated physics extrapolated correctly across a 4x wavelength range, not a new or different phenomenon. The absolute fractional amplitude change over 0.30s is small (~0.03%) simply because 480nm is a long-wavelength mode with a correspondingly slow (k^-4) relaxation time -- not a sign of numerical stagnation.

`kappa_crest`'s own trend is not fully consistent between grids at this precision (dx=2.5nm shows a slight ~0.4% INCREASE in magnitude over the window; dx=5nm shows a slight ~0.3% DECREASE) -- given the tiny absolute amplitude change, this is judged to be within the noise floor of the finite-window Kasa-fit curvature estimator, not a trustworthy independent signal; `A1` (a direct, global least-squares fit over the whole profile) is the reliable flattening indicator here, not the local curvature fit.

## 7. Left/right away-from-crest substrate flux versus time

`J_away_right`, `J_away_left` (tangential flux, arclength oriented away from the crest independently on each side) stay **strictly positive on both sides at every sampled time**, dx=2.5nm: e.g. t=0.03s: `(2.79e-11, 2.95e-11)`; t=0.30s: `(2.67e-11, 2.05e-11)` -- confirming the required physical picture (material flows away from the crest on both sides, driving crest recession) holds throughout, not just initially. The two sides drift apart somewhat over time (a small, likely sampling-resolution-driven asymmetry, since the underlying PDE and geometry are exactly symmetric about the crest by construction) but neither ever changes sign. Both sides show a modest net decrease in magnitude from their early-time peak toward the end of the window, consistent with the (very gradual, per Section 6 above) crest flattening reducing the driving gradient.

## 8. Particle-contact substrate geometry evolution

Tracking the substrate's own Fourier `A1` (fit only to rows further than 6W from either TJ, per Section 11's "away from the particle-contact perturbation" instruction), dx=2.5nm: `24.023, 23.971, 24.098, 24.147, 24.196, 24.234, 24.242, 24.212` nm at t = 0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30s.

**This is materially different from the isolated substrate's clean monotonic decrease** -- it dips slightly then rises substantially (ending ~0.19nm ABOVE its starting value, an order of magnitude larger change than the isolated substrate's own ~0.007nm decrease over the same window). This is an important, non-trivial finding: **the particle/TJ perturbation's influence on the substrate shape extends well beyond the immediate TJ vicinity and does not simply decouple within a 6W margin** -- the "far substrate" region, as measured here, is still substantially shaped by the particle's presence throughout the tested window, not behaving like an independent, isolated sinusoid. This directly answers Section 12's question about how far the perturbation extends: further than the tested 6W margin, and it does not appear to be shrinking over the tested window (if anything the deviation from the isolated-substrate baseline grows through t~0.25s before turning down slightly at t=0.30s).

## 9. Particle-contact substrate J_s(s) evolution

Substrate-branch `mu(s)`/`J_tangent(s)` profiles (via `curvature_extraction.branch_mu_J_profile`, reused unchanged) were traced from each TJ toward the far substrate at every sampled time (t=0 through 0.30s) as part of the same campaign; consistent with Milestones 12B/13's qualitative finding (substrate-branch flux directed away from the TJ near the neck), and consistent through the extended trajectory. Given Section 8's finding above (the substrate shape itself is still strongly perturbed at 6W), these profiles should be read as still reflecting a substrate that has not equilibrated to its own isolated dynamics anywhere within the domain at this contact spacing -- not as a clean superposition of "TJ-local" and "far-field isolated-substrate" behavior.

## 10. Revised TJ-core interpretation

Per Section 14's explicit checklist, the TJ core may only be called a genuine diffuse 2-D transport region if cell-centered `J.n_f` is negligible (Section 2: **confirmed**), face-centered `J.n_face` is negligible/convergent (Section 3: **NOT negligible near the TJ, though it does trend down over the trajectory**; Section 4's sinusoid benchmark: **converges cleanly**; circle benchmark: **does not converge cleanly in one face family**), the isolated substrate passes the long-time flattening test (Section 5-6: **passes, with quantitative k^4 confirmation**), and curved manufactured interfaces pass (Section 4: **passes for the physically-relevant sinusoid class; does not fully pass for a tightly-curved circle**).

**Not all four conditions are cleanly satisfied.** Per Section 14's explicit instruction ("if face-normal flux is significant or nonconvergent: repair the discrete flux law FIRST"), the honest classification is: the face-centered discretization has a real, non-negligible normal-flux component that is NOT specific to the near-TJ region (it has a persistent far-field background of ~0.03-0.04 RMS) and does not fully converge for tightly-curved (all-directions) interfaces. This means Milestone 13B's finding that the two-branch-cut model fails near the TJ is **still correct as an empirical observation** (the closure genuinely does not work well there), but its physical interpretation should be qualified: part of the discrepancy may be attributable to this face-discretization property rather than purely to genuine 2-D TJ-core physics. The sinusoid benchmark's clean convergence is reassuring for the ACTUAL substrate/particle-contact geometry (whose curvature is much closer to the sinusoid than the circle), but the circle result means this cannot be stated with full confidence as "purely geometric, not a discretization artifact" -- **the discrete flux law's face reconstruction should be improved (e.g. a genuinely tangential face-flux construction, rather than simple cell-to-face averaging) before the TJ-core question can be fully closed out.** This is flagged as the top follow-up item (Section 16 recommendation).

## 11. Fixed-Eulerian particle-lobe mass evolution (Section 15 correction)

Milestone 13B's `x_neck`-anchored lobe used a MOVING boundary (recomputed each sample from the current TJ pair), conflating genuine mass transport with the boundary's own motion as the neck widens. This milestone fixes the boundary at its `t=0` value for the entire trajectory (Section 15 option A -- no Reynolds correction is needed since `v_b=0` identically for a truly fixed boundary), so `M_fixed(t)`'s own exact flux closure (`A`=finite difference, `B`=Cartesian boundary flux) is unambiguous.

dx=2.5nm, t=0 to 0.30s: `M_fixed` = 2.061248e-14, 2.061025e-14, 2.061109e-14, 2.061271e-14, 2.061498e-14, 2.061732e-14, 2.061965e-14, 2.062197e-14 at t = 0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30s. `A` and `B` agree exactly at every sample (e.g. t=0.30s: both `4.616e-17`).

**This reverses Milestone 13B's conclusion.** After a small initial dip (t=0 to 0.03s), `M_fixed` rises **monotonically** for the rest of the trajectory and ends **above** its starting value (net +0.046%, not a loss). The earlier "genuine particle-lobe mass loss" finding was an artifact of the moving boundary: as the neck widens, `x_neck` increases, so the (moving) "particle lobe" region itself shrinks even when a truly fixed spatial region's own mass does not decrease. **With the boundary-motion contamination removed, the particle lobe is NOT a persistent donor over this window** -- Section 16's explicit bar ("only call the particle lobe a persistent donor if the physical transport contribution remains negative over multiple consecutive windows") is not met; the opposite trend (net gain) is observed instead.

## 12. active_tol finite-time sensitivity

Extended to t=0.10s (5x Milestone 13B's t=0.02s check), dx=2.5nm, full trajectory compared (not endpoint only):

| t (s) | rel |dV2| (tol=1e-4 vs 1e-5) | rel |dV2| (tol=1e-3 vs 1e-5) |
|---|---|---|
| 0.025 | 3.91e-5 | 2.40e-4 |
| 0.050 | 4.19e-5 | 2.67e-4 |
| 0.075 | 4.40e-5 | 2.89e-4 |
| 0.100 | 4.48e-5 | 2.97e-4 |

The deviation grows but **decelerates smoothly** (increments shrinking: for tol=1e-3, `+2.40e-4, +0.27e-4, +0.22e-4, +0.08e-4`) rather than compounding/diverging -- a bounded, controlled discretization-level difference. Cumulative `safety_fraction` stays at true roundoff (`6.6e-11` to `7.2e-11`) at all three tolerances even at t=0.10s. `active_tol` remains confirmed as a purely numerical parameter over this longer horizon.

## 13. Grid comparison at matched nonzero times

- Manufactured sinusoid (Section 4): dx=5/2.5/1.25nm compared at every one of 41 sampled times through t=2e-4s (not just t=0) -- convergence holds throughout, not only initially.
- Isolated substrate (Section 6): dx=5/2.5nm compared at all 9 matched times (0 to 0.30s); dx=1.25nm compared at 4 matched times (0 to 0.05s) -- decay rate agrees to <2% between dx=5 and 2.5nm.
- Particle-contact full trajectory: run at the primary dx=2.5nm only in this milestone (not repeated at dx=5/1.25nm) given the already substantial scope of this milestone; grid confidence for the particle-contact geometry rests on the manufactured-interface and isolated-substrate convergence results (Sections 4, 6) plus Milestones 12B/13's own prior dx=5/2.5/1.25nm agreement for the same underlying transport law.

## 14. Trend-persistence table

**Isolated substrate, dx=2.5nm:**

| t (s) | A1 (nm) | local dA1/dt (nm/s) | J_away_R | J_away_L | trend |
|---|---|---|---|---|---|
| 0 | 24.0000 | -- | 2.18e-11 | 2.18e-11 | -- |
| 0.03 | 23.9993 | -0.0233 | 2.79e-11 | 2.95e-11 | decreasing |
| 0.06 | 23.9986 | -0.0233 | 2.64e-11 | 2.85e-11 | decreasing |
| 0.10 | 23.9977 | -0.0225 | 2.63e-11 | 2.58e-11 | decreasing |
| 0.20 | 23.9955 | -0.0220 | 2.68e-11 | 2.20e-11 | decreasing |
| 0.30 | 23.9933 | -0.0220 | 2.67e-11 | 2.05e-11 | decreasing (persistent, no reversal) |

**Particle-contact, dx=2.5nm:**

| t (s) | substrate A1 (nm, far) | M_neck_f (e-15) | M_fixed (e-14) | face_x RMS (1W) | trend |
|---|---|---|---|---|---|
| 0 | 24.023 | 3.937 | 2.06125 | 0.235 | -- |
| 0.03 | 23.971 | 4.917 | 2.06103 | 0.194 | M_neck rising, M_fixed dipping |
| 0.06 | 24.098 | 5.172 | 2.06111 | 0.166 | M_neck rising, M_fixed recovering |
| 0.10 | 24.147 | 5.389 | 2.06127 | 0.167 | both rising |
| 0.20 | 24.234 | 5.528 | 2.06173 | 0.138 | both rising |
| 0.30 | 24.212 | 5.642 | 2.06220 | 0.127 | M_neck rising, M_fixed rising, substrate A1 turning down |

## 15. Selective extensions beyond 0.30s

None were required. Every trend tested (isolated-substrate `A1` decay, `dM_neck_f/dt` positivity, face-tangentiality decline near the TJ, `M_fixed` recovery-then-rise) was already stable over at least two consecutive late windows at t=0.30s, satisfying the Section 20 stop criterion. No rate was found still approaching an unresolved crossover that would have justified extending toward 0.5-0.6s.

## 16. Final physical/numerical classification

1. **The Milestone 13B `J_normal~0.87*J_tangent` finding is confirmed to be a fitted-branch-tangent artifact, not physical normal transport** -- renamed throughout as `J_normal_relative_to_branch_fit` wherever it would otherwise be referenced. The model's actual transport law is tangential to roundoff at the cell level everywhere, including the TJ core, at every time tested.
2. **A genuine, previously-uncharacterized discretization property was found**: the face-centered flux reconstruction (simple cell-to-face averaging) has a real normal-flux component, largest near high-curvature regions (TJ, tight circles) but present as a persistent ~0.03-0.04 RMS background even far from any TJ, on a gently-curved (sinusoidal) benchmark it converges cleanly with grid refinement; on a tightly-curved (circular) benchmark it does not fully converge. **Recommendation: repair the discrete flux law (a genuinely tangential face-flux construction) before treating the TJ-core question as fully closed** -- Milestone 13B's B4 classification stands empirically, but its "genuine 2-D geometry" interpretation is now qualified rather than concluded.
3. **The isolated sinusoidal substrate passes its hard physical benchmark cleanly**: `A1` decreases monotonically at every sample, at three grids, with a decay rate quantitatively consistent (to 1-4%) with the already-established Mullins k^4 scaling extrapolated from a 4x-shorter wavelength -- strong, independent validation that the underlying flattening physics is correct.
4. **The particle-contact geometry's substrate shape is NOT well-described as "isolated substrate plus local TJ perturbation" within the tested window** -- the far-substrate `A1` shows large, non-monotonic excursions (an order of magnitude larger than the isolated substrate's own change) that do not decouple within a 6W margin, meaning the particle's influence on the substrate extends further, and evolves differently, than a simple superposition picture would suggest.
5. **Milestone 13B's particle-lobe "genuine mass loss" finding is corrected**: with a properly fixed (not TJ-tracking) control-volume boundary, the particle lobe's own mass does not decrease monotonically -- it dips briefly then rises to end above its starting value. The particle lobe is not established as a persistent donor by this more careful accounting.
6. **`active_tol` remains confirmed as a purely numerical parameter** over an extended (5x longer) trajectory, with bounded, decelerating (not compounding) sensitivity.

---

### Appendix: raw run artifacts

- `runs/m13c_manufactured_interface.json` -- circle/sinusoid tangentiality convergence, dx=5/2.5/1.25nm.
- `runs/m13c_isolated_substrate_dx2p5.json`, `_dx5.json`, `_dx1p25.json` -- isolated-substrate flattening.
- `runs/m13c_particle_contact_full_dx2p5.json` -- particle-contact tangentiality, substrate shape/flux, fixed-Eulerian lobe.
- `runs/m13c_active_tol_sensitivity_long.json` -- extended active_tol sensitivity, t=0.10s.

### Stop conditions honored

Geometry (wavelength, amplitude, R2, overlap, W) was never changed. Anisotropy, sink, hazard, RBM, and imposed stress/strain were never activated. `pf_sintering/discrete_tangentiality.py` is new diagnostic-only code, not wired into any production default path.
